import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TemporalSSMBlock(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=3, padding=2, groups=self.d_inner)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # x: [B, S, d_model]
        x_transpose = x.transpose(1, 2)
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[..., :x.shape[1]]
        out = self.act(x_conv.transpose(1, 2)) * self.act(z)
        out = self.out_proj(out)
        return out + x

class FiMANet(nn.Module):
    def __init__(self, seq_len=10, hidden_size=256, num_layers=2,
                 use_imu=False, imu_channels=6, output_dim=1,
                 predict_uncertainty=False, predict_sign=False,
                 pair_encoder=False, pair_strides=(1,)):
        super().__init__()
        self.seq_len = seq_len
        self.use_imu = use_imu
        self.output_dim = output_dim
        self.predict_uncertainty = predict_uncertainty
        # Two-head: separate |Δz| regression from sign classification.
        self.predict_sign = predict_sign
        if predict_sign and predict_uncertainty:
            raise ValueError("predict_sign and predict_uncertainty are mutually exclusive")
        # Pair encoder: [f_curr | Δ_s1 | Δ_s2 | ...]; multi-scale strides capture
        # motion at multiple temporal gaps.
        self.pair_encoder = pair_encoder
        self.pair_strides = tuple(int(s) for s in pair_strides) if pair_encoder else ()
        if pair_encoder:
            if not self.pair_strides:
                raise ValueError("pair_encoder=True requires at least one stride")
            if min(self.pair_strides) < 1:
                raise ValueError("pair_strides must be positive")
        self._pair_max_stride = max(self.pair_strides) if self.pair_strides else 0
        # Shift the target slice so the last output aligns with motion(S-2 → S-1).
        self.pair_target_offset = max(0, self._pair_max_stride - 1)

        # Backbone (ResNet-18)
        base_resnet = models.resnet18(weights='IMAGENET1K_V1')
        base_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(base_resnet.conv1, base_resnet.bn1, base_resnet.relu, base_resnet.maxpool)
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        for param in self.stem.parameters(): param.requires_grad = False
        for param in self.layer1.parameters(): param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # (128+256+512) * 4 = 3584
        self.fusion = nn.Sequential(
            nn.Linear(3584, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # --- IMU branch (gated additive fusion) ---
        if use_imu:
            self.imu_encoder = nn.Sequential(
                nn.Linear(imu_channels, 64),
                nn.ReLU(),
                nn.LayerNorm(64),
                nn.Linear(64, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            # gate ∈ [0,1]: how much IMU info to add at each timestep
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )

        if pair_encoder:
            in_dim = hidden_size * (1 + len(self.pair_strides))
            self.pair_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

        self.pos_encoder = PositionalEncoding(hidden_size, max_len=seq_len)
        self.temporal_layers = nn.ModuleList([
            TemporalSSMBlock(d_model=hidden_size) for _ in range(num_layers)
        ])

        if predict_sign:
            # Two heads share the temporal features.
            self.head_mag = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, output_dim),
            )
            self.head_sign = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, output_dim),
            )
        else:
            head_out = output_dim * 2 if predict_uncertainty else output_dim
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, head_out)
            )

    def forward(self, x, imu=None):
        b, s, c, h, w = x.size()
        x_flat = x.view(b * s, c, h, w)

        x = self.stem(x_flat)
        x = self.layer1(x)

        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        p2 = self.pool(f2).flatten(1)
        p3 = self.pool(f3).flatten(1)
        p4 = self.pool(f4).flatten(1)

        combined = torch.cat([p2, p3, p4], dim=1)
        features = self.fusion(combined)
        features = features.view(b, s, -1)

        if self.use_imu and imu is not None:
            imu_feat = self.imu_encoder(imu)
            gate = self.fusion_gate(torch.cat([features, imu_feat], dim=-1))
            features = features + gate * imu_feat

        if self.pair_encoder:
            S = features.shape[1]
            ms = self._pair_max_stride
            L = S - ms
            f_curr = features[:, :L, :]
            parts = [f_curr]
            for s in self.pair_strides:
                f_far = features[:, s:s + L, :]
                parts.append(f_far - f_curr)
            features = self.pair_proj(torch.cat(parts, dim=-1))

        features = self.pos_encoder(features)
        for layer in self.temporal_layers:
            features = layer(features)

        # Heads emit one prediction per temporal position; inference reads [:, -1].
        if self.predict_sign:
            mag = self.head_mag(features)
            sign_logit = self.head_sign(features)
            if self.output_dim == 1:
                mag = mag.squeeze(-1)
                sign_logit = sign_logit.squeeze(-1)
            return mag, sign_logit

        out = self.head(features)
        if self.predict_uncertainty:
            mu = out[..., :self.output_dim]
            log_var = out[..., self.output_dim:]
            if self.output_dim == 1:
                mu = mu.squeeze(-1)
                log_var = log_var.squeeze(-1)
            return mu, log_var
        return out.squeeze(-1) if self.output_dim == 1 else out
