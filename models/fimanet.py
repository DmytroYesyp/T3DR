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
        # x: [Batch, Seq, d_model]
        x_transpose = x.transpose(1, 2) # [B, D, S] для Conv1d
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[..., :x.shape[1]]
        out = self.act(x_conv.transpose(1, 2)) * self.act(z)
        out = self.out_proj(out)
        return out + x

class FiMANet(nn.Module):
    def __init__(self, seq_len=10, hidden_size=256, num_layers=2,
                 use_imu=False, imu_channels=6, output_dim=1):
        super().__init__()
        self.seq_len = seq_len
        self.use_imu = use_imu
        self.output_dim = output_dim

        # Backbone (ResNet-18)
        base_resnet = models.resnet18(weights='IMAGENET1K_V1')
        base_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(base_resnet.conv1, base_resnet.bn1, base_resnet.relu, base_resnet.maxpool)
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2 # 128
        self.layer3 = base_resnet.layer3 # 256
        self.layer4 = base_resnet.layer4 # 512

        # Заморозка
        for param in self.stem.parameters(): param.requires_grad = False
        for param in self.layer1.parameters(): param.requires_grad = False

        # Pooling to 2x2 (щоб було 4 патчі на рівень)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # Fusion: (128+256+512) * 4 патчі = 3584
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

        self.pos_encoder = PositionalEncoding(hidden_size, max_len=seq_len)
        self.temporal_layers = nn.ModuleList([
            TemporalSSMBlock(d_model=hidden_size) for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
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

        combined = torch.cat([p2, p3, p4], dim=1) # [B*S, 3584]
        features = self.fusion(combined)
        features = features.view(b, s, -1)

        # Gated additive fusion: features = visual + gate * imu
        # Visual pathway is always preserved; IMU adds supplementary info.
        if self.use_imu and imu is not None:
            imu_feat = self.imu_encoder(imu)              # [B, S, hidden]
            gate = self.fusion_gate(
                torch.cat([features, imu_feat], dim=-1)   # [B, S, 2*hidden]
            )                                              # [B, S, hidden]
            features = features + gate * imu_feat

        features = self.pos_encoder(features)
        for layer in self.temporal_layers:
            features = layer(features)

        out = self.head(features[:, -1, :])  # [B, output_dim]
        return out.squeeze(-1) if self.output_dim == 1 else out
