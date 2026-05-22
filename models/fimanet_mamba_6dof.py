import torch
import torch.nn as nn
import torchvision.models as models
import math
from mamba_ssm import Mamba


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


class FiMANetMamba6DOF(nn.Module):
    """6-DoF variant of FiMANetMamba.

    Output convention matches the TUS-REC2024 baseline 'parameter' pred_type:
    (rx, ry, rz, tx, ty, tz) — ZYX Euler angles + translations, denoting the
    transformation from the current frame to the previous frame in image_mm
    coordinates (the calibration's image-mm frame, not the tool frame).
    """

    def __init__(self, seq_len=20, hidden_size=256, num_layers=2,
                 pair_encoder=True, pair_strides=(1,),
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.seq_len = seq_len
        self.pair_encoder = pair_encoder
        self.pair_strides = tuple(int(s) for s in pair_strides) if pair_encoder else ()
        if pair_encoder:
            if not self.pair_strides:
                raise ValueError("pair_encoder=True requires at least one stride")
            if min(self.pair_strides) < 1:
                raise ValueError("pair_strides must be positive")
        self._pair_max_stride = max(self.pair_strides) if self.pair_strides else 0
        self.pair_target_offset = max(0, self._pair_max_stride - 1)

        base = models.resnet18(weights='IMAGENET1K_V1')
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        for p in self.stem.parameters():   p.requires_grad = False
        for p in self.layer1.parameters(): p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.fusion = nn.Sequential(
            nn.Linear(3584, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
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
            Mamba(d_model=hidden_size, d_state=mamba_d_state,
                  d_conv=mamba_d_conv, expand=mamba_expand)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 6),
        )

    def forward(self, x):
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
        features = self.fusion(torch.cat([p2, p3, p4], dim=1)).view(b, s, -1)

        if self.pair_encoder:
            S = features.shape[1]
            L = S - self._pair_max_stride
            f_curr = features[:, :L, :]
            parts = [f_curr]
            for stride in self.pair_strides:
                parts.append(features[:, stride:stride + L, :] - f_curr)
            features = self.pair_proj(torch.cat(parts, dim=-1))

        features = self.pos_encoder(features)
        for layer in self.temporal_layers:
            features = layer(features)

        return self.head(features)  # [B, L, 6]
