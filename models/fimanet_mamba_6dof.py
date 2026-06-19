import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from mamba_ssm import Mamba


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        if L <= self.pe.size(1):
            return x + self.pe[:, :L, :]
        # Full-scan inference: sequence longer than the trained buffer. Extend the same
        # sinusoid on the fly (positions 0..max-1 are identical, so <=max is unchanged).
        pos = torch.arange(L, dtype=torch.float, device=x.device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(L, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.unsqueeze(0)


class BiMamba(nn.Module):
    """Forward + backward Mamba pass, summed. Gives each frame future context so the
    out-of-plane (tz) sign can be resolved (a causal pass cannot)."""
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.fwd(x) + self.bwd(x.flip(1)).flip(1)


class CorrEncoder(nn.Module):
    """Local correlation volume between adjacent-frame feature maps -> motion vector.
    For each cell, cosine-correlates against a (2d+1)^2 neighbourhood in the next frame:
    the correlation magnitude encodes inter-frame speckle decorrelation (the out-of-plane
    cue that global pooling erases); the peak location encodes in-plane shift."""
    def __init__(self, max_disp=4, out_dim=128):
        super().__init__()
        self.d = max_disp
        nd = (2 * max_disp + 1) ** 2
        self.enc = nn.Sequential(
            nn.Conv2d(nd, 96, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(nn.Linear(96 * 16, out_dim), nn.LayerNorm(out_dim), nn.ReLU(inplace=True))

    def forward(self, fa, fb):
        # fa, fb: [N, C, H, W] — feature maps of frame i and i+1
        fa = F.normalize(fa, dim=1)
        fb = F.normalize(fb, dim=1)
        H, W = fa.shape[-2:]
        d = self.d
        fb_pad = F.pad(fb, [d, d, d, d])
        corrs = [(fa * fb_pad[:, :, dy:dy + H, dx:dx + W]).sum(1, keepdim=True)
                 for dy in range(2 * d + 1) for dx in range(2 * d + 1)]
        cv = torch.cat(corrs, dim=1)  # [N, (2d+1)^2, H, W]
        return self.proj(self.enc(cv).flatten(1))  # [N, out_dim]


class SpatialSSM(nn.Module):
    """Bidirectional Mamba over the flattened spatial map, mean-pooled per frame (ReMamba idea)."""
    def __init__(self, in_ch, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.proj_in = nn.Linear(in_ch, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):  # x: [N, T, in_ch]
        x = self.norm(self.proj_in(x))
        y = self.fwd(x) + self.bwd(x.flip(1)).flip(1)
        return y.mean(dim=1)  # [N, d_model]


class FiMANetMamba6DOF(nn.Module):
    """6-DoF variant of FiMANetMamba.

    Output: (rz, ry, rx, tx, ty, tz) — ZYX Euler + translation, frame_{i+1} -> frame_i in
    the image_mm frame. Channel order (rz,ry,rx) matches train/eval decode; do NOT reorder.
    pool_size / freeze_early default to the old architecture (2x2 pool, stem+layer1 frozen)
    so old checkpoints load; training passes larger pool + trainable stem/layer1.
    """

    def __init__(self, seq_len=20, hidden_size=256, num_layers=2,
                 pair_encoder=True, pair_strides=(1,),
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 backbone='resnet18', pool_size=2, freeze_early=True, bidirectional=False,
                 use_corr=False, corr_disp=4, corr_dim=128, use_spatial_ssm=False):
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
        self.backbone_name = backbone

        # Backbone — channels differ across ResNet variants.
        # R18/R34: BasicBlock, channels 64/128/256/512
        # R50:     Bottleneck, channels 256/512/1024/2048
        if backbone == 'resnet18':
            base = models.resnet18(weights='IMAGENET1K_V1')
            ch2, ch3, ch4 = 128, 256, 512
        elif backbone == 'resnet34':
            base = models.resnet34(weights='IMAGENET1K_V1')
            ch2, ch3, ch4 = 128, 256, 512
        elif backbone == 'resnet50':
            base = models.resnet50(weights='IMAGENET1K_V2')
            ch2, ch3, ch4 = 512, 1024, 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}")

        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Unfreeze stem+layer1 (freeze_early=False) so conv1 can learn US speckle — the
        # only out-of-plane (tz) cue. freeze_early=True = old frozen backbone.
        if freeze_early:
            for p in self.stem.parameters():   p.requires_grad = False
            for p in self.layer1.parameters(): p.requires_grad = False

        # Larger pool preserves speckle that 2x2 averaged away. pool_size=2 = old behavior.
        self.pool_size = pool_size
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        fusion_in = (ch2 + ch3 + ch4) * pool_size * pool_size
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Tier-2: adjacent-frame correlation volume on the layer2 maps (ch2), fused into the
        # pair encoder. use_corr=False keeps the old architecture so prior checkpoints load.
        self.use_corr = use_corr
        if use_corr:
            self.corr_encoder = CorrEncoder(max_disp=corr_disp, out_dim=corr_dim)

        # Spatial-scan SSM over layer-3, fused with the pooled feature; per-frame dim stays
        # hidden_size so pair_proj is unchanged and old checkpoints still load.
        self.use_spatial_ssm = use_spatial_ssm
        if use_spatial_ssm:
            self.spatial_ssm = SpatialSSM(ch3, hidden_size, mamba_d_state, mamba_d_conv, mamba_expand)
            self.combine_proj = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size),
                nn.ReLU(), nn.Dropout(0.2))

        if pair_encoder:
            in_dim = hidden_size * (1 + len(self.pair_strides))
            if use_corr:
                in_dim += corr_dim
            self.pair_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

        self.pos_encoder = PositionalEncoding(hidden_size, max_len=seq_len)
        def _temporal():
            if bidirectional:
                return BiMamba(hidden_size, mamba_d_state, mamba_d_conv, mamba_expand)
            return Mamba(d_model=hidden_size, d_state=mamba_d_state,
                         d_conv=mamba_d_conv, expand=mamba_expand)
        self.temporal_layers = nn.ModuleList([_temporal() for _ in range(num_layers)])

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

        if self.use_spatial_ssm:
            spat = f3.flatten(2).transpose(1, 2)              # [b*s, H3*W3, ch3]
            spat = self.spatial_ssm(spat).view(b, s, -1)      # [b, s, hidden]
            features = self.combine_proj(torch.cat([features, spat], dim=-1))

        if self.pair_encoder:
            S = features.shape[1]
            L = S - self._pair_max_stride
            f_curr = features[:, :L, :]
            parts = [f_curr]
            for stride in self.pair_strides:
                parts.append(features[:, stride:stride + L, :] - f_curr)
            if self.use_corr:
                C2, Hf, Wf = f2.shape[1], f2.shape[2], f2.shape[3]
                f2_seq = f2.view(b, s, C2, Hf, Wf)
                fa = f2_seq[:, :L].reshape(b * L, C2, Hf, Wf)        # frames 0..L-1
                fb = f2_seq[:, 1:1 + L].reshape(b * L, C2, Hf, Wf)   # frames 1..L (adjacent)
                parts.append(self.corr_encoder(fa, fb).view(b, L, -1))
            features = self.pair_proj(torch.cat(parts, dim=-1))

        features = self.pos_encoder(features)
        for layer in self.temporal_layers:
            features = layer(features)

        return self.head(features)  # [B, L, 6]
