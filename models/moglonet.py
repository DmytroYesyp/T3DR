import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class MoGLoDataset(Dataset):
    def __init__(self, frames, tforms, seq_len=5):
        self.frames = frames
        self.tforms = tforms
        self.seq_len = seq_len
        self.samples = []
        
        real_z = tforms[:, 2, 3]
        
        for i in range(0, len(frames) - seq_len):
            if abs(real_z[i+seq_len] - real_z[i]) < 20.0:
                self.samples.append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start = self.samples[idx]
        
        seq_imgs = []
        for i in range(self.seq_len):
            img = self.frames[start + i]
            if img.max() > 1.0: img = img / 255.0
            seq_imgs.append(np.expand_dims(img, axis=0))
            
        seq_tensor = torch.tensor(np.stack(seq_imgs), dtype=torch.float32)
        
        z_prev = self.tforms[start + self.seq_len - 1, 2, 3]
        z_curr = self.tforms[start + self.seq_len, 2, 3]
        target = abs(z_curr - z_prev)
        
        return seq_tensor, torch.tensor(target, dtype=torch.float32)

class CorrelationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        prod = f1 * f2           
        
        return torch.cat([diff, prod], dim=1)

class GlobalLocalAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MoGLoNet(nn.Module):
    def __init__(self, seq_len=5, hidden_size=256):
        super().__init__()
        self.seq_len = seq_len
        
        # Backbone (ResNet-18)
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        
        self.corr = CorrelationLayer()
        self.att = GlobalLocalAttention(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(input_size=384, hidden_size=hidden_size, num_layers=2, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        b, s, c, h, w = x.size()
        
        x_flat = x.view(b * s, c, h, w)
        features = self.encoder(x_flat) 
        features = features.view(b, s, 128, features.size(2), features.size(3))
        
        lstm_inputs = []
        
        for t in range(1, s):
            f_prev = features[:, t-1]
            f_curr = features[:, t]
            
            corr_map = self.corr(f_prev, f_curr) 
            
            att_feat = self.att(f_curr)
            
            combined = torch.cat([att_feat, corr_map], dim=1) 
            
            vector = self.pool(combined).view(b, -1) 
            lstm_inputs.append(vector)
            
        lstm_input_seq = torch.stack(lstm_inputs, dim=1)
        
        lstm_out, _ = self.lstm(lstm_input_seq)
        final_feat = lstm_out[:, -1, :]
        
        out = self.fc(final_feat)
        return out.squeeze()