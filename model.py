# model.py
import torch
import torch.nn as nn

class ConvMixerBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 5):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel, padding=kernel//2, groups=dim, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = x + self.dw(x)
        return self.pw(x)

class ConvMixerKWS(nn.Module):
    def __init__(self, n_mels=64, n_classes=12, dim=128, depth=6, patch_time=1, patch_mel=1):
        super().__init__()
        # input: mel (B, n_mels, time) -> treat as image (1, H=n_mels, W=time)
        self.embed = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=(patch_mel, patch_time), stride=(patch_mel, patch_time), bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.blocks = nn.Sequential(*[ConvMixerBlock(dim, kernel=5) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, mel):   # mel: (B, n_mels, time)
        x = mel.unsqueeze(1)  # -> (B,1,H,W)
        x = self.embed(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
