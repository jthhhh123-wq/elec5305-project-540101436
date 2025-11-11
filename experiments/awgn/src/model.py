import torch
import torch.nn as nn

class ConvMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size, dropout=0.0):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.pwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.dwconv(x)
        x = self.pwconv(x)
        x = self.dropout(x)
        return x

class ConvMixer(nn.Module):
    """
    ConvMixer backbone for spectrograms.
    Input spec shape: (B, 1, n_mels, T)
    """
    def __init__(self, n_mels=64, n_classes=12, dim=256, depth=8, kernel_size=7, patch_size=4, dropout=0.0):
        super().__init__()
        # Patch embedding: stride=kernel=patch_size to reduce time resolution
        self.embed = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.blocks = nn.Sequential(*[
            ConvMixerBlock(dim, kernel_size, dropout=dropout) for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):  # x: (B,1,M,T)
        x = self.embed(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
