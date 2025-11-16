import torch
import torch.nn as nn


# ---------------------------
# SE Block
# ---------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ---------------------------
# ConvMixer Block (with SE)
# ---------------------------
class ConvMixerBlockSE(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.se = SEBlock(dim)

        self.pwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = x + self.se(self.block(x))   # SE 插在 depthwise conv 后
        return self.pwconv(x)


# ---------------------------
# ConvMixer + SE model
# ---------------------------
class ConvMixerSE(nn.Module):
    def __init__(self, n_mels, n_classes, dim, depth, kernel_size, patch_size, dropout=0.0):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

        blocks = []
        for _ in range(depth):
            blocks.append(ConvMixerBlockSE(dim, kernel_size))

        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x



