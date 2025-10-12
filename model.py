# -*- coding: utf-8 -*-
"""
Simple CNN model for Keyword Spotting (size-agnostic via global average pooling).

Input:  (B, 1, n_mels, T)  log-mel spectrogram
Output: logits over num_classes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KWSCNN(nn.Module):
    def __init__(self, num_classes: int = 10, n_mels: int = 40):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.25)

        # Make the representation length-invariant
        self.gap = nn.AdaptiveAvgPool2d((1, 1))   # -> (B, 32, 1, 1)

        # After GAP we always have 32 features, regardless of time frames
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B,16, n_mels/2, T/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B,32, n_mels/4, T/4)
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = self.gap(x)             # (B,32,1,1)
        x = torch.flatten(x, 1)     # (B,32)
        x = F.relu(self.fc1(x))     # (B,128)
        x = self.dropout(x)
        logits = self.fc2(x)        # (B,num_classes)
        return logits
