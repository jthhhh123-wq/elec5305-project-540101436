# -*- coding: utf-8 -*-
"""Feature extraction utilities for KWS.

We use librosa to compute log-mel spectrograms.
Make sure to keep the same parameters between train/eval.
"""
from typing import Tuple
import numpy as np
import librosa
import torch

DEFAULT_SR = 16000
DEFAULT_N_MELS = 40
WIN_LENGTH = 400     # 25 ms at 16 kHz
HOP_LENGTH = 160     # 10 ms at 16 kHz

def waveform_to_logmel(
    y: np.ndarray, 
    sr: int = DEFAULT_SR, 
    n_mels: int = DEFAULT_N_MELS,
    center: bool = True
) -> torch.Tensor:
    """Convert a mono waveform to a normalized log-mel spectrogram.

    Args:
        y: numpy array, mono waveform in float32.
        sr: sampling rate.
        n_mels: number of mel bands.
    Returns:
        torch.Tensor of shape (1, n_mels, time_frames) normalized.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=n_mels, center=center, power=2.0
    )
    logS = librosa.power_to_db(S + 1e-10)
    mu, sigma = logS.mean(), logS.std() + 1e-9
    logS = (logS - mu) / sigma
    feat = torch.tensor(logS, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)
    return feat
