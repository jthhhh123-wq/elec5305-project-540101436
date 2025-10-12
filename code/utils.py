# -*- coding: utf-8 -*-
"""Utility functions for KWS experiments (noise mixing, plotting, etc.)."""
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import os

def set_seed(seed: int = 42):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))

def mix_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise into clean audio to reach the target SNR (in dB).

    Args:
        clean: mono waveform float32
        noise: mono waveform float32
        snr_db: desired SNR in dB (20/10/0 etc.)
    """
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean)/len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]

    a = rms(clean)
    b = rms(noise)
    if b < 1e-12:
        return clean.copy()

    target_b = a / (10 ** (snr_db / 20.0))
    scale = target_b / b
    noisy = clean + scale * noise

    maxv = np.max(np.abs(noisy)) + 1e-9
    if maxv > 1.0:
        noisy = noisy / maxv
    return noisy.astype(np.float32)

def save_lineplot(xs, ys, xlabel, ylabel, title, savepath):
    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=200)
    plt.close()

def plot_confusion(cm, classes, title, savepath):
    import itertools
    cm = cm.astype('float')
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    plt.figure(figsize=(6,5))
    plt.imshow(cm_norm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    thresh = cm_norm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = f"{cm_norm[i, j]*100:.1f}"
        plt.text(j, i, val, ha='center', va='center',
                 color='white' if cm_norm[i, j] > thresh else 'black', fontsize=7)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=200)
    plt.close()
