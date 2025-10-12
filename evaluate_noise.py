# -*- coding: utf-8 -*-
"""
Noise Sensitivity Evaluation for Keyword Spotting (KWS)

This script:
1) Loads a trained baseline CNN.
2) Adds Gaussian noise at different SNRs to the input features.
3) Evaluates accuracy vs. SNR and (optionally) saves confusion matrices.

Outputs:
- <results_dir>/accuracy_vs_snr.png
- <results_dir>/confusion_30dB.png, ... (if --plot_confusion_all True)
"""

import os
import glob
import argparse
from typing import List

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from features import waveform_to_logmel, DEFAULT_SR
from model import KWSCNN


# ---------------------------
# Dataset (copied from training script)
# ---------------------------
class SpeechCommandsSubset(Dataset):
    """Load a subset of Speech Commands by class folders."""
    def __init__(self, root: str, classes: List[str]):
        self.items = []
        for cls in classes:
            files = glob.glob(os.path.join(root, cls, '*.wav'))
            self.items += [(fp, cls) for fp in files]
        if not self.items:
            raise RuntimeError("No wav files found. Check data_root and classes.")
        self.classes = classes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fp, cls = self.items[idx]
        y, sr = sf.read(fp)
        if sr != DEFAULT_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=DEFAULT_SR)
        y = y.astype(np.float32)
        feat = waveform_to_logmel(y)     # (1, n_mels, T)
        x = feat.unsqueeze(0)            # (1, 1, n_mels, T)
        y_id = self.classes.index(cls)
        return x.squeeze(0), y_id        # (1, n_mels, T), label


def collate_fn(batch):
    """Pad along time dimension to the longest in the batch."""
    xs, ys = zip(*batch)
    maxT = max(x.shape[-1] for x in xs)
    padded = []
    for x in xs:
        if x.shape[-1] < maxT:
            pad = maxT - x.shape[-1]
            # pad last dimension (time)
            x = torch.nn.functional.pad(x, (0, pad))
        padded.append(x)
    X = torch.stack(padded, dim=0)  # (B, 1, n_mels, T)
    Y = torch.tensor(ys, dtype=torch.long)
    return X, Y


# ---------------------------
# Evaluation with noise
# ---------------------------
@torch.no_grad()
def test_with_noise(model, loader, snr_db, device):
    """
    Evaluate model performance with additive Gaussian noise at a given SNR (dB).

    SNR(dB) = 10 * log10(P_signal / P_noise)
    noise_scale is computed to match the target SNR per batch.
    """
    model.eval()
    y_true, y_pred = [], []

    for X, Y in tqdm(loader, desc=f"SNR={snr_db} dB"):
        X = X.to(device)
        Y = Y.to(device)

        # Generate Gaussian noise
        noise = torch.randn_like(X)

        # Compute signal & noise power for scaling
        # (use mean over batch and all dims)
        signal_power = X.pow(2).mean()
        noise_power = signal_power / (10.0 ** (snr_db / 10.0) + 1e-12)
        scaled_noise = noise * torch.sqrt(noise_power / (noise.pow(2).mean() + 1e-12))

        noisy_X = X + scaled_noise

        logits = model(noisy_X)
        preds = torch.argmax(logits, dim=-1)

        y_true.extend(Y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred, classes, snr_level, out_dir="results"):
    """Plot and save confusion matrix for the given SNR level."""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=1)
    plt.title(f"Confusion Matrix at {snr_level} dB")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"confusion_{snr_level}dB.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--classes', type=str, nargs='+', required=True)
    parser.add_argument('--snrs', type=float, nargs='+', default=[30, 20, 10, 0, -5])
    parser.add_argument('--plot_confusion_all', type=bool, default=True)
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    # Device & model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KWSCNN(num_classes=len(args.classes)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Data
    dataset = SpeechCommandsSubset(args.data_root, args.classes)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Eval
    accuracies = []
    os.makedirs(args.results_dir, exist_ok=True)

    for snr_level in args.snrs:
        y_true, y_pred = test_with_noise(model, loader, snr_level, device)
        acc = np.mean(y_true == y_pred)
        accuracies.append(acc)
        print(f"SNR={snr_level:>4} dB  Accuracy={acc * 100:.2f}%  (N={len(y_true)})")

        if args.plot_confusion_all:
            plot_confusion_matrix(y_true, y_pred, args.classes, snr_level, args.results_dir)

    # Accuracy vs SNR
    plt.figure(figsize=(6, 4))
    plt.plot(args.snrs, np.array(accuracies) * 100.0, marker='o')
    plt.title("Accuracy vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "accuracy_vs_snr.png"))
    plt.close()

    print("All results saved to:", args.results_dir)


if __name__ == "__main__":
    main()
