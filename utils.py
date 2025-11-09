# utils.py
import torch
import torchaudio
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

class LogMelSpec(torch.nn.Module):
    def __init__(self, sr=16000, n_fft=400, hop=160, n_mels=64):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, x):  # x: (B,1,T)
        m = self.mel(x)    # (B, n_mels, time)
        m = self.db(m + 1e-6)
        # per-sample, per-channel norm
        mean = m.mean(dim=(-1,-2), keepdim=True)
        std  = m.std(dim=(-1,-2), keepdim=True) + 1e-6
        return (m - mean) / std

def add_gaussian_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    x: (B,1,T) waveform in [-1,1]
    """
    if snr_db is None:
        return x
    sig_p = x.pow(2).mean(dim=(-1,-2), keepdim=True) + 1e-12
    noise = torch.randn_like(x)
    noise_p = noise.pow(2).mean(dim=(-1,-2), keepdim=True) + 1e-12
    alpha = torch.sqrt(sig_p / (noise_p * (10**(snr_db/10))))
    return x + alpha * noise

def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
    f1  = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="macro")
    return acc, f1, preds

def save_confusion_matrix(y_true, y_pred, labels, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)

def set_seed(seed: int = 1234):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
