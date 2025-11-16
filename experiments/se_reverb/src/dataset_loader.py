import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import torch.nn.functional as F


# --------------------------
# Torchaudio split (no txt files needed)
# --------------------------
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, root, subset="training", download=True):
        super().__init__(root=root, download=download)
        self._walker = SPEECHCOMMANDS(root, subset=subset)._walker


def pad_trim(waveform, max_len):
    T = waveform.size(1)
    if T > max_len:
        return waveform[:, :max_len]
    if T < max_len:
        return torch.nn.functional.pad(waveform, (0, max_len - T))
    return waveform


def collate_fn(labels):
    mapping = {lb: i for i, lb in enumerate(labels)}

    def fn(batch):
        waves = []
        targets = []
        for waveform, sr, label, *_ in batch:
            if label not in mapping:
                label = "unknown"
            waves.append(waveform)
            targets.append(mapping[label])
        TARGET_LEN = 16000
        fixed_waves = []
        for w in waves:
            T = w.shape[-1]
            if T < TARGET_LEN:
                # 右侧补零
                pad_len = TARGET_LEN - T
                w = F.pad(w, (0, pad_len))
            elif T > TARGET_LEN:
                # 太长就截断
                w = w[..., :TARGET_LEN]
            fixed_waves.append(w)

        waves = torch.stack(fixed_waves)
        targets = torch.tensor(targets, dtype=torch.long)
        return waves, targets

    return fn


def make_loaders(root, cfg, train_noise=False, snr_db=20):
    labels = cfg["labels"]
    sr = cfg.get("sample_rate", 16000)
    bs = cfg["batch_size"]
    nw = cfg["num_workers"]

    # datasets
    train_set = SubsetSC(root, "training")
    val_set   = SubsetSC(root, "validation")
    test_set  = SubsetSC(root, "testing")

    # collate
    collate = collate_fn(labels)

    # loaders
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate)
    val_loader   = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate)
    test_loader  = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate)

    # -------------------------------
    # RETURN EXACTLY 4 VALUES
    # -------------------------------
    return train_loader, val_loader, test_loader, labels



