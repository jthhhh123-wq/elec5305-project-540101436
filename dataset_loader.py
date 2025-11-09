# dataset_loader.py
import torch
import torchaudio
from torch.utils.data import DataLoader
import os
from typing import Tuple, List

SAMPLE_RATE = 16000
TARGET_LEN = SAMPLE_RATE  # 1s clips (Speech Commands are ~1s)

class SpeechCommandsSubset(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root: str, subset: str = "training"):
        super().__init__(root=root, download=True, subset=subset)
        self.labels = sorted(list(set([x[2] for x in self])))
        self.lab2id = {lab: i for i, lab in enumerate(self.labels)}

    def __getitem__(self, n):
        wav, sr, label, *_ = super().__getitem__(n)
        wav = wav.mean(0, keepdim=True)  # mono
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        # crop/pad to TARGET_LEN
        T = wav.shape[-1]
        if T < TARGET_LEN:
            wav = torch.nn.functional.pad(wav, (0, TARGET_LEN - T))
        else:
            wav = wav[..., :TARGET_LEN]
        y = self.lab2id[label]
        return wav, y

def collate_fn(batch):
    xs, ys = [], []
    for x, y in batch:
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)

def build_loaders(data_root: str, batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    tr = SpeechCommandsSubset(root=os.path.join(data_root), subset="training")
    va = SpeechCommandsSubset(root=os.path.join(data_root), subset="validation")
    te = SpeechCommandsSubset(root=os.path.join(data_root), subset="testing")
    # align label maps
    labels = tr.labels
    va.lab2id = tr.lab2id
    te.lab2id = tr.lab2id
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, labels
