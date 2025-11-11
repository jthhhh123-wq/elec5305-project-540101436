import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

def _load_list(root, filename):
    path = os.path.join(root, filename)
    with open(path) as f:
        return [os.path.join(root, line.strip()) for line in f]

class SubsetSC(SPEECHCOMMANDS):
    """
    Same split rule as torchaudio tutorial: use provided txt lists.
    """
    def __init__(self, root, subset: str = "training", download=True):
        super().__init__(root=root, download=download)
        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = _load_list(self._path, "testing_list.txt")
        elif subset == "training":
            excludes = set(_load_list(self._path, "validation_list.txt") + _load_list(self._path, "testing_list.txt"))
            self._walker = [w for w in self._walker if w not in excludes]
        else:
            raise ValueError("subset must be 'training' | 'validation' | 'testing'")

def pad_trim(waveform: torch.Tensor, max_len: int):
    # waveform: (1, T)
    T = waveform.size(1)
    if T > max_len:
        return waveform[:, :max_len]
    if T < max_len:
        return torch.nn.functional.pad(waveform, (0, max_len - T))
    return waveform

def add_awgn(waveform: torch.Tensor, snr_db: float = 20.0):
    sig_p = waveform.pow(2).mean().clamp_min(1e-8)
    snr = 10 ** (snr_db / 10.0)
    noise_p = sig_p / snr
    noise = torch.randn_like(waveform) * noise_p.sqrt()
    return waveform + noise

class ToLogMel:
    def __init__(self, sample_rate=16000, n_mels=64, add_noise=False, snr_db=20.0):
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.add_noise = add_noise
        self.snr_db = snr_db
        self.max_len = sample_rate  # 1s clips

    def __call__(self, waveform: torch.Tensor):
        waveform = pad_trim(waveform, self.max_len)
        if self.add_noise:
            waveform = add_awgn(waveform, self.snr_db)
        spec = self.mel(waveform)          # (n_mels, T)
        spec = self.db(spec)
        return spec

def _label_to_index(word, label_list):
    try:
        return label_list.index(word)
    except ValueError:
        return label_list.index("unknown")

def collate_fn_factory(labels, sample_rate=16000):
    max_len = sample_rate
    def _fn(batch):
        waves, targets = [], []
        for waveform, sr, label, *_ in batch:
            waveform = pad_trim(waveform, max_len)
            waves.append(waveform)
            targets.append(_label_to_index(label, labels))
        waves = torch.stack(waves)                           # (B, 1, T)
        targets = torch.tensor(targets, dtype=torch.long)    # (B,)
        return waves, targets
    return _fn

def make_loaders(root, cfg, train_noise=False, snr_db=20.0):
    labels = cfg["labels"]
    sr = cfg.get("sample_rate", 16000)
    n_mels = cfg.get("n_mels", 64)

    train_set = SubsetSC(root, "training")
    valid_set = SubsetSC(root, "validation")
    test_set  = SubsetSC(root, "testing")

    train_tf = ToLogMel(sample_rate=sr, n_mels=n_mels, add_noise=train_noise, snr_db=snr_db)
    eval_tf  = ToLogMel(sample_rate=sr, n_mels=n_mels, add_noise=False)

    collate = collate_fn_factory(labels, sample_rate=sr)

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, collate_fn=collate, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=0, collate_fn=collate, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=0, collate_fn=collate, pin_memory=True)
    return train_loader, valid_loader, test_loader, train_tf, eval_tf, labels

