import os, random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

