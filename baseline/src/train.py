import argparse, yaml, os
import torch
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from .dataset_loader import make_loaders
from .model import ConvMixer
from .utils import set_seed, accuracy, ensure_dir

console = Console()

import os, torch

def to_spec_batch(waves, to_spec, chunk=16):
    """
    waves: (B, 1, T) on CPU or GPU
    returns: (B, 1, M, T') float32 contiguous (on CPU)
    """
    waves = waves.cpu()
    out = []
    with torch.no_grad():
        for i in range(0, waves.size(0), chunk):
            part = waves[i:i+chunk]              # (b,1,T)
            s = to_spec(part)                    # (b,M,T')  torchaudio 支持小批量
            if s.dim() == 3:
                s = s.unsqueeze(1)               # (b,1,M,T')
            out.append(s)
    specs = torch.cat(out, dim=0).to(torch.float32).contiguous()
    return specs

def train_one_epoch(model, opt, loader, to_spec, device):
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    n, loss_sum, acc_sum = 0, 0.0, 0.0
    for waves, targets in tqdm(loader, desc="train", leave=False):
        targets = targets.to(device)
        specs = to_spec_batch(waves, to_spec)  # CPU 上算特征
        specs = specs.to(device)  # 再搬到 GPU
        logits = model(specs)
        loss = ce(logits, targets)

        opt.zero_grad(); loss.backward(); opt.step()

        bs = waves.size(0)
        loss_sum += loss.item() * bs
        acc_sum  += (logits.argmax(1) == targets).float().sum().item()
        n += bs
    return loss_sum / n, acc_sum / n

@torch.no_grad()
def evaluate(model, loader, to_spec, device):
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    n, loss_sum, acc_sum = 0, 0.0, 0.0
    for waves, targets in tqdm(loader, desc="eval", leave=False):
        targets = targets.to(device)
        specs = to_spec_batch(waves, to_spec)  # 先 CPU
        specs = specs.to(device)  # 再搬到 GPU
        logits = model(specs)
        loss = ce(logits, targets)

        bs = waves.size(0)
        loss_sum += loss.item() * bs
        acc_sum  += (logits.argmax(1) == targets).float().sum().item()
        n += bs
    return loss_sum / n, acc_sum / n

def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # data & loaders
    train_loader, valid_loader, test_loader, train_tf, eval_tf, labels = make_loaders(
        root=args.data_dir, cfg=cfg, train_noise=args.train_noise, snr_db=args.snr_db
    )

    # model
    cm = cfg["convmixer"]
    model = ConvMixer(
        n_mels=cfg.get("n_mels", 64),
        n_classes=len(labels),
        dim=cm["dim"],
        depth=cm["depth"],
        kernel_size=cm["kernel_size"],
        patch_size=cm["patch_size"],
        dropout=cm.get("dropout", 0.0)
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    ensure_dir(args.ckpt_dir)
    best_val = 0.0

    # ---- Early Stopping 设置 ----
    best_val = 0.0
    patience = 5  # 容忍 5 个 epoch 无提升（你可以改成 3~7）
    wait = 0

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, opt, train_loader, train_tf, device)
        va_loss, va_acc = evaluate(model, valid_loader, eval_tf, device)

        table = Table(title=f"Epoch {epoch}")
        table.add_column("split");
        table.add_column("loss");
        table.add_column("acc")
        table.add_row("train", f"{tr_loss:.4f}", f"{tr_acc:.4f}")
        table.add_row("valid", f"{va_loss:.4f}", f"{va_acc:.4f}")
        console.print(table)

        # Early stopping 检查
        if va_acc > best_val:
            best_val = va_acc
            torch.save(
                {"state_dict": model.state_dict(), "labels": labels, "cfg": cfg},
                os.path.join(args.ckpt_dir, "baseline_best.pt")
            )
            console.print(f"[green]Saved best checkpoint (val_acc={best_val:.4f})[/green]")
            wait = 0  # 重置等待计数
        else:
            wait += 1
            if wait > patience:
                console.print(f"[yellow]Early stopping at epoch {epoch} (best val_acc={best_val:.4f})[/yellow]")
                break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--config", type=str, default="configs/baseline.yaml")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--train_noise", action="store_true", help="add AWGN during training")
    p.add_argument("--snr_db", type=float, default=20.0)
    args = p.parse_args()
    main(args)
