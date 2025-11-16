import argparse, yaml, os, torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from .dataset_loader import make_loaders
from .model_se import ConvMixerSE
from .utils import set_seed, accuracy, ensure_dir


console = Console()

# -------- Mel-Spec --------
import torchaudio
mel_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=320, n_mels=64
)

def to_spec(waves):
    spec = mel_fn(waves)  # 输出: [B, 1, n_mels, T]
    spec = torchaudio.functional.amplitude_to_DB(
        spec,
        multiplier=10.0,
        amin=1e-10,
        db_multiplier=0.0,
    )
    return spec   # 直接返回，不要再 unsqueeze




# -------- One epoch --------
def train_one_epoch(model, opt, loader, device):
    model.train()
    ce = torch.nn.CrossEntropyLoss()

    n, loss_sum, acc_sum = 0, 0.0, 0.0

    pbar = tqdm(loader, desc="train", leave=False)
    for waves, targets in pbar:
        targets = targets.to(device)

        specs = to_spec(waves)

        specs = specs.to(device)

        opt.zero_grad()
        logits = model(specs)
        loss = ce(logits, targets)
        loss.backward()
        opt.step()

        bs = waves.size(0)
        loss_sum += loss.item() * bs
        acc_sum += (logits.argmax(1) == targets).float().sum().item()
        n += bs

        pbar.set_postfix({"loss": f"{loss_sum/n:.4f}", "acc": f"{acc_sum/n:.4f}"})

    return loss_sum/n, acc_sum/n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = torch.nn.CrossEntropyLoss()

    n, loss_sum, acc_sum = 0, 0.0, 0.0

    for waves, targets in tqdm(loader, desc="eval", leave=False):
        targets = targets.to(device)

        specs = to_spec(waves)

        specs = specs.to(device)

        logits = model(specs)
        loss = ce(logits, targets)

        bs = waves.size(0)
        loss_sum += loss.item() * bs
        acc_sum += (logits.argmax(1) == targets).float().sum().item()
        n += bs

    return loss_sum/n, acc_sum/n


# -------- Main --------
def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    train_loader, val_loader, test_loader, labels = make_loaders(
        root=args.data_dir, cfg=cfg, train_noise=True
    )

    model = ConvMixerSE(
        n_mels=cfg["n_mels"],
        n_classes=len(labels),
        dim=cfg["dim"],
        depth=cfg["depth"],
        kernel_size=cfg["kernel_size"],
        patch_size=cfg["patch_size"],
        dropout=cfg.get("dropout", 0.0)
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    ensure_dir(args.ckpt_dir)
    best_val = 0
    wait = 0
    patience = 5

    console.print(f"[cyan]Training SE + Reverb model[/cyan]")

    for epoch in range(1, cfg["epochs"]+1):
        train_loss, train_acc = train_one_epoch(model, opt, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        table = Table(title=f"Epoch {epoch}")
        table.add_column("split")
        table.add_column("loss")
        table.add_column("acc")
        table.add_row("train", f"{train_loss:.4f}", f"{train_acc:.4f}")
        table.add_row("valid", f"{val_loss:.4f}", f"{val_acc:.4f}")
        console.print(table)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": labels,
                    "cfg": cfg
                },
                os.path.join(args.ckpt_dir, f"{cfg['tag']}_best.pt")
            )
            console.print(f"[green]Saved new best model (acc={best_val:.4f})[/green]")
            wait = 0
        else:
            wait += 1
            if wait > patience:
                console.print(f"[yellow]Early stopping (best={best_val:.4f})[/yellow]")
                break

    console.print(f"[cyan]Testing best model...[/cyan]")
    best_model = torch.load(os.path.join(args.ckpt_dir, f"{cfg['tag']}_best.pt"))
    model.load_state_dict(best_model["state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, device)
    console.print(f"[bold green]Test acc = {test_acc:.4f}[/bold green]")


# -------- CLI --------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--config", type=str, default="./configs/se_reverb.yaml")
    p.add_argument("--ckpt_dir", type=str, default="./runs/se_reverb")
    args = p.parse_args()
    main(args)

