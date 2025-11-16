# experiments/se_reverb/src/eval_noise_sweep.py

import os
import csv
import argparse
import yaml
import torch
import torchaudio

from .dataset_loader import make_loaders
from .model_se import ConvMixerSE
from .utils import ensure_dir

print(">>> EVAL NOISE SWEEP START <<<")

mel_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=320,
    n_mels=64,
)


def to_spec(waves: torch.Tensor) -> torch.Tensor:
    """
    waves: [B, 1, T]  (在 CPU 上)
    return: [B, 1, n_mels, T']  (在 CPU 上)
    """
    spec = mel_fn(waves)  # [B, 1, n_mels, T']
    spec = torchaudio.functional.amplitude_to_DB(
        spec,
        multiplier=10.0,
        amin=1e-10,
        db_multiplier=0.0,
    )
    return spec


@torch.no_grad()
def eval_snr(model, loader, device, snr_db: float):
    ce = torch.nn.CrossEntropyLoss()
    model.eval()

    n, loss_sum, acc_sum = 0, 0.0, 0.0

    for waves, targets in loader:
      
        targets = targets.to(device)

  
        noise = torch.randn_like(waves)
        noise_power = noise.pow(2).mean()
        sig_power = waves.pow(2).mean()
        scale = (sig_power / (10 ** (snr_db / 10)) / noise_power).sqrt()
        noisy = waves + scale * noise  # 仍在 CPU 上

 
        specs = to_spec(noisy)
        specs = specs.to(torch.float32).to(device)

        logits = model(specs)
        loss = ce(logits, targets)

        bs = waves.size(0)
        loss_sum += loss.item() * bs
        acc_sum += (logits.argmax(1) == targets).float().sum().item()
        n += bs

    return loss_sum / n, acc_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--config", type=str, default="./configs/se_reverb.yaml")
    p.add_argument("--ckpt", type=str, default="./runs/se_reverb/se_reverb_best.pt")
    p.add_argument("--out_csv", type=str, default="../../runs/acc_snr.csv")
    args = p.parse_args()


    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

 ）
    _, _, test_loader, labels = make_loaders(
        root=args.data_dir,
        cfg=cfg,
        train_noise=False,  
    )

    
    model = ConvMixerSE(
        n_mels=cfg["n_mels"],
        n_classes=len(labels),
        dim=cfg["dim"],
        depth=cfg["depth"],
        kernel_size=cfg["kernel_size"],
        patch_size=cfg["patch_size"],
        dropout=cfg.get("dropout", 0.0),
    ).to(device)


    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

 
    snrs = [20, 10, 0, -5]
    print("Evaluating noise robustness...")
    results = {}

    for snr in snrs:
        loss, acc = eval_snr(model, test_loader, device, snr)
        print(f"SNR={snr:>3} dB | Test loss={loss:.4f} | acc={acc:.4f}")
        results[snr] = (loss, acc)

 
    ensure_dir(os.path.dirname(args.out_csv))
    tag = cfg.get("tag", os.path.splitext(os.path.basename(args.config))[0])

    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        for snr, (loss, acc) in results.items():
            w.writerow([tag, snr, round(loss, 4), round(acc, 4)])

    print(f"\n✅ Results appended to {args.out_csv}")


if __name__ == "__main__":
    main()




