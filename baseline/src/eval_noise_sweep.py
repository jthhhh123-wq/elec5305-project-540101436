# baseline/src/eval_noise_sweep.py 
import torch, argparse, yaml
from .dataset_loader import make_loaders
from .model import ConvMixer
from .utils import ensure_dir
print(">>> EVAL NOISE SWEEP START <<<")

@torch.no_grad()
def eval_snr(model, loader, to_spec, device, snr_db):
    ce = torch.nn.CrossEntropyLoss()
    model.eval()
    n, loss_sum, acc_sum = 0, 0.0, 0.0

    for waves, targets in loader:
        targets = targets.to(device)

        noise = torch.randn_like(waves)
        noise_power = noise.pow(2).mean()
        sig_power = waves.pow(2).mean()
        scale = (sig_power / (10 ** (snr_db / 10)) / noise_power).sqrt()
        noisy = waves + scale * noise

        specs = to_spec(noisy)                       # CPU
        specs = specs.to(torch.float32).to(device)   # -> GPU

        logits = model(specs)
        loss = ce(logits, targets)

        bs = waves.size(0)
        loss_sum += loss.item() * bs
        acc_sum  += (logits.argmax(1) == targets).float().sum().item()
        n += bs

    return loss_sum / n, acc_sum / n


def main():
    import argparse, yaml, torch, os, csv
    from .dataset_loader import make_loaders
    from .model import ConvMixer
    from .eval_noise_sweep import eval_snr 

    # === Parse arguments ===
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--config", type=str, default="configs/baseline.yaml")
    p.add_argument("--ckpt", type=str, default="runs/baseline_gpu_25ep/baseline_best.pt")
    args = p.parse_args()

    # === Load configuration and model ===
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    labels = ckpt["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load test dataset ===
    _, _, test_loader, _, eval_tf, _ = make_loaders(root=args.data_dir, cfg=cfg)

    # === Build model ===
    model = ConvMixer(**cfg["convmixer"], n_classes=len(labels)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # === Evaluate different SNRs ===
    snrs = [20, 10, 0, -5]
    print("Evaluating noise robustness...")
    results = {}

    for snr in snrs:
        loss, acc = eval_snr(model, test_loader, eval_tf, device, snr)
        print(f"SNR={snr:>3} dB | Test loss={loss:.4f} | acc={acc:.4f}")
        results[snr] = (loss, acc)

    # === Append results to CSV ===
    out_csv = os.path.join("..", "runs", "acc_snr.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.config))[0]

    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        for snr, (loss, acc) in results.items():
            w.writerow([tag, snr, round(loss, 4), round(acc, 4)])

    print(f"\n✅ Results appended to {out_csv}")

if __name__ == "__main__":
    main()




