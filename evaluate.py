# evaluate.py
import argparse, os, torch, csv
from tqdm import tqdm
from dataset_loader import build_loaders
from utils import LogMelSpec, add_gaussian_snr, save_confusion_matrix
from model import ConvMixerKWS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

@torch.no_grad()
def eval_snr(model, loader, feat, device, snr_db=None):
    model.eval()
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc=f"eval snr={snr_db}", leave=False):
        x = x.float()
        if snr_db is not None:
            x = add_gaussian_snr(x, snr_db)
        x = x.to(device)
        logits = model(feat(x))
        y_true.append(y); y_pred.append(torch.argmax(logits,1).cpu())
    y_true = torch.cat([t.unsqueeze(0) for t in y_true]).flatten().numpy()
    y_pred = torch.cat([p for p in y_pred]).numpy()
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./_data")
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--mode", choices=["baseline","improved"], default="baseline")
    ap.add_argument("--ckpt", default=None, help="path to model_best.pth; default auto-resolve by mode")
    ap.add_argument("--snr_list", default="35,30,25,20,15,10,5,0,-5")
    ap.add_argument("--n_mels", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader, labels = build_loaders(args.data_root, batch_size=128)
    feat = LogMelSpec(n_mels=args.n_mels).to(device)

    if args.ckpt is None:
        args.ckpt = os.path.join(args.results_root, args.mode, "model_best.pth")
    ckpt = torch.load(args.ckpt, map_location=device)

    model = ConvMixerKWS(n_mels=args.n_mels, n_classes=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    save_dir = os.path.join(args.results_root, args.mode)
    os.makedirs(save_dir, exist_ok=True)

    # Clean accuracy + confusion matrix
    acc_clean, y_true, y_pred = eval_snr(model, test_loader, feat, device, snr_db=None)
    print(f"[CLEAN] acc={acc_clean:.4f}")
    save_confusion_matrix(y_true, y_pred, labels, os.path.join(save_dir, "confusion_matrix_clean.png"))

    # SNR curve
    snrs = [int(s) for s in args.snr_list.split(",")]
    rows = []
    accs = []
    for s in snrs:
        acc, _, _ = eval_snr(model, test_loader, feat, device, snr_db=s)
        print(f"[SNR {s:>3} dB] acc={acc:.4f}")
        rows.append([s, f"{acc:.4f}"])
        accs.append(acc)

    # save CSV
    csv_path = os.path.join(save_dir, "accuracy_vs_snr.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["snr_db","accuracy"]); w.writerows(rows)
    print("Saved:", csv_path)

    # plot
    fig = plt.figure(figsize=(6,4))
    plt.plot(snrs, accs, marker="o")
    plt.gca().invert_xaxis()  # higher SNR on left to right? flip for nicer look
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs SNR (Gaussian)")
    plt.grid(True)
    plt.tight_layout()
    png_path = os.path.join(save_dir, "accuracy_vs_snr.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", png_path)

if __name__ == "__main__":
    main()
