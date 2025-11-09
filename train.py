# train.py
import argparse, os, torch
from tqdm import tqdm
from dataset_loader import build_loaders
from utils import LogMelSpec, metrics_from_logits, set_seed
from model import ConvMixerKWS

def train_one_epoch(model, loader, feat, opt, device):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device).float()
        y = y.to(device)
        mel = feat(x)
        logits = model(mel)
        loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)

@torch.no_grad()
def evaluate(model, loader, feat, device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    ys, preds = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device).float()
        y = y.to(device)
        mel = feat(x)
        logits = model(mel)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        _, _, p = metrics_from_logits(logits, y)
        ys.append(y.cpu()); preds.append(p.cpu())
    ys = torch.cat(ys).numpy(); preds = torch.cat(preds).numpy()
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(ys, preds)
    f1  = f1_score(ys, preds, average="macro")
    return total_loss/max(1,n), acc, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./_data")
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--mode", choices=["baseline","improved"], default="baseline")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, labels = build_loaders(args.data_root, batch_size=args.batch_size)
    feat  = LogMelSpec(n_mels=args.n_mels).to(device)
    model = ConvMixerKWS(n_mels=args.n_mels, n_classes=len(labels), dim=args.dim, depth=args.depth).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    save_dir = os.path.join(args.results_root, args.mode)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "model_best.pth")
    report_txt = os.path.join(save_dir, "val_log.txt")

    best_acc = 0.0
    with open(report_txt, "w", encoding="utf-8") as f:
        for ep in range(1, args.epochs+1):
            tr_loss = train_one_epoch(model, train_loader, feat, opt, device)
            va_loss, va_acc, va_f1 = evaluate(model, val_loader, feat, device)
            line = f"[{ep:02d}] train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f} | val_f1={va_f1:.4f}"
            print(line); f.write(line+"\n"); f.flush()
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({"state_dict": model.state_dict(), "labels": labels, "cfg": vars(args)}, best_path)

    print("Best val acc:", best_acc)
    print("Best checkpoint saved to:", best_path)

if __name__ == "__main__":
    main()
