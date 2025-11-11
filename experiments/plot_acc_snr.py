# experiments/plot_acc_snr.py
# -*- coding: utf-8 -*-
"""
Read runs/acc_snr.csv and plot SNR–Accuracy curves.
CSV columns (with header): tag,snr,loss,acc
Example row: baseline,20,0.1633,0.9463
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Try reading with header; if no header, add it
    try:
        df = pd.read_csv(csv_path)
        if set(["tag", "snr", "loss", "acc"]).issubset(df.columns):
            return df[["tag", "snr", "loss", "acc"]]
        # Fallback: no header case
        df = pd.read_csv(csv_path, header=None, names=["tag", "snr", "loss", "acc"])
        return df
    except Exception:
        # Robust fallback
        df = pd.read_csv(csv_path, header=None, names=["tag", "snr", "loss", "acc"])
        return df

def plot_acc_snr(df: pd.DataFrame, out_path: str, title: str = "SNR vs Accuracy"):
    # Normalize types
    df["snr"]  = pd.to_numeric(df["snr"], errors="coerce")
    df["acc"]  = pd.to_numeric(df["acc"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["snr", "acc"])

    # Sort by SNR inside each tag so lines are not zigzag
    df = df.sort_values(by=["tag", "snr"])

    plt.figure(figsize=(7, 5), dpi=140)

    # Plot each tag as a separate curve
    for tag, sub in df.groupby("tag"):
        plt.plot(
            sub["snr"], sub["acc"],
            marker="o", linewidth=2, markersize=5, label=str(tag)
        )

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(title="Experiment", loc="best")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"✅ Figure saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="runs/acc_snr.csv",
                        help="Path to CSV with columns tag,snr,loss,acc")
    parser.add_argument("--out", type=str, default="runs/acc_snr.png",
                        help="Output image path")
    parser.add_argument("--title", type=str, default="SNR vs Accuracy",
                        help="Figure title")
    args = parser.parse_args()

    df = load_csv(args.csv)
    if df.empty:
        raise RuntimeError("CSV is empty or invalid.")
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(df.head())

    plot_acc_snr(df, args.out, args.title)

if __name__ == "__main__":
    main()
