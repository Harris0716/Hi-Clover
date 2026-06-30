# plot_from_log.py
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_log(log_path):
    text = Path(log_path).read_text(errors="ignore")

    pattern = re.compile(
        r"Epoch \[(\d+)\]\s+Val Triplet:\s+([0-9.]+).*?Log-Ratio:\s+([-+]?[0-9.]+)"
    )

    epochs, val_losses, log_ratios = [], [], []

    for m in pattern.finditer(text):
        epochs.append(int(m.group(1)))
        val_losses.append(float(m.group(2)))
        log_ratios.append(float(m.group(3)))

    if not epochs:
        raise ValueError("No 'Epoch [x] Val Triplet:' lines found in log.")

    return np.array(epochs), np.array(val_losses), np.array(log_ratios)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to train.log")
    parser.add_argument("--out", default=None, help="Output PDF/PNG path")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else log_path.parent / "loss_from_log_best_epoch.pdf"

    epochs, val_losses, log_ratios = parse_log(log_path)

    best_idx = int(np.argmin(val_losses))
    best_epoch = int(epochs[best_idx])
    best_val = float(val_losses[best_idx])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(epochs, val_losses, marker="o", label="Val Triplet")
    ax.axvline(
        best_epoch,
        linestyle="--",
        linewidth=2,
        label=f"Best epoch {best_epoch}"
    )
    ax.scatter(best_epoch, best_val, s=80, zorder=5)

    ax.set_title(f"Validation Triplet Loss\nBest Val={best_val:.4f} @ Epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation triplet loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val triplet: {best_val:.4f}")


if __name__ == "__main__":
    main()