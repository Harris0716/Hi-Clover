#!/usr/bin/env python3
"""
Plot training statistics from three (or more) *_history.npz files into one grid figure.

Expected npz keys from train_0615.py:
  train_losses, val_losses, val_log_ratio_history,
  grad_norm_backbone_history, lr_history,
  optional: best_epoch, best_val_loss

Example:
  python plot_three_dataset_training_grid.py \
    --npz Liver=outputs/liver_B_new/liver_B_new_history.npz \
    --npz NPC=outputs/NPC_B_new/NPC_B_new_history.npz \
    --npz "T Cell=outputs/TCell_B_new/TCell_B_new_history.npz" \
    --out three_dataset_training_stats.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_array(hist, key, default=None):
    """Read an array from npz. If missing, return default or raise error."""
    if key in hist.files:
        arr = np.asarray(hist[key])
        # convert scalar array to length-1 array if needed
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr
    if default is not None:
        return np.asarray(default)
    raise KeyError(f"Missing key '{key}' in npz. Available keys: {hist.files}")


def parse_labeled_npz(item):
    """Parse Label=path format."""
    if "=" not in item:
        path = Path(item)
        return path.stem.replace("_history", ""), path
    label, path = item.split("=", 1)
    return label.strip(), Path(path.strip())


def align_epochs(*arrays):
    """Use the shortest non-empty length so curves align safely."""
    lengths = [len(a) for a in arrays if a is not None and len(a) > 0]
    if not lengths:
        raise ValueError("No non-empty arrays found.")
    n = min(lengths)
    return np.arange(1, n + 1), [a[:n] if a is not None and len(a) > 0 else None for a in arrays]


def set_common_style(ax):
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(False)


def main():
    parser = argparse.ArgumentParser(
        description="Combine training history npz files into a multi-row training-statistics figure."
    )
    parser.add_argument(
        "--npz",
        action="append",
        required=True,
        help="Input as Label=path_to_history.npz. Use this option multiple times.",
    )
    parser.add_argument("--out", default="training_stats_grid.pdf", help="Output figure path, pdf/png supported.")
    parser.add_argument("--title", default=None, help="Optional overall figure title.")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Reference line for gradient clipping max norm.")
    parser.add_argument(
        "--mark_best",
        action="store_true",
        help="Mark best validation epoch on the loss plot.",
    )
    parser.add_argument(
        "--lr_log",
        action="store_true",
        help="Use log scale for learning rate axis.",
    )
    args = parser.parse_args()

    labeled_paths = [parse_labeled_npz(x) for x in args.npz]
    n_rows = len(labeled_paths)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 2.7 * n_rows),
        squeeze=False,
    )

    col_titles = [
        "Loss Evolution",
        "Log-Ratio (log(d(a,n)/d(a,p)))",
        "Gradient Norm",
        "Learning Rate",
    ]

    for row_idx, (label, npz_path) in enumerate(labeled_paths):
        if not npz_path.exists():
            raise FileNotFoundError(f"Cannot find npz file: {npz_path}")

        hist = np.load(npz_path, allow_pickle=True)

        train_losses = read_array(hist, "train_losses")
        val_losses = read_array(hist, "val_losses")
        log_ratio = read_array(hist, "val_log_ratio_history")
        grad_norm = read_array(hist, "grad_norm_backbone_history")
        lr_history = read_array(hist, "lr_history")

        epochs, (train_losses, val_losses, log_ratio, grad_norm, lr_history) = align_epochs(
            train_losses, val_losses, log_ratio, grad_norm, lr_history
        )

        # Left-side dataset label
        fig.text(
            0.035,
            1 - (row_idx + 0.5) / n_rows,
            label,
            ha="right",
            va="center",
            fontsize=22,
        )

        # 1. Loss
        ax = axes[row_idx, 0]
        ax.plot(epochs, train_losses, label="Train")
        ax.plot(epochs, val_losses, label="Val")
        if args.mark_best:
            if "best_epoch" in hist.files:
                best_idx = int(np.asarray(hist["best_epoch"]).item())
            else:
                best_idx = int(np.argmin(val_losses))
            best_idx = max(0, min(best_idx, len(epochs) - 1))
            ax.axvline(epochs[best_idx], linestyle="--", linewidth=1)
            ax.scatter([epochs[best_idx]], [val_losses[best_idx]], s=25, zorder=5)
        ax.set_title(col_titles[0], fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        set_common_style(ax)

        # 2. Log ratio
        ax = axes[row_idx, 1]
        ax.plot(epochs, log_ratio)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(col_titles[1], fontsize=9)
        set_common_style(ax)

        # 3. Gradient norm
        ax = axes[row_idx, 2]
        ax.plot(epochs, grad_norm)
        ax.axhline(args.max_norm, linestyle="--", linewidth=1)
        ax.set_title(col_titles[2], fontsize=9)
        set_common_style(ax)

        # 4. Learning rate
        ax = axes[row_idx, 3]
        ax.plot(epochs, lr_history)
        if args.lr_log and np.all(lr_history > 0):
            ax.set_yscale("log")
        ax.set_title(col_titles[3], fontsize=9)
        set_common_style(ax)

        # Only show x labels on last row to reduce clutter
        for col_idx in range(n_cols):
            if row_idx == n_rows - 1:
                axes[row_idx, col_idx].set_xlabel("Epoch", fontsize=8)
            else:
                axes[row_idx, col_idx].set_xlabel("")

    if args.title:
        fig.suptitle(args.title, fontsize=14, y=0.995)

    # Leave space on the left for dataset labels
    fig.tight_layout(rect=[0.08, 0.03, 0.99, 0.97])

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
