#!/usr/bin/env python3
"""
Paper-style multi-dataset training-statistics grid.

V3: same paper-style layout as V2, but follows the original color scheme:
train=blue, validation=orange, log-ratio=blue, gradient norm=teal, learning rate=orange,
best epoch=gray dashed, gradient clipping reference=red dashed.

Expected npz keys from train_0615.py:
  train_losses, val_losses, val_log_ratio_history,
  grad_norm_backbone_history, lr_history,
  optional: best_epoch, best_val_loss

Example:
  python plot_three_dataset_training_grid_paper_v3_original_colors.py \
    --npz "Liver=outputs/liver_B_new/liver_B_new_history.npz" \
    --npz "NPC=outputs/NPC_B_new/NPC_B_new_history.npz" \
    --npz "T Cell=outputs/TCell_B_new/TCell_B_new_history.npz" \
    --out B_three_dataset_training_stats_paper_v3.pdf \
    --mark_best \
    --lr_log
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext


def read_array(hist, key, default=None):
    if key in hist.files:
        arr = np.asarray(hist[key])
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr
    if default is not None:
        return np.asarray(default)
    raise KeyError(f"Missing key '{key}' in npz. Available keys: {hist.files}")


def parse_labeled_npz(item):
    if "=" not in item:
        path = Path(item)
        return path.stem.replace("_history", ""), path
    label, path = item.split("=", 1)
    return label.strip(), Path(path.strip())


def align_epochs(*arrays):
    lengths = [len(a) for a in arrays if a is not None and len(a) > 0]
    if not lengths:
        raise ValueError("No non-empty arrays found.")
    n = min(lengths)
    return np.arange(1, n + 1), [a[:n] if a is not None and len(a) > 0 else None for a in arrays]


def get_best_index(hist, val_losses, epochs):
    # best_epoch may be saved as 0-based index or 1-based epoch in different scripts.
    # Try to handle both safely.
    if "best_epoch" in hist.files:
        raw = int(np.asarray(hist["best_epoch"]).item())
        if raw in epochs:
            idx = int(np.where(epochs == raw)[0][0])
        else:
            idx = raw
    else:
        idx = int(np.argmin(val_losses))
    return max(0, min(idx, len(epochs) - 1))


def style_axis(ax):
    ax.tick_params(axis="both", labelsize=8, length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.grid(False)


def main():
    parser = argparse.ArgumentParser(description="Create a paper-style training-statistics grid from history npz files.")
    parser.add_argument("--npz", action="append", required=True,
                        help="Input as Label=path_to_history.npz. Use multiple times.")
    parser.add_argument("--out", default="training_stats_grid_paper.pdf")
    parser.add_argument("--max_norm", type=float, default=1.0,
                        help="Reference line for gradient clipping max norm.")
    parser.add_argument("--mark_best", action="store_true",
                        help="Draw best-validation-epoch vertical lines in all panels.")
    parser.add_argument("--annotate_best", action="store_true",
                        help="Write best epoch and best val loss in the loss panel.")
    parser.add_argument("--lr_log", action="store_true", help="Use log scale for learning rate axis.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Formal paper-like defaults. PDF remains vector; dpi matters mainly for raster formats.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })

    labeled_paths = [parse_labeled_npz(x) for x in args.npz]
    n_rows = len(labeled_paths)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13.2, 2.25 * n_rows + 0.8),
        squeeze=False,
        sharex=False,
    )

    column_titles = [
        "Triplet loss",
        r"Log-ratio, $\log(d(a,n)/d(a,p))$",
        "Gradient norm",
        "Learning rate",
    ]

    # Original color scheme
    color_train = "#1f77b4"   # blue
    color_val = "#ff7f0e"     # orange
    color_log = "#0000ff"     # blue, as in the original log-ratio panel
    color_grad = "#008080"    # teal, as in the original gradient-norm panel
    color_lr = "#ff7f0e"      # orange, as in the original learning-rate panel
    color_ref = "#ff0000"     # red dashed reference line for grad clipping
    color_best = "0.45"       # gray dashed line for best validation epoch

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

        best_idx = get_best_index(hist, val_losses, epochs)
        best_epoch = int(epochs[best_idx])
        best_val = float(val_losses[best_idx])

        # Row label on the left. It replaces repeated subplot titles.
        axes[row_idx, 0].annotate(
            label,
            xy=(-0.38, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=0,
        )

        # Column headers only once, on the first row.
        if row_idx == 0:
            for col_idx, title in enumerate(column_titles):
                axes[row_idx, col_idx].set_title(title, fontsize=10, fontweight="bold", pad=8)

        # 1. Loss
        ax = axes[row_idx, 0]
        ax.plot(epochs, train_losses, lw=1.35, color=color_train, label="Train")
        ax.plot(epochs, val_losses, lw=1.35, color=color_val, label="Validation")
        if args.mark_best:
            ax.axvline(best_epoch, color=color_best, ls="--", lw=0.9, alpha=0.8)
            ax.scatter([best_epoch], [best_val], s=18, color=color_best, zorder=5)
            if args.annotate_best:
                ax.text(
                    0.98, 0.93,
                    f"Best epoch: {best_epoch}\nVal loss: {best_val:.4g}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.8),
                )
        style_axis(ax)

        # 2. Log-ratio
        ax = axes[row_idx, 1]
        ax.plot(epochs, log_ratio, lw=1.35, color=color_log)
        if args.mark_best:
            ax.axvline(best_epoch, color=color_best, ls="--", lw=0.9, alpha=0.8)
        style_axis(ax)

        # 3. Gradient norm
        ax = axes[row_idx, 2]
        ax.plot(epochs, grad_norm, lw=1.35, color=color_grad)
        ax.axhline(args.max_norm, color=color_ref, ls="--", lw=0.8, alpha=0.75)
        style_axis(ax)

        # 4. Learning rate
        ax = axes[row_idx, 3]
        ax.plot(epochs, lr_history, lw=1.35, color=color_lr)
        if args.lr_log and np.all(lr_history > 0):
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        if args.mark_best:
            ax.axvline(best_epoch, color=color_best, ls="--", lw=0.9, alpha=0.8)
        style_axis(ax)

        # X-axis labels only on the bottom row.
        for col_idx in range(n_cols):
            if row_idx == n_rows - 1:
                axes[row_idx, col_idx].set_xlabel("Epoch", fontsize=9)
            else:
                axes[row_idx, col_idx].set_xticklabels([])

    # A single legend instead of repeated legends.
    legend_handles = [
        Line2D([0], [0], color=color_train, lw=1.5, label="Train"),
        Line2D([0], [0], color=color_val, lw=1.5, label="Validation"),
    ]
    if args.mark_best:
        legend_handles.append(Line2D([0], [0], color=color_best, ls="--", lw=1.0, label="Best validation epoch"))
    legend_handles.append(Line2D([0], [0], color=color_ref, ls="--", lw=1.0, label=f"Gradient clipping max norm = {args.max_norm:g}"))

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8,
        handlelength=2.3,
        columnspacing=1.3,
    )

    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.10, wspace=0.30, hspace=0.36)

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
