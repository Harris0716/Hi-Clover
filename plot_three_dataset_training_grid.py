#!/usr/bin/env python3
"""
Publication-ready 3x4 training statistics grid.
Optimized for standard academic paper column widths.
Subplot aspect ratios tuned to ~1.5:1 (width:height).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

def format_scientific(value, _pos):
    """Compact scientific notation for learning rate."""
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    coeff = value / (10 ** exp)
    if np.isclose(coeff, round(coeff), atol=1e-8):
        return f"{int(round(coeff))}e{exp}"
    return f"{coeff:.1f}e{exp}"

def load_data(npz_path):
    """Safely load arrays from NPZ."""
    history = np.load(npz_path, allow_pickle=True)
    
    data = {
        "train_loss": np.atleast_1d(history["train_losses"]).astype(float),
        "val_loss": np.atleast_1d(history["val_losses"]).astype(float),
        "log_ratio": np.atleast_1d(history["val_log_ratio_history"]).astype(float),
        "grad_norm": np.atleast_1d(history["grad_norm_backbone_history"]).astype(float),
        "lr": np.atleast_1d(history["lr_history"]).astype(float),
    }
    
    min_len = min(len(v) for v in data.values())
    epochs = np.arange(1, min_len + 1)
    for k in data:
        data[k] = data[k][:min_len]
        
    best_idx = int(np.argmin(data["val_loss"]))
    if "best_epoch" in history:
        raw_best = int(np.atleast_1d(history["best_epoch"]).item())
        if 0 <= raw_best < min_len:
            best_idx = raw_best
            
    best_epoch = epochs[best_idx]
    best_val_loss = data["val_loss"][best_idx]
    
    return epochs, data, best_epoch, best_val_loss

def style_ax(ax, hide_xticks=False):
    """Apply clean, publication-style formatting to an axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    ax.tick_params(axis='both', labelsize=10.5, width=0.8, length=4, colors='#333333')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    
    if ax.get_yscale() != 'log':
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        
    if hide_xticks:
        ax.set_xticklabels([])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", action="append", required=True, help="Label=path.npz")
    parser.add_argument("--out", default="training_stats.pdf")
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--mark_best", action="store_true")
    parser.add_argument("--lr_log", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # 全局字體與線條設定
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#333333"
    })

    datasets = []
    for item in args.npz:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(item).stem.replace("_history", "")
        datasets.append((name.strip(), Path(path.strip())))

    rows = len(datasets)
    cols = 4
    
    # 決定黃金比例的畫布大小
    # 14 (寬) x 7.5 (高) 可確保 4x3 的排列下，每個子圖比例約為 3:2
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(14, 7.5), 
        layout="constrained",
        sharex=False
    )
    
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Triplet loss", "Log-ratio", "Gradient norm", "Learning rate"]
    colors = {
        "train": "#1f77b4", "val": "#ff7f0e", "ratio": "#0000ff", 
        "grad": "#008080", "lr": "#6A3D9A", "best": "#666666", "ref": "#d62728"
    }

    for r, (name, path) in enumerate(datasets):
        epochs, data, best_ep, best_val = load_data(path)
        is_bottom = (r == rows - 1)

        # 1. Triplet Loss
        ax = axes[r, 0]
        ax.plot(epochs, data["train_loss"], c=colors["train"], lw=1.6)
        ax.plot(epochs, data["val_loss"], c=colors["val"], lw=1.6)
        if args.mark_best:
            ax.axvline(best_ep, c=colors["best"], ls="--", lw=1.2, alpha=0.8)
            ax.scatter(best_ep, best_val, s=24, c=colors["best"], zorder=3)
        style_ax(ax, hide_xticks=not is_bottom)
        
        # 資料集名稱 (Y軸標籤)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", 
                      fontsize=14, fontweight="bold", labelpad=18)

        # 2. Log-ratio
        ax = axes[r, 1]
        ax.plot(epochs, data["log_ratio"], c=colors["ratio"], lw=1.6)
        if args.mark_best:
            ax.axvline(best_ep, c=colors["best"], ls="--", lw=1.2, alpha=0.8)
        ax.set_ylim(bottom=0)
        style_ax(ax, hide_xticks=not is_bottom)

        # 3. Gradient Norm
        ax = axes[r, 2]
        ax.plot(epochs, data["grad_norm"], c=colors["grad"], lw=1.6)
        ax.axhline(args.max_norm, c=colors["ref"], ls="--", lw=1.2, alpha=0.7)
        ax.set_ylim(bottom=0)
        style_ax(ax, hide_xticks=not is_bottom)

        # 4. Learning Rate
        ax = axes[r, 3]
        ax.plot(epochs, data["lr"], c=colors["lr"], lw=1.6)
        if args.mark_best:
            ax.axvline(best_ep, c=colors["best"], ls="--", lw=1.2, alpha=0.8)
        if args.lr_log and np.all(data["lr"] > 0):
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(format_scientific))
        style_ax(ax, hide_xticks=not is_bottom)

        # 頂部欄位標題與底部 X 軸標籤
        if r == 0:
            for c in range(cols):
                axes[r, c].set_title(col_titles[c], fontsize=13, fontweight="bold", pad=12)
        if is_bottom:
            for c in range(cols):
                axes[r, c].set_xlabel("Epoch", fontsize=12, labelpad=8)

    # 頂部圖例 (調整大小與間距，使其符合大比例畫布)
    handles = [
        Line2D([0], [0], color=colors["train"], lw=1.8, label="Train"),
        Line2D([0], [0], color=colors["val"], lw=1.8, label="Validation")
    ]
    if args.mark_best:
        handles.append(Line2D([0], [0], color=colors["best"], ls="--", lw=1.2, label="Best val. epoch"))
    handles.append(Line2D([0], [0], color=colors["ref"], ls="--", lw=1.2, label=f"Clip max norm = {args.max_norm:g}"))

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=len(handles), frameon=False, fontsize=11.5, columnspacing=2.0)

    # 儲存
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved figure to: {args.out}")

if __name__ == "__main__":
    main()