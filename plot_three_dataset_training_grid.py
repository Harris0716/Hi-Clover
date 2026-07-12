#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def load_data(npz_path):
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

def apply_tufte_style(ax, hide_xticks=False):
    """應用學術視覺化標準的乾淨樣式"""
    # 移除上方與右側邊框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 柔化剩餘邊框顏色，凸顯數據本身
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # 調整刻度樣式
    ax.tick_params(axis='both', labelsize=10.5, width=0.8, length=4, colors='#333333')
    
    if hide_xticks:
        ax.set_xticklabels([])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", action="append", required=True)
    parser.add_argument("--out", default="training_stats.pdf")
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--mark_best", action="store_true")
    parser.add_argument("--lr_log", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # 全局字體與色彩設定
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
    
    # 14x7.5 的畫布設定，確保子圖比例落在最佳的 3:2
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(14, 7.5), 
        layout="constrained",
        sharex=False
    )
    
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # 已移除 Log-ratio 的公式
    col_titles = [
        "Triplet loss", 
        "Log-ratio", 
        "Gradient norm", 
        "Learning rate"
    ]
    
    # 使用具高辨識度的色碼
    colors = {
        "train": "#1f77b4", "val": "#ff7f0e", "ratio": "#0000ff", 
        "grad": "#008080", "lr": "#6A3D9A", "best": "#999999", "ref": "#ff9896"
    }

    for r, (name, path) in enumerate(datasets):
        epochs, data, best_ep, best_val = load_data(path)
        is_bottom = (r == rows - 1)

        # 1. Triplet Loss
        ax = axes[r, 0]
        ax.plot(epochs, data["train_loss"], c=colors["train"], lw=1.6)
        ax.plot(epochs, data["val_loss"], c=colors["val"], lw=1.6)
        if args.mark_best:
            ax.axvline(best_ep, c=colors["best"], ls="--", lw=1.2)
            ax.scatter(best_ep, best_val, s=25, c=colors["best"], zorder=3)
        
        ax.set_ylabel(name, rotation=0, ha="right", va="center", 
                      fontsize=14, fontweight="bold", labelpad=25)
        apply_tufte_style(ax, hide_xticks=not is_bottom)

        # 2. Log-ratio (嚴格鎖定原圖刻度)
        ax = axes[r, 1]
        ax.plot(epochs, data["log_ratio"], c=colors["ratio"], lw=1.6)
        if args.mark_best:
            ax.axvline(best_ep, c=colors["best"], ls="--", lw=1.2)
        ax.set_ylim(0, 0.9)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
        apply_tufte_style(ax, hide_xticks=not is_bottom)

        # 3. Gradient Norm (嚴格鎖定原圖刻度)
        ax = axes[r, 2]
        ax.plot(epochs, data["grad_norm"], c=colors["grad"], lw=1.6)
        ax.axhline(args.max_norm, c=colors["ref"], ls="--", lw=1.2)
        ax.set_ylim(0, 1.6)
        ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
        apply_tufte_style(ax, hide_xticks=not is_bottom)

        # 4. Learning Rate
        ax = axes[r, 3]
        ax.plot(epochs, data["lr"], c=colors["lr"], lw=1.6)
        if args.mark_best:
            ax.axvline(best_ep, c=colors["best"], ls="--", lw=1.2)
        if args.lr_log and np.all(data["lr"] > 0):
            ax.set_yscale("log")
        apply_tufte_style(ax, hide_xticks=not is_bottom)

        # 標題與 X 軸標籤設定
        for c in range(cols):
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=13, fontweight="bold", pad=12)
            if is_bottom:
                axes[r, c].set_xlabel("Epoch", fontsize=12, labelpad=8)

    # 頂部圖例排版
    handles = [
        Line2D([0], [0], color=colors["train"], lw=1.6, label="Train"),
        Line2D([0], [0], color=colors["val"], lw=1.6, label="Validation"),
    ]
    if args.mark_best:
        handles.append(Line2D([0], [0], color=colors["best"], ls="--", lw=1.2, label="Best validation epoch"))
    handles.append(Line2D([0], [0], color=colors["ref"], ls="--", lw=1.2, label=f"Gradient clipping max norm = {args.max_norm:g}"))

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=len(handles), frameon=False, fontsize=11.5, columnspacing=2.0)

    # 輸出設定
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()

if __name__ == "__main__":
    main()