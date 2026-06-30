#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def load_raw_dist(path):
    data = np.load(path)
    return data["dist"], data["lbl"]


def compute_best_mean_threshold(dist, lbl, steps=5000):
    """
    Select threshold using Train+Val only.

    Label convention:
      lbl == 0: replicate pair
      lbl == 1: condition pair

    Prediction rule:
      distance < threshold  -> replicate
      distance >= threshold -> condition

    The selected threshold maximizes:
      mean_perf = (replicate_accuracy + condition_accuracy) / 2
    """
    dist = np.asarray(dist).reshape(-1)
    lbl = np.asarray(lbl).reshape(-1)

    rep_dist = dist[lbl == 0]
    cond_dist = dist[lbl == 1]

    if len(rep_dist) == 0 or len(cond_dist) == 0:
        raise ValueError("Both replicate and condition distances are required to compute threshold.")

    d_min = float(np.min(dist))
    d_max = float(np.max(dist))

    # Include a small margin so thresholds at both extremes can be considered.
    eps = max((d_max - d_min) * 1e-6, 1e-8)
    candidates = np.linspace(d_min - eps, d_max + eps, steps)

    rep_acc = (rep_dist[:, None] < candidates[None, :]).mean(axis=0)
    cond_acc = (cond_dist[:, None] >= candidates[None, :]).mean(axis=0)
    mean_perf = (rep_acc + cond_acc) / 2.0

    best_idx = int(np.argmax(mean_perf))
    return float(candidates[best_idx]), float(rep_acc[best_idx]), float(cond_acc[best_idx]), float(mean_perf[best_idx])


def main():
    parser = argparse.ArgumentParser(
        description="Create a publication-style 2x3 distance distribution figure using Train+Val best-mean threshold calibration."
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing *_raw_dist.npz files")
    parser.add_argument("--out", default="combined_distance_distribution_paper_v4_best_mean.pdf", help="Output file path")
    parser.add_argument("--bins", type=int, default=170, help="Number of histogram bins per panel")
    parser.add_argument("--dpi", type=int, default=300, help="Output dpi")
    parser.add_argument("--threshold_steps", type=int, default=5000, help="Number of candidate thresholds scanned on Train+Val")
    args = parser.parse_args()

    file_prefixes = ["liver", "NPC", "TCell"]
    display_names = ["Liver", "NPC", "T Cell"]
    phases = ["train_val", "test"]
    phase_labels = ["Train + Val", "Test"]

    # Per-dataset x limits. T Cell is intentionally shorter to avoid empty space.
    xlims = {
        "liver": (0.0, 1.60),
        "NPC": (0.0, 1.80),
        "TCell": (0.0, 1.25),
    }

    color_rep = "#108690"
    color_cond = "#1D1E4E"
    color_thr = "#111111"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 8.8,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Recompute one threshold per dataset from Train+Val only.
    # This ignores the old threshold stored in *_raw_dist.npz.
    thresholds = {}
    print("Threshold calibration method: best mean performance on Train+Val only")
    for prefix in file_prefixes:
        train_val_path = os.path.join(args.data_dir, f"{prefix}_train_val_raw_dist.npz")
        if not os.path.exists(train_val_path):
            print(f"[Warning] Missing Train+Val file for threshold calibration: {train_val_path}")
            thresholds[prefix] = None
            continue

        train_val_dist, train_val_lbl = load_raw_dist(train_val_path)
        threshold, rep_acc, cond_acc, mean_perf = compute_best_mean_threshold(
            train_val_dist,
            train_val_lbl,
            steps=args.threshold_steps,
        )
        thresholds[prefix] = threshold
        print(
            f"  {prefix}: threshold={threshold:.4f}, "
            f"Train+Val Rep={rep_acc:.4f}, Cond={cond_acc:.4f}, Mean={mean_perf:.4f}"
        )

    fig, axes = plt.subplots(
        2, 3,
        figsize=(9.2, 5.25),
        sharex=False,
        sharey="col"
    )

    for i, phase in enumerate(phases):
        for j, prefix in enumerate(file_prefixes):
            ax = axes[i, j]
            path = os.path.join(args.data_dir, f"{prefix}_{phase}_raw_dist.npz")
            xmin, xmax = xlims[prefix]

            if not os.path.exists(path):
                ax.text(
                    0.5, 0.5,
                    f"Missing\n{prefix}_{phase}_raw_dist.npz",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_xlim(xmin, xmax)
                continue

            dist, lbl = load_raw_dist(path)
            threshold = thresholds[prefix]

            bins = np.linspace(xmin, xmax, args.bins)
            ax.hist(
                dist[lbl == 0],
                bins=bins,
                density=True,
                color=color_rep,
                alpha=0.5,
                edgecolor="none",
                label="Replicates",
            )
            ax.hist(
                dist[lbl == 1],
                bins=bins,
                density=True,
                color=color_cond,
                alpha=0.5,
                edgecolor="none",
                label="Conditions",
            )

            if threshold is not None:
                ax.axvline(
                    threshold,
                    color=color_thr,
                    linestyle="--",
                    linewidth=1.0,
                    label="Decision Threshold",
                )

            ax.set_xlim(xmin, xmax)
            ax.grid(False)
            ax.tick_params(direction="out")
            ax.xaxis.set_major_locator(MultipleLocator(0.25))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            if i == 0:
                ax.set_title(display_names[j], fontweight="bold", pad=7)
                ax.tick_params(labelbottom=False)

            # Keep panels clean.
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.supxlabel("Euclidean Distance", fontsize=10.5, y=0.055)
    fig.supylabel("Probability Density", fontsize=10.5, x=0.028)

    # Row labels placed between the global y-label and panels.
    fig.text(
        0.063, 0.635,
        phase_labels[0],
        va="center", ha="center",
        rotation="vertical",
        fontsize=9.8,
        fontweight="bold",
    )
    fig.text(
        0.063, 0.245,
        phase_labels[1],
        va="center", ha="center",
        rotation="vertical",
        fontsize=9.8,
        fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor=color_rep, edgecolor="none", alpha=0.5, label="Replicates"),
        Patch(facecolor=color_cond, edgecolor="none", alpha=0.5, label="Conditions"),
        Line2D([0], [0], color=color_thr, linestyle="--", linewidth=1.0, label="Decision Threshold"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.0,
    )

    plt.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.855,
        bottom=0.13,
        wspace=0.20,
        hspace=0.12,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
