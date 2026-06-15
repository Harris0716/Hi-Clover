#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def load_threshold(value):
    value = np.asarray(value)
    return float(value.reshape(-1)[0])


def main():
    parser = argparse.ArgumentParser(
        description="Create a publication-style 2x3 distance distribution figure."
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing *_raw_dist.npz files")
    parser.add_argument("--out", default="combined_distance_distribution_paper_v3.pdf", help="Output file path")
    parser.add_argument("--bins", type=int, default=170, help="Number of histogram bins per panel")
    parser.add_argument("--dpi", type=int, default=300, help="Output dpi")
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

    color_rep = "#9FC3CC"
    color_cond = "#7D809E"
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

            data = np.load(path)
            dist = data["dist"]
            lbl = data["lbl"]
            threshold = load_threshold(data["threshold"])

            bins = np.linspace(xmin, xmax, args.bins)
            ax.hist(
                dist[lbl == 0],
                bins=bins,
                density=True,
                color=color_rep,
                alpha=0.68,
                edgecolor="none",
                label="Replicates",
            )
            ax.hist(
                dist[lbl == 1],
                bins=bins,
                density=True,
                color=color_cond,
                alpha=0.68,
                edgecolor="none",
                label="Conditions",
            )
            ax.axvline(
                threshold,
                color=color_thr,
                linestyle="--",
                linewidth=1.0,
                label="Decision threshold",
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
        Patch(facecolor=color_rep, edgecolor="none", alpha=0.68, label="Replicates"),
        Patch(facecolor=color_cond, edgecolor="none", alpha=0.68, label="Conditions"),
        Line2D([0], [0], color=color_thr, linestyle="--", linewidth=1.0, label="Decision threshold"),
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
