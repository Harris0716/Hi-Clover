#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def load_threshold(value):
    value = np.asarray(value)
    return float(value.reshape(-1)[0])


def main():
    parser = argparse.ArgumentParser(
        description="Create a publication-style 2x3 distance distribution figure."
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing *_raw_dist.npz files")
    parser.add_argument("--out", default="combined_distance_distribution_paper_v2.pdf", help="Output file path")
    parser.add_argument("--bins", type=int, default=180, help="Number of histogram bins")
    parser.add_argument("--xmax", type=float, default=1.85, help="Shared x-axis upper limit")
    parser.add_argument("--dpi", type=int, default=300, help="Output dpi")
    args = parser.parse_args()

    file_prefixes = ["liver", "NPC", "TCell"]
    display_names = ["Liver", "NPC", "T Cell"]
    phases = ["train_val", "test"]
    phase_labels = ["Train + Val", "Test"]

    # Keep the original color identity, but use restrained opacity.
    color_rep = "#9FC3CC"
    color_cond = "#7D809E"
    color_thr = "#111111"

    # A cleaner, journal-style matplotlib setup.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        2, 3,
        figsize=(9.4, 5.5),
        sharex=True,
        sharey="col"
    )

    for i, phase in enumerate(phases):
        for j, prefix in enumerate(file_prefixes):
            ax = axes[i, j]
            path = os.path.join(args.data_dir, f"{prefix}_{phase}_raw_dist.npz")

            if not os.path.exists(path):
                ax.text(
                    0.5, 0.5,
                    f"Missing\n{prefix}_{phase}_raw_dist.npz",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_xlim(0, args.xmax)
                continue

            data = np.load(path)
            dist = data["dist"]
            lbl = data["lbl"]
            threshold = load_threshold(data["threshold"])

            bins = np.linspace(0, args.xmax, args.bins)
            ax.hist(
                dist[lbl == 0],
                bins=bins,
                density=True,
                color=color_rep,
                alpha=0.65,
                edgecolor="none",
                label="Replicates",
            )
            ax.hist(
                dist[lbl == 1],
                bins=bins,
                density=True,
                color=color_cond,
                alpha=0.65,
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

            ax.set_xlim(0, args.xmax)
            ax.grid(False)
            ax.tick_params(direction="out")

            # Column titles only once, with controlled spacing.
            if i == 0:
                ax.set_title(display_names[j], fontweight="bold", pad=7)

            # Remove repeated x tick labels from the top row.
            if i == 0:
                ax.tick_params(labelbottom=False)

    # Common labels: modest size, outside the panel area.
    fig.supxlabel("Euclidean distance", fontsize=11, y=0.055)
    fig.supylabel("Probability density", fontsize=11, x=0.028)

    # Row labels: placed separately from the y-axis label, not oversized.
    fig.text(
        0.058, 0.635,
        phase_labels[0],
        va="center", ha="center",
        rotation="vertical",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.058, 0.245,
        phase_labels[1],
        va="center", ha="center",
        rotation="vertical",
        fontsize=10,
        fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor=color_rep, edgecolor="none", alpha=0.65, label="Replicates"),
        Patch(facecolor=color_cond, edgecolor="none", alpha=0.65, label="Conditions"),
        Line2D([0], [0], color=color_thr, linestyle="--", linewidth=1.0, label="Decision threshold"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.0,
    )

    # More conservative layout: legend has its own space, labels don't collide.
    plt.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.86,
        bottom=0.13,
        wspace=0.22,
        hspace=0.10,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
