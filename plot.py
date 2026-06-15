import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def _scalar(x):
    x = np.asarray(x)
    return float(x.ravel()[0])


def main():
    parser = argparse.ArgumentParser(description="Plot combined distance distributions for Train+Val and Test.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing *_raw_dist.npz files.")
    parser.add_argument("--out", type=str, default="combined_distance_distribution_paper.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--xmax", type=float, default=1.8)
    parser.add_argument("--bins", type=int, default=180)
    args = parser.parse_args()

    # File names and display labels
    file_prefixes = ["liver", "NPC", "TCell"]
    display_names = ["Liver", "NPC", "T Cell"]
    phases = ["train_val", "test"]
    phase_labels = ["Train + Val", "Test"]

    # Colors following the original style
    color_rep = "#9FC3CC"
    color_cond = "#7D809E"
    color_thr = "black"

    # Paper-style typography. Times New Roman will be used if available.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 13,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14.5, 7.6),
        sharex=True,
        sharey="col"
    )

    x_min, x_max = 0.0, args.xmax
    bins = np.linspace(x_min, x_max, args.bins)

    for i, phase in enumerate(phases):
        for j, prefix in enumerate(file_prefixes):
            ax = axes[i, j]
            file_path = os.path.join(args.data_dir, f"{prefix}_{phase}_raw_dist.npz")

            if not os.path.exists(file_path):
                ax.text(
                    0.5, 0.5,
                    f"Missing:\n{prefix}_{phase}_raw_dist.npz",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )
                continue

            data = np.load(file_path)
            dist = data["dist"]
            lbl = data["lbl"]
            threshold = _scalar(data["threshold"])

            rep_dist = dist[lbl == 0]
            cond_dist = dist[lbl == 1]

            ax.hist(
                rep_dist,
                bins=bins,
                density=True,
                alpha=0.70,
                color=color_rep,
                label="Replicates",
            )
            ax.hist(
                cond_dist,
                bins=bins,
                density=True,
                alpha=0.70,
                color=color_cond,
                label="Conditions",
            )
            ax.axvline(
                threshold,
                color=color_thr,
                linestyle="--",
                linewidth=1.3,
                label="Decision threshold",
            )

            ax.set_xlim(x_min, x_max)
            ax.tick_params(direction="out", length=3.5, width=0.8)

            if i == 0:
                ax.set_title(display_names[j], pad=10)

            # Hide top/right spines for a cleaner paper figure
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Global axis labels
    fig.supxlabel("Euclidean distance", fontsize=17, y=0.055)
    fig.supylabel("Probability density", fontsize=17, x=0.028)

    # Row labels. Smaller and farther from the y-axis to avoid crowding.
    fig.text(
        0.073, 0.645,
        phase_labels[0],
        va="center", ha="center",
        rotation="vertical",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.073, 0.305,
        phase_labels[1],
        va="center", ha="center",
        rotation="vertical",
        fontsize=17,
        fontweight="bold",
    )

    # Single legend above the column titles, with enough gap to avoid overlap.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Normalize legend capitalization
    labels = ["Replicates", "Conditions", "Decision threshold"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=2.2,
    )

    # Spacing: leave separate room for legend, titles, row labels, and global labels.
    fig.subplots_adjust(
        left=0.115,
        right=0.985,
        top=0.845,
        bottom=0.135,
        wspace=0.16,
        hspace=0.18,
    )

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
