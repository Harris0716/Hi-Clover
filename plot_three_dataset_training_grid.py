#!/usr/bin/env python3
"""
Publication-ready training-statistics figure for a thesis.

Layout:
    rows    = datasets
    columns = Triplet loss, Log-ratio, Gradient norm, Learning rate

Expected NPZ keys:
    train_losses
    val_losses
    val_log_ratio_history
    grad_norm_backbone_history
    lr_history

Optional NPZ keys:
    best_epoch
    best_val_loss

Example:
    python plot_training_stats_thesis_clean.py \
        --npz "Liver=outputs/Liver_history.npz" \
        --npz "NPC=outputs/NPC_history.npz" \
        --npz "T Cell=outputs/TCell_history.npz" \
        --out training_stats_thesis_clean.pdf \
        --mark_best \
        --lr_log
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    MaxNLocator,
    NullFormatter,
    NullLocator,
)


# ---------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------
FIGURE_SIZE = (15.2, 8.6)

FONT_TITLE = 17
FONT_ROW = 16
FONT_TICK = 12.5
FONT_XLABEL = 13.5
FONT_LEGEND = 12
FONT_ANNOTATION = 9.5

LINE_WIDTH = 1.8
SPINE_WIDTH = 0.85
REFERENCE_WIDTH = 1.0

COLOR_TRAIN = "#1F77B4"
COLOR_VALIDATION = "#FF7F0E"
COLOR_LOG_RATIO = "#0000FF"
COLOR_GRADIENT = "#008080"
COLOR_LEARNING_RATE = "#6A3D9A"
COLOR_BEST = "#777777"
COLOR_CLIP = "#FF0000"


# ---------------------------------------------------------------------
# Input utilities
# ---------------------------------------------------------------------
def normalize_dataset_name(name):
    clean = name.strip()
    normalized = clean.lower().replace("_", " ").replace("-", " ")

    if normalized == "liver":
        return "Liver"
    if normalized == "npc":
        return "NPC"
    if normalized in {"tcell", "t cell"}:
        return "T Cell"

    return clean


def parse_npz_argument(item):
    if "=" in item:
        label, path = item.split("=", 1)
        return normalize_dataset_name(label), Path(path.strip())

    path = Path(item)
    label = path.stem.replace("_history", "")
    return normalize_dataset_name(label), path


def read_array(history, key):
    if key not in history.files:
        raise KeyError(
            f"Missing key '{key}' in {history.filename}. "
            f"Available keys: {history.files}"
        )

    array = np.asarray(history[key], dtype=float)

    if array.ndim == 0:
        array = array.reshape(1)

    return array


def align_arrays(*arrays):
    length = min(len(array) for array in arrays)

    if length <= 0:
        raise ValueError("History arrays must not be empty.")

    epochs = np.arange(1, length + 1)
    return epochs, [array[:length] for array in arrays]


def resolve_best_index(history, val_losses):
    fallback = int(np.argmin(val_losses))

    if "best_epoch" not in history.files:
        return fallback

    raw = int(np.asarray(history["best_epoch"]).item())
    candidates = []

    # Current training code: zero-based index.
    if 0 <= raw < len(val_losses):
        candidates.append(raw)

    # Compatibility with one-based epoch values.
    if 1 <= raw <= len(val_losses):
        candidates.append(raw - 1)

    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        return fallback

    if "best_val_loss" in history.files:
        target = float(np.asarray(history["best_val_loss"]).item())
        return min(
            candidates,
            key=lambda idx: abs(float(val_losses[idx]) - target),
        )

    return candidates[0]


# ---------------------------------------------------------------------
# Tick formatting
# ---------------------------------------------------------------------
def compact_scientific(value, _position):
    if value == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / (10 ** exponent)

    if np.isclose(coefficient, round(coefficient), atol=1e-8):
        coefficient_text = str(int(round(coefficient)))
    else:
        coefficient_text = f"{coefficient:.1f}".rstrip("0").rstrip(".")

    return f"{coefficient_text}e{exponent}"


def three_log_ticks(values):
    positive = values[np.isfinite(values) & (values > 0)]

    if len(positive) == 0:
        return np.array([])

    minimum = float(np.min(positive))
    maximum = float(np.max(positive))

    if np.isclose(minimum, maximum):
        return np.array([minimum])

    return np.geomspace(minimum, maximum, 3)


def style_axis(ax, show_x_labels):
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=FONT_TICK,
        length=4,
        width=0.85,
        pad=4,
    )

    if not show_x_labels:
        ax.tick_params(axis="x", labelbottom=False)

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=4, integer=True)
    )

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#555555")

    ax.grid(False)
    ax.margins(x=0.025)


# ---------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create a clean thesis training-statistics figure."
    )

    parser.add_argument(
        "--npz",
        action="append",
        required=True,
        help="Use Label=path_to_history.npz once per dataset.",
    )
    parser.add_argument(
        "--out",
        default="training_stats_thesis_clean.pdf",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--mark_best",
        action="store_true",
    )
    parser.add_argument(
        "--annotate_best",
        action="store_true",
    )
    parser.add_argument(
        "--lr_log",
        action="store_true",
    )
    parser.add_argument(
        "--log_ratio_ylim",
        nargs=2,
        type=float,
        default=(0.0, 0.9),
        metavar=("YMIN", "YMAX"),
    )
    parser.add_argument(
        "--grad_norm_ylim",
        nargs=2,
        type=float,
        default=(0.0, 1.6),
        metavar=("YMIN", "YMAX"),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    datasets = [parse_npz_argument(item) for item in args.npz]
    row_count = len(datasets)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FONT_TICK,
        "axes.linewidth": SPINE_WIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=FIGURE_SIZE)

    # Explicit spacer columns prevent y-axis labels from entering
    # neighbouring panels.
    grid = GridSpec(
        row_count,
        7,
        figure=fig,
        width_ratios=[
            1.0, 0.28,
            1.0, 0.28,
            1.0, 0.38,
            1.0,
        ],
        left=0.085,
        right=0.975,
        top=0.855,
        bottom=0.105,
        wspace=0.0,
        hspace=0.38,
    )

    panel_columns = [0, 2, 4, 6]
    axes = np.empty((row_count, 4), dtype=object)

    for row in range(row_count):
        for column, grid_column in enumerate(panel_columns):
            axes[row, column] = fig.add_subplot(
                grid[row, grid_column]
            )

    titles = [
        "Triplet loss",
        "Log-ratio",
        "Gradient norm",
        "Learning rate",
    ]

    for column, title in enumerate(titles):
        axes[0, column].set_title(
            title,
            fontsize=FONT_TITLE,
            fontweight="bold",
            pad=10,
        )

    row_names = []

    for row, (dataset_name, npz_path) in enumerate(datasets):
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Cannot find NPZ file: {npz_path}"
            )

        history = np.load(npz_path, allow_pickle=True)

        train_losses = read_array(history, "train_losses")
        val_losses = read_array(history, "val_losses")
        log_ratio = read_array(
            history,
            "val_log_ratio_history",
        )
        gradient_norm = read_array(
            history,
            "grad_norm_backbone_history",
        )
        learning_rate = read_array(
            history,
            "lr_history",
        )

        epochs, aligned = align_arrays(
            train_losses,
            val_losses,
            log_ratio,
            gradient_norm,
            learning_rate,
        )

        (
            train_losses,
            val_losses,
            log_ratio,
            gradient_norm,
            learning_rate,
        ) = aligned

        best_index = resolve_best_index(history, val_losses)
        best_epoch = int(epochs[best_index])
        best_val_loss = float(val_losses[best_index])

        show_x_labels = row == row_count - 1

        # ---------------------------------------------------------
        # Triplet loss
        # ---------------------------------------------------------
        ax = axes[row, 0]

        ax.plot(
            epochs,
            train_losses,
            color=COLOR_TRAIN,
            linewidth=LINE_WIDTH,
        )
        ax.plot(
            epochs,
            val_losses,
            color=COLOR_VALIDATION,
            linewidth=LINE_WIDTH,
        )

        if args.mark_best:
            ax.axvline(
                best_epoch,
                color=COLOR_BEST,
                linestyle="--",
                linewidth=REFERENCE_WIDTH,
            )
            ax.scatter(
                [best_epoch],
                [best_val_loss],
                s=28,
                color=COLOR_BEST,
                zorder=5,
            )

            if args.annotate_best:
                ax.text(
                    0.97,
                    0.93,
                    (
                        f"Epoch {best_epoch}\n"
                        f"Val {best_val_loss:.4g}"
                    ),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=FONT_ANNOTATION,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 2,
                    },
                )

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=4)
        )
        style_axis(ax, show_x_labels)

        # ---------------------------------------------------------
        # Log-ratio
        # ---------------------------------------------------------
        ax = axes[row, 1]

        ax.plot(
            epochs,
            log_ratio,
            color=COLOR_LOG_RATIO,
            linewidth=LINE_WIDTH,
        )

        if args.mark_best:
            ax.axvline(
                best_epoch,
                color=COLOR_BEST,
                linestyle="--",
                linewidth=REFERENCE_WIDTH,
            )

        ax.set_ylim(*args.log_ratio_ylim)
        ax.yaxis.set_major_locator(
            FixedLocator([0.0, 0.25, 0.50, 0.75])
        )
        style_axis(ax, show_x_labels)

        # ---------------------------------------------------------
        # Gradient norm
        # ---------------------------------------------------------
        ax = axes[row, 2]

        ax.plot(
            epochs,
            gradient_norm,
            color=COLOR_GRADIENT,
            linewidth=LINE_WIDTH,
        )
        ax.axhline(
            args.max_norm,
            color=COLOR_CLIP,
            linestyle="--",
            linewidth=REFERENCE_WIDTH,
            alpha=0.65,
        )

        ax.set_ylim(*args.grad_norm_ylim)
        ax.yaxis.set_major_locator(
            FixedLocator([0.0, 0.4, 0.8, 1.2, 1.6])
        )
        style_axis(ax, show_x_labels)

        # ---------------------------------------------------------
        # Learning rate
        # ---------------------------------------------------------
        ax = axes[row, 3]

        ax.plot(
            epochs,
            learning_rate,
            color=COLOR_LEARNING_RATE,
            linewidth=LINE_WIDTH,
        )

        positive_lr = learning_rate[
            np.isfinite(learning_rate)
            & (learning_rate > 0)
        ]

        if (
            args.lr_log
            and len(positive_lr) > 0
            and np.all(learning_rate > 0)
        ):
            ax.set_yscale("log")
            ticks = three_log_ticks(learning_rate)

            if len(ticks) > 0:
                ax.yaxis.set_major_locator(
                    FixedLocator(ticks)
                )
        else:
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=3)
            )

        ax.yaxis.set_major_formatter(
            FuncFormatter(compact_scientific)
        )

        # Disable all minor labels and ticks: only the three selected
        # learning-rate labels remain.
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.get_offset_text().set_visible(False)

        if args.mark_best:
            ax.axvline(
                best_epoch,
                color=COLOR_BEST,
                linestyle="--",
                linewidth=REFERENCE_WIDTH,
            )

        style_axis(ax, show_x_labels)

        if show_x_labels:
            for column in range(4):
                axes[row, column].set_xlabel(
                    "Epoch",
                    fontsize=FONT_XLABEL,
                    labelpad=6,
                )

        row_names.append(dataset_name)
        history.close()

    # Figure-level row labels cannot be clipped by subplot boundaries.
    fig.canvas.draw()

    for row, dataset_name in enumerate(row_names):
        position = axes[row, 0].get_position()
        center_y = (position.y0 + position.y1) / 2

        fig.text(
            0.030,
            center_y,
            dataset_name,
            ha="center",
            va="center",
            fontsize=FONT_ROW,
            fontweight="bold",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_TRAIN,
            linewidth=2.0,
            label="Train",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_VALIDATION,
            linewidth=2.0,
            label="Validation",
        ),
    ]

    if args.mark_best:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=COLOR_BEST,
                linestyle="--",
                linewidth=1.2,
                label="Best validation epoch",
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=COLOR_CLIP,
            linestyle="--",
            linewidth=1.2,
            label=(
                "Gradient clipping max norm "
                f"= {args.max_norm:g}"
            ),
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.975),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=2.4,
        handletextpad=0.55,
        columnspacing=1.25,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep the manually reserved margins.
    fig.savefig(
        output_path,
        dpi=args.dpi,
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
