#!/usr/bin/env python3
"""
Create a publication-ready 3 × 4 training-statistics figure.

This script is written from scratch for thesis use. It reads one history NPZ
file per dataset and draws:

    1. Triplet loss
    2. Log-ratio
    3. Gradient norm
    4. Learning rate

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
    python plot_training_stats_thesis.py \
        --npz "Liver=outputs/Liver_history.npz" \
        --npz "NPC=outputs/NPC_history.npz" \
        --npz "T Cell=outputs/TCell_history.npz" \
        --out training_stats_thesis.pdf \
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
from matplotlib.ticker import FuncFormatter, MaxNLocator


# ---------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------
FIGURE_SIZE = (13.2, 8.5)

TITLE_FONT_SIZE = 17
ROW_LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 13
X_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 12.5
ANNOTATION_FONT_SIZE = 10

LINE_WIDTH = 1.8
REFERENCE_LINE_WIDTH = 1.0
SPINE_WIDTH = 0.85

COLOR_TRAIN = "#1F77B4"
COLOR_VALIDATION = "#FF7F0E"
COLOR_LOG_RATIO = "#0000FF"
COLOR_GRADIENT = "#008080"
COLOR_LEARNING_RATE = "#6A3D9A"
COLOR_BEST = "#777777"
COLOR_CLIP = "#FF0000"


# ---------------------------------------------------------------------
# Input helpers
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


def align_histories(*arrays):
    lengths = [len(array) for array in arrays]

    if not lengths or min(lengths) == 0:
        raise ValueError("History arrays must not be empty.")

    length = min(lengths)
    epochs = np.arange(1, length + 1)

    return epochs, [array[:length] for array in arrays]


def resolve_best_index(history, val_losses):
    fallback = int(np.argmin(val_losses))

    if "best_epoch" not in history.files:
        return fallback

    raw = int(np.asarray(history["best_epoch"]).item())
    candidates = []

    # Current training code saves a zero-based index.
    if 0 <= raw < len(val_losses):
        candidates.append(raw)

    # Compatibility with older files that may save one-based epoch numbers.
    if 1 <= raw <= len(val_losses):
        candidates.append(raw - 1)

    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        return fallback

    if "best_val_loss" in history.files:
        target = float(np.asarray(history["best_val_loss"]).item())
        return min(
            candidates,
            key=lambda index: abs(float(val_losses[index]) - target),
        )

    return candidates[0]


# ---------------------------------------------------------------------
# Formatting helpers
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


def make_log_ticks(values, count=3):
    positive = values[np.isfinite(values) & (values > 0)]

    if len(positive) == 0:
        return None

    minimum = float(np.min(positive))
    maximum = float(np.max(positive))

    if np.isclose(minimum, maximum):
        return np.array([minimum])

    return np.geomspace(minimum, maximum, count)


def style_axis(ax, show_x_axis):
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONT_SIZE,
        length=4,
        width=0.85,
        pad=4,
    )

    if not show_x_axis:
        ax.tick_params(axis="x", labelbottom=False)

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=4, integer=True)
    )

    if ax.get_yscale() != "log":
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=4)
        )

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#555555")

    ax.grid(False)
    ax.margins(x=0.025)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create a thesis-ready training-statistics figure."
    )

    parser.add_argument(
        "--npz",
        action="append",
        required=True,
        help=(
            "Dataset history as Label=path_to_history.npz. "
            "Use this option once per dataset."
        ),
    )
    parser.add_argument(
        "--out",
        default="training_stats_thesis.pdf",
        help="Output PDF or image path.",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=1.0,
        help="Gradient clipping reference value.",
    )
    parser.add_argument(
        "--mark_best",
        action="store_true",
        help="Draw the best-validation-epoch line.",
    )
    parser.add_argument(
        "--annotate_best",
        action="store_true",
        help="Annotate the best epoch and validation loss.",
    )
    parser.add_argument(
        "--lr_log",
        action="store_true",
        help="Use logarithmic scale for learning rate.",
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

    dataset_files = [
        parse_npz_argument(item)
        for item in args.npz
    ]

    row_count = len(dataset_files)

    if row_count == 0:
        raise ValueError("At least one --npz argument is required.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": TICK_FONT_SIZE,
        "axes.linewidth": SPINE_WIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=FIGURE_SIZE)

    # Use a dedicated spacer before the learning-rate column.
    # This prevents its y-axis labels from overlapping the gradient panel
    # without moving those labels outside the page boundary.
    grid = GridSpec(
        row_count,
        5,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 0.18, 1.0],
        left=0.115,
        right=0.975,
        top=0.855,
        bottom=0.105,
        wspace=0.28,
        hspace=0.36,
    )

    axes = np.empty((row_count, 4), dtype=object)
    grid_columns = [0, 1, 2, 4]

    for row in range(row_count):
        for column, grid_column in enumerate(grid_columns):
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
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
            pad=10,
        )

    row_labels = []

    for row, (dataset_name, npz_path) in enumerate(dataset_files):
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

        epochs, histories = align_histories(
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
        ) = histories

        best_index = resolve_best_index(
            history,
            val_losses,
        )
        best_epoch = int(epochs[best_index])
        best_val_loss = float(val_losses[best_index])

        show_x_axis = row == row_count - 1

        # Triplet loss
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
                linewidth=REFERENCE_LINE_WIDTH,
                alpha=0.9,
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
                    fontsize=ANNOTATION_FONT_SIZE,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                        "pad": 2,
                    },
                )

        style_axis(ax, show_x_axis)

        # Log-ratio
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
                linewidth=REFERENCE_LINE_WIDTH,
                alpha=0.9,
            )

        ax.set_ylim(*args.log_ratio_ylim)
        style_axis(ax, show_x_axis)

        # Gradient norm
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
            linewidth=REFERENCE_LINE_WIDTH,
            alpha=0.65,
        )
        ax.set_ylim(*args.grad_norm_ylim)
        style_axis(ax, show_x_axis)

        # Learning rate
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
            ticks = make_log_ticks(learning_rate, count=3)

            if ticks is not None:
                ax.set_yticks(ticks)
        else:
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=3)
            )

        ax.yaxis.set_major_formatter(
            FuncFormatter(compact_scientific)
        )
        ax.yaxis.get_offset_text().set_visible(False)

        # Keep learning-rate labels on the left. The dedicated spacer
        # between columns 3 and 4 prevents overlap with gradient norm.
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        ax.tick_params(
            axis="y",
            which="both",
            labelleft=True,
            labelright=False,
            pad=4,
        )

        if args.mark_best:
            ax.axvline(
                best_epoch,
                color=COLOR_BEST,
                linestyle="--",
                linewidth=REFERENCE_LINE_WIDTH,
                alpha=0.9,
            )

        style_axis(ax, show_x_axis)

        if show_x_axis:
            for column in range(4):
                axes[row, column].set_xlabel(
                    "Epoch",
                    fontsize=X_LABEL_FONT_SIZE,
                    labelpad=6,
                )

        row_labels.append(dataset_name)

        history.close()

    # Figure-level row labels are never clipped by subplot boundaries.
    fig.canvas.draw()

    for row, label in enumerate(row_labels):
        position = axes[row, 0].get_position()
        center_y = (position.y0 + position.y1) / 2

        fig.text(
            0.048,
            center_y,
            label,
            ha="center",
            va="center",
            fontsize=ROW_LABEL_FONT_SIZE,
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
        bbox_to_anchor=(0.55, 0.975),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.4,
        handletextpad=0.55,
        columnspacing=1.25,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Intentionally avoid bbox_inches="tight" so reserved margins remain intact.
    fig.savefig(
        output_path,
        dpi=args.dpi,
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
