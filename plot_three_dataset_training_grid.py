#!/usr/bin/env python3
"""
Create a publication-style 3 × 4 training-statistics figure from history NPZ files.

Main layout changes:
- Larger plotting area with balanced left/right margins.
- Compact, readable top legend.
- More horizontal spacing before the learning-rate column.
- Compact scientific notation for learning-rate ticks.
- Column titles appear only once.
- X-axis labels and tick labels appear only on the bottom row.
- Export keeps the manually defined margins instead of using tight cropping.

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
    python plot_three_dataset_training_grid_refined.py \
        --npz "Liver=outputs/Liver_history.npz" \
        --npz "NPC=outputs/NPC_history.npz" \
        --npz "T Cell=outputs/TCell_history.npz" \
        --out three_dataset_training_stats_refined.pdf \
        --mark_best \
        --lr_log
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


def read_array(history, key, default=None):
    if key in history.files:
        array = np.asarray(history[key])

        if array.ndim == 0:
            array = array.reshape(1)

        return array.astype(float)

    if default is not None:
        return np.asarray(default, dtype=float)

    raise KeyError(
        f"Missing key '{key}' in NPZ. "
        f"Available keys: {history.files}"
    )


def normalize_display_label(label):
    clean = label.strip()
    normalized = (
        clean.lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    if normalized == "liver":
        return "Liver"

    if normalized == "npc":
        return "NPC"

    if normalized in {"tcell", "t cell"}:
        return "T Cell"

    return clean


def parse_labeled_npz(item):
    if "=" not in item:
        path = Path(item)
        label = path.stem.replace("_history", "")
        return normalize_display_label(label), path

    label, path = item.split("=", 1)
    return normalize_display_label(label), Path(path.strip())


def align_epochs(*arrays):
    lengths = [
        len(array)
        for array in arrays
        if array is not None and len(array) > 0
    ]

    if not lengths:
        raise ValueError("No non-empty arrays were found.")

    sample_count = min(lengths)
    epochs = np.arange(1, sample_count + 1)

    aligned = [
        array[:sample_count]
        if array is not None and len(array) > 0
        else None
        for array in arrays
    ]

    return epochs, aligned


def get_best_index(history, val_losses):
    """
    Resolve best_epoch robustly.

    The current training script stores best_epoch as a zero-based array index.
    For compatibility with older files, the function also checks whether a
    one-based interpretation better matches best_val_loss.
    """
    fallback_index = int(np.argmin(val_losses))

    if "best_epoch" not in history.files:
        return fallback_index

    raw = int(np.asarray(history["best_epoch"]).item())
    candidates = []

    if 0 <= raw < len(val_losses):
        candidates.append(raw)

    if 1 <= raw <= len(val_losses):
        candidates.append(raw - 1)

    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        return fallback_index

    if "best_val_loss" in history.files:
        target = float(np.asarray(history["best_val_loss"]).item())

        return min(
            candidates,
            key=lambda index: abs(float(val_losses[index]) - target),
        )

    # Prefer the current zero-based convention.
    return candidates[0]


def compact_scientific(value, _position):
    """Format ticks as 1e-3, 6e-4, etc., to avoid wide math-text labels."""
    if value == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / (10 ** exponent)

    if np.isclose(coefficient, round(coefficient), atol=1e-8):
        coefficient_text = f"{int(round(coefficient))}"
    else:
        coefficient_text = f"{coefficient:.1f}".rstrip("0").rstrip(".")

    return f"{coefficient_text}e{exponent}"


def style_axis(ax, show_x_ticks):
    ax.tick_params(
        axis="both",
        labelsize=10.5,
        length=3.5,
        width=0.8,
        pad=3.0,
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    if ax.get_yscale() != "log":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    if not show_x_ticks:
        ax.tick_params(axis="x", labelbottom=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#555555")

    ax.grid(False)
    ax.margins(x=0.03)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-style multi-dataset "
            "training-statistics grid."
        )
    )

    parser.add_argument(
        "--npz",
        action="append",
        required=True,
        help=(
            "Input as Label=path_to_history.npz. "
            "Use this option multiple times."
        ),
    )
    parser.add_argument(
        "--out",
        default="training_stats_grid_refined.pdf",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=1.0,
        help="Reference line for gradient clipping max norm.",
    )
    parser.add_argument(
        "--mark_best",
        action="store_true",
        help="Draw the best-validation-epoch line.",
    )
    parser.add_argument(
        "--annotate_best",
        action="store_true",
        help="Annotate best epoch and validation loss in loss panels.",
    )
    parser.add_argument(
        "--lr_log",
        action="store_true",
        help="Use logarithmic scale for the learning-rate axis.",
    )
    parser.add_argument(
        "--log_ratio_ylim",
        type=float,
        nargs=2,
        default=(0.0, 0.9),
        metavar=("YMIN", "YMAX"),
    )
    parser.add_argument(
        "--grad_norm_ylim",
        type=float,
        nargs=2,
        default=(0.0, 1.6),
        metavar=("YMIN", "YMAX"),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11.0,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    labeled_paths = [
        parse_labeled_npz(item)
        for item in args.npz
    ]

    row_count = len(labeled_paths)
    column_count = 4

    # Balanced for insertion into a portrait thesis page at full text width.
    figure_height = 2.05 * row_count + 1.25

    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(9.2, figure_height),
        squeeze=False,
        sharex=False,
    )

    column_titles = [
        "Triplet loss",
        "Log-ratio",
        "Gradient norm",
        "Learning rate",
    ]

    color_train = "#1f77b4"
    color_validation = "#ff7f0e"
    color_log_ratio = "#0000ff"
    color_gradient = "#008080"
    color_learning_rate = "#6A3D9A"
    color_reference = "#ff0000"
    color_best = "0.45"

    for row_index, (label, npz_path) in enumerate(labeled_paths):
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

        epochs, aligned_arrays = align_epochs(
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
        ) = aligned_arrays

        best_index = get_best_index(history, val_losses)
        best_epoch = int(epochs[best_index])
        best_val_loss = float(val_losses[best_index])

        # Dataset label: placed in the reserved left margin.
        axes[row_index, 0].annotate(
            label,
            xy=(-0.45, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=13.5,
            fontweight="bold",
            clip_on=False,
        )

        if row_index == 0:
            for column_index, title in enumerate(column_titles):
                axes[row_index, column_index].set_title(
                    title,
                    fontsize=13.0,
                    fontweight="bold",
                    pad=8,
                )

        show_bottom_axis = row_index == row_count - 1

        # 1. Triplet loss
        ax = axes[row_index, 0]

        ax.plot(
            epochs,
            train_losses,
            linewidth=1.60,
            color=color_train,
        )
        ax.plot(
            epochs,
            val_losses,
            linewidth=1.60,
            color=color_validation,
        )

        if args.mark_best:
            ax.axvline(
                best_epoch,
                color=color_best,
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
            )
            ax.scatter(
                [best_epoch],
                [best_val_loss],
                s=19,
                color=color_best,
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
                    fontsize=9.0,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 1.5,
                    },
                )

        style_axis(ax, show_bottom_axis)

        # 2. Log-ratio
        ax = axes[row_index, 1]

        ax.plot(
            epochs,
            log_ratio,
            linewidth=1.60,
            color=color_log_ratio,
        )

        if args.mark_best:
            ax.axvline(
                best_epoch,
                color=color_best,
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
            )

        ax.set_ylim(*args.log_ratio_ylim)
        style_axis(ax, show_bottom_axis)

        # 3. Gradient norm
        ax = axes[row_index, 2]

        ax.plot(
            epochs,
            gradient_norm,
            linewidth=1.60,
            color=color_gradient,
        )
        ax.axhline(
            args.max_norm,
            color=color_reference,
            linestyle="--",
            linewidth=0.85,
            alpha=0.58,
        )
        ax.set_ylim(*args.grad_norm_ylim)
        style_axis(ax, show_bottom_axis)

        # 4. Learning rate
        ax = axes[row_index, 3]

        ax.plot(
            epochs,
            learning_rate,
            linewidth=1.60,
            color=color_learning_rate,
        )

        if args.lr_log and np.all(learning_rate > 0):
            ax.set_yscale("log")

        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(
            FuncFormatter(compact_scientific)
        )

        if args.mark_best:
            ax.axvline(
                best_epoch,
                color=color_best,
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
            )

        style_axis(ax, show_bottom_axis)

        if show_bottom_axis:
            for column_index in range(column_count):
                axes[row_index, column_index].set_xlabel(
                    "Epoch",
                    fontsize=11.5,
                    labelpad=5,
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_train,
            linewidth=1.7,
            label="Train",
        ),
        Line2D(
            [0],
            [0],
            color=color_validation,
            linewidth=1.7,
            label="Validation",
        ),
    ]

    if args.mark_best:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color_best,
                linestyle="--",
                linewidth=1.0,
                label="Best val. epoch",
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=color_reference,
            linestyle="--",
            linewidth=1.0,
            label=(
                "Clip max norm "
                f"= {args.max_norm:g}"
            ),
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=10.5,
        handlelength=2.0,
        handletextpad=0.55,
        columnspacing=0.95,
    )

    # Deliberately reserve more space before the last column so its
    # scientific-notation labels never overlap the gradient panel.
    fig.subplots_adjust(
        left=0.18,
        right=0.985,
        top=0.865,
        bottom=0.16,
        wspace=0.48,
        hspace=0.35,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Do not use bbox_inches="tight": it would remove the balanced margins.
    fig.savefig(
        output_path,
        dpi=args.dpi,
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()