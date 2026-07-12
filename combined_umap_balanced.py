#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from HiSiNet.HiCDatasetClass import HiCDatasetDec
import HiSiNet.models as models


DATASET_KEYS = ["Liver", "NPC", "TCell"]

DISPLAY_NAMES = {
    "Liver": "Liver",
    "NPC": "NPC",
    "TCell": "T Cell",
}

# 各資料集放在圖上方的簡短 Sample ID。
TOP_LEGENDS = {
    "Liver": [
        "NIPBL R1",
        "NIPBL R2",
        "TAM R1",
        "TAM R2",
    ],
    "NPC": [
        "Ctrl R1",
        "Ctrl R2",
        "Aux R1",
        "Aux R2",
    ],
    "TCell": [
        "DP R1",
        "DP R2",
        "SP R1",
        "SP R2",
    ],
}

UMAP_COLORS = ["#1F77B4", "#AEC7E8", "#D62728", "#FF9896"]


def parse_key_value(items, arg_name):
    out = {}

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"{arg_name} must use key=value format, got: {item}"
            )

        key, value = item.split("=", 1)
        out[key] = value

    return out


def resolve_json_path(json_file):
    json_path = os.path.abspath(os.path.expanduser(json_file))

    if os.path.exists(json_path):
        return json_path

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for base in [os.path.dirname(script_dir), script_dir, os.getcwd()]:
        candidate = os.path.normpath(os.path.join(base, json_file))

        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(f"config not found: {json_file}")


def load_model(
    model_name,
    ckpt_path,
    device,
    mask=False,
    embedding_dim=128,
):
    model = getattr(models, model_name)(
        mask=mask,
        embedding_dim=embedding_dim,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    state = OrderedDict(
        (key.replace("module.", ""), value)
        for key, value in state.items()
    )

    model.load_state_dict(state)
    model.eval()

    return model


def is_rep2_path(path):
    filename = os.path.basename(path).upper()
    return any(token in filename for token in ["R2", "REP2"])


def sample_embeddings_for_paths(
    model,
    paths,
    device,
    total_samples=5000,
    batch_size=64,
    seed=42,
):
    embeddings = []
    labels = []

    samples_per_file = max(
        1,
        int(total_samples) // max(1, len(paths)),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    with torch.no_grad():
        for path in paths:
            dataset = HiCDatasetDec.load(path)

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )

            is_r2 = is_rep2_path(path)
            count = 0

            for batch in loader:
                inputs = batch[0].to(device)
                class_ids = batch[-1].cpu().numpy()
                batch_embeddings = (
                    model.forward_one(inputs).cpu().numpy()
                )

                remaining = samples_per_file - count

                if remaining <= 0:
                    break

                sample_count = min(len(class_ids), remaining)

                embeddings.append(batch_embeddings[:sample_count])

                for class_id in class_ids[:sample_count]:
                    # class_id == 1 對應第一個條件，否則為第二個條件。
                    if int(class_id) == 1:
                        labels.append(2 if is_r2 else 1)
                    else:
                        labels.append(4 if is_r2 else 3)

                count += sample_count

                if count >= samples_per_file:
                    break

    if not embeddings:
        raise RuntimeError(
            "No embeddings were collected. "
            "Please check dataset paths and files."
        )

    return np.vstack(embeddings), np.asarray(labels, dtype=int)


def paths_for_subset(config, key, subset):
    if subset == "train_val":
        return (
            config[key]["training"]
            + config[key]["validation"]
        )

    if subset == "test":
        return config[key]["test"]

    raise ValueError(f"Unknown subset: {subset}")


def plot_one_panel(
    ax,
    coords,
    labels,
    title=None,
    point_size=4.0,
    alpha=0.45,
):
    handles = []

    for label_id, color in zip(
        [1, 2, 3, 4],
        UMAP_COLORS,
    ):
        selected = labels == label_id

        handle = ax.scatter(
            coords[selected, 0],
            coords[selected, 1],
            s=point_size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )

        handles.append(handle)

    if title:
        # 標題放在水平圖例上方。
        ax.set_title(
            title,
            fontsize=17.0,
            fontweight="bold",
            pad=34,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(0.80)
        ax.spines[side].set_color("#666666")

    return handles


def add_top_legend(ax, handles, labels):
    """
    將每個資料集的四個 Sample ID 排成一行，
    放在該資料集上方，不再占用圖的右側空間。
    """
    legend = ax.legend(
        handles,
        labels,
        ncol=4,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.008, 1.0, 0.060),
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
        fontsize=11.2,
        handletextpad=0.30,
        columnspacing=0.45,
        markerscale=1.80,
    )

    # 確保圖例中的點不會因原始散點透明度而太淡。
    for handle in legend.legend_handles:
        try:
            handle.set_alpha(0.9)
        except AttributeError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-style 2x3 UMAP figure "
            "with horizontal Sample ID legends above the panels."
        )
    )

    parser.add_argument(
        "--json_file",
        required=True,
        help="Path to config.json",
    )
    parser.add_argument(
        "--model_name",
        default="TripletLeNetBatchNormSE",
    )

    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help=(
            "Checkpoint mapping, e.g. "
            "--ckpt Liver=outputs/Liver/best.ckpt"
        ),
    )

    parser.add_argument(
        "--out",
        default="combined_umap_top_legend.pdf",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--mask",
        action="store_true",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
    )

    args = parser.parse_args()

    checkpoints = parse_key_value(args.ckpt, "--ckpt")

    missing = [
        key
        for key in DATASET_KEYS
        if key not in checkpoints
    ]

    if missing:
        raise ValueError(
            f"Missing checkpoint(s) for: {missing}. "
            f"Required keys: {DATASET_KEYS}"
        )

    json_path = resolve_json_path(args.json_file)

    with open(json_path, encoding="utf-8") as file:
        config = json.load(file)

    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"Using device: {device}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12.0,
        "axes.titlesize": 17.0,
        "axes.labelsize": 15.0,
        "legend.fontsize": 11.2,
        "axes.linewidth": 0.85,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # 移除原本右側圖例欄後，三欄子圖可放得更大。
    fig = plt.figure(figsize=(12.2, 7.4))

    grid = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.082,
        right=0.995,
        top=0.825,
        bottom=0.105,
        wspace=0.085,
        hspace=0.075,
    )

    axes = [
        [fig.add_subplot(grid[row, col]) for col in range(3)]
        for row in range(2)
    ]

    saved_data = {}

    subsets = ["train_val", "test"]
    row_labels = ["Train+Val", "Test"]

    for col, key in enumerate(DATASET_KEYS):
        print(
            f"\nLoading model for {key}: "
            f"{checkpoints[key]}"
        )

        model = load_model(
            args.model_name,
            checkpoints[key],
            device,
            mask=args.mask,
            embedding_dim=args.embedding_dim,
        )

        top_handles = None

        for row, subset in enumerate(subsets):
            print(f"  Processing {key} / {subset}...")

            paths = paths_for_subset(
                config,
                key,
                subset,
            )

            embeddings, labels = sample_embeddings_for_paths(
                model,
                paths,
                device,
                total_samples=args.total_samples,
                batch_size=args.batch_size,
                seed=args.seed + row * 100 + col,
            )

            reducer = umap.UMAP(
                random_state=args.seed,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                metric="euclidean",
            )

            coordinates = reducer.fit_transform(embeddings)

            title = DISPLAY_NAMES[key] if row == 0 else None

            handles = plot_one_panel(
                axes[row][col],
                coordinates,
                labels,
                title=title,
                point_size=args.point_size,
                alpha=args.alpha,
            )

            if row == 0:
                top_handles = handles

            saved_data[
                f"{key}_{subset}_coords"
            ] = coordinates
            saved_data[
                f"{key}_{subset}_labels"
            ] = labels

        # 四個 Sample ID 直接橫向放在該欄上方。
        add_top_legend(
            axes[0][col],
            top_handles,
            TOP_LEGENDS[key],
        )

    fig.canvas.draw()

    top_left_position = axes[0][0].get_position()
    bottom_left_position = axes[1][0].get_position()
    top_right_position = axes[0][2].get_position()

    y_train = (
        top_left_position.y0 + top_left_position.y1
    ) / 2
    y_test = (
        bottom_left_position.y0 + bottom_left_position.y1
    ) / 2

    x_center_panels = (
        top_left_position.x0 + top_right_position.x1
    ) / 2

    fig.text(
        x_center_panels,
        0.040,
        "UMAP Dimension 1",
        ha="center",
        va="center",
        fontsize=15.0,
    )

    fig.text(
        0.021,
        (
            bottom_left_position.y0
            + top_left_position.y1
        ) / 2,
        "UMAP Dimension 2",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=15.0,
    )

    fig.text(
        0.060,
        y_train,
        row_labels[0],
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=14.0,
        fontweight="bold",
    )

    fig.text(
        0.060,
        y_test,
        row_labels[1],
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=14.0,
        fontweight="bold",
    )

    output_directory = os.path.dirname(
        os.path.abspath(args.out)
    )

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    fig.savefig(
        args.out,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)

    print(f"\nSaved figure: {args.out}")

    if args.save_npz:
        npz_output = (
            os.path.splitext(args.out)[0]
            + "_umap_coords.npz"
        )

        np.savez_compressed(
            npz_output,
            **saved_data,
        )

        print(
            f"Saved UMAP coordinates: {npz_output}"
        )


if __name__ == "__main__":
    main()
