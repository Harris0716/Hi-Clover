#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import umap

from HiSiNet.HiCDatasetClass import HiCDatasetDec
import HiSiNet.models as models


DATASET_KEYS = ["Liver", "NPC", "TCell"]

DISPLAY_NAMES = {
    "Liver": "Liver",
    "NPC": "NPC",
    "TCell": "T Cell",
}

LEGENDS = {
    "Liver": [
        "Liver NIPBL R1",
        "Liver NIPBL R2",
        "Liver TAM R1",
        "Liver TAM R2",
    ],
    "NPC": [
        "NPC Ctrl R1",
        "NPC Ctrl R2",
        "NPC Treat (Aux) R1",
        "NPC Treat (Aux) R2",
    ],
    "TCell": [
        "T Cell Ctrl (DP) R1",
        "T Cell Ctrl (DP) R2",
        "T Cell Treat (SP) R1",
        "T Cell Treat (SP) R2",
    ],
}

UMAP_COLORS = ["#1F77B4", "#AEC7E8", "#D62728", "#FF9896"]


def parse_key_value(items, arg_name):
    out = {}

    for item in items:
        if "=" not in item:
            raise ValueError(f"{arg_name} must use key=value format, got: {item}")

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


def load_model(model_name, ckpt_path, device, mask=False, embedding_dim=128):
    model = getattr(models, model_name)(
        mask=mask,
        embedding_dim=embedding_dim,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())

    model.load_state_dict(state)
    model.eval()

    return model


def is_rep2_path(path):
    up = os.path.basename(path).upper()
    return any(token in up for token in ["R2", "REP2"])


def sample_embeddings_for_paths(
    model,
    paths,
    device,
    total_samples=5000,
    batch_size=64,
    seed=42,
):
    embs = []
    labels = []

    samples_per_file = max(1, int(total_samples) // max(1, len(paths)))

    generator = torch.Generator()
    generator.manual_seed(seed)

    with torch.no_grad():
        for path in paths:
            ds = HiCDatasetDec.load(path)

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )

            is_r2 = is_rep2_path(path)
            count = 0

            for batch in loader:
                x = batch[0].to(device)
                class_ids = batch[-1].cpu().numpy()
                emb = model.forward_one(x).cpu().numpy()

                remaining = samples_per_file - count

                if remaining <= 0:
                    break

                n = min(len(class_ids), remaining)

                embs.append(emb[:n])

                for cid in class_ids[:n]:
                    # class_id == 1 -> labels 1/2; otherwise -> labels 3/4.
                    if int(cid) == 1:
                        labels.append(2 if is_r2 else 1)
                    else:
                        labels.append(4 if is_r2 else 3)

                count += n

                if count >= samples_per_file:
                    break

    if not embs:
        raise RuntimeError("No embeddings were collected. Please check dataset paths and files.")

    return np.vstack(embs), np.asarray(labels, dtype=int)


def paths_for_subset(config, key, subset):
    if subset == "train_val":
        return config[key]["training"] + config[key]["validation"]

    if subset == "test":
        return config[key]["test"]

    raise ValueError(f"Unknown subset: {subset}")


def plot_one_panel(
    ax,
    coords,
    labels,
    legend_labels,
    title=None,
    point_size=4.0,
    alpha=0.45,
):
    handles = []

    for label_id, color, legend_name in zip([1, 2, 3, 4], UMAP_COLORS, legend_labels):
        idx = labels == label_id

        h = ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=point_size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            label=legend_name,
            rasterized=True,
        )

        handles.append(h)

    if title:
        ax.set_title(title, fontweight="bold", pad=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#666666")

    return handles


def add_right_legend(legend_ax, col_legend_handles):
    legend_ax.axis("off")

    legend_ax.text(
        0.0,
        0.98,
        "Sample ID",
        fontsize=15,
        fontweight="bold",
        va="top",
    )

    y = 0.88

    for key in DATASET_KEYS:
        legend_ax.text(
            0.0,
            y,
            DISPLAY_NAMES[key],
            fontsize=13.5,
            fontweight="bold",
            va="top",
        )

        y -= 0.075

        handles = col_legend_handles[key]
        labels = LEGENDS[key]

        for h, label in zip(handles, labels):
            color = h.get_facecolor()[0]

            legend_ax.scatter(
                0.04,
                y,
                s=70,
                color=color,
                alpha=0.80,
                edgecolors="none",
            )

            legend_ax.text(
                0.12,
                y,
                label,
                fontsize=12.4,
                va="center",
            )

            y -= 0.058

        y -= 0.060

    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Create a publication-style 2x3 UMAP figure with a large right-side legend."
    )

    parser.add_argument("--json_file", required=True, help="Path to config.json")
    parser.add_argument("--model_name", default="TripletLeNetBatchNormSE")

    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="Checkpoint mapping, e.g. --ckpt liver=outputs/liver/best.ckpt",
    )

    parser.add_argument("--out", default="combined_umap_right_large_legend.pdf")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--total_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_neighbors", type=int, default=80)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--point_size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save_npz", action="store_true")

    args = parser.parse_args()

    ckpts = parse_key_value(args.ckpt, "--ckpt")

    missing = [k for k in DATASET_KEYS if k not in ckpts]

    if missing:
        raise ValueError(f"Missing checkpoint(s) for: {missing}. Required keys: {DATASET_KEYS}")

    json_path = resolve_json_path(args.json_file)

    with open(json_path) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.titlesize": 14.0,
        "axes.labelsize": 12.0,
        "legend.fontsize": 12.0,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(11.8, 6.4))

    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 0.78],
        height_ratios=[1.0, 1.0],
        left=0.08,
        right=0.98,
        top=0.90,
        bottom=0.14,
        wspace=0.16,
        hspace=0.12,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    legend_ax = fig.add_subplot(gs[:, 3])
    legend_ax.axis("off")

    saved_data = {}

    subsets = ["train_val", "test"]
    row_labels = ["Train + Val", "Test"]
    col_legend_handles = {}

    for col, key in enumerate(DATASET_KEYS):
        print(f"\nLoading model for {key}: {ckpts[key]}")

        model = load_model(
            args.model_name,
            ckpts[key],
            device,
            mask=args.mask,
            embedding_dim=args.embedding_dim,
        )

        for row, subset in enumerate(subsets):
            print(f"  Processing {key} / {subset}...")

            paths = paths_for_subset(config, key, subset)

            embs, labels = sample_embeddings_for_paths(
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

            coords = reducer.fit_transform(embs)

            title = DISPLAY_NAMES[key] if row == 0 else None

            handles = plot_one_panel(
                axes[row][col],
                coords,
                labels,
                LEGENDS[key],
                title=title,
                point_size=args.point_size,
                alpha=args.alpha,
            )

            if row == 0:
                col_legend_handles[key] = handles

            saved_data[f"{key}_{subset}_coords"] = coords
            saved_data[f"{key}_{subset}_labels"] = labels

    add_right_legend(legend_ax, col_legend_handles)

    fig.canvas.draw()

    pos_top = axes[0][0].get_position()
    pos_bottom = axes[1][0].get_position()

    y_train = (pos_top.y0 + pos_top.y1) / 2
    y_test = (pos_bottom.y0 + pos_bottom.y1) / 2

    fig.text(
        0.455,
        0.065,
        "UMAP Dimension 1",
        ha="center",
        va="center",
        fontsize=13.0,
    )

    fig.text(
        0.028,
        (pos_bottom.y0 + pos_top.y1) / 2,
        "UMAP Dimension 2",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=13.0,
    )

    fig.text(
        0.052,
        y_train,
        row_labels[0],
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12.0,
        fontweight="bold",
    )

    fig.text(
        0.052,
        y_test,
        row_labels[1],
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12.0,
        fontweight="bold",
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure: {args.out}")

    if args.save_npz:
        npz_out = os.path.splitext(args.out)[0] + "_umap_coords.npz"
        np.savez_compressed(npz_out, **saved_data)
        print(f"Saved UMAP coordinates: {npz_out}")


if __name__ == "__main__":
    main()