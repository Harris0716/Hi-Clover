import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
import argparse, json, os, pandas as pd, matplotlib.pyplot as plt
import time
from scipy.integrate import simpson
from numpy import minimum
from collections import OrderedDict
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
import umap

from HiSiNet.HiCDatasetClass import HiCDatasetDec, SiameseHiCDataset, GroupedHiCDataset
import HiSiNet.models as models
from HiSiNet.reference_dictionaries import reference_genomes


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def format_seconds(seconds):
    seconds = float(seconds)
    minutes = seconds / 60
    hours = seconds / 3600

    return f"{seconds:.2f}s ({minutes:.2f} min, {hours:.4f} h)"


# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Latent Space Evaluation and Visualization')
parser.add_argument('model_name', type=str)
parser.add_argument('json_file', type=str)
parser.add_argument('model_infile', type=str)
parser.add_argument('--mask', action='store_true', help='Mask diagonal')

# threshold_data: "train_val" = use train+val for threshold (default, stabler); "val" = validation only
parser.add_argument(
    '--threshold_data',
    type=str,
    default='train_val',
    choices=['val', 'train_val'],
    help='Data used to calibrate decision threshold. train_val=train+val (default, stabler); val=validation only.'
)

parser.add_argument(
    '--threshold_method',
    type=str,
    default='best_mean',
    choices=['best_mean', 'intersection'],
    help='Threshold calibration method. best_mean maximizes balanced mean performance on threshold_data; intersection uses the first histogram intersection.'
)

parser.add_argument("data_inputs", nargs='+')
parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
parser.add_argument(
    '--fixed_threshold',
    type=float,
    default=None,
    help='Use a manually specified threshold and skip threshold calibration.'
)
parser.add_argument(
    '--test_only',
    action='store_true',
    help='Evaluate only the test set. Training and validation mlhic files will not be loaded for evaluation.'
)

args = parser.parse_args()

script_start_time = time.perf_counter()


def test_triplet(model, dataloader, device):
    distances, labels = [], []

    model.eval()

    with torch.no_grad():
        for data in dataloader:
            o1 = model.forward_one(data[0].to(device))
            o2 = model.forward_one(data[1].to(device))

            # keep Euclidean distance
            distances.extend(F.pairwise_distance(o1, o2).cpu().numpy())
            labels.extend(data[2].numpy())

    return np.array(distances), np.array(labels)


def _threshold_by_best_mean(distances, labels, n_grid=5000):
    """Choose threshold using only calibration data.

    Replicate pairs are positive when distance < threshold.
    Condition pairs are positive when distance >= threshold.
    The selected threshold maximizes balanced mean performance:
        (replicate_accuracy + condition_accuracy) / 2
    """
    rep_dist = distances[labels == 0]
    cond_dist = distances[labels == 1]

    lo = float(np.min(distances))
    hi = float(np.percentile(distances, 99.9))
    grid = np.linspace(lo, hi, n_grid)

    rep_sorted = np.sort(rep_dist)
    cond_sorted = np.sort(cond_dist)

    # rep_rate = P(rep distance < threshold)
    rep_counts = np.searchsorted(rep_sorted, grid, side='left')
    rep_rate = rep_counts / max(len(rep_sorted), 1)

    # cond_rate = P(condition distance >= threshold)
    cond_less = np.searchsorted(cond_sorted, grid, side='left')
    cond_rate = 1.0 - cond_less / max(len(cond_sorted), 1)

    mean_perf = (rep_rate + cond_rate) / 2
    best_i = int(np.argmax(mean_perf))

    return float(grid[best_i])


def _threshold_by_first_intersection(rep_dist, cond_dist, rng):
    a = np.histogram(rep_dist, bins=rng, density=True)
    b = np.histogram(cond_dist, bins=rng, density=True)

    idx = np.where(np.diff(np.sign(a[0] - b[0])))[0]

    return float(a[1][idx[0]]) if len(idx) > 0 else float(a[1][len(a[1]) // 2])


def calculate_metrics(distances, labels, fixed_threshold=None, threshold_method='best_mean'):
    rng = np.linspace(distances.min(), np.percentile(distances, 99.5), 200)

    rep_dist = distances[labels == 0]
    cond_dist = distances[labels == 1]

    a = np.histogram(rep_dist, bins=rng, density=True)
    b = np.histogram(cond_dist, bins=rng, density=True)

    if fixed_threshold is None:
        if threshold_method == 'best_mean':
            threshold = _threshold_by_best_mean(distances, labels)
        elif threshold_method == 'intersection':
            threshold = _threshold_by_first_intersection(rep_dist, cond_dist, rng)
        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method}")
    else:
        threshold = float(fixed_threshold)

    overlap = minimum(a[0], b[0])
    sep_idx = 1 - simpson(overlap, x=(a[1][1:] + a[1][:-1]) / 2)

    return {
        "intersect": threshold,
        "rep_rate": np.sum(rep_dist < threshold) / len(rep_dist),
        "cond_rate": np.sum(cond_dist >= threshold) / len(cond_dist),
        "sep_index": sep_idx,
        "hist_data": (a, b, rng)
    }


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = eval("models." + args.model_name)(
    mask=args.mask,
    embedding_dim=args.embedding_dim
).to(device)

sd = torch.load(args.model_infile, map_location=device, weights_only=True)

model.load_state_dict(
    OrderedDict([(k.replace("module.", ""), v) for k, v in sd.items()])
)

json_path = os.path.abspath(os.path.expanduser(args.json_file))

if not os.path.exists(json_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for base in [os.path.dirname(script_dir), script_dir, os.getcwd()]:
        candidate = os.path.normpath(os.path.join(base, args.json_file))

        if os.path.exists(candidate):
            json_path = os.path.abspath(candidate)
            break

if not os.path.exists(json_path):
    raise FileNotFoundError(f"config not found: {args.json_file} (tried {json_path})")

with open(json_path) as f:
    dataset_config = json.load(f)

m_dir = os.path.dirname(args.model_infile)
m_base = os.path.basename(args.model_infile).split('.ckpt')[0]

cell_name = args.data_inputs[0]
cell_title = cell_name

# set Legend
if "NPC" in cell_name.upper():
    lgd = [
        "NPC Ctrl R1",
        "NPC Ctrl R2",
        "NPC Treat (Aux) R1",
        "NPC Treat (Aux) R2"
    ]
elif "LIVER" in cell_name.upper():
    lgd = [
        "Liver NIPBL R1",
        "Liver NIPBL R2",
        "Liver TAM R1",
        "Liver TAM R2"
    ]
elif "TCELL" in cell_name.upper():
    lgd = [
        "TCells Ctrl (DP) R1",
        "TCells Ctrl (DP) R2",
        "TCells Treat (SP) R1",
        "TCells Treat (SP) R2"
    ]
else:
    lgd = [
        f"{cell_name} R1",
        f"{cell_name} R2",
        "Treat R1",
        "Treat R2"
    ]

param_info = f"Model: {args.model_name} | Embed: {args.embedding_dim} | Mask: {args.mask}"

timing_records = []


# ---------------------------------------------------------
# Step 1: Obtain threshold
# ---------------------------------------------------------
if args.fixed_threshold is not None:
    if args.fixed_threshold < 0:
        raise ValueError("--fixed_threshold must be greater than or equal to 0.")

    fixed_threshold = float(args.fixed_threshold)
    threshold_source = "Manual"
    threshold_data_used = "manual"
    threshold_method_used = "manual"
    threshold_time_sec = 0.0

    timing_records.append({
        "dataset": cell_name,
        "stage": "threshold_manual",
        "subset": "manual",
        "time_sec": 0.0,
        "time_min": 0.0,
        "time_hour": 0.0
    })

    print(f"Step 1: Using manually specified threshold ({cell_name})...")
    print(f"  -> Threshold = {fixed_threshold:.4f}")
    print("  -> Threshold calibration skipped.")
else:
    threshold_source = "Validation" if args.threshold_data == "val" else "Train+Val"
    threshold_data_used = args.threshold_data
    threshold_method_used = args.threshold_method

    print(f"Step 1: Calibrating Threshold from {threshold_source} ({cell_name})...")

    threshold_start_time = time.perf_counter()

    if args.threshold_data == "val":
        cal_paths = [
            p
            for d in args.data_inputs
            for p in dataset_config[d]["validation"]
        ]
    else:
        cal_paths = [
            p
            for d in args.data_inputs
            for p in (dataset_config[d]["training"] + dataset_config[d]["validation"])
        ]

    cal_ds = GroupedHiCDataset([
        SiameseHiCDataset(
            [HiCDatasetDec.load(p) for p in cal_paths],
            reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]]
        )
    ])

    cal_loader = DataLoader(cal_ds, batch_size=128)
    cal_dist, cal_lbl = test_triplet(model, cal_loader, device)

    fixed_threshold = calculate_metrics(
        cal_dist,
        cal_lbl,
        threshold_method=threshold_method_used
    )["intersect"]

    threshold_end_time = time.perf_counter()
    threshold_time_sec = threshold_end_time - threshold_start_time

    timing_records.append({
        "dataset": cell_name,
        "stage": "threshold_calibration",
        "subset": args.threshold_data,
        "time_sec": threshold_time_sec,
        "time_min": threshold_time_sec / 60,
        "time_hour": threshold_time_sec / 3600
    })

    print(
        f"  -> Threshold = {fixed_threshold:.4f} "
        f"(method={args.threshold_method}, from {threshold_source})"
    )
    print(f"  -> Threshold calibration time: {format_seconds(threshold_time_sec)}")


# ---------------------------------------------------------
# Step 2: Evaluate train_val and test
# ---------------------------------------------------------
results = []

subsets_to_evaluate = ["test"] if args.test_only else ["train_val", "test"]

for subset in subsets_to_evaluate:
    subset_name = "TRAIN/VAL" if subset == "train_val" else "TEST"

    print(f"\nStep 2: Processing {subset.upper()}...")

    subset_total_start_time = time.perf_counter()

    paths = [
        p
        for d in args.data_inputs
        for p in (
            dataset_config[d]["training"] + dataset_config[d]["validation"]
            if subset == "train_val"
            else dataset_config[d]["test"]
        )
    ]

    # -----------------------------------------------------
    # Core evaluation timing:
    # dataset construction + model inference + distance calculation + metrics
    # This is the value recommended for thesis Testing Time.
    # -----------------------------------------------------
    eval_start_time = time.perf_counter()

    ds = GroupedHiCDataset([
        SiameseHiCDataset(
            [HiCDatasetDec.load(p) for p in paths],
            reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]]
        )
    ])

    loader = DataLoader(ds, batch_size=128)

    dist, lbl = test_triplet(model, loader, device)

    data = calculate_metrics(dist, lbl, fixed_threshold=fixed_threshold)

    eval_end_time = time.perf_counter()
    eval_time_sec = eval_end_time - eval_start_time

    timing_records.append({
        "dataset": cell_name,
        "stage": "core_evaluation",
        "subset": subset,
        "time_sec": eval_time_sec,
        "time_min": eval_time_sec / 60,
        "time_hour": eval_time_sec / 3600
    })

    print(f"  -> Core evaluation time ({subset}): {format_seconds(eval_time_sec)}")

    # sample calculation of embeddings
    embedding_start_time = time.perf_counter()

    embs, detailed_lbls = [], []
    samples_per_file = max(1, 5000 // len(paths))

    with torch.no_grad():
        for p in paths:
            temp_ds = HiCDatasetDec.load(p)
            ldr = DataLoader(temp_ds, batch_size=64, shuffle=True)

            is_r2 = 1 if any(x in p.upper() for x in ['R2', 'REP2']) else 0
            count = 0

            for batch in ldr:
                eb = model.forward_one(batch[0].to(device))

                num = min(len(batch[-1]), samples_per_file - count)

                if num > 0:
                    embs.extend(eb[:num].cpu().numpy())

                    for cid in batch[-1][:num].numpy():
                        detailed_lbls.append(
                            (1 if is_r2 == 0 else 2)
                            if cid == 1
                            else (3 if is_r2 == 0 else 4)
                        )

                    count += num

                if count >= samples_per_file:
                    break

    embs = np.array(embs)
    detailed_lbls = np.array(detailed_lbls)

    embedding_end_time = time.perf_counter()
    embedding_time_sec = embedding_end_time - embedding_start_time

    timing_records.append({
        "dataset": cell_name,
        "stage": "embedding_sampling_for_visualization",
        "subset": subset,
        "time_sec": embedding_time_sec,
        "time_min": embedding_time_sec / 60,
        "time_hour": embedding_time_sec / 3600
    })

    print(f"  -> Embedding sampling time ({subset}): {format_seconds(embedding_time_sec)}")

    # calculate the binary silhouette coefficient
    binary_lbls = [0 if (l == 1 or l == 2) else 1 for l in detailed_lbls]
    sil_score = silhouette_score(embs, binary_lbls, metric='euclidean')

    mean_perf = (data["rep_rate"] + data["cond_rate"]) / 2

    results.append({
        "threshold_data": threshold_data_used,
        "threshold_method": threshold_method_used,
        "intersect": fixed_threshold,
        "set": subset,
        "rep_rate": data["rep_rate"],
        "cond_rate": data["cond_rate"],
        "mean_performance": mean_perf,
        "sep_index": data["sep_index"],
        "silhouette": sil_score,
        "core_eval_time_sec": eval_time_sec,
        "core_eval_time_min": eval_time_sec / 60,
        "core_eval_time_hour": eval_time_sec / 3600
    })

    # -----------------------------------------------------
    # 1. Histogram
    # -----------------------------------------------------
    plt.figure(figsize=(9, 6))

    plt.hist(
        dist[lbl == 0],
        bins=data["hist_data"][2],
        density=True,
        label='Replicates',
        alpha=0.5,
        color='#108690'
    )

    plt.hist(
        dist[lbl == 1],
        bins=data["hist_data"][2],
        density=True,
        label='Conditions',
        alpha=0.5,
        color='#1D1E4E'
    )

    plt.axvline(
        fixed_threshold,
        color='k',
        ls='--',
        label=f'Threshold ({fixed_threshold:.2f})'
    )

    plt.title(
        f"Distance Distribution: {cell_title} ({subset_name})\n"
        f"{param_info}\n"
        f"Separation Index: {data['sep_index']:.4f} | Mean Performance: {mean_perf:.4f}",
        fontsize=12,
        fontweight='bold'
    )

    plt.xlabel("Euclidean Distance")
    plt.ylabel("Probability Density")
    plt.legend()

    plt.savefig(
        os.path.join(m_dir, f"{m_base}_{subset}_dist_hist.pdf"),
        bbox_inches='tight'
    )

    plt.close()

    np.savez_compressed(
        os.path.join(m_dir, f"{cell_name}_{subset}_raw_dist.npz"),
        dist=dist,
        lbl=lbl,
        threshold=fixed_threshold,
        threshold_method=threshold_method_used,
        threshold_data=threshold_data_used
    )

    cmap = ListedColormap(['#1F77B4', '#AEC7E8', '#D62728', '#FF9896'])

    # -----------------------------------------------------
    # 2. t-SNE
    # -----------------------------------------------------
    # print(f"Calculating t-SNE for {subset}...")
    # res_tsne = TSNE(
    #     n_components=2,
    #     perplexity=40,
    #     random_state=42,
    #     early_exaggeration=20,
    #     metric='euclidean'
    # ).fit_transform(embs)
    #
    # plt.figure(figsize=(10, 8))
    # scat = plt.scatter(
    #     res_tsne[:, 0],
    #     res_tsne[:, 1],
    #     c=detailed_lbls,
    #     cmap=cmap,
    #     s=10,
    #     alpha=0.5
    # )
    #
    # plt.legend(
    #     handles=scat.legend_elements()[0],
    #     labels=lgd,
    #     title="Sample ID"
    # )
    #
    # plt.title(
    #     f"Latent Space: t-SNE Projection | {cell_title} ({subset_name})\n"
    #     f"{param_info}\n"
    #     f"Silhouette Score: {sil_score:.4f}",
    #     fontsize=12,
    #     fontweight='bold'
    # )
    #
    # plt.savefig(
    #     os.path.join(m_dir, f"{m_base}_{subset}_tsne.pdf"),
    #     bbox_inches='tight'
    # )
    #
    # plt.close()

    # -----------------------------------------------------
    # 3. UMAP
    # -----------------------------------------------------
    umap_start_time = time.perf_counter()

    print(f"Calculating UMAP for {subset}...")

    res_umap = umap.UMAP(
        random_state=42,
        n_neighbors=80,
        min_dist=0.1,
        metric='euclidean'
    ).fit_transform(embs)

    plt.figure(figsize=(10, 8))

    scat = plt.scatter(
        res_umap[:, 0],
        res_umap[:, 1],
        c=detailed_lbls,
        cmap=cmap,
        s=10,
        alpha=0.5
    )

    plt.legend(
        handles=scat.legend_elements()[0],
        labels=lgd,
        title="Sample ID"
    )

    plt.title(
        f"Latent Space: UMAP Projection | {cell_title} ({subset_name})\n"
        f"{param_info}\n"
        f"Silhouette Score: {sil_score:.4f}",
        fontsize=12,
        fontweight='bold'
    )

    plt.savefig(
        os.path.join(m_dir, f"{m_base}_{subset}_umap.pdf"),
        bbox_inches='tight'
    )

    plt.close()

    umap_end_time = time.perf_counter()
    umap_time_sec = umap_end_time - umap_start_time

    timing_records.append({
        "dataset": cell_name,
        "stage": "umap_visualization",
        "subset": subset,
        "time_sec": umap_time_sec,
        "time_min": umap_time_sec / 60,
        "time_hour": umap_time_sec / 3600
    })

    print(f"  -> UMAP visualization time ({subset}): {format_seconds(umap_time_sec)}")

    subset_total_end_time = time.perf_counter()
    subset_total_time_sec = subset_total_end_time - subset_total_start_time

    timing_records.append({
        "dataset": cell_name,
        "stage": "subset_total",
        "subset": subset,
        "time_sec": subset_total_time_sec,
        "time_min": subset_total_time_sec / 60,
        "time_hour": subset_total_time_sec / 3600
    })

    print(f"  -> Total subset processing time ({subset}): {format_seconds(subset_total_time_sec)}")

    if subset == "test":
        print("\n" + "=" * 60)
        print(f"TEST evaluation time for thesis ({cell_name}): {format_seconds(eval_time_sec)}")
        print("This excludes UMAP plotting and visualization time.")
        print("=" * 60 + "\n")


# ---------------------------------------------------------
# Output Summary CSV
# ---------------------------------------------------------
summary_df = pd.DataFrame(results)

# Reorder columns: subset, threshold source, threshold value, metrics, timing
col_order = [
    "set",
    "threshold_data",
    "threshold_method",
    "intersect",
    "rep_rate",
    "cond_rate",
    "mean_performance",
    "sep_index",
    "silhouette",
    "core_eval_time_sec",
    "core_eval_time_min",
    "core_eval_time_hour"
]

summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

# Format for readability: round floats
summary_df_display = summary_df.copy()

for col in [
    "rep_rate",
    "cond_rate",
    "mean_performance",
    "sep_index",
    "silhouette",
    "intersect",
    "core_eval_time_sec",
    "core_eval_time_min",
    "core_eval_time_hour"
]:
    if col in summary_df_display.columns:
        summary_df_display[col] = summary_df_display[col].round(4)

threshold_tag = (
    f"manual_{fixed_threshold:.4f}"
    if args.fixed_threshold is not None
    else f"{args.threshold_data}_{args.threshold_method}"
)

out_csv = os.path.join(
    m_dir,
    f"{m_base}_performance_summary_threshold_{threshold_tag}.csv"
)

summary_df_display.to_csv(out_csv, index=False)

print(f"\nEvaluation Complete. CSV saved: {out_csv}")
print(
    f"  Threshold source: {threshold_source} "
    f"(method={threshold_method_used}, threshold = {fixed_threshold:.4f})"
)


# ---------------------------------------------------------
# Output Timing CSV
# ---------------------------------------------------------
script_end_time = time.perf_counter()
script_total_time_sec = script_end_time - script_start_time

timing_records.append({
    "dataset": cell_name,
    "stage": "script_total",
    "subset": "all",
    "time_sec": script_total_time_sec,
    "time_min": script_total_time_sec / 60,
    "time_hour": script_total_time_sec / 3600
})

timing_df = pd.DataFrame(timing_records)

timing_df_display = timing_df.copy()

for col in ["time_sec", "time_min", "time_hour"]:
    timing_df_display[col] = timing_df_display[col].round(4)

timing_csv = os.path.join(
    m_dir,
    f"{m_base}_timing_summary_threshold_{threshold_tag}.csv"
)

timing_df_display.to_csv(timing_csv, index=False)

print(f"Timing CSV saved: {timing_csv}")

print("\n" + "=" * 60)
print(f"Total script runtime ({cell_name}): {format_seconds(script_total_time_sec)}")
print("=" * 60)