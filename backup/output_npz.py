import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
import argparse, json, os, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import simpson
from numpy import minimum
from collections import OrderedDict

# 移除未使用之視覺化與聚類評估套件
# from sklearn.manifold import TSNE
# from sklearn.metrics import silhouette_score
# from matplotlib.colors import ListedColormap
# import umap

from HiSiNet.HiCDatasetClass import HiCDatasetDec, SiameseHiCDataset, GroupedHiCDataset
import HiSiNet.models as models
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Latent Space Evaluation - Distance Distribution Only')
parser.add_argument('model_name', type=str)
parser.add_argument('json_file', type=str)
parser.add_argument('model_infile', type=str)
parser.add_argument('--mask', type=bool, default=True)
parser.add_argument('--threshold_data', type=str, default='train_val', choices=['val', 'train_val'],
                    help='Data used to calibrate decision threshold (intersect). train_val=train+val (default, stabler); val=validation only.')    
parser.add_argument("data_inputs", nargs='+')
args = parser.parse_args()

def test_triplet(model, dataloader, device):
    distances, labels = [], []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            o1, o2 = model.forward_one(data[0].to(device)), model.forward_one(data[1].to(device))
            # keep Euclidean distance
            distances.extend(F.pairwise_distance(o1, o2).cpu().numpy())
            labels.extend(data[2].numpy())
    return np.array(distances), np.array(labels)

def calculate_metrics(distances, labels, fixed_threshold=None):
    rng = np.linspace(distances.min(), np.percentile(distances, 99.5), 200)
    rep_dist, cond_dist = distances[labels == 0], distances[labels == 1]
    a, b = np.histogram(rep_dist, bins=rng, density=True), np.histogram(cond_dist, bins=rng, density=True)
    if fixed_threshold is None:
        idx = np.where(np.diff(np.sign(a[0] - b[0])))[0]
        intersect = a[1][idx[0]] if len(idx) > 0 else a[1][len(a[1])//2]
    else: intersect = fixed_threshold
    overlap = minimum(a[0], b[0])
    sep_idx = 1 - simpson(overlap, x=(a[1][1:] + a[1][:-1]) / 2)
    return {"intersect": intersect, "rep_rate": np.sum(rep_dist < intersect) / len(rep_dist),
            "cond_rate": np.sum(cond_dist >= intersect) / len(cond_dist),
            "sep_index": sep_idx, "hist_data": (a, b, rng)}

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = eval("models." + args.model_name)(mask=args.mask).to(device)
sd = torch.load(args.model_infile, map_location=device, weights_only=True)
model.load_state_dict(OrderedDict([(k.replace("module.", ""), v) for k, v in sd.items()]))

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
with open(json_path) as f: dataset_config = json.load(f)
m_dir, m_base = os.path.dirname(args.model_infile), os.path.basename(args.model_infile).split('.ckpt')[0]

cell_name = args.data_inputs[0]
cell_title = cell_name

# parse the file name parameters
parts = m_base.replace('_best', '').split('_')
if len(parts) >= 5:
    param_info = f"Model: {parts[0]} | LR: {parts[1]} | Batch: {parts[2]} | Seed: {parts[3]} | Margin: {parts[4]}"
else:
    param_info = m_base.replace('_best', '').replace('_', ' | ')

# Step 1: Calibrate threshold (intersect)
threshold_source = "Validation" if args.threshold_data == "val" else "Train+Val"
print(f"Step 1: Calibrating Threshold from {threshold_source} ({cell_name})...")
if args.threshold_data == "val":
    cal_paths = [p for d in args.data_inputs for p in dataset_config[d]["validation"]]
else:
    cal_paths = [p for d in args.data_inputs for p in (dataset_config[d]["training"] + dataset_config[d]["validation"])]
cal_ds = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(p) for p in cal_paths], 
                        reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]])])
cal_dist, cal_lbl = test_triplet(model, DataLoader(cal_ds, batch_size=128), device)
fixed_threshold = calculate_metrics(cal_dist, cal_lbl)["intersect"]
print(f"  -> Intersect (threshold) = {fixed_threshold:.4f} (from {threshold_source})")

results = []
for subset in ["train_val", "test"]:
    subset_name = "TRAIN/VAL" if subset == "train_val" else "TEST"
    print(f"\nStep 2: Processing {subset.upper()}...")
    
    paths = [p for d in args.data_inputs for p in (dataset_config[d]["training"] + dataset_config[d]["validation"] if subset == "train_val" else dataset_config[d]["test"])]
    
    ds = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(p) for p in paths], 
                            reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]])])
    dist, lbl = test_triplet(model, DataLoader(ds, batch_size=128), device)
    data = calculate_metrics(dist, lbl, fixed_threshold=fixed_threshold)

    # 註解區塊：已移除耗時的 embeddings 萃取與 silhouette_score 計算

    mean_perf = (data["rep_rate"] + data["cond_rate"]) / 2
    
    results.append({
        "threshold_data": args.threshold_data,
        "intersect": fixed_threshold,
        "set": subset, 
        "rep_rate": data["rep_rate"], 
        "cond_rate": data["cond_rate"], 
        "mean_performance": mean_perf,
        "sep_index": data["sep_index"]
    })

    # --- Histogram ---
    plt.figure(figsize=(9, 6))
    plt.hist(dist[lbl == 0], bins=data["hist_data"][2], density=True, label='Replicates', alpha=0.5, color='#108690')
    plt.hist(dist[lbl == 1], bins=data["hist_data"][2], density=True, label='Conditions', alpha=0.5, color='#1D1E4E')
    plt.axvline(fixed_threshold, color='k', ls='--', label=f'Threshold ({fixed_threshold:.2f})')
    
    plt.title(f"Distance Distribution: {cell_title} ({subset_name})\n"
              f"{param_info}\n"
              f"Separation Index: {data['sep_index']:.4f} | Mean Performance: {mean_perf:.4f}", 
              fontsize=12, fontweight='bold')
    
    plt.xlabel("Euclidean Distance"); plt.ylabel("Probability Density"); plt.legend()
    plt.savefig(os.path.join(m_dir, f"{m_base}_{subset}_dist_hist.pdf"), bbox_inches='tight')
    plt.close()
    
    # 儲存供矩陣圖使用的 raw data
    np.savez_compressed(
        os.path.join(m_dir, f"{cell_name}_{subset}_raw_dist.npz"),
        dist=dist,
        lbl=lbl,
        threshold=fixed_threshold
    )

# ---------------------------------------------------------
# Output Summary (CSV)
# ---------------------------------------------------------
summary_df = pd.DataFrame(results)

# 重新排序與格式化 (不包含 silhouette)
col_order = ["set", "threshold_data", "intersect", "rep_rate", "cond_rate", "mean_performance", "sep_index"]
summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

summary_df_display = summary_df.copy()
for col in ["rep_rate", "cond_rate", "mean_performance", "sep_index", "intersect"]:
    if col in summary_df_display.columns:
        summary_df_display[col] = summary_df_display[col].round(4)

out_csv = os.path.join(m_dir, f"{m_base}_performance_summary_threshold_{args.threshold_data}.csv")
summary_df_display.to_csv(out_csv, index=False)
print(f"\nEvaluation Complete. CSV saved: {out_csv}")
print(f"  Threshold calibrated from: {threshold_source} (intersect = {fixed_threshold:.4f})")