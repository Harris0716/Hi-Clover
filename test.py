import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from numpy import minimum
from collections import OrderedDict
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

# 嘗試匯入 umap
try:
    import umap
except ImportError:
    print("Warning: umap-learn not installed. Please run 'pip install umap-learn'")
    umap = None

# 匯入自定義模組
from HiSiNet.HiCDatasetClass import HiCDatasetDec, SiameseHiCDataset, GroupedHiCDataset
import HiSiNet.models as models
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Latent Space Evaluation and Visualization Module')
parser.add_argument('model_name', type=str, help='Model architecture name')
parser.add_argument('json_file', type=str, help='JSON config for data paths')
parser.add_argument('model_infile', type=str, help='Path to the trained .ckpt file')
parser.add_argument('--mask', type=bool, default=True, help='Apply diagonal mask')
parser.add_argument("data_inputs", nargs='+', help="Biological conditions to evaluate (e.g., NIPBL)")
args = parser.parse_args()

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def test_triplet_by_siamese(model, dataloader, device):
    distances, labels = [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input1, input2, label = data[0].to(device), data[1].to(device), data[2]
            output1 = model.forward_one(input1)
            output2 = model.forward_one(input2)
            dist = F.pairwise_distance(output1, output2, p=2)
            distances.extend(dist.cpu().numpy())
            labels.extend(label.numpy())
    return np.array(distances), np.array(labels)

def calculate_metrics(distances, labels, fixed_threshold=None):
    # 使用 99.5 百分位作為繪圖邊界，避免離群值影響視覺效果
    mx, mn = np.percentile(distances, 99.5), distances.min()
    rng = np.linspace(mn, mx, 200)
    rep_dist, cond_dist = distances[labels == 0], distances[labels == 1]
    
    a = np.histogram(rep_dist, bins=rng, density=True)
    b = np.histogram(cond_dist, bins=rng, density=True)
    
    bin_centers = (a[1][1:] + a[1][:-1]) / 2
    
    if fixed_threshold is None:
        # 尋找兩分佈交叉點作為最佳門檻
        diff = a[0] - b[0]
        idx = np.where(np.diff(np.sign(diff)))[0]
        intersect = a[1][idx[0]] if len(idx) > 0 else a[1][len(a[1])//2]
    else:
        intersect = fixed_threshold

    # 計算 Separation Index (SI)
    overlap = minimum(a[0], b[0])
    sep_index = 1 - simpson(overlap, x=bin_centers)
    
    rep_rate = np.sum(rep_dist < intersect) / len(rep_dist)
    cond_rate = np.sum(cond_dist >= intersect) / len(cond_dist)
    
    return {
        "intersect": intersect, 
        "rep_rate": rep_rate, 
        "cond_rate": cond_rate, 
        "mean_perf": (rep_rate + cond_rate) / 2, 
        "sep_index": sep_index, 
        "hist_data": (a, b, rng)
    }

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載模型
model = eval("models." + args.model_name)(mask=args.mask).to(device)
state_dict = torch.load(args.model_infile, map_location=device, weights_only=True)
new_state_dict = OrderedDict([(k.replace("module.", ""), v) for k, v in state_dict.items()])
model.load_state_dict(new_state_dict)

with open(args.json_file) as f:
    dataset_config = json.load(f)

model_dir = os.path.dirname(args.model_infile)
model_base_name = os.path.basename(args.model_infile).split('.ckpt')[0]

# --- STEP 1: 從 Validation Set 獲取固定門檻 ---
print(">>> Step 1: Calibrating Decision Threshold from Validation Data...")
val_paths = []
for d in args.data_inputs:
    val_paths.extend(dataset_config[d]["validation"])

val_ds = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(p) for p in val_paths], 
                            reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]])])
val_loader = DataLoader(val_ds, batch_size=128, sampler=SequentialSampler(val_ds))
val_dist, val_lbl = test_triplet_by_siamese(model, val_loader, device)
fixed_threshold = calculate_metrics(val_dist, val_lbl)["intersect"]
print(f"Optimal Threshold Captured: {fixed_threshold:.4f}")

# --- STEP 2: 評估與視覺化 ---
results = []
for subset in ["train_val", "test"]:
    print(f"\n>>> Step 2: Evaluating {subset.upper()} Set...")
    paths = []
    if subset == "train_val":
        for d in args.data_inputs:
            paths.extend(dataset_config[d]["training"] + dataset_config[d]["validation"])
    else:
        for d in args.data_inputs:
            paths.extend(dataset_config[d]["test"])

    # A. 距離統計
    ds = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(p) for p in paths], 
                            reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]])])
    loader = DataLoader(ds, batch_size=128, sampler=SequentialSampler(ds))
    dist, lbl = test_triplet_by_siamese(model, loader, device)
    data = calculate_metrics(dist, lbl, fixed_threshold=fixed_threshold)
    
    results.append({
        "set": subset,
        "rep_rate": round(data["rep_rate"], 4),
        "cond_rate": round(data["cond_rate"], 4),
        "mean_perf": round(data["mean_perf"], 4),
        "sep_index": round(data["sep_index"], 4)
    })

    # 繪製 Distance Histogram
    plt.figure(figsize=(9, 6))
    plt.hist(dist[lbl == 0], bins=data["hist_data"][2], density=True, label='Technical Replicates', alpha=0.5, color='#108690')
    plt.hist(dist[lbl == 1], bins=data["hist_data"][2], density=True, label='Biological Conditions', alpha=0.5, color='#1D1E4E')
    plt.axvline(fixed_threshold, color='k', linestyle='--', label=f'Threshold ({fixed_threshold:.2f})')
    plt.title(f"Pairwise Distance Distribution - {subset.upper()}\nSeparation Index: {data['sep_index']:.4f}", fontweight='bold')
    plt.xlabel("Euclidean Distance"); plt.ylabel("Probability Density"); plt.legend()
    plt.savefig(os.path.join(model_dir, f"{model_base_name}_{subset}_dist_hist.pdf"), bbox_inches='tight')
    plt.close()

    # B. 潛在空間投射 (t-SNE & UMAP)
    print(f"Generating Projection Plots for {subset}...")
    test_embeddings, detailed_labels = [], []
    samples_per_file = max(1, 5000 // len(paths))
    
    model.eval()
    with torch.no_grad():
        for p in paths:
            temp_ds = HiCDatasetDec.load(p)
            temp_loader = DataLoader(temp_ds, batch_size=64, shuffle=True)
            is_r2 = 1 if any(x in p.upper() for x in ['R2', 'REP2']) else 0
            count = 0
            for batch in temp_loader:
                img, cids = batch[0].to(device), batch[-1].cpu().numpy()
                emb = model.forward_one(img)
                num = min(len(cids), samples_per_file - count)
                if num > 0:
                    test_embeddings.extend(emb[:num].cpu().numpy())
                    for cid in cids[:num]:
                        # 標籤對應：1=NIPBL, 2=TAM
                        if cid == 1: final_lbl = 1 if is_r2 == 0 else 2 # 藍 / 淺藍
                        else: final_lbl = 3 if is_r2 == 0 else 4        # 紅 / 淺紅
                        detailed_labels.append(final_lbl)
                    count += num
                if count >= samples_per_file: break

    test_embeddings, detailed_labels = np.array(test_embeddings), np.array(detailed_labels)
    cmap = ListedColormap(['#1F77B4', '#AEC7E8', '#D62728', '#FF9896'])
    lgd_labels = ["NIPBL R1", "NIPBL R2", "TAM R1", "TAM R2"]

    # t-SNE Plot
    print(f"Calculating t-SNE ({subset})...")
    res_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(test_embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(res_tsne[:, 0], res_tsne[:, 1], c=detailed_labels, cmap=cmap, s=20, alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=lgd_labels, title="Samples")
    plt.title(f"Latent Space Visualization (t-SNE) - {subset.upper()}", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(model_dir, f"{model_base_name}_{subset}_tsne.pdf"), bbox_inches='tight')
    plt.close()

    # UMAP Plot
    if umap:
        print(f"Calculating UMAP ({subset})...")
        res_umap = umap.UMAP(random_state=42).fit_transform(test_embeddings)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(res_umap[:, 0], res_umap[:, 1], c=detailed_labels, cmap=cmap, s=20, alpha=0.7)
        plt.legend(handles=scatter.legend_elements()[0], labels=lgd_labels, title="Samples")
        plt.title(f"Latent Space Visualization (UMAP) - {subset.upper()}", fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(model_dir, f"{model_base_name}_{subset}_umap.pdf"), bbox_inches='tight')
        plt.close()

# ---------------------------------------------------------
# Output Metrics
# ---------------------------------------------------------
summary_df = pd.DataFrame(results)
summary_df.to_csv(os.path.join(model_dir, f"{model_base_name}_performance_summary.csv"), index=False)
print(f"\nEvaluation Complete. All files saved to: {model_dir}")