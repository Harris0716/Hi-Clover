import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
import matplotlib.pyplot as plt
import argparse
import json
import os
import pandas as pd
from scipy.integrate import simpson
from numpy import minimum
from collections import OrderedDict
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

# 導入 Dataset 與模型定義
from HiSiNet.HiCDatasetClass import HiCDatasetDec, SiameseHiCDataset, GroupedHiCDataset
import HiSiNet.models as models
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Triplet network testing module')
parser.add_argument('model_name', type=str, help='Model name from models.py')
parser.add_argument('json_file', type=str, help='JSON file containing data paths')
parser.add_argument('model_infile', type=str, help='Path to the trained .ckpt file')
parser.add_argument('--mask', type=bool, default=True, help='Mask diagonal (default: True)')
parser.add_argument("data_inputs", nargs='+', help="Keys for testing datasets")

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

def calculate_metrics(distances, labels):
    mx = np.percentile(distances, 99.5) 
    mn = distances.min()
    rng = np.linspace(mn, mx, 200)
    rep_dist = distances[labels == 0]
    cond_dist = distances[labels == 1]
    a = np.histogram(rep_dist, bins=rng, density=True)
    b = np.histogram(cond_dist, bins=rng, density=True)
    rep_density, cond_density = a[0], b[0]
    bin_edges = a[1]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    diff = rep_density - cond_density
    idx = np.where(np.diff(np.sign(diff)))[0]
    intersect = bin_edges[idx[0]] if len(idx) > 0 else bin_edges[len(bin_edges)//2]
    overlap = minimum(rep_density, cond_density)
    separation_index = 1 - simpson(overlap, x=bin_centers)
    rep_rate = np.sum(rep_dist < intersect) / len(rep_dist)
    cond_rate = np.sum(cond_dist >= intersect) / len(cond_dist)
    return {
        "intersect": intersect, 
        "rep_rate": rep_rate, 
        "cond_rate": cond_rate, 
        "mean_perf": (rep_rate + cond_rate) / 2, 
        "sep_index": separation_index, 
        "hist_data": (a, b, rng)
    }

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = eval("models." + args.model_name)(mask=args.mask).to(device)
state_dict = torch.load(args.model_infile, map_location=device, weights_only=True)
new_state_dict = OrderedDict([(k.replace("module.", ""), v) for k, v in state_dict.items()])
model.load_state_dict(new_state_dict)

with open(args.json_file) as f:
    dataset_config = json.load(f)

model_dir = os.path.dirname(args.model_infile)
model_base_name = os.path.basename(args.model_infile).split('.ckpt')[0]
results = []

for subset in ["train_val", "test"]:
    print(f"--- Processing {subset} set ---")
    paths = []
    if subset == "train_val":
        for d in args.data_inputs:
            paths.extend(dataset_config[d]["training"] + dataset_config[d]["validation"])
    else:
        for d in args.data_inputs:
            paths.extend(dataset_config[d]["test"])

    # 1. 距離分佈測試 (Siamese Mode)
    ds = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(p) for p in paths], 
                            reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]])])
    loader = DataLoader(ds, batch_size=100, sampler=SequentialSampler(ds))
    dist, lbl = test_triplet_by_siamese(model, loader, device)
    data = calculate_metrics(dist, lbl)
    
    # 將數據加入結果列表 (移除 intersect 欄位)
    results.append({
        "set": subset,
        "rep_rate": round(data["rep_rate"], 4),
        "cond_rate": round(data["cond_rate"], 4),
        "mean_perf": round(data["mean_perf"], 4),
        "sep_index": round(data["sep_index"], 4)
    })

    # 繪製直方圖：保留 Threshold 標註
    plt.figure(figsize=(9, 7))
    plt.hist(dist[lbl == 0], bins=data["hist_data"][2], density=True, label='replicates', alpha=0.5, color='#108690')
    plt.hist(dist[lbl == 1], bins=data["hist_data"][2], density=True, label='conditions', alpha=0.5, color='#1D1E4E')
    plt.axvline(data["intersect"], color='k', linestyle='--')
    
    # 在圖上標註 Threshold 數值
    plt.text(data["intersect"]*1.05, plt.gca().get_ylim()[1]*0.9, f'Threshold: {data["intersect"]:.2f}', 
             fontweight='bold', fontsize=10)
    
    plt.title(f"Distance Distribution ({subset})\nSI: {data['sep_index']:.4f}", fontweight='bold')
    plt.xlabel("Euclidean Distance"); plt.ylabel("Probability Density")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(model_dir, f"{model_base_name}_{subset}_distribution.pdf"), bbox_inches='tight')
    plt.close()

    # 2. 平衡 4 色 t-SNE 視覺化
    print(f"Generating Balanced 4-color t-SNE for {subset}...")
    test_embeddings, detailed_labels = [], []
    samples_per_file = max(1, 5000 // len(paths))
    
    model.eval()
    with torch.no_grad():
        for p in paths:
            temp_ds = HiCDatasetDec.load(p)
            temp_loader = DataLoader(temp_ds, batch_size=64, shuffle=True)
            is_r2 = 1 if ('R2' in p or 'rep2' in p) else 0
            file_count = 0
            for i, batch in enumerate(temp_loader):
                img, class_ids = batch[0].to(device), batch[-1].cpu().numpy()
                emb = model.forward_one(img) # 提取 Anchor Embedding
                test_embeddings.extend(emb.cpu().numpy())
                for cid in class_ids:
                    if cid == 1: final_lbl = 1 if is_r2 == 0 else 2
                    else: final_lbl = 3 if is_r2 == 0 else 4
                    detailed_labels.append(final_lbl)
                file_count += len(class_ids)
                if file_count >= samples_per_file: break

    test_embeddings = np.array(test_embeddings)
    detailed_labels = np.array(detailed_labels)
    if len(test_embeddings) > 5000:
        idx = np.random.choice(len(test_embeddings), 5000, replace=False)
        test_embeddings, detailed_labels = test_embeddings[idx], detailed_labels[idx]

    tsne_res = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(test_embeddings)
    
    plt.figure(figsize=(12, 9))
    custom_colors = ['#1F77B4', '#AEC7E8', '#D62728', '#FF9896']
    my_cmap = ListedColormap(custom_colors)
    scatter = plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=detailed_labels, cmap=my_cmap, s=20, alpha=0.7)
    
    unique_ids = np.unique(detailed_labels).astype(int)
    legend_map = {1: "NIPBL R1", 2: "NIPBL R2", 3: "TAM R1", 4: "TAM R2"}
    handles, _ = scatter.legend_elements()
    plt.legend(handles=handles, labels=[legend_map[i] for i in unique_ids], title="Samples", loc='best')
    plt.title(f"Replicates vs Conditions Visibility ({subset})\n{model_base_name}", fontsize=14, fontweight='bold')
    
    tsne_fig_path = os.path.join(model_dir, f"{model_base_name}_{subset}_4color_tsne.pdf")
    plt.savefig(tsne_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved t-SNE Plot: {tsne_fig_path}")

# ---------------------------------------------------------
# 輸出 CSV 
# ---------------------------------------------------------
summary_df = pd.DataFrame(results)
summary_df = summary_df[["set", "rep_rate", "cond_rate", "mean_perf", "sep_index"]]
summary_df.to_csv(os.path.join(model_dir, f"{model_base_name}_summary.csv"), index=False)
print(f"All process completed. Summary saved to CSV without 'intersect'.")