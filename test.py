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

# 導入你現有的 Dataset 與模型定義
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
    """
    使用 Siamese 方式評估 Triplet 模型：計算一對影像間的距離
    """
    distances = []
    labels = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input1, input2, label = data
            input1, input2 = input1.to(device), input2.to(device)
            
            output1 = model.forward_one(input1)
            output2 = model.forward_one(input2)

            dist = F.pairwise_distance(output1, output2, p=2)
            
            distances.extend(dist.cpu().numpy())
            labels.extend(label.numpy())

    return np.array(distances), np.array(labels)

def calculate_metrics(distances, labels, set_name="test"):
    """
    計算分離指數 (Separation Index) 與平均表現 (Mean Performance)
    """
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

    # 1. 計算交叉點 (Intersect) 作為分類閾值
    diff = rep_density - cond_density
    idx = np.where(np.diff(np.sign(diff)))[0]
    intersect = bin_edges[idx[0]] if len(idx) > 0 else bin_edges[len(bin_edges)//2]

    # 2. 計算分離指數 (Separation Index) - 論文定義 
    overlap = minimum(rep_density, cond_density)
    separation_index = 1 - simpson(overlap, x=bin_centers)

    # 3. 計算正確率 - 論文定義 
    replicate_correct = np.sum(rep_dist < intersect)
    condition_correct = np.sum(cond_dist >= intersect)
    
    rep_rate = replicate_correct / len(rep_dist)
    cond_rate = condition_correct / len(cond_dist)
    mean_perf = (rep_rate + cond_rate) / 2

    return {
        "distances": distances,
        "labels": labels,
        "intersect": intersect,
        "rep_rate": rep_rate,
        "cond_rate": cond_rate,
        "mean_perf": mean_perf,
        "sep_index": separation_index,
        "hist_data": (a, b, rng)
    }

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 載入模型
model = eval("models." + args.model_name)(mask=args.mask).to(device)
state_dict = torch.load(args.model_infile, map_location=device, weights_only=True)

# 處理 DataParallel 的 module. 前綴
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

with open(args.json_file) as f:
    dataset_config = json.load(f)

results = {}

# 分別測試 Train/Val (合併) 與 Test
for subset in ["train_val", "test"]:
    print(f"--- Processing {subset} set ---")
    
    # 建立路徑列表
    if subset == "train_val":
        paths = []
        for d in args.data_inputs:
            paths.extend(dataset_config[d]["training"] + dataset_config[d]["validation"])
    else:
        paths = []
        for d in args.data_inputs:
            paths.extend(dataset_config[d]["test"])

    # 建立 Siamese 資料集
    # 標籤 0 為 replicates，1 為 conditions [cite: 745]
    ds = GroupedHiCDataset([
        SiameseHiCDataset(
            [HiCDatasetDec.load(p) for p in paths],
            reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]]
        )
    ])
    
    loader = DataLoader(ds, batch_size=100, sampler=SequentialSampler(ds))
    dist, lbl = test_triplet_by_siamese(model, loader, device)
    
    results[subset] = calculate_metrics(dist, lbl, subset)

# ---------------------------------------------------------
# 繪圖與存檔
# ---------------------------------------------------------
for subset, data in results.items():
    a, b, rng = data["hist_data"]
    plt.figure(figsize=(8, 6))
    plt.hist(data["distances"][data["labels"] == 0], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
    plt.hist(data["distances"][data["labels"] == 1], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
    plt.axvline(data["intersect"], color='k', linestyle='--', label=f'Threshold: {data["intersect"]:.2f}')
    
    plt.title(f"Distance Distribution ({subset})\nSeparation Index: {data['sep_index']:.4f}")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Probability Density")
    plt.legend()
    
    save_fig = f"{args.model_infile.split('.ckpt')[0]}_{subset}_distribution.pdf"
    plt.savefig(save_fig)
    plt.close()
    print(f"Saved plot: {save_fig}")

# 輸出 CSV
summary_path = f"{args.model_infile.split('.ckpt')[0]}_summary.csv"
summary_df = pd.DataFrame({
    "set": results.keys(),
    "replicate_rate": [r["rep_rate"] for r in results.values()],
    "condition_rate": [r["cond_rate"] for r in results.values()],
    "mean_performance": [r["mean_perf"] for r in results.values()],
    "separation_index": [r["sep_index"] for r in results.values()]
}).round(4)

summary_df.to_csv(summary_path, index=False)
print(f"Saved summary: {summary_path}")