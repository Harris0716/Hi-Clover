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
    """
    使用 Siamese 方式評估 Triplet 模型：計算一對影像間的距離
    """
    distances = []
    labels = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # 修正點：改用索引讀取 (data[0], data[1], data[2])，以增加對 Dataset 修改後的兼容性
            input1 = data[0].to(device)
            input2 = data[1].to(device)
            label = data[2] # 這是 Siamese 的 0/1 標籤 (0: Replicate, 1: Condition)
            
            output1 = model.forward_one(input1)
            output2 = model.forward_one(input2)

            dist = F.pairwise_distance(output1, output2, p=2)
            
            distances.extend(dist.cpu().numpy())
            labels.extend(label.numpy())

    return np.array(distances), np.array(labels)

def calculate_metrics(distances, labels):
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

    # 2. 計算分離指數 (Separation Index)
    overlap = minimum(rep_density, cond_density)
    separation_index = 1 - simpson(overlap, x=bin_centers)

    # 3. 計算正確率
    replicate_correct = np.sum(rep_dist < intersect)
    condition_correct = np.sum(cond_dist >= intersect)
    
    rep_rate = replicate_correct / len(rep_dist)
    cond_rate = condition_correct / len(cond_dist)
    mean_perf = (rep_rate + cond_rate) / 2

    return {
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

# 處理 DataParallel 的 module. 前綴問題
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

with open(args.json_file) as f:
    dataset_config = json.load(f)

# 獲取模型檔案所在的資料夾路徑與基礎檔名
model_dir = os.path.dirname(args.model_infile)
model_base_name = os.path.basename(args.model_infile).split('.ckpt')[0]

results = {}

# 定義字體與繪圖參數
TITLE_SIZE, LABEL_SIZE = 14, 12

# 分別測試 Train/Val (合併) 與 Test
for subset in ["train_val", "test"]:
    print(f"--- Processing {subset} set ---")
    
    # 建立路徑列表
    paths = []
    if subset == "train_val":
        for d in args.data_inputs:
            paths.extend(dataset_config[d]["training"] + dataset_config[d]["validation"])
    else:
        for d in args.data_inputs:
            paths.extend(dataset_config[d]["test"])

    # 建立 Siamese 資料集
    ds = GroupedHiCDataset([
        SiameseHiCDataset(
            [HiCDatasetDec.load(p) for p in paths],
            reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]]
        )
    ])
    
    loader = DataLoader(ds, batch_size=100, sampler=SequentialSampler(ds))
    
    # 執行距離測試
    dist, lbl = test_triplet_by_siamese(model, loader, device)
    
    # 計算指標
    data = calculate_metrics(dist, lbl)
    results[subset] = data

    # ---------------------------------------------------------
    # 繪圖 1: Distance Distribution Histogram
    # ---------------------------------------------------------
    a, b, rng = data["hist_data"]
    plt.figure(figsize=(9, 7))
    plt.hist(dist[lbl == 0], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
    plt.hist(dist[lbl == 1], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
    plt.axvline(data["intersect"], color='k', linestyle='--', label=f'Threshold: {data["intersect"]:.2f}')
    
    full_title = f"Distance Distribution ({subset})\nSI: {data['sep_index']:.4f} | Model: {model_base_name}"
    plt.title(full_title, fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    plt.xlabel("Euclidean Distance", fontsize=LABEL_SIZE)
    plt.ylabel("Probability Density", fontsize=LABEL_SIZE)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    dist_fig_path = os.path.join(model_dir, f"{model_base_name}_{subset}_distribution.pdf")
    plt.savefig(dist_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Distribution Plot: {dist_fig_path}")

    # ---------------------------------------------------------
    # 4 色修正版: 平衡抽樣 (確保涵蓋所有 R1/R2/TAM/NIPBL)
    # ---------------------------------------------------------
    print(f"Generating Balanced 4-color t-SNE for {subset}...")
    from matplotlib.colors import ListedColormap
    
    test_embeddings, detailed_labels = [], []
    model.eval()
    
    # 計算每個檔案應該抽取的樣本數，以總數約 5000 為目標
    samples_per_file = max(1, 5000 // len(paths))
    
    with torch.no_grad():
        for p in paths:
            temp_ds = HiCDatasetDec.load(p)
            # 使用 shuffle=True 確保抽到的是該檔案中隨機的區域
            temp_loader = DataLoader(temp_ds, batch_size=64, shuffle=True)
            
            is_r2 = 1 if ('R2' in p or 'rep2' in p) else 0
            file_count = 0 # 紀錄目前檔案已抽取的數量
            
            for i, batch in enumerate(temp_loader):
                img = batch[0].to(device)
                class_ids = batch[-1].cpu().numpy() 
                
                emb = model.forward_one(img)
                test_embeddings.extend(emb.cpu().numpy())
                
                for cid in class_ids:
                    if cid == 1: # NIPBL
                        final_lbl = 1 if is_r2 == 0 else 2
                    else: # TAM
                        final_lbl = 3 if is_r2 == 0 else 4
                    detailed_labels.append(final_lbl)
                
                file_count += len(class_ids)
                # 如果這個檔案抽夠了，就換下一個檔案
                if file_count >= samples_per_file:
                    break 

    # 確保最終陣列長度對齊
    test_embeddings = np.array(test_embeddings)
    detailed_labels = np.array(detailed_labels)
    
    # 如果樣本太多，進行最後一次全局隨機裁切到 5000
    if len(test_embeddings) > 5000:
        indices = np.random.choice(len(test_embeddings), 5000, replace=False)
        test_embeddings = test_embeddings[indices]
        detailed_labels = detailed_labels[indices]

    print(f"Final plot contains {len(test_embeddings)} points from {len(paths)} files.")

    # --- 繪圖邏輯保持不變 ---
    tsne_res = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(test_embeddings)
    
    custom_colors = ['#1F77B4', '#AEC7E8', '#D62728', '#FF9896'] 
    my_cmap = ListedColormap(custom_colors)
    
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(tsne_res[:, 0], tsne_res[:, 1], 
                          c=detailed_labels, cmap=my_cmap, s=20, alpha=0.7, edgecolors='none')

    legend_labels = ["NIPBL R1", "NIPBL R2", "TAM R1", "TAM R2"]
    # 確保 legend 只顯示圖中實際存在的類別
    unique_ids = np.unique(detailed_labels).astype(int)
    handles, _ = scatter.legend_elements()
    actual_labels = [legend_labels[i-1] for i in unique_ids]
    
    plt.legend(handles=handles, labels=actual_labels, title="Samples", loc='best')
    plt.title(f"Replicates vs Conditions Visibility ({subset})", fontsize=14, fontweight='bold')
    # ... 其餘存檔邏輯 ...

# ---------------------------------------------------------
# 輸出 CSV 統計總表
# ---------------------------------------------------------
summary_path = os.path.join(model_dir, f"{model_base_name}_summary.csv")
summary_df = pd.DataFrame({
    "set": results.keys(),
    "replicate_rate": [r["rep_rate"] for r in results.values()],
    "condition_rate": [r["cond_rate"] for r in results.values()],
    "mean_performance": [r["mean_perf"] for r in results.values()],
    "separation_index": [r["sep_index"] for r in results.values()]
}).round(4)

summary_df.to_csv(summary_path, index=False)
print(f"Saved All Results to: {summary_path}")