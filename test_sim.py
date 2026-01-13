import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
import argparse, json, os, pandas as pd, matplotlib.pyplot as plt
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
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Latent Space Evaluation and Visualization')
parser.add_argument('model_name', type=str)
parser.add_argument('json_file', type=str)
parser.add_argument('model_infile', type=str)
parser.add_argument('--mask', type=bool, default=True)
parser.add_argument("data_inputs", nargs='+')
args = parser.parse_args()

def test_triplet(model, dataloader, device):
    distances, labels = [], []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            o1, o2 = model.forward_one(data[0].to(device)), model.forward_one(data[1].to(device))
            # 使用 Cosine Distance (1 - Cosine Similarity)
            cos_sim = F.cosine_similarity(o1, o2)
            distances.extend((1 - cos_sim).cpu().numpy())
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = eval("models." + args.model_name)(mask=args.mask).to(device)
sd = torch.load(args.model_infile, map_location=device, weights_only=True)
model.load_state_dict(OrderedDict([(k.replace("module.", ""), v) for k, v in sd.items()]))

with open(args.json_file) as f: dataset_config = json.load(f)
m_dir, m_base = os.path.dirname(args.model_infile), os.path.basename(args.model_infile).split('.ckpt')[0]

cell_name = args.data_inputs[0]
cell_title = cell_name

if "NPC" in cell_name.upper():
    lgd = ["NPC Ctrl R1", "NPC Ctrl R2", "NPC Treat R1", "NPC Treat R2"]
elif "NIPBL" in cell_name.upper():
    lgd = ["NIPBL R1", "NIPBL R2", "TAM R1", "TAM R2"]
else:
    lgd = [f"{cell_name} R1", f"{cell_name} R2", "Treat R1", "Treat R2"]

param_title = m_base.replace('_best', '').replace('_', ' | ')

print(f"Step 1: Calibrating Threshold from Validation ({cell_name})...")
val_paths = [p for d in args.data_inputs for p in dataset_config[d]["validation"]]
v_ds = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(p) for p in val_paths], 
                        reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]])])
v_dist, v_lbl = test_triplet(model, DataLoader(v_ds, batch_size=128), device)
fixed_threshold = calculate_metrics(v_dist, v_lbl)["intersect"]

results = []
for subset in ["train_val", "test"]:
    print(f"\nStep 2: Processing {subset.upper()}...")
    paths = [p for d in args.data_inputs for p in (dataset_config[d]["training"] + dataset_config[d]["validation"] if subset == "train_val" else dataset_config[d]["test"])]
    
    ds = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(p) for p in paths], 
                            reference=reference_genomes[dataset_config[args.data_inputs[0]]["reference"]])])
    dist, lbl = test_triplet(model, DataLoader(ds, batch_size=128), device)
    data = calculate_metrics(dist, lbl, fixed_threshold=fixed_threshold)

    # 抽樣計算視覺化與輪廓係數
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
                        detailed_lbls.append((1 if is_r2 == 0 else 2) if cid == 1 else (3 if is_r2 == 0 else 4))
                    count += num
                if count >= samples_per_file: break

    embs, detailed_lbls = np.array(embs), np.array(detailed_lbls)
    
    # 修改處：將原本 4 類改為 2 類 (Ctrl=0, Treat=1) 計算輪廓係數
    binary_lbls = [0 if (l == 1 or l == 2) else 1 for l in detailed_lbls]
    sil_score = silhouette_score(embs, binary_lbls, metric='cosine')
    
    mean_perf = (data["rep_rate"] + data["cond_rate"]) / 2 
    
    results.append({
        "set": subset, 
        "rep_rate": data["rep_rate"], 
        "cond_rate": data["cond_rate"], 
        "mean_performance": mean_perf,
        "sep_index": data["sep_index"],
        "silhouette": sil_score
    })

    # 1. Histogram
    plt.figure(figsize=(9, 6))
    plt.hist(dist[lbl == 0], bins=data["hist_data"][2], density=True, label='Replicates', alpha=0.5, color='#108690')
    plt.hist(dist[lbl == 1], bins=data["hist_data"][2], density=True, label='Conditions', alpha=0.5, color='#1D1E4E')
    plt.axvline(fixed_threshold, color='k', ls='--', label=f'Threshold ({fixed_threshold:.2f})')
    plt.title(f"Distance Distribution ({subset}) | {cell_title}\n{param_title}\nSI: {data['sep_index']:.4f} | Mean Perf: {mean_perf:.4f}", fontweight='bold')
    plt.xlabel("$1 - Cosine Similarity$"); plt.ylabel("Probability Density"); plt.legend()
    plt.savefig(os.path.join(m_dir, f"{m_base}_{subset}_dist_hist.pdf"), bbox_inches='tight'); plt.close()

    cmap = ListedColormap(['#1F77B4', '#AEC7E8', '#D62728', '#FF9896'])

    # 2. t-SNE
    print(f"Calculating t-SNE for {subset}...")
    res_tsne = TSNE(n_components=2, perplexity=40, random_state=42, early_exaggeration=20, metric='cosine').fit_transform(embs)
    plt.figure(figsize=(10, 8)); scat = plt.scatter(res_tsne[:,0], res_tsne[:,1], c=detailed_lbls, cmap=cmap, s=10, alpha=0.5)
    plt.legend(handles=scat.legend_elements()[0], labels=lgd, title="Samples")
    plt.title(f"Latent Space Visualization (t-SNE) - {subset.upper()} | {cell_title}\n{param_title}\nSilhouette: {sil_score:.4f}", fontweight='bold')
    plt.savefig(os.path.join(m_dir, f"{m_base}_{subset}_tsne.pdf"), bbox_inches='tight'); plt.close()

    # 3. UMAP
    print(f"Calculating UMAP for {subset}...")
    res_umap = umap.UMAP(random_state=42, n_neighbors=80, min_dist=0.1, metric='cosine').fit_transform(embs)
    plt.figure(figsize=(10, 8)); scat = plt.scatter(res_umap[:,0], res_umap[:,1], c=detailed_lbls, cmap=cmap, s=10, alpha=0.5)
    plt.legend(handles=scat.legend_elements()[0], labels=lgd, title="Samples")
    plt.title(f"Latent Space Visualization (UMAP) - {subset.upper()} | {cell_title}\n{param_title}\nSilhouette: {sil_score:.4f}", fontweight='bold')
    plt.savefig(os.path.join(m_dir, f"{m_base}_{subset}_umap.pdf"), bbox_inches='tight'); plt.close()

# ---------------------------------------------------------
# Output Summary 
# ---------------------------------------------------------
summary_df = pd.DataFrame(results)
summary_df.to_csv(os.path.join(m_dir, f"{m_base}_performance_summary.csv"), index=False, float_format='%.4f')
print(f"Evaluation Complete. CSV saved for {cell_name}.")