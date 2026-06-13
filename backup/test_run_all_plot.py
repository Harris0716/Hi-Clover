import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import json
import os
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from numpy import minimum
from collections import OrderedDict

from HiSiNet.HiCDatasetClass import HiCDatasetDec, SiameseHiCDataset, GroupedHiCDataset
import HiSiNet.models as models
from HiSiNet.reference_dictionaries import reference_genomes

def test_triplet(model, dataloader, device):
    distances, labels = [], []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            o1, o2 = model.forward_one(data[0].to(device)), model.forward_one(data[1].to(device))
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
    else: 
        intersect = fixed_threshold
    return intersect, rng

def main():
    parser = argparse.ArgumentParser(description='Evaluate 3 checkpoints and plot 2x3 latent space distribution.')
    parser.add_argument('--model_name', type=str, default='TripletLeNetBatchNorm')
    parser.add_argument('--json_file', type=str, required=True, help='Path to config.json')
    parser.add_argument('--ckpt_liver', type=str, required=True, help='Path to Liver .ckpt')
    parser.add_argument('--ckpt_npc', type=str, required=True, help='Path to NPC .ckpt')
    parser.add_argument('--ckpt_tcell', type=str, required=True, help='Path to TCell .ckpt')
    parser.add_argument('--mask', type=bool, default=True)
    parser.add_argument('--out_pdf', type=str, default='combined_latent_space_distribution.pdf')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 讀取 JSON 設定檔
    with open(args.json_file) as f: 
        dataset_config = json.load(f)

    cell_configs = [
        {'name': 'Liver', 'display': 'Liver', 'ckpt': args.ckpt_liver},
        {'name': 'NPC', 'display': 'NPC', 'ckpt': args.ckpt_npc},
        {'name': 'TCell', 'display': 'T Cell', 'ckpt': args.ckpt_tcell}
    ]

    # 儲存繪圖所需數據
    plot_data = {}

    for cell in cell_configs:
        cell_name = cell['name']
        print(f"Processing {cell_name}...")
        
        # 1. 載入模型與對應權重
        model = eval("models." + args.model_name)(mask=args.mask).to(device)
        sd = torch.load(cell['ckpt'], map_location=device, weights_only=True)
        model.load_state_dict(OrderedDict([(k.replace("module.", ""), v) for k, v in sd.items()]), strict=False)
        
        plot_data[cell_name] = {}
        ref_genome = reference_genomes[dataset_config[cell_name]["reference"]]

        # 2. 評估 Train + Val (計算閾值)
        print(f"  -> Evaluating Train+Val...")
        tv_paths = dataset_config[cell_name]["training"] + dataset_config[cell_name]["validation"]
        tv_ds = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(p) for p in tv_paths], reference=ref_genome)])
        tv_dist, tv_lbl = test_triplet(model, DataLoader(tv_ds, batch_size=128), device)
        
        threshold, tv_rng = calculate_metrics(tv_dist, tv_lbl, fixed_threshold=None)
        plot_data[cell_name]['train_val'] = {'dist': tv_dist, 'lbl': tv_lbl, 'threshold': threshold, 'rng': tv_rng}

        # 3. 評估 Test (套用已計算之閾值)
        print(f"  -> Evaluating Test...")
        test_paths = dataset_config[cell_name]["test"]
        test_ds = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(p) for p in test_paths], reference=ref_genome)])
        test_dist, test_lbl = test_triplet(model, DataLoader(test_ds, batch_size=128), device)
        
        _, test_rng = calculate_metrics(test_dist, test_lbl, fixed_threshold=threshold)
        plot_data[cell_name]['test'] = {'dist': test_dist, 'lbl': test_lbl, 'threshold': threshold, 'rng': test_rng}

    # ==========================================
    # 開始繪製 2x3 圖表
    # ==========================================
    print("Generating combined plot...")
    phases = ['train_val', 'test']
    phase_labels = ['Train + Val', 'Test']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey='col')

    for i, phase in enumerate(phases):
        for j, cell in enumerate(cell_configs):
            ax = axes[i, j]
            c_name = cell['name']
            
            data = plot_data[c_name][phase]
            dist = data['dist']
            lbl = data['lbl']
            threshold = data['threshold']
            rng = data['rng']
            
            ax.hist(dist[lbl == 0], bins=rng, density=True, alpha=0.5, color='#108690', label='Replicates')
            ax.hist(dist[lbl == 1], bins=rng, density=True, alpha=0.5, color='#1D1E4E', label='Conditions')
            ax.axvline(threshold, color='k', ls='--', linewidth=1.5, label='Decision Threshold')

            if i == 0:
                ax.set_title(cell['display'], fontsize=18, fontweight='bold', pad=15)

    # 全域排版設定
    fig.supxlabel("Euclidean Distance", fontsize=18, y=0.02)
    fig.supylabel("Probability Density", fontsize=18, x=0.02)

    fig.text(0.04, 0.72, phase_labels[0], va='center', rotation='vertical', fontsize=18, fontweight='bold')
    fig.text(0.04, 0.28, phase_labels[1], va='center', rotation='vertical', fontsize=18, fontweight='bold')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=15, frameon=False)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.15, hspace=0.08)
    
    plt.savefig(args.out_pdf, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as {args.out_pdf}")

if __name__ == '__main__':
    main()