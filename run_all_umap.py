import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import json
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
import umap

from HiSiNet.HiCDatasetClass import HiCDatasetDec
import HiSiNet.models as models

def get_embeddings_and_labels(model, dataset_paths, device, samples_limit=5000):
    """提取樣本的特徵向量與對應標籤，供 UMAP 與 Silhouette 評估使用"""
    embs, detailed_lbls = [], []
    if not dataset_paths:
        return np.array(embs), np.array(detailed_lbls)
        
    samples_per_file = max(1, samples_limit // len(dataset_paths))
    model.eval()
    
    with torch.no_grad():
        for p in dataset_paths:
            temp_ds = HiCDatasetDec.load(p)
            ldr = DataLoader(temp_ds, batch_size=64, shuffle=True)
            # 依據檔名判斷是否為 Replicate 2
            is_r2 = 1 if any(x in p.upper() for x in ['R2', 'REP2']) else 0
            count = 0
            
            for batch in ldr:
                eb = model.forward_one(batch[0].to(device))
                num = min(len(batch[-1]), samples_per_file - count)
                if num > 0:
                    embs.extend(eb[:num].cpu().numpy())
                    for cid in batch[-1][:num].numpy():
                        # 定義標籤: 1: Ctrl R1, 2: Ctrl R2, 3: Treat R1, 4: Treat R2
                        detailed_lbls.append((1 if is_r2 == 0 else 2) if cid == 1 else (3 if is_r2 == 0 else 4))
                    count += num
                if count >= samples_per_file: 
                    break
                    
    return np.array(embs), np.array(detailed_lbls)

def main():
    parser = argparse.ArgumentParser(description='Evaluate 3 checkpoints and plot 2x3 UMAP latent space distribution.')
    parser.add_argument('--model_name', type=str, default='TripletLeNetBatchNorm')
    parser.add_argument('--json_file', type=str, required=True, help='Path to config.json')
    parser.add_argument('--ckpt_liver', type=str, required=True, help='Path to Liver .ckpt')
    parser.add_argument('--ckpt_npc', type=str, required=True, help='Path to NPC .ckpt')
    parser.add_argument('--ckpt_tcell', type=str, required=True, help='Path to TCell .ckpt')
    parser.add_argument('--mask', type=bool, default=True)
    parser.add_argument('--out_pdf', type=str, default='combined_umap_projection.pdf')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.json_file) as f: 
        dataset_config = json.load(f)

    cell_configs = [
        {
            'name': 'Liver', 'display': 'Liver', 'ckpt': args.ckpt_liver,
            'legend': ["Liver NIPBL R1", "Liver NIPBL R2", "Liver TAM R1", "Liver TAM R2"]
        },
        {
            'name': 'NPC', 'display': 'NPC', 'ckpt': args.ckpt_npc,
            'legend': ["NPC Ctrl R1", "NPC Ctrl R2", "NPC Treat (Aux) R1", "NPC Treat (Aux) R2"]
        },
        {
            'name': 'TCell', 'display': 'T Cell', 'ckpt': args.ckpt_tcell,
            'legend': ["TCells Ctrl (DP) R1", "TCells Ctrl (DP) R2", "TCells Treat (SP) R1", "TCells Treat (SP) R2"]
        }
    ]

    plot_data = {}

    for cell in cell_configs:
        cell_name = cell['name']
        print(f"Processing {cell_name}...")
        
        # 1. 載入模型與權重
        model = eval("models." + args.model_name)(mask=args.mask).to(device)
        sd = torch.load(cell['ckpt'], map_location=device, weights_only=True)
        model.load_state_dict(OrderedDict([(k.replace("module.", ""), v) for k, v in sd.items()]), strict=False)
        
        plot_data[cell_name] = {}

        for phase in ['train_val', 'test']:
            print(f"  -> Extracting embeddings for {phase}...")
            if phase == 'train_val':
                paths = dataset_config[cell_name]["training"] + dataset_config[cell_name]["validation"]
            else:
                paths = dataset_config[cell_name]["test"]

            embs, detailed_lbls = get_embeddings_and_labels(model, paths, device)
            
            # 計算輪廓係數 (Silhouette Score)，將標籤簡化為 Control (0) 與 Condition (1)
            sil_score = 0.0
            if len(embs) > 1:
                binary_lbls = [0 if (l == 1 or l == 2) else 1 for l in detailed_lbls]
                sil_score = silhouette_score(embs, binary_lbls, metric='euclidean')

            print(f"  -> Computing UMAP for {phase}...")
            # 執行 UMAP 降維
            res_umap = umap.UMAP(random_state=42, n_neighbors=80, min_dist=0.1, metric='euclidean').fit_transform(embs)
            
            plot_data[cell_name][phase] = {
                'umap': res_umap,
                'labels': detailed_lbls,
                'sil_score': sil_score
            }

    # ==========================================
    # 開始繪製 2x3 UMAP 圖表
    # ==========================================
    print("Generating combined UMAP plot...")
    phases = ['train_val', 'test']
    phase_labels = ['Train + Val', 'Test']
    cmap = ListedColormap(['#1F77B4', '#AEC7E8', '#D62728', '#FF9896'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=False, sharey=False)

    for i, phase in enumerate(phases):
        for j, cell in enumerate(cell_configs):
            ax = axes[i, j]
            c_name = cell['name']
            
            data = plot_data[c_name][phase]
            res_umap = data['umap']
            lbls = data['labels']
            sil_score = data['sil_score']
            
            # 繪製散佈圖
            scat = ax.scatter(res_umap[:, 0], res_umap[:, 1], c=lbls, cmap=cmap, s=5, alpha=0.6)
            
            # 子圖標題與輪廓係數
            if i == 0:
                # 獨立繪製粗體的細胞名稱 (位於子圖最上方)
                ax.text(0.5, 1.12, cell['display'], transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
                # 設定非粗體的輪廓係數
                ax.set_title(f"Silhouette Score: {sil_score:.4f}", 
                             fontsize=12, fontweight='normal', pad=8)
            else:
                # 第二列僅顯示非粗體的輪廓係數
                ax.set_title(f"Silhouette Score: {sil_score:.4f}", 
                             fontsize=12, fontweight='normal', pad=8)

            # 隱藏座標軸刻度以保持版面整潔
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 為每個子圖加入對應的圖例
            legend = ax.legend(handles=scat.legend_elements()[0], labels=cell['legend'], 
                               loc='best', fontsize=8, title="Sample ID")
            legend.get_title().set_fontsize('9')

    fig.supxlabel("UMAP Dimension 1", fontsize=16, y=0.02)
    fig.supylabel("UMAP Dimension 2", fontsize=16, x=0.02)

    fig.text(0.04, 0.72, phase_labels[0], va='center', rotation='vertical', fontsize=18, fontweight='bold')
    fig.text(0.04, 0.28, phase_labels[1], va='center', rotation='vertical', fontsize=18, fontweight='bold')

    plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.08, wspace=0.15, hspace=0.15)
    
    plt.savefig(args.out_pdf, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as {args.out_pdf}")

if __name__ == '__main__':
    main()