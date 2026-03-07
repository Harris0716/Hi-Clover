import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = "0223_final/npz" 

# 為了確保標題大小寫正確，將檔名與顯示名稱分開設定
file_prefixes = ['liver', 'NPC', 'TCell']  
display_names = ['Liver', 'NPC', 'T Cell']
phases = ['train_val', 'test']
phase_labels = ['Train + Val', 'Test']

# 【關鍵修正 1】：修改長寬比為 (15, 9) 讓子圖更接近黃金比例
# 【關鍵修正 2】：將 sharey=True 改為 sharey='col' (按欄共用 Y 軸)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey='col')

for i, phase in enumerate(phases):
    for j, cell in enumerate(file_prefixes):
        ax = axes[i, j]
        file_path = os.path.join(data_dir, f"{cell}_{phase}_raw_dist.npz")
        
        if os.path.exists(file_path):
            data = np.load(file_path)
            dist = data['dist']
            lbl = data['lbl']
            threshold = data['threshold']
            
            rng = np.linspace(dist.min(), np.percentile(dist, 99.5), 200)
            
            ax.hist(dist[lbl == 0], bins=rng, density=True, alpha=0.5, color='#108690', label='Replicates')
            ax.hist(dist[lbl == 1], bins=rng, density=True, alpha=0.5, color='#1D1E4E', label='Conditions')
            ax.axvline(threshold, color='k', ls='--', linewidth=1.5, label='Threshold')
        else:
            ax.text(0.5, 0.5, f"Missing:\n{cell}_{phase}", ha='center', va='center')

        # 設定第一列的標題
        if i == 0:
            ax.set_title(display_names[j], fontsize=18, fontweight='bold', pad=15)

# 全域座標軸標籤
fig.supxlabel("Euclidean Distance", fontsize=18, y=0.02)
fig.supylabel("Probability Density", fontsize=18, x=0.02)

# 列標示 (Train/Test)
fig.text(0.04, 0.72, phase_labels[0], va='center', rotation='vertical', fontsize=18, fontweight='bold')
fig.text(0.04, 0.28, phase_labels[1], va='center', rotation='vertical', fontsize=18, fontweight='bold')

# 全域圖例
handles, labels = axes[0, 0].get_legend_handles_labels()
# 將文字稍微修飾得更具論文專業感
labels = ['Replicates', 'Conditions', 'Decision Threshold']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=15, frameon=False)

# 【關鍵修正 3】：微調子圖間距
# 因為現在每欄都有獨立的 Y 軸，wspace (左右間距) 必須適度加大至 0.15，避免刻度文字與左側圖片重疊
plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.15, hspace=0.08)

plt.savefig('combined_latent_space_distribution_proportional.pdf', dpi=300, bbox_inches='tight')
plt.show()