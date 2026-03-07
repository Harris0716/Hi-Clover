import numpy as np
import matplotlib.pyplot as plt
import os

# 設定您的檔案所在目錄 (請替換為實際的 m_dir 路徑)
data_dir = "0223_final/npz" 

# 定義矩陣結構 (需與您存檔的 cell_name 一致)
datasets = ['Liver', 'NPC', 'TCell']  
phases = ['train_val', 'test']
phase_labels = ['Train / Val', 'Test']

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

for i, phase in enumerate(phases):
    for j, cell in enumerate(datasets):
        ax = axes[i, j]
        file_path = os.path.join(data_dir, f"{cell}_{phase}_raw_dist.npz")
        
        if os.path.exists(file_path):
            # 讀取數據
            data = np.load(file_path)
            dist = data['dist']
            lbl = data['lbl']
            threshold = data['threshold']
            
            # 建立與原程式相同的 bin 範圍
            rng = np.linspace(dist.min(), np.percentile(dist, 99.5), 200)
            
            # 繪製 Histogram (保持原程式的顏色與透明度設定)
            ax.hist(dist[lbl == 0], bins=rng, density=True, alpha=0.5, color='#108690', label='Replicates')
            ax.hist(dist[lbl == 1], bins=rng, density=True, alpha=0.5, color='#1D1E4E', label='Conditions')
            ax.axvline(threshold, color='k', ls='--', linewidth=1.5, label='Threshold')
        else:
            ax.text(0.5, 0.5, f"Missing:\n{cell}_{phase}", ha='center', va='center')

        # 設定第一列的標題
        if i == 0:
            ax.set_title(cell, fontsize=16, fontweight='bold', pad=12)

# 全域座標軸標籤
fig.supxlabel("Euclidean Distance", fontsize=16, y=0.04)
fig.supylabel("Probability Density", fontsize=16, x=0.04)

# 列標示 (Train/Test)
fig.text(0.06, 0.75, phase_labels[0], va='center', rotation='vertical', fontsize=16, fontweight='bold')
fig.text(0.06, 0.25, phase_labels[1], va='center', rotation='vertical', fontsize=16, fontweight='bold')

# 全域單一圖例 (擷取第一張圖的圖例即可)
handles, labels = axes[0, 0].get_legend_handles_labels()
# 將 Threshold 標籤簡化，因各圖閾值可能微幅不同，統一標示意義即可
labels = ['Replicates', 'Conditions', 'Decision Threshold']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=14, frameon=False)

# 精確調整邊距，消除白邊
plt.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.12, wspace=0.05, hspace=0.05)

plt.savefig('combined_latent_space_distribution.pdf', dpi=300, bbox_inches='tight')
plt.show()