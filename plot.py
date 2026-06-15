import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# User settings
# =========================
data_dir = "0616_dist_pdfs/npz"   # 改成你的 npz 資料夾
out_path = "combined_distance_distribution_paper.pdf"

# 檔名前綴 vs 圖上顯示名稱
file_prefixes = ["liver", "NPC", "TCell"]
display_names = ["Liver", "NPC", "T Cell"]

phases = ["train_val", "test"]
phase_labels = ["Train + Val", "Test"]

# 顏色（沿用你目前這版的風格）
COLOR_REP = "#9FC3CC"   # replicates
COLOR_COND = "#7D809E"  # conditions
COLOR_THR = "black"     # threshold

# =========================
# Figure style
# =========================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 14,
    "axes.linewidth": 1.0,
})

fig, axes = plt.subplots(
    2, 3,
    figsize=(15.5, 8.8),
    sharex=True,
    sharey="col"
)

# =========================
# Plot panels
# =========================
for i, phase in enumerate(phases):
    for j, cell in enumerate(file_prefixes):
        ax = axes[i, j]
        file_path = os.path.join(data_dir, f"{cell}_{phase}_raw_dist.npz")

        if not os.path.exists(file_path):
            ax.text(
                0.5, 0.5,
                f"Missing:\n{cell}_{phase}_raw_dist.npz",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=12
            )
            ax.set_axis_off()
            continue

        data = np.load(file_path)

        dist = data["dist"]
        lbl = data["lbl"]
        threshold = data["threshold"]

        # threshold 可能是 scalar，也可能是 shape=(1,)
        if np.ndim(threshold) > 0:
            threshold = float(np.ravel(threshold)[0])
        else:
            threshold = float(threshold)

        rep_dist = dist[lbl == 0]
        cond_dist = dist[lbl == 1]

        # bins 設定：
        # 用資料整體範圍的 99.5 percentile，避免極端尾巴把圖拉太長
        x_min = max(0.0, np.min(dist))
        x_max = np.percentile(dist, 99.5)
        bins = np.linspace(x_min, x_max, 160)

        ax.hist(
            rep_dist,
            bins=bins,
            density=True,
            alpha=0.75,
            color=COLOR_REP,
            label="Replicates"
        )

        ax.hist(
            cond_dist,
            bins=bins,
            density=True,
            alpha=0.75,
            color=COLOR_COND,
            label="Conditions"
        )

        ax.axvline(
            threshold,
            color=COLOR_THR,
            linestyle="--",
            linewidth=1.8,
            label="Decision Threshold"
        )

        # 第一列才放資料集標題
        if i == 0:
            ax.set_title(display_names[j], pad=14)

        # 只保留左邊與下方刻度，讓版面乾淨一點
        ax.tick_params(direction="out", length=4, width=1)

# =========================
# Global labels
# =========================
fig.supxlabel("Euclidean Distance", fontsize=24, y=0.04)
fig.supylabel("Probability Density", fontsize=24, x=0.03)

# 列標籤
fig.text(
    0.055, 0.72, phase_labels[0],
    va="center", ha="center",
    rotation="vertical",
    fontsize=24, fontweight="bold"
)
fig.text(
    0.055, 0.28, phase_labels[1],
    va="center", ha="center",
    rotation="vertical",
    fontsize=24, fontweight="bold"
)

# =========================
# Global legend
# =========================
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    frameon=False,
    fontsize=20
)

# =========================
# Layout
# =========================
plt.subplots_adjust(
    left=0.09,
    right=0.99,
    top=0.90,
    bottom=0.12,
    wspace=0.15,
    hspace=0.08
)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()