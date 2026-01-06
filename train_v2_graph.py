# Adagrad -> AdamW, add cosine annealing (LR scheduler)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
import json
import os
import time
import matplotlib.pyplot as plt

# 新增：用於 t-SNE 降維視覺化
from sklearn.manifold import TSNE

# 導入 Dataset 與模型定義
from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
from torch_plus.loss import TripletLoss
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Triplet network for Hi-C Replicate Analysis')
parser.add_argument('model_name', type=str, help='Model from models.py')
parser.add_argument('json_file', type=str, help='JSON dictionary with file paths')
parser.add_argument('learning_rate', type=float, help='Learning rate (Paper suggests 0.01)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size (Paper: 128)')
parser.add_argument('--epoch_training', type=int, default=100, help='Max epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=20, help='Enforced epochs before early stop')
parser.add_argument('--outpath', type=str, default="outputs/", help='Output directory')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--mask', type=bool, default=True, help='Mask diagonal (Paper: True)')
parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')
parser.add_argument("data_inputs", nargs='+', help="Keys for training and validation")

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# 設備設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.device_count() if torch.cuda.is_available() else 'CPU'} device.")

# 固定隨機種子
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ---------------------------------------------------------
# 檔名參數資訊定義 (用於統一標題和檔名)
# ---------------------------------------------------------
file_param_info_str = f"{args.model_name}_lr{args.learning_rate}_bs{args.batch_size}_m{args.margin}_wd{args.weight_decay}"
base_save_path = os.path.join(args.outpath, file_param_info_str)

# 繪圖用的通用標題
plot_title_base = (
    f"Model: {args.model_name} | LR: {args.learning_rate} | Batch: {args.batch_size}\n"
    f"Margin: {args.margin} | Seed: {args.seed} | Weight Decay: {args.weight_decay}"
)

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
with open(args.json_file) as f:
    dataset_config = json.load(f)

print("Loading Datasets...")
train_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset(
        [HiCDatasetDec.load(data_path) for data_path in dataset_config[data_name]["training"]],
        reference=reference_genomes[dataset_config[data_name]["reference"]])
    for data_name in args.data_inputs])

val_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset(
        [HiCDatasetDec.load(data_path) for data_path in dataset_config[data_name]["validation"]],
        reference=reference_genomes[dataset_config[data_name]["reference"]])
    for data_name in args.data_inputs])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=4, pin_memory=True)
# 注意：驗證 batch size 設大一點有助於 t-SNE 取樣效率，但要視 GPU 記憶體而定
val_loader = DataLoader(val_dataset, batch_size=256, sampler=SequentialSampler(val_dataset), num_workers=4, pin_memory=True)

no_of_batches = len(train_loader)
batches_validation = len(val_loader)

# ---------------------------------------------------------
# Model & Optimizer
# ---------------------------------------------------------
model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

torch.save(model.state_dict(), base_save_path + '_initial.ckpt')

criterion = TripletLoss(margin=args.margin)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_training)

# ---------------------------------------------------------
# Training Loop & Stats Containers
# ---------------------------------------------------------
best_val_loss = float('inf')
prev_val_loss = float('inf')
train_losses = []
val_losses = []

# 新增功能 [2]: 記錄 Learning Rate 變化
lr_history = []
# 新增功能 [5]: 記錄梯度範數變化
grad_norm_history = []
# 新增功能 [1]: 用於儲存最佳模型的距離分佈資料
best_d_ap_dist = []
best_d_an_dist = []

def get_stats(a_out, p_out, n_out):
    d_ap = F.pairwise_distance(a_out, p_out, p=2)
    d_an = F.pairwise_distance(a_out, n_out, p=2)
    return d_ap.mean().item(), d_an.mean().item()

print(f"Starting training for {args.epoch_training} epochs...")
total_start_time = time.time()

for epoch in range(args.epoch_training):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    epoch_grad_norms = [] # 每個 batch 的梯度範數

    for i, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        a_out, p_out, n_out = model(anchor, positive, negative)
        
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        
        # 新增功能 [5]: 計算並記錄梯度範數 (在 clip 之前)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        epoch_grad_norms.append(total_norm)

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 100 == 0 or (i + 1) == no_of_batches:
            d_ap_mean, d_an_mean = get_stats(a_out, p_out, n_out)
            print(f"Epoch [{epoch+1}/{args.epoch_training}], Step [{i+1}/{no_of_batches}], "
                  f"Loss: {running_loss/(i+1):.4f}, d(a,p): {d_ap_mean:.4f}, d(a,n): {d_an_mean:.4f}")
    
    # 紀錄本 Epoch 平均梯度範數
    grad_norm_history.append(np.mean(epoch_grad_norms))

    # ===========================
    # Validation Phase
    # ===========================
    model.eval()
    val_loss = 0.0
    # 新增功能 [1]: 暫存本 Epoch 的距離分佈
    current_val_d_ap = []
    current_val_d_an = []

    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            a_out, p_out, n_out = model(anchor, positive, negative)
            val_loss += criterion(a_out, p_out, n_out).item()
            
            # 新增功能 [1]: 收集距離資料
            d_ap = F.pairwise_distance(a_out, p_out, p=2)
            d_an = F.pairwise_distance(a_out, n_out, p=2)
            current_val_d_ap.extend(d_ap.cpu().numpy())
            current_val_d_an.extend(d_an.cpu().numpy())

    avg_val_loss = val_loss / batches_validation
    train_losses.append(running_loss / no_of_batches)
    val_losses.append(avg_val_loss)
    
    # 新增功能 [2]: 記錄當前 LR
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1}/{args.epoch_training}] Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.2e}, Time: {epoch_duration:.2f}s")

    scheduler.step()

    # 儲存最佳模型與其對應的距離分佈
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), base_save_path + '_best.ckpt')
        # 新增功能 [1]: 更新最佳模型的距離分佈資料
        best_d_ap_dist = current_val_d_ap
        best_d_an_dist = current_val_d_an
        print(f"Best model saved to {base_save_path}_best.ckpt")

    # 早停策略
    if epoch >= args.epoch_enforced_training:
        if avg_val_loss > 1.1 * prev_val_loss:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        prev_val_loss = avg_val_loss

# ---------------------------------------------------------
# 訓練結束：統計與繪圖
# ---------------------------------------------------------
torch.save(model.state_dict(), base_save_path + '_last.ckpt')
total_duration = time.time() - total_start_time
hours, rem = divmod(total_duration, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nTraining Completed. Total Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

# 定義通用繪圖函數以簡化代碼
def save_plot(fig, filename_suffix):
    plot_path = base_save_path + filename_suffix
    plt.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")

# --- Plot 0: Loss Curve (原有的) ---
fig_loss = plt.figure(figsize=(12, 7))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2, color='orange')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Triplet Loss', fontsize=12)
plt.title("Training History: Loss Curve\n" + plot_title_base, fontsize=12, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
save_plot(fig_loss, '_loss_curve.png')

# --- 新增功能 [1]: Distance Distribution Histogram (Best Model) ---
if best_d_ap_dist and best_d_an_dist:
    fig_dist = plt.figure(figsize=(12, 7))
    # 設定 bins，讓兩個分佈更容易比較
    bins = np.linspace(min(min(best_d_ap_dist), min(best_d_an_dist)),
                       max(max(best_d_ap_dist), max(best_d_an_dist)), 100)
    plt.hist(best_d_ap_dist, bins=bins, alpha=0.6, label='Positive Pair Distance d(a,p)', color='green', density=True)
    plt.hist(best_d_an_dist, bins=bins, alpha=0.6, label='Negative Pair Distance d(a,n)', color='red', density=True)
    plt.xlabel('L2 Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title("Distance Distribution (Best Validation Model)\n" + plot_title_base, fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    save_plot(fig_dist, '_dist_hist_best.png')

# --- 新增功能 [2]: Learning Rate Schedule ---
fig_lr = plt.figure(figsize=(12, 5))
plt.plot(lr_history, label='Learning Rate', color='purple', linewidth=2)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title("Learning Rate Schedule (Cosine Annealing)\n" + plot_title_base, fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log') # LR 通常用對數座標看比較清楚
save_plot(fig_lr, '_lr_schedule.png')

# --- 新增功能 [5]: Gradient Norm History ---
fig_grad = plt.figure(figsize=(12, 5))
plt.plot(grad_norm_history, label='Average Gradient Norm (L2)', color='teal', linewidth=1.5)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Gradient Norm', fontsize=12)
plt.title("Training Dynamics: Gradient Norm History\n" + plot_title_base, fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
# 可以添加一條線顯示 clip threshold
plt.axhline(y=1.0, color='r', linestyle=':', label='Clip Threshold (1.0)')
plt.legend()
save_plot(fig_grad, '_grad_norm.png')

# --- 新增功能 [3]: t-SNE Embedding Visualization (Best Model) ---
print("Generating t-SNE visualization for best model (this might take a moment)...")
# 載入最佳模型
model.load_state_dict(torch.load(base_save_path + '_best.ckpt'))
model.eval()

embeddings_to_plot = []
# 為了速度和記憶體，我們只取驗證集的一部分樣本進行 t-SNE (例如最多 3000 個點)
max_tsne_samples = 3000 
sample_count = 0

with torch.no_grad():
    for anchor, _, _ in val_loader:
        anchor = anchor.to(device)
        # 我們只需要 anchor 的 embedding 來觀察空間分佈
        a_out, _, _ = model(anchor, anchor, anchor) # 這裡只關心 a_out，後兩個輸入不重要
        embeddings_to_plot.extend(a_out.cpu().numpy())
        sample_count += anchor.size(0)
        if sample_count >= max_tsne_samples:
            break

embeddings_to_plot = np.array(embeddings_to_plot[:max_tsne_samples])

# 執行 t-SNE
tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(embeddings_to_plot)

fig_tsne = plt.figure(figsize=(10, 10))
# 因為 Triplet 資料通常沒有明確的類別標籤，我們暫時只繪製點的分佈結構
# 如果你的 dataset 能回傳標籤，可以改用 scatter(..., c=labels) 來上色
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5, s=15, color='royalblue')
plt.xlabel('t-SNE Dim 1', fontsize=12)
plt.ylabel('t-SNE Dim 2', fontsize=12)
plt.title(f"t-SNE of Anchor Embeddings (Best Model, N={len(tsne_results)})\n" + plot_title_base, fontsize=10, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
save_plot(fig_tsne, '_tsne_best.png')

print("\nAll Visualizations Completed.")