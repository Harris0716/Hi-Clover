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
import time  # 新增：用於時間統計
import matplotlib.pyplot as plt  # 新增：用於繪圖

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
# 檔名參數資訊定義
# ---------------------------------------------------------
file_param_info = f"{args.model_name}_{args.learning_rate}_{args.batch_size}_{args.seed}_{args.margin}"
base_save_path = os.path.join(args.outpath, file_param_info)

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
val_loader = DataLoader(val_dataset, batch_size=100, sampler=SequentialSampler(val_dataset), num_workers=4, pin_memory=True)

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
# Training Loop
# ---------------------------------------------------------
best_val_loss = float('inf')
prev_val_loss = float('inf')
train_losses = []  # 新增：紀錄訓練 Loss 繪圖用
val_losses = []    # 新增：紀錄驗證 Loss 繪圖用

def get_stats(a_out, p_out, n_out):
    d_ap = F.pairwise_distance(a_out, p_out, p=2).mean().item()
    d_an = F.pairwise_distance(a_out, n_out, p=2).mean().item()
    return d_ap, d_an

print(f"Starting training for {args.epoch_training} epochs...")
total_start_time = time.time()  # 新增：總時間開始

for epoch in range(args.epoch_training):
    epoch_start_time = time.time()  # 新增：單個 Epoch 時間開始
    model.train()
    running_loss = 0.0
    
    for i, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        if i == 0:
            with torch.no_grad():
                print(f"\n[Debug Epoch {epoch+1}] Input Max: {anchor.max().item():.4f}, Mean: {anchor.mean().item():.4f}")

        optimizer.zero_grad()
        a_out, p_out, n_out = model(anchor, positive, negative)
        
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 100 == 0 or (i + 1) == no_of_batches:
            d_ap, d_an = get_stats(a_out, p_out, n_out)
            print(f"Epoch [{epoch+1}/{args.epoch_training}], Step [{i+1}/{no_of_batches}], "
                  f"Loss: {running_loss/(i+1):.4f}, d(a,p): {d_ap:.4f}, d(a,n): {d_an:.4f}")

    # Validation Phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            a_out, p_out, n_out = model(anchor, positive, negative)
            val_loss += criterion(a_out, p_out, n_out).item()
    
    avg_val_loss = val_loss / batches_validation
    
    # 新增：儲存 Loss 紀錄
    train_losses.append(running_loss / no_of_batches)
    val_losses.append(avg_val_loss)
    
    # 新增：印出單個 Epoch 耗時
    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1}/{args.epoch_training}] Validation Loss: {avg_val_loss:.4f}, Time: {epoch_duration:.2f}s")

    scheduler.step()

    # 儲存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), base_save_path + '_best.ckpt')
        print(f"Best model saved to {base_save_path}_best.ckpt")

    # 早停策略 (維持你原本的 1.1 倍邏輯)
    if epoch >= args.epoch_enforced_training:
        if avg_val_loss > 1.1 * prev_val_loss:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        prev_val_loss = avg_val_loss

# ---------------------------------------------------------
# 結束統計與繪圖 (標題強化版)
# ---------------------------------------------------------
# 儲存最後模型
torch.save(model.state_dict(), base_save_path + '_last.ckpt')

# 計算總訓練時間
total_duration = time.time() - total_start_time
hours, rem = divmod(total_duration, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nTraining Completed. Total Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

# 自動繪製 Loss 曲線圖
plt.figure(figsize=(12, 7)) # 稍微加寬畫布以容納標題
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)

plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)

# --- 關鍵修改：動態標題 ---
# 使用 \n 換行讓標題更整潔
plot_title = (
    f"Training History: {args.model_name}\n"
    f"LR: {args.learning_rate} | Batch: {args.batch_size} | "
    f"Margin: {args.margin} | Seed: {args.seed} | Weight Decay: {args.weight_decay}"
)
plt.title(plot_title, fontsize=14, fontweight='bold', pad=15)

plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout() # 自動調整佈局確保文字不被切掉

plot_path = base_save_path + '_loss_curve.png'
plt.savefig(plot_path, dpi=300) # 提高解析度到 300dpi
plt.close()

print(f"Loss curve saved with parameters in title to: {plot_path}")