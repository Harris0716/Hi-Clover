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

# 導入你現有的 Dataset 與模型定義
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

criterion = TripletLoss(margin=args.margin)
# 論文實作發現固定 LR 穩定，但 AdamW + CosineAnnealing 對現代訓練效果更佳
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_training)

# ---------------------------------------------------------
# Debug 工具函數
# ---------------------------------------------------------
def get_stats(a_out, p_out, n_out):
    # 計算正負樣本的平均歐式距離
    d_ap = F.pairwise_distance(a_out, p_out, p=2).mean().item()
    d_an = F.pairwise_distance(a_out, n_out, p=2).mean().item()
    return d_ap, d_an

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
best_val_loss = float('inf')
prev_val_loss = float('inf')

print(f"Starting training for {args.epoch_training} epochs...")

for epoch in range(args.epoch_training):
    model.train()
    running_loss = 0.0
    
    for i, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # 每輪第一個 Batch 檢查輸入數據標準化狀態 
        if i == 0:
            with torch.no_grad():
                print(f"\n[Debug Epoch {epoch+1}] Input Max: {anchor.max().item():.4f}, Mean: {anchor.mean().item():.4f}")

        optimizer.zero_grad()
        a_out, p_out, n_out = model(anchor, positive, negative)
        
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        
        # 梯度剪裁，防止梯度爆炸或消失
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        # 每 100 步監控距離變化
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
    print(f"Epoch [{epoch+1}/{args.epoch_training}] Validation Loss: {avg_val_loss:.4f}")

    scheduler.step()

    # 模型存檔與 Early Stopping 
    model_save_name = f"{args.outpath}/{args.model_name}_seed{args.seed}.ckpt"
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_save_name.replace(".ckpt", "_best.ckpt"))
        print("Best model saved.")

    # 論文早停策略：強制訓練一定週期後，若驗證損失上升超過 10% 則停 
    if epoch >= args.epoch_enforced_training:
        if avg_val_loss > 1.1 * prev_val_loss:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        prev_val_loss = avg_val_loss

print("Training Completed.")