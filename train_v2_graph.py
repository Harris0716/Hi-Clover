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
parser.add_argument('learning_rate', type=float, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epoch_training', type=int, default=100, help='Max epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=20, help='Enforced epochs')
parser.add_argument('--outpath', type=str, default="outputs/", help='Output directory')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--mask', type=bool, default=True, help='Mask diagonal')
parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')
parser.add_argument("data_inputs", nargs='+', help="Keys for training and validation")

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.device_count() if torch.cuda.is_available() else 'CPU'} device.")

# fixed random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# file name parameter information
file_param_info_str = f"{args.model_name}_lr{args.learning_rate}_bs{args.batch_size}_m{args.margin}_wd{args.weight_decay}"
base_save_path = os.path.join(args.outpath, file_param_info_str)
plot_title_base = f"Model: {args.model_name} | LR: {args.learning_rate} | Margin: {args.margin}"

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

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, sampler=SequentialSampler(val_dataset), num_workers=2, pin_memory=True)

# ---------------------------------------------------------
# Model & Optimizer
# ---------------------------------------------------------
model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

criterion = TripletLoss(margin=args.margin)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_training)

# ---------------------------------------------------------
# Stats Containers
# ---------------------------------------------------------
best_val_loss = float('inf')
prev_val_loss = float('inf')
train_losses, val_losses, lr_history, grad_norm_history = [], [], [], []
best_d_ap_dist, best_d_an_dist = [], []
val_d_ap_history, val_d_an_history = [], []
val_log_ratio_history = [] 

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
print(f"Starting training for {args.epoch_training} epochs...")
total_start_time = time.time()

for epoch in range(args.epoch_training):
    epoch_start_time = time.time()
    model.train()
    running_loss, epoch_grad_norms = 0.0, []
    
    for i, data in enumerate(train_loader):
        anchor, positive, negative = data[0].to(device), data[1].to(device), data[2].to(device)

        optimizer.zero_grad()
        a_out, p_out, n_out = model(anchor, positive, negative)
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        epoch_grad_norms.append(total_norm.item())
        
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
            with torch.no_grad():
                d_ap_batch = F.pairwise_distance(a_out, p_out, p=2).mean().item()
                d_an_batch = F.pairwise_distance(a_out, n_out, p=2).mean().item()
            # 保持原本的 Log 格式
            print(f"Epoch [{epoch+1}/{args.epoch_training}], Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {running_loss/(i+1):.4f}, d(a,p): {d_ap_batch:.4f}, d(a,n): {d_an_batch:.4f}")

    # Validation Phase
    model.eval()
    val_loss, current_val_d_ap, current_val_d_an = 0.0, [], []
    with torch.no_grad():
        for data in val_loader:
            anchor, positive, negative = data[0].to(device), data[1].to(device), data[2].to(device)
            a_out, p_out, n_out = model(anchor, positive, negative)
            val_loss += criterion(a_out, p_out, n_out).item()
            
            d_ap = F.pairwise_distance(a_out, p_out, p=2)
            d_an = F.pairwise_distance(a_out, n_out, p=2)
            current_val_d_ap.extend(d_ap.cpu().numpy())
            current_val_d_an.extend(d_an.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_d_ap = np.mean(current_val_d_ap)
    avg_val_d_an = np.mean(current_val_d_an)
    
    # 計算 Log-Ratio
    log_ratio = np.log10((avg_val_d_an + 1e-6) / (avg_val_d_ap + 1e-6))
    val_log_ratio_history.append(log_ratio)

    train_losses.append(running_loss / len(train_loader))
    val_losses.append(avg_val_loss)
    val_d_ap_history.append(avg_val_d_ap)
    val_d_an_history.append(avg_val_d_an)
    lr_history.append(optimizer.param_groups[0]['lr'])
    grad_norm_history.append(np.mean(epoch_grad_norms))

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1}] Val Loss: {avg_val_loss:.4f}, "
          f"d(a,p): {avg_val_d_ap:.4f}, d(a,n): {avg_val_d_an:.4f}, Log-Ratio: {log_ratio:.4f}, Time: {epoch_duration:.2f}s")

    scheduler.step()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), base_save_path + '_best.ckpt')
        best_d_ap_dist, best_d_an_dist = current_val_d_ap, current_val_d_an
        print(f"*** Best model saved (Loss: {best_val_loss:.4f}) ***")

    if epoch >= args.epoch_enforced_training and avg_val_loss > 1.1 * prev_val_loss:
        print("Early stopping triggered.")
        break
    prev_val_loss = avg_val_loss

# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------
def save_fig(fig, suffix):
    plt.tight_layout()
    fig.savefig(base_save_path + suffix, dpi=300)
    plt.close(fig)

# 1. 訓練收斂圖
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(train_losses, label='Train'); ax[0].plot(val_losses, label='Val'); ax[0].set_title('Model Convergence: Loss'); ax[0].legend()
ax[1].plot(lr_history, color='purple'); ax[1].set_title('Learning Rate'); ax[1].set_yscale('log')
ax[2].plot(grad_norm_history, color='teal'); ax[2].set_title('Gradient Norm'); ax[2].axhline(1.0, color='r', linestyle='--')
save_fig(fig, '_training_stats.png')

# 2. Log-Ratio 獨立趨勢圖 (優化標題)
fig_log = plt.figure(figsize=(10, 6))
plt.plot(range(1, len(val_log_ratio_history) + 1), val_log_ratio_history, color='blue', marker='s', linewidth=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Epochs'); plt.ylabel('log10(d_an / d_ap)')
plt.title("Embedding Separation Quality (Log-Ratio)"); plt.grid(True, alpha=0.3)
save_fig(fig_log, '_log_ratio_evolution.png')

# 3. 驗證集距離分佈 (Histogram)
fig_dist = plt.figure(figsize=(10, 6))
plt.hist(best_d_ap_dist, bins=50, alpha=0.6, label='Positive d(a,p)', color='g', density=True)
plt.hist(best_d_an_dist, bins=50, alpha=0.6, label='Negative d(a,n)', color='r', density=True)
plt.title("Validation Set: Pairwise Distance Distribution"); plt.legend()
save_fig(fig_dist, '_dist_hist.png')

print(f"\nTraining Complete. Total Time: {(time.time()-total_start_time)/60:.2f} mins")