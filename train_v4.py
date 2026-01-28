# Final Robust Version: Adagrad + Jitter + Gradient Clipping
# Strategy: High Accuracy, High Replicate Rate, Stable Training

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse, json, os, time
import matplotlib.pyplot as plt

from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
from torch_plus.loss import TripletLoss
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Triplet network (Adagrad + Jitter + Clipping)')
parser.add_argument('model_name', type=str, help='Model from models.py')
parser.add_argument('json_file', type=str, help='JSON dictionary with file paths')
parser.add_argument('learning_rate', type=float, help='Learning rate (Suggest 0.01 for Adagrad)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epoch_training', type=int, default=100, help='Max epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=20, help='Enforced epochs')
parser.add_argument('--outpath', type=str, default="outputs/", help='Output directory')
parser.add_argument('--seed', type=int, default=30004, help='Random seed')
parser.add_argument('--mask', type=bool, default=False, help='Mask diagonal')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--margin', type=float, default=0.5, help='Margin for triplet loss')
parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping max norm')
parser.add_argument("data_inputs", nargs='+', help="Keys for training and validation")

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ---------------------------------------------------------
# parameters
# ---------------------------------------------------------
file_param_info = f"{args.model_name}_{args.learning_rate}_{args.batch_size}_{args.seed}_{args.margin}"
base_save_path = os.path.join(args.outpath, file_param_info)

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
with open(args.json_file) as f: dataset_config = json.load(f)

train_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(p) for p in dataset_config[n]["training"]], 
    reference=reference_genomes[dataset_config[n]["reference"]]) for n in args.data_inputs])

val_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(p) for p in dataset_config[n]["validation"]], 
    reference=reference_genomes[dataset_config[n]["reference"]]) for n in args.data_inputs])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=100, sampler=SequentialSampler(val_dataset), num_workers=4, pin_memory=True)

# ---------------------------------------------------------
# Model & Optimizer
# ---------------------------------------------------------
model = eval("models." + args.model_name)(mask=args.mask).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

# 使用 Hard Margin (0.5)
criterion = TripletLoss(margin=args.margin)

# 使用 Adagrad (適合稀疏特徵)
optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)

# ---------------------------------------------------------
# Data Augmentation (Brightness Jitter)
# ---------------------------------------------------------
jitter_transform = T.ColorJitter(brightness=0.2, contrast=0.2)

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
best_val_loss = float('inf')
patience_counter = 0 
train_losses, val_losses, val_log_ratio_history, grad_norm_history = [], [], [], []
best_ap_dist, best_an_dist = [], []

print(f"Starting training: {file_param_info}")
print(f"Config: Adagrad + Jitter + Gradient Clipping (max_norm={args.max_norm})")

total_start_time = time.time()

for epoch in range(args.epoch_training):
    epoch_start = time.time()
    model.train()
    running_loss, e_norms = 0.0, []
    
    for i, data in enumerate(train_loader):
        a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
        
        # Apply Jitter (Training Only)
        a = jitter_transform(a)
        p = jitter_transform(p)
        n = jitter_transform(n)

        optimizer.zero_grad()
        a_out, p_out, n_out = model(a, p, n)
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        
        # -----------------------------------------------------
        # [修改關鍵點] Gradient Clipping
        # -----------------------------------------------------
        # 以前是用 float('inf') 只觀察不剪裁
        # 現在改用 args.max_norm (預設 1.0) 進行強制剪裁
        # 這會限制梯度的最大長度，防止 Loss 暴衝，保護 Adagrad
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        e_norms.append(grad_norm.item()) # 紀錄原本的 norm 大小
        
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
            d_ap = F.pairwise_distance(a_out, p_out).mean().item()
            d_an = F.pairwise_distance(a_out, n_out).mean().item()
            print(f"Epoch [{epoch+1}/{args.epoch_training}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/(i+1):.4f}, d(a,p): {d_ap:.4f}, d(a,n): {d_an:.4f}")

    # Validation Phase
    model.eval()
    val_loss_sum, c_ap, c_an = 0.0, [], []
    with torch.no_grad():
        for data in val_loader:
            a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
            # 驗證時不使用 Jitter
            ao, po, no = model(a, p, n)
            val_loss_sum += criterion(ao, po, no).item()
            c_ap.extend(F.pairwise_distance(ao, po).cpu().numpy())
            c_an.extend(F.pairwise_distance(ao, no).cpu().numpy())
    
    avg_v = val_loss_sum / len(val_loader)
    avg_ap, avg_an = np.mean(c_ap), np.mean(c_an)
    l_ratio = np.log10((avg_an + 1e-6) / (avg_ap + 1e-6))
    
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(avg_v)
    val_log_ratio_history.append(l_ratio)
    grad_norm_history.append(np.mean(e_norms))

    print(f"Epoch [{epoch+1}] Val Loss: {avg_v:.4f}, Log-Ratio: {l_ratio:.4f}, Time: {time.time()-epoch_start:.2f}s")

    if avg_v < best_val_loss:
        best_val_loss = avg_v
        patience_counter = 0
        torch.save(model.state_dict(), base_save_path + '_best.ckpt')
        best_ap_dist, best_an_dist = c_ap, c_an
    else:
        if epoch >= args.epoch_enforced_training:
            patience_counter += 1
            print(f"-> No improvement. Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def save_fig(fig, suffix):
    plt.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(base_save_path + suffix, dpi=300); plt.close(fig)

fig1, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].plot(train_losses, label='Train'); ax[0].plot(val_losses, label='Val'); ax[0].set_title('Loss Evolution'); ax[0].legend()
ax[1].plot(val_log_ratio_history, color='blue'); ax[1].set_title('Log-Ratio (log(a_n/a_p))'); ax[1].axhline(0, color='k', ls='--')
# 這裡畫一條紅線表示 Clipping 的閾值
ax[2].plot(grad_norm_history, color='teal'); ax[2].set_title('Gradient Norm'); ax[2].axhline(args.max_norm, color='r', ls='--', label='Clip Threshold')
ax[2].legend()
fig1.suptitle(f"Training Metrics | Model: {args.model_name}\nLR: {args.learning_rate} | Margin: {args.margin} | Jitter+Clip"); save_fig(fig1, '_training_stats.pdf')

fig2 = plt.figure(figsize=(10, 7))
plt.hist(best_ap_dist, bins=50, alpha=0.6, label='Positives d(a,p)', color='g', density=True)
plt.hist(best_an_dist, bins=50, alpha=0.6, label='Negatives d(a,n)', color='r', density=True)
plt.title(f"Best Model Distance Distribution\n{file_param_info}"); plt.legend()
save_fig(fig2, '_val_dist_hist.pdf')

print(f"Training Complete. Total Time: {(time.time()-total_start_time)/60:.2f} mins")