# Ultimate Version: AdamW + Scheduler + Brightness Jitter + Constrained Loss
# Goal: High Accuracy (>0.76) AND High Separation Index (>0.6) AND Good UMAP

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
# from torch_plus.loss import TripletLoss # 我們改用自定義的 Constrained Loss
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# [新增] Constrained Triplet Loss
# 解決 UMAP 分不開的問題，強迫正樣本聚在一起
# ---------------------------------------------------------
class ConstrainedTripletLoss(nn.Module):
    def __init__(self, margin=1.0, lambda_c=0.1):
        super(ConstrainedTripletLoss, self).__init__()
        self.margin = margin
        self.lambda_c = lambda_c # 緊緻度權重

    def forward(self, anchor, positive, negative):
        # 1. 計算距離
        d_ap = F.pairwise_distance(anchor, positive)
        d_an = F.pairwise_distance(anchor, negative)
        
        # 2. 標準 Triplet Loss (確保相對距離)
        triplet_loss = F.relu(d_ap - d_an + self.margin).mean()
        
        # 3. 緊緻度約束 (Compactness Penalty)
        # 強迫 d(a,p) 趨近於 0，而不只是比 d(a,n) 小
        compactness_loss = (d_ap ** 2).mean()
        
        return triplet_loss + self.lambda_c * compactness_loss

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Triplet network (Ultimate Version)')
parser.add_argument('model_name', type=str, help='Model from models.py')
parser.add_argument('json_file', type=str, help='JSON dictionary with file paths')
parser.add_argument('learning_rate', type=float, help='Learning rate (Suggest 0.001 for Adam)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epoch_training', type=int, default=100, help='Max epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=20, help='Enforced epochs')
parser.add_argument('--outpath', type=str, default="outputs/", help='Output directory')
parser.add_argument('--seed', type=int, default=30004, help='Random seed')
parser.add_argument('--mask', type=bool, default=False, help='Mask diagonal')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--margin', type=float, default=0.5, help='Margin for triplet loss')
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
file_param_info = f"{args.model_name}_AdamW_{args.learning_rate}_{args.batch_size}_{args.seed}_{args.margin}"
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
# Model & Optimizer & Loss
# ---------------------------------------------------------
model = eval("models." + args.model_name)(mask=args.mask).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

# [修改 1] 使用 Constrained Loss 來提升 Separation Index
# lambda_c=0.1 是一個保守的起始值，如果 d(a,p) 還是太大，可以試試 0.2
criterion = ConstrainedTripletLoss(margin=args.margin, lambda_c=0.1)

# [修改 2] 使用 AdamW 取代 Adagrad
# Weight Decay 1e-4 用於防止過擬合
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

# [修改 3] 加入 Learning Rate Scheduler
# 當 Val Loss 卡住 5 個 Epoch 不動時，LR 減半
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# [保留] Data Augmentation
jitter_transform = T.ColorJitter(brightness=0.2, contrast=0.2)

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
best_val_loss = float('inf')
patience_counter = 0 
train_losses, val_losses, val_log_ratio_history, grad_norm_history = [], [], [], []
best_ap_dist, best_an_dist = [], []

print(f"Starting training: {file_param_info}")
print(f"Config: AdamW + Scheduler + Brightness Jitter + Constrained Loss")

total_start_time = time.time()

for epoch in range(args.epoch_training):
    epoch_start = time.time()
    model.train()
    running_loss, e_norms = 0.0, []
    
    for i, data in enumerate(train_loader):
        a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
        
        # Apply Jitter
        a = jitter_transform(a)
        p = jitter_transform(p)
        n = jitter_transform(n)

        optimizer.zero_grad()
        a_out, p_out, n_out = model(a, p, n)
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        e_norms.append(grad_norm)
        
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
            d_ap = F.pairwise_distance(a_out, p_out).mean().item()
            d_an = F.pairwise_distance(a_out, n_out).mean().item()
            # 這裡印出的 Loss 會包含 Compactness term，所以數值可能比單純 Triplet Loss 高一點
            print(f"Epoch [{epoch+1}/{args.epoch_training}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/(i+1):.4f}, d(a,p): {d_ap:.4f}, d(a,n): {d_an:.4f}")

    # Validation Phase
    model.eval()
    val_loss_sum, c_ap, c_an = 0.0, [], []
    with torch.no_grad():
        for data in val_loader:
            a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
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
    
    # [修改 4] 更新 Scheduler
    scheduler.step(avg_v)

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
ax[2].plot(grad_norm_history, color='teal'); ax[2].set_title('Gradient Norm'); ax[2].axhline(1.0, color='r', ls='--')
fig1.suptitle(f"Training Metrics | Model: {args.model_name}\nAdamW | Scheduler | Constrained Loss"); save_fig(fig1, '_training_stats.pdf')

fig2 = plt.figure(figsize=(10, 7))
plt.hist(best_ap_dist, bins=50, alpha=0.6, label='Positives d(a,p)', color='g', density=True)
plt.hist(best_an_dist, bins=50, alpha=0.6, label='Negatives d(a,n)', color='r', density=True)
plt.title(f"Best Model Distance Distribution\n{file_param_info}"); plt.legend()
save_fig(fig2, '_val_dist_hist.pdf')

print(f"Training Complete. Total Time: {(time.time()-total_start_time)/60:.2f} mins")