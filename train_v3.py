import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse, json, os, time
import matplotlib.pyplot as plt

from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
from torch_plus.loss import TripletLoss  # 改回標準 Triplet Loss 以配合 Margin
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Triplet network training (V3: Fixed Overfitting + Input Norm)')
parser.add_argument('model_name', type=str, help='Model from models.py')
parser.add_argument('json_file', type=str, help='JSON dictionary with file paths')
parser.add_argument('learning_rate', type=float, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epoch_training', type=int, default=100, help='Max epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=20, help='Enforced epochs')
parser.add_argument('--outpath', type=str, default="outputs/", help='Output directory')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--mask', type=lambda x: x.lower() == 'true', default=True, help='Mask diagonal')
parser.add_argument('--margin', type=float, default=0.5, help='Margin for triplet loss')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for AdamW (Recommended: 1e-3)')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--clip_val', type=float, default=2.0, help='Max gradient norm')
parser.add_argument('--input_norm', type=lambda x: x.lower() == 'true', default=True, help='Apply per-image z-score normalization')
parser.add_argument("data_inputs", nargs='+', help="Keys for training and validation")

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# --- title setting  ---
cell_line = args.data_inputs[0] + " data"
# 檔名加入 V3 標記，方便區分
f_info = f"{args.model_name}_V3_lr{args.learning_rate}_bs{args.batch_size}_wd{args.weight_decay}"
base_save_path = os.path.join(args.outpath, f_info)
param_title = (f"Model: {args.model_name} | LR: {args.learning_rate} | BS: {args.batch_size}\n"
               f"Margin: {args.margin} | WD: {args.weight_decay} | Norm: {args.input_norm}")

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

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, sampler=SequentialSampler(val_dataset), num_workers=0, pin_memory=True)

# ---------------------------------------------------------
# Model, Optimizer & Criterion
# ---------------------------------------------------------
model = eval("models." + args.model_name)(mask=args.mask).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

# [修正 1] 使用標準 TripletLoss 以支援 Margin 參數 (這對 T-cell 很重要)
criterion = TripletLoss(margin=args.margin)

# [修正 2] 使用 AdamW + Weight Decay (解決過擬合)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# [修正 3] 使用 ReduceLROnPlateau (針對 T-cell 瓶頸更有效)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses, val_log_ratio_history, grad_norm_history = [], [], [], []

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
print(f"Starting training V3 for {args.epoch_training} epochs...")
print(f"Config: Input Norm={args.input_norm}, Weight Decay={args.weight_decay}")
total_start_time = time.time()

for epoch in range(args.epoch_training):
    epoch_start = time.time()
    model.train()
    running_loss, e_norms = 0.0, []
    
    for i, data in enumerate(train_loader):
        a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
        
        # [關鍵修正 4] 加回 Input Normalization (T-cell 救星)
        if args.input_norm:
            a = (a - a.mean(dim=(2, 3), keepdim=True)) / (a.std(dim=(2, 3), keepdim=True) + 1e-8)
            p = (p - p.mean(dim=(2, 3), keepdim=True)) / (p.std(dim=(2, 3), keepdim=True) + 1e-8)
            n = (n - n.mean(dim=(2, 3), keepdim=True)) / (n.std(dim=(2, 3), keepdim=True) + 1e-8)

        optimizer.zero_grad()
        ao, po, no = model(a, p, n)
        loss = criterion(ao, po, no)
        loss.backward()
        
        # Gradient Clipping
        norm_val = nn.utils.clip_grad_norm_(model.parameters(), args.clip_val).item()
        e_norms.append(norm_val)
        
        optimizer.step()
        running_loss += loss.item()
        
        # Logging steps
        if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
            with torch.no_grad():
                dap = F.pairwise_distance(ao, po).mean().item()
                dan = F.pairwise_distance(ao, no).mean().item()
            print(f"Epoch [{epoch+1}/{args.epoch_training}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/(i+1):.4f}, d(a,p): {dap:.4f}, d(a,n): {dan:.4f}")

    # Validation Phase
    model.eval()
    v_loss, c_ap, c_an = 0.0, [], []
    with torch.no_grad():
        for data in val_loader:
            a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
            
            # Validation 也要做一樣的正規化
            if args.input_norm:
                a = (a - a.mean(dim=(2, 3), keepdim=True)) / (a.std(dim=(2, 3), keepdim=True) + 1e-8)
                p = (p - p.mean(dim=(2, 3), keepdim=True)) / (p.std(dim=(2, 3), keepdim=True) + 1e-8)
                n = (n - n.mean(dim=(2, 3), keepdim=True)) / (n.std(dim=(2, 3), keepdim=True) + 1e-8)

            ao, po, no = model(a, p, n)
            v_loss += criterion(ao, po, no).item()
            c_ap.extend(F.pairwise_distance(ao, po).cpu().numpy())
            c_an.extend(F.pairwise_distance(ao, no).cpu().numpy())
    
    avg_v = v_loss / len(val_loader)
    avg_ap, avg_an = np.mean(c_ap), np.mean(c_an)
    l_ratio = np.log10((avg_an + 1e-6) / (avg_ap + 1e-6))
    
    val_losses.append(avg_v); val_log_ratio_history.append(l_ratio)
    train_losses.append(running_loss / len(train_loader))
    grad_norm_history.append(np.mean(e_norms))
    
    # Update Scheduler
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_v)

    print(f"Epoch [{epoch+1}] Val Loss: {avg_v:.4f}, Log-Ratio: {l_ratio:.4f}, LR: {current_lr:.6f}, Time: {time.time()-epoch_start:.2f}s")
    
    # Save & Early Stopping
    if avg_v < best_val_loss:
        best_val_loss = avg_v
        patience_counter = 0
        torch.save(model.state_dict(), base_save_path + '_best.ckpt')
        best_ap_dist, best_an_dist = c_ap, c_an
        print(f"-> Model Saved (Best Loss: {best_val_loss:.4f})")
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
ax[1].plot(val_log_ratio_history, color='blue'); ax[1].set_title('Embedding Separation Quality (Log Ratio)'); ax[1].axhline(0, color='k', ls='--')
ax[2].plot(grad_norm_history, color='teal'); ax[2].set_title('Gradient Norm'); ax[2].axhline(args.clip_val, color='r', ls='--')
fig1.suptitle(f"Training Metrics | {cell_line}\n{param_title}"); save_fig(fig1, '_training_stats.pdf')

fig2 = plt.figure(figsize=(10, 7))
plt.hist(best_ap_dist, bins=50, alpha=0.6, label='Positives d(a,p)', color='g', density=True)
plt.hist(best_an_dist, bins=50, alpha=0.6, label='Negatives d(a,n)', color='r', density=True)
plt.title(f"Validation Set: Pairwise Distance Distribution | {cell_line}\n{param_title}"); plt.legend()
save_fig(fig2, '_val_dist_hist.pdf')

print(f"Training Complete. Total Time: {(time.time()-total_start_time)/60:.2f} mins")