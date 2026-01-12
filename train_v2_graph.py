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

parser = argparse.ArgumentParser(description='Triplet network training for Hi-C Replicates')
parser.add_argument('model_name', type=str)
parser.add_argument('json_file', type=str)
parser.add_argument('learning_rate', type=float)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch_training', type=int, default=100)
parser.add_argument('--epoch_enforced_training', type=int, default=20)
parser.add_argument('--outpath', type=str, default="outputs/")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mask', type=bool, default=True)
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument("data_inputs", nargs='+')
args = parser.parse_args()

os.makedirs(args.outpath, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

f_info = f"{args.model_name}_lr{args.learning_rate}_bs{args.batch_size}_m{args.margin}"
base_save_path = os.path.join(args.outpath, f_info)

with open(args.json_file) as f:
    dataset_config = json.load(f)

train_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(p) for p in dataset_config[n]["training"]], 
    reference=reference_genomes[dataset_config[n]["reference"]]) for n in args.data_inputs])

val_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(p) for p in dataset_config[n]["validation"]], 
    reference=reference_genomes[dataset_config[n]["reference"]]) for n in args.data_inputs])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, sampler=SequentialSampler(val_dataset), num_workers=4, pin_memory=True)

model = eval("models." + args.model_name)(mask=args.mask).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

criterion = TripletLoss(margin=args.margin)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_training)

best_val_loss, prev_val_loss = float('inf'), float('inf')
train_losses, val_losses, val_log_ratio_history = [], [], []
grad_norm_history = []

total_start_time = time.time()
for epoch in range(args.epoch_training):
    epoch_start = time.time()
    model.train()
    running_loss, e_norms = 0.0, []
    for i, data in enumerate(train_loader):
        a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
        optimizer.zero_grad()
        ao, po, no = model(a, p, n)
        loss = criterion(ao, po, no)
        loss.backward()
        e_norms.append(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{args.epoch_training}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/(i+1):.4f}")

    model.eval()
    v_loss, c_ap, c_an = 0.0, [], []
    with torch.no_grad():
        for data in val_loader:
            a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
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

    print(f"Epoch [{epoch+1}] Val Loss: {avg_v:.4f}, Log-Ratio: {l_ratio:.4f}, Time: {time.time()-epoch_start:.2fs}")
    if avg_v < best_val_loss:
        best_val_loss = avg_v
        torch.save(model.state_dict(), base_save_path + '_best.ckpt')
        best_ap_dist, best_an_dist = c_ap, c_an
    
    scheduler.step()
    if epoch >= args.epoch_enforced_training and avg_v > 1.1 * prev_val_loss: break
    prev_val_loss = avg_v

def save_fig(fig, suffix):
    plt.tight_layout(); fig.savefig(base_save_path + suffix, dpi=300); plt.close(fig)

# 1. Convergence Metrics
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(train_losses, label='Train Loss'); ax[0].plot(val_losses, label='Val Loss')
ax[0].set_title('Model Convergence: Loss'); ax[0].legend()
ax[1].plot(grad_norm_history, color='teal'); ax[1].set_title('Gradient Stability (Norm)'); ax[1].axhline(1.0, color='r', ls='--')
save_fig(fig, '_training_convergence.png')

# 2. Separation Quality (Log-Ratio)
fig_log = plt.figure(figsize=(8, 5))
plt.plot(range(1, len(val_log_ratio_history)+1), val_log_ratio_history, color='blue', marker='o', markersize=4)
plt.axhline(0, color='black', ls='--', alpha=0.5)
plt.xlabel('Training Epochs'); plt.ylabel('log10 (d_an / d_ap)')
plt.title("Embedding Separation Quality (Validation Set)"); plt.grid(True, alpha=0.3)
save_fig(fig_log, '_separation_quality.png')

# 3. Validation Distribution
fig_dist = plt.figure(figsize=(8, 5))
plt.hist(best_ap_dist, bins=50, alpha=0.6, label='Positives d(a,p)', color='g', density=True)
plt.hist(best_an_dist, bins=50, alpha=0.6, label='Negatives d(a,n)', color='r', density=True)
plt.title("Validation Set: Pairwise Distance Distribution"); plt.legend()
save_fig(fig_dist, '_val_distribution.png')

print(f"Training Complete. Total Time: {(time.time()-total_start_time)/60:.2f} mins")