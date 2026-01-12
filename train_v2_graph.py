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
from sklearn.manifold import TSNE
from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
from torch_plus.loss import TripletLoss
from HiSiNet.reference_dictionaries import reference_genomes

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

file_param_info_str = f"{args.model_name}_lr{args.learning_rate}_bs{args.batch_size}_m{args.margin}_wd{args.weight_decay}"
base_save_path = os.path.join(args.outpath, file_param_info_str)
plot_title_base = f"Model: {args.model_name} | LR: {args.learning_rate} | Margin: {args.margin}"

with open(args.json_file) as f:
    dataset_config = json.load(f)

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

model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

criterion = TripletLoss(margin=args.margin)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_training)

best_val_loss = float('inf')
prev_val_loss = float('inf')
train_losses, val_losses, lr_history, grad_norm_history = [], [], [], []
val_d_ap_history, val_d_an_history, val_log_ratio_history = [], [], []

print(f"Starting training for {args.epoch_training} epochs...")
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

    model.eval()
    val_loss, cur_val_d_ap, cur_val_d_an = 0.0, [], []
    with torch.no_grad():
        for data in val_loader:
            anchor, positive, negative = data[0].to(device), data[1].to(device), data[2].to(device)
            a_out, p_out, n_out = model(anchor, positive, negative)
            val_loss += criterion(a_out, p_out, n_out).item()
            cur_val_d_ap.extend(F.pairwise_distance(a_out, p_out).cpu().numpy())
            cur_val_d_an.extend(F.pairwise_distance(a_out, n_out).cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    avg_d_ap, avg_d_an = np.mean(cur_val_d_ap), np.mean(cur_val_d_an)
    log_ratio = np.log10((avg_d_an + 1e-6) / (avg_d_ap + 1e-6))
    
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(avg_val_loss); val_d_ap_history.append(avg_d_ap); val_d_an_history.append(avg_d_an)
    val_log_ratio_history.append(log_ratio); lr_history.append(optimizer.param_groups[0]['lr'])
    grad_norm_history.append(np.mean(epoch_grad_norms))

    print(f"Epoch [{epoch+1}/{args.epoch_training}] Val Loss: {avg_val_loss:.4f}, d(a,p): {avg_d_ap:.4f}, d(a,n): {avg_d_an:.4f}, Log-Ratio: {log_ratio:.4f}")

    scheduler.step()
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), base_save_path + '_best.ckpt')
        best_d_ap_dist, best_d_an_dist = cur_val_d_ap, cur_val_d_an

    if epoch >= args.epoch_enforced_training and avg_val_loss > 1.1 * prev_val_loss:
        break
    prev_val_loss = avg_val_loss

def save_fig(fig, suffix):
    plt.tight_layout(); fig.savefig(base_save_path + suffix, dpi=300); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(train_losses, label='Train'); ax[0].plot(val_losses, label='Val'); ax[0].set_title('Loss'); ax[0].legend()
ax[1].plot(lr_history, color='purple'); ax[1].set_title('Learning Rate'); ax[1].set_yscale('log')
ax[2].plot(grad_norm_history, color='teal'); ax[2].set_title('Gradient Norm')
save_fig(fig, '_training_stats.png')

fig_evol, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(val_d_ap_history, label='Avg d(a,p)', color='g', marker='o')
ax1.plot(val_d_an_history, label='Avg d(a,n)', color='r', marker='o')
ax1.set_ylabel('Distance'); ax1.legend(loc='upper left')
ax2 = ax1.twinx(); ax2.plot(val_log_ratio_history, label='Log Ratio', color='b', linestyle='--')
ax2.set_ylabel('log10(d_an/d_ap)'); ax2.legend(loc='upper right')
save_fig(fig_evol, '_distance_log_ratio.png')

fig_dist = plt.figure(figsize=(10, 6))
plt.hist(best_d_ap_dist, bins=50, alpha=0.6, label='Positive d(a,p)', color='g', density=True)
plt.hist(best_d_an_dist, bins=50, alpha=0.6, label='Negative d(a,n)', color='r', density=True)
plt.title("Distance Distribution (Best Model)"); plt.legend()
save_fig(fig_dist, '_dist_hist.png')