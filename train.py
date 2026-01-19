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
from torch_plus.loss import TripletLoss, soft_margin_triplet_loss
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Triplet network training for Hi-C Replicate Analysis')
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
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# --- title setting  ---
cell_line = args.data_inputs[0] + " data"
param_title = (f"Cell Line: {cell_line} | Model: {args.model_name} | Seed: {args.seed} | LR: {args.learning_rate}\n"
               f"BS: {args.batch_size} | Margin: {args.margin} | WD: {args.weight_decay}")

f_info = f"{args.model_name}_lr{args.learning_rate}_bs{args.batch_size}_m{args.margin}_wd{args.weight_decay}"
base_save_path = os.path.join(args.outpath, f_info)

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

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, sampler=SequentialSampler(val_dataset), num_workers=2, pin_memory=True)

model = eval("models." + args.model_name)(mask=args.mask).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

criterion = soft_margin_triplet_loss
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_training)

best_val_loss, prev_val_loss = float('inf'), float('inf')
train_losses, val_losses, val_log_ratio_history, grad_norm_history = [], [], [], []

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
print(f"Starting training for {args.epoch_training} epochs...")
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
        
        if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
            with torch.no_grad():
                dap = F.pairwise_distance(ao, po).mean().item()
                dan = F.pairwise_distance(ao, no).mean().item()
            print(f"Epoch [{epoch+1}/{args.epoch_training}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/(i+1):.4f}, d(a,p): {dap:.4f}, d(a,n): {dan:.4f}")

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

    print(f"Epoch [{epoch+1}] Val Loss: {avg_v:.4f}, Log-Ratio: {l_ratio:.4f}, Time: {time.time()-epoch_start:.2f}s")
    
    if avg_v < best_val_loss:
        best_val_loss = avg_v
        torch.save(model.state_dict(), base_save_path + '_best.ckpt')
        best_ap_dist, best_an_dist = c_ap, c_an
    
    scheduler.step()
    if epoch >= args.epoch_enforced_training and avg_v > 1.1 * prev_val_loss: break
    prev_val_loss = avg_v

def save_fig(fig, suffix):
    plt.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(base_save_path + suffix, dpi=300); plt.close(fig)

fig1, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].plot(train_losses, label='Train'); ax[0].plot(val_losses, label='Val'); ax[0].set_title('Loss Evolution'); ax[0].legend()
ax[1].plot(val_log_ratio_history, color='blue'); ax[1].set_title('Embedding Separation Quality'); ax[1].axhline(0, color='k', ls='--')
ax[2].plot(grad_norm_history, color='teal'); ax[2].set_title('Gradient Norm (Stability)'); ax[2].axhline(1.0, color='r', ls='--')
fig1.suptitle(f"Training Metrics | {cell_line}\n{param_title}"); save_fig(fig1, '_training_stats.pdf')

fig2 = plt.figure(figsize=(10, 7))
plt.hist(best_ap_dist, bins=50, alpha=0.6, label='Positives d(a,p)', color='g', density=True)
plt.hist(best_an_dist, bins=50, alpha=0.6, label='Negatives d(a,n)', color='r', density=True)
plt.title(f"Validation Set: Pairwise Distance Distribution | {cell_line}\n{param_title}"); plt.legend()
save_fig(fig2, '_val_dist_hist.pdf')

print(f"Training Complete. Total Time: {(time.time()-total_start_time)/60:.2f} mins")