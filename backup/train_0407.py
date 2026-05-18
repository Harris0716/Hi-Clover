# All setting are same as Twins but using Triplet Network (baseline)
# Add patience mechnism
# hard margin triplet loss
# Adagrad
# [Modified] Fixed GPU tensor error, added Gradient Clipping, NO Scheduler
# [Modified] Added AdamW optimizer and CosineAnnealingLR scheduler support
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import argparse, json, os, time
import matplotlib.pyplot as plt

from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
from torch_plus.loss import TripletLoss
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Triplet network (v1 logic with fixed naming)')
parser.add_argument('--model_name', type=str, help='Model from models.py')
parser.add_argument('--json_file', type=str, help='JSON dictionary with file paths')
parser.add_argument('--learning_rate', type=float,default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epoch_training', type=int, default=100, help='Max epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=20, help='Enforced epochs')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--mask', action='store_true', help='Mask diagonal')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping max norm')
parser.add_argument('--scheduler', type=str, default='none', choices=['plateau', 'cosine', 'none'], help='LR scheduler: plateau=ReduceLROnPlateau, cosine=CosineAnnealingLR, none=fixed LR (default)')
parser.add_argument('--lr_patience', type=int, default=3, help='[plateau] Epochs without val improvement before reducing LR')
parser.add_argument('--lr_factor', type=float, default=0.5, help='[plateau] LR multiplier when reducing')
parser.add_argument('--min_lr', type=float, default=1e-6, help='[plateau] Minimum LR')
parser.add_argument('--T_max', type=int, default=50, help='[cosine] CosineAnnealingLR T_max (half cycle length)')
parser.add_argument('--eta_min', type=float, default=1e-6, help='[cosine] Minimum LR for cosine scheduler')
parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 weight decay for optimizer (e.g. 1e-4, 1e-3 to reduce overfitting)')
parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients before updating weights')
parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers (use 0 to avoid shm error with large batch)')
parser.add_argument('--semi_hard', action='store_true', help='Use Semi-Hard Negative Mining (dist_ap < dist_an < dist_ap + margin)')
parser.add_argument('--jitter_brightness', type=float, default=0.0, help='ColorJitter brightness (0=off, e.g. 0.2 for augmentation)')
parser.add_argument('--jitter_contrast', type=float, default=0.0, help='ColorJitter contrast (0=off, e.g. 0.2 for augmentation)')
parser.add_argument('--anti_diag_flip', action='store_true', help='Anti-Diagonal Flip')
parser.add_argument('--h_flip', action='store_true', help='Horizontal Flip') 
parser.add_argument('--optimizer', type=str, default='adagrad', choices=['adagrad', 'adamw'], help='Optimizer choice: adagrad or adamw')
parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
parser.add_argument('--outpath', type=str, default="outputs/", help='Output directory')
parser.add_argument("--data_inputs", nargs='+', help="Keys for training and validation")

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# ---------------------------------------------------------
# Print command-line arguments to log
# ---------------------------------------------------------
print("-" * 50)
print("Command Line Arguments")  
for key, value in vars(args).items():
    print(f"  {key}: {value}")
print("-" * 50)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ---------------------------------------------------------
# parameters
# ---------------------------------------------------------
# file_param_info = f"{args.model_name}_{args.learning_rate}_{args.batch_size}_{args.seed}_{args.margin}"
file_param_info = f"{args.model_name}_{args.optimizer}_{args.scheduler}_{args.learning_rate}_{args.batch_size}_{args.seed}_{args.margin}"
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

# calculate the number of triplets generated
num_train_triplets = len(train_dataset)
num_val_triplets = len(val_dataset)
print(f"num_train_triplets: {num_train_triplets:,}") 
print(f"num_val_triplets: {num_val_triplets:,}") 
print(f"total_num_triplets: {num_train_triplets + num_val_triplets:,}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=100, sampler=SequentialSampler(val_dataset), num_workers=args.num_workers, pin_memory=True)

# ---------------------------------------------------------
# Model & Optimizer
# ---------------------------------------------------------
model = eval("models." + args.model_name)(mask=args.mask, embedding_dim=args.embedding_dim).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

criterion = TripletLoss(margin=args.margin)
if args.optimizer == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
else:
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

scheduler = None
if args.scheduler == 'plateau':
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
elif args.scheduler == 'cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

# [Modified] Jitter wrapped with RandomApply with 50% trigger probability
use_jitter = args.jitter_brightness > 0 or args.jitter_contrast > 0
if use_jitter:
    base_jitter = T.ColorJitter(brightness=args.jitter_brightness or 0.0, contrast=args.jitter_contrast or 0.0)
    jitter_transform = T.RandomApply([base_jitter], p=0.5)
else:
    jitter_transform = None

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
best_val_loss = float('inf')  
patience_counter = 0 
train_losses, val_losses, val_log_ratio_history, grad_norm_history, lr_history = [], [], [], [], []
best_ap_dist, best_an_dist = [], []

wd_str = f" | weight_decay={args.weight_decay}" if args.weight_decay > 0 else ""
semi_str = " | Semi-Hard Mining: ON" if args.semi_hard else ""
sched_label = {"plateau": "ReduceLROnPlateau", "cosine": "CosineAnnealingLR", "none": "None"}[args.scheduler]
print(f"Starting training: {file_param_info} | Optimizer: {args.optimizer.upper()} | Scheduler: {sched_label}" + wd_str + semi_str)
total_start_time = time.time()

accumulation_steps = args.accumulation_steps

try:
    for epoch in range(args.epoch_training):
        epoch_start = time.time()
        model.train()
        running_loss, e_norms = 0.0, []
        semi_hard_count = 0
        total_sample_count = 0
        fallback_count = 0
        
        optimizer.zero_grad()
        
        for i, data in enumerate(train_loader):
            a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
            
            # [New] 50% chance to synchronously flip the triplet along the anti-diagonal (y = -x)
            if args.anti_diag_flip:
                flip_mask = torch.rand(a.size(0), device=device) > 0.5
                if flip_mask.any():
                    # Flip matrix along y = -x: equivalent to transpose then flip height and width
                    a[flip_mask] = a[flip_mask].transpose(-2, -1).flip(-2, -1)
                    p[flip_mask] = p[flip_mask].transpose(-2, -1).flip(-2, -1)
                    n[flip_mask] = n[flip_mask].transpose(-2, -1).flip(-2, -1)

            # [New] 50% chance to apply random horizontal flip
            if args.h_flip:
                h_flip_mask = torch.rand(a.size(0), device=device) > 0.5
                if h_flip_mask.any():
                    a[h_flip_mask] = a[h_flip_mask].flip(-1)
                    p[h_flip_mask] = p[h_flip_mask].flip(-1)
                    n[h_flip_mask] = n[h_flip_mask].flip(-1)

            if jitter_transform is not None:
                a, p, n = jitter_transform(a), jitter_transform(p), jitter_transform(n)
            
            # 1. Forward pass
            a_out, p_out, n_out = model(a, p, n)
            
            # [New] Semi-Hard Mining logic
            if args.semi_hard:
                with torch.no_grad():
                    # Compute Euclidean distances
                    d_ap_batch = F.pairwise_distance(a_out, p_out)
                    d_an_batch = F.pairwise_distance(a_out, n_out)
                    # Selection condition: d(a,p) < d(a,n) < d(a,p) + margin
                    sh_mask = (d_an_batch > d_ap_batch) & (d_an_batch < d_ap_batch + args.margin)
                
                # If there are samples satisfying the condition in this batch, compute loss only on them
                total_sample_count += a.size(0)
                if sh_mask.any():
                    semi_hard_count += sh_mask.sum().item()
                    loss = criterion(a_out[sh_mask], p_out[sh_mask], n_out[sh_mask])
                else:
                    # fallback: use full batch when no semi-hard samples found
                    fallback_count += 1
                    loss = criterion(a_out, p_out, n_out)
            else:
                # Full-batch loss
                loss = criterion(a_out, p_out, n_out)
            
            # 2. Normalize and backpropagate
            loss = loss / accumulation_steps
            loss.backward()
            
            # 3. Accumulated gradient update
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                e_norms.append(grad_norm.item()) 
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                # During monitoring, show current sample distance statistics
                d_ap = F.pairwise_distance(a_out, p_out).mean().item()
                d_an = F.pairwise_distance(a_out, n_out).mean().item()
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
        lr_before = optimizer.param_groups[0]['lr']
        lr_history.append(lr_before)

        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(avg_v)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < lr_before:
                print(f"-> LR changed {lr_before:.2e} -> {current_lr:.2e}")

        lr_str = f", LR: {lr_before:.2e}" if scheduler else ""
        print(f"Epoch [{epoch+1}] Val Loss: {avg_v:.4f}, Log-Ratio: {l_ratio:.4f}, Time: {time.time()-epoch_start:.2f}s{lr_str}")
        if args.semi_hard:
            sh_ratio = semi_hard_count / max(total_sample_count, 1)
            print(f"  Semi-hard ratio: {sh_ratio:.4f} ({semi_hard_count}/{total_sample_count}), Fallback batches: {fallback_count}")
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            patience_counter = 0
            
            current_date = time.strftime("%Y%m%d")
            torch.save(model.state_dict(), f"{base_save_path}_{current_date}_best.ckpt")
            
            best_ap_dist, best_an_dist = c_ap, c_an
        else:
            if epoch >= args.epoch_enforced_training:
                patience_counter += 1
                print(f"-> No improvement. Patience: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
except KeyboardInterrupt:
    print("\nTraining interrupted by user (Ctrl+C). Plotting loss curves...")

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def save_fig(fig, suffix):
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(base_save_path + suffix, dpi=300)
    plt.close(fig)

# Aggregate detailed parameter information for figure titles  
aug_features = []
if args.anti_diag_flip: aug_features.append("AntiDiag")
if args.h_flip: aug_features.append("HFlip")
if args.semi_hard: aug_features.append("SemiHard")
aug_str = f" | {'+'.join(aug_features)}" if aug_features else ""
sched_str = f" | {sched_label}" if scheduler else ""

info_text = (f"Opt: {args.optimizer.upper()} | LR: {args.learning_rate} | WD: {args.weight_decay} | "
             f"Margin: {args.margin} | Batch: {args.batch_size}{aug_str}{sched_str}")

# Figure 1: Training Stats
n_plots = 4 if lr_history and len(set(lr_history)) > 1 else 3
fig1, ax = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
ax[0].plot(train_losses, label='Train')
ax[0].plot(val_losses, label='Val')
ax[0].set_title('Loss Evolution')
ax[0].legend()

ax[1].plot(val_log_ratio_history, color='blue')
ax[1].set_title('Log-Ratio (log(a_n/a_p))')
ax[1].axhline(0, color='k', ls='--')

ax[2].plot(grad_norm_history, color='teal')
ax[2].set_title('Gradient Norm')
ax[2].axhline(args.max_norm, color='r', ls='--')

if n_plots == 4:
    ax[3].plot(lr_history, color='orange')
    ax[3].set_title('Learning Rate')
    ax[3].set_yscale('log')

fig1.suptitle(f"Training Metrics | Model: {args.model_name}\n{info_text}")
save_fig(fig1, '_training_stats.pdf')

# Figure 2: Distance Distribution (skip if interrupted before first best model)
if best_ap_dist and best_an_dist:
    fig2 = plt.figure(figsize=(10, 7))
    plt.hist(best_ap_dist, bins=50, alpha=0.6, label='Positives d(a,p)', color='g', density=True)
    plt.hist(best_an_dist, bins=50, alpha=0.6, label='Negatives d(a,n)', color='r', density=True)
    plt.title(f"Best Model Distance Distribution\nModel: {args.model_name}\n{info_text}")
    plt.legend()
    save_fig(fig2, '_val_dist_hist.pdf')

print(f"Training Complete. Total Time: {(time.time()-total_start_time)/60:.2f} mins")