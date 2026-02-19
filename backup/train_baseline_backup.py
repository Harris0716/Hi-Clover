# All setting are same as Twins but using Triplet Network (baseline)
# Add patience mechanism; optional AdamW + ReduceLROnPlateau (Adagrad: fixed LR only)
# hard margin triplet loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
parser = argparse.ArgumentParser(description='Triplet network (v1 logic with fixed naming)')
parser.add_argument('model_name', type=str, help='Model from models.py')
parser.add_argument('json_file', type=str, help='JSON dictionary with file paths')
parser.add_argument('--optimizer', type=str, default='adagrad', choices=['adagrad', 'adamw'],
                    help='adagrad: fixed LR. adamw: ReduceLROnPlateau adapts LR by val loss.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='LR (fixed for Adagrad; initial for AdamW then scheduler)')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (AdamW only)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epoch_training', type=int, default=100, help='Max epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=20, help='Enforced epochs')
parser.add_argument('--outpath', type=str, default="outputs/", help='Output directory')
parser.add_argument('--seed', type=int, default=30004, help='Random seed')
parser.add_argument('--mask', type=bool, default=False, help='Mask diagonal')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping (stop training)')
parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping max norm')
parser.add_argument('--lr_patience', type=int, default=4, help='Epochs without val improvement before reducing LR (AdamW only)')
parser.add_argument('--lr_factor', type=float, default=0.5, help='LR multiplier when reducing (AdamW only)')
parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum LR (AdamW only)')
parser.add_argument('--no_scheduler', action='store_true', help='AdamW only: use fixed LR (no ReduceLROnPlateau). Can reduce overfitting when val loss diverges from train.')
parser.add_argument('--adagrad_weight_decay', type=float, default=0.0, help='L2 weight decay for Adagrad (optional; try 1e-5 or 1e-4 for light regularization).')
parser.add_argument('--amp', action='store_true', help='Use mixed precision training (CUDA only); can speed up both Adagrad and AdamW.')
parser.add_argument("data_inputs", nargs='+', help="Keys for training and validation")

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ---------------------------------------------------------
# parameters
# ---------------------------------------------------------
_sch = "_nosch" if (args.optimizer == "adamw" and args.no_scheduler) else ""
file_param_info = f"{args.model_name}_{args.optimizer}{_sch}_{args.learning_rate}_{args.batch_size}_{args.seed}_{args.margin}"
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

# 計算生成的triplets數量
num_train_triplets = len(train_dataset)
num_val_triplets = len(val_dataset)
print(f"num_train_triplets: {num_train_triplets:,}") 
print(f"num_val_triplets: {num_val_triplets:,}") 
print(f"total_num_triplets: {num_train_triplets + num_val_triplets:,}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=100, sampler=SequentialSampler(val_dataset), num_workers=4, pin_memory=True)

# ---------------------------------------------------------
# Model & Optimizer
# ---------------------------------------------------------
model = eval("models." + args.model_name)(mask=args.mask).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

criterion = TripletLoss(margin=args.margin)
if args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.adagrad_weight_decay)
    scheduler = None
else:
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None if args.no_scheduler else ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)

# AMP (CUDA only)
use_amp = getattr(args, 'amp', False) and device.type == "cuda"
scaler = torch.amp.GradScaler("cuda") if use_amp else None

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses, val_log_ratio_history, grad_norm_history, lr_history = [], [], [], [], []
best_ap_dist, best_an_dist = [], []

print(f"Starting training: {file_param_info}")
total_start_time = time.time()

try:
    for epoch in range(args.epoch_training):
        epoch_start = time.time()
        model.train()
        running_loss, e_norms = 0.0, []
        
        for i, data in enumerate(train_loader):
            a = data[0].to(device, non_blocking=True)
            p = data[1].to(device, non_blocking=True)
            n = data[2].to(device, non_blocking=True)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    a_out, p_out, n_out = model(a, p, n)
                    loss = criterion(a_out, p_out, n_out)
                scaler.scale(loss).backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                e_norms.append(grad_norm.item())
                scaler.step(optimizer)
                scaler.update()
            else:
                a_out, p_out, n_out = model(a, p, n)
                loss = criterion(a_out, p_out, n_out)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                e_norms.append(grad_norm.item())
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
                a = data[0].to(device, non_blocking=True)
                p = data[1].to(device, non_blocking=True)
                n = data[2].to(device, non_blocking=True)
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
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        if scheduler is not None:
            scheduler.step(avg_v)

        lr_str = f", LR: {current_lr:.2e}" if scheduler else ""
        print(f"Epoch [{epoch+1}] Val Loss: {avg_v:.4f}, Log-Ratio: {l_ratio:.4f}{lr_str}, Time: {time.time()-epoch_start:.2f}s")

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            patience_counter = 0
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state, base_save_path + '_best.ckpt')
            best_ap_dist, best_an_dist = list(c_ap), list(c_an)
        else:
            if epoch >= args.epoch_enforced_training:
                patience_counter += 1
                print(f"-> No improvement. Patience: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
except KeyboardInterrupt:
    print("\nInterrupted by user (Ctrl+C).")

# ---------------------------------------------------------
# Visualization (always run if we have any history, including after early stop or Ctrl+C)
# ---------------------------------------------------------
finally:
    def save_fig(fig, suffix):
        plt.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(base_save_path + suffix, dpi=300); plt.close(fig)

    if train_losses:
        # Figure 1: Training Stats (Loss, Log-Ratio, Grad Norm, LR)
        fig1, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax[0, 0].plot(train_losses, label='Train'); ax[0, 0].plot(val_losses, label='Val'); ax[0, 0].set_title('Loss Evolution'); ax[0, 0].legend()
        ax[0, 1].plot(val_log_ratio_history, color='blue'); ax[0, 1].set_title('Log-Ratio (log(a_n/a_p))'); ax[0, 1].axhline(0, color='k', ls='--')
        ax[1, 0].plot(grad_norm_history, color='teal'); ax[1, 0].set_title('Gradient Norm'); ax[1, 0].axhline(args.max_norm, color='r', ls='--')
        ax[1, 1].semilogy(lr_history, color='green'); ax[1, 1].set_title('Learning Rate'); ax[1, 1].set_xlabel('Epoch')
        fig1.suptitle(f"Training Metrics | Model: {args.model_name} | {args.optimizer.upper()}\nLR: {args.learning_rate} | Margin: {args.margin}"); save_fig(fig1, '_training_stats.pdf')
        print(f"Saved: {base_save_path}_training_stats.pdf")

    if best_ap_dist and best_an_dist:
        fig2 = plt.figure(figsize=(10, 7))
        plt.hist(best_ap_dist, bins=50, alpha=0.6, label='Positives d(a,p)', color='g', density=True)
        plt.hist(best_an_dist, bins=50, alpha=0.6, label='Negatives d(a,n)', color='r', density=True)
        plt.title(f"Best Model Distance Distribution\n{file_param_info}"); plt.legend()
        save_fig(fig2, '_val_dist_hist.pdf')
        print(f"Saved: {base_save_path}_val_dist_hist.pdf")

    print(f"Training Complete. Total Time: {(time.time()-total_start_time)/60:.2f} mins")