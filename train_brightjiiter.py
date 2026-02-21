# Final Robust Version: Adagrad + Jitter + Gradient Clipping
# Strategy: High Accuracy, High Replicate Rate, Stable Training

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.transforms as T
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse, json, os, subprocess, sys, time
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
parser.add_argument('--no-mask', dest='mask', action='store_false', default=True, help='Disable diagonal masking (default: mask is on)')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--margin', type=float, default=0.5, help='Margin for triplet loss')
parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping max norm')
parser.add_argument('--adagrad_weight_decay', type=float, default=0.0, help='L2 weight decay for Adagrad')
parser.add_argument('--hard_mining', action='store_true', help='Only backprop on triplets that violate margin (hard examples)')
parser.add_argument('--run_eval', action='store_true', help='Run test.py evaluation after training (intersect, rep_rate, cond_rate)')
parser.add_argument('--threshold_data', type=str, default='train_val', choices=['val', 'train_val'],
                    help='Data for threshold (intersect) calibration: train_val=train+val (default); val=validation only. Used when --run_eval.')
parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'none'],
                    help='LR scheduler: plateau=ReduceLROnPlateau (default), cosine=CosineAnnealingLR, none=fixed LR')
parser.add_argument('--lr_patience', type=int, default=3, help='[plateau] Epochs without val improvement before reducing LR')
parser.add_argument('--lr_factor', type=float, default=0.5, help='[plateau] LR multiplier when reducing')
parser.add_argument('--min_lr', type=float, default=1e-6, help='[plateau/cosine] Minimum LR (eta_min for cosine)')
parser.add_argument("data_inputs", nargs='+', help="Keys for training and validation")

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ---------------------------------------------------------
# parameters
# ---------------------------------------------------------
_hard = "_hard" if args.hard_mining else ""
file_param_info = f"{args.model_name}_{args.learning_rate}_{args.batch_size}_{args.seed}_{args.margin}{_hard}"
base_save_path = os.path.join(args.outpath, file_param_info)

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
with open(args.json_file, encoding='utf-8') as f: dataset_config = json.load(f)

train_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(p) for p in dataset_config[n]["training"]], 
    reference=reference_genomes[dataset_config[n]["reference"]]) for n in args.data_inputs])

val_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(p) for p in dataset_config[n]["validation"]], 
    reference=reference_genomes[dataset_config[n]["reference"]]) for n in args.data_inputs])

num_train_triplets = len(train_dataset)
num_val_triplets = len(val_dataset)
print(f"num_train_triplets: {num_train_triplets:,}, num_val_triplets: {num_val_triplets:,}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=100, sampler=SequentialSampler(val_dataset), num_workers=4, pin_memory=True)

# ---------------------------------------------------------
# Model & Optimizer
# ---------------------------------------------------------
model = getattr(models, args.model_name)(mask=args.mask).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

# 使用 Hard Margin (0.5)
criterion = TripletLoss(margin=args.margin)

# 使用 Adagrad (適合稀疏特徵)
optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.adagrad_weight_decay)
if args.scheduler == 'plateau':
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
elif args.scheduler == 'cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch_training, eta_min=args.min_lr)
else:
    scheduler = None

# ---------------------------------------------------------
# Data Augmentation (Brightness Jitter)
# ---------------------------------------------------------
jitter_transform = T.ColorJitter(brightness=0.2, contrast=0.2)

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
best_val_loss = float('inf')
patience_counter = 0 
train_losses, val_losses, val_log_ratio_history, grad_norm_history, lr_history = [], [], [], [], []
best_ap_dist, best_an_dist = [], []

print(f"Starting training: {file_param_info}")
sched_str = {"plateau": "ReduceLROnPlateau", "cosine": "CosineAnnealingLR"}.get(args.scheduler, "")
print(f"Config: Adagrad + Jitter + Gradient Clipping (max_norm={args.max_norm})" + (" | Hard Mining" if args.hard_mining else "") + (f" | {sched_str}" if sched_str else ""))

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
            
            # Apply Jitter (Training Only)
            a = jitter_transform(a)
            p = jitter_transform(p)
            n = jitter_transform(n)

            optimizer.zero_grad()
            a_out, p_out, n_out = model(a, p, n)

            if args.hard_mining:
                d_ap = F.pairwise_distance(a_out, p_out)
                d_an = F.pairwise_distance(a_out, n_out)
                loss_val = F.relu(d_ap - d_an + args.margin)
                mask = loss_val > 1e-16
                loss = loss_val[mask].mean() if mask.any() else loss_val.mean()
            else:
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
            if args.scheduler == 'plateau':
                scheduler.step(avg_v)
            else:
                scheduler.step()

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
# Visualization (always run if we have any history)
# ---------------------------------------------------------
finally:
    def save_fig(fig, suffix):
        plt.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(base_save_path + suffix, dpi=300); plt.close(fig)

    if train_losses:
        fig1, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax[0, 0].plot(train_losses, label='Train'); ax[0, 0].plot(val_losses, label='Val'); ax[0, 0].set_title('Loss Evolution'); ax[0, 0].legend()
        ax[0, 1].plot(val_log_ratio_history, color='blue'); ax[0, 1].set_title('Log-Ratio (log(a_n/a_p))'); ax[0, 1].axhline(0, color='k', ls='--')
        ax[1, 0].plot(grad_norm_history, color='teal'); ax[1, 0].set_title('Gradient Norm'); ax[1, 0].axhline(args.max_norm, color='r', ls='--', label='Clip Threshold'); ax[1, 0].legend()
        ax[1, 1].semilogy(lr_history, color='green'); ax[1, 1].set_title('Learning Rate'); ax[1, 1].set_xlabel('Epoch')
        fig1.suptitle(f"Training Metrics | Model: {args.model_name}\nLR: {args.learning_rate} | Margin: {args.margin} | Jitter+Clip"); save_fig(fig1, '_training_stats.pdf')
        print(f"Saved: {base_save_path}_training_stats.pdf")

    if best_ap_dist and best_an_dist:
        fig2 = plt.figure(figsize=(10, 7))
        plt.hist(best_ap_dist, bins=50, alpha=0.6, label='Positives d(a,p)', color='g', density=True)
        plt.hist(best_an_dist, bins=50, alpha=0.6, label='Negatives d(a,n)', color='r', density=True)
        plt.title(f"Best Model Distance Distribution\n{file_param_info}"); plt.legend()
        save_fig(fig2, '_val_dist_hist.pdf')
        print(f"Saved: {base_save_path}_val_dist_hist.pdf")

    print(f"Training Complete. Total Time: {(time.time()-total_start_time)/60:.2f} mins")

    # Optional: run test.py for intersect & performance metrics
    if args.run_eval:
        ckpt_path = os.path.abspath(base_save_path + '_best.ckpt')
        if os.path.exists(ckpt_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            test_script = os.path.join(script_dir, 'test.py')
            json_path = os.path.abspath(args.json_file)
            config_dir = os.path.dirname(json_path) if os.path.dirname(json_path) else os.getcwd()
            cmd = [sys.executable, test_script, args.model_name, json_path, ckpt_path]
            cmd.extend(args.data_inputs)
            cmd.extend(['--threshold_data', args.threshold_data])
            print(f"\n--- Running evaluation (threshold_data={args.threshold_data}) ---")
            subprocess.run(cmd, cwd=config_dir)
        else:
            print(f"Skipping --run_eval: checkpoint not found ({ckpt_path})")