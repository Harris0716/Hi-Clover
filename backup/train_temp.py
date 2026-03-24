import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse, json, os, time
import matplotlib.pyplot as plt

# Project modules           
from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Triplet Hard Mining (Fast & Robust)')
parser.add_argument('model_name', type=str)
parser.add_argument('json_file', type=str)
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch_training', type=int, default=150)
parser.add_argument('--epoch_enforced_training', type=int, default=30)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--margin', type=float, default=0.5)
parser.add_argument('--max_norm', type=float, default=1.0)
parser.add_argument('--outpath', type=str, default="outputs/")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mask', type=str, default="true")
parser.add_argument('--amp', action='store_true',
                    help='Use mixed precision training (CUDA only)')
parser.add_argument("data_inputs", nargs='+')

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# Device & Acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True 

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
with open(args.json_file) as f: dataset_config = json.load(f)

train_sets, val_sets = [], []
for n in args.data_inputs:
    ref = reference_genomes[dataset_config[n]["reference"]]
    train_sets.append(TripletHiCDataset([HiCDatasetDec.load(p) for p in dataset_config[n]["training"]], reference=ref))
    val_sets.append(TripletHiCDataset([HiCDatasetDec.load(p) for p in dataset_config[n]["validation"]], reference=ref))

train_loader = DataLoader(GroupedTripletHiCDataset(train_sets), batch_size=args.batch_size, 
                          sampler=RandomSampler(GroupedTripletHiCDataset(train_sets)), 
                          num_workers=4, pin_memory=True)

val_loader = DataLoader(GroupedTripletHiCDataset(val_sets), batch_size=100, 
                        sampler=SequentialSampler(GroupedTripletHiCDataset(val_sets)), 
                        num_workers=4, pin_memory=True)

# ---------------------------------------------------------
# Model & Optimizer
# ---------------------------------------------------------
mask_bool = args.mask.lower() == 'true'
model = eval("models." + args.model_name)(mask=mask_bool).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

if args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    scheduler = None
else:
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # LR scheduler kept internal to avoid over-complicating CLI arguments
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# AMP support (CUDA only)
use_amp = args.amp and device.type == "cuda"
scaler = torch.amp.GradScaler("cuda") if use_amp else None

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
file_info = f"{args.model_name}_{args.optimizer}_HardMining_{args.learning_rate}_{args.margin}"
base_path = os.path.join(args.outpath, file_info)

best_val_loss = float('inf')
patience_cnt = 0
train_losses, val_losses = [], []
val_log_ratio_history, grad_norm_history, lr_history = [], [], []
best_ap_dist, best_an_dist = [], []

print(f"Start Training: {base_path}")
print(f"Config: Hard Mining (Margin={args.margin}), Unpack-Safe, Fast-Mode")
total_start_time = time.time()

try:
    for epoch in range(args.epoch_training):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        e_norms = []

        # [Safety] Use enumerate and avoid direct unpacking
        for i, batch_data in enumerate(train_loader):
            # [Safety] Explicitly take only the first three tensors, ignore any extra index/label
            a = batch_data[0].to(device, non_blocking=True)
            p = batch_data[1].to(device, non_blocking=True)
            n = batch_data[2].to(device, non_blocking=True)
            
            optimizer.zero_grad()

            if use_amp:
                # Mixed precision forward + loss
                with torch.amp.autocast("cuda"):
                    a_out, p_out, n_out = model(a, p, n)

                    # -------------------------------------------------
                    # Hard Mining Logic
                    # -------------------------------------------------
                    d_ap = F.pairwise_distance(a_out, p_out)
                    d_an = F.pairwise_distance(a_out, n_out)
                    
                    # 1. Compute raw loss
                    loss_val = F.relu(d_ap - d_an + args.margin)
                    
                    # 2. Build a mask: only keep samples with loss > 0 (hard examples)
                    # 1e-16 is for numerical stability
                    mask = loss_val > 1e-16
                    
                    # 3. Aggregate loss
                    if mask.any():
                        loss = loss_val[mask].mean()  # average over hard examples only
                    else:
                        loss = loss_val.mean()        # 0.0 when all triplets satisfy the margin
                    # -------------------------------------------------

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # Must unscale before clipping (clipping scaled grads kills updates)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                e_norms.append(grad_norm.item())
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 forward + loss
                a_out, p_out, n_out = model(a, p, n)

                # -------------------------------------------------
                # Hard Mining Logic
                # -------------------------------------------------
                d_ap = F.pairwise_distance(a_out, p_out)
                d_an = F.pairwise_distance(a_out, n_out)
                
                # 1. Compute raw loss
                loss_val = F.relu(d_ap - d_an + args.margin)
                
                # 2. Build a mask: only keep samples with loss > 0 (hard examples)
                # 1e-16 is for numerical stability
                mask = loss_val > 1e-16
                
                # 3. Aggregate loss
                if mask.any():
                    loss = loss_val[mask].mean()  # average over hard examples only
                else:
                    loss = loss_val.mean()        # 0.0 when all triplets satisfy the margin
                # -------------------------------------------------

                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                e_norms.append(grad_norm.item())
                optimizer.step()
            
            running_loss += loss.item()

            # Print every 100 batches so you know training is progressing
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epoch_training}] Step [{i+1}/{len(train_loader)}] "
                      f"Loss: {running_loss/(i+1):.4f}")

        # Validation
        model.eval()
        val_loss_sum = 0.0
        c_ap, c_an = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                # [Safety] Validation loader also uses safe unpacking
                a = batch_data[0].to(device, non_blocking=True)
                p = batch_data[1].to(device, non_blocking=True)
                n = batch_data[2].to(device, non_blocking=True)
                
                ao, po, no = model(a, p, n)
                
                # Validation uses standard mean loss over the batch
                d_ap_v = F.pairwise_distance(ao, po)
                d_an_v = F.pairwise_distance(ao, no)
                val_loss_sum += F.relu(d_ap_v - d_an_v + args.margin).mean().item()
                c_ap.extend(d_ap_v.cpu().numpy())
                c_an.extend(d_an_v.cpu().numpy())
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_ap, avg_an = np.mean(c_ap), np.mean(c_an)
        l_ratio = np.log10((avg_an + 1e-6) / (avg_ap + 1e-6))
        
        # LR Update
        curr_lr = optimizer.param_groups[0]['lr']
        if scheduler: scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_log_ratio_history.append(l_ratio)
        grad_norm_history.append(np.mean(e_norms))
        lr_history.append(curr_lr)

        print(f"Epoch [{epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Log-Ratio: {l_ratio:.4f} | LR: {curr_lr:.2e} | Time: {time.time()-epoch_start:.1f}s")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), base_path + '_best.ckpt')
            best_ap_dist, best_an_dist = list(c_ap), list(c_an)
        else:
            if epoch >= args.epoch_enforced_training:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"Early Stopping at Epoch {epoch+1}")
                    break

except KeyboardInterrupt:
    print("\nInterrupted.")

# ---------------------------------------------------------
# Visualization (runs even after early stop or Ctrl+C)
# ---------------------------------------------------------
finally:
    def save_fig(fig, suffix):
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(base_path + suffix, dpi=300)
        plt.close(fig)

    if train_losses:
        # Figure 1: four-panel monitoring plot (Loss, Log-Ratio, Gradient Norm, Learning Rate)
        fig1, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax[0, 0].plot(train_losses, label='Train')
        ax[0, 0].plot(val_losses, label='Val')
        ax[0, 0].set_title('Loss Evolution')
        ax[0, 0].legend()
        ax[0, 0].set_xlabel('Epoch')

        ax[0, 1].plot(val_log_ratio_history, color='blue')
        ax[0, 1].set_title('Embedding Separation (log₁₀ d(a,n)/d(a,p))')
        ax[0, 1].axhline(0, color='k', ls='--')
        ax[0, 1].set_xlabel('Epoch')

        ax[1, 0].plot(grad_norm_history, color='teal')
        ax[1, 0].set_title('Gradient Norm (Stability)')
        ax[1, 0].axhline(args.max_norm, color='r', ls='--')
        ax[1, 0].set_xlabel('Epoch')

        ax[1, 1].semilogy(lr_history, color='green')
        ax[1, 1].set_title('Learning Rate')
        ax[1, 1].set_xlabel('Epoch')

        fig1.suptitle(f"Hard Mining Training Metrics | {args.model_name} | Margin={args.margin}")
        save_fig(fig1, '_training_stats.pdf')
        print(f"Saved: {base_path}_training_stats.pdf")

        # Optional: keep a standalone loss curve figure
        fig_loss = plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.title(f"Loss Curve (Margin {args.margin})")
        plt.legend()
        plt.xlabel('Epoch')
        save_fig(fig_loss, '_loss.pdf')

    if best_ap_dist and best_an_dist:
        fig2 = plt.figure(figsize=(10, 7))
        plt.hist(best_ap_dist, bins=50, alpha=0.6, label='Positives d(a,p)', color='g', density=True)
        plt.hist(best_an_dist, bins=50, alpha=0.6, label='Negatives d(a,n)', color='r', density=True)
        plt.title(f"Best Model: Validation Distance Distribution | {file_info}")
        plt.legend()
        plt.xlabel('Distance')
        save_fig(fig2, '_val_dist_hist.pdf')
        print(f"Saved: {base_path}_val_dist_hist.pdf")

    print(f"Done. Total Time: {(time.time() - total_start_time) / 60:.2f} mins")