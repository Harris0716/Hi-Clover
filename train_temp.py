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

# ==========================================================================================
# [Class Definition] Online Hard Negative Mining Loss
# ==========================================================================================
class OnlineHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0, epsilon=1e-16):
        super(OnlineHardTripletLoss, self).__init__()
        self.margin = margin
        self.epsilon = epsilon

    def forward(self, anchor, positive, negative):
        # 1. 計算歐式距離 (Euclidean Distance)
        d_ap = F.pairwise_distance(anchor, positive, p=2)
        d_an = F.pairwise_distance(anchor, negative, p=2)

        # 2. 計算每個樣本的原始 Loss
        # F.relu 將滿足 Margin (簡單樣本) 的 Loss 歸零
        loss_per_sample = F.relu(d_ap - d_an + self.margin)

        # 3. 篩選困難樣本 (Hard Sample Selection)
        # mask 為布林值，標記哪些樣本是困難的 (Loss > 0)
        mask_hard = loss_per_sample > self.epsilon
        hard_loss = loss_per_sample[mask_hard]

        # 4. 計算最終 Loss (僅對困難樣本取平均)
        if hard_loss.numel() > 0:
            loss = hard_loss.mean()
        else:
            # 極少見情況：Batch 內所有樣本都已經完美分開
            loss = loss_per_sample.mean() # 實際上是 0.0

        # 計算統計資訊供監控
        hard_ratio = mask_hard.float().mean() # 該 Batch 中有多少比例是困難樣本
        
        return loss, hard_ratio, d_ap.mean(), d_an.mean()

# ==========================================================================================
# Argument Parser
# ==========================================================================================
parser = argparse.ArgumentParser(description='Triplet Network with Online Hard Negative Mining')
parser.add_argument('model_name', type=str, help='Model name defined in models.py')
parser.add_argument('json_file', type=str, help='Path to JSON config')
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adagrad', 'adamw'], help='Optimizer selection')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for AdamW')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epoch_training', type=int, default=150, help='Max training epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=30, help='Minimum training epochs')
parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
parser.add_argument('--margin', type=float, default=0.3, help='Margin for triplet loss')
parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping norm')
parser.add_argument('--lr_patience', type=int, default=5, help='LR scheduler patience')
parser.add_argument('--lr_factor', type=float, default=0.5, help='LR decay factor')
parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
parser.add_argument('--outpath', type=str, default="outputs/", help='Output directory')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--mask', type=str, default="true", help='Mask diagonal (true/false string)') # 修正布林值解析問題
parser.add_argument('--no_scheduler', action='store_true', help='Disable LR scheduler')
parser.add_argument("data_inputs", nargs='+', help="Data keys from JSON")

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# 處理 mask 參數 (argparse 傳遞布林值的小技巧)
mask_flag = args.mask.lower() == 'true'

# Device & Optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 開啟 CuDNN Benchmark 以加速固定尺寸輸入的卷積運算
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ==========================================================================================
# Parameters & Logging
# ==========================================================================================
_sch = "_nosch" if (args.optimizer == "adamw" and args.no_scheduler) else ""
file_param_info = f"{args.model_name}_{args.optimizer}{_sch}_HardMining_{args.learning_rate}_{args.batch_size}_{args.seed}_{args.margin}"
base_save_path = os.path.join(args.outpath, file_param_info)

# ==========================================================================================
# Data Loading
# ==========================================================================================
with open(args.json_file) as f: dataset_config = json.load(f)

# 建構 Dataset
train_datasets = []
val_datasets = []
for n in args.data_inputs:
    # Training Data
    train_hic_list = [HiCDatasetDec.load(p) for p in dataset_config[n]["training"]]
    train_datasets.append(TripletHiCDataset(train_hic_list, reference=reference_genomes[dataset_config[n]["reference"]]))
    # Validation Data
    val_hic_list = [HiCDatasetDec.load(p) for p in dataset_config[n]["validation"]]
    val_datasets.append(TripletHiCDataset(val_hic_list, reference=reference_genomes[dataset_config[n]["reference"]]))

train_dataset = GroupedTripletHiCDataset(train_datasets)
val_dataset = GroupedTripletHiCDataset(val_datasets)

print(f"Num Train Triplets: {len(train_dataset):,}") 
print(f"Num Val Triplets: {len(val_dataset):,}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=100, sampler=SequentialSampler(val_dataset), num_workers=4, pin_memory=True)

# ==========================================================================================
# Model, Loss & Optimizer
# ==========================================================================================
model = eval("models." + args.model_name)(mask=mask_flag).to(device)
if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

# 初始化自定義 Loss
criterion_train = OnlineHardTripletLoss(margin=args.margin).to(device)
# 驗證時通常使用標準平均 Loss 以評估整體分佈，但也可以參考 Hard Ratio
criterion_val = TripletLoss(margin=args.margin).to(device) # 這是您原本的 Loss 函數

if args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    scheduler = None
else:
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None if args.no_scheduler else ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)

# ==========================================================================================
# Training Loop
# ==========================================================================================
best_val_loss = float('inf')
patience_counter = 0
history = {
    'train_loss': [], 'val_loss': [], 'log_ratio': [], 
    'grad_norm': [], 'lr': [], 'hard_ratio': []
}
best_dist = {'ap': [], 'an': []}

print(f"Starting Training: {file_param_info}")
print(f"Config: Hard Mining enabled, Margin={args.margin}")
total_start_time = time.time()

try:
    for epoch in range(args.epoch_training):
        epoch_start = time.time()
        model.train()
        
        # Accumulators
        run_loss = 0.0
        run_hard_ratio = 0.0
        run_d_ap = 0.0
        run_d_an = 0.0
        grad_norms = []
        
        num_batches = len(train_loader)
        
        for i, (a, p, n) in enumerate(train_loader):
            a, p, n = a.to(device), p.to(device), n.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            a_out, p_out, n_out = model(a, p, n)
            
            # Loss Calculation (Using Hard Mining Class)
            loss, hard_ratio, d_ap_mean, d_an_mean = criterion_train(a_out, p_out, n_out)
            
            loss.backward()
            
            # Gradient Clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            grad_norms.append(grad_norm.item())
            
            optimizer.step()
            
            # Statistics
            run_loss += loss.item()
            run_hard_ratio += hard_ratio.item()
            run_d_ap += d_ap_mean.item()
            run_d_an += d_an_mean.item()

            # Logging every 100 steps
            if (i + 1) % 100 == 0 or (i + 1) == num_batches:
                avg_loss = run_loss / (i + 1)
                avg_hr = (run_hard_ratio / (i + 1)) * 100 # Percentage
                print(f"Epoch [{epoch+1}/{args.epoch_training}] Step [{i+1}/{num_batches}] | "
                      f"Loss: {avg_loss:.4f} | Hard%: {avg_hr:.1f}% | "
                      f"d(a,p): {run_d_ap/(i+1):.3f} | d(a,n): {run_d_an/(i+1):.3f}")

        # ================= Validation =================
        model.eval()
        val_loss_accum = 0.0
        c_ap, c_an = [], []
        
        with torch.no_grad():
            for a, p, n in val_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                ao, po, no = model(a, p, n)
                
                # 計算標準驗證 Loss (平均)
                val_loss_accum += criterion_val(ao, po, no).item()
                
                # 收集距離分佈
                c_ap.extend(F.pairwise_distance(ao, po).cpu().numpy())
                c_an.extend(F.pairwise_distance(ao, no).cpu().numpy())
        
        # Metrics Calculation
        avg_val_loss = val_loss_accum / len(val_loader)
        avg_ap = np.mean(c_ap)
        avg_an = np.mean(c_an)
        log_ratio = np.log10((avg_an + 1e-6) / (avg_ap + 1e-6))
        
        # Current LR
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler: scheduler.step(avg_val_loss)

        # Update History
        history['train_loss'].append(run_loss / num_batches)
        history['val_loss'].append(avg_val_loss)
        history['log_ratio'].append(log_ratio)
        history['grad_norm'].append(np.mean(grad_norms))
        history['lr'].append(current_lr)
        history['hard_ratio'].append(run_hard_ratio / num_batches)

        # Print Epoch Summary
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}] Result: Val Loss: {avg_val_loss:.4f} | Log-Ratio: {log_ratio:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        print("-" * 80)

        # Checkpoint & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), base_save_path + '_best.ckpt')
            best_dist['ap'] = c_ap
            best_dist['an'] = c_an
        else:
            if epoch >= args.epoch_enforced_training:
                patience_counter += 1
                print(f"Early Stopping Counter: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at Epoch {epoch+1}")
                    break

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

# ==========================================================================================
# Visualization
# ==========================================================================================
finally:
    def save_plot(fig, name):
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = base_save_path + name
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved plot: {path}")

    if history['train_loss']:
        # Plot 1: Loss & Log-Ratio
        fig1, ax = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        ax[0, 0].plot(history['train_loss'], label='Train (Hard Mining)')
        ax[0, 0].plot(history['val_loss'], label='Val (Standard)')
        ax[0, 0].set_title('Loss History')
        ax[0, 0].set_xlabel('Epoch')
        ax[0, 0].legend()
        ax[0, 0].grid(True, alpha=0.3)
        
        # Log Ratio
        ax[0, 1].plot(history['log_ratio'], color='blue')
        ax[0, 1].set_title('Log-Ratio (Separation Index)')
        ax[0, 1].axhline(0, color='k', linestyle='--')
        ax[0, 1].grid(True, alpha=0.3)
        
        # Hard Ratio & Grad Norm
        ax[1, 0].plot(history['hard_ratio'], color='orange')
        ax[1, 0].set_title('Hard Sample Ratio (Difficulty)')
        ax[1, 0].set_ylim(0, 1.0)
        ax[1, 0].set_ylabel('Ratio (0-1)')
        ax[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        ax[1, 1].semilogy(history['lr'], color='green')
        ax[1, 1].set_title('Learning Rate')
        ax[1, 1].grid(True, alpha=0.3)
        
        fig1.suptitle(f"Training Metrics | {args.model_name} | Hard Mining (m={args.margin})")
        save_plot(fig1, '_metrics.pdf')

    if best_dist['ap'] and best_dist['an']:
        # Plot 2: Distance Distribution
        fig2 = plt.figure(figsize=(10, 7))
        plt.hist(best_dist['ap'], bins=50, alpha=0.6, label='Positive Dist', color='g', density=True)
        plt.hist(best_dist['an'], bins=50, alpha=0.6, label='Negative Dist', color='r', density=True)
        plt.axvline(args.margin, color='k', linestyle='--', label=f'Margin ({args.margin})')
        plt.title(f"Distance Distribution (Best Model)\n{file_param_info}")
        plt.xlabel('Euclidean Distance')
        plt.legend()
        save_plot(fig2, '_dist_hist.pdf')

    total_time = (time.time() - total_start_time) / 60
    print(f"Experiment finished in {total_time:.2f} minutes.")