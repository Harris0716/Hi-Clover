# All setting are same as Twins but using Triplet Network (baseline)
# Add patience mechnism
# hard margin triplet loss
# Adagrad
# [Modified] Fixed GPU tensor error, added Gradient Clipping, NO Scheduler
# [Modified] Added AdamW optimizer and CosineAnnealingLR scheduler support
# [CHANGED] Joint loss now uses pair-wise BCE instead of single-sample CrossEntropy
# [CHANGED] pair_classifier uses separate optimizer (ref: Twins design)
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
parser.add_argument('--random_flip', action='store_true', help='Random horizontal flip (50% prob) instead of 2x augmentation')
parser.add_argument('--optimizer', type=str, default='adagrad', choices=['adagrad', 'adamw'], help='Optimizer choice: adagrad or adamw')
parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
parser.add_argument('--joint_loss', action='store_true', help='Use joint triplet + BCE loss (pair-wise)')
parser.add_argument('--ce_weight', type=float, default=0.5, help='Weight for BCE loss in joint loss (lambda)')
# [CHANGED] 新增 clf_lr，pair_classifier 使用獨立 learning rate
parser.add_argument('--clf_lr', type=float, default=1e-3, help='Learning rate for pair_classifier (independent optimizer)')
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
aug_tag = []
if args.h_flip:          aug_tag.append("hflip")
if args.random_flip:     aug_tag.append("rflip")
if args.anti_diag_flip:  aug_tag.append("adflip")
if args.semi_hard:       aug_tag.append("semihard")
if args.mask:            aug_tag.append("mask")
aug_str_tag = "_".join(aug_tag) if aug_tag else "noaug"

joint_tag = f"joint{args.ce_weight}" if args.joint_loss else "nojoint"

jitter_tag = ""
if args.jitter_brightness > 0 or args.jitter_contrast > 0:
    jitter_tag = f"_jit{args.jitter_brightness}c{args.jitter_contrast}"

file_param_info = (
    f"{args.model_name}"
    f"_{args.optimizer}"
    f"_{args.scheduler}"
    f"_lr{args.learning_rate}"
    f"_bs{args.batch_size}"
    f"_wd{args.weight_decay}"
    f"_emb{args.embedding_dim}"
    f"_margin{args.margin}"
    f"_acc{args.accumulation_steps}"
    f"_pat{args.patience}"
    f"_maxnorm{args.max_norm}"
    f"_seed{args.seed}"
    f"_{aug_str_tag}"
    f"_{joint_tag}"
    f"{jitter_tag}"
)
base_save_path = os.path.join(args.outpath, file_param_info)

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
with open(args.json_file) as f: dataset_config = json.load(f)

train_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(p) for p in dataset_config[n]["training"]],
    reference=reference_genomes[dataset_config[n]["reference"]]) for n in args.data_inputs],
    h_flip=args.h_flip,
    random_flip=args.random_flip)

# validation 不做翻轉
val_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(p) for p in dataset_config[n]["validation"]],
    reference=reference_genomes[dataset_config[n]["reference"]]) for n in args.data_inputs])

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

# [CHANGED] 把 backbone 和 pair_classifier 的參數分開
# backbone optimizer：只更新非 pair_classifier 的參數
# clf optimizer：只更新 pair_classifier 的參數，使用獨立的 learning rate
if args.joint_loss:
    backbone_params = [p for n, p in model.named_parameters() if 'pair_classifier' not in n]
    clf_params = [p for n, p in model.named_parameters() if 'pair_classifier' in n]
else:
    backbone_params = list(model.parameters())
    clf_params = []

if args.optimizer == 'adamw':
    optimizer = optim.AdamW(backbone_params, lr=args.learning_rate, weight_decay=args.weight_decay)
else:
    optimizer = optim.Adagrad(backbone_params, lr=args.learning_rate, weight_decay=args.weight_decay)

# [CHANGED] Joint loss 初始化：pair_classifier 使用獨立 optimizer
scheduler = None
ce_criterion = None
pair_clf = None
optimizer_clf = None
if args.joint_loss:
    ce_criterion = nn.BCEWithLogitsLoss()
    pair_clf = model.module.pair_classifier if hasattr(model, 'module') else model.pair_classifier
    if args.optimizer == 'adamw':
        optimizer_clf = optim.AdamW(clf_params, lr=args.clf_lr, weight_decay=args.weight_decay)
    else:
        optimizer_clf = optim.Adagrad(clf_params, lr=args.clf_lr, weight_decay=args.weight_decay)
    print(f"Joint loss enabled: triplet_loss + {args.ce_weight} * BCE_loss (pair-wise)")
    print(f"Backbone lr: {args.learning_rate}, Classifier lr: {args.clf_lr}")

if args.scheduler == 'plateau':
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
elif args.scheduler == 'cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

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
joint_str = f" | Joint BCE (λ={args.ce_weight}, clf_lr={args.clf_lr})" if args.joint_loss else ""
sched_label = {"plateau": "ReduceLROnPlateau", "cosine": "CosineAnnealingLR", "none": "None"}[args.scheduler]
print(f"Starting training: {file_param_info} | Optimizer: {args.optimizer.upper()} | Scheduler: {sched_label}" + wd_str + semi_str + joint_str)
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
        if optimizer_clf is not None:
            optimizer_clf.zero_grad()

        for i, data in enumerate(train_loader):
            a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)

            if args.anti_diag_flip:
                flip_mask = torch.rand(a.size(0), device=device) > 0.5
                if flip_mask.any():
                    a[flip_mask] = a[flip_mask].transpose(-2, -1).flip(-2, -1)
                    p[flip_mask] = p[flip_mask].transpose(-2, -1).flip(-2, -1)
                    n[flip_mask] = n[flip_mask].transpose(-2, -1).flip(-2, -1)

            if jitter_transform is not None:
                a, p, n = jitter_transform(a), jitter_transform(p), jitter_transform(n)

            # 1. Forward pass
            a_out, p_out, n_out = model(a, p, n)

            # Semi-Hard Mining
            if args.semi_hard:
                with torch.no_grad():
                    d_ap_batch = F.pairwise_distance(a_out, p_out)
                    d_an_batch = F.pairwise_distance(a_out, n_out)
                    sh_mask = (d_an_batch > d_ap_batch) & (d_an_batch < d_ap_batch + args.margin)

                total_sample_count += a.size(0)
                if sh_mask.any():
                    semi_hard_count += sh_mask.sum().item()
                    triplet_loss = criterion(a_out[sh_mask], p_out[sh_mask], n_out[sh_mask])
                else:
                    fallback_count += 1
                    triplet_loss = criterion(a_out, p_out, n_out)
            else:
                triplet_loss = criterion(a_out, p_out, n_out)

            # Joint loss: pair-wise BCE
            if args.joint_loss:
                if args.semi_hard and sh_mask.any():
                    a_out_used = a_out[sh_mask]
                    p_out_used = p_out[sh_mask]
                    n_out_used = n_out[sh_mask]
                    B = sh_mask.sum().item()
                else:
                    a_out_used = a_out
                    p_out_used = p_out
                    n_out_used = n_out
                    B = a.size(0)

                feat_ap = torch.abs(a_out_used - p_out_used)
                feat_an = torch.abs(a_out_used - n_out_used)

                logit_ap = pair_clf(feat_ap).squeeze(1)
                logit_an = pair_clf(feat_an).squeeze(1)

                logits = torch.cat([logit_ap, logit_an], dim=0)
                bce_labels = torch.cat([
                    torch.zeros(B, device=device),
                    torch.ones(B, device=device)
                ]).float()

                bce_loss = ce_criterion(logits, bce_labels)
                loss = triplet_loss + args.ce_weight * bce_loss
            else:
                loss = triplet_loss

            # 2. Backpropagate
            loss = loss / accumulation_steps
            loss.backward()

            # 3. Accumulated gradient update
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                # [CHANGED] backbone 只用 triplet 梯度更新，clip 只針對 backbone
                grad_norm = nn.utils.clip_grad_norm_(backbone_params, max_norm=args.max_norm)
                e_norms.append(grad_norm.item())
                optimizer.step()
                optimizer.zero_grad()
                # [CHANGED] pair_clf 單獨更新，不影響 backbone
                if optimizer_clf is not None:
                    optimizer_clf.step()
                    optimizer_clf.zero_grad()

            running_loss += loss.item() * accumulation_steps

            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                d_ap = F.pairwise_distance(a_out, p_out).mean().item()
                d_an = F.pairwise_distance(a_out, n_out).mean().item()
                print(f"Epoch [{epoch+1}/{args.epoch_training}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/(i+1):.4f}, d(a,p): {d_ap:.4f}, d(a,n): {d_an:.4f}")

        # Validation Phase
        # [CHANGED] early stopping 只看 val triplet loss，與 Twins 設計一致
        model.eval()
        val_loss_sum, val_triplet_sum, c_ap, c_an = 0.0, 0.0, [], []
        with torch.no_grad():
            for data in val_loader:
                a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
                ao, po, no = model(a, p, n)
                val_triplet_loss = criterion(ao, po, no)
                val_triplet_sum += val_triplet_loss.item()

                if args.joint_loss:
                    B = a.size(0)
                    feat_ap = torch.abs(ao - po)
                    feat_an = torch.abs(ao - no)
                    logit_ap = pair_clf(feat_ap).squeeze(1)
                    logit_an = pair_clf(feat_an).squeeze(1)
                    logits = torch.cat([logit_ap, logit_an], dim=0)
                    bce_labels = torch.cat([
                        torch.zeros(B, device=device),
                        torch.ones(B, device=device)
                    ]).float()
                    val_bce_loss = ce_criterion(logits, bce_labels)
                    val_loss_sum += (val_triplet_loss + args.ce_weight * val_bce_loss).item()
                else:
                    val_loss_sum += val_triplet_loss.item()

                c_ap.extend(F.pairwise_distance(ao, po).cpu().numpy())
                c_an.extend(F.pairwise_distance(ao, no).cpu().numpy())

        avg_v = val_loss_sum / len(val_loader)
        avg_triplet = val_triplet_sum / len(val_loader)
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
                scheduler.step(avg_triplet)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < lr_before:
                print(f"-> LR changed {lr_before:.2e} -> {current_lr:.2e}")

        lr_str = f", LR: {lr_before:.2e}" if scheduler else ""
        print(f"Epoch [{epoch+1}] Val Loss (joint): {avg_v:.4f}, Val Triplet: {avg_triplet:.4f}, Log-Ratio: {l_ratio:.4f}, Time: {time.time()-epoch_start:.2f}s{lr_str}")
        if args.semi_hard:
            sh_ratio = semi_hard_count / max(total_sample_count, 1)
            print(f"  Semi-hard ratio: {sh_ratio:.4f} ({semi_hard_count}/{total_sample_count}), Fallback batches: {fallback_count}")

        # [CHANGED] early stopping 用 avg_triplet，與 Twins 設計一致
        if avg_triplet < best_val_loss:
            best_val_loss = avg_triplet
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

aug_features = []
if args.anti_diag_flip: aug_features.append("AntiDiag")
if args.h_flip: aug_features.append("HFlip")
if args.semi_hard: aug_features.append("SemiHard")
if args.joint_loss: aug_features.append(f"JointBCE(λ={args.ce_weight})")
aug_str = f" | {'+'.join(aug_features)}" if aug_features else ""
sched_str = f" | {sched_label}" if scheduler else ""

info_text = (f"Opt: {args.optimizer.upper()} | LR: {args.learning_rate} | WD: {args.weight_decay} | "
             f"Margin: {args.margin} | Batch: {args.batch_size}{aug_str}{sched_str}")

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

if best_ap_dist and best_an_dist:
    fig2 = plt.figure(figsize=(10, 7))
    plt.hist(best_ap_dist, bins=50, alpha=0.6, label='Positives d(a,p)', color='g', density=True)
    plt.hist(best_an_dist, bins=50, alpha=0.6, label='Negatives d(a,n)', color='r', density=True)
    plt.title(f"Best Model Distance Distribution\nModel: {args.model_name}\n{info_text}")
    plt.legend()
    save_fig(fig2, '_val_dist_hist.pdf')

print(f"Training Complete. Total Time: {(time.time()-total_start_time)/60:.2f} mins")