# All setting are same as Twins but using Triplet Network (baseline)
# Add patience mechnism
# hard margin triplet loss
# Adagrad
# [Modified] Fixed GPU tensor error, added Gradient Clipping, NO Scheduler
# [Modified] Added AdamW optimizer and CosineAnnealingLR scheduler support
# [CHANGED] Joint loss now uses pair-wise BCE instead of single-sample CrossEntropy
# [CHANGED] pair_classifier uses separate optimizer (ref: Twins design)
# [FIXED v3] val_losses now consistently tracks triplet-only loss (same as early stopping criterion)
# [FIXED v4] Forward/backward correctly inside per-batch loop (indentation bug resolved)
# [FIXED v4] BCE graph isolated from backbone via .detach() — no retain_graph needed
# [FIXED v4] Gradient clipping applied to both backbone and clf separately
# [CHANGED v5] semi_hard now does TRUE batch-internal semi-hard mining
#              (negative is selected from the whole batch, not just filtered)
#              class label inferred as (1 - a_lbl) for binary condition setup
# [ADDED v6] --soft_margin flag: use SoftMarginTripletLoss (softplus, no hard cutoff)
# [ADDED v7] --batch_hard flag: always select hardest negative (skip semi-hard window)
# [CHANGED v8] True joint loss: BCE updates both backbone and pair_classifier.
#              Use a single optimizer for all model parameters.
#              No .detach() in pair-wise BCE branch.
# [ADDED v8] Save training history as .npz and mark best validation epoch in loss plot.
# [ADDED v9] Reproducibility controls: deterministic mode, seeded DataLoader workers, and seeded sampler.

import os
# Required by PyTorch for deterministic CUDA matmul on some CUDA versions when deterministic mode is enabled.
# It is harmless if deterministic mode is not used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import argparse, json, time
import matplotlib.pyplot as plt

from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
from torch_plus.loss import TripletLoss, SoftMarginTripletLoss
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Triplet network (v8 true joint, single optimizer)')
parser.add_argument('--model_name', type=str, help='Model from models.py')
parser.add_argument('--json_file', type=str, help='JSON dictionary with file paths')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epoch_training', type=int, default=100, help='Max epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=20, help='Enforced epochs')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--deterministic', action='store_true',
                    help='Enable deterministic PyTorch/cuDNN settings for reproducible GPU experiments')
parser.add_argument('--deterministic_warn_only', action='store_true',
                    help='When --deterministic is enabled, warn instead of error if a non-deterministic op is used')
parser.add_argument('--mask', action='store_true', help='Mask diagonal')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss / semi-hard window')
parser.add_argument('--soft_margin', action='store_true',
                    help='Use SoftMarginTripletLoss (softplus, no hard cutoff). Ignores --margin for loss.')
parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping max norm')
parser.add_argument('--scheduler', type=str, default='none',
                    choices=['plateau', 'cosine', 'none'],
                    help='LR scheduler')
parser.add_argument('--lr_patience', type=int, default=3)
parser.add_argument('--lr_factor', type=float, default=0.5)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--T_max', type=int, default=50)
parser.add_argument('--eta_min', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--semi_hard', action='store_true',
                    help='Batch-internal semi-hard negative mining')
parser.add_argument('--batch_hard', action='store_true',
                    help='Batch-internal hardest negative mining (always pick closest diff-class). '
                         'Takes priority over --semi_hard if both are set.')
parser.add_argument('--jitter_brightness', type=float, default=0.0)
parser.add_argument('--jitter_contrast', type=float, default=0.0)
parser.add_argument('--anti_diag_flip', action='store_true')
parser.add_argument('--h_flip', action='store_true')
parser.add_argument('--random_flip', action='store_true')
parser.add_argument('--optimizer', type=str, default='adagrad',
                    choices=['adagrad', 'adamw'])
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--joint_loss', action='store_true')
parser.add_argument('--ce_weight', type=float, default=0.5)
parser.add_argument('--clf_lr', type=float, default=1e-3,
                    help='Kept for compatibility. Ignored in v8 single-optimizer true joint mode.')
parser.add_argument('--outpath', type=str, default="outputs/")
parser.add_argument('--run_name', type=str, default=None,
                    help='Short output file prefix. If set, output files become <run_name>_best.ckpt, <run_name>_history.npz, etc.')
parser.add_argument("--data_inputs", nargs='+')

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# batch_hard 與 semi_hard 都需要做 batch-internal mining；batch_hard 優先
do_mining = args.semi_hard or args.batch_hard

print("-" * 50)
print("Command Line Arguments")
for key, value in vars(args).items():
    print(f"  {key}: {value}")
print("-" * 50)

def seed_everything(seed, deterministic=False, warn_only=False):
    """Set all practical random seeds used in this training script.

    Note: PYTHONHASHSEED is most effective when also set before launching Python:
      PYTHONHASHSEED=42 python train_reproducible.py ...
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        print("Deterministic mode: ON")
        print(f"  CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
        print("  cuDNN benchmark=False, deterministic=True, TF32 disabled")
    else:
        print("Deterministic mode: OFF (seed is still fixed)")


def seed_worker(worker_id):
    """Make DataLoader workers deterministic."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


seed_everything(args.seed, deterministic=args.deterministic, warn_only=args.deterministic_warn_only)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------
# File naming
# ---------------------------------------------------------
def _safe_num(x):
    """Make a short, filesystem-friendly numeric tag."""
    s = f"{x:g}"
    return s.replace("-", "m").replace(".", "p")


def _short_model_name(name):
    """Shorten common model names to keep output filenames readable."""
    mapping = {
        "TripletLeNetBatchNormSE": "SE",
        "TripletLeNetBatchNormSE_Joint": "SEJoint",
        "TripletLeNetBatchNormSE_Dilated": "SEDilated",
        "TripletLeNetBatchNormSE_Dilated_Joint": "SEDilatedJoint",
    }
    if name in mapping:
        return mapping[name]
    return (name.replace("Triplet", "")
                .replace("LeNet", "LN")
                .replace("BatchNorm", "BN"))


def _build_short_run_name(args):
    """Build a concise default filename prefix.

    The output directory already records the full experiment context, so the
    filename only keeps important identifiers. Use --run_name to override this.
    """
    data_tag = "-".join(args.data_inputs) if args.data_inputs else "data"
    model_tag = _short_model_name(args.model_name)

    method = []
    if args.h_flip:
        method.append("hflip")
    if args.random_flip:
        method.append("rflip")
    if args.anti_diag_flip:
        method.append("adflip")
    if args.batch_hard:
        method.append("batchhard")
    elif args.semi_hard:
        method.append("semihard")
    if args.mask:
        method.append("mask")

    if args.joint_loss:
        method.append(f"TJce{_safe_num(args.ce_weight)}")
    else:
        method.append("nojoint")

    loss_tag = "softmargin" if args.soft_margin else f"m{_safe_num(args.margin)}"
    opt_tag = f"{args.optimizer}-{args.scheduler}"
    lr_tag = f"lr{_safe_num(args.learning_rate)}"
    seed_tag = f"s{args.seed}"

    return "_".join([data_tag, model_tag] + method + [loss_tag, opt_tag, lr_tag, seed_tag])


file_param_info = args.run_name if args.run_name else _build_short_run_name(args)
base_save_path = os.path.join(args.outpath, file_param_info)
print(f"Output file prefix: {file_param_info}")

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
with open(args.json_file) as f:
    dataset_config = json.load(f)

train_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset(
        [HiCDatasetDec.load(p) for p in dataset_config[n]["training"]],
        reference=reference_genomes[dataset_config[n]["reference"]]
    ) for n in args.data_inputs],
    h_flip=args.h_flip, random_flip=args.random_flip)

val_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset(
        [HiCDatasetDec.load(p) for p in dataset_config[n]["validation"]],
        reference=reference_genomes[dataset_config[n]["reference"]]
    ) for n in args.data_inputs])

print(f"num_train_triplets: {len(train_dataset):,}")
print(f"num_val_triplets:   {len(val_dataset):,}")

# Seeded sampler + seeded DataLoader workers keep shuffling and worker-side randomness reproducible.
# Use different generators so that sampler order and worker base seeds do not consume the same RNG stream.
train_sampler = RandomSampler(train_dataset, generator=make_generator(args.seed + 101))
val_sampler   = SequentialSampler(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          sampler=train_sampler,
                          num_workers=args.num_workers, pin_memory=True,
                          worker_init_fn=seed_worker,
                          generator=make_generator(args.seed + 202))
val_loader   = DataLoader(val_dataset, batch_size=100,
                          sampler=val_sampler,
                          num_workers=args.num_workers, pin_memory=True,
                          worker_init_fn=seed_worker,
                          generator=make_generator(args.seed + 303))

# ---------------------------------------------------------
# Model & Optimizer
# ---------------------------------------------------------
model = eval("models." + args.model_name)(
    mask=args.mask, embedding_dim=args.embedding_dim).to(device)
if torch.cuda.device_count() > 1:
    if args.deterministic:
        print("Warning: multi-GPU DataParallel may still introduce small run-to-run differences.")
        print("         For strict reproducibility tests, prefer CUDA_VISIBLE_DEVICES=<one_gpu>.")
    model = nn.DataParallel(model)

if args.soft_margin:
    criterion = SoftMarginTripletLoss()
    print("Loss: SoftMarginTripletLoss (softplus, no hard cutoff; --margin ignored for loss)")
else:
    criterion = TripletLoss(margin=args.margin)
    print(f"Loss: TripletLoss (hard margin = {args.margin})")

if args.batch_hard:
    print("Mining: batch-hard (always select hardest diff-class negative)")
elif args.semi_hard:
    print(f"Mining: semi-hard (window = {args.margin})")
else:
    print("Mining: none (use dataset-paired negative)")

if args.joint_loss:
    backbone_params = [p for n, p in model.named_parameters() if 'pair_classifier' not in n]
    clf_params      = [p for n, p in model.named_parameters() if 'pair_classifier' in n]
else:
    backbone_params = list(model.parameters())
    clf_params      = []

def make_optimizer(params, lr):
    if args.optimizer == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    return optim.Adagrad(params, lr=lr, weight_decay=args.weight_decay)

# v8 true joint uses a single optimizer for all model parameters.
# When --joint_loss is enabled, BCE gradients are allowed to flow through
# pair_classifier back into the embedding backbone.
optimizer     = make_optimizer(list(model.parameters()), args.learning_rate)
scheduler     = None
ce_criterion  = None
pair_clf      = None

if args.joint_loss:
    ce_criterion  = nn.BCEWithLogitsLoss()
    pair_clf      = model.module.pair_classifier if hasattr(model, 'module') else model.pair_classifier
    print(f"True joint loss enabled: triplet_loss + {args.ce_weight} * BCE_loss (pair-wise)")
    print(f"Single optimizer lr: {args.learning_rate}; --clf_lr is ignored in this version.")

if args.scheduler == 'plateau':
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor,
                                  patience=args.lr_patience, min_lr=args.min_lr)
elif args.scheduler == 'cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

jitter_transform = None
if args.jitter_brightness > 0 or args.jitter_contrast > 0:
    jitter_transform = T.RandomApply([
        T.ColorJitter(brightness=args.jitter_brightness, contrast=args.jitter_contrast)], p=0.5)

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
sched_label = {"plateau": "ReduceLROnPlateau",
               "cosine":  "CosineAnnealingLR",
               "none":    "None"}[args.scheduler]

best_val_loss    = float('inf')
best_epoch       = -1
patience_counter = 0
train_losses, val_losses                         = [], []
train_bce_losses, val_bce_losses                 = [], []
train_total_losses, val_total_losses             = [], []
val_log_ratio_history                            = []
grad_norm_backbone_history, grad_norm_clf_history = [], []
lr_history                                       = []
best_ap_dist, best_an_dist                       = [], []

total_start_time   = time.time()
accumulation_steps = args.accumulation_steps

try:
    for epoch in range(args.epoch_training):
        epoch_start          = time.time()
        model.train()
        running_triplet_loss = 0.0
        running_bce_loss     = 0.0
        e_norms_backbone     = []
        e_norms_clf          = []
        semi_hard_count      = 0   # 找得到 semi-hard negative 的 anchor 數
        fallback_count       = 0   # 找不到 semi-hard、改用 hardest negative 的 anchor 數
        total_sample_count   = 0

        optimizer.zero_grad()

        # ── Per-batch loop ────────────────────────────────────────
        for i, data in enumerate(train_loader):
            a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
            a_lbl   = data[3].to(device)   # anchor 的 class id（二元：0 / 1）

            if args.anti_diag_flip:
                flip_mask = torch.rand(a.size(0), device=device) > 0.5
                if flip_mask.any():
                    a[flip_mask] = a[flip_mask].transpose(-2, -1).flip(-2, -1)
                    p[flip_mask] = p[flip_mask].transpose(-2, -1).flip(-2, -1)
                    n[flip_mask] = n[flip_mask].transpose(-2, -1).flip(-2, -1)

            if jitter_transform is not None:
                a, p, n = jitter_transform(a), jitter_transform(p), jitter_transform(n)

            # ── Forward ──────────────────────────────────────────
            a_out, p_out, n_out = model(a, p, n)

            # ── Batch-internal Mining (semi-hard or batch-hard) ───
            if do_mining:
                B = a.size(0)

                # Build the gradient-carrying pool outside no_grad.
                # Selection is done with detached embeddings, but the selected
                # negative used in triplet loss still keeps gradient.
                pool_emb_grad = torch.cat([a_out, p_out, n_out], dim=0)       # [3B, D]

                with torch.no_grad():
                    pool_emb_select = pool_emb_grad.detach()
                    pool_lbl = torch.cat([a_lbl, a_lbl, 1 - a_lbl], dim=0)    # [3B]

                    d_ap     = F.pairwise_distance(a_out, p_out).detach()    # [B]
                    dist_mat = torch.cdist(a_out.detach(), pool_emb_select)  # [B, 3B]

                    # Candidate negatives must be from a different class.
                    diff_class = pool_lbl.unsqueeze(0) != a_lbl.unsqueeze(1) # [B, 3B]

                    large = torch.tensor(float('inf'), device=device)

                    # Hardest negative: nearest different-class sample.
                    hard_dist = torch.where(diff_class, dist_mat, large)
                    _, hard_min_idx = hard_dist.min(dim=1)

                    if args.batch_hard:
                        selected_idx = hard_min_idx
                        semi_hard_count    += 0
                        fallback_count     += B
                        total_sample_count += B
                    else:
                        # semi-hard: d(a,p) < d(a,n) < d(a,p) + margin
                        sh_cond = (dist_mat > d_ap.unsqueeze(1)) & \
                                  (dist_mat < (d_ap + args.margin).unsqueeze(1))
                        sh_mask = diff_class & sh_cond
                        sh_dist = torch.where(sh_mask, dist_mat, large)
                        sh_min_d, sh_min_idx = sh_dist.min(dim=1)
                        has_sh = torch.isfinite(sh_min_d)

                        selected_idx = torch.where(has_sh, sh_min_idx, hard_min_idx)

                        semi_hard_count    += has_sh.sum().item()
                        fallback_count     += (~has_sh).sum().item()
                        total_sample_count += B

                selected_neg_emb = pool_emb_grad[selected_idx]               # [B, D]
                triplet_loss = criterion(a_out, p_out, selected_neg_emb)
                a_u, p_u, n_u = a_out, p_out, selected_neg_emb
            else:
                B = a.size(0)
                triplet_loss = criterion(a_out, p_out, n_out)
                a_u, p_u, n_u = a_out, p_out, n_out

            # ── Backward ─────────────────────────────────────────
            if args.joint_loss:
                # v8 true joint: do NOT detach embeddings.
                # BCE loss updates both pair_classifier and embedding backbone.
                feat_ap  = torch.abs(a_u - p_u)
                feat_an  = torch.abs(a_u - n_u)
                logit_ap = pair_clf(feat_ap).squeeze(1)
                logit_an = pair_clf(feat_an).squeeze(1)
                logits   = torch.cat([logit_ap, logit_an], dim=0)
                bce_labels = torch.cat([
                    torch.zeros(B, device=device),
                    torch.ones(B,  device=device)
                ]).float()
                bce_loss = ce_criterion(logits, bce_labels)

                total_loss = triplet_loss + args.ce_weight * bce_loss
                (total_loss / accumulation_steps).backward()

                running_triplet_loss += triplet_loss.item()
                running_bce_loss     += bce_loss.item()
            else:
                (triplet_loss / accumulation_steps).backward()
                running_triplet_loss += triplet_loss.item()

            # ── Gradient update after accumulation ───────────────
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                grad_norm_b = nn.utils.clip_grad_norm_(backbone_params, max_norm=args.max_norm)
                e_norms_backbone.append(grad_norm_b.item())

                if args.joint_loss and len(clf_params) > 0:
                    grad_norm_c = nn.utils.clip_grad_norm_(clf_params, max_norm=args.max_norm)
                    e_norms_clf.append(grad_norm_c.item())

                optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                d_ap_log = F.pairwise_distance(a_out, p_out).mean().item()
                d_an_log = F.pairwise_distance(a_out, n_u).mean().item()
                print(f"Epoch [{epoch+1}/{args.epoch_training}], "
                      f"Step [{i+1}/{len(train_loader)}], "
                      f"Triplet: {running_triplet_loss/(i+1):.4f}, "
                      f"d(a,p): {d_ap_log:.4f}, d(a,n): {d_an_log:.4f}")

        # ── Validation ───────────────────────────────────────────
        # 驗證階段使用 dataset 預配的固定 triplet，不做 mining，
        # 以確保 early-stopping 的指標在所有設定下一致可比。
        model.eval()
        val_triplet_sum = 0.0
        val_bce_sum     = 0.0
        c_ap, c_an      = [], []

        with torch.no_grad():
            for data in val_loader:
                a, p, n = data[0].to(device), data[1].to(device), data[2].to(device)
                ao, po, no = model(a, p, n)

                vt = criterion(ao, po, no)
                val_triplet_sum += vt.item()

                if args.joint_loss:
                    B       = a.size(0)
                    f_ap    = torch.abs(ao - po)
                    f_an    = torch.abs(ao - no)
                    l_ap    = pair_clf(f_ap).squeeze(1)
                    l_an    = pair_clf(f_an).squeeze(1)
                    lg      = torch.cat([l_ap, l_an], dim=0)
                    bl      = torch.cat([torch.zeros(B, device=device),
                                         torch.ones(B,  device=device)]).float()
                    val_bce_sum += ce_criterion(lg, bl).item()

                c_ap.extend(F.pairwise_distance(ao, po).cpu().numpy())
                c_an.extend(F.pairwise_distance(ao, no).cpu().numpy())

        avg_triplet = val_triplet_sum / len(val_loader)
        avg_bce     = val_bce_sum     / len(val_loader) if args.joint_loss else 0.0
        l_ratio     = np.log10((np.mean(c_an) + 1e-6) / (np.mean(c_ap) + 1e-6))

        # val_losses = triplet only, consistent with early stopping
        avg_train_triplet = running_triplet_loss / len(train_loader)
        avg_train_bce     = running_bce_loss / len(train_loader) if args.joint_loss else 0.0
        avg_train_total   = avg_train_triplet + args.ce_weight * avg_train_bce if args.joint_loss else avg_train_triplet
        avg_val_total     = avg_triplet + args.ce_weight * avg_bce if args.joint_loss else avg_triplet

        train_losses.append(avg_train_triplet)
        val_losses.append(avg_triplet)
        train_bce_losses.append(avg_train_bce)
        val_bce_losses.append(avg_bce)
        train_total_losses.append(avg_train_total)
        val_total_losses.append(avg_val_total)
        val_log_ratio_history.append(l_ratio)
        grad_norm_backbone_history.append(np.mean(e_norms_backbone) if e_norms_backbone else 0.0)
        grad_norm_clf_history.append(np.mean(e_norms_clf) if e_norms_clf else 0.0)

        lr_before = optimizer.param_groups[0]['lr']
        lr_history.append(lr_before)

        if scheduler is not None:
            scheduler.step(avg_triplet) if args.scheduler == 'plateau' else scheduler.step()
            cur_lr = optimizer.param_groups[0]['lr']
            if cur_lr < lr_before:
                print(f"-> LR changed {lr_before:.2e} -> {cur_lr:.2e}")

        bce_str = f", Val BCE: {avg_bce:.4f}" if args.joint_loss else ""
        lr_str  = f", LR: {lr_before:.2e}" if scheduler else ""
        print(f"Epoch [{epoch+1}] Val Triplet: {avg_triplet:.4f}{bce_str}, "
              f"Log-Ratio: {l_ratio:.4f}, "
              f"Time: {time.time()-epoch_start:.2f}s{lr_str}")

        if do_mining and not args.batch_hard:
            sh_r = semi_hard_count / max(total_sample_count, 1)
            print(f"  Semi-hard hit rate: {sh_r:.4f} "
                  f"({semi_hard_count}/{total_sample_count}), "
                  f"Fallback(hardest): {fallback_count}")
        elif args.batch_hard:
            print(f"  Batch-hard mining: all {total_sample_count} anchors used hardest negative")

        # Early stopping based on triplet loss only
        if avg_triplet < best_val_loss:
            best_val_loss    = avg_triplet
            best_epoch       = epoch
            patience_counter = 0
            current_date     = time.strftime("%Y%m%d")
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
    print("\nTraining interrupted. Plotting...")

# ---------------------------------------------------------
# Save Training History
# ---------------------------------------------------------
history_path = base_save_path + "_history.npz"
np.savez_compressed(
    history_path,
    train_losses=np.array(train_losses),
    val_losses=np.array(val_losses),
    train_bce_losses=np.array(train_bce_losses),
    val_bce_losses=np.array(val_bce_losses),
    train_total_losses=np.array(train_total_losses),
    val_total_losses=np.array(val_total_losses),
    val_log_ratio_history=np.array(val_log_ratio_history),
    grad_norm_backbone_history=np.array(grad_norm_backbone_history),
    grad_norm_clf_history=np.array(grad_norm_clf_history),
    lr_history=np.array(lr_history),
    best_epoch=np.array(best_epoch),
    best_val_loss=np.array(best_val_loss),
    ce_weight=np.array(args.ce_weight),
    joint_loss=np.array(args.joint_loss),
    true_joint_single_optimizer=np.array(True),
    deterministic=np.array(args.deterministic),
    deterministic_warn_only=np.array(args.deterministic_warn_only),
    cublas_workspace_config=np.array(os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")),
    run_name=np.array(file_param_info),
)
print(f"Training history saved: {history_path}")

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def save_fig(fig, suffix):
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(base_save_path + suffix, dpi=300)
    plt.close(fig)

aug_features = []
if args.anti_diag_flip: aug_features.append("AntiDiag")
if args.h_flip:         aug_features.append("HFlip")
if args.batch_hard:     aug_features.append("BatchHard")
elif args.semi_hard:    aug_features.append("SemiHard")
if args.joint_loss:     aug_features.append(f"JointBCE(λ={args.ce_weight})")
aug_str   = f" | {'+'.join(aug_features)}" if aug_features else ""
sched_str = f" | {sched_label}" if scheduler else ""
loss_str  = "SoftMargin" if args.soft_margin else f"Margin: {args.margin}"
info_text = (f"Opt: {args.optimizer.upper()} | LR: {args.learning_rate} | "
             f"WD: {args.weight_decay} | {loss_str} | "
             f"Batch: {args.batch_size}{aug_str}{sched_str}")

has_lr_change = lr_history and len(set(lr_history)) > 1
n_cols = 5 if (args.joint_loss and has_lr_change) else \
         4 if (args.joint_loss or has_lr_change) else 3

fig1, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
col = 0

axes[col].plot(train_losses, label='Train Triplet')
axes[col].plot(val_losses,   label='Val Triplet')
if len(val_losses) > 0 and best_epoch >= 0:
    axes[col].axvline(best_epoch, color='red', ls='--', linewidth=2,
                      label=f'Best Epoch ({best_epoch + 1})')
    axes[col].scatter(best_epoch, val_losses[best_epoch], color='red', s=60, zorder=5)
    axes[col].set_title(f'Triplet Loss (early-stopping metric)\n'
                        f'Best Val={best_val_loss:.4f} @ Epoch {best_epoch + 1}')
else:
    axes[col].set_title('Triplet Loss (early-stopping metric)')
axes[col].legend()
col += 1

if args.joint_loss:
    axes[col].plot(val_bce_losses, color='purple', label='Val BCE')
    axes[col].set_title('Val BCE Loss (diagnostic)')
    axes[col].legend()
    col += 1

axes[col].plot(val_log_ratio_history, color='blue')
axes[col].set_title('Log-Ratio log(d(a,n)/d(a,p))')
axes[col].axhline(0, color='k', ls='--')
col += 1

axes[col].plot(grad_norm_backbone_history, color='teal',   label='Backbone')
if args.joint_loss:
    axes[col].plot(grad_norm_clf_history,  color='orange', label='Classifier')
axes[col].set_title('Gradient Norm (after clipping)')
axes[col].axhline(args.max_norm, color='r', ls='--')
axes[col].legend()
col += 1

if has_lr_change:
    axes[col].plot(lr_history, color='orange')
    axes[col].set_title('Learning Rate')
    axes[col].set_yscale('log')

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