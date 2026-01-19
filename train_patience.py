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
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str)
parser.add_argument('json_file', type=str)
parser.add_argument('learning_rate', type=float)
parser.add_argument("data_inputs", nargs='+')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch_training', type=int, default=100)
parser.add_argument('--epoch_enforced_training', type=int, default=20)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--outpath', type=str, default="outputs/")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mask', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=1e-4)

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed); np.random.seed(args.seed)

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

criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_training)

best_val_loss, patience_counter = float('inf'), 0
train_losses, val_losses, grad_norm_history = [], [], []

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
for epoch in range(args.epoch_training):
    model.train()
    running_loss, e_norms = 0.0, []
    for data in train_loader:
        a, p, n = [d.to(device) for d in data]
        optimizer.zero_grad()
        ao, po, no = [F.normalize(o, p=2, dim=1) for o in model(a, p, n)]

        # 核心優化：Batch Hard Spatial Loss
        dist_mat = torch.cdist(ao, ao, p=2)
        dist_mat.fill_diagonal_(float('inf'))
        # 0.1(條件區分) : 0.9(空間對齊) 權重
        loss = 0.1 * criterion(ao, po, no) + 0.9 * criterion(ao, po, ao[dist_mat.argmin(dim=1)])

        loss.backward()
        e_norms.append(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    v_loss, c_ap, c_an = 0.0, [], []
    with torch.no_grad():
        for data in val_loader:
            a, p, n = [d.to(device) for d in data]
            ao, po, no = [F.normalize(o, p=2, dim=1) for o in model(a, p, n)]
            
            dist_mat = torch.cdist(ao, ao, p=2)
            dist_mat.fill_diagonal_(float('inf'))
            v_loss += (0.1 * criterion(ao, po, no) + 0.9 * criterion(ao, po, ao[dist_mat.argmin(dim=1)])).item()
            c_ap.extend(F.pairwise_distance(ao, po).cpu().numpy())
            c_an.extend(F.pairwise_distance(ao, no).cpu().numpy())
    
    avg_v = v_loss / len(val_loader)
    train_losses.append(running_loss / len(train_loader)); val_losses.append(avg_v)
    grad_norm_history.append(np.mean(e_norms))

    print(f"Epoch [{epoch+1}] Val Loss: {avg_v:.4f}")
    
    if avg_v < best_val_loss:
        best_val_loss, patience_counter = avg_v, 0
        torch.save(model.state_dict(), base_save_path + '_best.ckpt')
        best_ap_dist, best_an_dist = c_ap, c_an
    elif epoch >= args.epoch_enforced_training:
        patience_counter += 1
    
    scheduler.step()
    if patience_counter >= args.patience: break

# 繪圖
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(train_losses, label='Train'); ax[0].plot(val_losses, label='Val'); ax[0].legend()
ax[1].hist(best_ap_dist, bins=50, alpha=0.5, label='d(a,p)'); ax[1].hist(best_an_dist, bins=50, alpha=0.5, label='d(a,n)'); ax[1].legend()
plt.savefig(base_save_path + '_stats.pdf'); plt.close()