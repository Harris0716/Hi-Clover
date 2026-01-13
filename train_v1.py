# baseline code (Fixed Version)
# early stopping logic -> 1.1 * prev_val_loss
# plot loss curve
# Adagrad

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

# 導入 Dataset 與模型定義
from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
from torch_plus.loss import TripletLoss
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Triplet network (v1 logic with fixed naming)')
parser.add_argument('model_name', type=str, help='a string indicating a model from models')
parser.add_argument('json_file', type=str, help='a file location for the json dictionary containing file paths')
parser.add_argument('learning_rate', type=float, help='a float for the learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='an int for batch size')
parser.add_argument('--epoch_training', type=int, default=100, help='max epochs')
parser.add_argument('--epoch_enforced_training', type=int, default=20, help='enforced epochs before early stop')
parser.add_argument('--outpath', type=str, default="outputs/", help='a path for the output directory')
parser.add_argument('--seed', type=int, default=30004, help='an int for the seed')
parser.add_argument('--mask', type=bool, default=False, help='specifying if the diagonal should be masked')
parser.add_argument('--margin', type=float, default=1.0, help='margin for triplet loss')
parser.add_argument("data_inputs", nargs='+', help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.device_count() if torch.cuda.is_available() else 'CPU'} device.")

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ---------------------------------------------------------
# parameters
# ---------------------------------------------------------
file_param_info = f"{args.model_name}_{args.learning_rate}_{args.batch_size}_{args.seed}_{args.margin}"
base_save_path = os.path.join(args.outpath, file_param_info)

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
with open(args.json_file) as f:
    dataset_config = json.load(f)

print("Loading Datasets...")
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

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=100, sampler=SequentialSampler(val_dataset), num_workers=4, pin_memory=True)

no_of_batches = len(train_loader)
batches_validation = len(val_loader)

# ---------------------------------------------------------
# Model & Optimizer
# ---------------------------------------------------------
model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

torch.save(model.state_dict(), base_save_path + '_initial.ckpt')

criterion = TripletLoss(margin=args.margin)
optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)

# ---------------------------------------------------------
# Helper Function for Stats
# ---------------------------------------------------------
def get_stats(a_out, p_out, n_out):
    d_ap = F.pairwise_distance(a_out, p_out, p=2).mean().item()
    d_an = F.pairwise_distance(a_out, n_out, p=2).mean().item()
    return d_ap, d_an

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
best_val_loss = float('inf')
prev_val_loss_sum = float('inf')
train_losses = []
val_losses = []

print(f"Starting training: {file_param_info}")
total_start_time = time.time()

for epoch in range(args.epoch_training):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    
    for i, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        a_out, p_out, n_out = model(anchor, positive, negative)
        
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 100 == 0 or (i + 1) == no_of_batches:
            d_ap, d_an = get_stats(a_out, p_out, n_out)
            print(f"Epoch [{epoch+1}/{args.epoch_training}], Step [{i+1}/{no_of_batches}], "
                  f"Loss: {running_loss/(i+1):.4f}, d(a,p): {d_ap:.4f}, d(a,n): {d_an:.4f}")

    # Validation Phase
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            a_out, p_out, n_out = model(anchor, positive, negative)
            val_loss_sum += criterion(a_out, p_out, n_out).item()
    
    avg_val_loss = val_loss_sum / batches_validation
    train_losses.append(running_loss / no_of_batches)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{args.epoch_training}] Validation Loss: {avg_val_loss:.4f}, Time: {time.time() - epoch_start_time:.2f}s")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), base_save_path + '_best.ckpt')
        print(f"--> Best model saved with loss: {best_val_loss:.4f}")

    # Early stopping check (v1 logic)
    if epoch >= args.epoch_enforced_training:
        if val_loss_sum > 1.1 * prev_val_loss_sum:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        prev_val_loss_sum = val_loss_sum

    torch.save(model.state_dict(), base_save_path + '.ckpt')

# ---------------------------------------------------------
# Final Stats & Plotting
# ---------------------------------------------------------
total_duration = time.time() - total_start_time
hours, rem = divmod(total_duration, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nTraining Completed. Total Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

# loss curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f"Loss Curve: {args.model_name}\nMargin: {args.margin} | LR: {args.learning_rate}")
plt.legend()
plt.grid(True)

plot_path = base_save_path + '_loss_curve.png'
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Loss curve saved to: {plot_path}")