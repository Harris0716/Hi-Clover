import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
import torch
from torch_plus.loss import TripletLoss
import argparse
from HiSiNet.reference_dictionaries import reference_genomes
import json
import os

# ADDED
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Triplet network')
parser.add_argument('model_name',  type=str,
                    help='a string indicating a model from models')
parser.add_argument('json_file',  type=str,
                    help='a file location for the json dictionary containing file paths')
parser.add_argument('learning_rate',  type=float,
                    help='a float for the learning rate')
parser.add_argument('--batch_size',  type=int, default=17,
                    help='an int for batch size')
parser.add_argument('--epoch_training',  type=int, default=30,
                    help='max number of epochs')
parser.add_argument('--epoch_enforced_training',  type=int, default=0,
                    help='number of epochs to force training before early stopping can trigger')
# ADDED — patience argument
parser.add_argument('--patience', type=int, default=10,
                    help='number of epochs with no improvement before early stopping')
parser.add_argument('--outpath',  type=str, default="outputs/",
                    help='a path for the output directory')
parser.add_argument('--seed',  type=int, default=30004,
                    help='random seed')
parser.add_argument('--mask',  type=bool, default=False,
                    help='mask diagonal')
parser.add_argument('--margin',  type=float, default=1.0,
                    help='margin for triplet loss')
parser.add_argument("data_inputs", nargs='+', help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

os.makedirs(args.outpath, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("Using {} GPUs".format(n_gpu))
else:
    device = torch.device("cpu")
    print("Using CPU")

with open(args.json_file) as json_file:
    dataset = json.load(json_file)

torch.manual_seed(args.seed)

# Initialize dataset
Triplet = GroupedTripletHiCDataset([
    TripletHiCDataset(
        [HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["training"]],
        reference=reference_genomes[dataset[data_name]["reference"]])
    for data_name in args.data_inputs])

# shuffle
train_sampler = torch.utils.data.RandomSampler(Triplet)

batch_size, learning_rate = args.batch_size, args.learning_rate

dataloader = DataLoader(
    Triplet,
    batch_size=args.batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True)
no_of_batches = len(dataloader)

# Validation dataset
Triplet_validation = GroupedTripletHiCDataset(
    [TripletHiCDataset(
        [HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["validation"]],
        reference=reference_genomes[dataset[data_name]["reference"]])
     for data_name in args.data_inputs])

test_sampler = SequentialSampler(Triplet_validation)
dataloader_validation = DataLoader(
    Triplet_validation,
    batch_size=100,
    sampler=test_sampler,
    num_workers=4,
    pin_memory=True)
batches_validation = len(dataloader_validation)

# Model
model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

model_save_path = args.outpath + args.model_name + '_' + str(learning_rate) + '_' + str(batch_size) + '_' + str(args.seed)

# Save initial model
torch.save(model.state_dict(), model_save_path + '.ckpt')

# reduction='mean' means the loss is averaged over the batch
criterion = TripletLoss(margin=args.margin, reduction='mean')
optimizer = optim.Adagrad(model.parameters())

# Loss history
train_losses = []
val_losses = []

# Early stopping state
best_val_loss = float('inf')
patience_counter = 0

print(f"Early stopping: patience = {args.patience}, enforced epochs = {args.epoch_enforced_training}")

# Training loop
for epoch in range(args.epoch_training):
    model.train()
    running_loss = 0.0
    running_validation_loss = 0.0

    # Training phase
    for i, data in enumerate(dataloader):
        anchor, positive, negative = data
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        anchor_out, positive_out, negative_out = model(anchor, positive, negative)
        loss = criterion(anchor_out, positive_out, negative_out)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % no_of_batches == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1,
                args.epoch_training,
                i + 1,
                int(no_of_batches),
                running_loss / no_of_batches
            ))

    epoch_train_loss = running_loss / no_of_batches
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader_validation):
            anchor, positive, negative = data
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            loss = criterion(anchor_out, positive_out, negative_out)
            running_validation_loss += loss.item()

    epoch_val_loss = running_validation_loss / batches_validation
    val_losses.append(epoch_val_loss)

    print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(
        epoch + 1,
        args.epoch_training,
        epoch_val_loss
    ))

    # ======== NEW Early stopping with patience ========
    if epoch >= args.epoch_enforced_training:

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0

        # NEW — Save best model
        torch.save(model.state_dict(), model_save_path + '_best.ckpt')
        print("Best model saved.")
    else:
        patience_counter += 1

    if patience_counter >= args.patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    # ==================================================

    # NEW — Save last model
    torch.save(model.state_dict(), model_save_path + '_last.ckpt')

print("Training completed")

# Plot loss curve
plt.figure(figsize=(8,6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)

loss_curve_path = os.path.join(args.outpath, 'loss_curve.png')
plt.savefig(loss_curve_path, dpi=300)
plt.close()

print("Loss curve saved to:", loss_curve_path)
