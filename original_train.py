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
                    help='an int for no of epochs training can go on for')
parser.add_argument('--epoch_enforced_training',  type=int, default=0,
                    help='an int for number of epochs to force training for')
parser.add_argument('--outpath',  type=str, default="outputs/",
                    help='a path for the output directory')
parser.add_argument('--seed',  type=int, default=30004,
                    help='an int for the seed')
parser.add_argument('--mask',  type=bool, default=False,
                    help='an argument specifying if the diagonal should be masked')
parser.add_argument('--margin',  type=float, default=1.0,
                    help='margin for triplet loss')
parser.add_argument("data_inputs", nargs='+', help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

os.makedirs(args.outpath, exist_ok=True)

# # debug
# def debug_triplet_dataset(ds: TripletHiCDataset):
#     print("==== Triplet Dataset Summary ====")
#     print(f"Total triplets: {len(ds.data)}")
#     print(f"Total positions: {len(ds.positions)}")
#     print()

#     # Count by chromosome
#     print("Triplets per chromosome:")
#     for chrom, (start, end) in ds.chromosomes.items():
#         print(f"  {chrom}: {end - start} triplets")

#     print("\nTriplets per label pair:")
#     label_count = {}
#     for a, b in ds.labels:
#         key = (a, b)
#         label_count[key] = label_count.get(key, 0) + 1

#     for k, v in label_count.items():
#         print(f"  {k}: {v} triplets")

#     print("==== End Summary ====")

# ds = TripletHiCDataset([TAM_R1, TAM_R2, KO_R1, KO_R2])
# debug_triplet_dataset(ds)

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

# Initialize CNN parameters
batch_size, learning_rate = args.batch_size, args.learning_rate
no_of_batches = len(Triplet) // args.batch_size
dataloader = DataLoader(Triplet,
                       batch_size=args.batch_size,
                       sampler=train_sampler,
                       num_workers=4,
                       pin_memory=True)

# Create validation dataset and dataloader
Triplet_validation = GroupedTripletHiCDataset(
    [TripletHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["validation"]],
                       reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs])
test_sampler = SequentialSampler(Triplet_validation)
batches_validation = np.ceil(len(Triplet_validation) / 100)
dataloader_validation = DataLoader(Triplet_validation,
                                 batch_size=100,
                                 sampler=test_sampler,
                                 num_workers=4,
                                 pin_memory=True)

# Initialize the Triplet network
model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

model_save_path = args.outpath + args.model_name + '_' + str(learning_rate) + '_' + str(batch_size) + '_' + str(args.seed)

# Save initial model
torch.save(model.state_dict(), model_save_path + '.ckpt')

# Initialize loss function and optimizer
criterion = TripletLoss(margin=args.margin)
optimizer = optim.Adagrad(model.parameters())

# Training loop
prev_validation_loss = float('inf')  # Initialize validation loss for early stopping

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

    print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(
        epoch + 1,
        args.epoch_training,
        running_validation_loss / batches_validation
    ))

    # Early stopping check
    if epoch >= args.epoch_enforced_training:
        if running_validation_loss > 1.1 * prev_validation_loss:
            print("Early stopping triggered")
            break
        prev_validation_loss = running_validation_loss

    # Save model checkpoints
    torch.save(model.state_dict(), model_save_path + '.ckpt')

print("Training completed")
