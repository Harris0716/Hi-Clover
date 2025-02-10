import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import argparse
import json

from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedHiCDataset
import HiSiNet.models as models
from HiSiNet.reference_dictionaries import reference_genomes

parser = argparse.ArgumentParser(description='Triplet network testing module')
parser.add_argument('model_name', type=str, help='a string indicating a model from models')
parser.add_argument('json_file', type=str, help='a file location for the json dictionary containing file paths')
parser.add_argument('model_infile', type=str, help='a string indicating the model location file')
parser.add_argument('--mask', type=bool, default=False, help='whether to mask diagonal')
parser.add_argument("data_inputs", nargs='+',
                    help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

with open(args.json_file) as json_file:
    dataset = json.load(json_file)

# Set the device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# Load the state dict, removing 'module.' prefix for DataParallel models
state_dict = torch.load(args.model_infile)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()


# Define the triplet testing function
def test_triplet(model, dataloader):
    all_d_ap = []
    all_d_an = []

    with torch.no_grad():
        for _, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_out, positive_out, negative_out = model(anchor, positive, negative)

            # Compute distances
            d_ap, d_an = model.compute_distances(anchor_out, positive_out, negative_out)
            all_d_ap.append(d_ap.cpu().numpy())
            all_d_an.append(d_an.cpu().numpy())

    all_d_ap = np.concatenate(all_d_ap)
    all_d_an = np.concatenate(all_d_an)

    # Calculate accuracy
    triplet_accuracy = np.mean(all_d_ap < all_d_an)

    return all_d_ap, all_d_an, triplet_accuracy


# Create test dataset and DataLoader
Triplet_dataset = GroupedHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["test"]],
                      reference=reference_genomes[dataset[data_name]["reference"]])
    for data_name in args.data_inputs
])

test_sampler = SequentialSampler(Triplet_dataset)
dataloader = DataLoader(Triplet_dataset, batch_size=100, sampler=test_sampler)

# Execute the test
d_ap, d_an, accuracy = test_triplet(model, dataloader)

print(f'Triplet Accuracy: {accuracy:.4f} (d_ap < d_an)')

# Plot the distance distribution
plt.hist(d_ap, bins=50, alpha=0.5, label='Anchor-Positive Distance', color='#108690', density=True)
plt.hist(d_an, bins=50, alpha=0.5, label='Anchor-Negative Distance', color='#1D1E4E', density=True)
plt.axvline(np.mean(d_ap), color='blue', linestyle='dashed', label='Mean d_ap')
plt.axvline(np.mean(d_an), color='red', linestyle='dashed', label='Mean d_an')
plt.xlabel('Euclidean Distance')
plt.ylabel('Density')
plt.title(f'Triplet Test Distribution ({args.model_infile.split("/")[-1]})')
plt.legend()
plt.savefig(args.model_infile.split(".ckpt")[0] + "_triplet_test_distribution.pdf")
plt.close()
