import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn.functional as F
import torch.nn as nn
from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedHiCDataset
import HiSiNet.models as models
import torch
import matplotlib.pyplot as plt
import argparse
from HiSiNet.reference_dictionaries import reference_genomes
import json

parser = argparse.ArgumentParser(description='Triplet network testing module')
parser.add_argument('model_name', type=str, help='a string indicating a model from models')
parser.add_argument('json_file', type=str, help='a file location for the json dictionary containing file paths')
parser.add_argument('model_infile', type=str, help='a string indicating the model location file')
parser.add_argument('--mask', type=bool, default=False, help='an argument specifying if the diagonal should be masked')
parser.add_argument("data_inputs", nargs='+', help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

with open(args.json_file) as json_file:
    dataset = json.load(json_file)

# Modify test_triplet_model to use only anchor and positive distances
def test_triplet_model(model, dataloader):
    d_ap_list = []  # Store distances between anchor and positive

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            anchor, positive, _ = data  # Ignore negative
            anchor, positive = anchor.to(cuda), positive.to(cuda)

            # Forward pass
            anchor_out, positive_out = model(anchor, positive)  # Only anchor and positive outputs

            # Compute distances
            d_ap = F.pairwise_distance(anchor_out, positive_out)
            d_ap_list.append(d_ap.cpu().numpy())

    # Combine results into arrays
    d_ap = np.concatenate(d_ap_list)

    return d_ap

cuda = torch.device("cuda:0")
model = eval("models." + args.model_name)(mask=args.mask).to(cuda)
state_dict = torch.load(args.model_infile)
model.load_state_dict(state_dict, strict=False)

model.eval()

# Load train/validation dataset
TripletDataset = GroupedHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["training"] + dataset[data_name]["validation"]],
                      reference=reference_genomes[dataset[data_name]["reference"]])
    for data_name in args.data_inputs
])
test_sampler = SequentialSampler(TripletDataset)
dataloader = DataLoader(TripletDataset, batch_size=100, sampler=test_sampler)

# train/validation set
d_ap = test_triplet_model(model, dataloader)

# Find threshold for separation
mx = max(d_ap)
mn = min(d_ap)
rng = np.arange(mn, mx, (mx - mn) / 200)

# Plot for train/validation set
a = plt.hist(d_ap, bins=rng, density=True, label='Anchor-Positive', alpha=0.5, color='#108690')
intersect = a[1][np.argwhere(np.diff(np.sign(a[0] - a[0])))[0]]
plt.axvline(intersect, color='k')
plt.legend()
plt.title("Distance of train/validation from " + args.model_infile.split("/")[-1] + " on: " + ", ".join(args.data_inputs))
plt.ylabel("Density")
plt.xlabel("Euclidean Distance")
plt.savefig(args.model_infile.split(".ckpt")[0] + "_train_distribution.pdf")
plt.close()

# Load test dataset
TripletDataset = GroupedHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["test"]],
                      reference=reference_genomes[dataset[data_name]["reference"]])
    for data_name in args.data_inputs
])
test_sampler = SequentialSampler(TripletDataset)
dataloader = DataLoader(TripletDataset, batch_size=100, sampler=test_sampler)

# Test set
d_ap = test_triplet_model(model, dataloader)

# Calculate rate based on the intersection threshold
global_rate = np.mean(d_ap < intersect)
TR_rate = np.mean(d_ap < intersect)

print('Global rate: {:.4f}, Triplet Accuracy (Anchor-Positive < Intersection Threshold): {:.4f}'
      .format(global_rate, TR_rate))

# Plot for test set
a = plt.hist(d_ap, bins=rng, density=True, label='Anchor-Positive', alpha=0.5, color='#108690')
plt.axvline(intersect, color='k')
plt.legend()
plt.title("Distance of test from " + args.model_infile.split("/")[-1] + " on: " + ", ".join(args.data_inputs))
plt.ylabel("Density")
plt.xlabel("Euclidean Distance")
plt.savefig(args.model_infile.split(".ckpt")[0] + "_test_distribution.pdf")
plt.close()
