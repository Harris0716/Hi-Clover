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

# Parsing arguments
parser = argparse.ArgumentParser(description='Triplet network testing module')
parser.add_argument('model_name', type=str, help='a string indicating a model from models')
parser.add_argument('json_file', type=str, help='a file location for the json dictionary containing file paths')
parser.add_argument('model_infile', type=str, help='a string indicating the model location file')
parser.add_argument('--mask', type=bool, default=False, help='whether to mask diagonal')
parser.add_argument("data_inputs", nargs='+',
                    help="keys from dictionary containing paths for training and validation sets.")
args = parser.parse_args()

# Load the dataset dictionary
with open(args.json_file) as json_file:
    dataset = json.load(json_file)

# Set device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# Load the model state dict
state_dict = torch.load(args.model_infile, weights_only=True)
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


# Function to plot the distance distributions
def plot_triplet_distribution(d_ap, d_an, model_name, data_inputs, phase='train-validation'):
    plt.hist(d_ap, bins=50, alpha=0.5, label='Anchor-Positive Distance', color='#108690', density=True)
    plt.hist(d_an, bins=50, alpha=0.5, label='Anchor-Negative Distance', color='#1D1E4E', density=True)
    plt.axvline(np.mean(d_ap), color='blue', linestyle='dashed', label='Mean d_ap')
    plt.axvline(np.mean(d_an), color='red', linestyle='dashed', label='Mean d_an')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.title(f'Triplet {phase} Distribution ({model_name})')
    plt.legend()
    plt.savefig(f"{model_name}_{phase}_triplet_distribution.pdf")
    plt.close()


# Create training and validation dataset and dataloader
Triplet_train_val = GroupedHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(data_path) for data_path in
                       dataset[data_name]["training"] + dataset[data_name]["validation"]],
                      reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs
])

train_val_sampler = SequentialSampler(Triplet_train_val)
dataloader_train_val = DataLoader(Triplet_train_val, batch_size=100, sampler=train_val_sampler)

# Execute the testing phase for training and validation datasets
train_val_d_ap, train_val_d_an, train_val_accuracy = test_triplet(model, dataloader_train_val)
plot_triplet_distribution(train_val_d_ap, train_val_d_an, args.model_infile.split("/")[-1], args.data_inputs,
                          phase='train-validation')

# Create test dataset and dataloader
Triplet_test = GroupedHiCDataset([
    TripletHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["test"]],
                      reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs
])

test_sampler = SequentialSampler(Triplet_test)
dataloader_test = DataLoader(Triplet_test, batch_size=100, sampler=test_sampler)

# Execute the testing phase for the test dataset
test_d_ap, test_d_an, test_accuracy = test_triplet(model, dataloader_test)
plot_triplet_distribution(test_d_ap, test_d_an, args.model_infile.split("/")[-1], args.data_inputs, phase='test')

# Optionally print accuracy for both phases
print(f"Triplet Train/Validation Accuracy: {train_val_accuracy:.4f} (d_ap < d_an)")
print(f"Triplet Test Accuracy: {test_accuracy:.4f} (d_ap < d_an)")
