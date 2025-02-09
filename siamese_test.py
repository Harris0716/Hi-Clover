import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import argparse
import json

from HiSiNet.HiCDatasetClass import HiCDatasetDec, SiameseHiCDataset, GroupedHiCDataset
import HiSiNet.models as models
from HiSiNet.reference_dictionaries import reference_genomes

parser = argparse.ArgumentParser(description='Siamese network testing module')
parser.add_argument('model_name', type=str,
                    help='a string indicating a model from models')
parser.add_argument('json_file', type=str,
                    help='a file location for the json dictionary containing file paths')
parser.add_argument('model_infile', type=str,
                    help='a string indicating the model location file')
parser.add_argument('--mask', type=bool, default=False,
                    help='an argument specifying if the diagonal should be masked')
parser.add_argument("data_inputs", nargs='+', help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

with open(args.json_file) as json_file:
    dataset = json.load(json_file)

# Define pairwise similarity testing function
def test_pairwise_similarity(model, dataloader):
    distances = np.array([])
    labels = np.array([])

    for _, data in enumerate(dataloader):
        sample1, sample2, label = data
        sample1, sample2 = sample1.to(device), sample2.to(device)

        # Forward pass using two inputs only (ignore negative branch)
        output1, output2, _ = model(sample1, sample2, sample2)
        predicted_distance = F.pairwise_distance(output1, output2)

        # Record results
        distances = np.concatenate((distances, predicted_distance.cpu().detach().numpy()))
        labels = np.concatenate((labels, label.cpu().detach().numpy()))

    return distances, labels

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("Using {} GPUs".format(n_gpu))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load model
model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load(args.model_infile))
model.eval()

# Load training and validation datasets
Siamese = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["training"] + dataset[data_name]["validation"])],
                                reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs])

test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=100, sampler=test_sampler)

# Test the model on training and validation data
distances, labels = test_pairwise_similarity(model, dataloader)

mx = max(distances)
mn = min(distances[distances > 0])
rng = np.arange(mn, mx, (mx - mn) / 200)

# Plot the distance distribution for training/validation
plt.hist(distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
plt.hist(distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
plt.axvline(np.mean(distances), color='k')
plt.xticks(np.arange(0, np.ceil(mx), 1))
plt.legend()
plt.title("Distance of train and validation from {} on: {}".format(args.model_infile.split("/")[-1], ", ".join(args.data_inputs)))
plt.ylabel("Density")
plt.xlabel("Euclidean Distance of Representation")
plt.savefig(args.model_infile.split(".ckpt")[0] + "_train_distribution.pdf")
plt.close()

# Load test dataset
Siamese = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["test"])],
                                reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs])

test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=100, sampler=test_sampler)

# Test the model on test data
distances, labels = test_pairwise_similarity(model, dataloader)

# Compute classification rates
global_rate = sum(((distances < np.mean(distances)) == (labels == 0))) / len(distances)
TR_rate = sum((distances < np.mean(distances)) & (labels == 0)) / sum(labels == 0)
BC_rate = sum((distances > np.mean(distances)) & (labels == 1)) / sum(labels == 1)

print('Global rate: {:.4f}, Replicate rate: {:.4f}, Condition rate: {:.4f}'
      .format(global_rate, TR_rate, BC_rate))

# Plot the distance distribution for test set
plt.hist(distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
plt.hist(distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
plt.axvline(np.mean(distances), color='k')
plt.xticks(np.arange(0, np.ceil(mx), 1))
plt.title("Distance of test from {} on: {}".format(args.model_infile.split("/")[-1], ", ".join(args.data_inputs)))
plt.ylabel("Density")
plt.xlabel("Euclidean Distance of Representation")
plt.legend()
plt.savefig(args.model_infile.split(".ckpt")[0] + "_test_distribution.pdf")
plt.close()
