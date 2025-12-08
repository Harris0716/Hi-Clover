import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn.functional as F
from HiSiNet.HiCDatasetClass import HiCDatasetDec, GroupedHiCDataset, SiameseHiCDataset
import matplotlib.pyplot as plt
import argparse
from HiSiNet.reference_dictionaries import reference_genomes
import json
from scipy.integrate import simpson
from numpy import minimum

parser = argparse.ArgumentParser(description='Triplet network testing module')
parser.add_argument('json_file', type=str, help='a file location for the json dictionary containing file paths')
parser.add_argument("data_inputs", nargs='+', help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

with open(args.json_file) as json_file:
    dataset = json.load(json_file)

def test_triplet_by_siamese(dataloader):
    distances = np.array([])
    labels = np.array([])

    with np.errstate(divide='ignore', invalid='ignore'):
        for _, data in enumerate(dataloader):
            input1, input2, label = data
            input1 = input1.view(input1.size(0), -1)
            input2 = input2.view(input2.size(0), -1)
            # baseline: use Euclidean distance
            predicted = F.pairwise_distance(input1, input2)
            distances = np.concatenate((distances, predicted.cpu().detach().numpy()))
            labels = np.concatenate((labels, label.cpu().detach().numpy()))

    return distances, labels

# Load train/validation dataset
Siamese = GroupedHiCDataset([SiameseHiCDataset(
    [HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["training"] + dataset[data_name]["validation"])],
    reference=reference_genomes[dataset[data_name]["reference"]]
) for data_name in args.data_inputs])
test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=100, sampler=test_sampler)

# train/validation set
distances, labels = test_triplet_by_siamese(dataloader)

mx = max(distances)
mn = min(distances[distances > 0])
rng = np.arange(mn, mx, (mx - mn) / 200)

a = plt.hist(distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
b = plt.hist(distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
intersect = a[1][np.argwhere(np.diff(np.sign(a[0] - b[0])))[0]]
plt.axvline(intersect, color='k')
plt.xticks(np.arange(0, np.ceil(mx), step=5), fontsize=10)
plt.legend()
plt.title("Distance of Train/Val from baseline (Euclidean)")
plt.ylabel("Density")
plt.xlabel("Euclidean Distance of Representation")
plt.savefig("baseline_train_distribution.pdf")
plt.close()

print('----train/val-----')
# Separation Index
rep_density = a[0]
cond_density = b[0]
bin_edges = a[1]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_widths = np.diff(bin_edges)

# overlap area
overlap = minimum(rep_density, cond_density)

# Separation Index (keyword argument style)
separation_index_simps = 1 - simpson(overlap, x=bin_centers)
separation_index_sum = 1 - np.sum(overlap * bin_widths)

print("Separation Index (Simpson): {:.4f}".format(separation_index_simps))
print("Separation Index (Sum)    : {:.4f}".format(separation_index_sum))

# mean performance
replicate_correct = np.sum((distances < intersect) & (labels == 0))
condition_correct = np.sum((distances >= intersect) & (labels == 1))
replicate_total = np.sum(labels == 0)
condition_total = np.sum(labels == 1)

replicate_rate = replicate_correct / replicate_total
condition_rate = condition_correct / condition_total
mean_performance = (replicate_rate + condition_rate) / 2

print("Replicate rate: {:.4f}".format(replicate_rate))
print("Condition rate: {:.4f}".format(condition_rate))
print("Mean Performance: {:.4f}".format(mean_performance))

# Dataset test
Siamese = GroupedHiCDataset([SiameseHiCDataset(
    [HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["test"])],
    reference=reference_genomes[dataset[data_name]["reference"]]
) for data_name in args.data_inputs])
test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=100, sampler=test_sampler)

# Test the model
distances, labels = test_triplet_by_siamese(dataloader)
mx = max(distances)
mn = min(distances[distances > 0])
rng = np.arange(mn, mx, (mx - mn) / 200)

a = plt.hist(distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
b = plt.hist(distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
intersect = a[1][np.argwhere(np.diff(np.sign(a[0] - b[0])))[0]]
plt.axvline(intersect, color='k')
plt.xticks(np.arange(0, np.ceil(mx), step=5), fontsize=10)
plt.title("Distance of Test from baseline (Euclidean)")
plt.ylabel("Density")
plt.xlabel("Euclidean Distance of Representation")
plt.legend()
plt.savefig("baseline_test_distribution.pdf")
plt.close()

print('----test-----')
# Separation Index
rep_density = a[0]
cond_density = b[0]
bin_edges = a[1]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_widths = np.diff(bin_edges)

overlap = minimum(rep_density, cond_density)

# Separation Index (keyword argument style)
separation_index_simps = 1 - simpson(overlap, x=bin_centers)
separation_index_sum = 1 - np.sum(overlap * bin_widths)

print("Separation Index (Simpson): {:.4f}".format(separation_index_simps))
print("Separation Index (Sum)    : {:.4f}".format(separation_index_sum))

# mean performance
replicate_correct = np.sum((distances < intersect) & (labels == 0))
condition_correct = np.sum((distances >= intersect) & (labels == 1))
replicate_total = np.sum(labels == 0)
condition_total = np.sum(labels == 1)

replicate_rate = replicate_correct / replicate_total
condition_rate = condition_correct / condition_total
mean_performance = (replicate_rate + condition_rate) / 2

print("Replicate rate: {:.4f}".format(replicate_rate))
print("Condition rate: {:.4f}".format(condition_rate))
print("Mean Performance: {:.4f}".format(mean_performance))
