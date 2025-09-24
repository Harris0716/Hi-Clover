import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
import torch
import matplotlib.pyplot as plt
import argparse
import json
from scipy.integrate import simpson
from numpy import minimum
from HiSiNet.HiCDatasetClass import HiCDatasetDec, GroupedHiCDataset, SiameseHiCDataset
from HiSiNet.reference_dictionaries import reference_genomes
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='Triplet network testing module (Pearson correlation distance)')
parser.add_argument('json_file', type=str, help='JSON file containing dataset paths')
parser.add_argument("data_inputs", nargs='+', help="keys from JSON for datasets to test")
args = parser.parse_args()

with open(args.json_file) as json_file:
    dataset = json.load(json_file)


# ===== Pearson correlation distance =====
def pearson_distance(mat1, mat2):
    """Flatten matrices and compute Pearson correlation distance"""
    m1 = mat1.flatten()
    m2 = mat2.flatten()
    corr, _ = pearsonr(m1, m2)
    return 1 - corr  # distance: smaller means more similar


# ===== Testing function =====
def test_triplet_by_siamese(dataloader):
    distances = np.array([])
    labels = np.array([])

    for _, data in enumerate(dataloader):
        input1, input2, label = data
        input1 = input1.cpu().numpy()
        input2 = input2.cpu().numpy()

        batch_scores = []
        for m1, m2 in zip(input1, input2):
            if m1.ndim == 1:
                side = int(np.sqrt(len(m1)))
                m1 = m1.reshape(side, side)
                m2 = m2.reshape(side, side)

            dist = pearson_distance(m1, m2)
            batch_scores.append(dist)

        distances = np.concatenate((distances, np.array(batch_scores)))
        labels = np.concatenate((labels, label.cpu().numpy()))

    return distances, labels


# ===== Helper to create Siamese Dataset from paths =====
def make_siamese_dataset(data_names, mode="trainval"):
    datasets = []
    for name in data_names:
        paths = dataset[name]
        if mode == "trainval":
            combined = paths.get("training", []) + paths.get("validation", [])
        elif mode == "test":
            combined = paths.get("test", [])
        datasets.append(
            SiameseHiCDataset([HiCDatasetDec.load(p) for p in combined],
                              reference=reference_genomes[paths["reference"]])
        )
    return GroupedHiCDataset(datasets)


# ===== Train/Validation =====
Siamese = make_siamese_dataset(args.data_inputs, mode="trainval")
test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=10, sampler=test_sampler)

distances, labels = test_triplet_by_siamese(dataloader)

mx = max(distances)
mn = min(distances[distances > 0])
rng = np.arange(mn, mx, (mx - mn) / 200)

a = plt.hist(distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
b = plt.hist(distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
intersect = a[1][np.argwhere(np.diff(np.sign(a[0] - b[0])))[0]]
plt.axvline(intersect, color='k')
plt.xticks(np.arange(0, np.ceil(mx), step=0.1), fontsize=8)
plt.legend()
plt.title("Distance of Train/Val (1 - Pearson correlation)")
plt.ylabel("Density")
plt.xlabel("Distance")
plt.savefig("train_distribution.pdf")
plt.close()

print('----train/val-----')
rep_density = a[0]
cond_density = b[0]
bin_edges = a[1]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_widths = np.diff(bin_edges)

overlap = minimum(rep_density, cond_density)
separation_index_simps = 1 - simpson(overlap, x=bin_centers)
separation_index_sum = 1 - np.sum(overlap * bin_widths)

print("Separation Index (Simpson): {:.4f}".format(separation_index_simps))
print("Separation Index (Sum)    : {:.4f}".format(separation_index_sum))

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


# ===== Test =====
Siamese_test = make_siamese_dataset(args.data_inputs, mode="test")
test_sampler = SequentialSampler(Siamese_test)
dataloader = DataLoader(Siamese_test, batch_size=10, sampler=test_sampler)

distances, labels = test_triplet_by_siamese(dataloader)

mx = max(distances)
mn = min(distances[distances > 0])
rng = np.arange(mn, mx, (mx - mn) / 200)

a = plt.hist(distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
b = plt.hist(distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
intersect = a[1][np.argwhere(np.diff(np.sign(a[0] - b[0])))[0]]
plt.axvline(intersect, color='k')
plt.xticks(np.arange(0, np.ceil(mx), step=0.1), fontsize=8)
plt.title("Distance of Test (1 - Pearson correlation)")
plt.ylabel("Density")
plt.xlabel("Distance")
plt.legend()
plt.savefig("test_distribution.pdf")
plt.close()

print('----test-----')
rep_density = a[0]
cond_density = b[0]
bin_edges = a[1]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_widths = np.diff(bin_edges)

overlap = minimum(rep_density, cond_density)
separation_index_simps = 1 - simpson(overlap, x=bin_centers)
separation_index_sum = 1 - np.sum(overlap * bin_widths)

print("Separation Index (Simpson): {:.4f}".format(separation_index_simps))
print("Separation Index (Sum)    : {:.4f}".format(separation_index_sum))

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
