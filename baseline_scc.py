import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch
import matplotlib.pyplot as plt
import argparse
import json
from scipy.integrate import simpson
from numpy import minimum
from HiSiNet.HiCDatasetClass import HiCDatasetDec, GroupedHiCDataset, SiameseHiCDataset
from HiSiNet.reference_dictionaries import reference_genomes
from hicrep import hicrepSCC  # HiRep SCC

parser = argparse.ArgumentParser(description='Triplet network testing module (HiRep SCC version)')
parser.add_argument('json_file', type=str, help='a file location for the json dictionary containing file paths')
parser.add_argument("data_inputs", nargs='+', help="keys from dictionary containing paths for training and validation sets.")
args = parser.parse_args()

with open(args.json_file) as json_file:
    dataset = json.load(json_file)


# ===== HiRep SCC function =====
def hicrep_scc(mat1, mat2, h=1, dBPMax=2000000, resolution=10000):
    """
    計算兩個 Hi-C contact maps 的 SCC
    mat1, mat2: numpy 2D contact maps
    """
    return hicrepSCC(mat1, mat2, h=h, dBPMax=dBPMax, resolution=resolution)


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
            # 假設 m1, m2 是 contact matrix
            if m1.ndim == 1:  # 如果 flatten，要 reshape
                side = int(np.sqrt(len(m1)))
                m1 = m1.reshape(side, side)
                m2 = m2.reshape(side, side)

            scc_val = hicrep_scc(m1, m2)
            dist = 1 - scc_val  # 轉換成距離
            batch_scores.append(dist)

        distances = np.concatenate((distances, np.array(batch_scores)))
        labels = np.concatenate((labels, label.cpu().detach().numpy()))

    return distances, labels


# ===== Train/Validation Dataset =====
Siamese = GroupedHiCDataset([SiameseHiCDataset(
    [HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["training"] + dataset[data_name]["validation"])],
    reference=reference_genomes[dataset[data_name]["reference"]]
) for data_name in args.data_inputs])
test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=10, sampler=test_sampler)  # 減少 batch_size，避免記憶體爆掉

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
plt.title("Distance of Train/Val from HiRep SCC")
plt.ylabel("Density")
plt.xlabel("1 - SCC Distance of Representation")
plt.savefig("scc_train_distribution.pdf")
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


# ===== Test Dataset =====
Siamese = GroupedHiCDataset([SiameseHiCDataset(
    [HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["test"])],
    reference=reference_genomes[dataset[data_name]["reference"]]
) for data_name in args.data_inputs])
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
plt.title("Distance of Test from HiRep SCC")
plt.ylabel("Density")
plt.xlabel("1 - SCC Distance of Representation")
plt.legend()
plt.savefig("scc_test_distribution.pdf")
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
