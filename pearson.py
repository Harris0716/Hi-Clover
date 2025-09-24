import torch
from torch.utils.data import DataLoader, SequentialSampler
import matplotlib.pyplot as plt
import argparse
import json
from scipy.integrate import simpson
from numpy import minimum
from HiSiNet.HiCDatasetClass import HiCDatasetDec, GroupedHiCDataset, SiameseHiCDataset
from HiSiNet.reference_dictionaries import reference_genomes

parser = argparse.ArgumentParser(description='Triplet network testing module (GPU Pearson correlation distance)')
parser.add_argument('json_file', type=str, help='JSON file containing dataset paths')
parser.add_argument("data_inputs", nargs='+', help="keys from JSON for datasets to test")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

with open(args.json_file) as json_file:
    dataset = json.load(json_file)

# ===== GPU batch Pearson correlation distance =====
def batch_pearson_distance(batch1, batch2):
    """
    batch1, batch2: torch.Tensor, shape (B, N, N)
    Returns: distance = 1 - Pearson correlation for each pair
    """
    B = batch1.shape[0]
    flat1 = batch1.view(B, -1)
    flat2 = batch2.view(B, -1)

    mean1 = flat1.mean(dim=1, keepdim=True)
    mean2 = flat2.mean(dim=1, keepdim=True)

    num = ((flat1 - mean1) * (flat2 - mean2)).sum(dim=1)
    denom = torch.sqrt(((flat1 - mean1)**2).sum(dim=1) * ((flat2 - mean2)**2).sum(dim=1))
    corr = num / denom
    return 1 - corr  # distance

# ===== Testing function =====
def test_triplet_by_siamese(dataloader):
    distances = []
    labels = []

    for _, data in enumerate(dataloader):
        input1, input2, label = data
        input1 = input1.to(device)
        input2 = input2.to(device)
        label = label.cpu().numpy()

        # reshape 1D -> NxN if needed
        if input1.ndim == 3 and input1.shape[2] == 1:
            N = int(input1.shape[1]**0.5)
            input1 = input1.view(input1.shape[0], N, N)
            input2 = input2.view(input2.shape[0], N, N)

        batch_dist = batch_pearson_distance(input1, input2)
        distances.append(batch_dist.cpu().numpy())
        labels.append(label)

    distances = np.concatenate(distances)
    labels = np.concatenate(labels)
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
dataloader = DataLoader(
    Siamese,
    batch_size=10,
    sampler=SequentialSampler(Siamese),
    num_workers=1  # 使用 1 個子進程讀取資料
)

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

# ===== Separation Index & Performance =====
rep_density = a[0]
cond_density = b[0]
bin_edges = a[1]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_widths = np.diff(bin_edges)

overlap = minimum(rep_density, cond_density)
separation_index_simps = 1 - simpson(overlap, x=bin_centers)
separation_index_sum = 1 - np.sum(overlap * bin_widths)

replicate_correct = np.sum((distances < intersect) & (labels == 0))
condition_correct = np.sum((distances >= intersect) & (labels == 1))
replicate_total = np.sum(labels == 0)
condition_total = np.sum(labels == 1)

replicate_rate = replicate_correct / replicate_total
condition_rate = condition_correct / condition_total
mean_performance = (replicate_rate + condition_rate) / 2

print('----train/val-----')
print("Separation Index (Simpson): {:.4f}".format(separation_index_simps))
print("Separation Index (Sum)    : {:.4f}".format(separation_index_sum))
print("Replicate rate: {:.4f}".format(replicate_rate))
print("Condition rate: {:.4f}".format(condition_rate))
print("Mean Performance: {:.4f}".format(mean_performance))


# ===== Test =====
Siamese_test = make_siamese_dataset(args.data_inputs, mode="test")
dataloader = DataLoader(Siamese_test, batch_size=10, sampler=SequentialSampler(Siamese_test))

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

rep_density = a[0]
cond_density = b[0]
bin_edges = a[1]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_widths = np.diff(bin_edges)

overlap = minimum(rep_density, cond_density)
separation_index_simps = 1 - simpson(overlap, x=bin_centers)
separation_index_sum = 1 - np.sum(overlap * bin_widths)

replicate_correct = np.sum((distances < intersect) & (labels == 0))
condition_correct = np.sum((distances >= intersect) & (labels == 1))
replicate_total = np.sum(labels == 0)
condition_total = np.sum(labels == 1)

replicate_rate = replicate_correct / replicate_total
condition_rate = condition_correct / condition_total
mean_performance = (replicate_rate + condition_rate) / 2

print('----test-----')
print("Separation Index (Simpson): {:.4f}".format(separation_index_simps))
print("Separation Index (Sum)    : {:.4f}".format(separation_index_sum))
print("Replicate rate: {:.4f}".format(replicate_rate))
print("Condition rate: {:.4f}".format(condition_rate))
print("Mean Performance: {:.4f}".format(mean_performance))
