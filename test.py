import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn.functional as F
import torch.nn as nn
from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedHiCDataset, SiameseHiCDataset
import HiSiNet.models as models
import torch
import matplotlib.pyplot as plt
import argparse
from HiSiNet.reference_dictionaries import reference_genomes
import json
from scipy.integrate import simpson
from numpy import minimum
import pandas as pd

parser = argparse.ArgumentParser(description='Triplet network testing module')
parser.add_argument('model_name', type=str, help='a string indicating a model from models')
parser.add_argument('json_file', type=str, help='a file location for the json dictionary containing file paths')
parser.add_argument('model_infile', type=str, help='a string indicating the model location file')
parser.add_argument('--mask', type=bool, default=False, help='an argument specifying if the diagonal should be masked')
parser.add_argument("data_inputs", nargs='+', help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

with open(args.json_file) as json_file:
    dataset = json.load(json_file)

def test_triplet_by_siamese(model, dataloader):
    distances = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            input1, input2, label = data
            input1, input2 = input1.to(cuda), input2.to(cuda)
            label = label.float()
            output1, output2, _ = model(input1, input2, input2)
            predicted = F.pairwise_distance(output1, output2)
            distances = np.concatenate((distances, predicted.cpu().numpy()))

            labels = np.concatenate((labels, label.numpy()))

    return distances, labels


cuda = torch.device("cuda:0")
model = eval("models." + args.model_name)(mask=args.mask).to(cuda)
state_dict = torch.load(args.model_infile, weights_only=True, map_location=cuda)

new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# ------------------------------ TRAIN/VAL ------------------------------------
Siamese = GroupedHiCDataset([
    SiameseHiCDataset(
        [HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["training"] + dataset[data_name]["validation"])],
        reference=reference_genomes[dataset[data_name]["reference"]]
    ) for data_name in args.data_inputs
])

dataloader = DataLoader(Siamese, batch_size=100, sampler=SequentialSampler(Siamese))
distances, labels = test_triplet_by_siamese(model, dataloader)

mx = max(distances)
mn = min(distances[distances > 0])
rng = np.arange(mn, mx, (mx - mn) / 200)

a = plt.hist(distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
b = plt.hist(distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')

# intersect
diff = a[0] - b[0]
idx = np.where(np.diff(np.sign(diff)))[0]

# to avoid the case that there is no intersection
if len(idx) == 0:
    intersect = a[1][len(a[1]) // 2]
else:
    intersect = a[1][idx[0]]


plt.axvline(intersect, color='k')
plt.xticks(np.arange(0, np.ceil(mx), step=5), fontsize=10)
plt.legend()
plt.title("Distance of Train/Val from " + args.model_infile.split("/")[-1] + " on: " + ", ".join(args.data_inputs))
plt.ylabel("Density")
plt.xlabel("Euclidean Distance of Representation")
plt.savefig(args.model_infile.split(".ckpt")[0] + "_train_distribution.pdf")
plt.close()

print("----train/val-----")

# Separation Index (train/val)
rep_density = a[0]
cond_density = b[0]
bin_edges = a[1]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

overlap = minimum(rep_density, cond_density)
train_separation_index = 1 - simpson(overlap, x=bin_centers)
print("Separation Index (Simpson): {:.4f}".format(train_separation_index))

# Metrics (train/val)
replicate_correct = np.sum((distances < intersect) & (labels == 0))
condition_correct = np.sum((distances >= intersect) & (labels == 1))
replicate_total = np.sum(labels == 0)
condition_total = np.sum(labels == 1)

train_replicate_rate = replicate_correct / replicate_total
train_condition_rate = condition_correct / condition_total
train_mean_performance = (train_replicate_rate + train_condition_rate) / 2

print("Replicate rate: {:.4f}".format(train_replicate_rate))
print("Condition rate: {:.4f}".format(train_condition_rate))
print("Mean Performance: {:.4f}".format(train_mean_performance))


# ------------------------------ TEST -----------------------------------------
Siamese = GroupedHiCDataset([
    SiameseHiCDataset(
        [HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["test"])],
        reference=reference_genomes[dataset[data_name]["reference"]]
    ) for data_name in args.data_inputs
])

dataloader = DataLoader(Siamese, batch_size=100, sampler=SequentialSampler(Siamese))
distances, labels = test_triplet_by_siamese(model, dataloader)

mx = max(distances)
mn = min(distances[distances > 0])
rng = np.arange(mn, mx, (mx - mn) / 200)

a = plt.hist(distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
b = plt.hist(distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')

intersect = a[1][np.argwhere(np.diff(np.sign(a[0] - b[0])))[0]]

plt.axvline(intersect, color='k')
plt.xticks(np.arange(0, np.ceil(mx), step=5), fontsize=10)
plt.legend()
plt.title("Distance of Test from " + args.model_infile.split("/")[-1] + " on: " + ", ".join(args.data_inputs))
plt.ylabel("Density")
plt.xlabel("Euclidean Distance of Representation")
plt.savefig(args.model_infile.split(".ckpt")[0] + "_test_distribution.pdf")
plt.close()

print("----test-----")

# Separation Index (test)
rep_density = a[0]
cond_density = b[0]
bin_edges = a[1]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

overlap = minimum(rep_density, cond_density)
test_separation_index = 1 - simpson(overlap, x=bin_centers)
print("Separation Index (Simpson): {:.4f}".format(test_separation_index))

# Metrics (test)
replicate_correct = np.sum((distances < intersect) & (labels == 0))
condition_correct = np.sum((distances >= intersect) & (labels == 1))
replicate_total = np.sum(labels == 0)
condition_total = np.sum(labels == 1)

test_replicate_rate = replicate_correct / replicate_total
test_condition_rate = condition_correct / condition_total
test_mean_performance = (test_replicate_rate + test_condition_rate) / 2

print("Replicate rate: {:.4f}".format(test_replicate_rate))
print("Condition rate: {:.4f}".format(test_condition_rate))
print("Mean Performance: {:.4f}".format(test_mean_performance))


# ------------------------------ EXPORT SUMMARY CSV ---------------------------
summary_path = args.model_infile.split(".ckpt")[0] + "_summary.csv"

summary_df = pd.DataFrame({
    "set": ["train_val", "test"],
    "replicate_rate": [train_replicate_rate, test_replicate_rate],
    "condition_rate": [train_condition_rate, test_condition_rate],
    "mean_performance": [train_mean_performance, test_mean_performance],
    "separation_index": [train_separation_index, test_separation_index]
})

# 四捨五入到小數點後4位
summary_df = summary_df.round(4)

summary_df.to_csv(summary_path, index=False)
print(f"[Saved] {summary_path}")
