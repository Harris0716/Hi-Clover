import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
import torch.nn as nn
from HiSiNet.HiCDatasetClass import HiCDatasetDec, GroupedHiCDataset, SiameseHiCDataset
import HiSiNet.models as models
import torch
import matplotlib.pyplot as plt
import argparse
from HiSiNet.reference_dictionaries import reference_genomes
import json
from scipy.integrate import simpson
from numpy import minimum
import csv
import os

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
            label = label.type(torch.FloatTensor).to(cuda)
            output1, output2, _ = model(input1, input2, input2)
            predicted = F.pairwise_distance(output1, output2)
            distances = np.concatenate((distances, predicted.cpu().detach().numpy()))
            labels  = np.concatenate((labels, label.cpu().detach().numpy()))

    return distances, labels

def compute_metrics(distances, labels, intersect):
    rep_correct = np.sum((distances < intersect) & (labels == 0))
    cond_correct = np.sum((distances >= intersect) & (labels == 1))
    rep_total = np.sum(labels == 0)
    cond_total = np.sum(labels == 1)

    rep_rate = rep_correct / rep_total if rep_total > 0 else 0
    cond_rate = cond_correct / cond_total if cond_total > 0 else 0
    mean_perf = (rep_rate + cond_rate) / 2

    # hist densities for separation index
    rng = np.linspace(min(distances[distances>0]), max(distances), 200)
    a = np.histogram(distances[(labels==0)], bins=rng, density=True)
    b = np.histogram(distances[(labels==1)], bins=rng, density=True)

    rep_density = a[0]
    cond_density = b[0]
    bin_edges = a[1]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_widths = np.diff(bin_edges)

    overlap = minimum(rep_density, cond_density)
    sep_simps = 1 - simpson(overlap, x=bin_centers)
    sep_sum = 1 - np.sum(overlap * bin_widths)

    return sep_simps, sep_sum, rep_rate, cond_rate, mean_perf

cuda = torch.device("cuda:0")
model = eval("models."+ args.model_name)(mask=args.mask).to(cuda)
state_dict = torch.load(args.model_infile, weights_only=True, map_location=cuda)
new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# --------- 準備 CSV 檔案 ---------
csv_file = "results.csv"
write_header = not os.path.exists(csv_file)
with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "Split",
            "SeparationIndex_Simpson", "SeparationIndex_Sum",
            "ReplicateRate", "ConditionRate", "MeanPerformance"
        ])
# --------------------------------

# train/val
Siamese = GroupedHiCDataset([
    SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["training"] + dataset[data_name]["validation"])],
    reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs
])
dataloader = DataLoader(Siamese, batch_size=100, sampler=SequentialSampler(Siamese))

distances, labels = test_triplet_by_siamese(model, dataloader)
a = plt.hist(distances[(labels==0)], bins=200, density=True)
b = plt.hist(distances[(labels==1)], bins=200, density=True)
intersect = a[1][np.argwhere(np.diff(np.sign(a[0]-b[0])))[0]]
plt.close()

sep_simps, sep_sum, rep_rate, cond_rate, mean_perf = compute_metrics(distances, labels, intersect)

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "train/val",
        "{:.4f}".format(sep_simps), "{:.4f}".format(sep_sum),
        "{:.4f}".format(rep_rate), "{:.4f}".format(cond_rate), "{:.4f}".format(mean_perf)
    ])

# test
Siamese = GroupedHiCDataset([
    SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["test"]],
    reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs
])
dataloader = DataLoader(Siamese, batch_size=100, sampler=SequentialSampler(Siamese))

distances, labels = test_triplet_by_siamese(model, dataloader)
a = plt.hist(distances[(labels==0)], bins=200, density=True)
b = plt.hist(distances[(labels==1)], bins=200, density=True)
intersect = a[1][np.argwhere(np.diff(np.sign(a[0]-b[0])))[0]]
plt.close()

sep_simps, sep_sum, rep_rate, cond_rate, mean_perf = compute_metrics(distances, labels, intersect)

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "test",
        "{:.4f}".format(sep_simps), "{:.4f}".format(sep_sum),
        "{:.4f}".format(rep_rate), "{:.4f}".format(cond_rate), "{:.4f}".format(mean_perf)
    ])
