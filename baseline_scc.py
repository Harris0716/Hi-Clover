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
from hicrep import hicrepSCC  # 正確匯入

parser = argparse.ArgumentParser(description='Triplet network testing module (HiRep SCC version)')
parser.add_argument('json_file', type=str, help='a file location for the json dictionary containing file paths')
parser.add_argument("data_inputs", nargs='+', help="keys from dictionary containing paths for training and validation sets.")
args = parser.parse_args()

with open(args.json_file) as json_file:
    dataset = json.load(json_file)


# ===== HiRep SCC wrapper =====
# ===== 修正後的 HiRep SCC wrapper =====
# 修改 baseline_scc.py 中的這部分
def hicrep_scc(mat1, mat2, h=1, bin_size=10000, dBPMax=500000):
    from scipy.ndimage import gaussian_filter
    from scipy.stats import pearsonr

    # 強制擠壓多餘維度（如 Batch 或 Channel），確保為 2D
    mat1 = np.squeeze(mat1)
    mat2 = np.squeeze(mat2)
    
    if mat1.ndim != 2:
        return 0.0

    # 1. 平滑化 (Smoothing)
    if h > 0:
        mat1_s = gaussian_filter(mat1, sigma=h)
        mat2_s = gaussian_filter(mat2, sigma=h)
    else:
        mat1_s, mat2_s = mat1, mat2

    # 2. 依照距離 (Strata) 分層計算相關係數[cite: 2]
    max_bin = int(dBPMax / bin_size)
    side = mat1_s.shape[0]
    corrs = []
    weights = []

    for k in range(min(max_bin, side)):
        v1 = np.diag(mat1_s, k)
        v2 = np.diag(mat2_s, k)
        
        # 排除變異數為 0 的層級以避免 Pearson 計算無效
        if np.std(v1) == 0 or np.std(v2) == 0:
            continue
            
        r, _ = pearsonr(v1, v2)
        if np.isnan(r):
            continue
            
        corrs.append(r)
        # 使用層級長度與標準差之乘積作為權重[cite: 2]
        weight = len(v1) * np.std(v1) * np.std(v2)
        weights.append(weight)

    # 3. 加權平均
    if not weights or np.sum(weights) == 0:
        return 0.0
        
    return np.average(corrs, weights=weights)

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
            # 如果 flatten，需要 reshape 回矩陣
            if m1.ndim == 1:
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
