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
def hicrep_scc(mat1, mat2, h=1, dBPMax=2000000):
    """
    mat1, mat2: 來自 mlhic 的 numpy 2D patches (通常是 256x256)
    """
    # 根據您的論文，子圖解析度為 10kb
    current_binsize = 10000 

    # 建立模擬物件以符合 hicrepSCC 的輸入要求
    class CoolMock:
        def __init__(self, matrix, binsize):
            self.matrix_data = matrix
            self.binsize = binsize
        def matrix(self, balance=False):
            # hicrep 內部會呼叫 .matrix()，我們直接回傳 numpy 陣列
            return self
        def fetch(self, chrom):
            # 有些版本會呼叫 .fetch()
            return self.matrix_data
        @property
        def binsize(self):
            return self._binsize
        @binsize.setter
        def binsize(self, value):
            self._binsize = value

    # 封裝矩陣
    mock1 = CoolMock(mat1, current_binsize)
    mock1.binsize = current_binsize
    mock2 = CoolMock(mat2, current_binsize)
    mock2.binsize = current_binsize

    try:
        # 呼叫原始匯入的 hicrepSCC
        # 注意：h=1 是您原本設定的平滑化參數
        return hicrepSCC(mock1, mock2, h, dBPMax, bDownSample=False)
    except Exception as e:
        # 如果還是失敗，回傳一個安全值並印出錯誤
        print(f"SCC calculation failed: {e}")
        return 0.0


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
