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

parser = argparse.ArgumentParser(description='Triplet network testing module')
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

# 加載 dataset 配置文件
with open(args.json_file) as json_file:
    dataset = json.load(json_file)

# 設置設備
if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("Using {} GPUs".format(n_gpu))
else:
    device = torch.device("cpu")
    print("Using CPU")

# 定義測試函數來計算 triplet 的距離
def test_triplet_similarity(model, dataloader):
    anchor_distances = np.array([])
    negative_distances = np.array([])
    labels = np.array([])

    for _, data in enumerate(dataloader):
        anchor, positive, negative, label = data
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # 使用 Triplet 網絡進行前向傳播
        anchor_out, positive_out, negative_out = model(anchor, positive, negative)

        # 計算 anchor 和 positive 之間的距離 (使用 torch.norm)
        positive_distance = torch.norm(anchor_out - positive_out, p=2, dim=1)  # Euclidean 距離
        # 計算 anchor 和 negative 之間的距離 (使用 torch.norm)
        negative_distance = torch.norm(anchor_out - negative_out, p=2, dim=1)  # Euclidean 距離

        # 記錄距離和標籤
        anchor_distances = np.concatenate((anchor_distances, positive_distance.cpu().detach().numpy()))
        negative_distances = np.concatenate((negative_distances, negative_distance.cpu().detach().numpy()))
        labels = np.concatenate((labels, label.cpu().detach().numpy()))

    return anchor_distances, negative_distances, labels

# 加載模型
model = eval("models." + args.model_name)(mask=args.mask)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load(args.model_infile, weights_only=True))
model.eval()

# 加載訓練和驗證數據集
Siamese = GroupedHiCDataset([TripletHiCDataset([HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["training"] + dataset[data_name]["validation"])],
                                reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs])

test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=100, sampler=test_sampler)

# 使用 Triplet 測試函數測試模型
anchor_distances, negative_distances, labels = test_triplet_similarity(model, dataloader)

# 計算最大和最小距離
mx = max(np.max(anchor_distances), np.max(negative_distances))
mn = min(np.min(anchor_distances[anchor_distances > 0]), np.min(negative_distances[negative_distances > 0]))
rng = np.arange(mn, mx, (mx - mn) / 200)

# 計算分類準確率
global_rate = sum(((anchor_distances < negative_distances) == (labels == 0))) / len(anchor_distances)
TR_rate = sum((anchor_distances < negative_distances) & (labels == 0)) / sum(labels == 0)
BC_rate = sum((anchor_distances > negative_distances) & (labels == 1)) / sum(labels == 1)

print('Global rate: {:.4f}, Replicate rate: {:.4f}, Condition rate: {:.4f}'
      .format(global_rate, TR_rate, BC_rate))

# 繪製測試集的距離分佈
plt.hist(anchor_distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
plt.hist(negative_distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
plt.axvline(np.mean(anchor_distances), color='k')
plt.xticks(np.arange(0, np.ceil(mx), 1))
plt.title("Triplet Network Distance for test on: {}".format(", ".join(args.data_inputs)))
plt.ylabel("Density")
plt.xlabel("Euclidean Distance of Representation")
plt.legend()
plt.savefig(args.model_infile.split(".ckpt")[0] + "_test_triplet_distribution.pdf")
plt.close()
