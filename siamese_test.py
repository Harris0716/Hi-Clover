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
parser.add_argument('model_name',  type=str,
                    help='a string indicating a model from models')
parser.add_argument('json_file',  type=str,
                    help='a file location for the json dictionary containing file paths')
parser.add_argument('model_infile',  type=str,
                    help='a string indicating the model location file')
parser.add_argument('--mask',  type=bool, default=False,
                    help='an argument specifying if the diagonal should be masked')
parser.add_argument("data_inputs", nargs='+', help="keys from dictionary containing paths for training and validation sets.")

args = parser.parse_args()

with open(args.json_file) as json_file:
    dataset = json.load(json_file)


# 定義從訓練好的 Triplet 模型中提取 CNN 作為 Siamese 的基礎
class SiameseFromTriplet(nn.Module):
    def __init__(self, triplet_model, mask=False):
        super(SiameseFromTriplet, self).__init__()
        self.encoder = triplet_model
        self.mask = mask

    def forward(self, input1, input2):
        # 兩個輸入的特徵提取
        output1 = self.encoder(input1)
        output2 = self.encoder(input2)

        return output1, output2

# 載入訓練好的 Triplet 模型
cuda = torch.device("cuda:0")
triplet_model = eval("models." + args.model_name)(mask=args.mask).to(cuda)
state_dict = torch.load(args.model_infile, weights_only=True)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
triplet_model.load_state_dict(new_state_dict)
triplet_model.eval()

# 基於 Triplet 模型構建 Siamese 模型
siamese_model = SiameseFromTriplet(triplet_model).to(cuda)

# 定義測試函數來計算相似度
def test_model(model, dataloader):
    distances = np.array([])
    labels = np.array([])
    for _, data in enumerate(dataloader):
        input1, input2, label = data
        input1, input2 = input1.to(cuda), input2.to(cuda)
        label = label.type(torch.FloatTensor).to(cuda)
        
        # 計算兩個樣本的輸出
        output1, output2 = model(input1, input2)
        
        # 計算兩者的歐式距離
        predicted = F.pairwise_distance(output1, output2)
        distances = np.concatenate((distances, predicted.cpu().detach().numpy()))
        labels = np.concatenate((labels, label.cpu().detach().numpy()))
    
    return distances, labels

# 加載訓練和驗證數據集
Siamese = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["training"] + dataset[data_name]["validation"])],
             reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs])
test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=100, sampler=test_sampler)

# 測試模型並計算距離
distances, labels = test_model(siamese_model, dataloader)

# 計算最大和最小距離
mx = max(distances)
mn = min(distances[distances > 0])
rng = np.arange(mn, mx, (mx - mn) / 200)

a = plt.hist(distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
b = plt.hist(distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
intersect = a[1][np.argwhere(np.diff(np.sign(a[0] - b[0])))[0]]
plt.axvline(intersect, color='k')
plt.xticks(np.arange(0, np.ceil(mx), 1))
plt.legend()
plt.title("distance of train and validation from " + args.model_infile.split("/")[-1] + " on: " + ", ".join(args.data_inputs))
plt.ylabel("density")
plt.xlabel("euclidean distance of representation")
plt.savefig(args.model_infile.split(".ckpt")[0] + "_train_distribution.pdf")
plt.close()

# 測試數據集
Siamese_test = GroupedHiCDataset([SiameseHiCDataset([HiCDatasetDec.load(data_path) for data_path in dataset[data_name]["test"]],
             reference=reference_genomes[dataset[data_name]["reference"]]) for data_name in args.data_inputs])
test_sampler = SequentialSampler(Siamese_test)
dataloader_test = DataLoader(Siamese_test, batch_size=100, sampler=test_sampler)

# 測試模型
distances, labels = test_model(siamese_model, dataloader_test)

# 計算分類準確率
global_rate = sum(((distances < intersect) == (labels == 0))) / len(distances)
TR_rate = sum((distances < intersect) & (labels == 0)) / sum(labels == 0)
BC_rate = sum((distances > intersect) & (labels == 1)) / sum(labels == 1)

print('global rate: {:.4f}, replicate rate: {:.4f}, condition rate: {:.4f}'
      .format(global_rate, TR_rate, BC_rate))

# 繪製測試集的距離分佈
a = plt.hist(distances[(labels == 0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
b = plt.hist(distances[(labels == 1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
plt.axvline(intersect, color='k')
plt.xticks(np.arange(0, np.ceil(mx), 1))
plt.title("distance of test from " + args.model_infile.split("/")[-1] + " on: " + ", ".join(args.data_inputs))
plt.ylabel("density")
plt.xlabel("euclidean distance of representation")
plt.legend()
plt.savefig(args.model_infile.split(".ckpt")[0] + "_test_distribution.pdf")
plt.close()
