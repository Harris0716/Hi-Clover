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
import random

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
            label = label.to(cuda, dtype=torch.float)
            output1, output2, _ = model(input1, input2, input2)
            predicted = F.pairwise_distance(output1,output2)
            distances = np.concatenate((distances, predicted.cpu().numpy()))
            labels  = np.concatenate((labels, label.cpu().numpy()))

    return distances, labels

# ======================
# Misclassified Pairs Function
# ======================
def plot_misclassified_pairs(distances, labels, intersect, Siamese, num_samples=10, out_prefix="misclassified_pair"):
    wrong_indices = np.where(((distances < intersect) & (labels == 1)) | 
                             ((distances >= intersect) & (labels == 0)))[0]

    if len(wrong_indices) == 0:
        print("⚠ 沒有找到任何分錯的 pair")
        return
    
    selected_indices = random.sample(list(wrong_indices), min(num_samples, len(wrong_indices)))
    print(f"從 {len(wrong_indices)} 個錯誤分類中，隨機挑選 {len(selected_indices)} 個繪圖...")

    for j, idx in enumerate(selected_indices):
        input1, input2, label = Siamese[idx]
        mat1 = input1.squeeze().cpu().numpy() if isinstance(input1, torch.Tensor) else np.array(input1)
        mat2 = input2.squeeze().cpu().numpy() if isinstance(input2, torch.Tensor) else np.array(input2)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(mat1, cmap="Reds", vmin=0, vmax=np.percentile(mat1, 99))
        plt.title("Matrix1")

        plt.subplot(1, 2, 2)
        plt.imshow(mat2, cmap="Reds", vmin=0, vmax=np.percentile(mat2, 99))

        plt.suptitle(f"Misclassified Pair {j+1}\nDist={distances[idx]:.2f}, True={int(label)}")
        plt.tight_layout()
        out_file = f"{out_prefix}_{j+1}.png"
        plt.savefig(out_file)
        plt.close()
        print(f"已儲存: {out_file}")


# ======================
# Model Load
# ======================
cuda = torch.device("cuda:0")
model_cls = getattr(models, args.model_name)
model = model_cls(mask=args.mask).to(cuda)
state_dict = torch.load(args.model_infile, map_location=cuda)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# ======================
# Load Test Dataset
# ======================
Siamese = GroupedHiCDataset([
    SiameseHiCDataset(
        [HiCDatasetDec.load(data_path) for data_path in (dataset[data_name]["test"])],
        reference=reference_genomes[dataset[data_name]["reference"]]
    ) for data_name in args.data_inputs
])
test_sampler = SequentialSampler(Siamese)
dataloader = DataLoader(Siamese, batch_size=100, sampler=test_sampler)

# ======================
# Evaluate Test Dataset
# ======================
distances, labels = test_triplet_by_siamese(model, dataloader)
mx = max(distances)
mn = min(distances[distances > 0])
rng = np.arange(mn, mx, (mx - mn) / 200)

a = plt.hist(distances[(labels==0)], bins=rng, density=True, label='replicates', alpha=0.5, color='#108690')
b = plt.hist(distances[(labels==1)], bins=rng, density=True, label='conditions', alpha=0.5, color='#1D1E4E')
diff = a[0] - b[0]
crossings = np.where(np.diff(np.sign(diff)))[0]
if len(crossings) > 0:
    intersect = a[1][crossings[0]]
else:
    intersect = (mn + mx) / 2
plt.axvline(intersect, color='k')
plt.legend()
plt.savefig(args.model_infile.split(".ckpt")[0]+"_test_distribution.pdf")
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

# ======================
# 額外輸出分錯樣本
# ======================
plot_misclassified_pairs(distances, labels, intersect, Siamese, num_samples=10,
                         out_prefix=args.model_infile.split(".ckpt")[0]+"_misclassified")
