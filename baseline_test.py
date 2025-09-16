from HiSiNet.HiCDatasetClass import HiCDatasetDec
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
# if replicates
dataset1 = HiCDatasetDec.load("input1.mlhic")
dataset2 = HiCDatasetDec.load("input2.mlhic")

patch1, label1 = dataset1[0]
patch2, label2 = dataset2[0] 

if isinstance(patch1, torch.Tensor) and isinstance(patch2, torch.Tensor):
    patch1 = patch1.numpy()
    patch2 = patch2.numpy()  

if patch1.ndim == 3 and patch2 == 3:
    patch1 = patch1[0]
    patch2 = patch2[0]

print(patch1.shape)
print(patch2.shape)

# plt.imshow(patch, cmap="Reds", interpolation="nearest")
# plt.colorbar(label="Hi-C contact count")
# plt.title(f"First patch (label={label})")
# plt.show()

print(F.pairwise_distance(patch1,patch2))