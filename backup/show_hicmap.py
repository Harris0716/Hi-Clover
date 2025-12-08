from HiSiNet.HiCDatasetClass import HiCDatasetDec

dataset = HiCDatasetDec.load("input.mlhic")
patch, label = dataset[0] 

import matplotlib.pyplot as plt
import torch

if isinstance(patch, torch.Tensor):
    patch = patch.numpy()

if patch.ndim == 3:
    patch = patch[0]

plt.imshow(patch, cmap="Reds", interpolation="nearest")
plt.colorbar(label="Hi-C contact count")
plt.title(f"First patch (label={label})")
plt.show()
