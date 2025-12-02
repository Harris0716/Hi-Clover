from HiSiNet.HiCDatasetClass import HiCDatasetDec

dataset = HiCDatasetDec.load("input.mlhic")
patch, label = dataset[0] 
patch = patch.numpy()
