from HiSiNet.HiCDatasetClass import HiCDatasetDec

dataset_train = HiCDatasetDec.load("liver_data/TAM_R1.mlhic")
dataset_val = HiCDatasetDec.load("liver_data/TAM_validation_R1.mlhic")
dataset_test = HiCDatasetDec.load("liver_data/TAM_test_R1.mlhic")

print(len(dataset_train))
print(len(dataset_val))
print(len(dataset_test))

print('total patches: ', len(dataset_train) + len(dataset_val) + len(dataset_test))
