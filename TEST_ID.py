from HiSiNet.HiCDatasetClass import HiCDatasetDec

# 載入你的 TAM 檔案
data = HiCDatasetDec.load('liver_data/TAM_R1.mlhic') #

# 印出 class_id
print(f"TAM 的 class_id 是: {data.metadata['class_id']}") #