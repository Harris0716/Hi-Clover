from HiSiNet.HiCDatasetClass import HiCDatasetDec

# 載入你的 TAM 檔案
tam = HiCDatasetDec.load('/work/u1696810/liver_data/TAM_R1.mlhic') 
nipbl = HiCDatasetDec.load('/work/u1696810/liver_data/NIPBL_R1.mlhic') 
# 印出 class_id
print(f"TAM 的 class_id 是: {data.metadata['class_id']}") 
print(f"NIPBL 的 class_id 是: {data.metadata['class_id']}") 