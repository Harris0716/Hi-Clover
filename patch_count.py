import pandas as pd
from HiSiNet.HiCDatasetClass import HiCDatasetDec

# 定義檔案路徑與樣本名稱的對應關係
# 格式: (條件, 複製組, 訓練集路徑, 驗證集路徑, 測試集路徑)
samples_info = [
    ("TAM", "R1", "/work/u1696810/liver_data/TAM_R1.mlhic", "/work/u1696810/liver_data/TAM_validation_R1.mlhic", "/work/u1696810/liver_data/TAM_test_R1.mlhic"),
    ("TAM", "R2", "/work/u1696810/liver_data/TAM_R2.mlhic", "/work/u1696810/liver_data/TAM_validation_R2.mlhic", "/work/u1696810/liver_data/TAM_test_R2.mlhic"),
    ("NIPBL",  "R1", "/work/u1696810/liver_data/NIPBL_R1.mlhic", "/work/u1696810/liver_data/NIPBL_validation_R1.mlhic", "/work/u1696810/liver_data/NIPBL_test_R1.mlhic"),
    ("NIPBL",  "R2", "/work/u1696810/liver_data/NIPBL_R2.mlhic", "/work/u1696810/liver_data/NIPBL_validation_R2.mlhic", "/work/u1696810/liver_data/NIPBL_test_R2.mlhic"),
]

summary_rows = []

print("正在讀取數據集並計算窗口數量...")

for condition, replicate, train_path, val_path, test_path in samples_info:
    try:
        # 讀取各個分割數據集
        ds_train = HiCDatasetDec.load(train_path)
        ds_val   = HiCDatasetDec.load(val_path)
        ds_test  = HiCDatasetDec.load(test_path)
        
        # 取得長度
        len_train = len(ds_train)
        len_val   = len(ds_val)
        len_test  = len(ds_test)
        
        # 整理成一列數據
        summary_rows.append({
            "Condition": condition,
            "Replicate": replicate,
            "Train Windows": len_train,
            "Validation Windows": len_val,
            "Test Windows": len_test,
            "Total": len_train + len_val + len_test
        })
    except Exception as e:
        print(f"warn: can not read {condition} {replicate} file。error: {e}")

# 建立 DataFrame
df_stats = pd.DataFrame(summary_rows)

# 設定表格顯示樣式
pd.options.display.float_format = '{:,.0f}'.format

# 顯示表格
print("\n=== Hi-C Dataset Summary Table (TAM & NIPBL) ===")
print(df_stats.to_string(index=False))

# 也可以計算整體的總計列
total_all = df_stats.iloc[:, 2:].sum()
print("-" * 60)
print(f"Grand Total Across All Samples: {total_all['Total']:,}")

# 匯出成 CSV 備用
df_stats.to_csv("dataset_statistics.csv", index=False)