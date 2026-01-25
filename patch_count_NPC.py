import pandas as pd
from HiSiNet.HiCDatasetClass import HiCDatasetDec

# === 修改區域開始 ===
# 定義檔案路徑與樣本名稱的對應關係 (針對 NPC 資料集)
# 格式: (條件, 複製組, 訓練集路徑, 驗證集路徑, 測試集路徑)
samples_info = [
    # D4 Auxin-treated Replicate 1
    ("D4_Auxin", "R1", 
     "/work/u1696810/NPC_data/d4auxrep1_10kb.mlhic", 
     "/work/u1696810/NPC_data/d4_validation_auxrep1_10kb.mlhic", 
     "/work/u1696810/NPC_data/d4_test_auxrep1_10kb.mlhic"),
    
    # D4 Auxin-treated Replicate 2
    ("D4_Auxin", "R2", 
     "/work/u1696810/NPC_data/d4auxrep2_10kb.mlhic", 
     "/work/u1696810/NPC_data/d4_validation_auxrep2_10kb.mlhic", 
     "/work/u1696810/NPC_data/d4_test_auxrep2_10kb.mlhic"),
    
    # D4 Control Replicate 1
    ("D4_Control", "R1", 
     "/work/u1696810/NPC_data/d4ctlrep1_10kb.mlhic", 
     "/work/u1696810/NPC_data/d4_validation_ctlrep1_10kb.mlhic", 
     "/work/u1696810/NPC_data/d4_test_ctlrep1_10kb.mlhic"),
    
    # D4 Control Replicate 2
    ("D4_Control", "R2", 
     "/work/u1696810/NPC_data/d4ctlrep2_10kb.mlhic", 
     "/work/u1696810/NPC_data/d4_validation_ctlrep2_10kb.mlhic", 
     "/work/u1696810/NPC_data/d4_test_ctlrep2_10kb.mlhic"),
]
# === 修改區域結束 ===

summary_rows = []

print("正在讀取 NPC 數據集並計算窗口數量...")

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
        print(f"成功讀取: {condition} - {replicate}")
        
    except Exception as e:
        print(f"warn: 無法讀取 {condition} {replicate} 的檔案。錯誤: {e}")

# 建立 DataFrame
df_stats = pd.DataFrame(summary_rows)

# 設定表格顯示樣式
pd.options.display.float_format = '{:,.0f}'.format

# 顯示表格
print("\n=== Hi-C Dataset Summary Table (NPC Data) ===")
if not df_stats.empty:
    print(df_stats.to_string(index=False))

    # 計算整體的總計列
    total_all = df_stats.iloc[:, 2:].sum()
    print("-" * 60)
    print(f"Grand Total Across All Samples: {total_all['Total']:,}")

    # 匯出成 CSV 備用 (檔名已更新)
    df_stats.to_csv("npc_dataset_statistics.csv", index=False)
    print("\n統計結果已保存至 npc_dataset_statistics.csv")
else:
    print("沒有讀取到任何數據，請檢查路徑。")