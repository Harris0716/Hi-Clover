import pandas as pd
from HiSiNet.HiCDatasetClass import HiCDatasetDec

# Define the mapping of file paths and sample names (for NPC dataset)
# Format: (condition, replicate, training path, validation path, test path)
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

summary_rows = []

print("reading NPC dataset and calculating window numbers...")

for condition, replicate, train_path, val_path, test_path in samples_info:
    try:
        # read the split datasets
        ds_train = HiCDatasetDec.load(train_path)
        ds_val   = HiCDatasetDec.load(val_path)
        ds_test  = HiCDatasetDec.load(test_path)
        
        # get the length
        len_train = len(ds_train)
        len_val   = len(ds_val)
        len_test  = len(ds_test)
        
        # organize the data into a row
        summary_rows.append({
            "Condition": condition,
            "Replicate": replicate,
            "Train Windows": len_train,
            "Validation Windows": len_val,
            "Test Windows": len_test,
            "Total": len_train + len_val + len_test
        })
        print(f"successfully read: {condition} - {replicate}")
        
    except Exception as e:
        print(f"warn: cannot read {condition} {replicate} file. error: {e}")

# create DataFrame
df_stats = pd.DataFrame(summary_rows)

# set the table display style
pd.options.display.float_format = '{:,.0f}'.format

# show the table
print("\n=== Hi-C Dataset Summary Table (NPC Data) ===")
if not df_stats.empty:
    print(df_stats.to_string(index=False))

    # calculate the total number of rows
    total_all = df_stats.iloc[:, 2:].sum()
    print("-" * 60)
    print(f"Grand Total Across All Samples: {total_all['Total']:,}")

    # export to CSV (file name has been updated)
    df_stats.to_csv("npc_dataset_statistics.csv", index=False)
    print("\nstatistics results have been saved to npc_dataset_statistics.csv")
else:
    print("no data read, please check the path.")