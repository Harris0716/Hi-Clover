import pandas as pd
from HiSiNet.HiCDatasetClass import HiCDatasetDec

# Define the mapping of file paths and sample names
# Format: (condition, replicate, training path, validation path, test path)
samples_info = [
    ("TAM", "R1", "/work/u1696810/liver_data/TAM_R1.mlhic", "/work/u1696810/liver_data/TAM_validation_R1.mlhic", "/work/u1696810/liver_data/TAM_test_R1.mlhic"),
    ("TAM", "R2", "/work/u1696810/liver_data/TAM_R2.mlhic", "/work/u1696810/liver_data/TAM_validation_R2.mlhic", "/work/u1696810/liver_data/TAM_test_R2.mlhic"),
    ("NIPBL",  "R1", "/work/u1696810/liver_data/NIPBL_R1.mlhic", "/work/u1696810/liver_data/NIPBL_validation_R1.mlhic", "/work/u1696810/liver_data/NIPBL_test_R1.mlhic"),
    ("NIPBL",  "R2", "/work/u1696810/liver_data/NIPBL_R2.mlhic", "/work/u1696810/liver_data/NIPBL_validation_R2.mlhic", "/work/u1696810/liver_data/NIPBL_test_R2.mlhic"),
]

summary_rows = []

print("reading dataset and calculating window numbers...")

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
    except Exception as e:
        print(f"warn: can not read {condition} {replicate} file. error: {e}")

# create DataFrame
df_stats = pd.DataFrame(summary_rows)

# set the table display style
pd.options.display.float_format = '{:,.0f}'.format

# show the table
print("\n=== Hi-C Dataset Summary Table (TAM & NIPBL) ===")
print(df_stats.to_string(index=False))

# calculate the total number of rows
total_all = df_stats.iloc[:, 2:].sum()
print("-" * 60)
print(f"Grand Total Across All Samples: {total_all['Total']:,}")

# export to CSV (file name has been updated)
df_stats.to_csv("dataset_statistics.csv", index=False)