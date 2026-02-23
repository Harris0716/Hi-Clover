import pandas as pd
from HiSiNet.HiCDatasetClass import HiCDatasetDec

samples_info = [
    # 1. Double Positive (DP) - R1
    ("CD69negDP", "R1", 
     "/work/u1696810/tcell_data/CD69negDPWTR1.mlhic", 
     "/work/u1696810/tcell_data/CD69negDPWTR1_validation.mlhic", 
     "/work/u1696810/tcell_data/CD69negDPWTR1_test.mlhic"),
    
    # 2. Double Positive (DP) - R2
    ("CD69negDP", "R2", 
     "/work/u1696810/tcell_data/CD69negDPWTR2.mlhic", 
     "/work/u1696810/tcell_data/CD69negDPWTR2_validation.mlhic", 
     "/work/u1696810/tcell_data/CD69negDPWTR2_test.mlhic"),
    
    # 3. CD4 Single Positive (CD4 SP) - R1
    ("CD69posCD4SP", "R1", 
     "/work/u1696810/tcell_data/CD69posCD4SPWTR1.mlhic", 
     "/work/u1696810/tcell_data/CD69posCD4SPWTR1_validation.mlhic", 
     "/work/u1696810/tcell_data/CD69posCD4SPWTR1_test.mlhic"),
    
    # 4. CD4 Single Positive (CD4 SP) - R2
    ("CD69posCD4SP", "R2", 
     "/work/u1696810/tcell_data/CD69posCD4SPWTR2.mlhic", 
     "/work/u1696810/tcell_data/CD69posCD4SPWTR2_validation.mlhic", 
     "/work/u1696810/tcell_data/CD69posCD4SPWTR2_test.mlhic"),
]

summary_rows = []

print("reading T-Cell dataset and calculating window numbers...")

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
print("\n=== Hi-C Dataset Summary Table (T-Cell Data) ===")
if not df_stats.empty:
    print(df_stats.to_string(index=False))

    # calculate the total number of rows
    total_all = df_stats.iloc[:, 2:].sum()
    print("-" * 60)
    print(f"Grand Total Across All Samples: {total_all['Total']:,}")

    # export to CSV (file name has been updated)
    df_stats.to_csv("tcell_dataset_statistics.csv", index=False)
    print("\nstatistics results have been saved to tcell_dataset_statistics.csv")
else:
    print("no data read, please check the path.")