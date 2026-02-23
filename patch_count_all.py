import argparse
import pandas as pd
from HiSiNet.HiCDatasetClass import HiCDatasetDec


DATASETS = {
    "tcell": {
        "read_msg": "reading T-Cell dataset and calculating window numbers...",
        "title": "Hi-C Dataset Summary Table (T-Cell Data)",
        "csv": "tcell_dataset_statistics.csv",
        "samples_info": [
            # 1. Double Positive (DP) - R1
            (
                "CD69negDP",
                "R1",
                "/work/u1696810/tcell_data/CD69negDPWTR1.mlhic",
                "/work/u1696810/tcell_data/CD69negDPWTR1_validation.mlhic",
                "/work/u1696810/tcell_data/CD69negDPWTR1_test.mlhic",
            ),
            # 2. Double Positive (DP) - R2
            (
                "CD69negDP",
                "R2",
                "/work/u1696810/tcell_data/CD69negDPWTR2.mlhic",
                "/work/u1696810/tcell_data/CD69negDPWTR2_validation.mlhic",
                "/work/u1696810/tcell_data/CD69negDPWTR2_test.mlhic",
            ),
            # 3. CD4 Single Positive (CD4 SP) - R1
            (
                "CD69posCD4SP",
                "R1",
                "/work/u1696810/tcell_data/CD69posCD4SPWTR1.mlhic",
                "/work/u1696810/tcell_data/CD69posCD4SPWTR1_validation.mlhic",
                "/work/u1696810/tcell_data/CD69posCD4SPWTR1_test.mlhic",
            ),
            # 4. CD4 Single Positive (CD4 SP) - R2
            (
                "CD69posCD4SP",
                "R2",
                "/work/u1696810/tcell_data/CD69posCD4SPWTR2.mlhic",
                "/work/u1696810/tcell_data/CD69posCD4SPWTR2_validation.mlhic",
                "/work/u1696810/tcell_data/CD69posCD4SPWTR2_test.mlhic",
            ),
        ],
    },
    "liver": {
        "read_msg": "reading liver dataset and calculating window numbers...",
        "title": "Hi-C Dataset Summary Table (TAM & NIPBL)",
        "csv": "liver_dataset_statistics.csv",
        "samples_info": [
            ("TAM", "R1",
             "/work/u1696810/liver_data/TAM_R1.mlhic",
             "/work/u1696810/liver_data/TAM_validation_R1.mlhic",
             "/work/u1696810/liver_data/TAM_test_R1.mlhic"),
            ("TAM", "R2",
             "/work/u1696810/liver_data/TAM_R2.mlhic",
             "/work/u1696810/liver_data/TAM_validation_R2.mlhic",
             "/work/u1696810/liver_data/TAM_test_R2.mlhic"),
            ("NIPBL", "R1",
             "/work/u1696810/liver_data/NIPBL_R1.mlhic",
             "/work/u1696810/liver_data/NIPBL_validation_R1.mlhic",
             "/work/u1696810/liver_data/NIPBL_test_R1.mlhic"),
            ("NIPBL", "R2",
             "/work/u1696810/liver_data/NIPBL_R2.mlhic",
             "/work/u1696810/liver_data/NIPBL_validation_R2.mlhic",
             "/work/u1696810/liver_data/NIPBL_test_R2.mlhic"),
        ],
    },
    "npc": {
        "read_msg": "reading NPC dataset and calculating window numbers...",
        "title": "Hi-C Dataset Summary Table (NPC Data)",
        "csv": "npc_dataset_statistics.csv",
        "samples_info": [
            # D4 Auxin-treated Replicate 1
            (
                "D4_Auxin",
                "R1",
                "/work/u1696810/NPC_data/d4auxrep1_10kb.mlhic",
                "/work/u1696810/NPC_data/d4_validation_auxrep1_10kb.mlhic",
                "/work/u1696810/NPC_data/d4_test_auxrep1_10kb.mlhic",
            ),
            # D4 Auxin-treated Replicate 2
            (
                "D4_Auxin",
                "R2",
                "/work/u1696810/NPC_data/d4auxrep2_10kb.mlhic",
                "/work/u1696810/NPC_data/d4_validation_auxrep2_10kb.mlhic",
                "/work/u1696810/NPC_data/d4_test_auxrep2_10kb.mlhic",
            ),
            # D4 Control Replicate 1
            (
                "D4_Control",
                "R1",
                "/work/u1696810/NPC_data/d4ctlrep1_10kb.mlhic",
                "/work/u1696810/NPC_data/d4_validation_ctlrep1_10kb.mlhic",
                "/work/u1696810/NPC_data/d4_test_ctlrep1_10kb.mlhic",
            ),
            # D4 Control Replicate 2
            (
                "D4_Control",
                "R2",
                "/work/u1696810/NPC_data/d4ctlrep2_10kb.mlhic",
                "/work/u1696810/NPC_data/d4_validation_ctlrep2_10kb.mlhic",
                "/work/u1696810/NPC_data/d4_test_ctlrep2_10kb.mlhic",
            ),
        ],
    },
}


def summarize_one_dataset(name: str, cfg: dict) -> None:
    samples_info = cfg["samples_info"]
    summary_rows = []

    print(cfg["read_msg"])

    for condition, replicate, train_path, val_path, test_path in samples_info:
        try:
            ds_train = HiCDatasetDec.load(train_path)
            ds_val = HiCDatasetDec.load(val_path)
            ds_test = HiCDatasetDec.load(test_path)

            len_train = len(ds_train)
            len_val = len(ds_val)
            len_test = len(ds_test)

            summary_rows.append(
                {
                    "Condition": condition,
                    "Replicate": replicate,
                    "Train Windows": len_train,
                    "Validation Windows": len_val,
                    "Test Windows": len_test,
                    "Total": len_train + len_val + len_test,
                }
            )
            print(f"successfully read: {condition} - {replicate}")
        except Exception as e:  # noqa: BLE001
            print(f"warn: cannot read {condition} {replicate} file. error: {e}")

    df_stats = pd.DataFrame(summary_rows)
    pd.options.display.float_format = "{:,.0f}".format

    print(f"\n=== {cfg['title']} ===")
    if df_stats.empty:
        print("no data read, please check the path.")
        return

    print(df_stats.to_string(index=False))

    total_all = df_stats.iloc[:, 2:].sum()
    print("-" * 60)
    print(f"Grand Total Across All Samples: {total_all['Total']:,}")

    csv_name = cfg["csv"]
    df_stats.to_csv(csv_name, index=False)
    print(f"\nstatistics results have been saved to {csv_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Hi-C window counts for TCell, liver, and NPC datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["tcell", "liver", "npc", "all"],
        help="Which dataset to summarize (default: all).",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for name in ["tcell", "liver", "npc"]:
            summarize_one_dataset(name, DATASETS[name])
            print("\n" + "=" * 80 + "\n")
    else:
        summarize_one_dataset(args.dataset, DATASETS[args.dataset])


if __name__ == "__main__":
    main()

