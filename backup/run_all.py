import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Hi-Clover Pipeline: Train and Test')
    
    # ---------------------------------------------------------
    # 1. 定義參數 (與 train 腳本一致)
    # ---------------------------------------------------------
    parser.add_argument('model_name', type=str, help='Model name (e.g., TripletLeNet)')
    parser.add_argument('json_file', type=str, help='JSON config file path')
    parser.add_argument('learning_rate', type=float, help='Learning rate')
    parser.add_argument("data_inputs", nargs='+', help="Keys for training and validation")
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch_training', type=int, default=100)
    parser.add_argument('--epoch_enforced_training', type=int, default=20)
    parser.add_argument('--outpath', type=str, default="outputs/")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask', type=str, default="true", choices=["true", "false"])
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 2. 執行訓練 (Train)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 1: Starting Training...")
    print("="*50)

    train_cmd = [
        "python", "Hi-Clover/train.py",
        args.model_name,
        args.json_file,
        str(args.learning_rate)
    ]
    # 加入 data_inputs (多個)
    train_cmd.extend(args.data_inputs)
    
    # 加入其他選用參數
    train_cmd.extend([
        "--batch_size", str(args.batch_size),
        "--epoch_training", str(args.epoch_training),
        "--epoch_enforced_training", str(args.epoch_enforced_training),
        "--outpath", args.outpath,
        "--seed", str(args.seed),
        "--margin", str(args.margin),
        "--weight_decay", str(args.weight_decay),
        "--mask", args.mask
    ])

    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Training failed with exit code {e.returncode}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 3. 構建 Checkpoint 路徑
    # ---------------------------------------------------------
    # 邏輯必須與 train_v2_graph.py 產生的 f_info 一致
    # f_info = f"{args.model_name}_lr{args.learning_rate}_bs{args.batch_size}_m{args.margin}_wd{args.weight_decay}"
    ckpt_name = f"{args.model_name}_lr{args.learning_rate}_bs{args.batch_size}_m{args.margin}_wd{args.weight_decay}_best.ckpt"
    ckpt_path = os.path.join(args.outpath, ckpt_name)

    # ---------------------------------------------------------
    # 4. 執行測試 (Test)
    # ---------------------------------------------------------
    if not os.path.exists(ckpt_path):
        print(f"\n[Error] Could not find checkpoint at {ckpt_path}")
        print("Please check if the outpath or naming logic matches.")
        sys.exit(1)

    print("\n" + "="*50)
    print(f"STEP 2: Starting Evaluation using {ckpt_name}...")
    print("="*50)

    test_cmd = [
        "python", "Hi-Clover/test.py",
        args.model_name,
        args.json_file,
        ckpt_path
    ]
    test_cmd.extend(args.data_inputs)
    test_cmd.extend(["--mask", args.mask])

    try:
        subprocess.run(test_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Testing failed with exit code {e.returncode}")
        sys.exit(1)

    print("\n" + "="*50)
    print("Pipeline Complete!")
    print("="*50)

if __name__ == "__main__":
    main()