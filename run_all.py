# run train.py and test.py sequentially
import subprocess
import os

# --- 配置參數 ---
config = {
    "model_name": "TripletLeNet",
    "json_file": "config.json",
    "lr": 0.001,
    "data_input": "NIPBL",
    "batch_size": 128,
    "epoch_training": 100,
    "epoch_enforced_training": 20,
    "outpath": "./output_path/",
    "seed": 42,
    "margin": 0.5,
    "weight_decay": 0.0001,
    "mask": "true"
}

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# 1. 執行訓練
train_cmd = [
    "python", "Hi-Clover/train_v2_graph.py",
    config["model_name"], config["json_file"], str(config["lr"]), config["data_input"],
    "--batch_size", str(config["batch_size"]),
    "--epoch_training", str(config["epoch_training"]),
    "--epoch_enforced_training", str(config["epoch_enforced_training"]),
    "--outpath", config["outpath"],
    "--seed", str(config["seed"]),
    "--margin", str(config["margin"]),
    "--weight_decay", str(config["weight_decay"]),
    "--mask", config["mask"]
]
run_command(train_cmd)

# 2. 構建 Checkpoint 路徑
# 根據你的 train 代碼: f_info = f"{args.model_name}_lr{args.learning_rate}_bs{args.batch_size}_m{args.margin}_wd{args.weight_decay}"
ckpt_info = f"{config['model_name']}_lr{config['lr']}_bs{config['batch_size']}_m{config['margin']}_wd{config['weight_decay']}_best.ckpt"
ckpt_path = os.path.join(config["outpath"], ckpt_info)

# 3. 執行測試
if os.path.exists(ckpt_path):
    test_cmd = [
        "python", "Hi-Clover/test.py",
        config["model_name"], config["json_file"], ckpt_path, config["data_input"],
        "--mask", config["mask"]
    ]
    run_command(test_cmd)
else:
    print(f"Error: Could not find {ckpt_path}")