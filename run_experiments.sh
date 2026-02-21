#!/bin/bash
# Hi-Clover experiment commands (margin=0.5, Adagrad + Plateau)
# 執行前請 cd 到 Hi-Clover 的上一層（config.json 所在目錄）
# 例如: cd /path/to/parent && bash Hi-Clover/run_experiments.sh

# --- Baseline ---
# python Hi-Clover/train_brightjiiter.py TripletLeNetBatchNorm config.json 0.01 TCell \
#   --batch_size 128 --patience 10 --margin 0.5 --epoch_enforced_training 5 \
#   --adagrad_weight_decay 0.002 --hard_mining --outpath ./exp_baseline --seed 42 --run_eval

# --- 1. learning_rate 0.005 ---
python Hi-Clover/train_brightjiiter.py TripletLeNetBatchNorm config.json 0.005 TCell \
  --batch_size 128 --patience 10 --margin 0.5 --epoch_enforced_training 5 \
  --adagrad_weight_decay 0.002 --hard_mining --outpath ./exp_lr005 --seed 42 --run_eval

# --- 2. hard_mining 關閉 ---
python Hi-Clover/train_brightjiiter.py TripletLeNetBatchNorm config.json 0.01 TCell \
  --batch_size 128 --patience 10 --margin 0.5 --epoch_enforced_training 5 \
  --adagrad_weight_decay 0.002 --outpath ./exp_soft --seed 42 --run_eval

# --- 3. batch_size 64 ---
python Hi-Clover/train_brightjiiter.py TripletLeNetBatchNorm config.json 0.01 TCell \
  --batch_size 64 --patience 10 --margin 0.5 --epoch_enforced_training 5 \
  --adagrad_weight_decay 0.002 --hard_mining --outpath ./exp_bs64 --seed 42 --run_eval
