#!/bin/bash
# 0223 TCell 三組實驗
# 執行: bash Hi-Clover/run_0223_exp_tcell.sh

OPTS="--batch_size 128 --epoch_training 100 --epoch_enforced_training 20 \
  --patience 5 --margin 0.5 --max_norm 1.0 --seed 42 --mask true"

echo "=========================================="
echo "[1/3] TripletLeNet (no BN)"
echo "=========================================="
python Hi-Clover/train_baseline_backup.py TripletLeNet Hi-Clover/config.json 0.01 TCell \
  $OPTS --outpath ./0223exp_tcell_noBN

echo ""
echo "=========================================="
echo "[2/3] TripletLeNetBatchNorm + weight_decay 0.001"
echo "=========================================="
python Hi-Clover/train_baseline_backup.py TripletLeNetBatchNorm Hi-Clover/config.json 0.01 TCell \
  $OPTS --outpath ./0223exp_tcell_BN_wd0.001 --weight_decay 0.001

echo ""
echo "=========================================="
echo "[3/3] TripletLeNet (no BN) + weight_decay 0.001"
echo "=========================================="
python Hi-Clover/train_baseline_backup.py TripletLeNet Hi-Clover/config.json 0.01 TCell \
  $OPTS --outpath ./0223exp_tcell_noBN_wd0.001 --weight_decay 0.001

echo ""
echo "All 3 experiments done."
