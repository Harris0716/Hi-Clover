#!/bin/bash
# 用實驗 2 的參數跑 liver 和 NPC（TripletLeNetBatchNorm + weight_decay 0.001）
# 執行: bash Hi-Clover/run_exp2_liver_npc.sh

OPTS="--batch_size 128 --epoch_training 100 --epoch_enforced_training 20 \
  --patience 5 --margin 0.5 --max_norm 1.0 --seed 42 --mask true \
  --weight_decay 0.001"

for DATA in liver NPC; do
  echo "=========================================="
  echo "Running Exp2 config on: $DATA"
  echo "=========================================="
  python Hi-Clover/train_baseline_backup.py TripletLeNetBatchNorm Hi-Clover/config.json 0.01 $DATA \
    $OPTS --outpath ./0223exp_${DATA}_BN_wd0.001
  echo ""
done

echo "Liver and NPC done (Exp2 params)."
