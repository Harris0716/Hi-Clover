#!/bin/bash
# 用同一組 baseline 參數跑三種資料：TCell, liver, NPC
# 執行前請 cd 到專案根目錄，確保 Hi-Clover/config.json 存在
#
# 用法: bash Hi-Clover/run_baseline_all_datasets.sh

BASELINE_OPTS="--batch_size 128 --epoch_training 100 --epoch_enforced_training 20 \
  --patience 10 --margin 0.5 --max_norm 1.0 \
  --seed 42 --mask true \
  --scheduler plateau --lr_patience 2"

for DATA in NPC TCell; do
  echo "=========================================="
  echo "Running baseline on: $DATA"
  echo "=========================================="
  python Hi-Clover/train_baseline_backup.py TripletLeNetBatchNorm Hi-Clover/config.json 0.01 $DATA \
    $BASELINE_OPTS \
    --outpath ./0222output_baseline_${DATA}
  echo ""
done

echo "All three datasets done."
