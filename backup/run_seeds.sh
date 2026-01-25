#!/bin/bash
SEEDS=(42 123 1025 2026 7)
MODEL="TripletLeNet"
LR=0.001
MARGIN=0.5
BATCH = 128

for SEED in "${SEEDS[@]}"
do
    OUT_DIR="./outputs_seeds/seed_${SEED}/"
    mkdir -p $OUT_DIR

    # 1. 訓練
    python Hi-Clover/train_v2.py $MODEL config.json $LR NIPBL \
        --outpath $OUT_DIR --seed $SEED --margin $MARGIN --mask true

    # 2. 定義模型路徑 (需與 train_v2.py 產生的檔名一致)
    CKPT="${OUT_DIR}${MODEL}_${LR}_${BATCH}_${SEED}_${MARGIN}_best.ckpt"

    # 3. 測試
    python Hi-Clover/test.py $MODEL config.json $CKPT NIPBL --mask true
done