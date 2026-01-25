#!/bin/bash
# original code

# 定義要測試的 Margin 列表
MARGINS=(0.3 0.5 0.8 1.0)

# 固定參數設定
MODEL="TripletLeNet"
LR=0.001
SEED=42
DATA="NIPBL"
BATCH=128

for MARGIN in "${MARGINS[@]}"
do
    echo "============================================"
    echo "正在執行 Margin: $MARGIN"
    echo "============================================"

    # 1. 定義輸出路徑
    OUT_DIR="./v1_outputs_margins/margin_${MARGIN}/"
    mkdir -p $OUT_DIR

    # 2. 訓練 (使用您提供的 train_v1.py 邏輯)
    python Hi-Clover/train.py $MODEL config.json $LR $DATA \
        --batch_size $BATCH \
        --epoch_training 100 \
        --epoch_enforced_training 20 \
        --outpath $OUT_DIR \
        --seed $SEED \
        --margin $MARGIN \
        --mask true

    # 3. 定義模型權重路徑
    CKPT="${OUT_DIR}${MODEL}_${LR}_${BATCH}_${SEED}_${MARGIN}_best.ckpt"

    # 4. 測試
    echo "開始測試 Margin $MARGIN 的最佳模型..."
    python Hi-Clover/test.py $MODEL config.json $CKPT $DATA --mask true
done

echo "所有 Margin 實驗已完成！"