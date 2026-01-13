#!/bin/bash

# --- 固定參數設定 ---
MODEL="TripletLeNet"
CONFIG="config.json"
LR="0.001"
DATA="NIPBL"
BS=128
EPOCH=100
ENFORCED=20
SEED=42
WD="0.0001"
MASK="true"

# --- 要測試的 Margin 列表 (在此修改你要跑的數值) ---
# 例如：0.2, 0.5, 0.8, 1.0
MARGINS=(0.3 0.5 0.8 1.0)

# --- 執行迴圈 ---
for M in "${MARGINS[@]}"
do
    # 自動產生輸出資料夾名稱，包含日期與 Margin 數值
    # 格式：日期_LR_BS_Seed_Margin_Data
    DATE="0113"
    CURRENT_OUTPATH="./${DATE}_${LR}_${BS}_${SEED}_M${M}_${DATA}/"

    echo "============================================================"
    echo "正在執行 Experiment: Margin = $M"
    echo "輸出路徑: $CURRENT_OUTPATH"
    echo "============================================================"

    # 執行 Python 整合腳本
    python run_all.py $MODEL $CONFIG $LR $DATA \
        --batch_size $BS \
        --epoch_training $EPOCH \
        --epoch_enforced_training $ENFORCED \
        --outpath "$CURRENT_OUTPATH" \
        --seed $SEED \
        --margin $M \
        --weight_decay $WD \
        --mask $MASK

    echo -e "Experiment with Margin $M finished.\n"
done

echo "所有實驗已完成！"