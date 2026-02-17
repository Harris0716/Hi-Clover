import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse, json, os, time
import matplotlib.pyplot as plt

# 專案模組
from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
# from torch_plus.loss import TripletLoss # 這裡不需要引用外部 loss，因為我們手寫 Hard Mining

# 引入 reference (根據您的環境配置)
from HiSiNet.reference_dictionaries import reference_genomes

# ---------------------------------------------------------
# 1. 參數設定 (Arguments)
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Robust Triplet Training with Hard Mining')
parser.add_argument('model_name', type=str)
parser.add_argument('json_file', type=str)
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch_training', type=int, default=150)
parser.add_argument('--epoch_enforced_training', type=int, default=30)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--max_norm', type=float, default=1.0)
parser.add_argument('--outpath', type=str, default="outputs/")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mask', type=str, default="true")
parser.add_argument("data_inputs", nargs='+')

args = parser.parse_args()
os.makedirs(args.outpath, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True # 加速固定輸入尺寸的運算

# 設定隨機種子以確保實驗可重現
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ---------------------------------------------------------
# 2. 資料讀取 (Data Loading)
# ---------------------------------------------------------
with open(args.json_file) as f: config = json.load(f)

train_sets, val_sets = [], []
for n in args.data_inputs:
    ref = reference_genomes[config[n]["reference"]]
    train_sets.append(TripletHiCDataset([HiCDatasetDec.load(p) for p in config[n]["training"]], reference=ref))
    val_sets.append(TripletHiCDataset([HiCDatasetDec.load(p) for p in config[n]["validation"]], reference=ref))

# 使用 GroupedTripletHiCDataset 整合多個來源
train_loader = DataLoader(GroupedTripletHiCDataset(train_sets), batch_size=args.batch_size, 
                          sampler=RandomSampler(GroupedTripletHiCDataset(train_sets)), 
                          num_workers=4, pin_memory=True)

val_loader = DataLoader(GroupedTripletHiCDataset(val_sets), batch_size=100, 
                        sampler=SequentialSampler(GroupedTripletHiCDataset(val_sets)), 
                        num_workers=4, pin_memory=True)

# ---------------------------------------------------------
# 3. 模型與優化器 (Model & Optimizer)
# ---------------------------------------------------------
# 解析 mask 參數
mask_bool = args.mask.lower() == 'true'
model = eval("models." + args.model_name)(mask=mask_bool).to(device)

if torch.cuda.device_count() > 1: 
    model = nn.DataParallel(model)

# 根據需求選擇優化器
if args.optimizer == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
else:
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    scheduler = None

# ---------------------------------------------------------
# 4. 訓練流程 (Training Loop)
# ---------------------------------------------------------
base_path = os.path.join(args.outpath, f"{args.model_name}_{args.optimizer}_HardMining")
best_val_loss = float('inf')
patience_cnt = 0
history = {'loss': [], 'val_loss': [], 'intersect': [], 'hard_ratio': []}
best_dist = {'ap': [], 'an': []}

print(f"Start Training: {base_path}")
print(f"Configs: Margin={args.margin}, LR={args.learning_rate}, Batch={args.batch_size}")

start_time = time.time()

try:
    for epoch in range(args.epoch_training):
        # ------------------ Training Phase ------------------
        model.train()
        run_loss, run_hard_ratio = 0.0, 0.0
        
        # 用於收集 "Train + Val" 的所有距離數據
        all_ap, all_an = [], []

        for i, batch in enumerate(train_loader):
            # [安全拆解] 確保只取前三個 Tensor，避免 ValueError
            a, p, n = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            optimizer.zero_grad()
            a_out, p_out, n_out = model(a, p, n)
            
            # --- Inline Hard Mining Logic ---
            d_ap = F.pairwise_distance(a_out, p_out)
            d_an = F.pairwise_distance(a_out, n_out)
            
            # 計算 Loss: 只取大於 0 的部分 (Violated Margin)
            loss_val = F.relu(d_ap - d_an + args.margin)
            mask = loss_val > 1e-16
            
            # 如果有困難樣本，只對困難樣本取平均；否則對全體取平均(通常為0)
            loss = loss_val[mask].mean() if mask.any() else loss_val.mean()
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            
            # 記錄數據
            run_loss += loss.item()
            run_hard_ratio += mask.float().mean().item()
            
            # 收集訓練集距離 (轉回 CPU 存入 list)
            all_ap.extend(d_ap.detach().cpu().numpy())
            all_an.extend(d_an.detach().cpu().numpy())

        # ------------------ Validation Phase ------------------
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                a, p, n = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                a_out, p_out, n_out = model(a, p, n)
                
                # Validation 使用標準平均 Loss 評估
                d_ap_v = F.pairwise_distance(a_out, p_out)
                d_an_v = F.pairwise_distance(a_out, n_out)
                val_loss_sum += F.relu(d_ap_v - d_an_v + args.margin).mean().item()
                
                # 收集驗證集距離 (加入同一個 list)
                all_ap.extend(d_ap_v.cpu().numpy())
                all_an.extend(d_an_v.cpu().numpy())

        # ------------------ Metrics Calculation ------------------
        avg_loss = run_loss / len(train_loader)
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_hard = (run_hard_ratio / len(train_loader)) * 100
        
        # [修正後的核心邏輯] 計算 PDF Intersection
        # 1. 轉換為 Numpy Array
        np_ap = np.array(all_ap)
        np_an = np.array(all_an)
        
        # 2. 決定共同的 Range (涵蓋正負樣本的全域範圍)
        # 這是避免直方圖截斷的關鍵
        global_min = min(np_ap.min(), np_an.min())
        global_max = max(np_ap.max(), np_an.max())
        
        # 3. 在相同 Range 下計算直方圖 (密度函數)
        hist_p, edges = np.histogram(np_ap, bins=100, range=(global_min, global_max), density=True)
        hist_n, _ = np.histogram(np_an, bins=100, range=(global_min, global_max), density=True)
        
        # 4. 計算交集面積: sum(min(P, N)) * bin_width
        bin_width = edges[1] - edges[0]
        intersect = np.sum(np.minimum(hist_p, hist_n)) * bin_width

        # ------------------ Logging & Checkpoint ------------------
        # 更新 LR
        curr_lr = optimizer.param_groups[0]['lr']
        if scheduler: 
            scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{args.epoch_training}] "
              f"Loss: {avg_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"Hard%: {avg_hard:.1f}% | Intersect: {intersect:.4f} | LR: {curr_lr:.2e}")

        history['loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)
        history['intersect'].append(intersect)
        history['hard_ratio'].append(avg_hard)

        # 儲存策略：以 Val Loss 為準 (確保模型泛化能力)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), base_path + '_best.ckpt')
            best_dist['ap'], best_dist['an'] = list(np_ap), list(np_an)
        else:
            if epoch >= args.epoch_enforced_training:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"Early Stopping Triggered at Epoch {epoch+1}")
                    break

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

# ---------------------------------------------------------
# 5. 結果視覺化 (Visualization)
# ---------------------------------------------------------
print("Saving visualization...")
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Loss Curve
ax[0, 0].plot(history['loss'], label='Train (Hard)')
ax[0, 0].plot(history['val_loss'], label='Val (Avg)')
ax[0, 0].set_title('Loss Evolution')
ax[0, 0].legend()

# Intersection Score (Train+Val)
ax[0, 1].plot(history['intersect'], color='purple')
ax[0, 1].set_title('PDF Intersection Area (Train+Val)')
ax[0, 1].set_ylabel('Area (lower is better)')

# Hard Ratio
ax[1, 0].plot(history['hard_ratio'], color='orange')
ax[1, 0].set_title('Hard Sample Ratio (%)')

# Distance Distribution (Best Model)
if len(best_dist['ap']) > 0:
    ax[1, 1].hist(best_dist['ap'], bins=100, alpha=0.6, label='Pos', density=True, color='g')
    ax[1, 1].hist(best_dist['an'], bins=100, alpha=0.6, label='Neg', density=True, color='r')
    ax[1, 1].axvline(args.margin, color='k', linestyle='--', label=f'Margin {args.margin}')
    ax[1, 1].set_title(f'Dist Distribution (Best)\nIntersect: {min(history["intersect"]):.4f}')
    ax[1, 1].legend()

plt.tight_layout()
plt.savefig(base_path + '_result.pdf')
print(f"Done. Results saved to {base_path}_result.pdf")
print(f"Total time: {(time.time() - start_time)/60:.1f} mins")