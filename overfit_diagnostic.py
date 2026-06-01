# ============================================================
# Overfit Diagnostic
# 目的：用現有 backbone 故意 overfit 極少量 T Cell triplet，
#       判斷「T Cell 學不起來」是 backbone 容量問題，還是泛化/資料問題。
#
# 判讀：
#   - train loss 能壓到接近 0 (例如 < 0.02)      -> backbone 容量「足夠」，問題在泛化或資料訊號
#   - train loss 卡在高位 (例如停在 0.15 左右降不下去) -> backbone 容量「不足」，要動 backbone
#
# 注意：這裡刻意關掉所有正則化（weight decay=0、augmentation 全關、
#       並把 dropout 設為 eval 行為），讓模型「全力記住」這一小撮資料。
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse, json

from HiSiNet.HiCDatasetClass import HiCDatasetDec, TripletHiCDataset, GroupedTripletHiCDataset
import HiSiNet.models as models
from torch_plus.loss import TripletLoss
from HiSiNet.reference_dictionaries import reference_genomes

parser = argparse.ArgumentParser(description='Overfit diagnostic')
parser.add_argument('--model_name', type=str, default='TripletLeNetBatchNormSE_Joint')
parser.add_argument('--json_file', type=str, required=True)
parser.add_argument('--data_inputs', nargs='+', required=True)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--mask', action='store_true')
parser.add_argument('--seed', type=int, default=42)
# 診斷專用
parser.add_argument('--n_samples', type=int, default=256,
                    help='故意 overfit 的 triplet 數量（取一小撮）')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=300,
                    help='在這一小撮資料上反覆訓練幾輪')
parser.add_argument('--log_every', type=int, default=10)
args = parser.parse_args()

print("-" * 50)
print("Overfit Diagnostic — Configuration")
for k, v in vars(args).items():
    print(f"  {k}: {v}")
print("-" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ------------------------------------------------------------
# 載入資料，取一小撮 triplet（不做任何 augmentation）
# ------------------------------------------------------------
with open(args.json_file) as f:
    dataset_config = json.load(f)

full_dataset = GroupedTripletHiCDataset([
    TripletHiCDataset(
        [HiCDatasetDec.load(p) for p in dataset_config[n]["training"]],
        reference=reference_genomes[dataset_config[n]["reference"]]
    ) for n in args.data_inputs])   # h_flip / random_flip 預設 False，刻意不增強

total = len(full_dataset)
n = min(args.n_samples, total)
print(f"Total triplets available: {total:,}; using a fixed subset of {n}")

# 固定取前 n 個（seed 已固定，可重現）；一次搬上 GPU 反覆使用
idxs = list(range(n))
a_list, p_list, n_list = [], [], []
for idx in idxs:
    item = full_dataset[idx]
    a_list.append(item[0])
    p_list.append(item[1])
    n_list.append(item[2])

a = torch.stack(a_list).to(device)   # [n, 1, 256, 256]
p = torch.stack(p_list).to(device)
ng = torch.stack(n_list).to(device)
print(f"Loaded subset tensors: a={tuple(a.shape)}")

# ------------------------------------------------------------
# 建立模型；關掉 dropout（用 eval 模式跑 BN/Dropout 會不更新統計，
# 但我們要 BN 正常更新，所以用 train 模式，改成手動把 Dropout 機率設 0）
# ------------------------------------------------------------
model = eval("models." + args.model_name)(
    mask=args.mask, embedding_dim=args.embedding_dim).to(device)

# 把所有 Dropout 的機率設為 0，排除 dropout 干擾「全力記憶」
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.0

model.train()   # 保持 train 模式讓 BN 正常運作

criterion = TripletLoss(margin=args.margin)
# weight_decay = 0，純粹看擬合能力
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

# ------------------------------------------------------------
# 在這一小撮資料上反覆訓練
# ------------------------------------------------------------
print("\nEpoch |   Loss   | d(a,p) | d(a,n) | log-ratio")
print("-" * 50)

best_loss = float('inf')
for epoch in range(args.epochs):
    optimizer.zero_grad()
    a_out, p_out, n_out = model(a, p, ng)
    loss = criterion(a_out, p_out, n_out)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % args.log_every == 0 or epoch == 0:
        with torch.no_grad():
            d_ap = F.pairwise_distance(a_out, p_out).mean().item()
            d_an = F.pairwise_distance(a_out, n_out).mean().item()
            lr_ratio = np.log10((d_an + 1e-6) / (d_ap + 1e-6))
        print(f"{epoch+1:5d} | {loss.item():.5f} | {d_ap:.4f} | {d_an:.4f} | {lr_ratio:+.4f}")

    best_loss = min(best_loss, loss.item())

# ------------------------------------------------------------
# 判讀
# ------------------------------------------------------------
print("-" * 50)
print(f"Lowest train loss reached: {best_loss:.5f}")
print("-" * 50)
if best_loss < 0.02:
    print("判讀：train loss 壓到接近 0 → backbone 容量【足夠】。")
    print("      T Cell 學不起來的瓶頸不在架構容量，而在泛化或資料訊號本身。")
    print("      → 動 backbone 不一定有用，建議查資料/泛化（augmentation、樣本對設計）。")
elif best_loss < 0.10:
    print("判讀：train loss 有明顯下降但未壓到極低 → backbone 容量【勉強】。")
    print("      加深/加寬/dilation 可能有邊際幫助，但別期待質變。")
else:
    print("判讀：train loss 卡在高位降不下去 → backbone 容量【不足】。")
    print("      確認瓶頸在架構，動 backbone（如 dilation / 加深）才有意義。")
print("-" * 50)