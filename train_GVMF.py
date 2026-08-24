import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from MLP import CNN_Phy_dir
import random
from scipy.interpolate import griddata
from zhijie import CNN_Phy_zhijie
def extract_features(self, img, phy_feat):
    x_img = self.flat(self.avgpool(self.backbone(img)))  # 512
    x_phy = self.phy_pre(phy_feat)  # 256
    x_img2 = self.cnn_pre(x_img)    # 256
    return x_img, x_img2, x_phy

def seed_everything(seed=100):
    """固定所有的随机种子，确保结果可复现"""
    # 1. 固定 Python 内置的随机种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. 固定 Numpy 的随机种子
    np.random.seed(seed)

    # 3. 固定 PyTorch 的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 4. 固定 cuDNN 的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(seed=100)

# --- 2. 数据集定义 ---
class LED_Dset_Phy(Dataset):
    def __init__(self, image_dir, phy_cache_path, is_train=True, transform=None):
        self.image_dir = image_dir
        self.is_train = is_train
        self.files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.phy_cache = np.load(phy_cache_path, allow_pickle=True).item()
        self.transform = transform # 传入变换

    def _parse_label_from_filename(self, fname):
        try:
            x, y = map(float, fname.replace('.png', '').replace('.jpg', '').split('_')[:2])
            return torch.tensor([x, y]).float()
        except:
            return torch.zeros(2)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.image_dir, fname)

        # 1. 读取并预处理
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(gray, (224, 224))
        img_3c = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        # 2. 应用图像增强
        if self.transform:
            img = self.transform(img_3c)
        else:
            img = torch.from_numpy(img_3c).float().permute(2, 0, 1) / 255.0

        # 3. 物理特征处理
        if fname in self.phy_cache:
            phy_feat = torch.from_numpy(self.phy_cache[fname]).float()
        else:
            phy_feat = torch.zeros(12).float()

        real_coord = self._parse_label_from_filename(fname)
        target_norm = real_coord / 2.25

        return {"img": img, "phy_feat": phy_feat, "target": target_norm, "real_xy": real_coord, "fname": fname}

# --- 3. 评估函数 ---
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    all_preds = []
    all_gts = []

    for batch in loader:
        img = batch['img'].to(device)
        phy = batch['phy_feat'].to(device)
        target = batch['target'].to(device)

        # 预测出归一化的坐标 [0, 1]
        pred_norm = model(img, phy)
        loss = criterion(pred_norm, target)
        losses.append(loss.item())

        # 还原回真实物理尺度 2.25m
        pred_xy = pred_norm.cpu().numpy() * 2.25
        gt_xy = batch['real_xy'].numpy()

        all_preds.append(pred_xy)
        all_gts.append(gt_xy)

    all_preds = np.vstack(all_preds)
    all_gts = np.vstack(all_gts)

    # 核心：计算欧式距离误差 (以米为单位)
    errors = np.linalg.norm(all_preds - all_gts, axis=1)

    return {
        "loss": np.mean(losses),
        "mean_err": errors.mean(),
        "max_err": errors.max(),
        "preds": all_preds,
        "gts": all_gts,
        "errors": errors
    }

import pandas as pd

@torch.no_grad()
def export_features_to_csv(model, loader, save_path="features.csv"):
    model.eval()

    rows = []

    for batch in tqdm(loader, desc="Extracting features"):
        img = batch['img'].to(device)
        phy = batch['phy_feat'].to(device)

        x_img_raw, x_img, x_phy = model.extract_features(img, phy)

        x_img_raw = x_img_raw.cpu().numpy()
        x_img = x_img.cpu().numpy()
        x_phy = x_phy.cpu().numpy()
        phy_raw = batch['phy_feat'].numpy()
        labels = batch['real_xy'].numpy()
        fnames = batch['fname']

        B = x_img.shape[0]

        for i in range(B):
            row = {}

            # 文件名
            row["fname"] = fnames[i]

            # label
            row["x"] = labels[i][0]
            row["y"] = labels[i][1]

            # 原始12维
            for j in range(12):
                row[f"phy_{j}"] = phy_raw[i][j]

            # CNN raw 512维
            for j in range(x_img_raw.shape[1]):
                row[f"cnn_raw_{j}"] = x_img_raw[i][j]

            # CNN压缩后256维
            for j in range(x_img.shape[1]):
                row[f"cnn_256_{j}"] = x_img[i][j]

            # phy embedding 256维
            for j in range(x_phy.shape[1]):
                row[f"phy_emb_{j}"] = x_phy[i][j]

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✅已保存: {save_path}")
# --- 4. 主训练函数 ---
def train_modelA():
    # 路径配置
    train_img_dir = r"C:\Users\diuneilaomu\Desktop\data\ltrain"
    train_phy_path = r"C:\Users\diuneilaomu\Desktop\A\A\ext\trainHL.npy"
    test_img_dir = r"C:\Users\diuneilaomu\Desktop\data\ltest"
    test_phy_path = r"C:\Users\diuneilaomu\Desktop\A\A\ext\testHL.npy"

    batch_size = 64
    epochs = 100
    lr = 1e-3

    print("正在加载数据集...")
    train_set = LED_Dset_Phy(
        image_dir=train_img_dir,
        phy_cache_path=train_phy_path,
        is_train=True,
    )
    val_set = LED_Dset_Phy(
        image_dir=test_img_dir,
        phy_cache_path=test_phy_path,
        is_train=False,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"训练集: {len(train_set)} 张, 测试集: {len(val_set)} 张")

    model = CNN_Phy_zhijie(phy_dim=12).to(device)

    criterion = nn.SmoothL1Loss(beta=0.1).to(device)

    # 优化器与学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best_mean_err = float('inf')
    best_val_res = None
    history_train_loss = []
    history_val_err = []

    for ep in range(epochs):
        model.train()
        train_loss_accum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep + 1}/{epochs}")

        for batch in pbar:
            img = batch['img'].to(device)
            phy = batch['phy_feat'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                pred = model(img, phy)
                loss = criterion(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item()
            pbar.set_postfix(train_loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        # 记录平均训练损失
        avg_train_loss = train_loss_accum / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # 验证
        val_res = evaluate(model, val_loader, criterion)
        history_val_err.append(val_res['mean_err'])

        print(
            f"[VAL] loss={val_res['loss']:.6f} | mean_err={val_res['mean_err']:.4f}m | max_err={val_res['max_err']:.4f}m")

        # 保存最优模型
        if val_res['mean_err'] < best_mean_err:
            best_mean_err = val_res['mean_err']
            best_val_res = val_res
            torch.save(model.state_dict(), "FM/FM_zhijie.pth")
            print(f"🚀保存最优模型! 当前最佳误差: {best_mean_err:.4f}m")

        scheduler.step()

    print(f"\n✅ 训练结束！最终最佳平均误差: {best_mean_err:.4f}m")

if __name__ == "__main__":
    train_modelA()