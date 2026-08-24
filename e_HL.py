import os
import cv2
import numpy as np
from tqdm import tqdm
# ==========================================
# 1. 核心特征提取算法 (返回 12维 原始特征)
# ==========================================
def extract_raw_12d_features(img_path, W=224, H=224, debug_vis=False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告: 无法读取图像 {img_path}")
        return np.zeros(12, dtype=np.float32)

    vis_img = img.copy() if debug_vis else None

    # 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H_orig, W_orig = gray.shape
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # 动态阈值计算
    min_val, max_val, _, _ = cv2.minMaxLoc(blurred)
    low_t = max(max_val * 0.2, 15) if max_val >= 30 else 30
    high_t = max(max_val * 0.7, 50) if max_val >= 30 else 100

    # A. 高阈值提取光斑中心 (质心)
    _, bin_high = cv2.threshold(blurred, high_t, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_high, connectivity=8)
    mask_high = np.zeros_like(bin_high)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_high[labels == largest_idx] = 255

    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_high)
    M = cv2.moments(masked_gray)
    if M["m00"] > 0:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        cx, cy = W_orig / 2.0, H_orig / 2.0

    # 中心偏移量
    dx = cx - (W_orig / 2.0)
    dy = cy - (H_orig / 2.0)

    # B. 低阈值提取光斑形态 (椭圆拟合)
    _, bin_low = cv2.threshold(blurred, low_t, 255, cv2.THRESH_BINARY)
    num_labels_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats(bin_low, connectivity=8)

    a, b, e, theta = 0.0, 0.0, 0.0, 0.0
    r1, r2, r3, r4 = 0.0, 0.0, 0.0, 0.0

    ellipse = None  # 用于可视化

    if num_labels_l > 1:
        largest_idx_l = 1 + np.argmax(stats_l[1:, cv2.CC_STAT_AREA])
        mask_low = np.zeros_like(bin_low)
        mask_low[labels_l == largest_idx_l] = 255

        contours, _ = cv2.findContours(mask_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 5 and cv2.contourArea(cnt) > 20:
                try:
                    (ex, ey), (d1, d2), angle = cv2.fitEllipse(cnt)
                    ellipse = ((ex, ey), (d1, d2), angle)

                    # 过滤异常狭长的拟合结果
                    if max(d1, d2) / (min(d1, d2) + 1e-6) < 6.0:
                        if d1 > d2:
                            a, b, theta = d1 / 2, d2 / 2, angle
                        else:
                            a, b, theta = d2 / 2, d1 / 2, angle + 90

                        if a > 0:
                            e = np.sqrt(1 - (b / a) ** 2) if b < a else 0.0

                        # 计算极径 (0, 90, 180, 270度)
                        theta_rad = np.radians(theta)
                        radii = []
                        for ang in [0, 90, 180, 270]:
                            phi = np.radians(ang)
                            term1 = (b * np.cos(phi - theta_rad)) ** 2
                            term2 = (a * np.sin(phi - theta_rad)) ** 2
                            r = (a * b) / np.sqrt(term1 + term2 + 1e-6)
                            radii.append(r)
                        r1, r2, r3, r4 = radii
                except:
                    pass

    # 可视化绘制
    if debug_vis:
        if ellipse is not None:
            cv2.ellipse(vis_img, ellipse, (0, 255, 0), 2)  # 绿色椭圆
        cv2.drawMarker(vis_img, (int(cx), int(cy)), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
        os.makedirs("debug_vis_HR", exist_ok=True)
        base_name = os.path.basename(img_path)
        save_path = os.path.join("debug_vis_HR", base_name)
        cv2.imwrite(save_path, vis_img)

    # 组装 12D 特征
    raw_feats = np.array([cx, cy, dx, dy, a, b, e, theta, r1, r2, r3, r4], dtype=np.float32)
    return raw_feats


def build_feature_cache():
    # 路径配置
    TRAIN_DIR = r"C:\Users\diuneilaomu\Desktop\data\ltrain"
    TEST_DIR = r"C:\Users\diuneilaomu\Desktop\data\ltest\ltest"

    TRAIN_SAVE_PATH = "trainHL.npy"
    TEST_SAVE_PATH = "testHL.npy"
    STATS_SAVE_PATH = "feature_stats_r.npy"  # 保存均值和方差供以后部署用

    # --- Step 1: 提取训练集原始特征并计算统计量 ---
    print("\n[1/4] 正在提取训练集原始特征...")
    train_files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.jpg', '.png'))]
    train_raw_dict = {}

    for fname in tqdm(train_files):
        img_path = os.path.join(TRAIN_DIR, fname)
        raw_feat = extract_raw_12d_features(img_path, debug_vis=False)
        train_raw_dict[fname] = raw_feat

    # 计算均值和标准差
    all_train_feats = np.vstack(list(train_raw_dict.values()))
    mean_val = np.mean(all_train_feats, axis=0)
    std_val = np.std(all_train_feats, axis=0) + 1e-6  # 加 1e-6 防止除以零

    print(f"训练集特征均值: {mean_val}")

    # --- Step 2: 归一化并保存训练集 ---
    print("\n[2/4] 正在归一化并保存训练集缓存...")
    train_norm_dict = {}
    for fname, raw_feat in train_raw_dict.items():
        norm_feat = (raw_feat - mean_val) / std_val
        train_norm_dict[fname] = norm_feat

    np.save(TRAIN_SAVE_PATH, train_norm_dict)
    print(f"已保存: {TRAIN_SAVE_PATH} (共 {len(train_norm_dict)} 条数据)")

    # --- Step 3: 提取测试集并应用训练集的统计量 ---
    print("\n[3/4] 正在提取测试集并进行归一化...")
    test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.png'))]
    test_norm_dict = {}

    for fname in tqdm(test_files):
        img_path = os.path.join(TEST_DIR, fname)
        raw_feat = extract_raw_12d_features(img_path, debug_vis=True)  # 可视化测试集
        # 【关键】测试集必须使用训练集的 mean 和 std 进行归一化
        norm_feat = (raw_feat - mean_val) / std_val
        test_norm_dict[fname] = norm_feat

    np.save(TEST_SAVE_PATH, test_norm_dict)
    print(f"已保存: {TEST_SAVE_PATH} (共 {len(test_norm_dict)} 条数据)")

    # --- Step 4: 保存统计量 (可选) ---
    print("\n[4/4] 正在保存均值与方差文件...")
    np.save(STATS_SAVE_PATH, {'mean': mean_val, 'std': std_val})
    print(f"已保存统计量至: {STATS_SAVE_PATH} (未来推理预测时使用)")

    print("\n✅ 所有缓存文件生成完毕！现在您可以畅快地运行 train_baseline.py 了！")


if __name__ == "__main__":
    build_feature_cache()