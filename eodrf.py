import os
import cv2
import numpy as np
from tqdm import tqdm
def ransac_fit_ellipse(pts, max_trials=200, residual_threshold=1.0, min_samples=5):
    """
    自定义 RANSAC 算法，用于对清洗后的平滑外凸点集进行鲁棒拟合
    """
    n_points = len(pts)
    if n_points < 5:
        return None, 0, n_points
    if n_points <= min_samples:
        try:
            return cv2.fitEllipse(pts), n_points, n_points
        except:
            return None, 0, n_points

    best_inliers_count = 0
    best_inlier_pts = None

    for _ in range(max_trials):
        idx = np.random.choice(n_points, min_samples, replace=False)
        sample_pts = pts[idx]

        try:
            ellipse = cv2.fitEllipse(sample_pts)
            (cx, cy), (d1, d2), angle = ellipse
            a, b = d1 / 2.0, d2 / 2.0
            if a < 1 or b < 1:
                continue
        except:
            continue

        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a

        phi = np.arctan2(y_rot, x_rot)
        r_point = np.hypot(x_rot, y_rot)

        r_ellipse = (a * b) / np.sqrt((b * np.cos(phi)) ** 2 + (a * np.sin(phi)) ** 2 + 1e-6)
        errors = np.abs(r_point - r_ellipse)

        inliers_mask = errors < residual_threshold
        inliers_count = np.sum(inliers_mask)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_inlier_pts = pts[inliers_mask]

    if best_inlier_pts is not None and len(best_inlier_pts) >= 5:
        try:
            return cv2.fitEllipse(best_inlier_pts), best_inliers_count, n_points
        except:
            return None, 0, n_points
    return None, 0, n_points

# 曲率与凸性分析
def extract_robust_integrated_curves(contour, epsilon_factor=0.005, sharp_turn_thresh=np.pi / 3,
                                     straight_angle_thresh=172):
    """
    整合 PDR 算法与 R 算法的过滤思想：
    1. 宏观：RDP 逼近 + 曲率符号(凸性) + 急转弯切割
    2. 微观：动态跨度滑动窗口 -> 剔除遮挡造成的直线边缘
    """
    N_orig = len(contour)
    if N_orig < 15:
        return contour.reshape(-1, 2)

    contour_pts = contour.reshape(-1, 2)
    peri = cv2.arcLength(contour, True)

    # ==========================================
    # Step 1: PDR  提取平滑外凸掩码
    # ==========================================
    convex_mask = np.zeros(N_orig, dtype=bool)
    approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True).reshape(-1, 2)
    N_approx = len(approx)

    if N_approx >= 3:
        area = cv2.contourArea(approx, oriented=True)
        is_ccw = area > 0
        valid_anchors = []
        current_segment = []

        for i in range(N_approx):
            p_prev = approx[(i - 1) % N_approx]
            p_curr = approx[i]
            p_next = approx[(i + 1) % N_approx]

            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            is_convex = (cross_product > 0) if is_ccw else (cross_product < 0)

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            theta = 0
            if norm_v1 * norm_v2 > 1e-6:
                cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
                theta = np.arccos(cos_theta)

            if not is_convex or theta > sharp_turn_thresh:
                if len(current_segment) > 0:
                    valid_anchors.extend(current_segment)
                    current_segment = []
            else:
                current_segment.append(p_curr)

        if len(current_segment) > 0:
            valid_anchors.extend(current_segment)

        for pt in valid_anchors:
            dists = np.linalg.norm(contour_pts - pt, axis=1)
            convex_mask[dists < (epsilon_factor * peri * 2)] = True
    else:
        convex_mask[:] = True

        # ==========================================
    # Step 2: 动态 dist
    # ==========================================
    straight_mask = np.zeros(N_orig, dtype=bool)
    # 🌟 策略二核心：动态计算跨度
    # 基准：长度 180 的轮廓，dist 取 6 比较合适 (180/30 = 6)
    #设置保底最小跨度 4，最大跨度 10 (防止轮廓过大导致跨度越过半个椭圆)
    dynamic_dist = max(4, min(int(N_orig / 30), 10))

    for i in range(N_orig):
        # 使用动态 dist 
        p1 = contour_pts[(i - dynamic_dist) % N_orig]
        p2 = contour_pts[i]
        p3 = contour_pts[(i + dynamic_dist) % N_orig]

        v1 = p1 - p2
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue

        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))

        # 真正的曲线点夹角会小于阈值，平直遮挡线会被剔除
        if angle < straight_angle_thresh:
            straight_mask[i] = True

    # ==========================================
    # Step 3: 取掩码交集，输出纯净点集
    # ==========================================
    final_mask = convex_mask & straight_mask

    return contour_pts[final_mask]
# ==========================================
# VLP 特征提取主干
# ==========================================
def extract_raw_12d_features(img_path, W=224, H=224, debug_vis=True):
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告: 无法读取图像 {img_path}")
        return np.zeros(12, dtype=np.float32), np.zeros((60, 2), dtype=np.float32), 0, 0

    vis_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H_orig, W_orig = gray.shape

    # 使用高斯模糊
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    min_val, max_val, _, _ = cv2.minMaxLoc(blurred)
    low_t = max(max_val * 0.2, 15) if max_val >= 30 else 30

    cx, cy = W_orig / 2.0, H_orig / 2.0
    dx, dy = 0.0, 0.0
    a, b, e, theta = 0.0, 0.0, 0.0, 0.0
    r1, r2, r3, r4 = 0.0, 0.0, 0.0, 0.0
    fixed_pts = np.full((60, 2), -1.0, dtype=np.float32)
    inlier_count, total_count = 0, 0

    # 结合 Canny与阈值分割 获取高斯光斑边缘
    _, bin_low = cv2.threshold(blurred, low_t, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(bin_low, 50, 150)

    # 查找边缘轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        # 取最长边缘（排除杂散噪点）
        cnt = max(contours, key=len)

        # 兜底质心
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

        # === 提取平滑且外凸的边缘曲线 ===

        pts = extract_robust_integrated_curves(
            cnt,
            epsilon_factor=0.005,
            sharp_turn_thresh=np.pi /2,
            straight_angle_thresh=174)
        n_points = len(pts)

        if n_points > 0:
            norm_pts = pts / np.array([W_orig, H_orig], dtype=np.float32)
            if n_points >= 60:
                idx = np.random.choice(n_points, 60, replace=False)
                fixed_pts = norm_pts[idx]
            else:
                fixed_pts[:n_points] = norm_pts

        for pt in pts:
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

        if n_points >= 5:
            try:
                ellipse, inlier_count, total_count = ransac_fit_ellipse(pts, max_trials=200, residual_threshold=1,
                                                                        min_samples=5)
                if ellipse is not None:
                    (ex, ey), (d1, d2), angle = ellipse
                    cx, cy = ex, ey
                    cv2.ellipse(vis_img, ellipse, (0, 255, 0), 2)

                    if max(d1, d2) / (min(d1, d2) + 1e-6) < 6.0:
                        if d1 > d2:
                            a, b, theta = d1 / 2, d2 / 2, angle
                        else:
                            a, b, theta = d2 / 2, d1 / 2, angle + 90
                        theta = theta % 180.0
                        if a > 0:
                            e = np.sqrt(1 - (b / a) ** 2) if b < a else 0.0

                        theta_rad = np.radians(theta)
                        for ang in [0, 90, 180, 270]:
                            phi = np.radians(ang)
                            term1 = (b * np.cos(phi - theta_rad)) ** 2
                            term2 = (a * np.sin(phi - theta_rad)) ** 2
                            r = (a * b) / np.sqrt(term1 + term2 + 1e-6)
                            if ang == 0:
                                r1 = r
                            elif ang == 90:
                                r2 = r
                            elif ang == 180:
                                r3 = r
                            elif ang == 270:
                                r4 = r

                        cv2.putText(vis_img, f"Fit OK: Inliers {inlier_count}/{total_count}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(vis_img, "Too Narrow", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    cv2.putText(vis_img, "RANSAC Failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as ex:
                print(f"拟合异常: {ex}")
        else:
            cv2.putText(vis_img, f"Filtered < 5 pts", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    dx = cx - (W_orig / 2.0)
    dy = cy - (H_orig / 2.0)
    cv2.drawMarker(vis_img, (int(cx), int(cy)), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)

    if debug_vis:
        os.makedirs("debug_vis", exist_ok=True)
        base_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join("debug_vis", base_name), vis_img)

    raw_feats = np.array([cx, cy, dx, dy, a, b, e, theta, r1, r2, r3, r4], dtype=np.float32)
    return raw_feats, fixed_pts, inlier_count, total_count

def build_feature_cache():
    TRAIN_DIR =  r"C:\Users\diuneilaomu\Desktop\data\ltrain\ltrain"
    TEST_DIR =  r"C:\Users\diuneilaomu\Desktop\data\ltest_occ2\ltest_occ2"

    TRAIN_SAVE_PATH =r"C:\Users\diuneilaomu\Desktop\A\A\ext\feat\xunlian.npy"
    TEST_SAVE_PATH =  r"C:\Users\diuneilaomu\Desktop\A\A\ext\feat\l2_ed.npy"

    os.makedirs("feat", exist_ok=True)

    # --- Step 1: 提取训练集原始特征 ---
    train_files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.jpg', '.png'))]
    train_raw_dict = {}
    train_pts_dict = {}

    for fname in tqdm(train_files, desc="Processing Train Set"):
        img_path = os.path.join(TRAIN_DIR, fname)
        raw_feat, pts, _, _ = extract_raw_12d_features(img_path, debug_vis=False)
        train_raw_dict[fname] = raw_feat
        train_pts_dict[fname] = pts

    all_train_feats = np.vstack(list(train_raw_dict.values()))
    mean_val = np.mean(all_train_feats, axis=0)
    std_val = np.std(all_train_feats, axis=0) + 1e-6

    print(f"\n训练集特征均值:\n{mean_val}")

    # --- Step 2: 归一化并保存训练集 ---
    train_norm_dict = {}
    for fname, raw_feat in train_raw_dict.items():
        norm_feat = (raw_feat - mean_val) / std_val
        train_norm_dict[fname] = norm_feat
    np.save(TRAIN_SAVE_PATH, train_norm_dict)
    print(f"已保存: {TRAIN_SAVE_PATH} (共 {len(train_norm_dict)} 条数据)")

    # --- Step 3: 提取测试集并计算内点比例 ---
    test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.png'))]
    test_norm_dict = {}
    test_pts_dict = {}

    # 用于存放测试集中每张图的内点比例
    test_ratios = []
    for fname in tqdm(test_files, desc="Processing Test Set"):
        img_path = os.path.join(TEST_DIR, fname)
        # 接收内点数和总点数
        raw_feat, pts, inlier_c, total_c = extract_raw_12d_features(img_path, debug_vis=True)
        test_pts_dict[fname] = pts
        norm_feat = (raw_feat - mean_val) / std_val
        test_norm_dict[fname] = norm_feat

        # 计算并记录比例
        if total_c > 0:
            test_ratios.append(inlier_c / total_c)

    np.save(TEST_SAVE_PATH, test_norm_dict)
    print(f"已保存: {TEST_SAVE_PATH} (共 {len(test_norm_dict)} 条数据)")

    # 打印最终的平均内点比例
    if len(test_ratios) > 0:
        avg_ratio = np.mean(test_ratios)
        print(f"\n📊 测试集平均内点比例 (Inlier Ratio): {avg_ratio:.4f} ({avg_ratio * 100:.2f}%)")
    else:
        print("\n📊 警告: 测试集中未提取到有效的 RANSAC 拟合数据。")

    print("\n✅ 所有缓存文件生成完毕！")

if __name__ == "__main__":
    build_feature_cache()
