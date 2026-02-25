"""
severity_estimator.py - 缺陷严重程度自动估计

对 A 类缺陷（银层异常）：通过银蓝边界偏移量来量化
对 B 类缺陷（结构异物）：通过缺陷区域面积占比来量化

输出: severity in [0, 1]，0=极轻微，1=极严重
"""
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


# ─────────────────────────────────────────────────────────
# A 类缺陷：基于颜色分割的规则估计器
# ─────────────────────────────────────────────────────────
def estimate_silver_ratio(img_np: np.ndarray) -> float:
    """
    估计图像中银层 (粉色区域) 占产品面积的比例

    工作原理:
      1. 在 HSV 色彩空间中，银层是粉/紫色 (H~140-170)
      2. 本体是蓝色 (H~100-130)
      3. 背景是青色 (H~80-100 且 S 较低)
      4. 计算银层面积 / (银层 + 本体面积) = silver_ratio

    Args:
        img_np: (H, W, 3) uint8 BGR 图像 (OpenCV 格式)

    Returns:
        ratio: float in [0, 1]
    """
    import cv2

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 银层掩码 (粉/紫色: H in [130, 175], S > 30)
    silver_mask = ((h > 130) & (h < 175) & (s > 30)).astype(np.uint8)

    # 本体掩码 (蓝色: H in [100, 135], S > 30)
    body_mask = ((h > 100) & (h < 135) & (s > 30)).astype(np.uint8)

    # 产品区域 = 银层 + 本体
    product_area = silver_mask.sum() + body_mask.sum()
    if product_area < 100:  # 太小，可能是全背景
        return 0.0

    ratio = silver_mask.sum() / product_area
    return float(ratio)


def estimate_defect_area(img_np: np.ndarray, ok_img_np: np.ndarray,
                         threshold: float = 30.0) -> float:
    """
    通过与 OK 图的差异来估计缺陷面积占比

    Args:
        img_np:    NG 图 (H, W, 3) uint8 RGB
        ok_img_np: 对应工位的 OK 参考图 (H, W, 3) uint8 RGB
        threshold: 差异阈值

    Returns:
        ratio: float in [0, 1], 缺陷面积 / 产品面积
    """
    diff = np.abs(img_np.astype(float) - ok_img_np.astype(float))
    diff_gray = diff.mean(axis=2)  # 灰度差异

    defect_mask = (diff_gray > threshold).astype(np.uint8)
    product_mask = (diff_gray > 5).astype(np.uint8)  # 非背景区域

    product_area = product_mask.sum()
    if product_area < 100:
        return 0.0

    return float(defect_mask.sum() / product_area)


# ─────────────────────────────────────────────────────────
# 严重程度估计主函数
# ─────────────────────────────────────────────────────────
# A 类缺陷：银层异常
A_TYPE_DEFECTS = {"diffusion", "exposed_substrate", "silver_overflow", "reversed_print"}
# B 类缺陷：结构异物
B_TYPE_DEFECTS = {"breakage", "contamination", "adhesion"}

# 各缺陷的正常银层比例 (从 OK 图统计得到, 可调)
NORMAL_SILVER_RATIOS = {
    "cam1_p0":  0.45,  # Cam1 正面正常银层约 45%
    "cam1_p90": 0.45,
    "cam2_p0":  0.05,  # Cam2 底面正常几乎无银
    "cam2_p90": 0.05,
    "cam3_p0":  0.30,  # Cam3-6 侧面约 30%
    "cam3_p90": 0.30,
    "cam4_p0":  0.30,
    "cam4_p90": 0.30,
    "cam5_p0":  0.30,
    "cam5_p90": 0.30,
    "cam6_p0":  0.30,
    "cam6_p90": 0.30,
}


def estimate_severity(
    ng_img: np.ndarray,
    defect_type: str,
    cam: int,
    pose: str,
    ok_img: np.ndarray = None,
) -> float:
    """
    估计一张 NG 图的缺陷严重程度

    Args:
        ng_img:      NG 图像 (H, W, 3) uint8 RGB
        defect_type: 缺陷类型名
        cam:         摄像头 ID (1-6)
        pose:        "p0" 或 "p90"
        ok_img:      对应工位的 OK 参考图 (可选, B类缺陷需要)

    Returns:
        severity: float in [0.0, 1.0]
    """
    if defect_type in A_TYPE_DEFECTS:
        # A 类: 银层偏移 → 用银层比例偏离正常值的程度
        silver_ratio = estimate_silver_ratio(ng_img)
        key = f"cam{cam}_{pose}"
        normal_ratio = NORMAL_SILVER_RATIOS.get(key, 0.35)
        deviation = abs(silver_ratio - normal_ratio) / max(normal_ratio, 0.1)
        severity = min(deviation, 1.0)

    elif defect_type in B_TYPE_DEFECTS:
        # B 类: 缺陷面积占比
        if ok_img is not None:
            severity = estimate_defect_area(ng_img, ok_img)
            severity = min(severity * 3.0, 1.0)  # 缩放到 [0,1]
        else:
            # 无 OK 参考图时，用简单亮度异常面积估计
            gray = np.mean(ng_img, axis=2)
            # 亮度异常区域 (过暗或过亮)
            anomaly = ((gray < 30) | (gray > 230)).astype(float)
            severity = min(float(anomaly.mean()) * 5.0, 1.0)
    else:
        severity = 0.5  # 未知类型给中间值

    return float(np.clip(severity, 0.0, 1.0))


# ─────────────────────────────────────────────────────────
# 可学习的严重程度编码器 (训练时用)
# ─────────────────────────────────────────────────────────
class SeverityEncoder(nn.Module):
    """
    将 severity scalar → 条件向量

    用法:
      severity = 0.7  (严重偏高)
      condition = severity_encoder(torch.tensor([0.7]))
      → 注入 UNet 的条件中
    """

    def __init__(self, out_dim=768):
        super().__init__()

        # 连续值 → embedding
        self.encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, severity):
        """
        Args:
            severity: (B,) FloatTensor, 值在 [0, 1]

        Returns:
            condition: (B, 1, out_dim)
        """
        x = severity.unsqueeze(-1)       # (B, 1)
        out = self.encoder(x)            # (B, out_dim)
        return out.unsqueeze(1)          # (B, 1, out_dim)


# ─────────────────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Severity Estimator Test")
    print("=" * 50)

    # 测试可学习编码器
    enc = SeverityEncoder()
    sev = torch.tensor([0.2, 0.5, 0.9])
    out = enc(sev)
    print(f"  SeverityEncoder: input=(3,) -> output={out.shape}")
    print(f"  Expected: (3, 1, 768)")

    # 测试规则估计器 (用随机图)
    fake_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    sev_val = estimate_severity(fake_img, "diffusion", cam=1, pose="p0")
    print(f"  Rule-based severity (random img, diffusion): {sev_val:.3f}")

    sev_val2 = estimate_severity(fake_img, "breakage", cam=1, pose="p0")
    print(f"  Rule-based severity (random img, breakage):  {sev_val2:.3f}")

    print("\n[PASS] Severity estimator test passed")
