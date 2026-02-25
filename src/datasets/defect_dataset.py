"""
DefectDataset: 私有电感 AOI 数据集 DataLoader

文件命名规范:
  OK : cam{X}_{pose}_ok_{index:05d}.png
  NG : cam{X}_{pose}_{defect}_{index:05d}.png

标签全部从文件名解析，无需额外标注文件。
"""
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────
# 常量定义
# ─────────────────────────────────────────────
DEFECT_TYPES = [
    "adhesion",
    "breakage",
    "contamination",
    "diffusion",
    "exposed_substrate",
    "reversed_print",
    "silver_overflow",
]

CAM_IDS = [1, 2, 3, 4, 5, 6]
POSES   = ["p0", "p90"]

DEFECT2IDX = {d: i for i, d in enumerate(DEFECT_TYPES)}
IDX2DEFECT = {i: d for d, i in DEFECT2IDX.items()}

CAM2IDX = {c: i for i, c in enumerate(CAM_IDS)}   # cam1→0 … cam6→5
POSE2IDX = {"p0": 0, "p90": 1}

# 物理兼容性矩阵 DVCP  (defect × cam)  1=合法, 0=非法
# 行: DEFECT_TYPES 顺序  列: cam1~cam6
DVCP_MATRIX = torch.tensor([
    # cam1 cam2 cam3 cam4 cam5 cam6
    [  1,   0,   1,   0,   0,   0],  # adhesion (Cam2 deleted, only Cam1/Cam3)
    [  1,   1,   0,   0,   0,   0],  # breakage
    [  1,   1,   0,   0,   0,   0],  # contamination
    [  1,   0,   0,   0,   0,   0],  # diffusion
    [  1,   0,   0,   0,   0,   0],  # exposed_substrate
    [  0,   1,   0,   0,   0,   0],  # reversed_print
    [  0,   0,   1,   1,   1,   1],  # silver_overflow
], dtype=torch.float32)


# ─────────────────────────────────────────────
# 文件名解析工具
# ─────────────────────────────────────────────
_NG_PATTERN = re.compile(
    r"cam(\d+)_(p\d+)_([a-z_]+?)(?:_gen)?_(\d{5})\.png", re.IGNORECASE
)
_OK_PATTERN = re.compile(
    r"cam(\d+)_(p\d+)_ok_(\d{5})\.png", re.IGNORECASE
)


def parse_ng_filename(fname: str) -> Optional[Dict]:
    """解析 NG 文件名，返回 {cam, pose, defect, index} 或 None"""
    m = _NG_PATTERN.match(fname)
    if m is None:
        return None
    cam, pose, defect, idx = m.group(1), m.group(2), m.group(3), m.group(4)
    if defect == "ok":          # 防止误匹配 OK 文件
        return None
    if defect not in DEFECT2IDX:
        return None
    return {
        "cam":    int(cam),
        "pose":   pose,
        "defect": defect,
        "index":  int(idx),
    }


def parse_ok_filename(fname: str) -> Optional[Dict]:
    """解析 OK 文件名，返回 {cam, pose, index} 或 None"""
    m = _OK_PATTERN.match(fname)
    if m is None:
        return None
    return {
        "cam":   int(m.group(1)),
        "pose":  m.group(2),
        "index": int(m.group(3)),
    }


# ─────────────────────────────────────────────
# Dataset 主类
# ─────────────────────────────────────────────
class DefectDataset(Dataset):
    """
    统一加载 OK 图 和 NG 图。

    Args:
        root:        数据集根目录，下含 OK/ 和 NG/ 子目录
        mode:        'ng' | 'ok' | 'all'
        cam_ids:     要包含的摄像头列表，None 表示全部
        poses:       要包含的姿态列表，None 表示全部
        defects:     要包含的缺陷类型列表，None 表示全部（仅 mode='ng'/'all' 时有效）
        transform:   图像变换
        max_per_class: 每个 (cam, defect) 组合最多取多少张（少样本实验用）
    """

    def __init__(
        self,
        root: str,
        mode: str = "ng",
        cam_ids: Optional[List[int]] = None,
        poses: Optional[List[str]] = None,
        defects: Optional[List[str]] = None,
        transform=None,
        max_per_class: Optional[int] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self.cam_ids = cam_ids or CAM_IDS
        self.poses   = poses   or POSES
        self.defects = defects or DEFECT_TYPES
        self.transform = transform or self._default_transform()
        self.max_per_class = max_per_class

        self.samples: List[Dict] = []
        self._load_samples()

    # --------------------------------------------------
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    # --------------------------------------------------
    def _load_samples(self):
        if self.mode in ("ng", "all"):
            self._load_ng()
        if self.mode in ("ok", "all"):
            self._load_ok()

    def _load_ng(self):
        ng_root = self.root / "NG"
        if not ng_root.exists():
            raise FileNotFoundError(f"NG directory not found: {ng_root}")

        counts: Dict[Tuple, int] = {}   # (cam, defect) → count

        for defect in sorted(ng_root.iterdir()):
            if not defect.is_dir() or defect.name not in self.defects:
                continue
            for pose_dir in sorted(defect.iterdir()):
                if not pose_dir.is_dir() or pose_dir.name not in self.poses:
                    continue
                for img_path in sorted(pose_dir.glob("*.png")):
                    info = parse_ng_filename(img_path.name)
                    if info is None:
                        continue
                    if info["cam"] not in self.cam_ids:
                        continue

                    key = (info["cam"], info["defect"])
                    cnt = counts.get(key, 0)
                    if self.max_per_class and cnt >= self.max_per_class:
                        continue
                    counts[key] = cnt + 1

                    self.samples.append({
                        "path":        str(img_path),
                        "label":       1,               # 1 = NG
                        "cam":         info["cam"],
                        "cam_idx":     CAM2IDX[info["cam"]],
                        "pose":        info["pose"],
                        "pose_idx":    POSE2IDX[info["pose"]],
                        "defect":      info["defect"],
                        "defect_idx":  DEFECT2IDX[info["defect"]],
                    })

    def _load_ok(self):
        ok_root = self.root / "OK"
        if not ok_root.exists():
            raise FileNotFoundError(f"OK directory not found: {ok_root}")

        for pose_dir in sorted(ok_root.iterdir()):
            if not pose_dir.is_dir() or pose_dir.name not in self.poses:
                continue
            for cam_dir in sorted(pose_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue
                for img_path in sorted(cam_dir.glob("*.png")):
                    info = parse_ok_filename(img_path.name)
                    if info is None:
                        continue
                    if info["cam"] not in self.cam_ids:
                        continue

                    self.samples.append({
                        "path":       str(img_path),
                        "label":      0,                # 0 = OK
                        "cam":        info["cam"],
                        "cam_idx":    CAM2IDX[info["cam"]],
                        "pose":       info["pose"],
                        "pose_idx":   POSE2IDX[info["pose"]],
                        "defect":     "ok",
                        "defect_idx": -1,               # -1 = 无缺陷
                    })

    # --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return {
            "image":      img,
            "label":      torch.tensor(s["label"],      dtype=torch.long),
            "cam_idx":    torch.tensor(s["cam_idx"],    dtype=torch.long),
            "pose_idx":   torch.tensor(s["pose_idx"],   dtype=torch.long),
            "defect_idx": torch.tensor(s["defect_idx"], dtype=torch.long),
            "path":       s["path"],
        }

    # --------------------------------------------------
    def get_stats(self) -> Dict:
        """打印数据集统计信息"""
        from collections import Counter
        counter_ok  = Counter()
        counter_ng  = Counter()
        for s in self.samples:
            key = f"cam{s['cam']}_{s['pose']}"
            if s["label"] == 0:
                counter_ok[key] += 1
            else:
                counter_ng[f"{key}_{s['defect']}"] += 1
        return {"ok": dict(counter_ok), "ng": dict(counter_ng)}


# ─────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    DATA_ROOT = r"E:\code\dataset\Generative_Modeling\data\datasets"

    print("=" * 60)
    print("测试 NG Dataset")
    ng_ds = DefectDataset(DATA_ROOT, mode="ng")
    print(f"  总样本数: {len(ng_ds)}")
    sample = ng_ds[0]
    print(f"  样本示例: cam={sample['cam_idx'].item()+1}, "
          f"pose_idx={sample['pose_idx'].item()}, "
          f"defect_idx={sample['defect_idx'].item()}, "
          f"image_shape={sample['image'].shape}")

    print("\n测试 OK Dataset")
    ok_ds = DefectDataset(DATA_ROOT, mode="ok")
    print(f"  总样本数: {len(ok_ds)}")

    print("\n测试 ALL Dataset")
    all_ds = DefectDataset(DATA_ROOT, mode="all")
    print(f"  总样本数: {len(all_ds)}")

    print("\n统计信息 (NG)")
    stats = ng_ds.get_stats()
    for k, v in sorted(stats["ng"].items()):
        print(f"  {k}: {v}")

    print("\n测试少样本模式 (max_per_class=20)")
    few_ds = DefectDataset(DATA_ROOT, mode="ng", max_per_class=20)
    print(f"  总样本数: {len(few_ds)}")

    print("\n测试 DataLoader")
    loader = DataLoader(ng_ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"  batch image shape: {batch['image'].shape}")
    print(f"  batch cam_idx:     {batch['cam_idx']}")
    print(f"  batch defect_idx:  {batch['defect_idx']}")

    print("\n[PASS] DataLoader test passed")
