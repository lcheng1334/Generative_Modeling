"""
evaluate_gen.py - 生成图像质量评估

指标:
  1. FID  (Frechet Inception Distance) ↓
  2. LPIPS (Learned Perceptual Image Patch Similarity) ↓
  3. PCS  (Physical Compatibility Score) ↑ — 本文首次提出

用法:
  python scripts/evaluate_gen.py \\
      --real_dir E:/code/dataset/.../NG \\
      --gen_dir outputs/samples \\
      --config configs/idgs.yaml
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.datasets.defect_dataset import (
    DEFECT_TYPES, CAM_IDS, DVCP_MATRIX,
    parse_ng_filename, DEFECT2IDX, CAM2IDX,
)


# ─────────────────────────────────────────────────────────
# 1. FID (需要 pytorch-fid 或 clean-fid)
# ─────────────────────────────────────────────────────────
def compute_fid(real_dir: str, gen_dir: str) -> float:
    """
    计算 FID 分数

    依赖: pip install pytorch-fid
    """
    try:
        from pytorch_fid import fid_score
        score = fid_score.calculate_fid_given_paths(
            [real_dir, gen_dir],
            batch_size=50,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dims=2048,
        )
        return score
    except ImportError:
        print("[WARNING] pytorch-fid not installed. Skipping FID.")
        return -1.0


# ─────────────────────────────────────────────────────────
# 2. LPIPS
# ─────────────────────────────────────────────────────────
def compute_lpips(real_dir: str, gen_dir: str, num_pairs=200) -> float:
    """
    计算 LPIPS 感知相似性（随机采样 real-gen 配对）

    依赖: pip install lpips
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex")
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        real_imgs = sorted(Path(real_dir).rglob("*.png"))
        gen_imgs  = sorted(Path(gen_dir).rglob("*.png"))

        if not real_imgs or not gen_imgs:
            print("[WARNING] No images found for LPIPS")
            return -1.0

        scores = []
        rng = np.random.RandomState(42)
        for _ in tqdm(range(min(num_pairs, len(gen_imgs))), desc="LPIPS"):
            r_idx = rng.randint(0, len(real_imgs))
            g_idx = rng.randint(0, len(gen_imgs))

            r_img = transform(Image.open(real_imgs[r_idx]).convert("RGB")).unsqueeze(0)
            g_img = transform(Image.open(gen_imgs[g_idx]).convert("RGB")).unsqueeze(0)

            if torch.cuda.is_available():
                r_img, g_img = r_img.cuda(), g_img.cuda()

            with torch.no_grad():
                d = loss_fn(r_img, g_img).item()
            scores.append(d)

        return float(np.mean(scores))

    except ImportError:
        print("[WARNING] lpips not installed. Skipping LPIPS.")
        return -1.0


# ─────────────────────────────────────────────────────────
# 3. PCS (Physical Compatibility Score) — 本文首次提出
# ─────────────────────────────────────────────────────────
def compute_pcs(gen_dir: str) -> dict:
    """
    Physical Compatibility Score (PCS)

    定义: 生成图中符合 DVCP 物理约束的比例

    计算方法:
      1. 从生成图文件名解析 (cam, defect) 组合
      2. 检查该组合在 DVCP 矩阵中是否合法
      3. PCS = 合法数 / 总数

    Returns:
        {
          "pcs": float,         # 总体 PCS
          "total": int,         # 总生成数
          "legal": int,         # 合法数
          "illegal": int,       # 非法数
          "illegal_examples": list,  # 非法组合示例
        }
    """
    gen_path = Path(gen_dir)
    total = 0
    legal = 0
    illegal_examples = []

    for img_file in gen_path.rglob("*.png"):
        info = parse_ng_filename(img_file.name)
        if info is None:
            continue

        total += 1
        d_idx = DEFECT2IDX.get(info["defect"], -1)
        c_idx = CAM2IDX.get(info["cam"], -1)

        if d_idx >= 0 and c_idx >= 0:
            if DVCP_MATRIX[d_idx, c_idx] > 0:
                legal += 1
            else:
                if len(illegal_examples) < 5:
                    illegal_examples.append(
                        f"cam{info['cam']}_{info['defect']} ({img_file.name})"
                    )

    pcs = legal / total if total > 0 else 0.0
    return {
        "pcs": pcs,
        "total": total,
        "legal": legal,
        "illegal": total - legal,
        "illegal_examples": illegal_examples,
    }


# ─────────────────────────────────────────────────────────
# 4. 各缺陷类别的 per-class FID (可选)
# ─────────────────────────────────────────────────────────
def compute_per_class_stats(gen_dir: str) -> dict:
    """统计每类缺陷的生成数量"""
    gen_path = Path(gen_dir)
    counts = defaultdict(int)
    for img_file in gen_path.rglob("*.png"):
        info = parse_ng_filename(img_file.name)
        if info:
            key = f"cam{info['cam']}_{info['defect']}"
            counts[key] += 1
    return dict(counts)


# ─────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────
def main(args):
    print("=" * 60)
    print("  IDGS Generation Quality Evaluation")
    print("=" * 60)

    # PCS (不需要 real 图)
    print("\n[1/3] Computing PCS (Physical Compatibility Score)...")
    pcs_result = compute_pcs(args.gen_dir)
    print(f"  PCS       : {pcs_result['pcs']:.4f}")
    print(f"  Total     : {pcs_result['total']}")
    print(f"  Legal     : {pcs_result['legal']}")
    print(f"  Illegal   : {pcs_result['illegal']}")
    if pcs_result['illegal_examples']:
        print(f"  Examples  : {pcs_result['illegal_examples']}")

    # Per-class stats
    stats = compute_per_class_stats(args.gen_dir)
    print("\n  Per-class generation counts:")
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}")

    # FID
    if args.real_dir:
        print("\n[2/3] Computing FID...")
        fid = compute_fid(args.real_dir, args.gen_dir)
        if fid >= 0:
            print(f"  FID       : {fid:.2f}")

        # LPIPS
        print("\n[3/3] Computing LPIPS...")
        lpips_score = compute_lpips(args.real_dir, args.gen_dir)
        if lpips_score >= 0:
            print(f"  LPIPS     : {lpips_score:.4f}")
    else:
        print("\n[2/3] Skipping FID (no --real_dir)")
        print("[3/3] Skipping LPIPS (no --real_dir)")

    print("\n[DONE] Evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir",  type=str, required=True)
    parser.add_argument("--real_dir", type=str, default=None)
    parser.add_argument("--config",   type=str, default="configs/idgs.yaml")
    args = parser.parse_args()
    main(args)
