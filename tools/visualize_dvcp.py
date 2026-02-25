"""
visualize_dvcp.py - DVCP 热力图可视化

用途: 论文 Figure 用
  1. Ground truth DVCP 矩阵 (制造工艺约束)
  2. 学到的 DVCP 矩阵 (训练后)
  3. 两者对比图

用法:
  python tools/visualize_dvcp.py --ckpt checkpoints/idgs/phase2_final
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 无 GUI 模式

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.datasets.defect_dataset import DEFECT_TYPES, CAM_IDS
from src.models.dvcp_module import DVCP_GT


DEFECT_LABELS = [
    "Adhesion", "Breakage", "Contamination",
    "Diffusion", "Exposed\nSubstrate", "Reversed\nPrint", "Silver\nOverflow"
]
CAM_LABELS = [f"Cam{c}" for c in CAM_IDS]


def plot_heatmap(matrix, title, save_path, cmap="YlOrRd",
                 vmin=0, vmax=1, fmt=".2f", annot=True):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(CAM_LABELS)))
    ax.set_xticklabels(CAM_LABELS, fontsize=12)
    ax.set_yticks(range(len(DEFECT_LABELS)))
    ax.set_yticklabels(DEFECT_LABELS, fontsize=11)

    if annot:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Camera Station", fontsize=12)
    ax.set_ylabel("Defect Type", fontsize=12)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Compatibility Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main(args):
    print("=" * 50)
    print("  DVCP Matrix Visualization")
    print("=" * 50)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ground Truth 矩阵
    gt_matrix = DVCP_GT.numpy()
    plot_heatmap(
        gt_matrix,
        "DVCP Ground Truth (Manufacturing Constraints)",
        out_dir / "dvcp_gt.png",
        cmap="RdYlGn",
        fmt=".0f",
    )

    # 2. 学到的矩阵 (如果有 checkpoint)
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        npy_path = ckpt_path / "dvcp_matrix_final.npy"
        if npy_path.exists():
            learned_matrix = np.load(str(npy_path))
            plot_heatmap(
                learned_matrix,
                "Learned DVCP Matrix (After Training)",
                out_dir / "dvcp_learned.png",
                cmap="YlOrRd",
            )

            # 3. 差异图
            diff = np.abs(learned_matrix - gt_matrix)
            plot_heatmap(
                diff,
                "DVCP Prediction Error |Learned - GT|",
                out_dir / "dvcp_diff.png",
                cmap="Reds",
                fmt=".3f",
            )
        else:
            print(f"  [SKIP] No learned matrix found at {npy_path}")
    else:
        print("  [SKIP] No checkpoint specified, only GT plotted")

    print("\n[DONE] Visualization complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/figures")
    args = parser.parse_args()
    main(args)
