"""
plot_training_log.py - SCI 论文级训练曲线可视化

特点:
  - Times New Roman 字体（SCI 标准）
  - 双 Y 轴（loss + lr）
  - 高 DPI 矢量输出
  - 标注关键点（最低 loss）
  - 收敛区域高亮

用法:
  python tools/plot_training_log.py --log logs/phase1.log
  python tools/plot_training_log.py --log logs/phase2.log
"""
import re
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── SCI 论文级全局样式 ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# 配色方案（SCI 友好，色盲安全）
COLORS = {
    "blue":   "#1f77b4",
    "red":    "#d62728",
    "orange": "#ff7f0e",
    "green":  "#2ca02c",
    "purple": "#9467bd",
    "gray":   "#7f7f7f",
}


def parse_phase1_log(log_path):
    epochs, losses, lrs = [], [], []
    pattern = re.compile(
        r"Epoch (\d+)/(\d+) \| loss=([\d.]+) \| lr=([\d.e+-]+)"
    )
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(3)))
                lrs.append(float(m.group(4)))
    return np.array(epochs), np.array(losses), np.array(lrs)


def parse_phase2_log(log_path):
    epochs, denoise, dvcp, total = [], [], [], []
    pattern = re.compile(
        r"Epoch (\d+)/(\d+) \| denoise=([\d.]+) \| dvcp=([\d.]+) \| total=([\d.]+)"
    )
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                denoise.append(float(m.group(3)))
                dvcp.append(float(m.group(4)))
                total.append(float(m.group(5)))
    return np.array(epochs), np.array(denoise), np.array(dvcp), np.array(total)


def moving_average(data, window=5):
    """计算移动平均"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_phase1(epochs, losses, lrs, save_path):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # ── 左 Y 轴: Loss ──
    ax1.plot(epochs, losses, color=COLORS["blue"], linewidth=0.8, alpha=0.35,
             label="_nolegend_")  # 原始数据（浅色）

    # 移动平均（主线）
    window = 5
    ma = moving_average(losses, window)
    ma_epochs = epochs[window-1:]
    ax1.plot(ma_epochs, ma, color=COLORS["blue"], linewidth=2.0,
             label="MSE Loss (moving avg.)")

    # 标注最低 loss
    min_idx = np.argmin(losses)
    ax1.annotate(
        f"Min: {losses[min_idx]:.4f}\n(Epoch {epochs[min_idx]})",
        xy=(epochs[min_idx], losses[min_idx]),
        xytext=(epochs[min_idx] + 8, losses[min_idx] + 0.005),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2),
        color=COLORS["red"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["red"], alpha=0.9),
    )

    # 收敛区域高亮（后 30% epochs）
    converge_start = int(len(epochs) * 0.7)
    ax1.axvspan(epochs[converge_start], epochs[-1],
                alpha=0.08, color=COLORS["green"],
                label="Convergence region")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss", color=COLORS["blue"])
    ax1.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax1.set_xlim(0, epochs[-1] + 1)

    # ── 右 Y 轴: Learning Rate ──
    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, color=COLORS["orange"], linewidth=1.5,
             linestyle="--", alpha=0.7, label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))

    # ── 合并图例 ──
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", framealpha=0.9,
               edgecolor="gray")

    ax1.set_title("Phase 1: SD v1.5 + LoRA Fine-tuning", fontweight="bold")
    ax1.grid(True, alpha=0.2, linestyle="--")

    plt.savefig(save_path)
    plt.savefig(save_path.replace(".png", ".pdf"))  # 矢量 PDF
    plt.close()
    print(f"  Saved: {save_path}")
    print(f"  Saved: {save_path.replace('.png', '.pdf')}")


def plot_phase2(epochs, denoise, dvcp, total, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── 左图: 三条 loss 曲线 ──
    window = 3
    for data, color, label in [
        (total,   COLORS["red"],    "Total Loss"),
        (denoise, COLORS["blue"],   "Denoise Loss"),
        (dvcp,    COLORS["orange"], "DVCP Loss"),
    ]:
        ax1.plot(epochs, data, color=color, linewidth=0.6, alpha=0.3)
        if len(data) >= window:
            ma = moving_average(data, window)
            ax1.plot(epochs[window-1:], ma, color=color, linewidth=2.0, label=label)
        else:
            ax1.plot(epochs, data, color=color, linewidth=2.0, label=label)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Phase 2: Joint Training Losses", fontweight="bold")
    ax1.legend(framealpha=0.9, edgecolor="gray")
    ax1.grid(True, alpha=0.2, linestyle="--")

    # ── 右图: DVCP loss 单独放大 ──
    ax2.plot(epochs, dvcp, color=COLORS["orange"], linewidth=0.8, alpha=0.4)
    if len(dvcp) >= window:
        ma = moving_average(dvcp, window)
        ax2.plot(epochs[window-1:], ma, color=COLORS["orange"],
                 linewidth=2.0, label="DVCP Loss")

    min_idx = np.argmin(dvcp)
    ax2.annotate(
        f"Min: {dvcp[min_idx]:.4f}",
        xy=(epochs[min_idx], dvcp[min_idx]),
        xytext=(epochs[min_idx] + 3, dvcp[min_idx] * 1.15),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2),
        color=COLORS["red"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLORS["red"], alpha=0.9),
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("DVCP Loss")
    ax2.set_title("DVCP Compatibility Learning", fontweight="bold")
    ax2.legend(framealpha=0.9, edgecolor="gray")
    ax2.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {save_path}")
    print(f"  Saved: {save_path.replace('.png', '.pdf')}")


def print_summary(epochs, losses):
    """打印训练摘要"""
    print(f"\n  Training Summary")
    print(f"  {'─'*40}")
    print(f"  Total epochs  : {len(epochs)}")
    print(f"  Initial loss  : {losses[0]:.4f}")
    print(f"  Final loss    : {losses[-1]:.4f}")
    print(f"  Min loss      : {losses.min():.4f} (Epoch {epochs[losses.argmin()]})")
    print(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    print(f"  Convergence   : {'Yes' if losses[-1] < losses[0] * 0.8 else 'Marginal'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--phase", type=int, default=None)
    args = parser.parse_args()

    log_path = Path(args.log)

    phase = args.phase
    if phase is None:
        phase = 2 if "phase2" in log_path.name else 1

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output or str(out_dir / f"phase{phase}_loss.png")

    print("=" * 50)
    print(f"  Phase {phase} Training Visualization")
    print("=" * 50)

    if phase == 1:
        epochs, losses, lrs = parse_phase1_log(str(log_path))
        print(f"  Parsed {len(epochs)} epochs")
        if epochs.size > 0:
            print_summary(epochs, losses)
            plot_phase1(epochs, losses, lrs, save_path)
        else:
            print("  ERROR: No epoch data found!")

    elif phase == 2:
        epochs, denoise, dvcp, total = parse_phase2_log(str(log_path))
        print(f"  Parsed {len(epochs)} epochs")
        if epochs.size > 0:
            print_summary(epochs, total)
            plot_phase2(epochs, denoise, dvcp, total, save_path)
        else:
            print("  ERROR: No epoch data found!")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
