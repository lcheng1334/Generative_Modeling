"""
plot_training_log.py - 从训练日志中提取 loss 并绘图

用法:
  python tools/plot_training_log.py --log logs/phase1.log --output outputs/figures/phase1_loss.png
  python tools/plot_training_log.py --log logs/phase2.log --output outputs/figures/phase2_loss.png
"""
import re
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_phase1_log(log_path):
    """解析 Phase 1 日志，提取 epoch, loss, lr"""
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
    return epochs, losses, lrs


def parse_phase2_log(log_path):
    """解析 Phase 2 日志，提取 epoch, denoise_loss, dvcp_loss, total_loss"""
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
    return epochs, denoise, dvcp, total


def plot_phase1(epochs, losses, lrs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, losses, color="#2196F3", linewidth=1.5, alpha=0.7, label="MSE Loss")
    # 滑动平均
    if len(losses) >= 5:
        window = min(5, len(losses))
        avg = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax1.plot(epochs[window-1:], avg, color="#F44336", linewidth=2, label=f"Moving Avg (w={window})")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Phase 1: SD LoRA Training Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # LR curve
    ax2.plot(epochs, lrs, color="#4CAF50", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_title("Learning Rate Schedule (Cosine)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_phase2(epochs, denoise, dvcp, total, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, denoise, color="#2196F3", linewidth=1.5, label="Denoise Loss", alpha=0.8)
    ax.plot(epochs, dvcp, color="#FF9800", linewidth=1.5, label="DVCP Loss", alpha=0.8)
    ax.plot(epochs, total, color="#F44336", linewidth=2, label="Total Loss")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Phase 2: DVCP + SAN Joint Training Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--phase", type=int, default=None,
                        help="1 or 2 (auto-detected from filename if not set)")
    args = parser.parse_args()

    log_path = Path(args.log)

    # 自动检测 phase
    phase = args.phase
    if phase is None:
        if "phase2" in log_path.name:
            phase = 2
        else:
            phase = 1

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output or str(out_dir / f"phase{phase}_loss.png")

    if phase == 1:
        epochs, losses, lrs = parse_phase1_log(str(log_path))
        print(f"Parsed {len(epochs)} epochs from {log_path}")
        if epochs:
            print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
            plot_phase1(epochs, losses, lrs, save_path)
        else:
            print("No epoch data found!")
    elif phase == 2:
        epochs, denoise, dvcp, total = parse_phase2_log(str(log_path))
        print(f"Parsed {len(epochs)} epochs from {log_path}")
        if epochs:
            plot_phase2(epochs, denoise, dvcp, total, save_path)
        else:
            print("No epoch data found!")


if __name__ == "__main__":
    main()
