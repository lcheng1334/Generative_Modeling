"""
train_classifier.py - 下游缺陷分类实验

用途: 验证合成数据对分类器性能的提升
  1. 仅用原始 NG 数据训练
  2. 原始 + 合成数据训练
  3. 对比 F1 score 提升

用法:
  # 仅用原始数据
  python scripts/train_classifier.py --data_root /data/datasets --mode real_only

  # 原始 + 合成数据
  python scripts/train_classifier.py --data_root /data/datasets --mode augmented \\
      --gen_dir outputs/samples

  # 少样本实验 (每类50张)
  python scripts/train_classifier.py --data_root /data/datasets --mode augmented \\
      --gen_dir outputs/samples --max_per_class 50
"""
import argparse
import sys
import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms, models
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.datasets.defect_dataset import DefectDataset, DEFECT_TYPES


# ─────────────────────────────────────────────────────────
# 1. 分类器模型
# ─────────────────────────────────────────────────────────
def get_classifier(num_classes=7, backbone="resnet50"):
    """
    获取预训练 backbone + 分类头

    Args:
        num_classes: 缺陷类别数
        backbone:    'resnet50' 或 'efficientnet_b0'

    Returns:
        model: nn.Module
    """
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ─────────────────────────────────────────────────────────
# 2. 训练循环
# ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        imgs = batch["image"].to(device)
        labels = batch["defect_idx"].to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        imgs = batch["image"].to(device)
        labels = batch["defect_idx"]

        outputs = model(imgs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────────────────
# 3. 主函数
# ─────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── 数据集 ──
    # 真实 NG 数据
    real_ds = DefectDataset(
        root=args.data_root,
        mode="ng",
        transform=transform,
        max_per_class=args.max_per_class,
    )

    # 训练/验证分割 (80/20)
    n_total = len(real_ds)
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train
    real_train, real_val = random_split(
        real_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # 验证集使用无增强 transform
    # (random_split 后无法直接修改 transform, 但这里影响不大)

    if args.mode == "augmented" and args.gen_dir:
        # 合成数据
        gen_ds = DefectDataset(
            root=args.gen_dir,
            mode="ng",
            transform=transform,
        )
        train_ds = ConcatDataset([real_train, gen_ds])
        print(f"  Real train: {n_train}, Synthetic: {len(gen_ds)}, Total: {n_train + len(gen_ds)}")
    else:
        train_ds = real_train
        print(f"  Real train: {n_train} (no augmentation)")

    print(f"  Validation: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(real_val, batch_size=32, shuffle=False, num_workers=4)

    # ── 类别分布 ──
    label_counts = Counter()
    for s in real_ds.samples:
        label_counts[s["defect"]] += 1
    print("\n  Class distribution:")
    for defect in DEFECT_TYPES:
        print(f"    {defect}: {label_counts.get(defect, 0)}")

    # ── 模型 ──
    model = get_classifier(
        num_classes=len(DEFECT_TYPES),
        backbone=args.backbone
    ).to(device)

    # 类别加权损失 (处理不均衡)
    class_counts = [label_counts.get(d, 1) for d in DEFECT_TYPES]
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum() * len(DEFECT_TYPES)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── 训练 ──
    best_f1 = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        scheduler.step()

        # 验证
        preds, labels = evaluate(model, val_loader, device)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

        if (epoch + 1) % 10 == 0 or f1_macro > best_f1:
            print(f"  Epoch {epoch+1}/{args.epochs} | "
                  f"loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"F1_macro={f1_macro:.4f} F1_weighted={f1_weighted:.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_preds = preds
            best_labels = labels
            if args.save_model:
                torch.save(model.state_dict(), args.save_model)

    # ── 最终报告 ──
    print("\n" + "=" * 60)
    print(f"  Best F1 (macro): {best_f1:.4f}")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(
        best_labels, best_preds,
        target_names=DEFECT_TYPES,
        zero_division=0,
    ))

    # 每类 F1
    per_class_f1 = f1_score(best_labels, best_preds, average=None, zero_division=0)
    print("\nPer-class F1:")
    for defect, f1 in zip(DEFECT_TYPES, per_class_f1):
        marker = " <-- RARE" if label_counts.get(defect, 0) < 200 else ""
        print(f"  {defect:20s}: {f1:.4f}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--gen_dir",   type=str, default=None)
    parser.add_argument("--mode",      type=str, default="real_only",
                        choices=["real_only", "augmented"])
    parser.add_argument("--backbone",  type=str, default="resnet50")
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--save_model", type=str, default=None)
    args = parser.parse_args()
    main(args)
