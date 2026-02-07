"""
光照条件分析脚本
分析6个视角的光照差异，为模型设计提供依据
"""
import sys
sys.path.insert(0, 'E:/code/Generative_Modeling')

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

from src.datasets.multiview_dataset import MultiViewDataset
from src.utils.image_utils import detect_background_type


def analyze_lighting_conditions(root_dir: str):
    """分析所有视角的光照条件"""
    
    print("=" * 60)
    print("光照条件分析")
    print("=" * 60)
    
    # 创建数据集（不使用变换，保持原始像素值）
    from torchvision import transforms
    simple_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()  # 只转为tensor，不归一化
    ])
    
    dataset = MultiViewDataset(
        root_dir=root_dir,
        group='all',
        label='OK',
        transform=simple_transform
    )
    
    print(f"总样本数: {len(dataset)}")
    
    # 按视角统计
    view_stats = defaultdict(lambda: {
        'brightness': [],
        'contrast': [],
        'bg_type': {'bright_field': 0, 'dark_field': 0}
    })
    
    # 采样分析（避免太慢）
    sample_size = min(100, len(dataset) // 6)
    print(f"每视角采样 {sample_size} 张进行分析...")
    
    view_indices = defaultdict(list)
    for idx in range(len(dataset)):
        view = dataset.samples[idx]['view']
        if len(view_indices[view]) < sample_size:
            view_indices[view].append(idx)
    
    for view, indices in view_indices.items():
        print(f"\n分析 {view}...")
        for idx in indices:
            sample = dataset[idx]
            image = sample['image']  # (C, H, W), values in [0, 1]
            
            # 计算亮度（平均值）
            brightness = image.mean().item() * 255
            
            # 计算对比度（标准差）
            contrast = image.std().item() * 255
            
            # 检测背景类型
            bg_type = detect_background_type(image)
            
            view_stats[view]['brightness'].append(brightness)
            view_stats[view]['contrast'].append(contrast)
            view_stats[view]['bg_type'][bg_type] += 1
    
    # 输出结果
    print("\n" + "=" * 60)
    print("分析结果")
    print("=" * 60)
    
    print("\n| 视角 | 平均亮度 | 对比度 | 明场% | 暗场% | 光照类型 |")
    print("|------|---------|--------|-------|-------|---------|")
    
    for view in ['Cam1', 'Cam2', 'Cam3', 'Cam4', 'Cam5', 'Cam6']:
        stats = view_stats[view]
        avg_brightness = np.mean(stats['brightness'])
        avg_contrast = np.mean(stats['contrast'])
        
        total = stats['bg_type']['bright_field'] + stats['bg_type']['dark_field']
        bright_pct = stats['bg_type']['bright_field'] / total * 100
        dark_pct = stats['bg_type']['dark_field'] / total * 100
        
        if dark_pct > 80:
            light_type = "暗场"
        elif bright_pct > 80:
            light_type = "明场"
        else:
            light_type = "混合"
        
        print(f"| {view} | {avg_brightness:.1f} | {avg_contrast:.1f} | {bright_pct:.0f}% | {dark_pct:.0f}% | {light_type} |")
    
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print("- Cam1, Cam2, Cam3, Cam5: 明场照明（青/绿色背景）")
    print("- Cam4, Cam6: 暗场照明（黑色背景）")
    print("- 建议：为明场和暗场分别设计处理策略")


if __name__ == '__main__':
    root = r'E:\code\Generative_Modeling\data\datasets'
    analyze_lighting_conditions(root)
