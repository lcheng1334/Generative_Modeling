"""
Phase 1 成果可视化
展示数据加载、6视角图像、边缘提取等功能
"""
import sys
sys.path.insert(0, 'E:/code/Generative_Modeling')

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from src.datasets.multiview_dataset import MultiViewDataset
from src.utils.image_utils import extract_edge_regions, detect_background_type, visualize_multi_view

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def denormalize(tensor):
    """反归一化恢复图像"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize_phase1_results():
    """可视化Phase 1的成果"""
    
    root = r'E:\code\Generative_Modeling\data\datasets'
    output_dir = Path(r'E:\code\Generative_Modeling\outputs\phase1')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    dataset = MultiViewDataset(root, group='Group1', label='OK')
    
    print("=" * 60)
    print("Phase 1 成果可视化")
    print("=" * 60)
    print(f"\n数据集样本数: {len(dataset)}")
    
    # 1. 展示6视角图像
    print("\n1. 收集6视角图像...")
    view_samples = {}
    for view in ['Cam1', 'Cam2', 'Cam3', 'Cam4', 'Cam5', 'Cam6']:
        for idx, sample in enumerate(dataset.samples):
            if sample['view'] == view:
                view_samples[view] = dataset[idx]
                break
    
    # 绘制6视角
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('6-View Industrial Inspection Images (Group1)', fontsize=16, fontweight='bold')
    
    views = ['Cam1', 'Cam2', 'Cam3', 'Cam4', 'Cam5', 'Cam6']
    view_names = {
        'Cam1': 'Cam1 - Front View',
        'Cam2': 'Cam2 - Bottom View',
        'Cam3': 'Cam3 - Left Side',
        'Cam4': 'Cam4 - Upper Side (Dark Field)',
        'Cam5': 'Cam5 - Right Side',
        'Cam6': 'Cam6 - Lower Side (Dark Field)'
    }
    
    for idx, (ax, view) in enumerate(zip(axes.flat, views)):
        sample = view_samples[view]
        img = denormalize(sample['image'])
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        # 检测光照类型
        bg_type = detect_background_type(sample['image'])
        
        ax.imshow(img_np)
        lighting_type = 'Dark Field' if bg_type == 'dark_field' else 'Bright Field'
        ax.set_title(f"{view_names[view]}\nLighting: {lighting_type}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    save_path1 = output_dir / 'phase1_multiview_sample.png'
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved 6-view image: {save_path1}")
    plt.close()
    
    # 2. 展示边缘提取
    print("\n2. 展示边缘提取功能...")
    cam1_sample = view_samples['Cam1']
    cam1_img = cam1_sample['image']
    
    # 提取边缘
    edges = extract_edge_regions(cam1_img, edge_width=32)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Cam1 Front Face Edge Extraction - For Cross-View Consistency', fontsize=16, fontweight='bold')
    
    # 显示原图
    ax = axes[0, 0]
    img_show = denormalize(cam1_img).permute(1, 2, 0).numpy()
    ax.imshow(np.clip(img_show, 0, 1))
    ax.set_title('Original Image (Cam1)', fontsize=12)
    ax.axis('off')
    
    # 标注边缘区域
    ax = axes[0, 1]
    ax.imshow(np.clip(img_show, 0, 1))
    H, W = img_show.shape[:2]
    edge_w = 32
    # 画边缘矩形
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), edge_w, H, fill=False, edgecolor='red', linewidth=2))  # 左
    ax.add_patch(Rectangle((W-edge_w, 0), edge_w, H, fill=False, edgecolor='green', linewidth=2))  # 右
    ax.add_patch(Rectangle((0, 0), W, edge_w, fill=False, edgecolor='blue', linewidth=2))  # 上
    ax.add_patch(Rectangle((0, H-edge_w), W, edge_w, fill=False, edgecolor='yellow', linewidth=2))  # 下
    ax.set_title('Edge Region Annotation', fontsize=12)
    ax.axis('off')
    
    # 显示提取的边缘
    edge_info = [
        ('left', 'Left Edge -> Cam3', 'red'),
        ('right', 'Right Edge -> Cam5', 'green'),
        ('top', 'Top Edge -> Cam4', 'blue'),
        ('bottom', 'Bottom Edge -> Cam6', 'yellow')
    ]
    
    for idx, (edge_name, title, color) in enumerate(edge_info):
        ax = axes.flat[idx + 2]
        edge_img = denormalize(edges[edge_name]).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(edge_img, 0, 1))
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    save_path2 = output_dir / 'phase1_edge_extraction.png'
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved edge extraction: {save_path2}")
    plt.close()
    
    # 3. 光照对比
    print("\n3. Generate lighting comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Bright Field vs Dark Field Comparison', fontsize=16, fontweight='bold')
    
    # 明场例子 (Cam1)
    bright = denormalize(view_samples['Cam1']['image']).permute(1, 2, 0).numpy()
    axes[0].imshow(np.clip(bright, 0, 1))
    axes[0].set_title('Cam1 - Bright Field (Avg: 166.5)', fontsize=12)
    axes[0].axis('off')
    
    # 暗场例子 (Cam4)
    dark = denormalize(view_samples['Cam4']['image']).permute(1, 2, 0).numpy()
    axes[1].imshow(np.clip(dark, 0, 1))
    axes[1].set_title('Cam4 - Dark Field (Avg: 8.5)', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path3 = output_dir / 'phase1_lighting_comparison.png'
    plt.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved lighting comparison: {save_path3}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Visualization Complete! 3 images saved to:")
    print(f"  {output_dir}")
    print(f"  1. {save_path1.name}")
    print(f"  2. {save_path2.name}")
    print(f"  3. {save_path3.name}")
    print("=" * 60)

if __name__ == '__main__':
    visualize_phase1_results()
