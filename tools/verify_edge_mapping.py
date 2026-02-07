"""
验证边缘提取的正确性
通过实际加载数据集中的图像来检查边缘对应关系
"""
import sys
sys.path.insert(0, 'E:/code/Generative_Modeling')

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.datasets.multiview_dataset import MultiViewDataset
from src.utils.image_utils import extract_edge_regions

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

def denormalize(tensor):
    """反归一化"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def verify_edge_mapping():
    """验证边缘映射关系"""
    
    root = r'E:\code\Generative_Modeling\data\datasets'
    output_dir = Path(r'E:\code\Generative_Modeling\outputs\phase1')
    
    # 创建数据集
    dataset = MultiViewDataset(root, group='Group1', label='OK')
    
    print("=" * 60)
    print("验证边缘映射关系")
    print("=" * 60)
    
    # 找到同一个产品的6个视角图像
    # 假设文件名中有相同的ID
    print("\n查找同一产品的6视角图像...")
    
    # 先收集每个视角的样本
    samples_by_view = {f'Cam{i}': [] for i in range(1, 7)}
    for sample in dataset.samples:
        view = sample['view']
        samples_by_view[view].append(sample)
    
    # 打印每个视角的样本数
    print("\n各视角样本数:")
    for view, samples in samples_by_view.items():
        print(f"  {view}: {len(samples)}")
    
    # 获取每个视角的第一个样本（虽然可能不是同一产品，但可以用来验证视角特征）
    view_images = {}
    for view in ['Cam1', 'Cam2', 'Cam3', 'Cam4', 'Cam5', 'Cam6']:
        if samples_by_view[view]:
            idx = dataset.samples.index(samples_by_view[view][0])
            sample = dataset[idx]
            view_images[view] = sample['image']
    
    # 绘制对比图
    fig = plt.figure(figsize=(18, 12))
    
    # 第一行：显示所有6个视角
    print("\n第一行：显示所有6个视角")
    for i, view in enumerate(['Cam1', 'Cam2', 'Cam3', 'Cam4', 'Cam5', 'Cam6']):
        ax = plt.subplot(3, 6, i+1)
        img = denormalize(view_images[view]).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(view, fontsize=10)
        ax.axis('off')
    
    # 第二行：Cam1的4个边缘
    print("第二行：Cam1的4个边缘")
    cam1_img = view_images['Cam1']
    edges = extract_edge_regions(cam1_img, edge_width=32)
    
    edge_list = [
        ('left', 'Left Edge (Cam1)', 7),
        ('right', 'Right Edge (Cam1)', 8),
        ('top', 'Top Edge (Cam1)', 9),
        ('bottom', 'Bottom Edge (Cam1)', 10)
    ]
    
    for edge_name, title, pos in edge_list:
        ax = plt.subplot(3, 6, pos)
        edge_img = denormalize(edges[edge_name]).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(edge_img, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # 第三行：对应的侧面视角（用于对比）
    print("第三行：对应的侧面视角")
    side_views = [
        ('Cam3', 'Cam3 (Left Side)', 13),
        ('Cam5', 'Cam5 (Right Side)', 14),
        ('Cam4', 'Cam4 (Upper Side)', 15),
        ('Cam6', 'Cam6 (Lower Side)', 16)
    ]
    
    for view, title, pos in side_views:
        ax = plt.subplot(3, 6, pos)
        img = denormalize(view_images[view]).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    plt.suptitle('Edge Mapping Verification: Cam1 Edges vs Side Views', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'edge_mapping_verification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved verification image: {save_path}")
    
    print("\n" + "=" * 60)
    print("请检查生成的图片，对比：")
    print("  - 第二行：Cam1的边缘提取")
    print("  - 第三行：对应的侧面视角")
    print("看看边缘区域是否与侧面视角匹配！")
    print("=" * 60)

if __name__ == '__main__':
    verify_edge_mapping()
