"""
Phase 1 最终效果图 - 优化布局版本
"""
import sys
sys.path.insert(0, 'E:/code/Generative_Modeling')

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')

from src.datasets.multiview_dataset import MultiViewDataset
from src.utils.image_utils import extract_edge_regions

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

def denormalize(tensor):
    """反归一化"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def create_clean_visualization():
    """创建布局清晰的可视化"""
    
    root = r'E:\code\Generative_Modeling\data\datasets'
    output_dir = Path(r'E:\code\Generative_Modeling\outputs\phase1')
    
    # 加载数据
    dataset = MultiViewDataset(root, group='Group1', label='OK')
    
    # 收集每个视角的一张图
    view_samples = {}
    for view in ['Cam1', 'Cam2', 'Cam3', 'Cam4', 'Cam5', 'Cam6']:
        for idx, sample in enumerate(dataset.samples):
            if sample['view'] == view:
                view_samples[view] = dataset[idx]
                break
    
    # 创建图（3行布局）
    fig = plt.figure(figsize=(18, 14))
    
    # ==================== Section 1: 6视角物理布局（十字型） ====================
    print("Section 1: Physical Layout (Cross Pattern)...")
    
    gs1 = fig.add_gridspec(3, 3, left=0.05, right=0.95, top=0.95, bottom=0.68, 
                           wspace=0.3, hspace=0.3)
    
    # 十字型布局
    ax_cam4 = fig.add_subplot(gs1[0, 1])  # 上
    ax_cam1 = fig.add_subplot(gs1[1, 1])  # 中心
    ax_cam3 = fig.add_subplot(gs1[1, 0])  # 左
    ax_cam5 = fig.add_subplot(gs1[1, 2])  # 右
    ax_cam6 = fig.add_subplot(gs1[2, 1])  # 下
    ax_cam2 = fig.add_subplot(gs1[2, 0])  # 底面（左下角）
    
    view_configs = [
        (ax_cam1, 'Cam1', 'Cam1\n(Front View)', 'black'),
        (ax_cam2, 'Cam2', 'Cam2\n(Bottom)', 'black'),
        (ax_cam3, 'Cam3', 'Cam3\n(Left Side)', 'red'),
        (ax_cam4, 'Cam4', 'Cam4\n(Upper Side)\nDark Field', 'blue'),
        (ax_cam5, 'Cam5', 'Cam5\n(Right Side)', 'green'),
        (ax_cam6, 'Cam6', 'Cam6\n(Lower Side)\nDark Field', 'orange'),
    ]
    
    for ax, view, title, color in view_configs:
        img = denormalize(view_samples[view]['image']).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.axis('off')
        # 彩色边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    # 添加箭头标注对应关系
    arrow_props = dict(arrowstyle='->', lw=2.5, color='gray', alpha=0.6)
    
    # ==================== Section 2: 边缘提取 ====================
    print("Section 2: Edge Extraction...")
    
    gs2 = fig.add_gridspec(2, 6, left=0.05, right=0.95, top=0.62, bottom=0.36,
                           wspace=0.4, hspace=0.4)
    
    cam1_img = view_samples['Cam1']['image']
    edges = extract_edge_regions(cam1_img, edge_width=32)
    
    # 2.1 原图
    ax = fig.add_subplot(gs2[0, 0:2])
    img_show = denormalize(cam1_img).permute(1, 2, 0).numpy()
    ax.imshow(np.clip(img_show, 0, 1))
    ax.set_title('Cam1 Front View\n(Original)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # 2.2 标注边缘
    ax = fig.add_subplot(gs2[0, 2:4])
    ax.imshow(np.clip(img_show, 0, 1))
    H, W = img_show.shape[:2]
    edge_w = 32
    
    # 绘制边缘框
    rect_configs = [
        (0, 0, edge_w, H, 'red', 'L'),
        (W-edge_w, 0, edge_w, H, 'green', 'R'),
        (0, 0, W, edge_w, 'blue', 'T'),
        (0, H-edge_w, W, edge_w, 'orange', 'B')
    ]
    
    for x, y, w, h, color, label in rect_configs:
        rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=4)
        ax.add_patch(rect)
    
    # 添加标签
    ax.text(edge_w/2, H/2, 'L', color='red', fontsize=24, fontweight='bold', 
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.text(W-edge_w/2, H/2, 'R', color='green', fontsize=24, fontweight='bold',
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.text(W/2, edge_w/2, 'T', color='blue', fontsize=24, fontweight='bold',
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.text(W/2, H-edge_w/2, 'B', color='orange', fontsize=24, fontweight='bold',
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_title('Edge Regions\nL/R/T/B', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # 2.3 提取的4个边缘
    edge_configs = [
        ('left', 'L → Cam3', (0, 4), 'red'),
        ('right', 'R → Cam5', (0, 5), 'green'),
        ('top', 'T → Cam4', (1, 4), 'blue'),
        ('bottom', 'B → Cam6', (1, 5), 'orange')
    ]
    
    for edge_name, title, pos, color in edge_configs:
        ax = fig.add_subplot(gs2[pos[0], pos[1]])
        edge_img = denormalize(edges[edge_name]).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(edge_img, 0, 1))
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
    
    # ==================== Section 3: 对应的侧面视角 ====================
    print("Section 3: Corresponding Side Views...")
    
    gs3 = fig.add_gridspec(1, 4, left=0.15, right=0.85, top=0.28, bottom=0.05,
                           wspace=0.5)
    
    side_configs = [
        ('Cam3', 'Cam3 (Left Side)', 0, 'red'),
        ('Cam5', 'Cam5 (Right Side)', 1, 'green'),
        ('Cam4', 'Cam4 (Upper Side)', 2, 'blue'),
        ('Cam6', 'Cam6 (Lower Side)', 3, 'orange')
    ]
    
    for view, title, col, color in side_configs:
        ax = fig.add_subplot(gs3[0, col])
        img = denormalize(view_samples[view]['image']).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(5)
    
    # 总标题和说明
    fig.suptitle('Phase 1: Multi-View Dataset & Cross-View Edge Correspondence', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    fig.text(0.5, 0.965, 'Part 1: 6-View Physical Layout (Cross Pattern)', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    fig.text(0.5, 0.635, 'Part 2: Edge Extraction from Cam1 Front View', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    fig.text(0.5, 0.295, 'Part 3: Corresponding Side Views (Cross-View Consistency)', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    save_path = output_dir / 'phase1_clean_layout.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved clean layout: {save_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Clean Layout Visualization Complete!")
    print("=" * 60)

if __name__ == '__main__':
    create_clean_visualization()
