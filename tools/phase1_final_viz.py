"""
Phase 1 最终效果图 - 清晰展示6视角关系和边缘对应
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
plt.rcParams['font.size'] = 11

def denormalize(tensor):
    """反归一化"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def create_final_visualization():
    """创建最终的Phase 1效果图"""
    
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
    
    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    
    # ============ 第一部分：6视角物理布局 ============
    print("Creating Part 1: Physical Layout...")
    
    # 按物理位置排列
    layout_positions = {
        'Cam4': (1, 2),  # 上侧面
        'Cam1': (2, 2),  # 正面（中心）
        'Cam6': (3, 2),  # 下侧面
        'Cam3': (2, 1),  # 左侧面
        'Cam5': (2, 3),  # 右侧面
        'Cam2': (2, 4),  # 底面
    }
    
    for view, (row, col) in layout_positions.items():
        ax = plt.subplot(4, 6, (row-1)*6 + col)
        img = denormalize(view_samples[view]['image']).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        
        # 视角名称
        view_names = {
            'Cam1': 'Cam1\n(Front View)',
            'Cam2': 'Cam2\n(Bottom)',
            'Cam3': 'Cam3\n(Left Side)',
            'Cam4': 'Cam4\n(Upper Side)',
            'Cam5': 'Cam5\n(Right Side)',
            'Cam6': 'Cam6\n(Lower Side)'
        }
        ax.set_title(view_names[view], fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # 添加箭头指示
    # 使用figure坐标系添加箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='red')
    
    # ============ 第二部分：边缘提取演示 ============
    print("Creating Part 2: Edge Extraction...")
    
    cam1_img = view_samples['Cam1']['image']
    edges = extract_edge_regions(cam1_img, edge_width=32)
    
    # 原图
    ax = plt.subplot(4, 6, 13)
    img_show = denormalize(cam1_img).permute(1, 2, 0).numpy()
    ax.imshow(np.clip(img_show, 0, 1))
    ax.set_title('Cam1 Original', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # 标注边缘
    ax = plt.subplot(4, 6, 14)
    ax.imshow(np.clip(img_show, 0, 1))
    H, W = img_show.shape[:2]
    edge_w = 32
    
    # 绘制边缘框并标注
    rect_left = patches.Rectangle((0, 0), edge_w, H, fill=False, edgecolor='red', linewidth=3)
    rect_right = patches.Rectangle((W-edge_w, 0), edge_w, H, fill=False, edgecolor='green', linewidth=3)
    rect_top = patches.Rectangle((0, 0), W, edge_w, fill=False, edgecolor='blue', linewidth=3)
    rect_bottom = patches.Rectangle((0, H-edge_w), W, edge_w, fill=False, edgecolor='yellow', linewidth=3)
    
    ax.add_patch(rect_left)
    ax.add_patch(rect_right)
    ax.add_patch(rect_top)
    ax.add_patch(rect_bottom)
    
    # 添加文字标注
    ax.text(edge_w/2, H/2, 'L', color='red', fontsize=20, fontweight='bold', 
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(W-edge_w/2, H/2, 'R', color='green', fontsize=20, fontweight='bold',
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(W/2, edge_w/2, 'T', color='blue', fontsize=20, fontweight='bold',
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(W/2, H-edge_w/2, 'B', color='yellow', fontsize=20, fontweight='bold',
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Edge Regions\n(L/R/T/B)', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # ============ 第三部分：边缘对应关系 ============
    print("Creating Part 3: Edge Correspondence...")
    
    # 显示4个提取的边缘
    edge_configs = [
        ('left', 'Left Edge (L)\n→ Cam3', 15, 'red'),
        ('right', 'Right Edge (R)\n→ Cam5', 16, 'green'),
        ('top', 'Top Edge (T)\n→ Cam4', 17, 'blue'),
        ('bottom', 'Bottom Edge (B)\n→ Cam6', 18, 'yellow')
    ]
    
    for edge_name, title, pos, color in edge_configs:
        ax = plt.subplot(4, 6, pos)
        edge_img = denormalize(edges[edge_name]).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(edge_img, 0, 1))
        ax.set_title(title, fontsize=9, fontweight='bold', color=color)
        ax.axis('off')
        # 添加边框
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    # ============ 第四部分：对应的侧面视角 ============
    print("Creating Part 4: Corresponding Side Views...")
    
    side_configs = [
        ('Cam3', 'Cam3\n(Left Side View)', 21, 'red'),
        ('Cam5', 'Cam5\n(Right Side View)', 22, 'green'),
        ('Cam4', 'Cam4\n(Upper Side View)', 23, 'blue'),
        ('Cam6', 'Cam6\n(Lower Side View)', 24, 'yellow')
    ]
    
    for view, title, pos, color in side_configs:
        ax = plt.subplot(4, 6, pos)
        img = denormalize(view_samples[view]['image']).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=9, fontweight='bold', color=color)
        ax.axis('off')
        # 添加边框
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    # 总标题
    fig.suptitle('Phase 1: Multi-View Dataset Analysis and Edge-to-Side Correspondence', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 添加说明文字
    fig.text(0.5, 0.72, 'Part 1: Physical Layout (Group1 - Vertical Entry)', 
             ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    fig.text(0.5, 0.46, 'Part 2-3: Edge Extraction from Cam1 Front View', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    fig.text(0.5, 0.21, 'Part 4: Corresponding Side Views (for Cross-View Consistency)', 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = output_dir / 'phase1_final_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved final visualization: {save_path}")
    
    print("\n" + "=" * 60)
    print("Phase 1 Final Visualization Complete!")
    print("=" * 60)

if __name__ == '__main__':
    create_final_visualization()
