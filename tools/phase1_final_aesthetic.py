"""
Phase 1 Final Visualization - Aesthetic & Clean Version
"""
import sys
sys.path.insert(0, 'E:/code/Generative_Modeling')

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')

from src.datasets.multiview_dataset import MultiViewDataset
from src.utils.image_utils import extract_edge_regions

# 设置全局字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5

def denormalize(tensor):
    """反归一化"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def create_aesthetic_visualization():
    """创建美观、无重叠的可视化图"""
    
    root = r'E:\code\Generative_Modeling\data\datasets'
    output_dir = Path(r'E:\code\Generative_Modeling\outputs\phase1')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据 Sample
    dataset = MultiViewDataset(root, group='Group1', label='OK')
    
    view_samples = {}
    for view in ['Cam1', 'Cam2', 'Cam3', 'Cam4', 'Cam5', 'Cam6']:
        for idx, sample in enumerate(dataset.samples):
            if sample['view'] == view:
                view_samples[view] = dataset[idx]
                break

    # 2. 设置画布
    # 增加高度以避免重叠
    fig = plt.figure(figsize=(24, 18), facecolor='white')
    
    # 定义主要Grid: 3行 (布局, 边缘提取, 对应关系)
    # height_ratios调整每部分的高度比例
    gs_main = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1, 1], hspace=0.4)

    # =========================================================================
    # Part 1: 物理布局 (Physical Layout) - 十字型结构
    # =========================================================================
    
    # 在第一行创建一个 3x4 的子网格来放置十字型布局
    # Cam4(上), Cam1(中), Cam6(下), Cam3(左), Cam5(右), Cam2(角落)
    # 布局逻辑:
    #       Cam4
    # Cam3  Cam1  Cam5
    #       Cam6        Cam2
    
    gs_part1 = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs_main[0], wspace=0.1, hspace=0.3)
    
    # 定义位置函数 (row, col)
    layout_map = {
        'Cam4': (0, 1), # 上
        'Cam3': (1, 0), # 左
        'Cam1': (1, 1), # 中
        'Cam5': (1, 2), # 右
        'Cam6': (2, 1), # 下
        'Cam2': (2, 3)  # 右下角独立展示
    }
    
    colors = {
        'Cam1': 'black', 'Cam2': 'gray',
        'Cam3': '#FF3333', # Red
        'Cam4': '#0066CC', # Blue
        'Cam5': '#009933', # Green
        'Cam6': '#FF9900'  # Orange
    }

    # 绘制 Part 1
    # 添加区域标题
    ax_header1 = fig.add_subplot(gs_main[0])
    ax_header1.axis('off')
    ax_header1.set_title('Part 1: 6-View Physical Layout (Cross Pattern)', fontsize=20, fontweight='bold', pad=20)

    for view, (row, col) in layout_map.items():
        ax = fig.add_subplot(gs_part1[row, col])
        img = denormalize(view_samples[view]['image']).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        
        # 边框颜色
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[view])
            spine.set_linewidth(3)
            
        # 标题 (放在下方或上方避免挤压)
        is_dark = view in ['Cam4', 'Cam6']
        lighting_text = '\n(Dark Field)' if is_dark else ''
        ax.set_title(f"{view}\n{lighting_text}", fontsize=12, color=colors[view], fontweight='bold')
        ax.axis('off')

    # =========================================================================
    # Part 2: 边缘提取流程 (Edge Extraction)
    # =========================================================================
    
    gs_part2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[1], wspace=0.2)
    
    # 标题
    ax_header2 = fig.add_subplot(gs_main[1])
    ax_header2.axis('off')
    ax_header2.set_title('Part 2: Edge Extraction Logic (From Cam1)', fontsize=20, fontweight='bold', pad=20)
    
    # 2.1 原始图
    ax_orig = fig.add_subplot(gs_part2[0, 0])
    cam1_img = view_samples['Cam1']['image']
    img_np = denormalize(cam1_img).permute(1, 2, 0).numpy()
    img_display = np.clip(img_np, 0, 1)
    
    ax_orig.imshow(img_display)
    ax_orig.set_title("Original Cam1", fontsize=14, pad=10)
    ax_orig.axis('off')
    
    # 2.2 标注图 (箭头指向)
    ax_arrow = fig.add_subplot(gs_part2[0, 1])
    ax_arrow.text(0.5, 0.5, "Extract\nRegions\n----->", fontsize=20, ha='center', va='center')
    ax_arrow.axis('off')
    
    # 2.3 边缘标注图
    ax_annot = fig.add_subplot(gs_part2[0, 2])
    ax_annot.imshow(img_display)
    H, W = img_display.shape[:2]
    w_edge = 32
    
    # 绘制框
    rects = [
        (0, 0, w_edge, H, colors['Cam3']), # Left -> Cam3
        (W-w_edge, 0, w_edge, H, colors['Cam5']), # Right -> Cam5
        (0, 0, W, w_edge, colors['Cam4']), # Top -> Cam4
        (0, H-w_edge, W, w_edge, colors['Cam6']) # Bottom -> Cam6
    ]
    
    for x, y, w, h, c in rects:
        rect = patches.Rectangle((x,y), w, h, linewidth=3, edgecolor=c, facecolor='none')
        ax_annot.add_patch(rect)
    
    ax_annot.set_title("Edge Regions Mapped", fontsize=14, pad=10)
    ax_annot.axis('off')
    
    # 2.4 提取结果(拼在一起展示)
    ax_result = fig.add_subplot(gs_part2[0, 3])
    ax_result.axis('off')
    
    # 在这个子图里再画4个小条
    # 使用 inset_axes 或者 subgridspec
    gs_strips = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_part2[0, 3], wspace=0.1, hspace=0.1)
    
    edges = extract_edge_regions(cam1_img, edge_width=32)
    strip_map = [
        (0, 0, 'left', 'Left -> Cam3', colors['Cam3']),
        (0, 1, 'right', 'Right -> Cam5', colors['Cam5']),
        (1, 0, 'top', 'Top -> Cam4', colors['Cam4']),
        (1, 1, 'bottom', 'Bot -> Cam6', colors['Cam6'])
    ]
    
    for r, c, key, txt, col in strip_map:
        ax_s = fig.add_subplot(gs_strips[r, c])
        strip_img = denormalize(edges[key]).permute(1, 2, 0).numpy()
        ax_s.imshow(np.clip(strip_img, 0, 1))
        for sp in ax_s.spines.values():
            sp.set_edgecolor(col)
            sp.set_linewidth(2)
        ax_s.set_title(txt, fontsize=10, color=col)
        ax_s.axis('off')

    # =========================================================================
    # Part 3: 跨视角一致性 (Cross-View Consistency)
    # =========================================================================
    
    # 这里的逻辑是：将 Cam1 的边缘 与 对应的侧面视角 并排展示
    
    ax_header3 = fig.add_subplot(gs_main[2])
    ax_header3.axis('off')
    ax_header3.set_title('Part 3: Verification - Does Edge Match Side View?', fontsize=20, fontweight='bold', pad=30)
    
    gs_part3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[2], wspace=0.3)
    
    # 4组对比
    # Group 1: Left Edge vs Cam3
    # Group 2: Top Edge vs Cam4
    # ...
    
    pairs = [
        ('Cam3', 'Left Edge', 'left', 0),
        ('Cam4', 'Top Edge', 'top', 1),
        ('Cam5', 'Right Edge', 'right', 2),
        ('Cam6', 'Bottom Edge', 'bottom', 3)
    ]
    
    for view_name, edge_label, edge_key, col_idx in pairs:
        # 每组里面再分两块：上(边缘) 下(侧面)
        gs_pair = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_part3[0, col_idx], height_ratios=[1, 3], hspace=0.05)
        
        c = colors[view_name]
        
        # 上：边缘
        ax_e = fig.add_subplot(gs_pair[0, 0])
        e_img = denormalize(edges[edge_key]).permute(1, 2, 0).numpy()
        ax_e.imshow(np.clip(e_img, 0, 1))
        ax_e.axis('off')
        ax_e.set_title(f"Cam1 {edge_label}", fontsize=12, color=c)
        
        # 下：侧面视图
        ax_v = fig.add_subplot(gs_pair[1, 0])
        v_img = denormalize(view_samples[view_name]['image']).permute(1, 2, 0).numpy()
        ax_v.imshow(np.clip(v_img, 0, 1))
        ax_v.axis('off')
        
        # 加个大框把这组括起来
        rect = patches.Rectangle((0,0), 1, 1, transform=ax_v.transAxes, linewidth=4, edgecolor=c, facecolor='none')
        ax_v.add_patch(rect)
        ax_v.text(0.5, -0.15, f"Matches\n{view_name}", transform=ax_v.transAxes, ha='center', va='top', fontsize=12, fontweight='bold', color=c)

    # 保存
    save_path = output_dir / 'phase1_aesthetic_layout.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
    print(f"\n[OK] Saved aesthetic visualization: {save_path}")

if __name__ == '__main__':
    create_aesthetic_visualization()
