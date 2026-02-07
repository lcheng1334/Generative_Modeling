"""
根据器件宽高比分类图像并移动

图1类型: 器件较窄（宽高比较小）
图2类型: 器件较宽（宽高比较大）- 需要移动到目标文件夹
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


def analyze_image(image_path):
    """
    分析图像中器件的宽高比
    通过检测粉色/紫色区域（器件边缘）来判断
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # 获取图像尺寸
    height, width = img_array.shape[:2]
    
    # 检测粉色/紫色区域（器件边缘的颜色）
    # 粉色区域的特征：R值高，B值中等，G值较低
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    # 检测粉色/紫色区域的掩码
    pink_mask = (r > 150) & (r < 230) & (g < 150) & (b > 100) & (b < 200)
    
    # 找到粉色区域的最左和最右边界
    pink_cols = np.any(pink_mask, axis=0)
    if not np.any(pink_cols):
        return None, None
    
    left_edge = np.argmax(pink_cols)
    right_edge = width - np.argmax(pink_cols[::-1]) - 1
    
    # 找到粉色区域的最上和最下边界
    pink_rows = np.any(pink_mask, axis=1)
    if not np.any(pink_rows):
        return None, None
    
    top_edge = np.argmax(pink_rows)
    bottom_edge = height - np.argmax(pink_rows[::-1]) - 1
    
    # 计算宽度和高度
    component_width = right_edge - left_edge
    component_height = bottom_edge - top_edge
    
    if component_height == 0:
        return None, None
    
    # 计算宽高比
    aspect_ratio = component_width / component_height
    
    return aspect_ratio, (component_width, component_height)


def classify_and_move_images(source_dir, target_dir, threshold=1.15, dry_run=True):
    """
    分类并移动宽型图像（图2类型）
    
    参数:
        source_dir: 源文件夹路径
        target_dir: 目标文件夹路径
        threshold: 宽高比阈值，大于此值的认为是"宽型"图像
        dry_run: 如果为True，只预览不实际移动
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 确保目标文件夹存在
    if not dry_run:
        target_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有bmp文件
    bmp_files = list(source_path.glob("*.bmp"))
    print(f"找到 {len(bmp_files)} 个BMP文件")
    
    narrow_images = []  # 图1类型（窄型）
    wide_images = []    # 图2类型（宽型）
    failed_images = []  # 分析失败的图像
    
    for i, img_path in enumerate(bmp_files):
        if (i + 1) % 50 == 0:
            print(f"处理进度: {i + 1}/{len(bmp_files)}")
        
        aspect_ratio, dimensions = analyze_image(img_path)
        
        if aspect_ratio is None:
            failed_images.append(img_path.name)
            continue
        
        if aspect_ratio > threshold:
            wide_images.append((img_path.name, aspect_ratio, dimensions))
        else:
            narrow_images.append((img_path.name, aspect_ratio, dimensions))
    
    print(f"\n分析完成!")
    print(f"窄型图像（图1类型）: {len(narrow_images)} 张")
    print(f"宽型图像（图2类型）: {len(wide_images)} 张")
    print(f"分析失败: {len(failed_images)} 张")
    
    # 显示一些示例
    print(f"\n窄型图像示例 (前5张):")
    for name, ratio, dims in narrow_images[:5]:
        print(f"  {name}: 宽高比={ratio:.3f}, 尺寸={dims}")
    
    print(f"\n宽型图像示例 (前5张):")
    for name, ratio, dims in wide_images[:5]:
        print(f"  {name}: 宽高比={ratio:.3f}, 尺寸={dims}")
    
    if dry_run:
        print(f"\n[预览模式] 将移动 {len(wide_images)} 张宽型图像到 {target_dir}")
        print("如果确认无误，请将 dry_run 设置为 False 来执行实际移动")
    else:
        print(f"\n正在移动 {len(wide_images)} 张图像...")
        for name, _, _ in wide_images:
            src = source_path / name
            dst = target_path / name
            shutil.move(str(src), str(dst))
        print(f"移动完成! 文件已移动到 {target_dir}")
    
    return narrow_images, wide_images, failed_images


if __name__ == "__main__":
    source_dir = r"E:\code\Generative_Modeling\data\datasets\OK\Cam1"
    target_dir = r"E:\code\Generative_Modeling\data\datasets\OK\Cam1_demo"
    
    # 首先用预览模式查看结果
    print("=" * 60)
    print("第一步: 预览模式 - 查看分类结果")
    print("=" * 60)
    
    narrow, wide, failed = classify_and_move_images(
        source_dir, target_dir, 
        threshold=1.15,  # 可以调整这个阈值
        dry_run=True     # True=预览模式, False=实际移动
    )
    
    # 如果预览结果正确，取消下面的注释来实际移动文件
    # print("\n" + "=" * 60)
    # print("第二步: 实际移动文件")
    # print("=" * 60)
    # classify_and_move_images(source_dir, target_dir, threshold=1.15, dry_run=False)
