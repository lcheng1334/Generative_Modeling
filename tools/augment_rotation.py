"""
数据增强工具 - 旋转变换

功能：
- 对Cam1和Cam2的图像进行旋转增强
- 可选择：左旋90°、右旋90°、旋转180°
- 支持批量处理

使用方法：
    # 全部变换（1张变4张）
    python tools/augment_rotation.py --folder "path/to/images" --all
    
    # 只左旋90°
    python tools/augment_rotation.py --folder "path/to/images" --left90
    
    # 左旋90° + 180°
    python tools/augment_rotation.py --folder "path/to/images" --left90 --rotate180
    
    # 预览模式（不实际保存）
    python tools/augment_rotation.py --folder "path/to/images" --all --preview
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import re
from tqdm import tqdm


def augment_images(folder_path: str, 
                   left90: bool = False, 
                   right90: bool = False, 
                   rotate180: bool = False,
                   preview: bool = False):
    """
    对Cam1和Cam2的图像进行旋转增强
    
    Args:
        folder_path: 图像文件夹路径
        left90: 是否左旋90°
        right90: 是否右旋90°
        rotate180: 是否旋转180°
        preview: 预览模式，不实际保存
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return
    
    # 查找Cam1和Cam2的图像
    pattern = re.compile(r'^Cam[12].*\.(png|jpg|bmp|PNG|JPG|BMP)$', re.IGNORECASE)
    
    images = []
    for ext in ['*.png', '*.jpg', '*.bmp', '*.PNG', '*.JPG', '*.BMP']:
        for img_path in folder.rglob(ext):
            if pattern.match(img_path.name):
                images.append(img_path)
    
    if not images:
        print(f"⚠️ 未找到Cam1或Cam2的图像")
        return
    
    print(f"找到 {len(images)} 张Cam1/Cam2图像")
    print(f"增强选项: 左旋90°={left90}, 右旋90°={right90}, 旋转180°={rotate180}")
    
    if preview:
        print("\n🔍 预览模式 - 不会实际保存文件\n")
    
    # 统计
    total_created = 0
    
    for img_path in tqdm(images, desc="处理图像", unit="张"):
        try:
            img = Image.open(img_path)
            stem = img_path.stem  # 文件名不含扩展名
            suffix = img_path.suffix  # 扩展名
            parent = img_path.parent  # 父目录
            
            # 左旋90° (逆时针)
            if left90:
                rotated = img.rotate(90, expand=True)
                new_name = f"{stem}_L90{suffix}"
                new_path = parent / new_name
                if not preview:
                    rotated.save(new_path)
                total_created += 1
            
            # 右旋90° (顺时针)
            if right90:
                rotated = img.rotate(-90, expand=True)
                new_name = f"{stem}_R90{suffix}"
                new_path = parent / new_name
                if not preview:
                    rotated.save(new_path)
                total_created += 1
            
            # 旋转180°
            if rotate180:
                rotated = img.rotate(180, expand=True)
                new_name = f"{stem}_180{suffix}"
                new_path = parent / new_name
                if not preview:
                    rotated.save(new_path)
                total_created += 1
                
        except Exception as e:
            tqdm.write(f"❌ 处理失败 {img_path.name}: {e}")
    
    print(f"\n{'预览' if preview else '完成'}! 共{'将创建' if preview else '创建了'} {total_created} 张增强图像")
    print(f"原始图像: {len(images)} 张")
    print(f"增强后总计: {len(images) + total_created} 张")


def main():
    parser = argparse.ArgumentParser(description="数据增强工具 - 旋转变换")
    
    parser.add_argument("--folder", "-f", type=str, required=True,
                        help="图像文件夹路径")
    
    # 变换选项
    parser.add_argument("--left90", "-l", action="store_true",
                        help="左旋90°（逆时针）")
    parser.add_argument("--right90", "-r", action="store_true",
                        help="右旋90°（顺时针）")
    parser.add_argument("--rotate180", "-180", action="store_true",
                        help="旋转180°")
    parser.add_argument("--all", "-a", action="store_true",
                        help="应用所有变换（左旋90°、右旋90°、旋转180°）")
    
    # 预览模式
    parser.add_argument("--preview", "-p", action="store_true",
                        help="预览模式，不实际保存文件")
    
    args = parser.parse_args()
    
    # 处理--all选项
    left90 = args.left90 or args.all
    right90 = args.right90 or args.all
    rotate180 = args.rotate180 or args.all
    
    if not (left90 or right90 or rotate180):
        print("❌ 请至少选择一种变换:")
        print("  --left90   左旋90°")
        print("  --right90  右旋90°")
        print("  --rotate180 旋转180°")
        print("  --all      全部变换")
        return
    
    print("=" * 50)
    print("数据增强工具 - 旋转变换")
    print("=" * 50)
    print(f"目标文件夹: {args.folder}")
    print(f"处理对象: Cam1*.png, Cam2*.png")
    print()
    
    augment_images(
        folder_path=args.folder,
        left90=left90,
        right90=right90,
        rotate180=rotate180,
        preview=args.preview
    )


if __name__ == "__main__":
    main()
