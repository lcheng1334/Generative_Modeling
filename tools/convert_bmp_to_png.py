"""
BMP to PNG Converter with Progress Bar
将datasets文件夹中的所有BMP文件转换为PNG格式，保持相同清晰度，并删除原始BMP文件
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_bmp_to_png(datasets_dir: str) -> None:
    """
    将指定目录下所有BMP文件转换为PNG格式
    
    Args:
        datasets_dir: 数据集目录路径
    """
    datasets_path = Path(datasets_dir)
    
    # 查找所有BMP文件
    bmp_files = list(datasets_path.rglob("*.bmp"))
    
    if not bmp_files:
        print("没有找到BMP文件")
        return
    
    print(f"找到 {len(bmp_files)} 个BMP文件，开始转换...")
    
    success_count = 0
    error_count = 0
    
    # 使用tqdm显示进度条
    for bmp_path in tqdm(bmp_files, desc="转换进度", unit="文件"):
        try:
            # 打开BMP图像
            with Image.open(bmp_path) as img:
                # 生成PNG文件路径（相同目录，相同文件名，不同扩展名）
                png_path = bmp_path.with_suffix(".png")
                
                # 保存为PNG格式，使用最高压缩但无损
                # compress_level=6 是默认值，提供较好的压缩与速度平衡
                img.save(png_path, "PNG", compress_level=6)
            
            # 删除原始BMP文件
            os.remove(bmp_path)
            success_count += 1
            
        except Exception as e:
            print(f"\n转换失败: {bmp_path}")
            print(f"  错误: {e}")
            error_count += 1
    
    print(f"\n转换完成!")
    print(f"  成功: {success_count} 个文件")
    if error_count > 0:
        print(f"  失败: {error_count} 个文件")


if __name__ == "__main__":
    datasets_dir = r"E:\code\Generative_Modeling\data\datasets"
    convert_bmp_to_png(datasets_dir)
