"""
BMP to PNG Converter
将BMP格式图像批量转换为PNG格式
"""
import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_bmp_to_png(input_dir, output_dir=None, recursive=False, delete_original=False):
    """
    将目录下的BMP文件转换为PNG格式
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则覆盖原文件所在目录
        recursive: 是否递归处理子目录
        delete_original: 是否删除原BMP文件
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 查找所有BMP文件
    if recursive:
        bmp_files = list(input_path.rglob("*.bmp"))
    else:
        bmp_files = list(input_path.glob("*.bmp"))
    
    if len(bmp_files) == 0:
        print(f"在 {input_dir} 中未找到BMP文件")
        return
    
    print(f"找到 {len(bmp_files)} 个BMP文件")
    
    # 转换文件
    success_count = 0
    for bmp_file in tqdm(bmp_files, desc="转换进度"):
        try:
            # 打开BMP图像
            img = Image.open(bmp_file)
            
            # 确定输出路径
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                # 保持相对目录结构
                relative_path = bmp_file.relative_to(input_path)
                png_file = output_path / relative_path.with_suffix('.png')
                png_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                png_file = bmp_file.with_suffix('.png')
            
            # 保存为PNG
            img.save(png_file, 'PNG')
            success_count += 1
            
            # 删除原文件（如果指定）
            if delete_original:
                bmp_file.unlink()
                
        except Exception as e:
            print(f"\n转换失败: {bmp_file}, 错误: {e}")
    
    print(f"\n成功转换 {success_count}/{len(bmp_files)} 个文件")
    if delete_original:
        print(f"已删除 {success_count} 个原始BMP文件")


def main():
    parser = argparse.ArgumentParser(
        description="批量将BMP图像转换为PNG格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个目录下的BMP文件
  python bmp_to_png_converter.py data/samples/inductor
  
  # 递归转换所有子目录
  python bmp_to_png_converter.py data/samples -r
  
  # 转换并输出到指定目录
  python bmp_to_png_converter.py data/samples -o data/png_output
  
  # 转换后删除原BMP文件
  python bmp_to_png_converter.py data/samples --delete
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='输入目录路径'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default=None,
        help='输出目录路径（默认覆盖原目录）'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='递归处理子目录'
    )
    
    parser.add_argument(
        '--delete',
        action='store_true',
        help='转换后删除原BMP文件'
    )
    
    args = parser.parse_args()
    
    convert_bmp_to_png(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        delete_original=args.delete
    )


if __name__ == "__main__":
    main()
