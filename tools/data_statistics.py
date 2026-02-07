"""
6视角样本数据统计工具
Quick data statistics for 6-view samples

用法:
    python tools/data_statistics.py --data_dir data/samples/inductor
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
from typing import Dict, List
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.common import setup_logger

logger = setup_logger(__name__)


def analyze_directory(data_dir: Path) -> Dict:
    """分析数据目录"""
    
    stats = {
        'total_files': 0,
        'image_files': 0,
        'by_extension': defaultdict(int),
        'by_view': defaultdict(list),
        'image_sizes': defaultdict(int),
        'total_size_mb': 0,
    }
    
    # 收集所有文件
    all_files = list(data_dir.glob('*.*'))
    all_files.extend(data_dir.glob('**/*.*'))
    
    view_keywords = ['正面', '底面', '前侧面', '后侧面', '左侧面', '右侧面']
    
    for file_path in all_files:
        if not file_path.is_file():
            continue
        
        stats['total_files'] += 1
        ext = file_path.suffix.lower()
        
        # 统计扩展名
        stats['by_extension'][ext] += 1
        
        # 检查是否是图像
        if ext in ['.bmp', '.png', '.jpg', '.jpeg']:
            stats['image_files'] += 1
            
            # 统计文件大小
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            stats['total_size_mb'] += file_size_mb
            
            # 识别视角
            file_name = file_path.stem
            for view in view_keywords:
                if view in file_name:
                    stats['by_view'][view].append(file_path)
                    break
            
            # 读取图像尺寸
            try:
                img = cv2.imread(str(file_path))
                if img is not None:
                    h, w = img.shape[:2]
                    size_key = f"{w}x{h}"
                    stats['image_sizes'][size_key] += 1
            except:
                pass
    
    return stats


def print_statistics(stats: Dict):
    """打印统计信息"""
    print("=" * 70)
    print("数据集统计报告")
    print("=" * 70)
    print()
    
    print(f"📁 文件统计:")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  图像文件数: {stats['image_files']}")
    print(f"  总大小: {stats['total_size_mb']:.2f} MB")
    print()
    
    print(f"📊 文件类型分布:")
    for ext, count in sorted(stats['by_extension'].items()):
        print(f"  {ext}: {count} 个文件")
    print()
    
    print(f"👁 视角分布:")
    total_views = 0
    for view, files in sorted(stats['by_view'].items()):
        print(f"  {view}: {len(files)} 张图像")
        total_views += len(files)
    print(f"  总计: {total_views} 张视角图像")
    print()
    
    # 检查完整性
    view_counts = [len(files) for files in stats['by_view'].values()]
    if view_counts:
        min_views = min(view_counts)
        max_views = max(view_counts)
        if min_views == max_views:
            print(f"✅ 数据完整: 每个视角都有 {min_views} 张图像")
        else:
            print(f"⚠️  数据不均衡: 视角图像数量在 {min_views} 到 {max_views} 之间")
    print()
    
    print(f"🖼️  图像尺寸分布:")
    for size, count in sorted(stats['image_sizes'].items()):
        print(f"  {size}: {count} 张图像")
    print()
    
    # 估算可用样本数
    if len(stats['by_view']) == 6:
        min_samples = min(len(files) for files in stats['by_view'].values())
        print(f"📦 估算可用样本组数: {min_samples}")
        print(f"   (假设每组包含6个视角)")
    
    print("=" * 70)


def list_files_by_view(stats: Dict, output_file: Path = None):
    """列出按视角分组的文件"""
    
    output = []
    output.append("\n视角文件清单:\n")
    output.append("=" * 70 + "\n")
    
    for view, files in sorted(stats['by_view'].items()):
        output.append(f"\n### {view} ({len(files)} 个文件)\n")
        for file_path in sorted(files):
            output.append(f"  - {file_path.name}\n")
    
    result = ''.join(output)
    print(result)
    
    if output_file:
        output_file.write_text(result, encoding='utf-8')
        logger.info(f"文件清单已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='数据统计工具')
    parser.add_argument('--data_dir', type=str, default='data/samples/inductor',
                        help='数据目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出报告文件路径')
    parser.add_argument('--list_files', action='store_true',
                        help='列出所有文件')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    logger.info(f"分析数据目录: {data_dir}")
    
    # 分析数据
    stats = analyze_directory(data_dir)
    
    # 打印统计
    print_statistics(stats)
    
    # 列出文件
    if args.list_files:
        output_file = Path(args.output) if args.output else None
        list_files_by_view(stats, output_file)


if __name__ == '__main__':
    main()
