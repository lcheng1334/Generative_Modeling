"""
6视角样本可视化工具
Visualize 6-view inductor samples

用法:
    python tools/visualize_samples.py --data_dir data/samples/inductor
    python tools/visualize_samples.py --data_dir data/samples/inductor --save_dir data/visualizations
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
from typing import List, Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.common import setup_logger

logger = setup_logger(__name__)


class SampleVisualizer:
    """6视角样本可视化器"""
    
    def __init__(self, data_dir: str, view_names: Optional[List[str]] = None):
        """
        Args:
            data_dir: 包含6视角图像的目录
            view_names: 视角名称列表
        """
        self.data_dir = Path(data_dir)
        self.view_names = view_names or [
            "正面", "底面", "前侧面", "后侧面", "左侧面", "右侧面"
        ]
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    def find_sample_groups(self) -> Dict[str, List[Path]]:
        """
        查找6视角图像组
        
        假设命名规则:
        - sample_001_view1.bmp, sample_001_view2.bmp, ...
        - 或者按文件夹组织: sample_001/view1.bmp, ...
        
        Returns:
            {sample_id: [path_view1, path_view2, ...]}
        """
        groups = {}
        
        # 方法1: 查找所有图像文件
        image_files = []
        for ext in ['*.bmp', '*.png', '*.jpg']:
            image_files.extend(self.data_dir.glob(ext))
            image_files.extend(self.data_dir.glob(f'**/{ext}'))
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        # 方法2: 按文件夹组织
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                sample_id = subdir.name
                views = sorted(subdir.glob('*.*'))
                views = [v for v in views if v.suffix.lower() in ['.bmp', '.png', '.jpg']]
                
                if len(views) >= 1:  # 至少有一张图
                    groups[sample_id] = views
        
        # 方法3: 按文件名前缀分组
        if not groups:
            from collections import defaultdict
            prefix_groups = defaultdict(list)
            
            for img_file in image_files:
                # 尝试提取前缀（去掉view数字）
                name = img_file.stem
                # 假设格式: prefix_viewX 或 prefix_X
                parts = name.split('_')
                if len(parts) >= 2:
                    prefix = '_'.join(parts[:-1])
                    prefix_groups[prefix].append(img_file)
            
            # 只保留有6张图的组
            for prefix, files in prefix_groups.items():
                if len(files) >= 1:
                    groups[prefix] = sorted(files)
        
        logger.info(f"找到 {len(groups)} 个样本组")
        return groups
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """加载图像"""
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"无法加载图像: {image_path}")
            # 返回占位符
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def create_grid_visualization(
        self, 
        images: List[np.ndarray], 
        titles: Optional[List[str]] = None,
        grid_size: tuple = (2, 3)
    ) -> np.ndarray:
        """
        创建网格可视化
        
        Args:
            images: 图像列表
            titles: 标题列表
            grid_size: 网格大小 (rows, cols)
        
        Returns:
            合并后的网格图像
        """
        rows, cols = grid_size
        n_images = len(images)
        
        if n_images == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 统一图像尺寸
        h, w = images[0].shape[:2]
        
        # 创建画布
        canvas_h = h * rows + 50 * rows  # 每行额外50px用于标题
        canvas_w = w * cols
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        for idx, img in enumerate(images):
            if idx >= rows * cols:
                break
            
            row = idx // cols
            col = idx % cols
            
            # 计算位置
            y_start = row * (h + 50) + 30  # 30px用于标题
            y_end = y_start + h
            x_start = col * w
            x_end = x_start + w
            
            # 放置图像
            canvas[y_start:y_end, x_start:x_end] = img
            
            # 添加标题
            if titles and idx < len(titles):
                title_y = row * (h + 50) + 20
                title_x = x_start + 10
                cv2.putText(
                    canvas, 
                    titles[idx], 
                    (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
        
        return canvas
    
    def visualize_sample(
        self, 
        sample_id: str, 
        image_paths: List[Path],
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        可视化单个样本的6视角
        
        Args:
            sample_id: 样本ID
            image_paths: 图像路径列表
            save_path: 保存路径
            show: 是否显示
        """
        logger.info(f"可视化样本: {sample_id}")
        
        # 加载图像
        images = [self.load_image(p) for p in image_paths[:6]]
        
        # 创建标题
        titles = []
        for i, path in enumerate(image_paths[:6]):
            if i < len(self.view_names):
                title = f"{self.view_names[i]} - {path.name}"
            else:
                title = path.name
            titles.append(title)
        
        # 创建网格
        grid = self.create_grid_visualization(images, titles)
        
        # 添加样本ID水印
        cv2.putText(
            grid,
            f"Sample: {sample_id}",
            (20, grid.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (100, 100, 100),
            2
        )
        
        # 保存
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            logger.info(f"保存到: {save_path}")
        
        # 显示
        if show:
            cv2.imshow(f"Sample: {sample_id}", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return grid
    
    def generate_statistics(self, groups: Dict[str, List[Path]]) -> Dict:
        """生成数据统计"""
        stats = {
            'total_samples': len(groups),
            'samples_with_6_views': 0,
            'samples_with_less_views': 0,
            'view_count_distribution': {},
            'image_sizes': [],
        }
        
        for sample_id, paths in groups.items():
            n_views = len(paths)
            if n_views == 6:
                stats['samples_with_6_views'] += 1
            else:
                stats['samples_with_less_views'] += 1
            
            stats['view_count_distribution'][n_views] = \
                stats['view_count_distribution'].get(n_views, 0) + 1
            
            # 检查图像尺寸
            if paths:
                img = self.load_image(paths[0])
                stats['image_sizes'].append(img.shape[:2])
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("数据集统计")
        logger.info("=" * 60)
        logger.info(f"总样本数: {stats['total_samples']}")
        logger.info(f"完整6视角样本: {stats['samples_with_6_views']}")
        logger.info(f"不完整样本: {stats['samples_with_less_views']}")
        logger.info("")
        logger.info("视角数量分布:")
        for n_views, count in sorted(stats['view_count_distribution'].items()):
            logger.info(f"  {n_views} 视角: {count} 个样本")
        
        if stats['image_sizes']:
            logger.info("")
            logger.info("图像尺寸:")
            unique_sizes = set(stats['image_sizes'])
            for size in unique_sizes:
                count = stats['image_sizes'].count(size)
                logger.info(f"  {size[0]}x{size[1]}: {count} 张图像")
        
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='6视角样本可视化工具')
    parser.add_argument('--data_dir', type=str, default='data/samples/inductor',
                        help='数据目录路径')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存可视化结果的目录（可选）')
    parser.add_argument('--max_samples', type=int, default=5,
                        help='最多可视化的样本数')
    parser.add_argument('--no_show', action='store_true',
                        help='不显示图像窗口')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = SampleVisualizer(args.data_dir)
    
    # 查找样本
    groups = visualizer.find_sample_groups()
    
    if not groups:
        logger.error("未找到任何样本！请检查数据目录。")
        return
    
    # 生成并打印统计
    stats = visualizer.generate_statistics(groups)
    visualizer.print_statistics(stats)
    
    # 可视化前N个样本
    save_dir = Path(args.save_dir) if args.save_dir else None
    
    for idx, (sample_id, paths) in enumerate(groups.items()):
        if idx >= args.max_samples:
            break
        
        save_path = None
        if save_dir:
            save_path = save_dir / f"{sample_id}_6views.png"
        
        visualizer.visualize_sample(
            sample_id, 
            paths, 
            save_path=save_path,
            show=not args.no_show
        )
    
    logger.info(f"可视化完成！共处理 {min(len(groups), args.max_samples)} 个样本。")


if __name__ == '__main__':
    main()
