"""
数据集验证工具
Validate dataset quality and completeness

用法:
    python tools/validate_dataset.py --data_dir data/samples/inductor/normal
    python tools/validate_dataset.py --data_dir data/samples/inductor --recursive
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.common import setup_logger

logger = setup_logger(__name__)


class DatasetValidator:
    """数据集验证器"""
    
    # 质量标准
    MIN_RESOLUTION = (640, 480)  # 最小分辨率
    MAX_FILE_SIZE_MB = 10  # 最大文件大小
    MIN_FILE_SIZE_KB = 50  # 最小文件大小（防止空文件）
    
    # 视角关键词
    VIEW_KEYWORDS = {
        '正面': ['正面', 'front', 'view1'],
        '底面': ['底面', 'bottom', 'view2'],
        '前侧面': ['前侧面', 'front_side', 'view3'],
        '后侧面': ['后侧面', 'back_side', 'view4'],
        '左侧面': ['左侧面', 'left', 'view5'],
        '右侧面': ['右侧面', 'right', 'view6'],
    }
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.issues = []
        self.warnings = []
        self.stats = {
            'total_samples': 0,
            'complete_samples': 0,
            'incomplete_samples': 0,
            'total_images': 0,
            'quality_issues': 0,
        }
    
    def find_sample_groups(self) -> Dict[str, Dict[str, Path]]:
        """
        查找样本组（每组应该有6个视角）
        
        Returns:
            {sample_id: {view_name: image_path}}
        """
        groups = defaultdict(dict)
        
        # 检查是否有子文件夹组织
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # 按文件夹组织
            for subdir in subdirs:
                sample_id = subdir.name
                for img_file in subdir.glob('*.*'):
                    if img_file.suffix.lower() in ['.bmp', '.png', '.jpg', '.jpeg']:
                        view_name = self._identify_view(img_file.stem)
                        if view_name:
                            groups[sample_id][view_name] = img_file
        else:
            # 按文件名前缀组织
            image_files = []
            for ext in ['.bmp', '.png', '.jpg', '.jpeg']:
                image_files.extend(self.data_dir.glob(f'*{ext}'))
            
            # 分组
            prefix_groups = defaultdict(list)
            for img_file in image_files:
                # 提取前缀
                parts = img_file.stem.split('_')
                if len(parts) >= 2:
                    # 假设格式: prefix_view 或 prefix_id_view
                    prefix = '_'.join(parts[:-1])
                    prefix_groups[prefix].append(img_file)
            
            # 转换为view字典
            for prefix, files in prefix_groups.items():
                for file in files:
                    view_name = self._identify_view(file.stem)
                    if view_name:
                        groups[prefix][view_name] = file
        
        return dict(groups)
    
    def _identify_view(self, filename: str) -> str:
        """识别视角名称"""
        filename_lower = filename.lower()
        for view_name, keywords in self.VIEW_KEYWORDS.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return view_name
        return None
    
    def check_completeness(self, groups: Dict) -> List[Dict]:
        """检查6视角完整性"""
        results = []
        
        for sample_id, views in groups.items():
            result = {
                'sample_id': sample_id,
                'num_views': len(views),
                'missing_views': [],
                'is_complete': False,
            }
            
            # 检查是否有全部6个视角
            expected_views = set(self.VIEW_KEYWORDS.keys())
            actual_views = set(views.keys())
            missing = expected_views - actual_views
            
            result['missing_views'] = list(missing)
            result['is_complete'] = len(missing) == 0
            
            if not result['is_complete']:
                self.warnings.append(
                    f"样本 {sample_id} 不完整: 缺少 {', '.join(missing)}"
                )
                self.stats['incomplete_samples'] += 1
            else:
                self.stats['complete_samples'] += 1
            
            results.append(result)
            self.stats['total_samples'] += 1
        
        return results
    
    def check_image_quality(self, image_path: Path) -> Tuple[bool, List[str]]:
        """
        检查单张图像质量
        
        Returns:
            (is_valid, issues)
        """
        issues = []
        
        # 检查文件大小
        file_size_mb = image_path.stat().st_size / (1024 * 1024)
        file_size_kb = image_path.stat().st_size / 1024
        
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            issues.append(f"文件过大: {file_size_mb:.2f}MB")
        
        if file_size_kb < self.MIN_FILE_SIZE_KB:
            issues.append(f"文件过小: {file_size_kb:.2f}KB（可能是空文件）")
        
        # 尝试加载图像
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                issues.append("无法读取图像")
                return False, issues
            
            h, w = img.shape[:2]
            
            # 检查分辨率
            if w < self.MIN_RESOLUTION[0] or h < self.MIN_RESOLUTION[1]:
                issues.append(f"分辨率过低: {w}x{h}")
            
            # 检查曝光（简单检查平均亮度）
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 30:
                issues.append(f"图像过暗: 平均亮度={mean_brightness:.1f}")
            elif mean_brightness > 225:
                issues.append(f"图像过亮: 平均亮度={mean_brightness:.1f}")
            
            # 检查模糊（Laplacian方差）
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = np.var(laplacian)
            
            if blur_score < 50:  # 阈值需要根据实际调整
                issues.append(f"可能模糊: blur_score={blur_score:.1f}")
            
        except Exception as e:
            issues.append(f"处理错误: {str(e)}")
            return False, issues
        
        return len(issues) == 0, issues
    
    def validate_all(self, groups: Dict) -> Dict:
        """验证所有样本"""
        validation_results = {
            'samples': [],
            'quality_report': {},
        }
        
        for sample_id, views in groups.items():
            sample_result = {
                'sample_id': sample_id,
                'views': {},
                'all_valid': True,
            }
            
            for view_name, image_path in views.items():
                is_valid, issues = self.check_image_quality(image_path)
                
                sample_result['views'][view_name] = {
                    'path': str(image_path),
                    'valid': is_valid,
                    'issues': issues,
                }
                
                if not is_valid:
                    sample_result['all_valid'] = False
                    self.stats['quality_issues'] += 1
                    self.issues.append(
                        f"{sample_id}/{view_name}: {', '.join(issues)}"
                    )
                
                self.stats['total_images'] += 1
            
            validation_results['samples'].append(sample_result)
        
        return validation_results
    
    def generate_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 70)
        report.append("数据集验证报告")
        report.append("=" * 70)
        report.append("")
        
        # 统计信息
        report.append("📊 统计信息:")
        report.append(f"  总样本数: {self.stats['total_samples']}")
        report.append(f"  完整样本数 (6视角): {self.stats['complete_samples']} ✓")
        report.append(f"  不完整样本数: {self.stats['incomplete_samples']}")
        report.append(f"  总图像数: {self.stats['total_images']}")
        report.append(f"  质量问题数: {self.stats['quality_issues']}")
        report.append("")
        
        # 完整性评估
        if self.stats['total_samples'] > 0:
            completeness = self.stats['complete_samples'] / self.stats['total_samples'] * 100
            report.append(f"📈 完整性: {completeness:.1f}%")
            
            if completeness == 100:
                report.append("  ✅ 所有样本都有完整的6视角！")
            elif completeness >= 80:
                report.append("  ⚠️  大部分样本完整，但仍有缺失")
            else:
                report.append("  ❌ 数据完整性较差，需要补充")
        report.append("")
        
        # 质量评估
        if self.stats['total_images'] > 0:
            quality_rate = (self.stats['total_images'] - self.stats['quality_issues']) / self.stats['total_images'] * 100
            report.append(f"🎯 质量合格率: {quality_rate:.1f}%")
            
            if quality_rate >= 95:
                report.append("  ✅ 图像质量优秀！")
            elif quality_rate >= 80:
                report.append("  ⚠️  质量良好，但有部分问题需要修复")
            else:
                report.append("  ❌ 质量问题较多，建议重新采集")
        report.append("")
        
        # 警告信息
        if self.warnings:
            report.append("⚠️  警告信息:")
            for warning in self.warnings[:10]:  # 最多显示10条
                report.append(f"  - {warning}")
            if len(self.warnings) > 10:
                report.append(f"  ... 还有 {len(self.warnings) - 10} 条警告")
            report.append("")
        
        # 严重问题
        if self.issues:
            report.append("❌ 质量问题:")
            for issue in self.issues[:10]:  # 最多显示10条
                report.append(f"  - {issue}")
            if len(self.issues) > 10:
                report.append(f"  ... 还有 {len(self.issues) - 10} 个问题")
            report.append("")
        
        # 总结
        report.append("=" * 70)
        report.append("📌 总结:")
        
        if self.stats['complete_samples'] >= 50 and self.stats['quality_issues'] < 10:
            report.append("  ✅ 数据集质量良好，可以开始训练！")
        elif self.stats['complete_samples'] >= 20:
            report.append("  ⚠️  数据集基本可用，但建议继续采集")
        else:
            report.append("  ❌ 数据集不足，需要继续采集")
        
        report.append("=" * 70)
        
        return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description='数据集验证工具')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出报告文件路径')
    parser.add_argument('--recursive', action='store_true',
                        help='递归检查子目录')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    logger.info(f"开始验证数据集: {data_dir}")
    
    # 创建验证器
    validator = DatasetValidator(data_dir)
    
    # 查找样本组
    groups = validator.find_sample_groups()
    logger.info(f"找到 {len(groups)} 个样本组")
    
    if not groups:
        logger.error("未找到任何样本！请检查数据目录和文件命名。")
        return
    
    # 检查完整性
    completeness_results = validator.check_completeness(groups)
    
    # 验证质量
    validation_results = validator.validate_all(groups)
    
    # 生成报告
    report = validator.generate_report()
    print(report)
    
    # 保存报告
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding='utf-8')
        logger.info(f"报告已保存到: {output_path}")


if __name__ == '__main__':
    main()
