"""
Multi-View Dataset for Industrial Inspection
支持6视角工业检测图像的加载与预处理
"""
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Optional, Tuple, Literal


class MultiViewDataset(Dataset):
    """
    多视角数据集加载器
    
    支持按视角(Cam1-6)和分组(Group1/2)加载OK/NG图像
    """
    
    # 视角名称到面的映射
    VIEW_NAMES = {
        'Cam1': '正面',
        'Cam2': '底面', 
        'Cam3': '左侧面',
        'Cam4': '上侧面',
        'Cam5': '右侧面',
        'Cam6': '下侧面'
    }
    
    # 光照条件分组
    BRIGHT_FIELD_VIEWS = ['Cam1', 'Cam2', 'Cam3', 'Cam5']  # 青/绿色背景
    DARK_FIELD_VIEWS = ['Cam4', 'Cam6']  # 黑色背景
    
    def __init__(
        self,
        root_dir: str,
        group: Literal['Group1', 'Group2', 'all'] = 'all',
        views: Optional[List[str]] = None,
        label: Literal['OK', 'NG', 'all'] = 'OK',
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (256, 256),
        return_path: bool = False
    ):
        """
        Args:
            root_dir: 数据集根目录 (e.g., E:/code/Generative_Modeling/data/datasets)
            group: 分组选择 ('Group1', 'Group2', 'all')
            views: 视角列表 (e.g., ['Cam1', 'Cam3'])，None表示全部
            label: 标签选择 ('OK', 'NG', 'all')
            transform: 图像变换
            image_size: 目标图像尺寸
            return_path: 是否返回图像路径
        """
        self.root_dir = Path(root_dir)
        self.group = group
        self.views = views or list(self.VIEW_NAMES.keys())
        self.label = label
        self.image_size = image_size
        self.return_path = return_path
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # 收集所有图像路径
        self.samples = self._collect_samples()
        
    def _collect_samples(self) -> List[Dict]:
        """收集所有符合条件的图像样本"""
        samples = []
        
        # 确定要加载的标签目录
        label_dirs = []
        if self.label in ['OK', 'all']:
            label_dirs.append(('OK', self.root_dir / 'OK'))
        if self.label in ['NG', 'all']:
            label_dirs.append(('NG', self.root_dir / 'NG'))
        
        for label_name, label_path in label_dirs:
            if not label_path.exists():
                continue
                
            if label_name == 'OK':
                # OK目录结构: OK/Group1/Cam1_Group1/images
                groups = []
                if self.group in ['Group1', 'all'] and (label_path / 'Group1').exists():
                    groups.append('Group1')
                if self.group in ['Group2', 'all'] and (label_path / 'Group2').exists():
                    groups.append('Group2')
                
                for grp in groups:
                    grp_path = label_path / grp
                    for view in self.views:
                        view_dir = grp_path / f'{view}_{grp}'
                        if view_dir.exists():
                            for img_file in view_dir.glob('*.png'):
                                samples.append({
                                    'path': img_file,
                                    'view': view,
                                    'group': grp,
                                    'label': 0,  # OK = 0
                                    'label_name': 'OK'
                                })
                            for img_file in view_dir.glob('*.bmp'):
                                samples.append({
                                    'path': img_file,
                                    'view': view,
                                    'group': grp,
                                    'label': 0,
                                    'label_name': 'OK'
                                })
            else:
                # NG目录结构: NG/Cam1/images (尚未分组)
                for view in self.views:
                    view_dir = label_path / view
                    if view_dir.exists():
                        for img_file in view_dir.glob('*.png'):
                            samples.append({
                                'path': img_file,
                                'view': view,
                                'group': 'unknown',
                                'label': 1,  # NG = 1
                                'label_name': 'NG'
                            })
                        for img_file in view_dir.glob('*.bmp'):
                            samples.append({
                                'path': img_file,
                                'view': view,
                                'group': 'unknown',
                                'label': 1,
                                'label_name': 'NG'
                            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['path']).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        result = {
            'image': image,
            'view': sample['view'],
            'group': sample['group'],
            'label': sample['label'],
            'label_name': sample['label_name']
        }
        
        if self.return_path:
            result['path'] = str(sample['path'])
        
        return result
    
    def get_view_samples(self, view: str) -> List[Dict]:
        """获取特定视角的所有样本"""
        return [s for s in self.samples if s['view'] == view]
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'total': len(self.samples),
            'by_view': {},
            'by_group': {},
            'by_label': {'OK': 0, 'NG': 0}
        }
        
        for sample in self.samples:
            view = sample['view']
            group = sample['group']
            label = sample['label_name']
            
            stats['by_view'][view] = stats['by_view'].get(view, 0) + 1
            stats['by_group'][group] = stats['by_group'].get(group, 0) + 1
            stats['by_label'][label] += 1
        
        return stats


class ViewSpecificDataset(Dataset):
    """
    特定视角的数据集 - 用于训练视角特定的模型
    
    只加载单个视角的图像，适合训练View-Specific VAE
    """
    
    def __init__(
        self,
        root_dir: str,
        view: str,
        group: Literal['Group1', 'Group2', 'all'] = 'all',
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        # 使用MultiViewDataset的子集
        self.full_dataset = MultiViewDataset(
            root_dir=root_dir,
            group=group,
            views=[view],
            label='OK',  # 只用OK样本训练正常外观
            transform=transform,
            image_size=image_size
        )
        
        self.view = view
        self.is_dark_field = view in MultiViewDataset.DARK_FIELD_VIEWS
    
    def __len__(self) -> int:
        return len(self.full_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.full_dataset[idx]


def create_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    val_split: float = 0.1
) -> Dict[str, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Returns:
        包含 'train' 和 'val' 的DataLoader字典
    """
    # 训练变换（带数据增强）
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证变换（无增强）
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建完整数据集
    full_dataset = MultiViewDataset(
        root_dir=root_dir,
        group='all',
        label='OK',
        transform=train_transform,
        image_size=image_size
    )
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }


if __name__ == '__main__':
    # 测试代码
    root = r'E:\code\Generative_Modeling\data\datasets'
    
    # 测试MultiViewDataset
    print("=" * 50)
    print("测试 MultiViewDataset")
    print("=" * 50)
    
    dataset = MultiViewDataset(root, group='all', label='OK')
    print(f"总样本数: {len(dataset)}")
    
    stats = dataset.get_statistics()
    print("\n按视角统计:")
    for view, count in sorted(stats['by_view'].items()):
        print(f"  {view}: {count}")
    
    print("\n按分组统计:")
    for group, count in sorted(stats['by_group'].items()):
        print(f"  {group}: {count}")
    
    # 测试加载单个样本
    sample = dataset[0]
    print(f"\n样本数据格式:")
    print(f"  image shape: {sample['image'].shape}")
    print(f"  view: {sample['view']}")
    print(f"  group: {sample['group']}")
    print(f"  label: {sample['label_name']}")
    
    # 测试DataLoader
    print("\n" + "=" * 50)
    print("测试 DataLoader")
    print("=" * 50)
    
    loaders = create_dataloaders(root, batch_size=8, num_workers=0)
    print(f"训练集批次数: {len(loaders['train'])}")
    print(f"验证集批次数: {len(loaders['val'])}")
    
    # 测试一个batch
    batch = next(iter(loaders['train']))
    print(f"\nBatch数据格式:")
    print(f"  images shape: {batch['image'].shape}")
    print(f"  views: {batch['view']}")
