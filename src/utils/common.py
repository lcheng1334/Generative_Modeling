"""
通用工具函数
Common utility functions
"""
import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional
import yaml
from loguru import logger


def set_seed(seed: int = 42):
    """
    设置随机种子以保证可复现性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> dict:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def create_directories(paths: list):
    """
    创建必要的目录
    
    Args:
        paths: 目录路径列表
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created {len(paths)} directories")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    获取计算设备
    
    Args:
        device: 指定设备 ("cuda", "cpu", None)
        
    Returns:
        torch.device对象
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(device)
    
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """用于跟踪训练指标的平均值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(log_dir: str, log_file: str = "training.log"):
    """
    设置日志系统
    
    Args:
        log_dir: 日志目录
        log_file: 日志文件名
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    # 移除默认handler
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # 添加文件输出
    logger.add(
        log_path,
        rotation="500 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    
    logger.info(f"Logger initialized. Log file: {log_path}")
