"""
图像处理工具函数
Image Processing Utilities
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Union, Optional, Dict, Literal
from pathlib import Path


def load_image(
    image_path: str,
    size: Optional[Tuple[int, int]] = None,
    to_tensor: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    加载图像
    
    Args:
        image_path: 图像路径
        size: 目标尺寸 (width, height)，None表示不调整大小
        to_tensor: 是否转换为tensor
        
    Returns:
        图像数组或tensor
    """
    # 尝试使用PIL加载（支持BMP）
    img = Image.open(image_path).convert('RGB')
    
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    
    img_array = np.array(img)
    
    if to_tensor:
        # 转换为 (C, H, W) 并归一化到 [0, 1]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        return img_tensor
    else:
        return img_array


def save_image(
    image: Union[np.ndarray, torch.Tensor],
    save_path: str
):
    """
    保存图像
    
    Args:
        image: 图像数组或tensor
        save_path: 保存路径
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(image, torch.Tensor):
        # (C, H, W) -> (H, W, C)
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # 确保值在 [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # 保存
    Image.fromarray(image).save(save_path)


def remove_background(
    image: np.ndarray,
    lower_color: Tuple[int, int, int] = (100, 200, 200),
    upper_color: Tuple[int, int, int] = (255, 255, 255),
    return_mask: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    去除背景（基于颜色阈值）
    
    Args:
        image: RGB图像 (H, W, 3)
        lower_color: 背景颜色下界 (B, G, R)
        upper_color: 背景颜色上界
        return_mask: 是否返回mask
        
    Returns:
        去背景后的图像，可选返回mask
    """
    # 转换为 HSV 可能效果更好
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 青色背景的HSV范围
    lower_hsv = np.array([80, 50, 50])
    upper_hsv = np.array([100, 255, 255])
    
    # 创建mask（背景为0，前景为255）
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = 255 - mask  # 反转
    
    # 形态学操作去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 应用mask
    result = cv2.bitwise_and(image, image, mask=mask)
    
    if return_mask:
        return result, mask
    else:
        return result


def get_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    获取mask的边界框
    
    Args:
        mask: 二值mask (H, W)
        
    Returns:
        (x, y, w, h) 边界框坐标
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    
    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return x, y, w, h


def crop_and_pad(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int],
    pad_value: int = 0
) -> np.ndarray:
    """
    裁剪并填充到目标尺寸
    
    Args:
        image: 输入图像
        bbox: 边界框 (x, y, w, h)
        target_size: 目标尺寸 (width, height)
        pad_value: 填充值
        
    Returns:
        处理后的图像
    """
    x, y, w, h = bbox
    
    # 裁剪
    cropped = image[y:y+h, x:x+w]
    
    # 计算缩放比例（保持宽高比）
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 居中填充
    result = np.full((target_size[1], target_size[0], 3), pad_value, dtype=np.uint8)
    offset_x = (target_size[0] - new_w) // 2
    offset_y = (target_size[1] - new_h) // 2
    result[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    
    return result


def visualize_multi_view(
    images: list,
    titles: Optional[list] = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    可视化多视角图像
    
    Args:
        images: 图像列表
        titles: 标题列表
        save_path: 保存路径（可选）
        
    Returns:
        拼接后的图像
    """
    num_images = len(images)
    
    if num_images == 6:
        # 2x3布局
        rows, cols = 2, 3
    elif num_images == 4:
        rows, cols = 2, 2
    else:
        # 单行
        rows, cols = 1, num_images
    
    # 获取单张图像尺寸
    h, w = images[0].shape[:2]
    
    # 创建画布
    canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        # 如果是tensor，转换为numpy
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        
        # 确保RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # 放置到画布
        canvas[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        # 添加标题
        if titles and idx < len(titles):
            cv2.putText(
                canvas, titles[idx],
                (col*w + 10, row*h + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )
    
    if save_path:
        Image.fromarray(canvas).save(save_path)
    
    return canvas


# =============================================================================
# 边缘区域提取 - 用于跨视角一致性约束
# =============================================================================

def extract_edge_regions(
    image: torch.Tensor,
    edge_width: int = 16,
    return_dict: bool = True
) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
    """
    从正面图像(Cam1)提取四个边缘区域
    
    这些边缘区域对应四个侧面视图:
    - left_edge -> Cam3 (左侧面)
    - right_edge -> Cam5 (右侧面)  
    - top_edge -> Cam4 (上侧面)
    - bottom_edge -> Cam6 (下侧面)
    
    Args:
        image: 输入图像 (C, H, W) 或 (B, C, H, W)
        edge_width: 边缘区域宽度（像素）
        return_dict: 是否返回字典格式
        
    Returns:
        四个边缘区域的字典或元组
    """
    # 处理batch维度
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, C, H, W = image.shape
    
    # 提取四个边缘区域
    left_edge = image[:, :, :, :edge_width]           # (B, C, H, edge_width)
    right_edge = image[:, :, :, -edge_width:]         # (B, C, H, edge_width)
    top_edge = image[:, :, :edge_width, :]            # (B, C, edge_width, W)
    bottom_edge = image[:, :, -edge_width:, :]        # (B, C, edge_width, W)
    
    if squeeze_output:
        left_edge = left_edge.squeeze(0)
        right_edge = right_edge.squeeze(0)
        top_edge = top_edge.squeeze(0)
        bottom_edge = bottom_edge.squeeze(0)
    
    if return_dict:
        return {
            'left': left_edge,      # 对应 Cam3
            'right': right_edge,    # 对应 Cam5
            'top': top_edge,        # 对应 Cam4
            'bottom': bottom_edge   # 对应 Cam6
        }
    else:
        return left_edge, right_edge, top_edge, bottom_edge


def get_edge_mapping() -> Dict[str, str]:
    """
    获取正面边缘到侧面视角的映射关系
    
    Returns:
        边缘名称到相机视角的映射
    """
    return {
        'left': 'Cam3',    # 正面左边缘 -> 左侧面
        'right': 'Cam5',   # 正面右边缘 -> 右侧面
        'top': 'Cam4',     # 正面上边缘 -> 上侧面
        'bottom': 'Cam6'   # 正面下边缘 -> 下侧面
    }


def compute_edge_similarity(
    edge_feature: torch.Tensor,
    side_feature: torch.Tensor,
    metric: Literal['mse', 'cosine', 'l1'] = 'mse'
) -> torch.Tensor:
    """
    计算边缘特征与侧面特征的相似度
    
    Args:
        edge_feature: 边缘特征 (B, C, H, W) 或 (C, H, W)
        side_feature: 侧面特征 (B, C, H, W) 或 (C, H, W)
        metric: 相似度度量方式
        
    Returns:
        相似度分数（越低越相似，用于损失函数）
    """
    # 确保维度匹配
    if edge_feature.dim() == 3:
        edge_feature = edge_feature.unsqueeze(0)
    if side_feature.dim() == 3:
        side_feature = side_feature.unsqueeze(0)
    
    # 展平特征
    edge_flat = edge_feature.flatten(start_dim=1)
    side_flat = side_feature.flatten(start_dim=1)
    
    if metric == 'mse':
        return F.mse_loss(edge_flat, side_flat)
    elif metric == 'cosine':
        # 1 - cosine_similarity (转换为损失)
        cos_sim = F.cosine_similarity(edge_flat, side_flat, dim=1)
        return 1 - cos_sim.mean()
    elif metric == 'l1':
        return F.l1_loss(edge_flat, side_flat)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def detect_background_type(
    image: Union[np.ndarray, torch.Tensor]
) -> Literal['bright_field', 'dark_field']:
    """
    检测图像的光照类型（明场/暗场）
    
    Args:
        image: 输入图像
        
    Returns:
        'bright_field' 或 'dark_field'
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        # (C, H, W) -> (H, W, C)
        image = image.permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    # 计算图像边缘区域的平均亮度
    h, w = image.shape[:2]
    border_size = min(h, w) // 10
    
    # 取四个角落的平均亮度
    corners = [
        image[:border_size, :border_size],
        image[:border_size, -border_size:],
        image[-border_size:, :border_size],
        image[-border_size:, -border_size:]
    ]
    
    avg_brightness = np.mean([c.mean() for c in corners])
    
    # 阈值判断
    if avg_brightness < 50:
        return 'dark_field'
    else:
        return 'bright_field'

