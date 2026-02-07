"""
Blinn-Phong重光照渲染模块
Blinn-Phong Relighting Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from typing import Tuple, Dict


class BlinnPhongRenderer(nn.Module):
    """基于Blinn-Phong模型的可微渲染器"""
    
    def __init__(
        self,
        light_position: Tuple[float, float, float] = (0, 10, -5),
        light_intensity: float = 1.0,
        ambient_intensity: float = 0.2
    ):
        """
        Args:
            light_position: 光源位置 (x, y, z)
            light_intensity: 光源强度
            ambient_intensity: 环境光强度
        """
        super().__init__()
        
        self.register_buffer(
            'light_position',
            torch.tensor(light_position, dtype=torch.float32)
        )
        self.light_intensity = light_intensity
        self.ambient_intensity = ambient_intensity
    
    def compute_normals_from_height(
        self,
        height_map: torch.Tensor,
        height_scale: float = 0.1
    ) -> torch.Tensor:
        """
        从高度图计算法线
        
        Args:
            height_map: (B, 1, H, W) 高度图
            height_scale: 高度缩放系数
            
        Returns:
            normals: (B, 3, H, W) 法线图（已归一化）
        """
        # 计算梯度
        grad = kornia.filters.spatial_gradient(height_map, mode='sobel')
        grad_x = grad[:, :, 0, :, :] * height_scale  # (B, 1, H, W)
        grad_y = grad[:, :, 1, :, :] * height_scale
        
        # 构建法线向量 N = [-∂h/∂x, -∂h/∂y, 1]
        normals = torch.cat([
            -grad_x,
            -grad_y,
            torch.ones_like(grad_x)
        ], dim=1)  # (B, 3, H, W)
        
        # 归一化
        normals = F.normalize(normals, dim=1)
        
        return normals
    
    def forward(
        self,
        albedo: torch.Tensor,
        normals: torch.Tensor,
        material_params: Dict[str, float],
        view_dir: Tuple[float, float, float] = (0, 0, -1)
    ) -> torch.Tensor:
        """
        执行Blinn-Phong渲染
        
        Args:
            albedo: (B, 3, H, W) RGB反照率
            normals: (B, 3, H, W) 法线图
            material_params: 材质参数字典 {'ka', 'kd', 'ks', 'shininess'}
            view_dir: 视线方向向量
            
        Returns:
            rendered: (B, 3, H, W) 渲染后的图像
        """
        B, C, H, W = albedo.shape
        device = albedo.device
        
        # 提取材质参数
        ka = material_params.get('ka', 0.2)
        kd = material_params.get('kd', 0.6)
        ks = material_params.get('ks', 0.3)
        shininess = material_params.get('shininess', 32.0)
        
        # 计算光源方向（从表面点到光源）
        # 简化：假设平行光
        light_dir = F.normalize(
            self.light_position.view(1, 3, 1, 1).expand(B, 3, H, W),
            dim=1
        )
        
        # 视线方向
        view_dir = torch.tensor(view_dir, device=device, dtype=torch.float32)
        view_dir = F.normalize(
            view_dir.view(1, 3, 1, 1).expand(B, 3, H, W),
            dim=1
        )
        
        # 1. 环境光分量
        ambient = ka * self.ambient_intensity * albedo
        
        # 2. 漫反射分量 (Lambertian)
        # I_diffuse = kd * max(N·L, 0) * albedo
        NdotL = torch.sum(normals * light_dir, dim=1, keepdim=True)
        NdotL = torch.clamp(NdotL, min=0.0)
        diffuse = kd * self.light_intensity * NdotL * albedo
        
        # 3. 镜面反射分量 (Blinn-Phong)
        # I_specular = ks * max(N·H, 0)^shininess
        # 其中 H = normalize(L + V) 是半角向量
        half_dir = F.normalize(light_dir + view_dir, dim=1)
        NdotH = torch.sum(normals * half_dir, dim=1, keepdim=True)
        NdotH = torch.clamp(NdotH, min=0.0)
        specular = ks * self.light_intensity * torch.pow(NdotH, shininess)
        specular = specular.expand_as(albedo)  # 扩展到RGB
        
        # 组合所有分量
        rendered = ambient + diffuse + specular
        
        # 裁剪到 [0, 1]
        rendered = torch.clamp(rendered, 0.0, 1.0)
        
        return rendered


def relight_with_height(
    albedo: torch.Tensor,
    height_map: torch.Tensor,
    material_params: Dict[str, float],
    light_position: Tuple[float, float, float] = (0, 10, -5),
    height_scale: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    便捷函数：从高度图重新渲染
    
    Args:
        albedo: (B, 3, H, W) RGB纹理
        height_map: (B, 1, H, W) 高度图
        material_params: 材质参数
        light_position: 光源位置
        height_scale: 高度缩放
        
    Returns:
        rendered: 渲染后图像
        normals: 计算的法线图
    """
    renderer = BlinnPhongRenderer(light_position=light_position)
    renderer = renderer.to(albedo.device)
    
    # 计算法线
    normals = renderer.compute_normals_from_height(height_map, height_scale)
    
    # 渲染
    rendered = renderer(albedo, normals, material_params)
    
    return rendered, normals


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建测试数据
    albedo = torch.rand(1, 3, 256, 256).to(device)
    height_map = torch.randn(1, 1, 256, 256).to(device) * 0.1
    
    material_params = {
        'ka': 0.2,
        'kd': 0.6,
        'ks': 0.8,
        'shininess': 64.0
    }
    
    # 渲染
    rendered, normals = relight_with_height(
        albedo, height_map, material_params
    )
    
    print(f"渲染结果形状: {rendered.shape}")
    print(f"法线图形状: {normals.shape}")
