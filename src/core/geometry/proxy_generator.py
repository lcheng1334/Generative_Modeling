"""
工字型电感的3D Proxy模型生成
H-shaped Inductor Proxy Model Generator
"""
import torch
import numpy as np
from typing import Tuple, Dict
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV


class InductorProxyGenerator:
    """生成工字型电感的3D代理模型"""
    
    def __init__(
        self,
        top_width: float = 1.0,
        bottom_width: float = 0.8,
        middle_height: float = 0.6,
        silver_ratio: float = 0.3,
        device: str = "cuda"
    ):
        """
        Args:
            top_width: 上横宽度（归一化）
            bottom_width: 下横宽度
            middle_height: 中间竖条高度
            silver_ratio: 镀银区域占比
            device: 计算设备
        """
        self.top_width = top_width
        self.bottom_width = bottom_width
        self.middle_height = middle_height
        self.silver_ratio = silver_ratio
        self.device = device
        
    def generate_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成工字型mesh的顶点和面
        
        Returns:
            vertices: (V, 3) 顶点坐标
            faces: (F, 3) 面索引
        """
        # 这里实现简化版本，实际可以更精细
        # 工字型可以拆解为3个长方体：上横、中间竖条、下横
        
        vertices_list = []
        faces_list = []
        vertex_offset = 0
        
        # 上横的顶点（8个）
        top_bar = self._create_box(
            width=self.top_width,
            height=0.2,
            depth=0.4,
            center=[0, self.middle_height/2 + 0.1, 0]
        )
        vertices_list.append(top_bar['vertices'])
        faces_list.append(top_bar['faces'] + vertex_offset)
        vertex_offset += len(top_bar['vertices'])
        
        # 中间竖条（8个顶点）
        middle_bar = self._create_box(
            width=0.3,
            height=self.middle_height,
            depth=0.4,
            center=[0, 0, 0]
        )
        vertices_list.append(middle_bar['vertices'])
        faces_list.append(middle_bar['faces'] + vertex_offset)
        vertex_offset += len(middle_bar['vertices'])
        
        # 下横（8个顶点）
        bottom_bar = self._create_box(
            width=self.bottom_width,
            height=0.2,
            depth=0.4,
            center=[0, -self.middle_height/2 - 0.1, 0]
        )
        vertices_list.append(bottom_bar['vertices'])
        faces_list.append(bottom_bar['faces'] + vertex_offset)
        
        vertices = np.concatenate(vertices_list, axis=0)
        faces = np.concatenate(faces_list, axis=0)
        
        # 转换为tensor
        vertices = torch.from_numpy(vertices).float().to(self.device)
        faces = torch.from_numpy(faces).long().to(self.device)
        
        return vertices, faces
    
    def _create_box(
        self,
        width: float,
        height: float,
        depth: float,
        center: list
    ) -> Dict[str, np.ndarray]:
        """
        创建长方体的顶点和面
        
        Args:
            width: 宽度（x方向）
            height: 高度（y方向）
            depth: 深度（z方向）
            center: 中心点坐标 [x, y, z]
            
        Returns:
            包含vertices和faces的字典
        """
        cx, cy, cz = center
        w, h, d = width/2, height/2, depth/2
        
        # 8个顶点
        vertices = np.array([
            [cx-w, cy-h, cz-d],  # 0: 左下前
            [cx+w, cy-h, cz-d],  # 1: 右下前
            [cx+w, cy+h, cz-d],  # 2: 右上前
            [cx-w, cy+h, cz-d],  # 3: 左上前
            [cx-w, cy-h, cz+d],  # 4: 左下后
            [cx+w, cy-h, cz+d],  # 5: 右下后
            [cx+w, cy+h, cz+d],  # 6: 右上后
            [cx-w, cy+h, cz+d],  # 7: 左上后
        ])
        
        # 12个三角形面（6个矩形面，每个分2个三角形）
        faces = np.array([
            # 前面
            [0, 1, 2], [0, 2, 3],
            # 后面
            [4, 6, 5], [4, 7, 6],
            # 左面
            [0, 3, 7], [0, 7, 4],
            # 右面
            [1, 5, 6], [1, 6, 2],
            # 上面
            [3, 2, 6], [3, 6, 7],
            # 下面
            [0, 4, 5], [0, 5, 1],
        ])
        
        return {'vertices': vertices, 'faces': faces}
    
    def create_uv_map(self, num_vertices: int) -> torch.Tensor:
        """
        创建UV坐标映射
        
        Args:
            num_vertices: 顶点数量
            
        Returns:
            uv_coords: (V, 2) UV坐标
        """
        # 简化版本：均匀分布
        # 实际需要根据面的法线方向分配到不同的UV区域
        uv_coords = torch.rand(num_vertices, 2).to(self.device)
        return uv_coords


def create_inductor_mesh(
    config: dict,
    device: str = "cuda"
) -> Meshes:
    """
    根据配置创建电感mesh
    
    Args:
        config: 配置字典
        device: 计算设备
        
    Returns:
        PyTorch3D Meshes对象
    """
    params = config['geometry']['params']
    
    generator = InductorProxyGenerator(
        top_width=params['top_width'],
        bottom_width=params['bottom_width'],
        middle_height=params['middle_height'],
        silver_ratio=params['silver_ratio'],
        device=device
    )
    
    vertices, faces = generator.generate_mesh()
    
    # 创建Meshes对象
    meshes = Meshes(
        verts=[vertices],
        faces=[faces]
    )
    
    return meshes


if __name__ == "__main__":
    # 测试代码
    generator = InductorProxyGenerator(device="cpu")
    vertices, faces = generator.generate_mesh()
    print(f"生成的顶点数: {len(vertices)}")
    print(f"生成的面数: {len(faces)}")
