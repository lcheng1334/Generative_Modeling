"""
Defect Injection Module for Multi-View Defect Synthesis

This module generates synthetic defects on OK images by:
1. Generating defect masks at specified locations
2. Sampling defect textures from NG samples or procedural generation
3. Blending defects onto OK images with cross-view consistency

Author: Multi-View Defect Synthesis Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
import random


class DefectMaskGenerator:
    """Generate defect masks with various shapes and patterns"""
    
    # Defect types and their typical characteristics
    DEFECT_CONFIGS = {
        'breakage': {'shape': 'irregular', 'edge_preference': True, 'size_range': (0.05, 0.15)},
        'adhesion': {'shape': 'blob', 'edge_preference': False, 'size_range': (0.1, 0.3)},
        'reversed_print': {'shape': 'region', 'edge_preference': False, 'size_range': (0.2, 0.4)},
        'silver_overflow': {'shape': 'edge_overflow', 'edge_preference': True, 'size_range': (0.03, 0.1)},
        'exposed_substrate': {'shape': 'patch', 'edge_preference': True, 'size_range': (0.02, 0.08)},
        'diffusion': {'shape': 'gradient', 'edge_preference': False, 'size_range': (0.05, 0.15)},
        'contamination': {'shape': 'spots', 'edge_preference': False, 'size_range': (0.01, 0.05)},
    }
    
    def __init__(self, image_size: int = 256):
        self.image_size = image_size
    
    def generate_mask(self, 
                      defect_type: str,
                      location: Optional[Tuple[int, int]] = None,
                      size_factor: float = 1.0) -> np.ndarray:
        """
        Generate a defect mask for the specified defect type.
        
        Args:
            defect_type: Type of defect from DEFECT_CONFIGS
            location: (x, y) center location, random if None
            size_factor: Multiplier for defect size
            
        Returns:
            Binary mask of shape (H, W) with values in [0, 1]
        """
        config = self.DEFECT_CONFIGS.get(defect_type, self.DEFECT_CONFIGS['contamination'])
        shape_type = config['shape']
        min_size, max_size = config['size_range']
        
        # Random size within range
        size = random.uniform(min_size, max_size) * size_factor
        radius = int(self.image_size * size / 2)
        
        # Random location if not specified
        if location is None:
            margin = radius + 10
            x = random.randint(margin, self.image_size - margin)
            y = random.randint(margin, self.image_size - margin)
        else:
            x, y = location
        
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        if shape_type == 'irregular':
            mask = self._generate_irregular_mask(x, y, radius)
        elif shape_type == 'blob':
            mask = self._generate_blob_mask(x, y, radius)
        elif shape_type == 'region':
            mask = self._generate_region_mask(x, y, radius)
        elif shape_type == 'edge_overflow':
            mask = self._generate_edge_overflow_mask(x, y, radius)
        elif shape_type == 'patch':
            mask = self._generate_patch_mask(x, y, radius)
        elif shape_type == 'gradient':
            mask = self._generate_gradient_mask(x, y, radius)
        elif shape_type == 'spots':
            mask = self._generate_spots_mask(radius)
        else:
            mask = self._generate_blob_mask(x, y, radius)
        
        return mask
    
    def _generate_irregular_mask(self, x: int, y: int, radius: int) -> np.ndarray:
        """Generate irregular shaped mask (for breakage)"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Create irregular polygon
        n_points = random.randint(5, 10)
        angles = np.sort(np.random.uniform(0, 2*np.pi, n_points))
        radii = np.random.uniform(0.5, 1.0, n_points) * radius
        
        points = []
        for angle, r in zip(angles, radii):
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            points.append([px, py])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1.0)
        
        # Add noise to edges
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def _generate_blob_mask(self, x: int, y: int, radius: int) -> np.ndarray:
        """Generate blob-shaped mask (for adhesion)"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Multiple overlapping circles
        n_circles = random.randint(3, 6)
        for _ in range(n_circles):
            cx = x + random.randint(-radius//2, radius//2)
            cy = y + random.randint(-radius//2, radius//2)
            cr = random.randint(radius//3, radius)
            cv2.circle(mask, (cx, cy), cr, 1.0, -1)
        
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask = (mask > 0.3).astype(np.float32)
        
        return mask
    
    def _generate_region_mask(self, x: int, y: int, radius: int) -> np.ndarray:
        """Generate rectangular region mask (for reversed_print)"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Rectangular region with some noise
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(self.image_size, x + radius)
        y2 = min(self.image_size, y + radius)
        
        mask[y1:y2, x1:x2] = 1.0
        
        # Add some edge irregularity
        noise = np.random.uniform(0, 1, mask.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (15, 15), 0)
        mask = mask * (noise > 0.4).astype(np.float32)
        
        return mask
    
    def _generate_edge_overflow_mask(self, x: int, y: int, radius: int) -> np.ndarray:
        """Generate edge overflow mask (for silver_overflow)"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Thin irregular line along edge
        edge_type = random.choice(['top', 'bottom', 'left', 'right'])
        
        if edge_type == 'top':
            y_pos = random.randint(20, 60)
            x_start = random.randint(50, 100)
            x_end = random.randint(150, 200)
            for xi in range(x_start, x_end):
                yi = y_pos + random.randint(-5, 5)
                cv2.circle(mask, (xi, yi), radius//2, 1.0, -1)
        elif edge_type == 'bottom':
            y_pos = random.randint(self.image_size - 60, self.image_size - 20)
            x_start = random.randint(50, 100)
            x_end = random.randint(150, 200)
            for xi in range(x_start, x_end):
                yi = y_pos + random.randint(-5, 5)
                cv2.circle(mask, (xi, yi), radius//2, 1.0, -1)
        elif edge_type == 'left':
            x_pos = random.randint(20, 60)
            y_start = random.randint(50, 100)
            y_end = random.randint(150, 200)
            for yi in range(y_start, y_end):
                xi = x_pos + random.randint(-5, 5)
                cv2.circle(mask, (xi, yi), radius//2, 1.0, -1)
        else:  # right
            x_pos = random.randint(self.image_size - 60, self.image_size - 20)
            y_start = random.randint(50, 100)
            y_end = random.randint(150, 200)
            for yi in range(y_start, y_end):
                xi = x_pos + random.randint(-5, 5)
                cv2.circle(mask, (xi, yi), radius//2, 1.0, -1)
        
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask
    
    def _generate_patch_mask(self, x: int, y: int, radius: int) -> np.ndarray:
        """Generate patch mask (for exposed_substrate)"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Small elliptical patches
        n_patches = random.randint(1, 3)
        for _ in range(n_patches):
            px = x + random.randint(-radius, radius)
            py = y + random.randint(-radius, radius)
            axes = (random.randint(radius//2, radius), random.randint(radius//3, radius//2))
            angle = random.randint(0, 180)
            cv2.ellipse(mask, (px, py), axes, angle, 0, 360, 1.0, -1)
        
        return mask
    
    def _generate_gradient_mask(self, x: int, y: int, radius: int) -> np.ndarray:
        """Generate gradient mask (for diffusion)"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Create distance-based gradient
        Y, X = np.ogrid[:self.image_size, :self.image_size]
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask = np.clip(1 - dist / (radius * 2), 0, 1).astype(np.float32)
        
        # Add noise
        noise = np.random.uniform(0.8, 1.2, mask.shape).astype(np.float32)
        mask = mask * noise
        mask = np.clip(mask, 0, 1)
        
        return mask
    
    def _generate_spots_mask(self, base_radius: int) -> np.ndarray:
        """Generate multiple spots mask (for contamination)"""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        n_spots = random.randint(3, 10)
        for _ in range(n_spots):
            x = random.randint(30, self.image_size - 30)
            y = random.randint(30, self.image_size - 30)
            radius = random.randint(2, base_radius)
            cv2.circle(mask, (x, y), radius, 1.0, -1)
        
        return mask


class DefectTextureGenerator:
    """Generate or sample defect textures"""
    
    def __init__(self, ng_data_dir: Optional[str] = None):
        """
        Args:
            ng_data_dir: Path to NG dataset for texture sampling
        """
        self.ng_data_dir = Path(ng_data_dir) if ng_data_dir else None
        self.texture_cache = {}
        
        if self.ng_data_dir and self.ng_data_dir.exists():
            self._load_texture_samples()
    
    def _load_texture_samples(self):
        """Load sample textures from NG dataset"""
        for defect_dir in self.ng_data_dir.iterdir():
            if defect_dir.is_dir():
                defect_type = defect_dir.name
                images = list(defect_dir.glob('*.png'))[:50]  # Limit to 50 samples
                self.texture_cache[defect_type] = images
    
    def generate_texture(self, 
                         mask: np.ndarray,
                         defect_type: str,
                         base_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate defect texture for the given mask.
        
        Args:
            mask: Binary mask of shape (H, W)
            defect_type: Type of defect
            base_image: Original image for color matching
            
        Returns:
            Texture image of shape (H, W, 3)
        """
        h, w = mask.shape
        
        # Try to sample from NG dataset first
        if defect_type in self.texture_cache and len(self.texture_cache[defect_type]) > 0:
            texture = self._sample_ng_texture(defect_type, (h, w))
        else:
            texture = self._generate_procedural_texture(defect_type, (h, w), base_image)
        
        return texture
    
    def _sample_ng_texture(self, defect_type: str, size: Tuple[int, int]) -> np.ndarray:
        """Sample texture from NG dataset"""
        img_path = random.choice(self.texture_cache[defect_type])
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return img.astype(np.float32) / 255.0
    
    def _generate_procedural_texture(self, 
                                      defect_type: str, 
                                      size: Tuple[int, int],
                                      base_image: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate procedural texture when NG samples unavailable"""
        h, w = size
        
        if defect_type == 'breakage':
            # Dark irregular texture
            texture = np.random.uniform(0.1, 0.3, (h, w, 3)).astype(np.float32)
            
        elif defect_type == 'silver_overflow':
            # Silver-like bright texture
            base = np.random.uniform(0.7, 0.9, (h, w)).astype(np.float32)
            texture = np.stack([base * 0.8, base * 0.75, base * 0.85], axis=-1)
            
        elif defect_type == 'exposed_substrate':
            # Ceramic-like neutral texture
            base = np.random.uniform(0.4, 0.6, (h, w)).astype(np.float32)
            texture = np.stack([base, base * 0.95, base * 0.9], axis=-1)
            
        elif defect_type == 'contamination':
            # Dark spots
            texture = np.random.uniform(0.05, 0.2, (h, w, 3)).astype(np.float32)
            
        else:
            # Default: slightly darker than base
            if base_image is not None:
                texture = base_image * 0.7
            else:
                texture = np.random.uniform(0.3, 0.5, (h, w, 3)).astype(np.float32)
        
        return texture


class DefectInjector(nn.Module):
    """
    Main defect injection module.
    
    Combines mask generation and texture generation to create
    synthetic defects on OK images.
    """
    
    def __init__(self, 
                 image_size: int = 256,
                 ng_data_dir: Optional[str] = None,
                 blend_mode: str = 'alpha'):
        """
        Args:
            image_size: Size of input/output images
            ng_data_dir: Path to NG dataset
            blend_mode: 'alpha', 'overlay', or 'replace'
        """
        super().__init__()
        
        self.image_size = image_size
        self.blend_mode = blend_mode
        
        self.mask_generator = DefectMaskGenerator(image_size)
        self.texture_generator = DefectTextureGenerator(ng_data_dir)
    
    def forward(self,
                image: Union[torch.Tensor, np.ndarray],
                defect_type: str,
                location: Optional[Tuple[int, int]] = None,
                size_factor: float = 1.0) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Inject defect into image.
        
        Args:
            image: Input OK image, shape (C, H, W) or (H, W, C)
            defect_type: Type of defect to inject
            location: Optional (x, y) location
            size_factor: Size multiplier
            
        Returns:
            Dictionary with:
                - 'image': Synthesized image with defect
                - 'mask': Binary defect mask
                - 'defect_type': Type of injected defect
        """
        # Convert to numpy if tensor
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image_np = image.cpu().numpy()
            if image_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                image_np = np.transpose(image_np, (1, 2, 0))
        else:
            image_np = image.copy()
        
        # Normalize to [0, 1] if needed
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # Generate mask
        mask = self.mask_generator.generate_mask(defect_type, location, size_factor)
        
        # Generate texture
        texture = self.texture_generator.generate_texture(mask, defect_type, image_np)
        
        # Blend
        synthesized = self._blend(image_np, texture, mask)
        
        # Convert back to tensor if needed
        if is_tensor:
            synthesized = np.transpose(synthesized, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            synthesized = torch.from_numpy(synthesized).float()
            mask = torch.from_numpy(mask).float()
        
        return {
            'image': synthesized,
            'mask': mask,
            'defect_type': defect_type
        }
    
    def _blend(self, 
               base: np.ndarray, 
               texture: np.ndarray, 
               mask: np.ndarray) -> np.ndarray:
        """Blend texture onto base image using mask"""
        mask_3d = np.expand_dims(mask, axis=-1)
        
        if self.blend_mode == 'alpha':
            # Simple alpha blending
            result = base * (1 - mask_3d) + texture * mask_3d
            
        elif self.blend_mode == 'overlay':
            # Overlay blending for more natural look
            result = base.copy()
            overlay = 2 * base * texture
            overlay = np.where(base < 0.5, overlay, 1 - 2 * (1 - base) * (1 - texture))
            result = base * (1 - mask_3d) + overlay * mask_3d
            
        elif self.blend_mode == 'replace':
            # Direct replacement
            result = base * (1 - mask_3d) + texture * mask_3d
            
        else:
            result = base * (1 - mask_3d) + texture * mask_3d
        
        return np.clip(result, 0, 1).astype(np.float32)
    
    def inject_batch(self,
                     images: torch.Tensor,
                     defect_types: List[str],
                     locations: Optional[List[Tuple[int, int]]] = None) -> Dict[str, torch.Tensor]:
        """
        Inject defects into a batch of images.
        
        Args:
            images: Batch of images, shape (B, C, H, W)
            defect_types: List of defect types for each image
            locations: Optional list of locations
            
        Returns:
            Dictionary with batched outputs
        """
        batch_size = images.shape[0]
        
        if locations is None:
            locations = [None] * batch_size
        
        results = []
        masks = []
        
        for i in range(batch_size):
            result = self(images[i], defect_types[i], locations[i])
            results.append(result['image'])
            masks.append(result['mask'])
        
        return {
            'images': torch.stack(results),
            'masks': torch.stack(masks),
            'defect_types': defect_types
        }


def demo():
    """Demonstrate defect injection"""
    import matplotlib.pyplot as plt
    
    # Create a simple test image
    test_image = np.random.uniform(0.3, 0.7, (256, 256, 3)).astype(np.float32)
    
    # Create injector
    injector = DefectInjector(image_size=256)
    
    # Test different defect types
    defect_types = ['breakage', 'silver_overflow', 'exposed_substrate', 'contamination']
    
    fig, axes = plt.subplots(2, len(defect_types), figsize=(16, 8))
    
    for i, defect_type in enumerate(defect_types):
        result = injector(test_image, defect_type)
        
        axes[0, i].imshow(result['image'])
        axes[0, i].set_title(f'{defect_type}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(result['mask'], cmap='gray')
        axes[1, i].set_title('Mask')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('defect_injection_demo.png')
    print("Demo saved to defect_injection_demo.png")


if __name__ == '__main__':
    demo()
