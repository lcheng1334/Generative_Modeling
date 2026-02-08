"""
Cross-View Consistency Loss for Multi-View Defect Synthesis

This module implements the cross-view geometric consistency constraints:
- Edge correspondence between front view (Cam1) and side views (Cam3-6)
- Ensures defects injected at edges appear consistently across views

Key insight:
- Cam1 (front) left edge corresponds to Cam3 (left side) upper edge
- Cam1 (front) right edge corresponds to Cam5 (right side) upper edge
- Cam1 (front) top edge corresponds to Cam4 (top side) edge
- Cam1 (front) bottom edge corresponds to Cam6 (bottom side) edge

Author: Multi-View Defect Synthesis Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class EdgeFeatureExtractor(nn.Module):
    """
    Extract features from edge regions of images.
    
    For the front view (Cam1), we extract features from the 4 edges.
    For side views (Cam3-6), we extract features from the corresponding edges.
    """
    
    def __init__(self, 
                 edge_width: int = 32,
                 feature_dim: int = 128,
                 pretrained_backbone: bool = False):
        """
        Args:
            edge_width: Width of edge region to extract
            feature_dim: Output feature dimension
            pretrained_backbone: Whether to use pretrained ResNet
        """
        super().__init__()
        
        self.edge_width = edge_width
        self.feature_dim = feature_dim
        
        # Simple CNN for edge feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim)
        )
    
    def extract_edge(self, 
                     image: torch.Tensor, 
                     edge: str) -> torch.Tensor:
        """
        Extract edge region from image.
        
        Args:
            image: Image tensor of shape (B, C, H, W)
            edge: 'left', 'right', 'top', 'bottom'
            
        Returns:
            Edge region tensor of shape (B, C, edge_width, H or W)
        """
        B, C, H, W = image.shape
        
        if edge == 'left':
            return image[:, :, :, :self.edge_width]
        elif edge == 'right':
            return image[:, :, :, -self.edge_width:]
        elif edge == 'top':
            return image[:, :, :self.edge_width, :]
        elif edge == 'bottom':
            return image[:, :, -self.edge_width:, :]
        else:
            raise ValueError(f"Unknown edge: {edge}")
    
    def forward(self, 
                image: torch.Tensor, 
                edge: str) -> torch.Tensor:
        """
        Extract features from edge region.
        
        Args:
            image: Input image (B, C, H, W)
            edge: Edge to extract ('left', 'right', 'top', 'bottom')
            
        Returns:
            Feature vector (B, feature_dim)
        """
        edge_region = self.extract_edge(image, edge)
        features = self.encoder(edge_region)
        return features


class CrossViewConsistencyLoss(nn.Module):
    """
    Compute cross-view consistency loss between corresponding edges.
    
    This enforces that defects at edges appear consistently across views:
    - If a defect appears on Cam1's left edge, it should also appear on Cam3
    - Similarly for other edge-view pairs
    """
    
    # Edge-to-view correspondence mapping
    # Format: {cam1_edge: (corresponding_cam, corresponding_edge)}
    EDGE_CORRESPONDENCE = {
        'left': ('cam3', 'top'),      # Cam1 left edge -> Cam3 top edge
        'right': ('cam5', 'top'),     # Cam1 right edge -> Cam5 top edge
        'top': ('cam4', 'top'),       # Cam1 top edge -> Cam4 top edge (approximate)
        'bottom': ('cam6', 'top'),    # Cam1 bottom edge -> Cam6 top edge (approximate)
    }
    
    def __init__(self,
                 edge_width: int = 32,
                 feature_dim: int = 128,
                 loss_type: str = 'l2'):
        """
        Args:
            edge_width: Width of edge region
            feature_dim: Feature dimension
            loss_type: 'l2', 'l1', or 'cosine'
        """
        super().__init__()
        
        self.edge_extractor = EdgeFeatureExtractor(edge_width, feature_dim)
        self.loss_type = loss_type
    
    def forward(self,
                images: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute cross-view consistency loss.
        
        Args:
            images: Dictionary mapping camera name to image tensor
                    {'cam1': (B,C,H,W), 'cam3': (B,C,H,W), ...}
            masks: Optional defect masks for weighted loss
            
        Returns:
            Dictionary with loss values:
                - 'total': Total consistency loss
                - 'left_cam3': Loss for left-cam3 pair
                - etc.
        """
        losses = {}
        total_loss = 0.0
        
        cam1 = images.get('cam1')
        if cam1 is None:
            return {'total': torch.tensor(0.0)}
        
        for cam1_edge, (target_cam, target_edge) in self.EDGE_CORRESPONDENCE.items():
            target_image = images.get(target_cam)
            
            if target_image is None:
                continue
            
            # Extract features from corresponding edges
            cam1_features = self.edge_extractor(cam1, cam1_edge)
            target_features = self.edge_extractor(target_image, target_edge)
            
            # Compute loss
            if self.loss_type == 'l2':
                loss = F.mse_loss(cam1_features, target_features)
            elif self.loss_type == 'l1':
                loss = F.l1_loss(cam1_features, target_features)
            elif self.loss_type == 'cosine':
                cos_sim = F.cosine_similarity(cam1_features, target_features)
                loss = 1 - cos_sim.mean()
            else:
                loss = F.mse_loss(cam1_features, target_features)
            
            # Apply mask weighting if provided
            if masks is not None:
                cam1_mask = masks.get('cam1')
                if cam1_mask is not None:
                    # Weight loss by mask intensity at edge
                    edge_mask = self._extract_edge_mask(cam1_mask, cam1_edge)
                    weight = edge_mask.mean() + 0.1  # Minimum weight of 0.1
                    loss = loss * weight
            
            loss_name = f'{cam1_edge}_{target_cam}'
            losses[loss_name] = loss
            total_loss = total_loss + loss
        
        losses['total'] = total_loss
        return losses
    
    def _extract_edge_mask(self, mask: torch.Tensor, edge: str) -> torch.Tensor:
        """Extract edge region from mask"""
        edge_width = self.edge_extractor.edge_width
        
        if edge == 'left':
            return mask[:, :edge_width]
        elif edge == 'right':
            return mask[:, -edge_width:]
        elif edge == 'top':
            return mask[:edge_width, :]
        elif edge == 'bottom':
            return mask[-edge_width:, :]
        else:
            return mask


class MultiViewSynthesisLoss(nn.Module):
    """
    Combined loss for multi-view defect synthesis.
    
    Combines:
    1. Reconstruction loss (L1/L2 on non-defect regions)
    2. Perceptual loss (feature matching)
    3. Cross-view consistency loss
    4. Adversarial loss (optional)
    """
    
    def __init__(self,
                 lambda_recon: float = 1.0,
                 lambda_perceptual: float = 0.1,
                 lambda_consistency: float = 0.5,
                 lambda_adversarial: float = 0.01):
        """
        Args:
            lambda_recon: Weight for reconstruction loss
            lambda_perceptual: Weight for perceptual loss
            lambda_consistency: Weight for consistency loss
            lambda_adversarial: Weight for adversarial loss
        """
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_perceptual = lambda_perceptual
        self.lambda_consistency = lambda_consistency
        self.lambda_adversarial = lambda_adversarial
        
        self.consistency_loss = CrossViewConsistencyLoss()
        
        # Perceptual loss using VGG features (lazy loading)
        self._vgg = None
    
    @property
    def vgg(self):
        """Lazy load VGG for perceptual loss"""
        if self._vgg is None:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self._vgg = nn.Sequential(*list(vgg.features[:16])).eval()
            for param in self._vgg.parameters():
                param.requires_grad = False
        return self._vgg
    
    def reconstruction_loss(self,
                            pred: torch.Tensor,
                            target: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute reconstruction loss on non-defect regions.
        
        Args:
            pred: Predicted image
            target: Original OK image
            mask: Defect mask (1 = defect, 0 = normal)
        """
        if mask is not None:
            # Only compute loss on non-defect regions
            inv_mask = 1 - mask.unsqueeze(1)  # (B, 1, H, W)
            diff = (pred - target) * inv_mask
            loss = (diff ** 2).sum() / (inv_mask.sum() + 1e-6)
        else:
            loss = F.mse_loss(pred, target)
        
        return loss
    
    def perceptual_loss(self,
                        pred: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using VGG features"""
        device = pred.device
        vgg = self.vgg.to(device)
        
        pred_features = vgg(pred)
        target_features = vgg(target)
        
        return F.mse_loss(pred_features, target_features)
    
    def forward(self,
                synthesized: Dict[str, torch.Tensor],
                original: Dict[str, torch.Tensor],
                masks: Dict[str, torch.Tensor],
                discriminator_output: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            synthesized: Dict of synthesized images per view
            original: Dict of original OK images per view
            masks: Dict of defect masks per view
            discriminator_output: Optional discriminator scores
            
        Returns:
            Dict of loss components and total
        """
        losses = {}
        
        # Reconstruction loss per view
        recon_loss = 0.0
        perceptual_loss = 0.0
        
        for cam_name in synthesized.keys():
            if cam_name in original:
                pred = synthesized[cam_name]
                target = original[cam_name]
                mask = masks.get(cam_name)
                
                recon_loss += self.reconstruction_loss(pred, target, mask)
                
                if self.lambda_perceptual > 0:
                    perceptual_loss += self.perceptual_loss(pred, target)
        
        losses['reconstruction'] = recon_loss
        losses['perceptual'] = perceptual_loss
        
        # Cross-view consistency loss
        consistency_losses = self.consistency_loss(synthesized, masks)
        losses['consistency'] = consistency_losses['total']
        
        # Adversarial loss
        if discriminator_output is not None:
            # Want discriminator to output 1 (real) for our generated images
            adv_loss = F.binary_cross_entropy_with_logits(
                discriminator_output, 
                torch.ones_like(discriminator_output)
            )
            losses['adversarial'] = adv_loss
        else:
            losses['adversarial'] = torch.tensor(0.0)
        
        # Total loss
        total = (self.lambda_recon * losses['reconstruction'] +
                 self.lambda_perceptual * losses['perceptual'] +
                 self.lambda_consistency * losses['consistency'] +
                 self.lambda_adversarial * losses['adversarial'])
        
        losses['total'] = total
        
        return losses


def demo():
    """Demonstrate consistency loss computation"""
    # Create dummy images for each view
    batch_size = 2
    images = {
        'cam1': torch.randn(batch_size, 3, 256, 256),
        'cam3': torch.randn(batch_size, 3, 256, 256),
        'cam4': torch.randn(batch_size, 3, 256, 256),
        'cam5': torch.randn(batch_size, 3, 256, 256),
        'cam6': torch.randn(batch_size, 3, 256, 256),
    }
    
    # Create consistency loss
    consistency_loss = CrossViewConsistencyLoss()
    
    # Compute loss
    losses = consistency_loss(images)
    
    print("Consistency Loss Components:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")


if __name__ == '__main__':
    demo()
