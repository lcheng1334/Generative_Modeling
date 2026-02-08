"""
VAE Reconstruction Evaluation Script

Evaluates the quality of VAE reconstruction on test images.
Computes: PSNR, SSIM, LPIPS, and MSE metrics.

Usage:
    python scripts/evaluate_vae.py \
        --vae_dir experiments/baseline_vae \
        --data_dir data/datasets/OK \
        --output_dir experiments/evaluation
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae import VanillaVAE
from src.datasets.multiview_dataset import MultiViewDataset
from torch.utils.data import DataLoader

# Try to import metrics, handle gracefully if not available
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: skimage not installed, PSNR/SSIM metrics unavailable")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed, LPIPS metric unavailable")


class VAEEvaluator:
    """Evaluate VAE reconstruction quality"""
    
    def __init__(self, 
                 vae_checkpoint_dir: str,
                 image_size: int = 256,
                 device: str = 'cuda'):
        self.image_size = image_size
        self.device = device
        
        # Load VAE models
        self.vaes = self._load_vaes(vae_checkpoint_dir)
        
        # LPIPS model
        if HAS_LPIPS:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
        else:
            self.lpips_model = None
    
    def _load_vaes(self, checkpoint_dir: str) -> Dict[str, VAE]:
        """Load VAE models for each camera"""
        vaes = {}
        checkpoint_path = Path(checkpoint_dir)
        
        for cam_idx in range(1, 7):
            cam_name = f'Cam{cam_idx}'
            ckpt_path = checkpoint_path / cam_name / 'best_vae.pth'
            
            if ckpt_path.exists():
                vae = VanillaVAE(
                    in_channels=3,
                    latent_dim=256,
                    hidden_dims=[32, 64, 128, 256],
                    image_size=self.image_size
                )
                state_dict = torch.load(ckpt_path, map_location=self.device)
                vae.load_state_dict(state_dict)
                vae.to(self.device)
                vae.eval()
                vaes[cam_name] = vae
                print(f"  Loaded VAE for {cam_name}")
            else:
                print(f"  Warning: No checkpoint for {cam_name}")
        
        return vaes
    
    def compute_mse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Mean Squared Error"""
        return F.mse_loss(pred, target).item()
    
    def compute_psnr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio"""
        if HAS_SKIMAGE:
            return psnr(target, pred, data_range=1.0)
        return 0.0
    
    def compute_ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Structural Similarity Index"""
        if HAS_SKIMAGE:
            return ssim(target, pred, data_range=1.0, channel_axis=2)
        return 0.0
    
    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute LPIPS perceptual distance"""
        if self.lpips_model is not None:
            with torch.no_grad():
                # LPIPS expects [-1, 1] range
                pred_scaled = pred * 2 - 1
                target_scaled = target * 2 - 1
                distance = self.lpips_model(pred_scaled, target_scaled)
                return distance.mean().item()
        return 0.0
    
    def evaluate_single(self, 
                        image: torch.Tensor, 
                        cam_name: str) -> Dict[str, float]:
        """Evaluate single image reconstruction"""
        if cam_name not in self.vaes:
            return {}
        
        vae = self.vaes[cam_name]
        image = image.to(self.device)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            recon, _, _ = vae(image)
        
        # Compute metrics
        metrics = {
            'mse': self.compute_mse(recon, image)
        }
        
        # Convert to numpy for PSNR/SSIM
        pred_np = recon[0].cpu().numpy().transpose(1, 2, 0)
        target_np = image[0].cpu().numpy().transpose(1, 2, 0)
        
        metrics['psnr'] = self.compute_psnr(pred_np, target_np)
        metrics['ssim'] = self.compute_ssim(pred_np, target_np)
        metrics['lpips'] = self.compute_lpips(recon, image)
        
        return metrics
    
    def evaluate_dataset(self,
                         data_dir: str,
                         num_samples: int = 100) -> Dict[str, Dict[str, float]]:
        """Evaluate on dataset"""
        results = {cam: {'mse': [], 'psnr': [], 'ssim': [], 'lpips': []} 
                   for cam in self.vaes.keys()}
        
        # Load dataset for each camera
        for cam_name in self.vaes.keys():
            cam_dir = Path(data_dir) / 'Group1' / cam_name
            if not cam_dir.exists():
                cam_dir = Path(data_dir) / cam_name
            
            if not cam_dir.exists():
                print(f"  Warning: Directory not found for {cam_name}")
                continue
            
            # Get images
            images = list(cam_dir.glob('*.png'))[:num_samples]
            
            print(f"\nEvaluating {cam_name} ({len(images)} images)...")
            
            for img_path in tqdm(images, desc=f"{cam_name}"):
                try:
                    # Load and preprocess
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((self.image_size, self.image_size))
                    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)
                    
                    # Evaluate
                    metrics = self.evaluate_single(img_tensor, cam_name)
                    
                    for key, value in metrics.items():
                        results[cam_name][key].append(value)
                        
                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
        
        # Compute averages
        avg_results = {}
        for cam_name, metrics in results.items():
            avg_results[cam_name] = {}
            for metric_name, values in metrics.items():
                if values:
                    avg_results[cam_name][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
        
        return avg_results
    
    def save_reconstruction_samples(self,
                                     data_dir: str,
                                     output_dir: str,
                                     num_samples: int = 10):
        """Save sample reconstruction comparisons"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for cam_name in self.vaes.keys():
            cam_dir = Path(data_dir) / 'Group1' / cam_name
            if not cam_dir.exists():
                cam_dir = Path(data_dir) / cam_name
            
            if not cam_dir.exists():
                continue
            
            images = sorted(list(cam_dir.glob('*.png')))[:num_samples]
            
            for i, img_path in enumerate(images):
                try:
                    # Load
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((self.image_size, self.image_size))
                    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    # Reconstruct
                    vae = self.vaes[cam_name]
                    with torch.no_grad():
                        recon, _, _ = vae(img_tensor)
                    
                    # Save comparison
                    orig = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
                    pred = recon[0].cpu().numpy().transpose(1, 2, 0)
                    
                    # Side by side
                    comparison = np.concatenate([orig, pred], axis=1)
                    comparison = (comparison * 255).astype(np.uint8)
                    
                    save_path = output_path / f'{cam_name}_sample_{i:03d}.png'
                    Image.fromarray(comparison).save(save_path)
                    
                except Exception as e:
                    print(f"Error: {e}")
        
        print(f"\nSaved reconstruction samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAE Reconstruction")
    
    parser.add_argument('--vae_dir', type=str, required=True,
                        help='Path to VAE checkpoint directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to OK dataset directory')
    parser.add_argument('--output_dir', type=str, default='experiments/evaluation',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--save_samples', action='store_true',
                        help='Save reconstruction samples')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VAE Reconstruction Evaluation")
    print("=" * 60)
    print(f"VAE directory: {args.vae_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Number of samples: {args.num_samples}")
    print()
    
    # Create evaluator
    evaluator = VAEEvaluator(
        vae_checkpoint_dir=args.vae_dir,
        device=args.device
    )
    
    # Evaluate
    results = evaluator.evaluate_dataset(
        data_dir=args.data_dir,
        num_samples=args.num_samples
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    for cam_name, metrics in results.items():
        print(f"\n{cam_name}:")
        for metric_name, values in metrics.items():
            print(f"  {metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f'vae_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Save samples
    if args.save_samples:
        evaluator.save_reconstruction_samples(
            data_dir=args.data_dir,
            output_dir=str(output_path / 'samples'),
            num_samples=10
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
