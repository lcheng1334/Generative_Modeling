"""
Multi-View Defect Synthesis Training Script

This script trains the complete synthesis pipeline:
1. Load OK images from all 6 views
2. Inject synthetic defects using DefectInjector
3. Enforce cross-view consistency using ConsistencyLoss
4. Save synthesized NG samples for downstream tasks

Multi-GPU Support:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_synthesis.py --data_dir ...

Usage:
    python scripts/train_synthesis.py --config configs/synthesis.yaml
    python scripts/train_synthesis.py --data_dir data/datasets/OK --num_samples 3000
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.multiview_dataset import MultiViewDataset
from src.core.generator.defect_injector import DefectInjector
from src.models.consistency_loss import CrossViewConsistencyLoss, MultiViewSynthesisLoss
from src.models.vae import VanillaVAE


class MultiViewDefectSynthesizer:
    """
    Main class for multi-view defect synthesis.
    
    Combines:
    - Pre-trained VAE models (one per view)
    - Defect injection module
    - Cross-view consistency constraints
    """
    
    DEFECT_TYPES = ['breakage', 'silver_overflow', 'exposed_substrate', 
                    'diffusion', 'contamination', 'reversed_print', 'adhesion']
    
    # Mapping of defect types to valid views
    DEFECT_VIEW_MAPPING = {
        'breakage': ['cam1', 'cam2'],
        'silver_overflow': ['cam3', 'cam4', 'cam5', 'cam6'],
        'exposed_substrate': ['cam1'],
        'diffusion': ['cam1'],
        'contamination': ['cam1', 'cam2'],
        'reversed_print': ['cam1', 'cam2'],
        'adhesion': ['cam1', 'cam2', 'cam3'],
    }
    
    def __init__(self,
                 vae_checkpoint_dir: str,
                 ng_data_dir: Optional[str] = None,
                 image_size: int = 256,
                 device: str = 'cuda',
                 multi_gpu: bool = False):
        """
        Args:
            vae_checkpoint_dir: Directory containing trained VAE checkpoints
            ng_data_dir: Directory containing classified NG samples
            image_size: Image size
            device: Device to run on
            multi_gpu: Whether to use multiple GPUs
        """
        self.image_size = image_size
        self.device = device
        self.multi_gpu = multi_gpu
        
        # Load pre-trained VAE models
        self.vaes = self._load_vaes(vae_checkpoint_dir)
        
        # Create defect injector
        self.defect_injector = DefectInjector(
            image_size=image_size,
            ng_data_dir=ng_data_dir
        )
        
        # Create consistency loss
        self.consistency_loss = MultiViewSynthesisLoss(
            lambda_recon=1.0,
            lambda_perceptual=0.1,
            lambda_consistency=0.5
        ).to(device)
        
        # Statistics for paper
        self.stats = {
            'total_generated': 0,
            'per_defect_count': {d: 0 for d in self.DEFECT_TYPES},
            'per_view_count': {f'cam{i}': 0 for i in range(1, 7)},
            'losses': []
        }
        
        print(f"Initialized synthesizer with {len(self.vaes)} VAE models")
        if multi_gpu:
            print(f"  Multi-GPU mode enabled with {torch.cuda.device_count()} GPUs")
    
    def _load_vaes(self, checkpoint_dir: str) -> Dict[str, VanillaVAE]:
        """Load pre-trained VAE models for each view"""
        vaes = {}
        checkpoint_path = Path(checkpoint_dir)
        
        for cam_idx in range(1, 7):
            cam_name = f'cam{cam_idx}'
            cam_dir = checkpoint_path / f'Cam{cam_idx}'
            ckpt_path = cam_dir / 'best_vae.pth'
            
            if ckpt_path.exists():
                # Use default hidden_dims=[32, 64, 128, 256, 512] to match trained model
                vae = VanillaVAE(
                    in_channels=3,
                    latent_dim=256
                    # hidden_dims uses default [32, 64, 128, 256, 512]
                )
                
                state_dict = torch.load(ckpt_path, map_location=self.device)
                vae.load_state_dict(state_dict)
                vae.to(self.device)
                vae.eval()
                
                # Multi-GPU support
                if self.multi_gpu and torch.cuda.device_count() > 1:
                    vae = nn.DataParallel(vae)
                
                vaes[cam_name] = vae
                print(f"  Loaded VAE for {cam_name}")
            else:
                print(f"  Warning: No checkpoint found for {cam_name}")
        
        return vaes
    
    def synthesize_defect(self,
                          ok_images: Dict[str, torch.Tensor],
                          defect_type: Optional[str] = None,
                          inject_all_views: bool = False) -> Dict[str, Dict]:
        """
        Synthesize defect on OK images.
        
        Args:
            ok_images: Dict of OK images per view
            defect_type: Type of defect to inject, random if None
            inject_all_views: Whether to inject in all valid views
            
        Returns:
            Dict with synthesized images, masks, and defect info per view
        """
        if defect_type is None:
            defect_type = random.choice(self.DEFECT_TYPES)
        
        # Get valid views for this defect type
        valid_views = self.DEFECT_VIEW_MAPPING.get(defect_type, ['cam1'])
        
        results = {}
        
        # Decide which views to inject
        if inject_all_views:
            inject_views = [v for v in valid_views if v in ok_images]
        else:
            # Random selection
            available = [v for v in valid_views if v in ok_images]
            if available:
                inject_views = [random.choice(available)]
            else:
                inject_views = []
        
        # Generate consistent defect location for edge defects
        if defect_type in ['silver_overflow', 'breakage']:
            # These defects should appear at edges consistently
            shared_location = self._generate_edge_location(defect_type)
        else:
            shared_location = None
        
        for cam_name, image in ok_images.items():
            if cam_name in inject_views:
                # Inject defect
                if shared_location and cam_name in valid_views:
                    location = self._transform_location(shared_location, 'cam1', cam_name)
                else:
                    location = None
                
                result = self.defect_injector(
                    image,
                    defect_type,
                    location=location
                )
                results[cam_name] = {
                    'image': result['image'],
                    'mask': result['mask'],
                    'defect_type': defect_type,
                    'has_defect': True
                }
                
                # Update statistics
                self.stats['per_view_count'][cam_name] += 1
            else:
                # Keep original
                results[cam_name] = {
                    'image': image,
                    'mask': torch.zeros(self.image_size, self.image_size),
                    'defect_type': None,
                    'has_defect': False
                }
        
        # Update defect statistics
        self.stats['per_defect_count'][defect_type] += 1
        self.stats['total_generated'] += 1
        
        return results
    
    def _generate_edge_location(self, defect_type: str) -> Tuple[int, int]:
        """Generate location at edge for edge-related defects"""
        edge = random.choice(['left', 'right', 'top', 'bottom'])
        margin = 30
        
        if edge == 'left':
            x = random.randint(margin, margin + 30)
            y = random.randint(50, self.image_size - 50)
        elif edge == 'right':
            x = random.randint(self.image_size - margin - 30, self.image_size - margin)
            y = random.randint(50, self.image_size - 50)
        elif edge == 'top':
            x = random.randint(50, self.image_size - 50)
            y = random.randint(margin, margin + 30)
        else:  # bottom
            x = random.randint(50, self.image_size - 50)
            y = random.randint(self.image_size - margin - 30, self.image_size - margin)
        
        return (x, y)
    
    def _transform_location(self, 
                            location: Tuple[int, int], 
                            from_cam: str, 
                            to_cam: str) -> Tuple[int, int]:
        """Transform defect location from one view to corresponding view"""
        x, y = location
        
        # Simple transformation based on edge correspondence
        # In practice, this would need calibration data
        if from_cam == 'cam1':
            if to_cam == 'cam3':  # Left side
                return (y, x)  # Rotate 90
            elif to_cam == 'cam5':  # Right side
                return (self.image_size - y, x)
            elif to_cam in ['cam4', 'cam6']:
                return location  # Keep same for top/bottom
        
        return location
    
    def generate_batch(self,
                       dataloader: DataLoader,
                       num_samples: int,
                       output_dir: str,
                       writer: Optional[SummaryWriter] = None,
                       defect_distribution: Optional[Dict[str, float]] = None):
        """
        Generate a batch of synthetic defect samples.
        
        Args:
            dataloader: DataLoader for OK images
            num_samples: Number of samples to generate
            output_dir: Directory to save samples
            writer: TensorBoard writer for logging
            defect_distribution: Optional distribution over defect types
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirs for each defect type
        for defect_type in self.DEFECT_TYPES:
            (output_path / defect_type).mkdir(exist_ok=True)
            (output_path / defect_type / 'masks').mkdir(exist_ok=True)
        
        generated = 0
        samples_processed = 0
        
        pbar = tqdm(total=num_samples, desc="Generating synthetic samples")
        
        while generated < num_samples:
            for batch in dataloader:
                if generated >= num_samples:
                    break
                
                # Get OK images for all views
                ok_images = {}
                for cam_idx in range(1, 7):
                    cam_key = f'cam{cam_idx}'
                    if cam_key in batch:
                        ok_images[cam_key] = batch[cam_key].to(self.device)
                
                if not ok_images:
                    continue
                    
                batch_size = list(ok_images.values())[0].shape[0]
                
                for b in range(batch_size):
                    if generated >= num_samples:
                        break
                    
                    # Select defect type
                    if defect_distribution:
                        defect_type = random.choices(
                            list(defect_distribution.keys()),
                            weights=list(defect_distribution.values())
                        )[0]
                    else:
                        defect_type = random.choice(self.DEFECT_TYPES)
                    
                    # Get single sample
                    sample_images = {k: v[b] for k, v in ok_images.items()}
                    
                    # Synthesize
                    results = self.synthesize_defect(sample_images, defect_type)
                    
                    # Save
                    self._save_sample(results, output_path, defect_type, generated)
                    
                    # Log to TensorBoard periodically
                    if writer and generated % 100 == 0:
                        self._log_to_tensorboard(writer, results, generated)
                    
                    generated += 1
                    samples_processed += 1
                    pbar.update(1)
        
        pbar.close()
        
        # Save final statistics
        self._save_statistics(output_path)
        
        print(f"\nGenerated {generated} synthetic samples in {output_dir}")
        print(f"\nDefect distribution:")
        for defect_type, count in self.stats['per_defect_count'].items():
            print(f"  {defect_type}: {count}")
    
    def _save_sample(self,
                     results: Dict[str, Dict],
                     output_dir: Path,
                     defect_type: str,
                     sample_idx: int):
        """Save synthesized sample to disk"""
        defect_dir = output_dir / defect_type
        
        for cam_name, data in results.items():
            if data['has_defect']:
                image = data['image']
                mask = data['mask']
                
                # Convert to numpy
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()
                    if image.shape[0] == 3:
                        image = np.transpose(image, (1, 2, 0))
                
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                # Ensure correct range
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                
                # Save image
                img_path = defect_dir / f'{cam_name}_synth_{sample_idx:06d}.png'
                Image.fromarray(image).save(img_path)
                
                # Save mask
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
                mask_path = defect_dir / 'masks' / f'{cam_name}_synth_{sample_idx:06d}_mask.png'
                Image.fromarray(mask).save(mask_path)
    
    def _log_to_tensorboard(self, 
                            writer: SummaryWriter, 
                            results: Dict[str, Dict],
                            step: int):
        """Log sample images to TensorBoard"""
        for cam_name, data in results.items():
            if data['has_defect']:
                image = data['image']
                if isinstance(image, np.ndarray):
                    # HWC -> CHW for tensorboard
                    if image.shape[-1] == 3:
                        image = np.transpose(image, (2, 0, 1))
                    image = torch.from_numpy(image)
                
                if image.max() > 1.0:
                    image = image / 255.0
                
                writer.add_image(
                    f'synthesis/{cam_name}/{data["defect_type"]}',
                    image,
                    step
                )
                break  # Only log one image per sample
    
    def _save_statistics(self, output_dir: Path):
        """Save generation statistics for paper"""
        stats_file = output_dir / 'generation_statistics.json'
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_generated': self.stats['total_generated'],
            'defect_distribution': self.stats['per_defect_count'],
            'view_distribution': self.stats['per_view_count']
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Also save as CSV for easy Excel import
        csv_file = output_dir / 'generation_statistics.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Defect Type', 'Count'])
            for defect_type, count in self.stats['per_defect_count'].items():
                writer.writerow([defect_type, count])
        
        print(f"\nStatistics saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Multi-View Defect Synthesis Training")
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to OK dataset directory')
    parser.add_argument('--vae_dir', type=str, required=True,
                        help='Path to trained VAE checkpoints')
    parser.add_argument('--ng_dir', type=str, default=None,
                        help='Path to classified NG dataset')
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                        help='Output directory for synthetic samples')
    parser.add_argument('--num_samples', type=int, default=3000,
                        help='Number of samples to generate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Use multiple GPUs')
    parser.add_argument('--log_dir', type=str, default='experiments/synthesis_logs',
                        help='TensorBoard log directory')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
        args.multi_gpu = False
    
    print("=" * 60)
    print("Multi-View Defect Synthesis")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"VAE directory: {args.vae_dir}")
    print(f"NG directory: {args.ng_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    if args.multi_gpu:
        print(f"Multi-GPU: Yes ({torch.cuda.device_count()} GPUs)")
    print()
    
    # Create TensorBoard writer
    log_dir = Path(args.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"TensorBoard logs: {log_dir}")
    print(f"  Run: tensorboard --logdir={args.log_dir}")
    print()
    
    # Create dataset
    dataset = MultiViewDataset(
        root_dir=args.data_dir,
        group='Group1',  # Use Group1 by default
        image_size=(args.image_size, args.image_size)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Loaded {len(dataset)} OK samples")
    print()
    
    # Create synthesizer
    synthesizer = MultiViewDefectSynthesizer(
        vae_checkpoint_dir=args.vae_dir,
        ng_data_dir=args.ng_dir,
        image_size=args.image_size,
        device=args.device,
        multi_gpu=args.multi_gpu
    )
    
    # Generate samples
    synthesizer.generate_batch(
        dataloader=dataloader,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        writer=writer
    )
    
    # Close TensorBoard writer
    writer.close()
    
    print("\nDone!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Statistics saved to: {args.output_dir}/generation_statistics.json")


if __name__ == '__main__':
    main()
