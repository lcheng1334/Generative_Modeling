"""
Multi-View Defect Synthesis Training Script

This script trains the complete synthesis pipeline:
1. Load OK images from all 6 views
2. Inject synthetic defects using DefectInjector
3. Save synthesized NG samples for downstream tasks

Usage:
    python scripts/train_synthesis.py --data_dir data/datasets --num_samples 3000
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
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv

from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.generator.defect_injector import DefectInjector
from src.models.vae import VanillaVAE


# Defect types and view mapping
DEFECT_TYPES = ['breakage', 'silver_overflow', 'exposed_substrate', 
                'diffusion', 'contamination', 'reversed_print', 'adhesion']


def load_image(path: Path, image_size: int = 256) -> torch.Tensor:
    """Load and preprocess a single image"""
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img)


def find_image_files(data_dir: Path) -> Dict[str, List[Path]]:
    """Find all OK images organized by camera"""
    images = {}
    
    # Try different directory structures
    for cam_idx in range(1, 7):
        cam_name = f'cam{cam_idx}'
        possible_dirs = [
            data_dir / 'OK' / 'Group1' / f'Cam{cam_idx}_Group1',
            data_dir / 'Group1' / f'Cam{cam_idx}_Group1',
            data_dir / f'Cam{cam_idx}_Group1',
            data_dir / f'Cam{cam_idx}',
        ]
        
        for d in possible_dirs:
            if d.exists():
                files = list(d.glob('*.png')) + list(d.glob('*.jpg'))
                if files:
                    images[cam_name] = files
                    print(f"  Found {len(files)} images for {cam_name}")
                    break
    
    return images


def load_vaes(checkpoint_dir: str, device: str) -> Dict[str, VanillaVAE]:
    """Load pre-trained VAE models for each view"""
    vaes = {}
    checkpoint_path = Path(checkpoint_dir)
    
    for cam_idx in range(1, 7):
        cam_name = f'cam{cam_idx}'
        cam_dir = checkpoint_path / f'Cam{cam_idx}'
        ckpt_path = cam_dir / 'best_vae.pth'
        
        if ckpt_path.exists():
            # Use default hidden_dims=[32, 64, 128, 256, 512] to match trained model
            vae = VanillaVAE(in_channels=3, latent_dim=256)
            state_dict = torch.load(ckpt_path, map_location=device)
            vae.load_state_dict(state_dict)
            vae.to(device)
            vae.eval()
            vaes[cam_name] = vae
            print(f"  Loaded VAE for {cam_name}")
        else:
            print(f"  Warning: No checkpoint found for {cam_name}")
    
    return vaes


def save_statistics(output_dir: Path, stats: Dict):
    """Save generation statistics for paper"""
    stats_file = output_dir / 'generation_statistics.json'
    
    stats_data = {
        'timestamp': datetime.now().isoformat(),
        'total_generated': stats['total_generated'],
        'defect_distribution': stats['per_defect_count'],
        'view_distribution': stats['per_view_count']
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    # Also save as CSV for easy Excel import
    csv_file = output_dir / 'generation_statistics.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Defect Type', 'Count'])
        for defect_type, count in stats['per_defect_count'].items():
            writer.writerow([defect_type, count])
    
    print(f"\nStatistics saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Multi-View Defect Synthesis")
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory (parent of OK folder)')
    parser.add_argument('--vae_dir', type=str, required=True,
                        help='Path to trained VAE checkpoints')
    parser.add_argument('--ng_dir', type=str, default=None,
                        help='Path to classified NG dataset for texture sampling')
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                        help='Output directory for synthetic samples')
    parser.add_argument('--num_samples', type=int, default=3000,
                        help='Number of samples to generate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--log_dir', type=str, default='experiments/synthesis_logs',
                        help='TensorBoard log directory')
    
    # Unused but kept for compatibility
    parser.add_argument('--batch_size', type=int, default=16, help='(unused)')
    parser.add_argument('--num_workers', type=int, default=8, help='(unused)')
    parser.add_argument('--multi_gpu', action='store_true', help='(unused)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Multi-View Defect Synthesis")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"VAE directory: {args.vae_dir}")
    print(f"NG directory: {args.ng_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print()
    
    # Create TensorBoard writer
    log_dir = Path(args.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"TensorBoard logs: {log_dir}")
    print(f"  Run: tensorboard --logdir={args.log_dir}")
    print()
    
    # Find images
    data_dir = Path(args.data_dir)
    print("Scanning for images...")
    image_files = find_image_files(data_dir)
    
    if not image_files:
        print("ERROR: No images found!")
        return
    
    total_images = sum(len(v) for v in image_files.values())
    print(f"Total: {total_images} images across {len(image_files)} cameras")
    print()
    
    # Load VAEs (optional, for latent space operations)
    print("Loading VAE models...")
    vaes = load_vaes(args.vae_dir, args.device)
    print(f"Loaded {len(vaes)} VAE models")
    print()
    
    # Create defect injector
    defect_injector = DefectInjector(
        image_size=args.image_size,
        ng_data_dir=args.ng_dir
    )
    
    # Statistics for paper
    stats = {
        'total_generated': 0,
        'per_defect_count': {d: 0 for d in DEFECT_TYPES},
        'per_view_count': {f'cam{i}': 0 for i in range(1, 7)}
    }
    
    # Create output directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for defect_type in DEFECT_TYPES:
        (output_path / defect_type).mkdir(exist_ok=True)
        (output_path / defect_type / 'masks').mkdir(exist_ok=True)
    
    # Generate samples
    print("Generating synthetic samples...")
    generated = 0
    
    # Get list of cameras with images
    available_cams = list(image_files.keys())
    
    pbar = tqdm(total=args.num_samples, desc="Generating")
    
    while generated < args.num_samples:
        # Random camera selection
        cam_name = random.choice(available_cams)
        
        # Random image from that camera
        img_path = random.choice(image_files[cam_name])
        
        try:
            # Load image
            image = load_image(img_path, args.image_size).to(args.device)
            
            # Random defect type
            defect_type = random.choice(DEFECT_TYPES)
            
            # Inject defect
            result = defect_injector(image, defect_type)
            
            # Save
            synth_image = result['image']
            mask = result['mask']
            
            # Convert to numpy
            if isinstance(synth_image, torch.Tensor):
                synth_image = synth_image.cpu().numpy()
                if synth_image.shape[0] == 3:
                    synth_image = np.transpose(synth_image, (1, 2, 0))
            
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Ensure correct range
            if synth_image.max() <= 1.0:
                synth_image = (synth_image * 255).astype(np.uint8)
            else:
                synth_image = synth_image.astype(np.uint8)
            
            # Save image
            img_save_path = output_path / defect_type / f'{cam_name}_synth_{generated:06d}.png'
            Image.fromarray(synth_image).save(img_save_path)
            
            # Save mask
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            mask_save_path = output_path / defect_type / 'masks' / f'{cam_name}_synth_{generated:06d}_mask.png'
            Image.fromarray(mask).save(mask_save_path)
            
            # Update statistics
            stats['total_generated'] += 1
            stats['per_defect_count'][defect_type] += 1
            stats['per_view_count'][cam_name] += 1
            
            # Log to TensorBoard periodically
            if generated % 100 == 0:
                if synth_image.shape[-1] == 3:
                    tb_image = np.transpose(synth_image, (2, 0, 1))
                else:
                    tb_image = synth_image
                writer.add_image(f'synthesis/{cam_name}/{defect_type}', 
                                tb_image.astype(np.float32) / 255.0, generated)
            
            generated += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"\n  Error processing {img_path}: {e}")
            continue
    
    pbar.close()
    
    # Save statistics
    save_statistics(output_path, stats)
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\nGenerated {generated} synthetic samples in {args.output_dir}")
    print(f"\nDefect distribution:")
    for defect_type, count in stats['per_defect_count'].items():
        print(f"  {defect_type}: {count}")
    
    print(f"\nView distribution:")
    for cam_name, count in stats['per_view_count'].items():
        print(f"  {cam_name}: {count}")
    
    print("\nDone!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Statistics saved to: {args.output_dir}/generation_statistics.json")


if __name__ == '__main__':
    main()
