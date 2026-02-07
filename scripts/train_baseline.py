"""
Train Baseline Models for 6 Views
支持多GPU服务器部署 (4x 3090)

Usage:
    # Train all views sequentially
    python scripts/train_baseline.py --all
    
    # Train specific view
    python scripts/train_baseline.py --view Cam1
    
    # Train with specific GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/train_baseline.py --view Cam1
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from src.models.vae import VanillaVAE
from src.core.trainer import VAETrainer
from src.datasets.multiview_dataset import ViewSpecificDataset


def get_config():
    """Training configuration"""
    return {
        'image_size': (256, 256),
        'batch_size': 32,  # 3090 can handle larger batch
        'latent_dim': 256,
        'learning_rate': 0.0005,
        'epochs': 50,  # More epochs for better quality
        'val_split': 0.1,
        'num_workers': 8,
    }


def train_view_model(view_name: str, data_root: str, output_dir: Path, config: dict):
    """Train VAE model for a single view"""
    print(f"\n{'='*60}")
    print(f"Training VAE for {view_name}")
    print(f"{'='*60}")
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Dataset transforms
    train_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = ViewSpecificDataset(
        root_dir=data_root,
        view=view_name,
        group='all',
        transform=train_transform,
        image_size=config['image_size']
    )
    
    # Split train/val
    total_size = len(full_dataset)
    val_size = int(total_size * config['val_split'])
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Batches: Train={len(train_loader)}, Val={len(val_loader)}")
    
    # Model
    model = VanillaVAE(
        in_channels=3, 
        latent_dim=config['latent_dim'],
        hidden_dims=[32, 64, 128, 256, 512]
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    
    # Output directory
    view_output_dir = output_dir / view_name
    view_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Trainer
    trainer = VAETrainer(
        model=model,
        optimizer_config={
            'lr': config['learning_rate'],
            'weight_decay': 1e-5,
            'scheduler_gamma': 0.95
        },
        device=device,
        log_dir=str(view_output_dir / 'logs')
    )
    
    # Train
    save_path = view_output_dir / 'best_vae.pth'
    trainer.train(
        train_loader, 
        val_loader, 
        epochs=config['epochs'], 
        save_path=str(save_path)
    )
    
    print(f"\nFinished training {view_name}")
    print(f"Model saved to: {save_path}")
    
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Train View-Specific VAE Models')
    parser.add_argument('--view', type=str, default=None, 
                       help='Specific view to train (e.g., Cam1). If not set, trains all.')
    parser.add_argument('--all', action='store_true', 
                       help='Train all 6 views sequentially')
    parser.add_argument('--data_root', type=str, 
                       default='../dataset/Generative_Modeling/data/datasets',
                       help='Path to dataset root (default: ../dataset/Generative_Modeling/data/datasets)')
    parser.add_argument('--output_dir', type=str,
                       default='./experiments/baseline_vae',
                       help='Output directory for models and logs')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Configuration
    config = get_config()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("Multi-View Defect Synthesis - Baseline VAE Training")
    print("="*60)
    print(f"Data Root: {data_root}")
    print(f"Output Dir: {output_dir}")
    print(f"Config: {config}")
    
    # Determine which views to train
    all_views = ['Cam1', 'Cam2', 'Cam3', 'Cam4', 'Cam5', 'Cam6']
    
    if args.view:
        views_to_train = [args.view]
    elif args.all:
        views_to_train = all_views
    else:
        print("\nError: Please specify --view or --all")
        print("Example: python train_baseline.py --view Cam1")
        print("Example: python train_baseline.py --all")
        return
    
    # Train each view
    for view in views_to_train:
        if view not in all_views:
            print(f"Warning: {view} is not a valid view name, skipping...")
            continue
        train_view_model(view, str(data_root), output_dir, config)
    
    print("\n" + "="*60)
    print("All training completed!")
    print("="*60)


if __name__ == '__main__':
    main()
