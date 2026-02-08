"""
Synthesis Quality Evaluation Script

Evaluates the quality of synthesized defect images.
Computes: FID (Fréchet Inception Distance), LPIPS, and visual comparisons.

Usage:
    python scripts/evaluate_synthesis.py \
        --synthetic_dir data/synthetic \
        --real_ng_dir data/datasets/NG/classify \
        --output_dir experiments/evaluation
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import metrics
try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    HAS_INCEPTION = True
except ImportError:
    HAS_INCEPTION = False
    print("Warning: torchvision inception not available")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed")

from scipy import linalg


class FIDCalculator:
    """Calculate Fréchet Inception Distance"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        if HAS_INCEPTION:
            self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            self.model.fc = torch.nn.Identity()  # Remove classification head
            self.model.to(device)
            self.model.eval()
        else:
            self.model = None
    
    def get_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract Inception features"""
        if self.model is None:
            return np.zeros((images.shape[0], 2048))
        
        # Resize to 299x299 for Inception
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = self.model(images)
        
        return features.cpu().numpy()
    
    def calculate_statistics(self, features: np.ndarray) -> tuple:
        """Calculate mean and covariance"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, 
                      real_features: np.ndarray, 
                      fake_features: np.ndarray) -> float:
        """Calculate FID between two feature sets"""
        mu1, sigma1 = self.calculate_statistics(real_features)
        mu2, sigma2 = self.calculate_statistics(fake_features)
        
        # Calculate FID
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)


class SynthesisEvaluator:
    """Evaluate synthesis quality"""
    
    def __init__(self, 
                 device: str = 'cuda',
                 image_size: int = 256):
        self.device = device
        self.image_size = image_size
        
        self.fid_calculator = FIDCalculator(device)
        
        if HAS_LPIPS:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
        else:
            self.lpips_model = None
    
    def load_images(self, 
                    image_dir: str, 
                    max_images: int = 1000) -> torch.Tensor:
        """Load images from directory"""
        image_dir = Path(image_dir)
        images = []
        
        # Find all images
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(list(image_dir.rglob(ext)))
        
        image_paths = image_paths[:max_images]
        
        for img_path in tqdm(image_paths, desc=f"Loading {image_dir.name}"):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((self.image_size, self.image_size))
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)
                images.append(img_tensor)
            except Exception as e:
                continue
        
        if images:
            return torch.stack(images)
        return torch.empty(0, 3, self.image_size, self.image_size)
    
    def evaluate_fid(self,
                     synthetic_dir: str,
                     real_dir: str,
                     max_images: int = 500) -> Dict[str, float]:
        """Calculate FID for each defect category"""
        results = {}
        
        synthetic_path = Path(synthetic_dir)
        real_path = Path(real_dir)
        
        # Get defect categories from synthetic dir
        categories = [d.name for d in synthetic_path.iterdir() if d.is_dir()]
        
        for category in categories:
            synth_cat_dir = synthetic_path / category
            real_cat_dir = real_path / category
            
            if not real_cat_dir.exists():
                print(f"  Skipping {category}: no real samples")
                continue
            
            print(f"\nEvaluating {category}...")
            
            # Load images
            synth_images = self.load_images(synth_cat_dir, max_images)
            real_images = self.load_images(real_cat_dir, max_images)
            
            if len(synth_images) < 10 or len(real_images) < 10:
                print(f"  Skipping {category}: insufficient samples")
                continue
            
            # Extract features
            synth_images = synth_images.to(self.device)
            real_images = real_images.to(self.device)
            
            synth_features = []
            real_features = []
            
            batch_size = 32
            for i in range(0, len(synth_images), batch_size):
                batch = synth_images[i:i+batch_size]
                synth_features.append(self.fid_calculator.get_features(batch))
            
            for i in range(0, len(real_images), batch_size):
                batch = real_images[i:i+batch_size]
                real_features.append(self.fid_calculator.get_features(batch))
            
            synth_features = np.concatenate(synth_features)
            real_features = np.concatenate(real_features)
            
            # Calculate FID
            fid = self.fid_calculator.calculate_fid(real_features, synth_features)
            results[category] = fid
            print(f"  {category} FID: {fid:.2f}")
        
        # Overall FID
        if results:
            results['mean'] = np.mean(list(results.values()))
        
        return results
    
    def evaluate_diversity(self, synthetic_dir: str, max_images: int = 100) -> Dict[str, float]:
        """Evaluate diversity of synthesized samples using LPIPS"""
        if self.lpips_model is None:
            return {}
        
        results = {}
        synthetic_path = Path(synthetic_dir)
        
        categories = [d.name for d in synthetic_path.iterdir() if d.is_dir()]
        
        for category in categories:
            cat_dir = synthetic_path / category
            images = self.load_images(cat_dir, max_images)
            
            if len(images) < 10:
                continue
            
            images = images.to(self.device)
            
            # Compute pairwise LPIPS distances
            distances = []
            num_pairs = min(100, len(images) * (len(images) - 1) // 2)
            
            indices = list(range(len(images)))
            for _ in range(num_pairs):
                i, j = np.random.choice(indices, size=2, replace=False)
                with torch.no_grad():
                    dist = self.lpips_model(
                        images[i:i+1] * 2 - 1,
                        images[j:j+1] * 2 - 1
                    )
                    distances.append(dist.item())
            
            results[category] = {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances))
            }
        
        return results
    
    def create_comparison_grid(self,
                               synthetic_dir: str,
                               real_dir: str,
                               output_path: str,
                               samples_per_category: int = 4):
        """Create visual comparison grid"""
        import matplotlib.pyplot as plt
        
        synthetic_path = Path(synthetic_dir)
        real_path = Path(real_dir)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        categories = [d.name for d in synthetic_path.iterdir() if d.is_dir()]
        
        for category in categories:
            synth_cat = synthetic_path / category
            real_cat = real_path / category
            
            synth_images = list(synth_cat.glob('*.png'))[:samples_per_category]
            real_images = list(real_cat.glob('*.png'))[:samples_per_category] if real_cat.exists() else []
            
            if not synth_images:
                continue
            
            n_cols = max(len(synth_images), len(real_images))
            fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6))
            
            if n_cols == 1:
                axes = axes.reshape(2, 1)
            
            # Real images
            for i in range(n_cols):
                if i < len(real_images):
                    img = Image.open(real_images[i])
                    axes[0, i].imshow(img)
                    axes[0, i].set_title('Real NG' if i == 0 else '')
                axes[0, i].axis('off')
            
            # Synthetic images
            for i in range(n_cols):
                if i < len(synth_images):
                    img = Image.open(synth_images[i])
                    axes[1, i].imshow(img)
                    axes[1, i].set_title('Synthetic' if i == 0 else '')
                axes[1, i].axis('off')
            
            plt.suptitle(f'{category}', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_path / f'comparison_{category}.png', dpi=150)
            plt.close()
        
        print(f"Saved comparison grids to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Synthesis Quality")
    
    parser.add_argument('--synthetic_dir', type=str, required=True,
                        help='Path to synthetic samples directory')
    parser.add_argument('--real_ng_dir', type=str, required=True,
                        help='Path to real NG samples directory')
    parser.add_argument('--output_dir', type=str, default='experiments/evaluation',
                        help='Output directory for results')
    parser.add_argument('--max_images', type=int, default=500,
                        help='Maximum images per category for FID')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--create_comparisons', action='store_true',
                        help='Create visual comparison grids')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Synthesis Quality Evaluation")
    print("=" * 60)
    print(f"Synthetic directory: {args.synthetic_dir}")
    print(f"Real NG directory: {args.real_ng_dir}")
    print()
    
    # Create evaluator
    evaluator = SynthesisEvaluator(device=args.device)
    
    # Evaluate FID
    print("\n--- FID Evaluation ---")
    fid_results = evaluator.evaluate_fid(
        synthetic_dir=args.synthetic_dir,
        real_dir=args.real_ng_dir,
        max_images=args.max_images
    )
    
    # Evaluate diversity
    print("\n--- Diversity Evaluation (LPIPS) ---")
    diversity_results = evaluator.evaluate_diversity(
        synthetic_dir=args.synthetic_dir
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    print("\nFID Scores (lower is better):")
    for category, fid in fid_results.items():
        print(f"  {category}: {fid:.2f}")
    
    print("\nDiversity (LPIPS, higher is more diverse):")
    for category, values in diversity_results.items():
        print(f"  {category}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'fid': fid_results,
        'diversity': diversity_results,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = output_path / f'synthesis_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Create comparisons
    if args.create_comparisons:
        evaluator.create_comparison_grid(
            synthetic_dir=args.synthetic_dir,
            real_dir=args.real_ng_dir,
            output_path=str(output_path / 'comparisons')
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
