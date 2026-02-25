"""
Augment: each image -> +180 rotation + horizontal flip
"""
import os
from PIL import Image

TARGETS = [
    r"E:\code\dataset\Generative_Modeling\data\datasets\NG\diffusion",
    r"E:\code\dataset\Generative_Modeling\data\datasets\NG\contamination",
]

def augment(target_dir):
    files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith('.png')])
    print(f"\n{os.path.basename(target_dir)}: {len(files)} files")
    
    created = 0
    for fname in files:
        fpath = os.path.join(target_dir, fname)
        base, ext = os.path.splitext(fname)
        if '_r180' in base or '_hflip' in base:
            continue
        img = Image.open(fpath)
        
        r180_path = os.path.join(target_dir, f"{base}_r180{ext}")
        if not os.path.exists(r180_path):
            img.rotate(180).save(r180_path)
            created += 1
        
        hflip_path = os.path.join(target_dir, f"{base}_hflip{ext}")
        if not os.path.exists(hflip_path):
            img.transpose(Image.FLIP_LEFT_RIGHT).save(hflip_path)
            created += 1
    
    total = len([f for f in os.listdir(target_dir) if f.lower().endswith('.png')])
    print(f"  Created: {created} | Total now: {total}")

if __name__ == "__main__":
    for t in TARGETS:
        augment(t)
    print("\nDone!")
