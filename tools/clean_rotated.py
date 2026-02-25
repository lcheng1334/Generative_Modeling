"""
Delete rotated (1080x1440) Cam1/Cam2 images from original NG folder.
ONLY touches Cam1 and Cam2 files. Cam3/4/5/6 are NOT affected.
"""
import os
from PIL import Image

NG_DIR = r"E:\code\dataset\Generative_Modeling\data\datasets\NG"

def clean_rotated():
    to_delete = []
    skipped = 0
    
    for defect_type in sorted(os.listdir(NG_DIR)):
        defect_dir = os.path.join(NG_DIR, defect_type)
        if not os.path.isdir(defect_dir):
            continue
        if 'renamed' in defect_type:
            continue
        
        for fname in os.listdir(defect_dir):
            if not fname.lower().endswith('.png'):
                continue
            
            # ONLY process Cam1 and Cam2
            if not (fname.startswith('Cam1') or fname.startswith('Cam2')):
                skipped += 1
                continue
            
            fpath = os.path.join(defect_dir, fname)
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
                    if w == 1080 and h == 1440:
                        to_delete.append((defect_type, fname, fpath))
            except:
                pass
    
    # Preview
    print(f"Cam3/4/5/6 files skipped (safe): {skipped}")
    print(f"Cam1/2 files to delete (1080x1440): {len(to_delete)}")
    print()
    
    # Group by defect type
    from collections import defaultdict
    by_type = defaultdict(list)
    for dt, fn, fp in to_delete:
        by_type[dt].append(fn)
    
    for dt in sorted(by_type.keys()):
        files = by_type[dt]
        print(f"  {dt}: {len(files)} files")
        for f in files[:3]:
            print(f"    {f}")
        if len(files) > 3:
            print(f"    ... and {len(files)-3} more")
    
    # Execute deletion
    print(f"\nDeleting {len(to_delete)} files...")
    for dt, fn, fp in to_delete:
        os.remove(fp)
    
    print("Done!")
    
    # Verify: count remaining
    print("\nRemaining files per defect type:")
    for defect_type in sorted(os.listdir(NG_DIR)):
        defect_dir = os.path.join(NG_DIR, defect_type)
        if not os.path.isdir(defect_dir) or 'renamed' in defect_type:
            continue
        count = len([f for f in os.listdir(defect_dir) if f.lower().endswith('.png')])
        print(f"  {defect_type}: {count}")

if __name__ == "__main__":
    clean_rotated()
