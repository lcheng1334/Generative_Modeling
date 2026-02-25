"""
Further analysis: augmented vs original images, Group detection in NG, capacitor check
"""
import os
from PIL import Image
from collections import Counter
import re

base_ok = r'E:\code\dataset\Generative_Modeling\data\datasets\OK'
base_classify = r'E:\code\dataset\Generative_Modeling\data\datasets\NG\classify'

# ======== 1. Augmentation analysis: how many are originals vs rotated? ========
print("=" * 60)
print("1. ORIGINAL vs AUGMENTED IMAGES IN CLASSIFY")
print("=" * 60)

for defect in sorted(os.listdir(base_classify)):
    d = os.path.join(base_classify, defect)
    if not os.path.isdir(d):
        continue
    files = os.listdir(d)
    
    originals = [f for f in files if not any(s in f for s in ['_180', '_L90', '_R90'])]
    aug_180 = [f for f in files if '_180' in f]
    aug_l90 = [f for f in files if '_L90' in f]
    aug_r90 = [f for f in files if '_R90' in f]
    
    print(f"  {defect:20s}: total={len(files):4d}, originals={len(originals):4d}, "
          f"_180={len(aug_180):4d}, _L90={len(aug_l90):4d}, _R90={len(aug_r90):4d}")

# ======== 2. Detect Group1/Group2 in NG by image orientation ========
print("\n" + "=" * 60)
print("2. NG IMAGE ORIENTATION (detect vertical/horizontal by size)")
print("=" * 60)

for defect in sorted(os.listdir(base_classify)):
    d = os.path.join(base_classify, defect)
    if not os.path.isdir(d):
        continue
    files = os.listdir(d)
    
    # Only check originals (no rotation suffix)
    originals = [f for f in files if not any(s in f for s in ['_180', '_L90', '_R90'])]
    
    landscape = 0  # 1440x1080 (width > height)
    portrait = 0   # 1080x1440 (height > width)
    small_landscape = 0  # 720x540
    small_portrait = 0   # 540x720
    other = 0
    
    for f in originals[:50]:  # sample 50
        try:
            img = Image.open(os.path.join(d, f))
            w, h = img.size
            if w == 1440 and h == 1080:
                landscape += 1
            elif w == 1080 and h == 1440:
                portrait += 1
            elif w == 720 and h == 540:
                small_landscape += 1
            elif w == 540 and h == 720:
                small_portrait += 1
            else:
                other += 1
        except:
            pass
    
    if originals:
        sampled = min(50, len(originals))
        print(f"  {defect:20s} (sampled {sampled}): "
              f"1440x1080={landscape}, 1080x1440={portrait}, "
              f"720x540={small_landscape}, 540x720={small_portrait}, other={other}")

# ======== 3. Check NG/Cam1 vs NG/Cam5 naming patterns ========
print("\n" + "=" * 60)
print("3. NG/Cam1 and NG/Cam5 NAMING PATTERNS")
print("=" * 60)

base_ng = r'E:\code\dataset\Generative_Modeling\data\datasets\NG'
for cam in ['Cam1', 'Cam5']:
    cam_path = os.path.join(base_ng, cam)
    if not os.path.isdir(cam_path):
        continue
    files = sorted(os.listdir(cam_path))
    print(f"\n  {cam}: {len(files)} files")
    print(f"  First 5: {files[:5]}")
    print(f"  Last 5: {files[-5:]}")
    
    # Check augmentation
    originals = [f for f in files if not any(s in f for s in ['_180', '_L90', '_R90'])]
    print(f"  Originals (no rotation): {len(originals)}")

# ======== 4. Per-cam original count in classify ========
print("\n" + "=" * 60)
print("4. ORIGINAL (non-augmented) IMAGES PER CAM PER DEFECT")
print("=" * 60)

print(f"{'Defect Type':20s} | {'Cam1':>5s} | {'Cam2':>5s} | {'Cam3':>5s} | {'Cam4':>5s} | {'Cam5':>5s} | {'Cam6':>5s} | {'Total':>5s}")
print("-" * 80)

for defect in ['adhesion','breakage','contamination','diffusion',
               'exposed_substrate','reversed_print','silver_overflow','other']:
    d = os.path.join(base_classify, defect)
    if not os.path.isdir(d):
        continue
    files = os.listdir(d)
    # Only originals
    originals = [f for f in files if not any(s in f for s in ['_180', '_L90', '_R90'])]
    
    cams = {}
    for i in range(1, 7):
        cams[f'Cam{i}'] = sum(1 for f in originals if f.startswith(f'Cam{i}'))
    total = len(originals)
    print(f"{defect:20s} | {cams['Cam1']:5d} | {cams['Cam2']:5d} | {cams['Cam3']:5d} | "
          f"{cams['Cam4']:5d} | {cams['Cam5']:5d} | {cams['Cam6']:5d} | {total:5d}")
