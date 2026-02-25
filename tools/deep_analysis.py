"""
Deep dataset analysis for paper preparation
"""
import os
from PIL import Image
from collections import defaultdict, Counter
import re

base_ok = r'E:\code\dataset\Generative_Modeling\data\datasets\OK'
base_ng = r'E:\code\dataset\Generative_Modeling\data\datasets\NG'
base_classify = os.path.join(base_ng, 'classify')

# ======== 1. Image Resolution Analysis ========
print("=" * 60)
print("1. IMAGE RESOLUTION ANALYSIS")
print("=" * 60)

# Check OK images
print("\n--- OK Images ---")
for group in ['Group1', 'Group2']:
    for cam_dir in sorted(os.listdir(os.path.join(base_ok, group))):
        cam_path = os.path.join(base_ok, group, cam_dir)
        if not os.path.isdir(cam_path):
            continue
        files = [f for f in os.listdir(cam_path) if f.lower().endswith(('.png','.jpg','.bmp'))]
        if files:
            img = Image.open(os.path.join(cam_path, files[0]))
            sizes = set()
            for f in files[:20]:  # sample 20
                im = Image.open(os.path.join(cam_path, f))
                sizes.add(im.size)
            print(f"  {group}/{cam_dir}: {len(files)} images, sizes: {sizes}")

# Check classify images
print("\n--- NG Classify Images ---")
for defect in sorted(os.listdir(base_classify)):
    d = os.path.join(base_classify, defect)
    if not os.path.isdir(d):
        continue
    files = [f for f in os.listdir(d) if f.lower().endswith(('.png','.jpg','.bmp'))]
    if files:
        sizes = set()
        for f in files[:20]:
            im = Image.open(os.path.join(d, f))
            sizes.add(im.size)
        print(f"  {defect}: sizes: {sizes}")

# ======== 2. Filename Pattern Analysis ========
print("\n" + "=" * 60)
print("2. FILENAME PATTERN ANALYSIS (checking for product ID)")
print("=" * 60)

# Analyze NG filenames for potential pairing info
for defect in ['breakage', 'silver_overflow', 'adhesion']:
    d = os.path.join(base_classify, defect)
    if not os.path.isdir(d):
        continue
    files = sorted(os.listdir(d))[:10]
    print(f"\n  {defect} (first 10 filenames):")
    for f in files:
        print(f"    {f}")

# Analyze OK filenames
print("\n  OK/Group1/Cam1_Group1 (first 10 filenames):")
ok_path = os.path.join(base_ok, 'Group1', 'Cam1_Group1')
for f in sorted(os.listdir(ok_path))[:10]:
    print(f"    {f}")

# ======== 3. Check if NG data has Group info in filenames ========
print("\n" + "=" * 60)
print("3. GROUP INFO IN NG FILENAMES")
print("=" * 60)

for defect in sorted(os.listdir(base_classify)):
    d = os.path.join(base_classify, defect)
    if not os.path.isdir(d):
        continue
    files = os.listdir(d)
    has_group = sum(1 for f in files if 'Group' in f or 'group' in f)
    has_g1 = sum(1 for f in files if 'Group1' in f)
    has_g2 = sum(1 for f in files if 'Group2' in f)
    print(f"  {defect}: total={len(files)}, has_Group={has_group}, G1={has_g1}, G2={has_g2}")

# ======== 4. Check filename numeric IDs for potential OK-NG matching ========
print("\n" + "=" * 60)
print("4. NUMERIC ID ANALYSIS (can we match OK-NG?)")
print("=" * 60)

# Extract numeric IDs from OK filenames
ok_ids = set()
ok_path = os.path.join(base_ok, 'Group1', 'Cam1_Group1')
for f in os.listdir(ok_path):
    nums = re.findall(r'\d+', f)
    if nums:
        ok_ids.add(nums[-1])  # last number might be ID
print(f"  OK/Group1/Cam1: {len(ok_ids)} unique numeric IDs")
print(f"  Sample IDs: {sorted(list(ok_ids))[:10]}")

# Extract from NG
ng_ids = set()
for f in os.listdir(os.path.join(base_classify, 'breakage'))[:100]:
    nums = re.findall(r'\d+', f)
    if nums:
        ng_ids.add(nums[-1])
print(f"  NG/breakage (first 100): {len(ng_ids)} unique numeric IDs")
print(f"  Sample IDs: {sorted(list(ng_ids))[:10]}")

# Check overlap
overlap = ok_ids & ng_ids
print(f"  Overlap between OK and NG IDs: {len(overlap)}")

# ======== 5. Check NG/Cam1-6 folder naming patterns ========
print("\n" + "=" * 60)
print("5. NG/Cam FOLDER DEFECT TYPE BREAKDOWN")
print("=" * 60)
for cam in ['Cam1', 'Cam5']:
    cam_path = os.path.join(base_ng, cam)
    if not os.path.isdir(cam_path):
        continue
    files = os.listdir(cam_path)
    types = Counter()
    for f in files:
        # Extract defect type from filename like Cam1_NG_暗缺陷(NG1)_xxx.png
        match = re.search(r'NG_(.+?)_\d', f)
        if match:
            types[match.group(1)] += 1
    print(f"\n  {cam} ({len(files)} files):")
    for t, c in types.most_common():
        print(f"    {t}: {c}")

# ======== 6. Image mode and color analysis ========
print("\n" + "=" * 60)
print("6. IMAGE MODE AND COLOR STATS")
print("=" * 60)
import numpy as np

for label, path in [
    ("OK/Group1/Cam1", os.path.join(base_ok, 'Group1', 'Cam1_Group1')),
    ("OK/Group1/Cam4", os.path.join(base_ok, 'Group1', 'Cam4_Group1')),
    ("NG/classify/breakage", os.path.join(base_classify, 'breakage')),
    ("NG/classify/silver_overflow", os.path.join(base_classify, 'silver_overflow')),
]:
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg'))][:5]
    if files:
        img = Image.open(os.path.join(path, files[0]))
        arr = np.array(img)
        print(f"  {label}: mode={img.mode}, shape={arr.shape}, dtype={arr.dtype}")
        print(f"    R: mean={arr[:,:,0].mean():.1f}, G: mean={arr[:,:,1].mean():.1f}, B: mean={arr[:,:,2].mean():.1f}")
