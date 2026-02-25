"""
NG folder rename script
- Detects original vs augmented (suffixes: _180, _L90, _R90, etc.)
- Detects pose from (NG1)=p0, (NG2)=p90; no marker -> 'pX' (unknown, to be resolved)
- Renames to: cam{X}_{pose}_{defect}_{index:05d}[_r{aug}].png
- Keeps defect-type subfolders with cam sub-subfolders
- Copies first (verify), then deletes originals
"""
import os
import re
import shutil
from collections import defaultdict

NG_DIR = r"E:\code\dataset\Generative_Modeling\data\datasets\NG"
OUTPUT_DIR = r"E:\code\dataset\Generative_Modeling\data\datasets\NG_renamed"

# Augmentation suffix patterns (order matters: match longest first)
AUG_PATTERNS = [
    ('_180_180', 'r180x2'),
    ('_180_L90', 'r180l90'),
    ('_180_R90', 'r180r90'),
    ('_L90_180', 'rl90x180'),
    ('_L90_L90', 'rl90x2'),
    ('_L90_R90', 'rl90r90'),
    ('_R90_180', 'rr90x180'),
    ('_R90_L90', 'rr90l90'),
    ('_R90_R90', 'rr90x2'),
    ('_180', 'r180'),
    ('_L90', 'rl90'),
    ('_R90', 'rr90'),
]

def parse_ng_filename(filename):
    """Parse an NG filename to extract cam, pose, ID, and augmentation info."""
    name = filename.replace('.png', '').replace('.PNG', '')
    
    # Extract augmentation suffix
    aug_tag = None
    for suffix, tag in AUG_PATTERNS:
        if name.endswith(suffix):
            aug_tag = tag
            name = name[:-len(suffix)]
            break
    
    # Extract cam id
    cam_match = re.match(r'Cam(\d+)', name)
    cam_id = cam_match.group(1) if cam_match else '0'
    
    # Extract pose from (NG1) or (NG2)
    if '(NG1)' in name:
        pose = 'p0'
    elif '(NG2)' in name:
        pose = 'p90'
    else:
        pose = 'px'  # unknown, will try to resolve later
    
    # Extract numeric ID (the unique part)
    # Remove known prefixes to get just the ID
    id_part = re.sub(r'Cam\d+_NG_(\(NG\d\)_)?', '', name)
    
    return {
        'cam': cam_id,
        'pose': pose,
        'orig_id': id_part,
        'aug': aug_tag,
        'is_augmented': aug_tag is not None
    }


def rename_ng():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    stats = defaultdict(lambda: {'total': 0, 'original': 0, 'augmented': 0, 
                                  'p0': 0, 'p90': 0, 'px': 0})
    
    for defect_type in sorted(os.listdir(NG_DIR)):
        defect_dir = os.path.join(NG_DIR, defect_type)
        if not os.path.isdir(defect_dir):
            continue
        if 'renamed' in defect_type or 'other' in defect_type:
            continue
        
        print(f"\n{'='*60}")
        print(f"  {defect_type}")
        print(f"{'='*60}")
        
        files = sorted([f for f in os.listdir(defect_dir) if f.lower().endswith('.png')])
        
        # Parse all files and group by (cam, orig_id) to assign sequential indices
        parsed = []
        for f in files:
            info = parse_ng_filename(f)
            info['old_name'] = f
            info['defect'] = defect_type
            parsed.append(info)
        
        # Group originals by cam to assign sequential index per cam
        # First, collect unique original IDs per cam
        cam_orig_ids = defaultdict(list)
        for p in parsed:
            if not p['is_augmented']:
                key = (p['cam'], p['orig_id'])
                if key not in [(c, i) for c, i in cam_orig_ids[p['cam']]]:
                    cam_orig_ids[p['cam']].append((p['cam'], p['orig_id']))
        
        # Build ID -> index mapping per cam
        cam_id_to_idx = {}
        for cam in sorted(cam_orig_ids.keys()):
            for idx, (_, orig_id) in enumerate(cam_orig_ids[cam], start=1):
                cam_id_to_idx[(cam, orig_id)] = idx
        
        # Also assign indices for augmented-only items (whose orig_id might also appear)
        for p in parsed:
            key = (p['cam'], p['orig_id'])
            if key not in cam_id_to_idx:
                cam = p['cam']
                next_idx = max([v for (c, _), v in cam_id_to_idx.items() if c == cam], default=0) + 1
                cam_id_to_idx[key] = next_idx
        
        # Now rename
        for p in parsed:
            cam = p['cam']
            pose = p['pose']
            idx = cam_id_to_idx.get((cam, p['orig_id']), 0)
            
            if p['aug']:
                new_name = f"cam{cam}_{pose}_{defect_type}_{idx:05d}_{p['aug']}.png"
            else:
                new_name = f"cam{cam}_{pose}_{defect_type}_{idx:05d}.png"
            
            # Output: NG_renamed/{defect_type}/cam{X}/{new_name}
            cam_folder = f"cam{cam}"
            out_dir = os.path.join(OUTPUT_DIR, defect_type, cam_folder)
            os.makedirs(out_dir, exist_ok=True)
            
            src = os.path.join(defect_dir, p['old_name'])
            dst = os.path.join(out_dir, new_name)
            shutil.copy2(src, dst)
            
            # Stats
            s = stats[defect_type]
            s['total'] += 1
            if p['is_augmented']:
                s['augmented'] += 1
            else:
                s['original'] += 1
            s[pose] += 1
        
        s = stats[defect_type]
        print(f"  Total: {s['total']} | Original: {s['original']} | Augmented: {s['augmented']}")
        print(f"  Pose:  p0={s['p0']} | p90={s['p90']} | unknown={s['px']}")
        
        # Show sample renames
        samples = parsed[:3]
        for p in samples:
            idx = cam_id_to_idx.get((p['cam'], p['orig_id']), 0)
            aug_part = f"_{p['aug']}" if p['aug'] else ""
            new = f"cam{p['cam']}_{p['pose']}_{defect_type}_{idx:05d}{aug_part}.png"
            print(f"    {p['old_name']}")
            print(f"    -> {new}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    grand_total = 0
    grand_orig = 0
    grand_aug = 0
    for dt in sorted(stats.keys()):
        s = stats[dt]
        grand_total += s['total']
        grand_orig += s['original']
        grand_aug += s['augmented']
        print(f"  {dt:25s}: {s['original']:5d} orig + {s['augmented']:5d} aug = {s['total']:5d} total")
    print(f"  {'TOTAL':25s}: {grand_orig:5d} orig + {grand_aug:5d} aug = {grand_total:5d} total")
    
    print(f"\nOutput: {OUTPUT_DIR}")
    print("Original files NOT deleted yet. Verify output, then confirm deletion.")

if __name__ == "__main__":
    rename_ng()
