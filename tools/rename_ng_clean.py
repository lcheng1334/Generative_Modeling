"""
NG Clean Rename Script
- All PNG files (original + augmented) treated as normal defect samples
- No augmentation tags in output filenames
- Output: cam{X}_{pose}_{defect}_{index:05d}.png
- Grouped by cam_id within each defect/pose folder
- Sequential index per cam_id
- Progress bar via tqdm
"""
import os
import re
import shutil
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

NG_BASE = r"E:\code\dataset\Generative_Modeling\data\datasets\NG"
OUT_BASE = r"E:\code\dataset\Generative_Modeling\data\datasets\NG_renamed"

DEFECTS = [
    "adhesion", "breakage", "contamination", "diffusion",
    "exposed_substrate", "reversed_print", "silver_overflow"
]

POSES = ["p0", "p90"]


def get_cam_id(fname):
    """Extract cam number from original filename."""
    m = re.match(r'[Cc]am(\d+)', fname)
    return m.group(1) if m else "0"


def main():
    # Remove old output if exists
    if os.path.exists(OUT_BASE):
        print(f"Removing existing output: {OUT_BASE}")
        shutil.rmtree(OUT_BASE)

    # Count total files for overall progress bar
    total_files = 0
    tasks = []
    for defect in DEFECTS:
        for pose in POSES:
            src_dir = os.path.join(NG_BASE, defect, pose)
            if not os.path.isdir(src_dir):
                continue
            files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith('.png')])
            if files:
                tasks.append((defect, pose, src_dir, files))
                total_files += len(files)

    summary = []

    with tqdm(total=total_files, unit="img", desc="Renaming NG") as pbar:
        for defect, pose, src_dir, files in tasks:
            dst_dir = os.path.join(OUT_BASE, defect, pose)
            os.makedirs(dst_dir, exist_ok=True)

            # Group by cam_id, preserve sort order
            cam_files = defaultdict(list)
            for f in files:
                cam = get_cam_id(f)
                cam_files[cam].append(f)

            count = 0
            for cam in sorted(cam_files.keys()):
                for idx, fname in enumerate(cam_files[cam], start=1):
                    new_name = f"cam{cam}_{pose}_{defect}_{idx:05d}.png"
                    src = os.path.join(src_dir, fname)
                    dst = os.path.join(dst_dir, new_name)
                    shutil.copy2(src, dst)
                    count += 1
                    pbar.update(1)

            summary.append((defect, pose, count))

    # Print summary table
    print(f"\n{'='*55}")
    print(f"  {'Defect':<22} {'Pose':<5} {'Files':>6}")
    print(f"{'='*55}")
    grand = 0
    for defect, pose, count in summary:
        print(f"  {defect:<22} {pose:<5} {count:>6}")
        grand += count
    print(f"{'='*55}")
    print(f"  {'TOTAL':<22} {'':5} {grand:>6}")
    print(f"\nOutput: {OUT_BASE}")
    print("Original NG files NOT deleted. Verify output before deleting.")

    # Show sample
    print(f"\nSample (adhesion/p0):")
    ap0 = os.path.join(OUT_BASE, "adhesion", "p0")
    if os.path.isdir(ap0):
        for f in sorted(os.listdir(ap0))[:5]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
