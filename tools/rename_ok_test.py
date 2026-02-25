"""
Rename preview script - OK/0 folder
Keeps Cam subfolders, renames files inside
Convention: cam{X}_p0_ok_{index:05d}.png
"""
import os
import shutil

SRC_DIR = r"E:\code\dataset\Generative_Modeling\data\datasets\OK\0°"
PREVIEW_DIR = r"E:\code\dataset\Generative_Modeling\data\datasets\OK\0_renamed_preview"

def rename_ok_p0():
    # Clean old preview
    if os.path.exists(PREVIEW_DIR):
        shutil.rmtree(PREVIEW_DIR)

    total = 0
    for cam_folder in sorted(os.listdir(SRC_DIR)):
        cam_path = os.path.join(SRC_DIR, cam_folder)
        if not os.path.isdir(cam_path):
            continue
        if "renamed" in cam_folder:
            continue

        # "Cam1_Group1" -> cam_id = "1"
        cam_id = cam_folder.split("_")[0].replace("Cam", "")

        # New subfolder name: "cam1" instead of "Cam1_Group1"
        new_subfolder = f"cam{cam_id}"
        dst_dir = os.path.join(PREVIEW_DIR, new_subfolder)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(cam_path) if f.lower().endswith('.png')])
        
        print(f"[{cam_folder}] -> [{new_subfolder}] : {len(files)} images")
        if files:
            print(f"  Before: {files[0]}")

        for idx, old_name in enumerate(files, start=1):
            new_name = f"cam{cam_id}_p0_ok_{idx:05d}.png"
            src = os.path.join(cam_path, old_name)
            dst = os.path.join(dst_dir, new_name)
            shutil.copy2(src, dst)
            total += 1

        if files:
            print(f"  After:  cam{cam_id}_p0_ok_00001.png ~ cam{cam_id}_p0_ok_{len(files):05d}.png")

    print(f"\nDone! {total} files copied to: {PREVIEW_DIR}")

if __name__ == "__main__":
    rename_ok_p0()
