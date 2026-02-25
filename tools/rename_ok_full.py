"""
Full OK folder rename script
- Renames OK/0  -> OK/p0/cam{X}/cam{X}_p0_ok_{index}.png
- Renames OK/90 -> OK/p90/cam{X}/cam{X}_p90_ok_{index}.png
- Deletes old folders (0, 90) and preview folder after completion
"""
import os
import shutil

BASE_DIR = r"E:\code\dataset\Generative_Modeling\data\datasets\OK"

POSE_MAP = {
    "0\u00b0": "p0",    # 0° -> p0
    "90\u00b0": "p90",  # 90° -> p90
}

def rename_all_ok():
    total = 0

    for old_pose_name, new_pose_name in POSE_MAP.items():
        src_pose_dir = os.path.join(BASE_DIR, old_pose_name)
        if not os.path.exists(src_pose_dir):
            print(f"[SKIP] {src_pose_dir} not found")
            continue

        dst_pose_dir = os.path.join(BASE_DIR, new_pose_name)
        print(f"\n{'='*50}")
        print(f"Processing: {old_pose_name} -> {new_pose_name}")
        print(f"{'='*50}")

        for cam_folder in sorted(os.listdir(src_pose_dir)):
            cam_path = os.path.join(src_pose_dir, cam_folder)
            if not os.path.isdir(cam_path):
                continue
            if "renamed" in cam_folder or "preview" in cam_folder:
                continue

            # "Cam1_Group1" or "Cam1_Group2" -> "1"
            cam_id = cam_folder.split("_")[0].replace("Cam", "")
            new_cam_folder = f"cam{cam_id}"
            dst_cam_dir = os.path.join(dst_pose_dir, new_cam_folder)
            os.makedirs(dst_cam_dir, exist_ok=True)

            files = sorted([f for f in os.listdir(cam_path) if f.lower().endswith('.png')])
            count = len(files)

            for idx, old_name in enumerate(files, start=1):
                new_name = f"cam{cam_id}_{new_pose_name}_ok_{idx:05d}.png"
                src = os.path.join(cam_path, old_name)
                dst = os.path.join(dst_cam_dir, new_name)
                shutil.copy2(src, dst)
                total += 1

            print(f"  {cam_folder} -> {new_cam_folder}/ : {count} files")
            if count > 0:
                print(f"    cam{cam_id}_{new_pose_name}_ok_00001.png ~ cam{cam_id}_{new_pose_name}_ok_{count:05d}.png")

    print(f"\n{'='*50}")
    print(f"Total: {total} files renamed")
    print(f"{'='*50}")

    # Delete old folders
    print("\nDeleting old folders...")
    for old_pose_name in POSE_MAP.keys():
        old_dir = os.path.join(BASE_DIR, old_pose_name)
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)
            print(f"  Deleted: {old_dir}")

    # Delete preview folders if they exist
    for name in os.listdir(BASE_DIR):
        if "preview" in name or "renamed" in name.lower():
            preview_path = os.path.join(BASE_DIR, name)
            if os.path.isdir(preview_path):
                shutil.rmtree(preview_path)
                print(f"  Deleted preview: {preview_path}")

    print("\nFinal structure:")
    for item in sorted(os.listdir(BASE_DIR)):
        item_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(item_path):
            sub_count = sum(len(fs) for _, _, fs in os.walk(item_path))
            print(f"  {item}/ ({sub_count} files)")
            for sub in sorted(os.listdir(item_path)):
                sub_path = os.path.join(item_path, sub)
                if os.path.isdir(sub_path):
                    file_count = len(os.listdir(sub_path))
                    print(f"    {sub}/ ({file_count} files)")

    print("\n[DONE] OK folder rename complete!")

if __name__ == "__main__":
    rename_all_ok()
