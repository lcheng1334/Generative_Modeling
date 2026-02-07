"""
重命名OK文件夹中的图像文件
1. 在文件名前加上文件夹名称（Cam1_, Cam2_等）
2. 把"良品"替换成"OK"

例如：
Cam1/良品_正面良品_1150511.bmp -> Cam1/Cam1_OK_正面OK_1150511.bmp
"""

import os
from pathlib import Path


def rename_ok_images(ok_folder: str, dry_run: bool = True):
    """
    重命名OK文件夹中的图像
    
    Args:
        ok_folder: OK文件夹路径
        dry_run: 如果为True，只打印将要执行的操作，不实际重命名
    """
    ok_path = Path(ok_folder)
    
    if not ok_path.exists():
        print(f"错误：文件夹不存在 {ok_folder}")
        return
    
    total_files = 0
    renamed_files = 0
    
    # 遍历Cam1-Cam6文件夹
    for cam_folder in sorted(ok_path.iterdir()):
        if not cam_folder.is_dir():
            continue
        
        cam_name = cam_folder.name  # 如 "Cam1"
        print(f"\n处理文件夹: {cam_name}")
        
        # 遍历该文件夹中的所有文件
        for file_path in sorted(cam_folder.iterdir()):
            if not file_path.is_file():
                continue
            
            total_files += 1
            old_name = file_path.name
            
            # 构建新文件名
            # 1. 把"良品"替换成"OK"
            new_name = old_name.replace("良品", "OK")
            # 2. 删除"正面"
            new_name = new_name.replace("正面", "")
            
            # 2. 在前面加上文件夹名称（如果还没有的话）
            if not new_name.startswith(cam_name + "_"):
                new_name = f"{cam_name}_{new_name}"
            
            # 如果文件名有变化
            if new_name != old_name:
                new_path = file_path.parent / new_name
                
                if dry_run:
                    print(f"  [预览] {old_name}")
                    print(f"      -> {new_name}")
                else:
                    # 检查目标文件是否已存在
                    if new_path.exists():
                        print(f"  [跳过] {new_name} 已存在")
                        continue
                    
                    file_path.rename(new_path)
                    print(f"  [重命名] {old_name} -> {new_name}")
                
                renamed_files += 1
    
    print(f"\n{'='*50}")
    print(f"总文件数: {total_files}")
    print(f"需要重命名: {renamed_files}")
    
    if dry_run:
        print("\n这是预览模式，没有实际修改文件。")
        print("确认无误后，请运行: python rename_ok_images.py --execute")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="重命名OK文件夹中的图像")
    parser.add_argument("--execute", action="store_true", 
                        help="实际执行重命名（默认只预览）")
    parser.add_argument("--folder", type=str, 
                        default=r"E:\code\Generative_Modeling\data\datasets\OK",
                        help="OK文件夹路径")
    
    args = parser.parse_args()
    
    print("="*50)
    print("OK图像重命名工具")
    print("="*50)
    print(f"目标文件夹: {args.folder}")
    print(f"模式: {'执行' if args.execute else '预览'}")
    print("="*50)
    
    rename_ok_images(args.folder, dry_run=not args.execute)
