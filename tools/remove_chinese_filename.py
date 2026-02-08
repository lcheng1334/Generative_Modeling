"""
文件名中文移除工具

功能：
- 读取文件夹内的所有图像
- 删除文件名中的中文字符
- 保留英文、数字、下划线等

使用方法：
    # 预览模式
    python tools/remove_chinese_filename.py --folder "path/to/images" --preview
    
    # 实际执行
    python tools/remove_chinese_filename.py --folder "path/to/images" --execute
"""

import os
import argparse
import re
from pathlib import Path
from tqdm import tqdm


def remove_chinese(text: str) -> str:
    """移除字符串中的中文字符"""
    # 匹配中文字符的正则表达式
    pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+')
    result = pattern.sub('', text)
    
    # 清理多余的下划线和连字符
    result = re.sub(r'_+', '_', result)  # 多个下划线变成一个
    result = re.sub(r'-+', '-', result)  # 多个连字符变成一个
    result = re.sub(r'^[_-]+', '', result)  # 开头的下划线/连字符
    result = re.sub(r'[_-]+$', '', result)  # 结尾的下划线/连字符（扩展名前）
    
    return result


def has_chinese(text: str) -> bool:
    """检查字符串是否包含中文"""
    pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
    return bool(pattern.search(text))


def process_folder(folder_path: str, preview: bool = True):
    """处理文件夹中的所有图像"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return
    
    # 查找所有图像
    images = []
    for ext in ['*.png', '*.jpg', '*.bmp', '*.PNG', '*.JPG', '*.BMP']:
        images.extend(list(folder.rglob(ext)))
    
    if not images:
        print("⚠️ 未找到任何图像")
        return
    
    print(f"找到 {len(images)} 张图像")
    
    if preview:
        print("\n🔍 预览模式 - 不会实际重命名\n")
    
    # 统计
    renamed_count = 0
    skipped_count = 0
    
    for img_path in tqdm(images, desc="处理图像", unit="张"):
        filename = img_path.name
        stem = img_path.stem
        suffix = img_path.suffix
        
        if not has_chinese(stem):
            skipped_count += 1
            continue
        
        # 移除中文
        new_stem = remove_chinese(stem)
        
        # 如果新文件名为空，使用原文件的hash
        if not new_stem or new_stem.strip() == '':
            import hashlib
            hash_str = hashlib.md5(filename.encode()).hexdigest()[:8]
            new_stem = f"img_{hash_str}"
        
        new_name = f"{new_stem}{suffix}"
        new_path = img_path.parent / new_name
        
        # 检查是否会重名
        if new_path.exists() and new_path != img_path:
            # 添加序号避免重名
            counter = 1
            while new_path.exists():
                new_stem_numbered = f"{new_stem}_{counter}"
                new_name = f"{new_stem_numbered}{suffix}"
                new_path = img_path.parent / new_name
                counter += 1
        
        tqdm.write(f"  {filename} → {new_name}")
        
        if not preview:
            try:
                img_path.rename(new_path)
            except Exception as e:
                tqdm.write(f"  ❌ 重命名失败: {e}")
                continue
        
        renamed_count += 1
    
    print(f"\n{'预览' if preview else '完成'}!")
    print(f"需要重命名: {renamed_count} 张")
    print(f"无需处理: {skipped_count} 张")


def main():
    parser = argparse.ArgumentParser(description="文件名中文移除工具")
    
    parser.add_argument("--folder", "-f", type=str, required=True,
                        help="图像文件夹路径")
    parser.add_argument("--preview", "-p", action="store_true",
                        help="预览模式，不实际重命名")
    parser.add_argument("--execute", "-e", action="store_true",
                        help="实际执行重命名")
    
    args = parser.parse_args()
    
    if not args.preview and not args.execute:
        print("❌ 请指定 --preview 或 --execute")
        return
    
    print("=" * 50)
    print("文件名中文移除工具")
    print("=" * 50)
    print(f"目标文件夹: {args.folder}")
    print()
    
    process_folder(args.folder, preview=not args.execute)


if __name__ == "__main__":
    main()
