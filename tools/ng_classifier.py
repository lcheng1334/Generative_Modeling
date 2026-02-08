"""
NG缺陷分类工具 - 8类版本

缺陷类型：
    1: 破损 (breakage)
    2: 粘连 (adhesion)
    3: 印反 (reversed_print) - 包括底面粘银
    4: 镀银超标 (silver_overflow)
    5: 露本体 (exposed_substrate)
    6: 扩散 (diffusion) - 包括印斜
    7: 脏污 (contamination) - 麻点+划痕
    8: 其他 (other)
    
导航：
    A/← : 上一张
    D/→ : 下一张
    S   : 跳过
    Q   : 退出
"""

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import os
import shutil
from pathlib import Path


class NGClassifier:
    """NG缺陷分类工具 - 8类版本"""
    
    # 缺陷类型映射：(文件夹名, 中文名, 颜色, 描述)
    DEFECT_TYPES = {
        '1': ('breakage', '破损', '#FF6B6B', '边角或表面破损'),
        '2': ('adhesion', '粘连', '#4ECDC4', '两个物料粘在一起'),
        '3': ('reversed_print', '印反', '#FFEAA7', '镀银位置翻转/底面粘银'),
        '4': ('silver_overflow', '镀银超标', '#DDA0DD', '镀银超出范围'),
        '5': ('exposed_substrate', '露本体', '#98D8C8', '镀银区域缺失'),
        '6': ('diffusion', '扩散', '#F7DC6F', '镀银扩散/印斜'),
        '7': ('contamination', '脏污', '#85C1E9', '麻点、划痕、污渍'),
        '8': ('other', '其他', '#AAB7B8', '其他无法归类'),
    }
    
    # 颜色主题
    BG_COLOR = '#1a1a2e'
    CARD_COLOR = '#16213e'
    TEXT_COLOR = '#eaeaea'
    HIGHLIGHT_COLOR = '#e94560'
    
    def __init__(self):
        self.source_dir = None
        self.target_dir = None
        self.images = []
        self.current_idx = 0
        self.root = None
        self.classified_count = 0
        
    def select_folders(self):
        """选择源文件夹和目标文件夹"""
        temp_root = tk.Tk()
        temp_root.withdraw()
        
        self.source_dir = filedialog.askdirectory(title="选择待分类的NG图像文件夹")
        if not self.source_dir:
            return False
            
        self.target_dir = filedialog.askdirectory(title="选择分类后存放的目标文件夹")
        if not self.target_dir:
            return False
            
        temp_root.destroy()
        
        # 创建所有缺陷类型子文件夹
        for key, (folder, name, color, desc) in self.DEFECT_TYPES.items():
            folder_path = Path(self.target_dir) / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            
        return True
    
    def scan_images(self):
        """扫描所有图像"""
        self.images = []
        source_path = Path(self.source_dir)
        
        for ext in ['*.png', '*.jpg', '*.bmp', '*.PNG', '*.JPG', '*.BMP']:
            self.images.extend(list(source_path.rglob(ext)))
        
        self.images = [str(p) for p in self.images]
        self.images.sort()
        print(f"找到 {len(self.images)} 张图像")
        
    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("🔍 NG缺陷分类工具 (8类)")
        self.root.configure(bg=self.BG_COLOR)
        self.root.state('zoomed')
        
        # 主容器
        main_container = tk.Frame(self.root, bg=self.BG_COLOR)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 左侧：图像显示
        left_frame = tk.Frame(main_container, bg=self.CARD_COLOR)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_title = tk.Label(
            left_frame, text="当前图像", 
            font=('Microsoft YaHei UI', 14, 'bold'),
            bg=self.CARD_COLOR, fg=self.TEXT_COLOR, pady=10
        )
        self.image_title.pack(fill=tk.X)
        
        self.image_label = tk.Label(left_frame, bg='#0d1117')
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.filename_label = tk.Label(
            left_frame, text="", font=('Consolas', 10),
            bg=self.CARD_COLOR, fg='#888888', pady=5
        )
        self.filename_label.pack(fill=tk.X)
        
        # 右侧：控制面板
        right_frame = tk.Frame(main_container, bg=self.BG_COLOR, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # 进度
        progress_card = tk.Frame(right_frame, bg=self.CARD_COLOR)
        progress_card.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(progress_card, text="📊 分类进度",
                 font=('Microsoft YaHei UI', 12, 'bold'),
                 bg=self.CARD_COLOR, fg=self.TEXT_COLOR, pady=8).pack()
        
        self.progress_label = tk.Label(
            progress_card, text="0 / 0",
            font=('Microsoft YaHei UI', 24, 'bold'),
            bg=self.CARD_COLOR, fg=self.HIGHLIGHT_COLOR
        )
        self.progress_label.pack()
        
        self.classified_label = tk.Label(
            progress_card, text="已分类: 0",
            font=('Microsoft YaHei UI', 10),
            bg=self.CARD_COLOR, fg='#888888', pady=5
        )
        self.classified_label.pack()
        
        # 快捷键
        shortcut_card = tk.Frame(right_frame, bg=self.CARD_COLOR)
        shortcut_card.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(shortcut_card, text="⌨️ 按键分类",
                 font=('Microsoft YaHei UI', 12, 'bold'),
                 bg=self.CARD_COLOR, fg=self.TEXT_COLOR, pady=8).pack()
        
        shortcuts_frame = tk.Frame(shortcut_card, bg=self.CARD_COLOR)
        shortcuts_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        for key in ['1', '2', '3', '4', '5', '6', '7', '8']:
            folder, name, color, desc = self.DEFECT_TYPES[key]
            
            item_frame = tk.Frame(shortcuts_frame, bg=self.CARD_COLOR)
            item_frame.pack(fill=tk.X, pady=3)
            
            tk.Label(item_frame, text=f" {key} ",
                     font=('Consolas', 12, 'bold'),
                     bg=color, fg='#000000', width=3).pack(side=tk.LEFT, padx=(0, 8))
            
            tk.Label(item_frame, text=name,
                     font=('Microsoft YaHei UI', 11, 'bold'),
                     bg=self.CARD_COLOR, fg=self.TEXT_COLOR, width=5, anchor='w').pack(side=tk.LEFT)
            
            tk.Label(item_frame, text=f"({desc})",
                     font=('Microsoft YaHei UI', 9),
                     bg=self.CARD_COLOR, fg='#888888', anchor='w').pack(side=tk.LEFT)
        
        # 导航
        nav_frame = tk.Frame(shortcut_card, bg=self.CARD_COLOR)
        nav_frame.pack(fill=tk.X, padx=10, pady=8)
        
        tk.Label(nav_frame,
                 text="A/← 上一张 | D/→ 下一张 | S 跳过 | Q 退出",
                 font=('Microsoft YaHei UI', 9),
                 bg=self.CARD_COLOR, fg='#888888').pack()
        
        # 状态
        status_card = tk.Frame(right_frame, bg=self.CARD_COLOR)
        status_card.pack(fill=tk.X)
        
        tk.Label(status_card, text="📝 最近操作",
                 font=('Microsoft YaHei UI', 12, 'bold'),
                 bg=self.CARD_COLOR, fg=self.TEXT_COLOR, pady=8).pack()
        
        self.status_label = tk.Label(
            status_card, text="等待分类...",
            font=('Microsoft YaHei UI', 11),
            bg=self.CARD_COLOR, fg=self.HIGHLIGHT_COLOR, pady=10
        )
        self.status_label.pack()
        
        self.root.bind('<Key>', self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
        
    def show_current_image(self):
        """显示当前图像"""
        if self.current_idx >= len(self.images):
            messagebox.showinfo("🎉 完成", f"所有图像已浏览完成！\n共分类: {self.classified_count} 张")
            self.on_quit()
            return
            
        img_path = self.images[self.current_idx]
        
        self.progress_label.config(text=f"{self.current_idx + 1} / {len(self.images)}")
        self.classified_label.config(text=f"已分类: {self.classified_count}")
        
        filename = Path(img_path).name
        short_name = filename[:40] + "..." if len(filename) > 40 else filename
        self.filename_label.config(text=filename)
        self.image_title.config(text=f"当前图像 - {short_name}")
        
        try:
            img = Image.open(img_path)
            
            display_width = 750
            display_height = 550
            
            img_ratio = img.width / img.height
            display_ratio = display_width / display_height
            
            if img_ratio > display_ratio:
                new_width = display_width
                new_height = int(display_width / img_ratio)
            else:
                new_height = display_height
                new_width = int(display_height * img_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo, text='')
            self.image_label.image = photo
            
        except Exception as e:
            self.image_label.config(image='', text=f"无法加载图像:\n{str(e)[:100]}")
            
    def move_to_category(self, category_folder, category_name):
        """移动当前图像"""
        if self.current_idx >= len(self.images):
            return
            
        img_path = self.images[self.current_idx]
        target_folder = Path(self.target_dir) / category_folder
        
        try:
            filename = Path(img_path).name
            target_path = target_folder / filename
            shutil.move(img_path, str(target_path))
            
            self.classified_count += 1
            self.status_label.config(text=f"✅ 分类为: {category_name}")
            
            self.images.pop(self.current_idx)
            
            if self.current_idx >= len(self.images):
                self.current_idx = max(0, len(self.images) - 1)
                
            self.show_current_image()
            
        except Exception as e:
            self.status_label.config(text=f"❌ 移动失败: {str(e)[:50]}")
            
    def on_key_press(self, event):
        """处理键盘"""
        key = event.keysym
        
        if key in ['1', '2', '3', '4', '5', '6', '7', '8']:
            folder, name, color, desc = self.DEFECT_TYPES[key]
            self.move_to_category(folder, name)
        elif key in ['a', 'A', 'Left']:
            if self.current_idx > 0:
                self.current_idx -= 1
                self.show_current_image()
                self.status_label.config(text="◀ 上一张")
        elif key in ['d', 'D', 'Right', 's', 'S']:
            if self.current_idx < len(self.images) - 1:
                self.current_idx += 1
                self.show_current_image()
                self.status_label.config(text="▶ 下一张" if key not in ['s', 'S'] else "⏭ 跳过")
        elif key in ['q', 'Q', 'Escape']:
            self.on_quit()
            
    def on_quit(self):
        """退出"""
        if messagebox.askyesno("退出", f"确定要退出吗？\n已分类: {self.classified_count} 张"):
            self.root.destroy()
            
    def run(self):
        """运行"""
        if not self.select_folders():
            return
            
        self.scan_images()
        
        if not self.images:
            messagebox.showinfo("提示", "未找到任何图像")
            return
            
        self.create_gui()
        self.show_current_image()
        self.root.mainloop()


if __name__ == '__main__':
    classifier = NGClassifier()
    classifier.run()
