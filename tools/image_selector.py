"""
图像筛选工具 - GUI版本
- 启动后显示界面，可选择文件夹
- 创建对应的 _demo 文件夹
- A键：上一张图
- D键：下一张图
- 空格键：移动当前图像到 _demo 文件夹
- Q键或ESC：退出
"""

import os
import shutil
from pathlib import Path
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox


class ImageSelectorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("图像筛选工具")
        self.root.geometry("500x300")
        self.root.resizable(False, False)
        
        # 设置样式
        self.root.configure(bg='#2b2b2b')
        
        self.folder_path = tk.StringVar()
        
        self.setup_ui()
        
    def setup_ui(self):
        # 标题
        title_label = tk.Label(
            self.root, 
            text="图像筛选工具", 
            font=('Microsoft YaHei', 18, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # 说明
        info_text = """选择一个Cam文件夹，工具会自动创建对应的_demo文件夹
浏览图像时：
  A键：上一张  |  D键：下一张
  空格键：移动到_demo  |  Q键：退出"""
        
        info_label = tk.Label(
            self.root, 
            text=info_text, 
            font=('Microsoft YaHei', 10),
            bg='#2b2b2b',
            fg='#cccccc',
            justify='left'
        )
        info_label.pack(pady=10)
        
        # 文件夹选择框
        folder_frame = tk.Frame(self.root, bg='#2b2b2b')
        folder_frame.pack(pady=20, padx=20, fill='x')
        
        self.folder_entry = tk.Entry(
            folder_frame, 
            textvariable=self.folder_path,
            font=('Consolas', 10),
            width=40
        )
        self.folder_entry.pack(side='left', padx=(0, 10))
        
        browse_btn = tk.Button(
            folder_frame, 
            text="浏览...", 
            command=self.browse_folder,
            font=('Microsoft YaHei', 10),
            bg='#404040',
            fg='white',
            width=8
        )
        browse_btn.pack(side='left')
        
        # 开始按钮
        start_btn = tk.Button(
            self.root, 
            text="开始筛选", 
            command=self.start_selection,
            font=('Microsoft YaHei', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            width=15,
            height=2
        )
        start_btn.pack(pady=20)
        
        # 状态标签
        self.status_label = tk.Label(
            self.root, 
            text="", 
            font=('Microsoft YaHei', 9),
            bg='#2b2b2b',
            fg='#888888'
        )
        self.status_label.pack(pady=5)
        
    def browse_folder(self):
        # 设置默认路径
        initial_dir = r"E:\code\Generative_Modeling\data\datasets\OK"
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()
            
        folder = filedialog.askdirectory(
            title="选择Cam文件夹",
            initialdir=initial_dir
        )
        
        if folder:
            self.folder_path.set(folder)
            # 统计文件数量
            count = len([f for f in Path(folder).iterdir() 
                        if f.is_file() and f.suffix.lower() in {'.bmp', '.jpg', '.jpeg', '.png'}])
            self.status_label.config(text=f"找到 {count} 张图像")
            
    def start_selection(self):
        folder = self.folder_path.get().strip()
        
        if not folder:
            messagebox.showwarning("警告", "请先选择文件夹！")
            return
            
        if not os.path.exists(folder):
            messagebox.showerror("错误", f"文件夹不存在：{folder}")
            return
        
        # 隐藏主窗口
        self.root.withdraw()
        
        # 运行图像选择器
        try:
            self.image_selector(folder)
        finally:
            # 恢复主窗口
            self.root.deiconify()
            
    def image_selector(self, source_folder: str):
        """图像筛选核心功能"""
        source_path = Path(source_folder)
        
        # 创建 _demo 文件夹
        demo_folder = source_path.parent / f"{source_path.name}_demo"
        demo_folder.mkdir(exist_ok=True)
        print(f"目标文件夹: {demo_folder}")
        
        # 获取所有图像文件
        image_extensions = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        images = sorted([f for f in source_path.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions])
        
        if not images:
            messagebox.showinfo("提示", "文件夹中没有图像文件")
            return
        
        print(f"找到 {len(images)} 张图像")
        
        current_index = 0
        moved_count = 0
        
        # 创建窗口
        window_name = f"Image Selector - {source_path.name} (A:Prev D:Next SPACE:Move Q:Quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 900)
        
        while True:
            # 检查是否还有图像
            if not images:
                messagebox.showinfo("完成", f"所有图像已处理完毕！\n已移动: {moved_count} 张")
                break
            
            # 确保索引在有效范围内
            current_index = max(0, min(current_index, len(images) - 1))
            
            # 读取当前图像
            current_image_path = images[current_index]
            img = cv2.imread(str(current_image_path))
            
            if img is None:
                print(f"无法读取图像: {current_image_path.name}")
                current_index += 1
                continue
            
            # 创建带信息的图像副本
            display_img = img.copy()
            
            # 添加顶部信息条
            cv2.rectangle(display_img, (0, 0), (display_img.shape[1], 50), (40, 40, 40), -1)
            
            # 进度信息
            progress_text = f"[{current_index + 1}/{len(images)}]  Moved: {moved_count}"
            cv2.putText(display_img, progress_text, (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 文件名（右侧）
            name_text = current_image_path.name
            text_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(display_img, name_text, 
                        (display_img.shape[1] - text_size[0] - 10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 底部操作提示
            cv2.rectangle(display_img, (0, display_img.shape[0] - 35), 
                         (display_img.shape[1], display_img.shape[0]), (40, 40, 40), -1)
            hint_text = "A: Prev  |  D: Next  |  SPACE: Move to Demo  |  Q: Quit"
            cv2.putText(display_img, hint_text, (10, display_img.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            
            # 显示图像
            cv2.imshow(window_name, display_img)
            
            # 等待按键
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('a') or key == ord('A'):
                # 上一张
                if current_index > 0:
                    current_index -= 1
                    
            elif key == ord('d') or key == ord('D'):
                # 下一张
                if current_index < len(images) - 1:
                    current_index += 1
                    
            elif key == ord(' '):
                # 空格键：移动图像
                src_file = current_image_path
                dst_file = demo_folder / current_image_path.name
                
                try:
                    shutil.move(str(src_file), str(dst_file))
                    print(f"已移动: {current_image_path.name}")
                    moved_count += 1
                    
                    # 从列表中移除
                    images.pop(current_index)
                    
                    # 调整索引
                    if current_index >= len(images):
                        current_index = len(images) - 1
                        
                except Exception as e:
                    print(f"移动失败: {e}")
                    
            elif key == ord('q') or key == ord('Q') or key == 27:
                # 退出
                break
        
        cv2.destroyAllWindows()
        
        # 更新状态
        self.status_label.config(
            text=f"完成！已移动 {moved_count} 张图像到 {demo_folder.name}"
        )
        
        print(f"\n筛选完成！已移动: {moved_count} 张图像")
        print(f"目标文件夹: {demo_folder}")
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ImageSelectorApp()
    app.run()
