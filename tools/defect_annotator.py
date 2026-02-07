"""
电感缺陷分类标注工具

功能：
- 6视角联合显示
- 键盘快捷键快速分类
- 自动保存标注结果
- 支持中断续标

使用方法：
    python tools/defect_annotator.py --data_dir data/datasets/CM_New_3010-6不良品
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import json
import glob
from pathlib import Path
from datetime import datetime

class DefectAnnotator:
    """缺陷分类标注工具"""
    
    # 7类缺陷及其快捷键
    DEFECT_TYPES = {
        '1': '破损',
        '2': '镀银超标', 
        '3': '扩散',
        '4': '印反',
        '5': '露本体',
        '6': '麻点',
        '7': '划痕',
        '0': '其他/跳过',
    }
    
    # 相机视角说明
    CAM_LABELS = {
        'Cam1': '正面',
        'Cam2': '底面',
        'Cam3': '下侧面',
        'Cam4': '右侧面',
        'Cam5': '上侧面',
        'Cam6': '左侧面',
    }
    
    def __init__(self, data_dir: str, output_file: str = None):
        self.data_dir = Path(data_dir)
        self.output_file = output_file or self.data_dir / 'annotations.json'
        
        # 加载已有标注
        self.annotations = self._load_annotations()
        
        # 扫描所有图像
        self.samples = self._scan_samples()
        self.current_idx = self._find_start_index()
        
        # 创建GUI
        self._create_gui()
        
    def _load_annotations(self) -> dict:
        """加载已有标注"""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_annotations(self):
        """保存标注结果"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
    
    def _scan_samples(self) -> list:
        """扫描所有样本（按产品ID分组）"""
        # 查找所有Cam1图像作为基准
        cam1_dir = self.data_dir / self.data_dir.name / 'Cam1'
        if not cam1_dir.exists():
            # 尝试其他路径结构
            cam1_dir = self.data_dir / 'Cam1'
        
        if not cam1_dir.exists():
            raise ValueError(f"找不到Cam1目录: {cam1_dir}")
        
        # 获取所有图像
        images = list(cam1_dir.glob('*.bmp')) + list(cam1_dir.glob('*.png'))
        
        samples = []
        for img_path in sorted(images):
            # 提取产品ID（文件名的最后部分，去除扩展名）
            filename = img_path.stem
            product_id = filename.split('_')[-1]
            
            # 查找6个视角的图像
            sample = {
                'product_id': product_id,
                'filename': filename,
                'images': {}
            }
            
            for cam in self.CAM_LABELS.keys():
                cam_dir = img_path.parent.parent / cam
                # 查找匹配的图像
                for ext in ['*.bmp', '*.png']:
                    matching = list(cam_dir.glob(f'*_{product_id}{ext[1:]}'))
                    if matching:
                        sample['images'][cam] = str(matching[0])
                        break
            
            if len(sample['images']) > 0:
                samples.append(sample)
        
        return samples
    
    def _find_start_index(self) -> int:
        """找到第一个未标注的样本"""
        for i, sample in enumerate(self.samples):
            if sample['product_id'] not in self.annotations:
                return i
        return 0
    
    def _create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title('电感缺陷分类标注工具')
        self.root.geometry('1400x900')
        self.root.configure(bg='#1a1a2e')
        
        # 顶部信息栏
        info_frame = tk.Frame(self.root, bg='#16213e', height=60)
        info_frame.pack(fill='x', padx=10, pady=5)
        info_frame.pack_propagate(False)
        
        self.progress_label = tk.Label(
            info_frame, 
            text='', 
            font=('微软雅黑', 14),
            fg='#e94560',
            bg='#16213e'
        )
        self.progress_label.pack(side='left', padx=20, pady=15)
        
        self.filename_label = tk.Label(
            info_frame,
            text='',
            font=('Consolas', 11),
            fg='#0f4c75',
            bg='#16213e'
        )
        self.filename_label.pack(side='right', padx=20, pady=15)
        
        # 6视角图像显示区
        self.image_frame = tk.Frame(self.root, bg='#1a1a2e')
        self.image_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.image_labels = {}
        self.cam_title_labels = {}
        
        # 2行3列布局
        for i, cam in enumerate(self.CAM_LABELS.keys()):
            row, col = i // 3, i % 3
            
            container = tk.Frame(self.image_frame, bg='#0f3460', bd=2)
            container.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            # 相机标题
            title = tk.Label(
                container,
                text=f'{cam} - {self.CAM_LABELS[cam]}',
                font=('微软雅黑', 11, 'bold'),
                fg='#e94560',
                bg='#0f3460'
            )
            title.pack(pady=5)
            self.cam_title_labels[cam] = title
            
            # 图像显示
            img_label = tk.Label(container, bg='#1a1a2e')
            img_label.pack(padx=5, pady=5)
            self.image_labels[cam] = img_label
        
        # 配置grid权重
        for i in range(2):
            self.image_frame.grid_rowconfigure(i, weight=1)
        for i in range(3):
            self.image_frame.grid_columnconfigure(i, weight=1)
        
        # 底部缺陷类型按钮和说明
        bottom_frame = tk.Frame(self.root, bg='#16213e', height=120)
        bottom_frame.pack(fill='x', padx=10, pady=5)
        bottom_frame.pack_propagate(False)
        
        # 缺陷类型按钮
        btn_frame = tk.Frame(bottom_frame, bg='#16213e')
        btn_frame.pack(pady=10)
        
        self.defect_buttons = {}
        for key, defect_name in self.DEFECT_TYPES.items():
            btn = tk.Button(
                btn_frame,
                text=f'[{key}] {defect_name}',
                font=('微软雅黑', 12),
                width=12,
                height=2,
                bg='#0f3460',
                fg='white',
                activebackground='#e94560',
                command=lambda k=key: self._on_classify(k)
            )
            btn.pack(side='left', padx=5)
            self.defect_buttons[key] = btn
        
        # 操作说明
        help_label = tk.Label(
            bottom_frame,
            text='快捷键: 1-7选择缺陷类型, 0跳过, ←/→ 前后切换, Ctrl+S 保存, Q 退出',
            font=('微软雅黑', 10),
            fg='#7f8c8d',
            bg='#16213e'
        )
        help_label.pack(pady=5)
        
        # 绑定键盘事件
        self.root.bind('<Key>', self._on_key_press)
        self.root.bind('<Left>', lambda e: self._navigate(-1))
        self.root.bind('<Right>', lambda e: self._navigate(1))
        self.root.bind('<Control-s>', lambda e: self._save_annotations())
        
        # 显示第一个样本
        self._show_current_sample()
        
    def _show_current_sample(self):
        """显示当前样本"""
        if not self.samples:
            messagebox.showinfo('提示', '没有找到待标注的样本')
            return
        
        sample = self.samples[self.current_idx]
        
        # 更新进度
        annotated = len(self.annotations)
        total = len(self.samples)
        self.progress_label.config(
            text=f'进度: {annotated}/{total} ({annotated*100//total}%) | 当前: {self.current_idx+1}'
        )
        self.filename_label.config(text=sample['filename'])
        
        # 显示6个视角图像
        for cam in self.CAM_LABELS.keys():
            if cam in sample['images']:
                try:
                    img = Image.open(sample['images'][cam])
                    # 调整大小以适应显示
                    img.thumbnail((400, 280), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.image_labels[cam].config(image=photo)
                    self.image_labels[cam].image = photo
                except Exception as e:
                    self.image_labels[cam].config(image='', text=f'加载失败\n{e}')
            else:
                self.image_labels[cam].config(image='', text='无图像')
        
        # 高亮已标注的类型
        if sample['product_id'] in self.annotations:
            annotated_type = self.annotations[sample['product_id']]['defect_key']
            for key, btn in self.defect_buttons.items():
                if key == annotated_type:
                    btn.config(bg='#e94560')
                else:
                    btn.config(bg='#0f3460')
        else:
            for btn in self.defect_buttons.values():
                btn.config(bg='#0f3460')
    
    def _on_classify(self, defect_key: str):
        """分类当前样本"""
        sample = self.samples[self.current_idx]
        
        self.annotations[sample['product_id']] = {
            'filename': sample['filename'],
            'defect_key': defect_key,
            'defect_name': self.DEFECT_TYPES[defect_key],
            'timestamp': datetime.now().isoformat(),
        }
        
        # 自动保存
        self._save_annotations()
        
        # 跳到下一个
        self._navigate(1)
    
    def _on_key_press(self, event):
        """处理键盘事件"""
        if event.char in self.DEFECT_TYPES:
            self._on_classify(event.char)
        elif event.char.lower() == 'q':
            self._on_quit()
    
    def _navigate(self, delta: int):
        """导航到其他样本"""
        new_idx = self.current_idx + delta
        if 0 <= new_idx < len(self.samples):
            self.current_idx = new_idx
            self._show_current_sample()
    
    def _on_quit(self):
        """退出程序"""
        self._save_annotations()
        messagebox.showinfo('保存成功', f'已保存 {len(self.annotations)} 条标注到:\n{self.output_file}')
        self.root.quit()
    
    def run(self):
        """运行标注工具"""
        self.root.mainloop()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='电感缺陷分类标注工具')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='不良品数据目录')
    parser.add_argument('--output', type=str, default=None,
                        help='标注结果输出文件 (JSON格式)')
    
    args = parser.parse_args()
    
    print('='*50)
    print('电感缺陷分类标注工具')
    print('='*50)
    print(f'数据目录: {args.data_dir}')
    print()
    print('缺陷类型:')
    for key, name in DefectAnnotator.DEFECT_TYPES.items():
        print(f'  [{key}] {name}')
    print()
    print('快捷键:')
    print('  1-7: 选择缺陷类型')
    print('  0: 其他/跳过')
    print('  ←/→: 前后切换样本')
    print('  Ctrl+S: 手动保存')
    print('  Q: 退出')
    print('='*50)
    
    annotator = DefectAnnotator(args.data_dir, args.output)
    print(f'找到 {len(annotator.samples)} 个待标注样本')
    print(f'已标注 {len(annotator.annotations)} 个')
    print()
    
    annotator.run()


if __name__ == '__main__':
    main()
