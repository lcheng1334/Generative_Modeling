# 依赖安装指南（CUDA 13.0版本）

## 当前安装进度

### ✅ 已完成
- [x] 创建conda环境 `Generate` (Python 3.10)
- [/] 正在安装 PyTorch 2.5.1 + CUDA 12.1（下载中，约1.21GB）

### 📦 待安装
- [ ] PyTorch3D
- [ ] 其他Python包（requirements.txt）

---

## 自动安装（推荐）

当前PyTorch正在后台自动安装。完成后会继续安装其余依赖。

**预计总时间**：15-30分钟（取决于网速）

---

## 手动安装（如遇问题）

如果自动安装遇到问题，可以手动执行以下命令：

### 1. 激活环境
```bash
conda activate Generate
```

### 2. 安装PyTorch（CUDA 12.1，兼容CUDA 13.0）
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. 安装PyTorch3D
```bash
# 方法1：使用conda（推荐）
conda install pytorch3d -c pytorch3d -y

# 方法2：如果conda失败，使用pip从源码安装
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 4. 安装其他依赖
```bash
pip install -r requirements.txt
```

---

## 验证安装

安装完成后，运行以下命令验证：

```python
# 启动Python
python

# 检查PyTorch
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU设备: {torch.cuda.get_device_name(0)}")

# 检查PyTorch3D
import pytorch3d
print(f"PyTorch3D版本: {pytorch3d.__version__}")
```

**预期输出**：
```
PyTorch版本: 2.5.1
CUDA可用: True
CUDA版本: 12.1
GPU设备: NVIDIA xxxx
PyTorch3D版本: 0.7.x
```

---

## 常见问题

### Q1: PyTorch3D conda安装失败
**解决**：使用pip从源码安装
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Q2: CUDA版本不匹配警告
**说明**：CUDA 12.1可以兼容CUDA 13.0，这是正常的。PyTorch会向下兼容。

### Q3: 显存不足
**解决**：
- 训练时减小batch_size
- 使用mixed_precision="fp16"
- 如果显存<8GB，考虑在服务器上训练

---

## 下一步

安装完成后，运行：

```bash
# 测试工字型Proxy生成
python src/core/geometry/proxy_generator.py

# 测试Blinn-Phong渲染
python src/core/rendering/blinn_phong.py

# 可视化样本数据
python tools/visualize_samples.py --data_dir data/samples/inductor
```
