# Multi-View Defect Synthesis

基于多视角一致性约束的工业缺陷合成系统

## 项目结构
```
├── src/                    # 核心代码
│   ├── models/             # 模型定义 (VAE, etc.)
│   ├── core/               # 训练器
│   ├── datasets/           # 数据加载器
│   └── utils/              # 工具函数
├── scripts/                # 训练脚本
│   ├── train_baseline.py   # 主训练脚本
│   ├── run_all_views.sh    # Linux并行训练
│   └── run_all_views.bat   # Windows训练
├── tools/                  # 辅助工具
└── configs/                # 配置文件
```

## 快速开始

### 1. 环境配置
```bash
# Linux服务器
chmod +x setup_server.sh
./setup_server.sh

# 或手动安装
conda create -n defect_gen python=3.10 -y
conda activate defect_gen
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_server.txt
```

### 2. 数据准备
将数据放在项目外部目录，修改训练脚本中的路径：
```bash
# 默认数据路径配置
--data_root /path/to/your/data/datasets
```

### 3. 训练
```bash
# 训练所有视角 (4 GPU并行)
./scripts/run_all_views.sh

# 或训练单个视角
CUDA_VISIBLE_DEVICES=0 python scripts/train_baseline.py --view Cam1 --epochs 50
```

### 4. 查看训练日志
```bash
tensorboard --logdir experiments/baseline_vae
```

## 硬件要求
- GPU: 至少1张RTX 3060 或更高
- 推荐: 4x RTX 3090 (并行训练)
- 显存: 每视角约4-6GB

## 技术路线
1. **Phase 1**: 数据准备 ✅
2. **Phase 2**: 基线模型训练 (VAE) ⏳
3. **Phase 3**: 多视角一致性合成
4. **Phase 4**: 实验验证
5. **Phase 5**: 论文撰写
