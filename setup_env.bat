@echo off
REM 快速环境搭建脚本
echo ========================================
echo 创建虚拟环境: Generate
echo ========================================

REM 创建conda环境
conda create -n Generate python=3.10 -y

REM 激活环境
call conda activate Generate

REM 安装PyTorch (CUDA 12.1版本，兼容CUDA 13.0)
echo.
echo ========================================
echo 安装PyTorch及CUDA支持 (CUDA 12.1)
echo ========================================
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

REM 安装PyTorch3D
echo.
echo ========================================
echo 安装PyTorch3D
echo ========================================
conda install pytorch3d -c pytorch3d -y

REM 安装其他依赖
echo.
echo ========================================
echo 安装其他Python包
echo ========================================
pip install -r requirements.txt

echo.
echo ========================================
echo 环境创建完成！
echo 使用以下命令激活环境:
echo conda activate Generate
echo ========================================
pause
