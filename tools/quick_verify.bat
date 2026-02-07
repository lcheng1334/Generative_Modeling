@echo off
chcp 65001 > nul
echo ========================================
echo 环境验证脚本
echo ========================================
echo.

call conda activate Generate
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 无法激活Generate环境
    pause
    exit /b 1
)

echo [信息] 已激活Generate环境
echo.

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
if %ERRORLEVEL% NEQ 0 (
    echo [错误] PyTorch未安装或有问题
    pause
    exit /b 1
)

echo.
echo ========================================
echo 环境验证成功！
echo ========================================
pause
