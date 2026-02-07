"""
验证环境安装
Verify environment installation
"""
import sys

def check_installation():
    """检查关键包的安装情况"""
    
    print("=" * 60)
    print("环境安装验证")
    print("=" * 60)
    print()
    
    results = {}
    
    # 1. PyTorch
    print("1. 检查 PyTorch...")
    try:
        import torch
        results['PyTorch'] = True
        print(f"   ✓ PyTorch 版本: {torch.__version__}")
        print(f"   ✓ CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA 版本: {torch.version.cuda}")
            print(f"   ✓ GPU 设备: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        results['PyTorch'] = False
        print(f"   ✗ PyTorch 安装失败: {e}")
    print()
    
    # 2. PyTorch3D
    print("2. 检查 PyTorch3D...")
    try:
        import pytorch3d
        results['PyTorch3D'] = True
        print(f"   ✓ PyTorch3D 版本: {pytorch3d.__version__}")
    except Exception as e:
        results['PyTorch3D'] = False
        print(f"   ⚠ PyTorch3D 未安装（可选，可后续手动安装）")
        print(f"   提示: 需要时可运行 'pip install fvcore iopath pytorch3d'")
    print()
    
    # 3. OpenCV
    print("3. 检查 OpenCV...")
    try:
        import cv2
        results['OpenCV'] = True
        print(f"   ✓ OpenCV 版本: {cv2.__version__}")
    except Exception as e:
        results['OpenCV'] = False
        print(f"   ✗ OpenCV 安装失败: {e}")
    print()
    
    # 4. Kornia
    print("4. 检查 Kornia...")
    try:
        import kornia
        results['Kornia'] = True
        print(f"   ✓ Kornia 版本: {kornia.__version__}")
    except Exception as e:
        results['Kornia'] = False
        print(f"   ✗ Kornia 安装失败: {e}")
    print()
    
    # 5. Diffusers
    print("5. 检查 Diffusers...")
    try:
        import diffusers
        results['Diffusers'] = True
        print(f"   ✓ Diffusers 版本: {diffusers.__version__}")
    except Exception as e:
        results['Diffusers'] = False
        print(f"   ✗ Diffusers 安装失败: {e}")
    print()
    
    # 6. Transformers
    print("6. 检查 Transformers...")
    try:
        import transformers
        results['Transformers'] = True
        print(f"   ✓ Transformers 版本: {transformers.__version__}")
    except Exception as e:
        results['Transformers'] = False
        print(f"   ✗ Transformers 安装失败: {e}")
    print()
    
    # 7. Ultralytics (YOLO)
    print("7. 检查 Ultralytics YOLO...")
    try:
        import ultralytics
        results['Ultralytics'] = True
        print(f"   ✓ Ultralytics 版本: {ultralytics.__version__}")
    except Exception as e:
        results['Ultralytics'] = False
        print(f"   ✗ Ultralytics 安装失败: {e}")
    print()
    
    # 8. 其他工具
    print("8. 检查其他工具...")
    try:
        import numpy as np
        import PIL
        from loguru import logger
        results['Others'] = True
        print(f"   ✓ NumPy 版本: {np.__version__}")
        print(f"   ✓ Pillow 版本: {PIL.__version__}")
        print(f"   ✓ Loguru: 已安装")
    except Exception as e:
        results['Others'] = False
        print(f"   ✗ 部分工具安装失败: {e}")
    print()
    
    # 总结
    print("=" * 60)
    print("安装总结")
    print("=" * 60)
    
    total = len(results)
    success = sum(results.values())
    
    print(f"✓ 成功安装: {success}/{total}")
    print(f"✗ 失败/未安装: {total - success}/{total}")
    print()
    
    if results.get('PyTorch') and results.get('PyTorch') and torch.cuda.is_available():
        print("🎉 核心组件已就绪！可以开始开发。")
    else:
        print("⚠ 部分核心组件缺失，请检查安装。")
    
    if not results.get('PyTorch3D'):
        print()
        print("💡 关于 PyTorch3D:")
        print("   PyTorch3D 可以稍后安装，不影响大部分功能。")
        print("   如需安装，可以尝试:")
        print("   pip install fvcore iopath")
        print("   pip install 'pytorch3d @ https://github.com/facebookresearch/pytorch3d/archive/refs/heads/main.zip'")


if __name__ == "__main__":
    check_installation()
