@echo off
REM Multi-GPU Training Script for Windows Server
REM Run this on the 4x 3090 server

set DATA_ROOT=./data/datasets
set OUTPUT_DIR=./experiments/baseline_vae
set EPOCHS=50
set BATCH_SIZE=32

echo Starting training for all 6 views...
echo This will train views sequentially. For parallel training, use run_all_views.sh on Linux.

echo.
echo ===== Training Cam1 =====
set CUDA_VISIBLE_DEVICES=0
python scripts/train_baseline.py --view Cam1 --data_root %DATA_ROOT% --output_dir %OUTPUT_DIR% --epochs %EPOCHS% --batch_size %BATCH_SIZE%

echo.
echo ===== Training Cam2 =====
set CUDA_VISIBLE_DEVICES=1
python scripts/train_baseline.py --view Cam2 --data_root %DATA_ROOT% --output_dir %OUTPUT_DIR% --epochs %EPOCHS% --batch_size %BATCH_SIZE%

echo.
echo ===== Training Cam3 =====
set CUDA_VISIBLE_DEVICES=2
python scripts/train_baseline.py --view Cam3 --data_root %DATA_ROOT% --output_dir %OUTPUT_DIR% --epochs %EPOCHS% --batch_size %BATCH_SIZE%

echo.
echo ===== Training Cam4 =====
set CUDA_VISIBLE_DEVICES=3
python scripts/train_baseline.py --view Cam4 --data_root %DATA_ROOT% --output_dir %OUTPUT_DIR% --epochs %EPOCHS% --batch_size %BATCH_SIZE%

echo.
echo ===== Training Cam5 =====
set CUDA_VISIBLE_DEVICES=0
python scripts/train_baseline.py --view Cam5 --data_root %DATA_ROOT% --output_dir %OUTPUT_DIR% --epochs %EPOCHS% --batch_size %BATCH_SIZE%

echo.
echo ===== Training Cam6 =====
set CUDA_VISIBLE_DEVICES=1
python scripts/train_baseline.py --view Cam6 --data_root %DATA_ROOT% --output_dir %OUTPUT_DIR% --epochs %EPOCHS% --batch_size %BATCH_SIZE%

echo.
echo All training completed!
pause
