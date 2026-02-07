#!/bin/bash
# Multi-GPU Training Script for 4x 3090 Server
# Each GPU trains one view in parallel

# Data path - ADJUST THIS TO YOUR SERVER PATH
DATA_ROOT="../dataset/Generative_Modeling/data/datasets"
OUTPUT_DIR="./experiments/baseline_vae"
EPOCHS=50
BATCH_SIZE=32

echo "Starting parallel training on 4 GPUs..."

# Round 1: Train Cam1-4 in parallel
CUDA_VISIBLE_DEVICES=0 python scripts/train_baseline.py --view Cam1 --data_root $DATA_ROOT --output_dir $OUTPUT_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE &
CUDA_VISIBLE_DEVICES=1 python scripts/train_baseline.py --view Cam2 --data_root $DATA_ROOT --output_dir $OUTPUT_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE &
CUDA_VISIBLE_DEVICES=2 python scripts/train_baseline.py --view Cam3 --data_root $DATA_ROOT --output_dir $OUTPUT_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE &
CUDA_VISIBLE_DEVICES=3 python scripts/train_baseline.py --view Cam4 --data_root $DATA_ROOT --output_dir $OUTPUT_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE &

# Wait for round 1 to complete
wait

echo "Round 1 complete (Cam1-4). Starting Round 2 (Cam5-6)..."

# Round 2: Train Cam5-6
CUDA_VISIBLE_DEVICES=0 python scripts/train_baseline.py --view Cam5 --data_root $DATA_ROOT --output_dir $OUTPUT_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE &
CUDA_VISIBLE_DEVICES=1 python scripts/train_baseline.py --view Cam6 --data_root $DATA_ROOT --output_dir $OUTPUT_DIR --epochs $EPOCHS --batch_size $BATCH_SIZE &

wait

echo "All 6 views training completed!"
