#!/bin/bash
# Description: Enhanced memory-optimized video generation using HunyuanVideo model

# Disable MPS memory limits for maximum memory utilization
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0

# Other optimizations
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection
export MPS_USE_GUARD_MODE=1
export MPS_ENABLE_MEMORY_GUARD=1
export PYTORCH_MPS_SYNC_OPERATIONS=1
export PYTORCH_MPS_AGGRESSIVE_MEMORY_CLEANUP=1

# Function to clear system caches
clear_system_caches() {
    echo "Clearing system caches..."
    sudo purge
    sleep 2
}

# Function to check available memory
check_memory() {
    free_memory=$(vm_stat | awk '/free/ {print $3}' | sed 's/\.//')
    echo "Available memory pages: $free_memory"
}

echo "Stage 1: Initial system cleanup"
clear_system_caches
check_memory

echo "Stage 2: Starting with minimal settings"
# Start with very conservative settings
python3 sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 256 256 \
    --video-length 5 \
    --infer-steps 15 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --vae-tiling \
    --precision fp16 \
    --vae-precision fp16 \
    --text-encoder-precision fp16 \
    --text-encoder-precision-2 fp16 \
    --save-path ./results

# Clear memory between stages
clear_system_caches
check_memory

echo "Stage 3: Attempting medium settings"
python3 sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 544 960 \
    --video-length 13 \
    --infer-steps 20 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --vae-tiling \
    --precision fp16 \
    --vae-precision fp16 \
    --text-encoder-precision fp16 \
    --text-encoder-precision-2 fp16 \
    --save-path ./results

# Clear memory between stages
clear_system_caches
check_memory

echo "Stage 4: Attempting full settings (64GB+ RAM only)"
# Only attempt if previous stages succeeded
python3 sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 720 1280 \
    --video-length 25 \
    --infer-steps 25 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --vae-tiling \
    --precision fp16 \
    --vae-precision fp16 \
    --text-encoder-precision fp16 \
    --text-encoder-precision-2 fp16 \
    --save-path ./results

echo "Complete! Check ./results for generated videos"
