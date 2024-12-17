#!/bin/bash
# Description: This script demonstrates memory-optimized video generation using HunyuanVideo model

# Set conservative memory settings
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection
export MPS_USE_GUARD_MODE=1
export MPS_ENABLE_MEMORY_GUARD=1
export PYTORCH_MPS_SYNC_OPERATIONS=1
export PYTORCH_MPS_AGGRESSIVE_MEMORY_CLEANUP=1

# Start with conservative settings (recommended)
python3 sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 544 960 \
    --video-length 65 \
    --infer-steps 25 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --vae-tiling \
    --save-path ./results

# For 64GB+ RAM after successful conservative generation:
# python3 sample_video_mps.py \
#     --mmgp-mode \
#     --mmgp-config configs/mmgp_mlx.json \
#     --video-size 720 1280 \
#     --video-length 65 \
#     --infer-steps 25 \
#     --prompt "A cat walks on the grass, realistic style." \
#     --seed 42 \
#     --embedded-cfg-scale 6.0 \
#     --flow-shift 7.0 \
#     --vae-tiling \
#     --save-path ./results
