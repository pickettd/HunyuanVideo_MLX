#!/bin/bash

# Default values
PROMPT="A beautiful sunset over the ocean, with waves gently rolling onto the beach"
VIDEO_SIZE="540 960"
VIDEO_LENGTH=16
SEED=42
GUIDANCE_SCALE=7.0
NUM_INFERENCE_STEPS=25
PRECISION="fp16"
VAE_PRECISION="fp16"
VAE_TILING="--vae-tiling"
SAVE_PATH="results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --video-size)
            VIDEO_SIZE="$2 $3"
            shift 3
            ;;
        --video-length)
            VIDEO_LENGTH="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --guidance-scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --num-inference-steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --vae-precision)
            VAE_PRECISION="$2"
            shift 2
            ;;
        --no-vae-tiling)
            VAE_TILING=""
            shift
            ;;
        --save-path)
            SAVE_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set PYTHONPATH to include project root
export PYTHONPATH=".:$PYTHONPATH"

# Run video generation
python sample_video.py \
    --prompt "$PROMPT" \
    --video-size $VIDEO_SIZE \
    --video-length $VIDEO_LENGTH \
    --seed $SEED \
    --guidance-scale $GUIDANCE_SCALE \
    --num-inference-steps $NUM_INFERENCE_STEPS \
    --precision $PRECISION \
    --vae-precision $VAE_PRECISION \
    $VAE_TILING \
    --save-path "$SAVE_PATH"
