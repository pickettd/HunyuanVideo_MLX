# HunyuanVideo MLX

A streamlined, Mac-native port of Tencent's text-to-video AI model, optimized specifically for Apple Silicon. Generate high-quality videos from text descriptions with native performance on your M1/M2/M3 Mac.

## Why This Port?

This project makes HunyuanVideo accessible on Apple Silicon Macs by:
- Leveraging native Metal acceleration through MLX
- Optimizing memory usage for Mac hardware
- Simplifying the setup process
- Providing Mac-specific performance tuning

## Setup & Usage

```bash
# One-line setup
/install_mlx.sh

# Download model weights (requires Hugging Face token)
python download_weights.py

# Copy environment example and configure
cp .env.example .env
# Add your token: HF_TOKEN=your_token_here
```

Before running `download_weights.py`, make sure you have:
1. A Hugging Face account and access token (get it from https://huggingface.co/settings/tokens)
2. Created a `.env` file with your token and memory settings

## Memory-Optimized Generation (Recommended)

Using MMGP (Mixed Model Generation Pipeline) mode for optimal memory management:

### For 32GB RAM Macs (M1/M2/M3 Pro/Max):
```bash
# Conservative settings for stable generation
python sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 384 640 \
    --prompt "your prompt here" \
    --video-length 16

# Medium quality once stable
python sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 544 960 \
    --prompt "your prompt here" \
    --video-length 16
```

### For 64GB+ RAM Macs (M3 Max/Ultra):
```bash
# High quality generation
python sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 720 1280 \
    --prompt "your prompt here" \
    --video-length 24
```

MMGP mode automatically:
- Detects your Mac's RAM and applies optimal settings
- Uses appropriate precision (fp16/fp32)
- Manages memory watermarks
- Enables VAE tiling
- Provides staged model loading

## Memory Management Tips

### Environment Settings
```bash
# Conservative settings (recommended for first run)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3

# After successful generation, can try:
# For 32GB RAM:
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5

# For 64GB+ RAM:
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7
```

### Video Size Guidelines
- Start with smaller sizes and gradually increase:
  * Initial testing: 384x640
  * Medium quality: 544x960
  * High quality: 720x1280 (64GB+ RAM only)

### Video Length
- Keep video length as 4n+1 (5, 9, 13, 17, etc.)
- Start with shorter lengths (16 frames) and increase if stable
- Longer videos (24+ frames) recommended only for 64GB+ RAM

## Key Features

- **Mac-Native Performance**: Built specifically for Apple Silicon using Metal acceleration
- **Memory Optimization**: 
  * MMGP (Mixed Model Generation Pipeline)
  * Automatic hardware detection
  * Dynamic precision adjustment
  * VAE tiling support
- **Multiple Resolutions**: Support for various video sizes and aspect ratios
- **Real-time Monitoring**: Built-in resource monitoring
- **Easy Setup**: Streamlined installation process

## Hardware Requirements

### M3 Max/Ultra with 64GB+ RAM
- Direct 720p generation with float32 precision
- Higher watermark ratio (0.8) supported
- Batch processing enabled
- 40 inference steps for optimal quality

### Other M-series with 32GB RAM
- Recommended 544x960 or smaller
- Float16 precision for memory efficiency
- Conservative watermark ratio (0.4-0.6)
- Memory-efficient processing

### Minimum Requirements
- Apple Silicon Mac (M1/M2/M3)
- macOS 12.3 or later
- 32GB RAM minimum
- Python 3.10 or later

## Troubleshooting

If you encounter memory issues:
1. Close all other applications
2. Use MMGP mode with conservative settings
3. Reduce video resolution and length
4. Clear Python environment between runs
5. Monitor resources with `python monitor_resources.py`

## License

Licensed under the Tencent Hunyuan Community License - see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

- Original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) by Tencent
- [MLX](https://github.com/ml-explore/mlx) framework by Apple
