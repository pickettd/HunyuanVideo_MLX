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

## Memory-Optimized Generation (New!)

The system now uses a chunked generation approach for better memory efficiency:

### For 32GB RAM Macs (M1/M2/M3 Pro/Max):
```bash
# Memory-efficient settings (recommended)
python sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 544 960 \
    --prompt "your prompt here" \
    --video-length 65

# Medium quality once stable
python sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 720 720 \
    --prompt "your prompt here" \
    --video-length 65
```

### For 64GB+ RAM Macs (M3 Max/Ultra):
```bash
# High quality generation
python sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 720 1280 \
    --prompt "your prompt here" \
    --video-length 129
```

### Memory Optimization Features

The new chunked generation system provides:
- Automatic chunk size calculation based on available RAM
- Frame overlap for smooth transitions between chunks
- Dynamic precision adjustment
- Aggressive memory cleanup
- Progress tracking and detailed feedback

### Resolution Guidelines

Start with smaller resolutions and gradually increase based on stability:

| RAM   | Recommended Resolution | Max Video Length |
|-------|----------------------|------------------|
| 32GB  | 544x960 (540p)      | 65 frames       |
| 32GB  | 720x720 (square)    | 65 frames       |
| 64GB  | 720x1280 (720p)     | 129 frames      |
| 64GB+ | 1280x720 (720p)     | 129 frames      |

## Memory Management Tips

### Environment Settings
```bash
# Conservative settings (recommended for first run)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2

# After successful generation, can try:
# For 32GB RAM:
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3

# For 64GB+ RAM:
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
```

### Optimization Steps

If you encounter memory issues:
1. Start with 540p resolution (544x960)
2. Use shorter video length (65 frames)
3. Close other applications
4. Clear Python environment between runs
5. Monitor resources with `python monitor_resources.py`
6. Gradually increase settings once stable

### Web Interface

The included Gradio interface provides:
- Memory-optimized default settings
- Resolution recommendations
- Real-time feedback
- Detailed error messages with optimization suggestions

Run the web interface:
```bash
python gradio_server.py
```

## Key Features

- **Mac-Native Performance**: Built specifically for Apple Silicon using Metal acceleration
- **Memory Optimization**: 
  * Chunked video generation
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
- Higher watermark ratio (0.6) supported
- Larger chunk sizes for faster processing
- 129-frame videos supported

### Other M-series with 32GB RAM
- Recommended 544x960 or smaller
- Float16 precision for memory efficiency
- Conservative watermark ratio (0.3-0.4)
- Memory-efficient chunked processing

### Minimum Requirements
- Apple Silicon Mac (M1/M2/M3)
- macOS 12.3 or later
- 32GB RAM minimum
- Python 3.10 or later

## Troubleshooting

If you encounter memory issues:
1. Close all other applications
2. Use recommended resolution for your RAM
3. Start with shorter video length
4. Clear Python environment between runs
5. Monitor resources with `python monitor_resources.py`
6. Check logs for specific optimization suggestions

## License

Licensed under the Tencent Hunyuan Community License - see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

- Original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) by Tencent
- [MLX](https://github.com/ml-explore/mlx) framework by Apple
