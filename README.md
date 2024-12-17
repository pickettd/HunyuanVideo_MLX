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

## Memory-Optimized Generation (Updated!)

The system now uses an enhanced chunked generation approach for better memory efficiency:

### For All Mac Configurations:
```bash
# Memory-efficient settings (recommended)
python sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 544 960 \
    --prompt "your prompt here" \
    --video-length 65

# After stable generation, try:
python sample_video_mps.py \
    --mmgp-mode \
    --mmgp-config configs/mmgp_mlx.json \
    --video-size 720 720 \
    --prompt "your prompt here" \
    --video-length 65
```

### Memory Optimization Features

The enhanced chunked generation system provides:
- Smaller chunk sizes (8 frames) with reduced overlap
- Aggressive memory cleanup between chunks
- Dynamic chunk size reduction on OOM
- Conservative memory watermark settings
- Forced fp16 precision for efficiency

### Resolution Guidelines

Start with smaller resolutions and gradually increase based on stability:

| RAM   | Initial Resolution | Max Resolution | Video Length |
|-------|-------------------|----------------|--------------|
| 32GB  | 544x960 (540p)   | 720x720        | 65 frames   |
| 64GB  | 544x960 (540p)   | 720x1280       | 65 frames   |
| 64GB+ | 720x720          | 720x1280       | 129 frames  |

## Memory Management Tips

### Environment Settings
```bash
# Conservative settings (recommended for all configurations)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2
```

### Optimization Steps

For optimal performance:
1. Start with 540p resolution (544x960)
2. Use shorter video length (65 frames)
3. Close other applications
4. Monitor resources with `python monitor_resources.py`
5. Clear Python environment between runs
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
- **Enhanced Memory Optimization**: 
  * Smaller chunk sizes with reduced overlap
  * Aggressive memory cleanup
  * Conservative watermark settings
  * Forced fp16 precision
  * Dynamic chunk size reduction
  * VAE tiling support
- **Multiple Resolutions**: Support for various video sizes and aspect ratios
- **Real-time Monitoring**: Built-in resource monitoring
- **Easy Setup**: Streamlined installation process

## Hardware Requirements

### All M-series Macs
- Start with 544x960 resolution
- Use fp16 precision for memory efficiency
- Conservative watermark ratio (0.3)
- Memory-efficient chunked processing

### Minimum Requirements
- Apple Silicon Mac (M1/M2/M3)
- macOS 12.3 or later
- 32GB RAM minimum
- Python 3.10 or later

## Troubleshooting

If you encounter memory issues:
1. Ensure conservative memory settings are used:
   ```bash
   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3
   PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2
   ```
2. Start with 544x960 resolution
3. Use 65 frames video length
4. Monitor memory with `python monitor_resources.py`
5. Clear Python environment between runs
6. Close other applications

## License

Licensed under the Tencent Hunyuan Community License - see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

- Original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) by Tencent
- [MLX](https://github.com/ml-explore/mlx) framework by Apple
