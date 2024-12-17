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
curl -s https://raw.githubusercontent.com/gregcmartin/HunyuanVideo_MLX/main/install_mlx.sh | bash

# Download model weights (requires Hugging Face token)
python download_weights.py

# Generate your first video
python sample_video_mps.py \
    --video-size 720 1280 \
    --prompt "a cat is running, realistic." \
    --save-path ./results
```

Before running `download_weights.py`, make sure you have:
1. A Hugging Face account and access token (get it from https://huggingface.co/settings/tokens)
2. Created a `.env` file with your token: `HF_TOKEN=your_token_here`

## Key Features

- **Mac-Native Performance**: Built specifically for Apple Silicon using Metal acceleration
- **Memory Optimization**: Efficient implementation that works well with Mac memory architecture
- **Multiple Resolutions**: Support for various video sizes and aspect ratios
  - 540p (544x960, 960x544, etc.)
  - 720p (720x1280, 1280x720, etc.)
- **Real-time Monitoring**: Built-in resource monitoring for optimal performance
- **Easy Setup**: Streamlined installation process for Mac users
- **MMGP Optimization**: Mixed Model Generation Pipeline optimized for each Mac model

## Hardware Requirements & Optimization

### M3 Max/Ultra with 64GB+ RAM
- Direct 720p generation with float32 precision
- Higher watermark ratio (0.8) for better performance
- Batch processing enabled
- 40 inference steps for optimal quality

### Other M-series with 32GB RAM
- Two-phase generation (540p → 720p)
- Float16 precision for memory efficiency
- Balanced watermark ratio (0.7)
- Memory-efficient processing

### Minimum Requirements
- Apple Silicon Mac (M1/M2/M3)
- macOS 12.3 or later
- 32GB RAM minimum
- Python 3.10 or later

## Example Configurations

```bash
# High-quality 720p video (M3 Max/Ultra 64GB)
python sample_video_mps.py \
    --video-size 720 1280 \
    --prompt "your prompt here"

# Memory-efficient 540p video (32GB RAM)
python sample_video_mps.py \
    --video-size 544 960 \
    --prompt "your prompt here"

# Square video
python sample_video_mps.py \
    --video-size 720 720 \
    --prompt "your prompt here"
```

## Performance Tips

- Use appropriate settings for your Mac model (see Hardware Requirements)
- Close memory-intensive applications before generating videos
- Monitor system resources with `python monitor_resources.py`
- First generation may be slower due to Metal shader compilation
- Set memory ratio for optimal performance:
  ```bash
  # For 64GB RAM configurations:
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
  
  # For 32GB RAM configurations:
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
  ```

## License

Licensed under the Tencent Hunyuan Community License - see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

- Original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) by Tencent
- [MLX](https://github.com/ml-explore/mlx) framework by Apple
