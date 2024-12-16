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

# Generate your first video
python sample_video_mps.py \
    --video-size 544 960 \
    --prompt "a cat is running, realistic." \
    --save-path ./results
```

## Key Features

- **Mac-Native Performance**: Built specifically for Apple Silicon using Metal acceleration
- **Memory Optimization**: Efficient implementation that works well with Mac memory architecture
- **Multiple Resolutions**: Support for various video sizes and aspect ratios
  - 540p (544x960, 960x544, etc.)
  - 720p (720x1280, 1280x720, etc.)
- **Real-time Monitoring**: Built-in resource monitoring for optimal performance
- **Easy Setup**: Streamlined installation process for Mac users

## Hardware Requirements

- **Mac**: Any Apple Silicon Mac (M1/M2/M3)
- **OS**: macOS 12.3 or later
- **RAM**: 
  - Minimum: 32GB
  - Recommended: 64GB (for 720p videos)
- **Python**: Version 3.10 or later

## Example Configurations

```bash
# Portrait video (9:16)
python sample_video_mps.py \
    --video-size 544 960 \
    --prompt "your prompt here"

# Landscape video (16:9)
python sample_video_mps.py \
    --video-size 960 544 \
    --prompt "your prompt here"

# Square video (1:1)
python sample_video_mps.py \
    --video-size 720 720 \
    --prompt "your prompt here"

# Advanced options
python sample_video_mps.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 30 \
    --prompt "your prompt here" \
    --flow-reverse \
    --embedded-cfg-scale 6.0 \
    --save-path ./results
```

## Performance Tips

- Use 540p resolution for faster generation and lower memory usage
- Close memory-intensive applications before generating videos
- Monitor system resources with `python monitor_resources.py`
- First generation may be slower due to Metal shader compilation
- Set memory ratio for optimal performance:
  ```bash
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
  ```

## License

Licensed under the Tencent Hunyuan Community License - see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

- Original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) by Tencent
- [MLX](https://github.com/ml-explore/mlx) framework by Apple
