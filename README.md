# HunyuanVideo MLX

[![License](https://img.shields.io/badge/license-Tencent%20Hunyuan-blue.svg)](LICENSE.txt)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.0.10+-blue.svg)](https://github.com/ml-explore/mlx)
[![macOS](https://img.shields.io/badge/macOS-12.3+-blue.svg)](https://support.apple.com/macos)

A streamlined, Mac-native port of Tencent's text-to-video AI model, optimized specifically for Apple Silicon using MLX. Generate high-quality videos from text descriptions with native performance on your M1/M2/M3 Mac.

[Paper](https://arxiv.org/abs/2412.03603) | [Original Repo](https://github.com/Tencent/HunyuanVideo)

## Features

- **MLX Optimization**: Built from the ground up using Apple's MLX framework
- **Metal Acceleration**: Native Metal Performance Shaders (MPS) support
- **Memory Efficient**: Optimized memory usage with chunked processing
- **Mac-Native**: Designed specifically for Apple Silicon
- **High Quality**: Maintains the high quality of the original model

## Directory Structure

```
ckpts/
├── hunyuan-video-t2v-720p/
│   ├── transformers/          # Main model weights
│   └── vae/                   # VAE model
├── text_encoder/              # Primary text encoder (LLaVA)
├── text_encoder_2/            # Secondary text encoder (CLIP)
└── llava-llama-3-8b-v1_1-transformers/  # Raw LLaVA model
```

## Requirements

- macOS 12.3 or later
- Apple Silicon Mac (M1/M2/M3)
- Python 3.10 or later
- 16GB RAM minimum (32GB+ recommended)

## Setup

1. Install MLX and dependencies:
```bash
# Install MLX
pip install mlx

# Install other requirements
pip install -r requirements.txt
```

2. Download model weights:
```bash
# First set up your Hugging Face token
cp .env.example .env
# Edit .env and add: HF_TOKEN=your_token_here

# Download weights
python download_weights.py
```

## Generation

Basic usage with MLX optimization:
```bash
python sample_video_mps.py \
    --prompt "your prompt here" \
    --video-size 544 960 \
    --video-length 13 \
    --precision fp16 \
    --vae-precision fp16 \
    --text-encoder-precision fp16 \
    --text-encoder-precision-2 fp16
```

### Video Parameters

- **Resolution**: Start with 544x960 (540p) and adjust based on your Mac's capabilities
- **Length**: Must satisfy (video_length - 1) % 4 == 0
  - Valid lengths: 5, 9, 13, 17, 21, 25
  - Single frame generation (video_length=1) also supported
- **Precision**: fp16 recommended for optimal Metal performance

### Hardware Recommendations

| Mac Model | RAM   | Resolution  | Max Length | Notes |
|-----------|-------|-------------|------------|-------|
| M1        | 16GB  | 544x960    | 13 frames  | Conservative settings |
| M2        | 32GB  | 720x720    | 17 frames  | Balanced performance |
| M3        | 64GB+ | 720x1280   | 25 frames  | Maximum quality |

## MLX Optimizations

The implementation includes several MLX-specific optimizations:

1. **Memory Management**
   - Chunked processing for video frames
   - Aggressive cache clearing
   - Memory-efficient VAE implementation

2. **Metal Performance**
   - Native MPS backend utilization
   - fp16 precision throughout pipeline
   - Optimized tensor operations

3. **Pipeline Optimizations**
   - Efficient text encoding with MLX
   - Streamlined VAE processing
   - Memory-aware attention computation

## Environment Variables

```bash
# MLX/Metal Optimization Settings
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection
export MPS_USE_GUARD_MODE=1
export MPS_ENABLE_MEMORY_GUARD=1
export PYTORCH_MPS_SYNC_OPERATIONS=1
export PYTORCH_MPS_AGGRESSIVE_MEMORY_CLEANUP=1
```

## Troubleshooting

If you encounter memory issues:

1. Start with conservative settings:
   - 544x960 resolution
   - 13 frames video length
   - fp16 precision

2. Monitor resources:
```bash
python monitor_resources.py
```

3. Clear environment between runs:
```bash
# Reset Python environment
python -c "import torch; torch.mps.empty_cache()"
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork and clone the repository
2. Install development dependencies:
```bash
./install_mlx.sh --with-dev
```
3. Create a feature branch
4. Make your changes
5. Submit a pull request

For more details on our development process, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under the Tencent Hunyuan Community License - see [LICENSE.txt](LICENSE.txt) for details.

## Acknowledgements

- Original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) by Tencent
- [MLX](https://github.com/ml-explore/mlx) framework by Apple
