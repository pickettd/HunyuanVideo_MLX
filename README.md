# HunyuanVideo MLX

[![License](https://img.shields.io/badge/license-Tencent%20Hunyuan-blue.svg)](LICENSE.txt)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3116/)
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
- Python 3.11 (required for package compatibility)
- 16GB RAM minimum (32GB+ recommended)

To install Python 3.11:
```bash
# Using Homebrew
brew install python@3.11

# Or download from python.org
open https://www.python.org/downloads/release/python-3116/
```

## Setup

1. **Ensure System Requirements**:
   - macOS 12.3 or later
   - Apple Silicon Mac (M1/M2/M3)
   - Python 3.11 (required)
   - 16GB RAM minimum (32GB+ recommended)

2. **Install Dependencies**:
```bash
# Clone the repository
git clone https://github.com/yourusername/HunyuanVideo_MLX.git
cd HunyuanVideo_MLX

# Run the installation script (creates venv and installs dependencies)
./install_mlx.sh
```

3. **Configure Environment**:
```bash
# The install script creates .env from template
# Edit .env and add your Hugging Face token:
# Get token from: https://huggingface.co/settings/tokens
nano .env  # or use any text editor
```

4. **Download Model Weights**:
```bash
# Activate the virtual environment (if not already activated)
source venv/bin/activate

# Download weights (requires HF_TOKEN in .env)
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
