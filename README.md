# HunyuanVideo MLX Port

This is a native port of [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) optimized for Apple Silicon using MLX and Metal Performance Shaders (MPS). This port aims to provide efficient video generation on Apple Silicon devices.

## Features

- Native Apple Silicon support using MLX and MPS
- MMGP (Mixed Model Generation Pipeline) support for memory optimization
- Optimized for M1/M2/M3 Macs
- Memory-efficient generation pipeline

## System Requirements

- Apple Silicon Mac (M1/M2/M3)
- macOS 12.3 or later
- Minimum 32GB RAM recommended
- 64GB RAM for higher resolutions

## Installation

1. Install Miniconda for Apple Silicon:
```bash
# Download and install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
```

2. Clone this repository:
```bash
git clone https://github.com/gregcmartin/HunyuanVideo_MLX.git
cd HunyuanVideo_MLX
```

3. Run the installation script:
```bash
chmod +x install_mps.sh
./install_mps.sh
```

## Usage

### Basic Generation

```bash
python sample_video_mps.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --save-path ./results
```

### MMGP (Mixed Model Generation Pipeline)

MMGP allows using different models at different stages for optimal memory usage and quality:

```bash
python sample_video_mps.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --save-path ./results \
    --mmgp-mode \
    --mmgp-config configs/mmgp_example.json
```

## Performance Tips

1. Memory Management:
   - Use MMGP for efficient memory usage
   - Start with lower resolutions first
   - Monitor memory usage with Activity Monitor

2. Generation Speed:
   - Use lighter models for initial steps
   - Adjust batch size based on available memory
   - Close other GPU-intensive applications

3. Quality Optimization:
   - Use higher quality models for final refinement
   - Experiment with different MMGP configurations
   - Balance between speed and quality using step distribution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the same terms as the original HunyuanVideo project.

## Acknowledgments

- Original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) project by Tencent
- [MLX](https://github.com/ml-explore/mlx) by Apple
- Apple Metal team for MPS support

## Status

This is an active work in progress. Current focus areas:
- [ ] MLX integration
- [x] MPS support
- [x] MMGP implementation
- [ ] Performance optimization
- [ ] Memory usage improvements
