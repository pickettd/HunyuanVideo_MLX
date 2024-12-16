# Installation Guide for HunyuanVideo MLX

## System Requirements

1. Hardware:
- Apple Silicon Mac (M1/M2/M3)
- Minimum 32GB RAM (45GB required for 544x960)
- Recommended 64GB RAM (60GB required for 720x1280)

2. Software:
- macOS 12.3 or later
- Python 3.10 or later
- Git

## Step-by-Step Installation

1. Create and activate a new conda environment:
```bash
conda create -n hunyuan-video python=3.10
conda activate hunyuan-video
```

2. Install PyTorch with MPS support:
```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

3. Install dependencies:
```bash
# Install base requirements
pip install -r requirements_mps.txt

# Install additional dependencies
pip install ninja flash-attention --no-build-isolation
```

4. Set up Hugging Face token:
```bash
# Export your Hugging Face token
export HF_TOKEN="your_token_here"

# Or use the Hugging Face CLI
huggingface-cli login
```

5. Download model weights:
```bash
# Create model directories
mkdir -p ckpts/hunyuan-video-t2v-720p/transformers
mkdir -p ckpts/vae
mkdir -p ckpts/text_encoder

# Download weights using the script
python download_weights.py
```

6. Verify installation:
```bash
# Check system compatibility
python check_system.py

# Run benchmark
python benchmark.py
```

## Environment Variables

Set these environment variables for optimal performance:

```bash
# Set MPS high watermark ratio
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
```

## Troubleshooting

1. If you encounter MPS errors:
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

2. If you encounter memory errors:
```bash
# Monitor resources during operation
python monitor_resources.py
```

3. For model weight issues:
```bash
# Verify model files
python check_system.py
```

## Next Steps

1. See QUICKSTART.md for usage examples
2. Check MODEL_SETUP.md for model configuration
3. Review README.md for full documentation

## Support

If you encounter issues:
1. Check the troubleshooting section in README.md
2. Run diagnostics with check_system.py
3. Monitor resources with monitor_resources.py
4. Review logs in the results directory
