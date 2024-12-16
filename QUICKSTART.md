# Quick Start Guide

## Initial Setup

1. Run system check to verify compatibility:
```bash
python check_system.py
```

2. Set up environment variables and activate virtual environment:
```bash
./setup_env.sh
```

3. Create Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install dependencies:
```bash
# Install PyTorch with MPS support
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install other requirements
pip install -r requirements_mps.txt
pip install ninja flash-attention --no-build-isolation
```

5. Download model weights:
```bash
python download_weights.py
```

## Basic Video Generation

Generate a video at lower resolution (recommended for first test):
```bash
python sample_video_mps.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "An Australian Shepherd dog catching a frisbee in mid-air, slow motion, cinematic style" \
    --flow-reverse \
    --save-path ./results
```

## Memory-Efficient High Resolution (MMGP)

For higher resolution with optimized memory usage:
```bash
python sample_video_mps.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "An Australian Shepherd dog catching a frisbee in mid-air, slow motion, cinematic style" \
    --flow-reverse \
    --mmgp-mode \
    --mmgp-config configs/mmgp_example.json \
    --save-path ./results
```

## Monitoring Tools

1. Monitor system resources during generation:
```bash
python monitor_resources.py
```

2. Run performance benchmark:
```bash
python benchmark.py
```

## Memory Usage Tips

1. Start with lower resolutions (544x960) to test system compatibility
2. Use MMGP for higher resolutions to optimize memory usage
3. Monitor system resources during generation
4. Close other memory-intensive applications
5. Set environment variables for optimal performance:
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
```

## Common Issues

1. If you see "MPS not available":
   ```bash
   python check_system.py
   ```
   This will verify your system compatibility and provide specific recommendations.

2. If you encounter memory errors:
   - Use `monitor_resources.py` to track memory usage
   - Try using MMGP mode
   - Reduce video resolution
   - Close other applications

3. If generation is slow:
   - Use MMGP with more steps on lighter model
   - Monitor CPU/Memory usage
   - Consider reducing inference steps

## Maintenance

Clean up temporary files and manage disk space:
```bash
./cleanup.sh
```

## Next Steps

- See README.md for detailed documentation
- Check DIRECTORY_STRUCTURE.md for project organization
- Join the community discussions on GitHub
