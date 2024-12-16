# HunyuanVideo MLX Port

Native port of [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) optimized for Apple Silicon using MLX and Metal Performance Shaders (MPS).

## ⚠️ Important Note About Model Weights

Currently, only the main transformer model is automatically downloadable. The VAE and text encoder models need to be downloaded manually from the official HunyuanVideo repository. We are working on updating the download script.

Required Model Files:
1. ✓ Main transformer model (automatically downloaded)
   - Location: `ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt`
   - Size: ~25GB

2. ⚠️ VAE model (manual download required)
   - Location: `ckpts/vae/884-16c-hy.pt`
   - Download from: [HunyuanVideo Repository](https://huggingface.co/tencent/HunyuanVideo)

3. ⚠️ Text encoder model (manual download required)
   - Location: `ckpts/text_encoder/llm.pt`
   - Download from: [HunyuanVideo Repository](https://huggingface.co/tencent/HunyuanVideo)

## System Requirements

- Apple Silicon Mac (M1/M2/M3)
- macOS 12.3 or later
- Minimum 32GB RAM recommended
- 64GB RAM for higher resolutions

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/gregcmartin/HunyuanVideo_MLX.git
cd HunyuanVideo_MLX
```

2. Create and activate Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
# Install PyTorch with MPS support
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install other requirements
pip install -r requirements_mps.txt
pip install ninja flash-attention --no-build-isolation
```

4. Download model weights:
```bash
# Download main transformer model
python download_weights.py

# Manually download VAE and text encoder models
# Place them in their respective directories:
# - ckpts/vae/884-16c-hy.pt
# - ckpts/text_encoder/llm.pt
```

5. Generate a video:
```bash
python sample_video_mps.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "An Australian Shepherd dog catching a frisbee in mid-air, slow motion, cinematic style" \
    --flow-reverse \
    --save-path ./results
```

## Utility Tools

The port includes several tools to help you optimize and monitor performance:

1. System Check:
```bash
python check_system.py
```

2. Resource Monitor:
```bash
python monitor_resources.py
```

3. Performance Benchmark:
```bash
python benchmark.py
```

4. Environment Setup:
```bash
./setup_env.sh
```

## Advanced Usage: MMGP

MMGP (Mixed Model Generation Pipeline) allows using different models at different stages for optimal memory usage and quality:

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

## Performance Tips

1. Memory Management:
   - Start with lower resolutions (544x960) first
   - Use MMGP for efficient memory usage
   - Monitor memory usage with `monitor_resources.py`
   - Close other memory-intensive applications

2. Generation Speed:
   - Use lighter models for initial steps
   - Adjust batch size based on available memory
   - Consider using fewer inference steps for faster generation

3. Quality Optimization:
   - Use higher quality models for final refinement
   - Experiment with different MMGP configurations
   - Balance between speed and quality using step distribution

## Performance Expectations

Generation times will vary based on your specific Apple Silicon chip and available memory:

- M1 Pro/Max (32GB) with MMGP:
  - 544x960 → 720x1280: ~12-18 minutes per video
  - Pure 544x960: ~10-15 minutes per video

- M2 Pro/Max (32GB+) with MMGP:
  - 544x960 → 720x1280: ~10-15 minutes per video
  - Pure 544x960: ~8-12 minutes per video

- M3 Pro/Max (48GB+) with MMGP:
  - 544x960 → 720x1280: ~8-12 minutes per video
  - Pure 544x960: ~6-10 minutes per video

## Troubleshooting

1. If you encounter "MPS not available" error:
   - Ensure you're on macOS 12.3 or later
   - Run `python check_system.py` to verify compatibility
   - Set required environment variables:
     ```bash
     export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
     ```

2. If you encounter memory errors:
   - Use `monitor_resources.py` to track memory usage
   - Try using MMGP with more steps on the lighter model
   - Reduce video resolution or batch size
   - Close other applications

3. If you encounter missing model errors:
   - Verify all model files are in their correct locations
   - Check file permissions
   - Run `python check_system.py` to verify model paths

## Directory Structure

See [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) for a complete overview of the project organization.

## Quick Reference

See [QUICKSTART.md](QUICKSTART.md) for a condensed guide to getting started quickly.

## License

This project is licensed under the same terms as the original HunyuanVideo project.

## Acknowledgments

- Original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) project by Tencent
- [MLX](https://github.com/ml-explore/mlx) by Apple
- Apple Metal team for MPS support
