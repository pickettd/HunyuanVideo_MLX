# HunyuanVideo for Apple Silicon

This is a native port of HunyuanVideo for Apple Silicon (M1/M2/M3) Macs, utilizing Metal Performance Shaders (MPS) for hardware acceleration.

## System Requirements

- Apple Silicon Mac (M1/M2/M3)
- macOS 12.3 or later
- Minimum 32GB RAM recommended
- 64GB RAM for higher resolutions

## Installation

1. Install Miniconda for Apple Silicon from [here](https://docs.conda.io/en/latest/miniconda.html)

2. Clone the repository:
```bash
git clone https://github.com/tencent/HunyuanVideo
cd HunyuanVideo
```

3. Make the installation script executable and run it:
```bash
chmod +x install_mps.sh
./install_mps.sh
```

4. Activate the environment:
```bash
conda activate HunyuanVideo-MPS
```

## Usage

### Basic Usage

1. Download the model weights as described in the main README.

2. Generate a video using the MPS-optimized script:
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

MMGP allows you to use different models at different stages of the generation process. This can be particularly useful on Apple Silicon to optimize memory usage and generation quality.

1. Create an MMGP configuration file (or use the example in configs/mmgp_example.json):
```json
{
    "models": {
        "model_540p": "ckpts/hunyuan-video-t2v-540p",
        "model_720p": "ckpts/hunyuan-video-t2v-720p"
    },
    "schedule": [
        {
            "model": "model_540p",
            "start_step": 0,
            "end_step": 25
        },
        {
            "model": "model_720p",
            "start_step": 25,
            "end_step": 50
        }
    ]
}
```

2. Run generation with MMGP:
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

MMGP Benefits for Apple Silicon:
- More efficient memory usage by using lighter models in early stages
- Better quality by using higher-resolution models for refinement
- Flexible configuration to balance between speed and quality

## Important Notes

1. This port uses FP32 precision by default as it's most stable on MPS.
2. Generation will be slower compared to CUDA GPUs due to:
   - Limited memory bandwidth on unified memory
   - FP32 precision instead of mixed precision
   - Some operations falling back to CPU

3. Memory Usage:
   - Start with lower resolutions (544x960) first
   - Monitor memory usage with Activity Monitor
   - If you encounter memory issues, try using MMGP with a lighter model for initial steps
   - Consider reducing batch size or video length

4. Known Limitations:
   - No multi-GPU support (not applicable to Apple Silicon)
   - No CPU offloading support
   - Autocast is disabled as it's not supported on MPS
   - Flash attention is used in CPU mode

## MMGP Performance Tips

1. Memory Optimization:
   - Use lighter models (540p) for initial denoising steps
   - Switch to higher quality models (720p) only for final refinement
   - Adjust step distribution based on your available memory

2. Quality vs Speed:
   - More steps with the lighter model = faster generation
   - More steps with the higher quality model = better details
   - Experiment with different step distributions

3. Recommended Configurations:
   - For 32GB RAM:
     ```json
     {
         "schedule": [
             {"model": "model_540p", "start_step": 0, "end_step": 35},
             {"model": "model_720p", "start_step": 35, "end_step": 50}
         ]
     }
     ```
   - For 64GB RAM:
     ```json
     {
         "schedule": [
             {"model": "model_540p", "start_step": 0, "end_step": 25},
             {"model": "model_720p", "start_step": 25, "end_step": 50}
         ]
     }
     ```

## Troubleshooting

1. If you encounter "MPS not available" error:
   - Ensure you're on macOS 12.3 or later
   - Verify PyTorch is installed with MPS support
   - Run this test code:
     ```python
     import torch
     print(f"Is MPS available: {torch.backends.mps.is_available()}")
     print(f"Is MPS built: {torch.backends.mps.is_built()}")
     ```

2. If you encounter memory errors:
   - Try using MMGP with more steps on the lighter model
   - Reduce video resolution or batch size
   - Close other memory-intensive applications
   - Try setting environment variable:
     ```bash
     export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
     ```

3. If generation is too slow:
   - Use MMGP with more steps on the lighter model
   - Consider using fewer inference steps
   - Try adjusting the batch size
   - Close other GPU-intensive applications

## Performance Expectations

Generation times will vary based on your specific Apple Silicon chip and available memory. Here are rough estimates:

- M1 Pro/Max (32GB) with MMGP:
  - 544x960 → 720x1280: ~12-18 minutes per video
  - Pure 544x960: ~10-15 minutes per video

- M2 Pro/Max (32GB+) with MMGP:
  - 544x960 → 720x1280: ~10-15 minutes per video
  - Pure 544x960: ~8-12 minutes per video

- M3 Pro/Max (48GB+) with MMGP:
  - 544x960 → 720x1280: ~8-12 minutes per video
  - Pure 544x960: ~6-10 minutes per video

These are approximate times and may vary based on your specific configuration and workload.

## Contributing

If you encounter any issues or have improvements for the MPS port, please:

1. Test with the latest PyTorch nightly build first
2. Check if the issue is MPS-specific by running a small test case
3. Submit an issue with:
   - Your exact hardware configuration
   - macOS version
   - PyTorch version
   - Complete error message
   - Steps to reproduce
   - MMGP configuration if applicable
