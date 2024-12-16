# Quick Start Guide

## ⚠️ Model Weights Status

1. ✓ Main transformer model (automatic download):
```bash
python download_weights.py
```

2. ⚠️ Manual Downloads Required:
- VAE model: Place in `ckpts/vae/884-16c-hy.pt`
- Text encoder model: Place in `ckpts/text_encoder/llm.pt`
Download both from: [HunyuanVideo Repository](https://huggingface.co/tencent/HunyuanVideo)

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

## Model Setup

1. Download main transformer model:
```bash
python download_weights.py
```

2. Manually download and place additional models:
- Download VAE and text encoder models from HunyuanVideo repository
- Place in correct directories:
  ```
  ckpts/
  ├── hunyuan-video-t2v-720p/    # ✓ (automatically downloaded)
  │   └── transformers/
  │       └── mp_rank_00_model_states.pt
  ├── vae/                       # ⚠️ (manual download)
  │   └── 884-16c-hy.pt
  └── text_encoder/             # ⚠️ (manual download)
      └── llm.pt
  ```

3. Verify model setup:
```bash
python check_system.py
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

## Common Issues

1. Missing Model Files:
- Verify all model files are downloaded
- Check file permissions
- Run `python check_system.py` to verify paths

2. Memory Issues:
- Use `monitor_resources.py` to track usage
- Try MMGP mode for better memory management
- Reduce resolution or batch size
- Set environment variable:
  ```bash
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
  ```

3. Performance Issues:
- Start with lower resolution first
- Use MMGP for memory optimization
- Monitor system resources
- Close other applications

## Maintenance

Clean up temporary files and manage disk space:
```bash
./cleanup.sh
```

## Next Steps

- See README.md for detailed documentation
- Check DIRECTORY_STRUCTURE.md for project organization
- Join the community discussions on GitHub
