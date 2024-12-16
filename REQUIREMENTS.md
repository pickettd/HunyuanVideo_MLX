# HunyuanVideo Requirements

## Required Components

For video generation, HunyuanVideo requires three essential components:

1. Main Transformer Model ✓
   - Status: Downloaded (23.9 GB)
   - Location: `ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt`
   - Purpose: Core video generation model

2. VAE Model ✓
   - Status: Downloaded (0.9 GB)
   - Location: `ckpts/vae/884-16c-hy.pt`
   - Purpose: Video encoding/decoding

3. Text Encoder ⚠️
   - Status: Required but not yet available
   - Expected Location: `ckpts/text_encoder/llm.pt`
   - Purpose: Converts text prompts into embeddings
   - Source: Hunyuan-Large model

## Current Status

Video generation is not possible until all three components are present. The text encoder is a critical component that:
1. Processes the input text prompts
2. Creates embeddings that guide the video generation
3. Cannot be bypassed or substituted

## Next Steps

1. Monitor Official Sources:
   - [HunyuanVideo Repository](https://github.com/Tencent/HunyuanVideo)
   - [Hunyuan-Large Repository](https://github.com/Tencent/Tencent-Hunyuan-Large)
   - [Project Website](https://aivideo.hunyuan.tencent.com)

2. Alternative Solutions:
   - Watch for community-provided text encoders
   - Check for compatible alternatives
   - Monitor for official releases

3. System Preparation:
   - System is ready for video generation
   - All other components are in place
   - Environment is properly configured

## System Requirements

1. Hardware:
   - Apple Silicon Mac (M1/M2/M3)
   - 32GB RAM minimum (45GB for 544x960)
   - 64GB RAM recommended (60GB for 720x1280)

2. Software:
   - macOS 12.3 or later
   - Python 3.10 or later
   - PyTorch with MPS support

3. Environment Variables:
   - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 recommended

## Verification

You can verify your setup with:
```bash
python check_system.py
```

This will show:
- MPS availability
- System specifications
- Model component status
- Environment configuration

## Updates

We will update this document when:
1. The text encoder becomes available
2. Alternative solutions are verified
3. System requirements change
