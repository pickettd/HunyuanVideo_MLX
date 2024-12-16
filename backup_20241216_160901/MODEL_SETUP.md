# HunyuanVideo Model Setup Guide

## Current Status

1. ✓ Main Transformer Model
- Successfully downloads from Hugging Face
- Location: `ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt`
- Size: ~24GB

2. ⚠️ VAE Model
- Not currently available through automatic download
- Required Location: `ckpts/vae/884-16c-hy.pt`
- Alternative Sources:
  1. Official Repository: Check releases
  2. Model Zoo: Check available conversions
  3. Community: Check for shared conversions

3. ⚠️ Text Encoder Model
- Not currently available through automatic download
- Required Location: `ckpts/text_encoder/llm.pt`
- Alternative Sources:
  1. Official Repository: Check releases
  2. Model Zoo: Check available conversions
  3. Community: Check for shared conversions

## Temporary Solutions

While we work on fixing the automatic downloads, here are some temporary solutions:

1. Check Official Sources:
- Visit [HunyuanVideo Official Repository](https://github.com/Tencent/HunyuanVideo)
- Look for model releases or conversion scripts
- Check documentation for alternative download links

2. Use Alternative Models:
- Check if compatible VAE models can be used
- Look for compatible text encoder models
- Note: This may affect generation quality

3. Manual Setup:
- If you have access to the models from another source:
  ```bash
  # Create directories
  mkdir -p ckpts/vae
  mkdir -p ckpts/text_encoder
  
  # Copy or move files
  mv path/to/vae.pt ckpts/vae/884-16c-hy.pt
  mv path/to/text_encoder.pt ckpts/text_encoder/llm.pt
  ```

## Verification

After obtaining the models, verify your setup:

```bash
python check_system.py
```

Expected output should show all models present:
```
=== Model Weights ===
✓ Main model: mp_rank_00_model_states.pt (23.9 GB)
✓ VAE model: 884-16c-hy.pt
✓ Text encoder: llm.pt
```

## Next Steps

We are:
1. Working on fixing automatic downloads
2. Looking for official sources of VAE and text encoder models
3. Developing conversion scripts if needed

Please check back for updates or join the discussion in:
- GitHub Issues
- Community Discord
- Official Forums

## Need Help?

If you have access to the working models or know where to find them:
1. Open an issue on GitHub
2. Share the source (if permitted)
3. Help us update the documentation

## Updates

We will update this document as soon as we have a solution for the missing models. Watch the repository for updates.
