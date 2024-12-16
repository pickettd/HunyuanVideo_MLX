# Text Encoder Setup Guide

## Current Status

We have successfully downloaded:
1. ✓ Main transformer model (23.9 GB)
2. ✓ VAE model (0.9 GB)

For the text encoder, there are several options:

## Option 1: Direct Download

The text encoder is part of the Hunyuan-Large model. You can:

1. Visit [Hunyuan-Large Model](https://huggingface.co/tencent/Tencent-Hunyuan-Large)
2. Download the A52B-Instruct version
3. Place it in `ckpts/text_encoder/llm.pt`

## Option 2: Use Alternative Text Encoder

According to the documentation, HunyuanVideo uses a pre-trained Multimodal Large Language Model (MLLM) as its text encoder. You can:

1. Use the [Hunyuan-Large original code](https://github.com/Tencent/Tencent-Hunyuan-Large)
2. Follow their setup instructions
3. Convert the model to the required format

## Option 3: Wait for Official Release

The text encoder might be released separately or as part of a future update. You can:

1. Watch the [HunyuanVideo Repository](https://github.com/Tencent/HunyuanVideo)
2. Monitor the [Issues Page](https://github.com/Tencent/HunyuanVideo/issues)
3. Check for updates in their documentation

## Manual Setup Instructions

If you have access to the text encoder model:

1. Create the directory:
```bash
mkdir -p ckpts/text_encoder
```

2. Place the model file:
```bash
mv path/to/your/text_encoder.pt ckpts/text_encoder/llm.pt
```

3. Verify setup:
```bash
python check_system.py
```

## Alternative Solutions

While waiting for the text encoder:

1. You can still experiment with:
   - System setup and configuration
   - Resource monitoring
   - Performance benchmarking
   - VAE model testing

2. The following components are ready:
   - Main transformer model
   - VAE model
   - System utilities
   - Monitoring tools

## Getting Help

If you need assistance:

1. Check the official channels:
   - [HunyuanVideo Repository](https://github.com/Tencent/HunyuanVideo)
   - [Hunyuan-Large Repository](https://github.com/Tencent/Tencent-Hunyuan-Large)
   - [Project Website](https://aivideo.hunyuan.tencent.com)

2. Join the community:
   - GitHub Issues
   - WeChat Group
   - Discord Server

## Updates

We will update this document when:
1. The official text encoder is released
2. Alternative solutions are verified
3. New setup methods become available
