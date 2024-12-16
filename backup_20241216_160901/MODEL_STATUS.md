# HunyuanVideo Model Status Update

## Current Status (December 16, 2023)

1. ✓ Main Transformer Model
- Successfully downloaded (23.9 GB)
- Location: `ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt`
- Status: Working

2. ✓ VAE Model
- Successfully downloaded (0.9 GB)
- Location: `ckpts/vae/884-16c-hy.pt`
- Status: Working

3. ⚠️ Text Encoder Model
- Currently unavailable through direct download
- Expected Location: `ckpts/text_encoder/llm.pt`
- Status: Investigating alternative sources
- Note: The model uses a pre-trained MLLM (Multimodal Large Language Model) as text encoder

## Next Steps

1. For Text Encoder:
- Investigating the correct source for the text encoder model
- Checking if it needs to be downloaded from a separate repository
- Looking into the Hunyuan-Large model mentioned in documentation

2. Current Capabilities:
- Main transformer model is ready
- VAE model is ready
- System can be tested for partial functionality

3. Action Items:
- Monitor the official repository for text encoder updates
- Check the Hunyuan-Large repository for compatible text encoder
- Test system with available components

## Updates

We have successfully downloaded:
1. Main transformer model (23.9 GB)
2. VAE model (0.9 GB)

The text encoder appears to be part of a separate package. According to the documentation, HunyuanVideo uses a pre-trained Multimodal Large Language Model (MLLM) as its text encoder. We are investigating the correct source and will update this document when we have more information.

## Stay Updated

To stay informed about model releases:
1. Watch the [HunyuanVideo Repository](https://github.com/Tencent/HunyuanVideo)
2. Check the [Hugging Face Model Page](https://huggingface.co/tencent/HunyuanVideo)
3. Monitor the [Hunyuan-Large Repository](https://github.com/Tencent/Tencent-Hunyuan-Large)

## Need Help?

If you have information about the text encoder model:
1. Check if you have access to the Hunyuan-Large model
2. Verify compatibility with HunyuanVideo
3. Share information through official channels

We appreciate your patience as we work to provide a complete solution.
