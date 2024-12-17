import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

import torch
import argparse
import gc

def add_mmgp_args(parser):
    group = parser.add_argument_group(title="MMGP args")
    group.add_argument(
        "--mmgp-mode",
        action="store_true",
        help="Enable MMGP (Mixed Model Generation Pipeline) mode",
    )
    group.add_argument(
        "--mmgp-config",
        type=str,
        default="configs/mmgp_mlx.json",
        help="Path to MMGP configuration file",
    )
    return parser

def check_mps_settings():
    """Verify MPS settings before running"""
    high_ratio = os.getenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
    low_ratio = os.getenv('PYTORCH_MPS_LOW_WATERMARK_RATIO')
    
    if not high_ratio or not low_ratio:
        logger.error("MPS watermark ratios not set!")
        logger.info("Please set the following environment variables:")
        logger.info("export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6")
        logger.info("export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5")
        return False
    
    try:
        high_ratio = float(high_ratio)
        low_ratio = float(low_ratio)
        
        if high_ratio > 1.0 or low_ratio > 1.0 or high_ratio < low_ratio:
            logger.error(f"Invalid watermark ratios: high={high_ratio}, low={low_ratio}")
            logger.info("High ratio should be <= 1.0 and greater than low ratio")
            return False
            
    except ValueError:
        logger.error("Invalid watermark ratio values")
        return False
    
    return True

def clear_memory():
    """Clear CUDA memory cache"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def main():
    # Check MPS settings first
    if not check_mps_settings():
        return

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because PyTorch was not built with MPS enabled")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device")
        return

    # Clear memory before starting
    clear_memory()

    # Parse arguments
    parser = argparse.ArgumentParser(description="HunyuanVideo inference script")
    parser = add_mmgp_args(parser)
    args = parse_args(namespace=parser.parse_args([]))
    
    # Enable MMGP mode by default for memory optimization
    args.mmgp_mode = True
    
    # Use memory-efficient settings
    args.precision = "fp16"  # Use float16 for memory efficiency
    args.vae_precision = "fp16"
    args.text_encoder_precision = "fp16"
    args.disable_autocast = False  # Enable autocast for memory efficiency
    args.vae_tiling = True  # Enable VAE tiling
    
    # Set device to MPS
    device = torch.device("mps")
    
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    try:
        # Load models with MMGP optimization
        hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
            models_root_path, 
            args=args,
            device=device
        )
        
        # Get the updated args
        args = hunyuan_video_sampler.args

        # Clear memory before inference
        clear_memory()

        # Start sampling with optimized settings
        outputs = hunyuan_video_sampler.predict(
            prompt=args.prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=30,  # Reduced steps for memory efficiency
            guidance_scale=7.0,  # Balanced guidance scale
            num_videos_per_prompt=1,  # Generate one video at a time
            flow_shift=args.flow_shift,
            batch_size=1,  # Process one batch at a time
            embedded_guidance_scale=args.embedded_cfg_scale
        )
        samples = outputs['samples']
        
        # Save samples
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("Out of memory error occurred. Try:")
            logger.info("1. Reducing video resolution")
            logger.info("2. Reducing video length")
            logger.info("3. Using fewer inference steps")
            logger.info("4. Clearing other applications from memory")
        raise e
    finally:
        # Clean up memory
        clear_memory()

if __name__ == "__main__":
    main()
