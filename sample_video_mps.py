import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import torch
import gc
import json
import psutil
import numpy as np

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.utils.chunked_generation import generate_video_chunks, clear_memory
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

def free_memory():
    """Aggressively free system memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    if torch.backends.mps.is_available():
        # Force synchronize before clearing
        torch.mps.synchronize()
        # Multiple synchronization cycles
        for _ in range(3):
            torch.mps.empty_cache()
            torch.mps.synchronize()
    
    # Multiple GC cycles
    for _ in range(3):
        gc.collect()

def staged_model_loading(models_root_path, args, device):
    """Load models in stages with aggressive memory management"""
    try:
        logger.info("Stage 1: Initial setup and memory cleanup...")
        free_memory()
        
        # Set minimal memory limits for loading
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
        
        # Force minimal settings
        args.batch_size = 1
        args.precision = 'fp16'
        args.vae_precision = 'fp16'
        args.text_encoder_precision = 'fp16'
        args.disable_autocast = False
        args.vae_tiling = True
        
        logger.info("Stage 2: Loading model with unlimited memory...")
        try:
            hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
                models_root_path,
                args=args,
                device=device
            )
            
            # Force synchronization and cleanup
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            free_memory()
            
            logger.info("Model loaded successfully!")
            return hunyuan_video_sampler
            
        finally:
            # Set conservative memory settings
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.3'  # Reduced from 0.4
            os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.2'   # Reduced from 0.3
    
    except Exception as e:
        logger.error(f"Error during model loading: {str(e)}")
        raise e

def main():
    # Kill any existing Python processes
    os.system("pkill -9 Python")
    time.sleep(2)  # Wait for processes to be killed
    
    # Set conservative environment variables
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
    os.environ['MPS_USE_GUARD_MODE'] = '1'
    os.environ['MPS_ENABLE_MEMORY_GUARD'] = '1'
    os.environ['PYTORCH_MPS_SYNC_OPERATIONS'] = '1'
    os.environ['PYTORCH_MPS_AGGRESSIVE_MEMORY_CLEANUP'] = '1'
    
    # New: Set initial conservative memory limits
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.3'  # Start with conservative limit
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.2'

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because PyTorch was not built with MPS enabled")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device")
        return

    # Initial memory cleanup
    free_memory()

    # Parse arguments
    args = parse_args()
    
    # Force conservative settings
    args.video_size = [256, 384]  # Minimum size for initial load
    args.video_length = 33  # Minimum length
    args.precision = 'fp16'
    args.vae_precision = 'fp16'
    args.text_encoder_precision = 'fp16'
    args.disable_autocast = False
    args.vae_tiling = True
    
    # Set device to MPS
    device = torch.device("mps")
    
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)

    try:
        # Load models with staged loading
        hunyuan_video_sampler = staged_model_loading(models_root_path, args, device)
        args = hunyuan_video_sampler.args

        # Clear memory before inference
        free_memory()

        logger.info("Starting video generation with optimized chunked approach...")
        
        # Calculate optimal chunk size based on available memory
        total_ram = psutil.virtual_memory().total / (1024**3)  # Total RAM in GB
        if total_ram >= 64:
            chunk_size = 8  # Reduced from 16 for better memory management
            overlap = 2    # Reduced from 4 but still maintains smooth transitions
        else:
            chunk_size = 4  # Minimal chunk size
            overlap = 1    # Minimal overlap
        
        logger.info(f"Using chunk size: {chunk_size} frames with {overlap} frame overlap")
        
        outputs = generate_video_chunks(
            model=hunyuan_video_sampler,
            prompt=args.prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            chunk_size=chunk_size,
            overlap=overlap,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=25,
            guidance_scale=7.0,
            num_videos_per_prompt=1,
            flow_shift=args.flow_shift,
            batch_size=1,
            embedded_guidance_scale=args.embedded_cfg_scale
        )
        
        samples = outputs['samples']
        
        # Save samples
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample saved to: {save_path}')

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise e
    finally:
        # Final cleanup with multiple cycles
        for _ in range(3):
            free_memory()

if __name__ == "__main__":
    main()
