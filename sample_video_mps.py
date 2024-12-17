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
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

def aggressive_memory_cleanup():
    """More aggressive memory cleanup routine"""
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clear MPS cache with multiple cycles
    if hasattr(torch.mps, 'empty_cache'):
        for _ in range(5):  # Increased cycles
            torch.mps.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                time.sleep(0.1)  # Small delay between cycles
    
    # Multiple GC cycles
    for _ in range(5):  # Increased cycles
        gc.collect()
        time.sleep(0.1)  # Small delay between cycles
    
    # Force Python garbage collection
    gc.collect(generation=2)

def get_system_memory():
    """Get current system memory usage"""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / (1024**3),  # GB
        'available': memory.available / (1024**3),  # GB
        'percent': memory.percent
    }

def staged_model_loading(models_root_path, args, device):
    """Enhanced staged model loading with better memory management"""
    try:
        logger.info("Stage 1: Initial setup and aggressive memory cleanup...")
        aggressive_memory_cleanup()
        
        memory_info = get_system_memory()
        logger.info(f"Available memory before loading: {memory_info['available']:.2f}GB")
        
        # Stage 2: Disable MPS memory limits for initial loading
        logger.info("Stage 2: Disabling MPS memory limits...")
        original_high = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.3')
        original_low = os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.2')
        
        # Disable memory limits temporarily for loading
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
        
        # Force minimal settings
        args.batch_size = 1
        args.precision = 'fp16'
        args.vae_precision = 'fp16'
        args.text_encoder_precision = 'fp16'
        args.text_encoder_precision_2 = 'fp16'
        args.disable_autocast = False
        args.vae_tiling = True
        
        try:
            logger.info("Stage 3: Attempting model loading with disabled memory limits...")
            hunyuan_video_sampler = None
            
            # Multiple attempts with increasing delays between tries
            for attempt in range(3):
                try:
                    aggressive_memory_cleanup()
                    time.sleep(2)  # Increased delay before attempt
                    
                    # Set default device type for autocast
                    torch.set_default_device(device)
                    torch.set_default_dtype(torch.float16)
                    
                    # Enable autocast for mixed precision
                    with torch.autocast(device_type='mps', dtype=torch.float16, enabled=True):
                        hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
                            models_root_path,
                            args=args,
                            device=device
                        )
                    
                    logger.info("Model loaded successfully!")
                    break
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) and attempt < 2:
                        logger.warning(f"OOM on attempt {attempt + 1}, cleaning up and retrying...")
                        aggressive_memory_cleanup()
                        time.sleep(3)  # Increased delay between retries
                    else:
                        raise e
            
            if hunyuan_video_sampler is None:
                raise RuntimeError("Failed to load model after multiple attempts")
            
            # Force synchronization and cleanup
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            aggressive_memory_cleanup()
            
            return hunyuan_video_sampler
            
        finally:
            # Keep memory limits disabled if loading was successful
            if hunyuan_video_sampler is not None:
                logger.info("Stage 4: Keeping memory limits disabled for generation...")
            else:
                # Restore original memory settings only if loading failed
                logger.info("Stage 4: Restoring original memory settings...")
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = original_high
                os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = original_low
    
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
    
    # Disable MPS memory limits
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because PyTorch was not built with MPS enabled")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device")
        return

    # Initial memory cleanup
    aggressive_memory_cleanup()

    # Parse arguments
    args = parse_args()
    
    # Set device to MPS
    device = torch.device("mps")
    
    # Set default device type and dtype
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float16)
    
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)

    try:
        # Load models with enhanced staged loading
        hunyuan_video_sampler = staged_model_loading(models_root_path, args, device)
        args = hunyuan_video_sampler.args

        # Clear memory before inference
        aggressive_memory_cleanup()

        logger.info("Starting video generation...")
        
        # Get system memory info
        memory_info = get_system_memory()
        logger.info(f"Total RAM: {memory_info['total']:.2f}GB, Available: {memory_info['available']:.2f}GB")
        
        # Generate video with autocast
        with torch.autocast(device_type='mps', dtype=torch.float16, enabled=True):
            outputs = hunyuan_video_sampler.predict(
                prompt=args.prompt,
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
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
        for _ in range(5):
            aggressive_memory_cleanup()

if __name__ == "__main__":
    main()
