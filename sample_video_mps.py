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
import json

def check_mps_settings():
    """Verify MPS settings before running"""
    required_vars = {
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.4',
        'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.3',
        'PYTORCH_MPS_ALLOCATOR_POLICY': 'garbage_collection',
        'MPS_USE_GUARD_MODE': '1',
        'MPS_ENABLE_MEMORY_GUARD': '1',
        'PYTORCH_MPS_SYNC_OPERATIONS': '1'
    }
    
    for var, default_value in required_vars.items():
        if not os.getenv(var):
            logger.warning(f"{var} not set! Setting to default: {default_value}")
            os.environ[var] = default_value
    
    try:
        high_ratio = float(os.getenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO'))
        low_ratio = float(os.getenv('PYTORCH_MPS_LOW_WATERMARK_RATIO'))
        
        if high_ratio > 1.0 or low_ratio > 1.0 or high_ratio < low_ratio:
            logger.error(f"Invalid watermark ratios: high={high_ratio}, low={low_ratio}")
            logger.info("High ratio should be <= 1.0 and greater than low ratio")
            return False
            
    except ValueError:
        logger.error("Invalid watermark ratio values")
        return False
    
    return True

def clear_memory():
    """Aggressively clear memory"""
    if torch.backends.mps.is_available():
        # Force synchronous operations to complete
        torch.mps.synchronize()
        # Clear MPS cache
        torch.mps.empty_cache()
    # Force garbage collection
    gc.collect()
    # Additional garbage collection cycle
    gc.collect()

def load_mmgp_config(config_path):
    """Load and validate MMGP configuration"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['models', 'schedule', 'notes']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in MMGP config")
        
        return config
    except Exception as e:
        logger.error(f"Error loading MMGP config: {str(e)}")
        raise

def get_mac_model_settings(config):
    """Get settings based on available RAM"""
    total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3)  # GB
    
    if total_ram >= 64:
        return config['notes']['memory_optimization']['M3_Max_64GB']['settings']
    else:
        return config['notes']['memory_optimization']['M3_Max_32GB']['settings']

def map_precision(precision_str):
    """Map precision string to correct format"""
    precision_map = {
        'float32': 'fp32',
        'float16': 'fp16',
        'bfloat16': 'bf16'
    }
    return precision_map.get(precision_str, 'fp16')  # Default to fp16 if unknown

def staged_model_loading(models_root_path, args, device):
    """Load models in stages with memory clearing between each stage"""
    try:
        logger.info("Stage 1: Initial setup...")
        clear_memory()
        
        logger.info("Stage 2: Loading model with conservative memory settings...")
        # Temporarily lower watermark ratio during model loading
        original_high_ratio = os.getenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
        original_low_ratio = os.getenv('PYTORCH_MPS_LOW_WATERMARK_RATIO')
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.4'
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.3'
        
        try:
            hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
                models_root_path, 
                args=args,
                device=device
            )
            clear_memory()
            
            logger.info("Model loaded successfully!")
            return hunyuan_video_sampler
            
        finally:
            # Restore original watermark ratios
            if original_high_ratio:
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = original_high_ratio
            if original_low_ratio:
                os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = original_low_ratio
    
    except Exception as e:
        logger.error(f"Error during model loading: {str(e)}")
        if "out of memory" in str(e):
            logger.error("\nMemory error during model loading. Try these steps:")
            logger.info("1. Close all other applications")
            logger.info("2. Restart your Python environment")
            logger.info("3. Set even lower watermark ratios in .env:")
            logger.info("   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3")
            logger.info("   PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2")
            logger.info("4. If issues persist, try reducing model precision further")
        raise e

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
    args = parse_args()
    
    # Handle MMGP mode
    if args.mmgp_mode:
        logger.info("Loading MMGP configuration...")
        mmgp_config = load_mmgp_config(args.mmgp_config)
        settings = get_mac_model_settings(mmgp_config)
        
        # Apply MMGP settings with correct precision mapping
        precision = map_precision(settings.get('precision', 'float16'))
        args.precision = precision
        args.vae_precision = precision
        args.text_encoder_precision = precision
        args.disable_autocast = False
        args.vae_tiling = True
        
        logger.info(f"Using precision: {precision}")
        
        if 'batch_processing' in settings and settings['batch_processing'] == 'enabled':
            args.batch_size = 1  # Start conservative
    else:
        # Use default memory-efficient settings
        args.precision = "fp16"
        args.vae_precision = "fp16"
        args.text_encoder_precision = "fp16"
        args.disable_autocast = False
        args.vae_tiling = True
    
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
        # Load models with staged loading
        hunyuan_video_sampler = staged_model_loading(models_root_path, args, device)
        
        # Get the updated args
        args = hunyuan_video_sampler.args

        # Clear memory before inference
        clear_memory()

        logger.info("Starting video generation...")
        # Start sampling with optimized settings
        outputs = hunyuan_video_sampler.predict(
            prompt=args.prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=25,  # Further reduced steps for memory efficiency
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
            logger.info(f'Sample saved to: {save_path}')

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("\nOut of memory error occurred. Try these steps:")
            logger.info("1. Close all other applications")
            logger.info("2. Reduce video resolution (e.g., --video-size 384 640)")
            logger.info("3. Reduce video length")
            logger.info("4. Lower watermark ratios in .env")
            logger.info("5. If issues persist, try restarting your Python environment")
        raise e
    finally:
        # Clean up memory
        clear_memory()

if __name__ == "__main__":
    main()
