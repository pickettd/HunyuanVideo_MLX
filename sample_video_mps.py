import os
import json
import argparse
import mlx.core as mx
import numpy as np
from loguru import logger
import imageio
from pathlib import Path

from hyvideo.utils.memory_utils import print_memory_usage
from hyvideo.inference import HunyuanVideo

def load_mlx_config(config_path: str = "configs/mlx_config.json"):
    """Load MLX configuration with defaults"""
    if not os.path.exists(config_path):
        logger.warning(f"Config not found at {config_path}, using hardcoded defaults")
        return None
        
    with open(config_path) as f:
        return json.load(f)

def main():
    # Load MLX config
    config = load_mlx_config()
    
    parser = argparse.ArgumentParser()
    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--video-size", type=int, nargs=2, 
                       default=config["generation_defaults"]["video_size"] if config else [544, 960])
    parser.add_argument("--video-length", type=int,
                       default=config["generation_defaults"]["video_length"] if config else 13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float,
                       default=config["generation_defaults"]["guidance_scale"] if config else 7.0)
    parser.add_argument("--num-inference-steps", type=int,
                       default=config["generation_defaults"]["num_inference_steps"] if config else 40)
    parser.add_argument("--save-path", type=str, default="results")
    
    # Model configuration
    model_settings = config["model_settings"] if config else {}
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"],
                       default=model_settings.get("precision", "fp16"))
    parser.add_argument("--model", type=str,
                       default=model_settings.get("model", "HYVideo-T/2"))
    parser.add_argument("--text-states-dim", type=int,
                       default=model_settings.get("text_states_dim", 4096))
    parser.add_argument("--text-states-dim-2", type=int,
                       default=model_settings.get("text_states_dim_2", 768))
    parser.add_argument("--vae", type=str,
                       default=model_settings.get("vae", "884-16c-hy"))
    parser.add_argument("--text-encoder", type=str,
                       default=model_settings.get("text_encoder", "llm"))
    parser.add_argument("--tokenizer", type=str,
                       default=model_settings.get("tokenizer", "llm"))
    parser.add_argument("--text-len", type=int,
                       default=model_settings.get("text_len", 256))
    parser.add_argument("--denoise-type", type=str,
                       default=model_settings.get("denoise_type", "flow"))
    parser.add_argument("--flow-shift", type=float,
                       default=model_settings.get("flow_shift", 5.0))
    parser.add_argument("--flow-solver", type=str,
                       default=model_settings.get("flow_solver", "midpoint"))
    parser.add_argument("--latent-channels", type=int,
                       default=model_settings.get("latent_channels", 16))
    parser.add_argument("--hidden-size", type=int,
                       default=model_settings.get("hidden_size", 3072))
    parser.add_argument("--heads-num", type=int,
                       default=model_settings.get("heads_num", 24))
    parser.add_argument("--mlp-width-ratio", type=float,
                       default=model_settings.get("mlp_width_ratio", 4.0))
    parser.add_argument("--rope-theta", type=float,
                       default=model_settings.get("rope_theta", 10000.0))
    
    # Additional settings
    parser.add_argument("--hidden-state-skip-layer", type=int, default=0)
    parser.add_argument("--apply-final-norm", action="store_true", default=False)
    parser.add_argument("--reproduce", action="store_true", default=False)
    parser.add_argument("--use-cpu-offload", action="store_true", default=False)
    
    # MLX optimization settings
    mlx_settings = config["mlx_optimization"]["precision_settings"] if config else {}
    parser.add_argument("--text-encoder-precision", type=str,
                       default=mlx_settings.get("text_encoder", "fp16"))
    parser.add_argument("--vae-precision", type=str,
                       default=mlx_settings.get("vae", "fp16"))
    parser.add_argument("--vae-tiling", action="store_true", default=True)
    
    # Add prompt template arguments
    parser.add_argument("--prompt-template", type=str, default=None)
    parser.add_argument("--prompt-template-video", type=str, default=None)
    parser.add_argument("--text-encoder-2", type=str, default=None)
    parser.add_argument("--text-encoder-precision-2", type=str, default="fp16")
    parser.add_argument("--tokenizer-2", type=str, default=None)
    parser.add_argument("--text-len-2", type=int, default=256)
    parser.add_argument("--flow-reverse", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Apply MLX memory optimizations if config exists
    if config and "memory_settings" in config["mlx_optimization"]:
        mem_settings = config["mlx_optimization"]["memory_settings"]
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(mem_settings["high_watermark_ratio"])
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(mem_settings["low_watermark_ratio"])
        os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = mem_settings["allocator_policy"]
        os.environ["MPS_USE_GUARD_MODE"] = "1" if mem_settings["use_guard_mode"] else "0"
        os.environ["MPS_ENABLE_MEMORY_GUARD"] = "1" if mem_settings["enable_memory_guard"] else "0"
        os.environ["PYTORCH_MPS_SYNC_OPERATIONS"] = "1" if mem_settings["sync_operations"] else "0"
        os.environ["PYTORCH_MPS_AGGRESSIVE_MEMORY_CLEANUP"] = "1" if mem_settings["aggressive_cleanup"] else "0"
    
    # Print generation parameters
    logger.info("Starting video generation...")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Video size: {args.video_size}")
    logger.info(f"Video length: {args.video_length}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Guidance scale: {args.guidance_scale}")
    logger.info(f"Inference steps: {args.num_inference_steps}")
    
    try:
        # Print initial memory state
        available_memory = print_memory_usage()
        logger.info(f"Available memory before loading: {available_memory:.2f}GB")
        
        # Set random seed
        mx.random.seed(args.seed)
        np.random.seed(args.seed)
        
        # Initialize MLX pipeline with memory cleanup
        logger.info("Initializing MLX pipeline...")
        checkpoint_path = "ckpts"
        
        # Clear memory before initialization
        import gc, torch
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        pipeline = HunyuanVideo.from_pretrained(checkpoint_path, args)
        
        # Clear memory after initialization
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        # Generate video
        logger.info("Generating video...")
        outputs = pipeline.predict(
            prompt=args.prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            guidance_scale=args.guidance_scale,
            infer_steps=args.num_inference_steps,
            seed=args.seed
        )
        
        # Save video
        logger.info("Saving video...")
        video = outputs.videos[0]  # First video in batch
        save_path = os.path.join(args.save_path, f"{args.seed}.mp4")
        
        # Convert to uint8 range [0, 255]
        video = ((video + 1) * 127.5).astype(np.uint8)
        
        # Save frames as video
        imageio.mimsave(save_path, [frame for frame in video], fps=8)
        logger.info(f"Video saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
