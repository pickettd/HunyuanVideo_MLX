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
        default=None,
        help="Path to MMGP configuration file",
    )
    return parser

def main():
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because PyTorch was not built with MPS enabled")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device")
        return

    # Parse arguments
    parser = argparse.ArgumentParser(description="HunyuanVideo inference script")
    parser = add_mmgp_args(parser)
    args = parse_args(namespace=parser.parse_args([]))
    print(args)
    
    # Override CUDA-specific settings
    args.precision = "fp32"  # MPS works best with fp32
    args.use_cpu_offload = False  # CPU offload not needed for MPS
    args.vae_precision = "fp32"
    args.text_encoder_precision = "fp32"
    args.disable_autocast = True  # Disable autocast as it's not supported on MPS
    
    # Set device to MPS
    device = torch.device("mps")
    
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, 
        args=args,
        device=device
    )
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
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

if __name__ == "__main__":
    main()
