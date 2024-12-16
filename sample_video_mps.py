import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

import torch

def main():
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because PyTorch was not built with MPS enabled")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device")
        return

    args = parse_args()
    print(args)
    
    # Override CUDA-specific settings
    args.precision = "fp32"  # MPS works best with fp32
    args.use_cpu_offload = False  # CPU offload not needed for MPS
    args.vae_precision = "fp32"
    args.text_encoder_precision = "fp32"
    args.disable_autocast = True  # Disable autocast as it's not supported on MPS
    
    # Set device to MPS
    device = torch.device("mps")
    
    # Handle MMGP settings
    if args.mmgp_mode:
        if not args.mmgp_config:
            raise ValueError("MMGP config file must be provided when using MMGP mode")
        
        logger.info(f"Using MMGP mode with config: {args.mmgp_config}")
        
        # Load MMGP config
        import json
        with open(args.mmgp_config, 'r') as f:
            mmgp_config = json.load(f)
        
        # Validate MMGP config
        required_keys = ['models', 'schedule']
        if not all(key in mmgp_config for key in required_keys):
            raise ValueError(f"MMGP config must contain all of: {required_keys}")
        
        # Create models root path
        models_root_paths = {}
        for model_name, model_path in mmgp_config['models'].items():
            model_path = Path(model_path)
            if not model_path.exists():
                raise ValueError(f"Model path not exists: {model_path}")
            models_root_paths[model_name] = model_path
        
        # Create samplers for each model
        samplers = {}
        for model_name, model_path in models_root_paths.items():
            logger.info(f"Loading model: {model_name} from {model_path}")
            samplers[model_name] = HunyuanVideoSampler.from_pretrained(
                model_path,
                args=args,
                device=device
            )
        
        # Process generation schedule
        schedule = mmgp_config['schedule']
        current_latents = None
        current_seeds = None
        
        for stage in schedule:
            model_name = stage['model']
            start_step = stage.get('start_step', 0)
            end_step = stage.get('end_step', args.infer_steps)
            
            logger.info(f"Running stage with model {model_name} from step {start_step} to {end_step}")
            
            # Update inference steps for this stage
            stage_steps = end_step - start_step
            
            outputs = samplers[model_name].predict(
                prompt=args.prompt,
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
                seed=args.seed if current_seeds is None else current_seeds,
                negative_prompt=args.neg_prompt,
                infer_steps=stage_steps,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=args.num_videos,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale,
                start_step=start_step,
                latents=current_latents
            )
            
            # Update latents and seeds for next stage
            current_latents = outputs.get('latents', None)
            current_seeds = outputs.get('seeds', None)
            samples = outputs['samples']
        
    else:
        # Original single-model pipeline
        models_root_path = Path(args.model_base)
        if not models_root_path.exists():
            raise ValueError(f"`models_root` not exists: {models_root_path}")
        
        hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
            models_root_path, 
            args=args,
            device=device
        )
        
        args = hunyuan_video_sampler.args
        
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
    
    # Create save folder
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # Save samples
    for i, sample in enumerate(samples):
        sample = samples[i].unsqueeze(0)
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
        save_videos_grid(sample, save_path, fps=24)
        logger.info(f'Sample save to: {save_path}')

if __name__ == "__main__":
    main()
