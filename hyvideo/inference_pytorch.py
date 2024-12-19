import os
import gc
import time
import random
import functools
from typing import List, Optional, Tuple, Union
from collections import OrderedDict

from pathlib import Path
from loguru import logger

import torch
import torch.distributed as dist
import mlx.core as mx
import mlx.nn as nn

from hyvideo.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE
from hyvideo.vae import load_vae
from hyvideo.modules import load_model
from hyvideo.text_encoder import TextEncoder
from hyvideo.utils.data_utils import align_to
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline

# Enable MLX optimizations
mx.set_default_device(mx.gpu)

def aggressive_memory_cleanup():
    """Aggressively clear memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.mps, 'empty_cache'):
        for _ in range(3):
            torch.mps.synchronize()
            torch.mps.empty_cache()
    for _ in range(3):
        gc.collect()

class Inference(object):
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=None,
        logger=None,
    ):
        self.vae = vae
        self.vae_kwargs = vae_kwargs
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.model = model
        self.pipeline = pipeline
        self.use_cpu_offload = use_cpu_offload
        self.args = args
        # Prioritize MPS over CPU
        self.device = device if device is not None else (
            torch.device("mps") if torch.backends.mps.is_available() 
            else torch.device("cpu")
        )
        self.logger = logger

    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, **kwargs):
        """Initialize the Inference pipeline."""
        logger.info(f"Got text-to-video model root path: {pretrained_model_path}")
        
        # Prioritize MPS over CPU
        if device is None:
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # Disable gradient
        torch.set_grad_enabled(False)

        # Build main model
        logger.info("Building model...")
        factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
        in_channels = args.latent_channels
        out_channels = args.latent_channels

        try:
            # Clear memory before model loading
            aggressive_memory_cleanup()
            
            model = load_model(
                args,
                in_channels=in_channels,
                out_channels=out_channels,
                factor_kwargs=factor_kwargs,
            )
            model = model.to(device)
            
            # Load model weights with memory management
            model = cls.load_state_dict(args, model, pretrained_model_path)
            model.eval()

            # Build VAE with memory management
            aggressive_memory_cleanup()
            vae, _, s_ratio, t_ratio = load_vae(
                args.vae,
                args.vae_precision,
                logger=logger,
                device=device if not args.use_cpu_offload else "cpu",
            )
            vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

            # Text encoder settings
            if args.prompt_template_video is not None:
                crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
            elif args.prompt_template is not None:
                crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
            else:
                crop_start = 0
            max_length = args.text_len + crop_start

            prompt_template = PROMPT_TEMPLATE[args.prompt_template] if args.prompt_template is not None else None
            prompt_template_video = PROMPT_TEMPLATE[args.prompt_template_video] if args.prompt_template_video is not None else None

            # Build text encoder with memory management
            aggressive_memory_cleanup()
            text_encoder = TextEncoder(
                text_encoder_type=args.text_encoder,
                max_length=max_length,
                text_encoder_precision=args.text_encoder_precision,
                tokenizer_type=args.tokenizer,
                prompt_template=prompt_template,
                prompt_template_video=prompt_template_video,
                hidden_state_skip_layer=args.hidden_state_skip_layer,
                apply_final_norm=args.apply_final_norm,
                reproduce=args.reproduce,
                logger=logger,
                device=device if not args.use_cpu_offload else "cpu",
            )
            
            text_encoder_2 = None
            if args.text_encoder_2 is not None:
                aggressive_memory_cleanup()
                text_encoder_2 = TextEncoder(
                    text_encoder_type=args.text_encoder_2,
                    max_length=args.text_len_2,
                    text_encoder_precision=args.text_encoder_precision_2,
                    tokenizer_type=args.tokenizer_2,
                    reproduce=args.reproduce,
                    logger=logger,
                    device=device if not args.use_cpu_offload else "cpu",
                )

            return cls(
                args=args,
                vae=vae,
                vae_kwargs=vae_kwargs,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                model=model,
                use_cpu_offload=args.use_cpu_offload,
                device=device,
                logger=logger,
            )
            
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
            raise

    @staticmethod
    def load_state_dict(args, model, pretrained_model_path):
        """Load model weights with safe handling"""
        try:
            # Construct correct model path
            model_dir = Path(pretrained_model_path) / "hunyuan-video-t2v-720p" / "transformers"
            model_files = list(model_dir.glob("*_model_states.pt"))
            
            if not model_files:
                raise ValueError(f"No model weights found in {model_dir}")
                
            model_path = model_files[0]
            if len(model_files) > 1:
                logger.warning(f"Multiple model weights found in {model_dir}, using {model_path}")

            logger.info(f"Loading torch model {model_path}...")
            
            # Load with safe settings
            state_dict = torch.load(
                model_path,
                map_location='cpu',  # Load to CPU first
                weights_only=True    # Safe loading mode
            )

            # Handle DataParallel prefix
            if "module" in state_dict:
                state_dict = state_dict["module"]
            
            # Remove DataParallel prefix if present in keys
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # Remove 'module.' prefix
                new_state_dict[k] = v

            # Move to target device after loading
            model.load_state_dict(new_state_dict, strict=True)
            return model
            
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
            raise

class HunyuanVideoSampler(Inference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = self.load_diffusion_pipeline(
            args=self.args,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            model=self.model,
            device=self.device,
        )
        self.default_negative_prompt = NEGATIVE_PROMPT

    def load_diffusion_pipeline(self, args, vae, text_encoder, text_encoder_2, model, scheduler=None, device=None, progress_bar_config=None):
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        pipeline = HunyuanVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            progress_bar_config=progress_bar_config,
            args=args,
        )
        
        if self.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        return pipeline

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size)
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(s % self.model.patch_size[idx] == 0 for idx, s in enumerate(latents_size))
            rope_sizes = [s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes

        head_dim = self.model.hidden_size // self.model.heads_num
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
            
        assert sum(rope_dim_list) == head_dim

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.args.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def predict(self, prompt, height=192, width=336, video_length=129, seed=None,
               negative_prompt=None, infer_steps=50, guidance_scale=6, flow_shift=5.0,
               embedded_guidance_scale=None, batch_size=1, num_videos_per_prompt=1, **kwargs):
        """Generate video from text prompt"""
        out_dict = {}

        # Handle seeds
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 1_000_000) for _ in range(batch_size * num_videos_per_prompt)]
        elif isinstance(seed, int):
            seeds = [seed + i for _ in range(batch_size) for i in range(num_videos_per_prompt)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) + j for i in range(batch_size) for j in range(num_videos_per_prompt)]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(f"Invalid seed length: {len(seed)}")
        else:
            raise ValueError(f"Invalid seed type: {type(seed)}")
            
        generator = [torch.Generator(self.device).manual_seed(s) for s in seeds]
        out_dict["seeds"] = seeds

        # Validate dimensions
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}x{video_length}")
        if video_length != 1 and (video_length - 1) % 4 != 0:
            raise ValueError(f"`video_length` has to be 1 or a multiple of 4 but is {video_length}.")

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length
        out_dict["size"] = (target_height, target_width, target_video_length)

        # Handle prompts
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be string, got {type(prompt)}")
        prompt = [prompt.strip()]

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(f"negative_prompt must be string, got {type(negative_prompt)}")
        negative_prompt = [negative_prompt.strip()]

        # Setup scheduler
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver
        )
        self.pipeline.scheduler = scheduler

        # Get position embeddings
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            target_video_length, target_height, target_width
        )
        n_tokens = freqs_cos.shape[0]

        # Log generation parameters
        logger.debug(f"""
            height: {target_height}
            width: {target_width}
            video_length: {target_video_length}
            prompt: {prompt}
            negative_prompt: {negative_prompt}
            seed: {seed}
            steps: {infer_steps}
            videos_per_prompt: {num_videos_per_prompt}
            guidance_scale: {guidance_scale}
            n_tokens: {n_tokens}
            flow_shift: {flow_shift}
            embedded_guidance_scale: {embedded_guidance_scale}
        """)

        # Generate
        try:
            start_time = time.time()
            samples = self.pipeline(
                prompt=prompt,
                height=target_height,
                width=target_width,
                video_length=target_video_length,
                num_inference_steps=infer_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                output_type="pil",
                freqs_cis=(freqs_cos, freqs_sin),
                n_tokens=n_tokens,
                embedded_guidance_scale=embedded_guidance_scale,
                data_type="video" if target_video_length > 1 else "image",
                is_progress_bar=True,
                vae_ver=self.args.vae,
                enable_tiling=self.args.vae_tiling,
            )[0]
            
            out_dict["samples"] = samples
            out_dict["prompts"] = prompt

            gen_time = time.time() - start_time
            logger.info(f"Generation successful, time: {gen_time:.2f}s")
            
            return out_dict
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
