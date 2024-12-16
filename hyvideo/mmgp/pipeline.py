import os
import mlx.core as mx
from pathlib import Path
from typing import Dict, Optional, Union, List
from loguru import logger

from ..diffusion.pipelines.pipeline_hunyuan_video import HunyuanVideoPipeline
from .config import MMGPConfig, ModelConfig

class MLXMMGPPipeline:
    """Multi-Model Generation Pipeline for HunyuanVideo optimized for Apple Silicon"""
    
    def __init__(
        self,
        config: MMGPConfig,
        precision: str = "float16",  # MLX default precision
        use_mps: bool = True  # Use Metal Performance Shaders
    ):
        self.config = config
        self.precision = precision
        self.use_mps = use_mps
        self.current_model: Optional[str] = None
        self.pipeline: Optional[HunyuanVideoPipeline] = None
        
        # Configure MLX for optimal Apple Silicon performance
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
        
    def load_model(self, model_config: ModelConfig) -> None:
        """Load a specific model into the pipeline with MLX optimizations"""
        if self.current_model == model_config.name:
            return  # Model already loaded
            
        logger.info(f"Loading model: {model_config.name} ({model_config.description})")
        
        # Unload current model if any
        if self.pipeline is not None:
            del self.pipeline
            mx.clear_memory_pool()  # MLX memory management
            
        # Load new model
        checkpoint_path = Path(model_config.checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Model checkpoint not found: {checkpoint_path}")
            
        # Configure pipeline with MLX optimizations
        self.pipeline = HunyuanVideoPipeline.from_pretrained(
            checkpoint_path,
            precision=self.precision,
            use_mps=self.use_mps
        )
        
        self.current_model = model_config.name
        
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        video_length: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        callback=None,
        **kwargs
    ) -> mx.array:
        """Generate video using multiple models with MLX optimizations"""
        
        # Validate total steps matches schedule
        total_steps = sum(
            step["end_step"] - step["start_step"] 
            for step in self.config.schedule
        )
        if total_steps != num_inference_steps:
            raise ValueError(
                f"Total steps in schedule ({total_steps}) does not match "
                f"requested inference steps ({num_inference_steps})"
            )
            
        # Initialize generation
        latents = None
        current_step = 0
        
        # Set random seed if provided
        if seed is not None:
            mx.random.seed(seed)
            
        # Generate through each model in schedule
        for schedule_item in self.config.schedule:
            model_name = schedule_item["model"]
            model_config = self.config.models[model_name]
            start_step = schedule_item["start_step"]
            end_step = schedule_item["end_step"]
            steps_for_model = end_step - start_step
            
            # Load appropriate model
            self.load_model(model_config)
            
            # Configure memory usage based on model resolution
            if "540p" in model_name:
                # More aggressive memory optimization for lower resolution
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
            else:
                # Conservative memory usage for higher resolution
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
            
            logger.info(
                f"Generating steps {start_step}-{end_step} with {model_name}"
            )
            
            # MLX-optimized generation
            result = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                video_length=video_length,
                num_inference_steps=steps_for_model,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                latents=latents,  # Pass previous latents for continuation
                callback=callback,
                precision=self.precision,
                use_mps=self.use_mps,
                **kwargs
            )
            
            # Update latents for next model
            latents = result.videos
            current_step = end_step
            
            # Clear MLX memory after each model
            mx.clear_memory_pool()
            
            # Progress callback
            if callback:
                callback(current_step, num_inference_steps, latents)
                
        return latents
        
    @classmethod
    def from_config_file(
        cls,
        config_path: str,
        precision: str = "float16",
        use_mps: bool = True
    ) -> "MLXMMGPPipeline":
        """Create MLX-optimized pipeline from a config file"""
        config = MMGPConfig.from_json(config_path)
        return cls(config, precision=precision, use_mps=use_mps)
