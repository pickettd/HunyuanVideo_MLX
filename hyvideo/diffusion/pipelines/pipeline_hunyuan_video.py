import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Union, List, Dict, Any, Tuple, Callable
import numpy as np
from dataclasses import dataclass
import torch
from loguru import logger

from hyvideo.vae.mlx_vae import MLXVAE
from hyvideo.utils.quantization import quantize_model_weights

# For VAE decoding
torch.set_grad_enabled(False)  # Ensure gradients are disabled

# Default quantization settings
DEFAULT_QUANT_CONFIG = {
    "enabled": False,
    "bits": 4,
    "exclude_modules": ["vae"]  # Don't quantize VAE by default
}

# Enable MLX optimizations and memory management
mx.set_default_device(mx.gpu)

def clear_mlx_cache():
    """Clear MLX memory cache"""
    # MLX automatically manages memory, but we can force garbage collection
    import gc
    gc.collect()

def get_mlx_activation(name: str):
    """Get MLX activation function"""
    if name.lower() == "silu":
        return nn.SiLU
    elif name.lower() == "gelu":
        return nn.GELU
    else:
        raise ValueError(f"Unknown activation function: {name}")

def get_timestep_embedding(timesteps, embedding_dim):
    """Create sinusoidal timestep embeddings."""
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = mx.log(mx.array([10000.0])) / (half_dim - 1)
    emb = mx.exp(mx.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=1)
    
    if embedding_dim % 2 == 1:  # zero pad if needed
        emb = mx.pad(emb, [(0, 0), (0, 1)])
        
    return emb

def mlx_rearrange_qkv(x: mx.array, heads_num: int) -> Tuple[mx.array, mx.array, mx.array]:
    """Custom rearrange function for QKV computation in MLX"""
    # Input shape: (batch_size, seq_len, 3 * heads_num * head_dim)
    batch_size, seq_len, _ = x.shape
    head_dim = x.shape[-1] // (3 * heads_num)
    
    # Reshape to (batch_size, seq_len, 3, heads_num, head_dim)
    x = x.reshape(batch_size, seq_len, 3, heads_num, head_dim)
    
    # Permute to (3, batch_size, seq_len, heads_num, head_dim)
    x = x.transpose(2, 0, 1, 3, 4)
    
    # Split into q, k, v
    q, k, v = x[0], x[1], x[2]
    
    return q, k, v

def compute_attention(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
    """Compute scaled dot-product attention"""
    # q, k, v shapes: (batch_size, seq_len, num_heads, head_dim)
    
    # Compute attention scores
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale
    attn = mx.softmax(attn, axis=-1)
    
    # Apply attention to values
    out = attn @ v
    
    # Reshape output to (batch_size, seq_len, num_heads * head_dim)
    batch_size, seq_len, num_heads, head_dim = out.shape
    out = out.reshape(batch_size, seq_len, num_heads * head_dim)
    
    return out

def to_mlx(data):
    """Convert numpy array or torch tensor to MLX array"""
    if data is None:
        return None
    if hasattr(data, 'numpy'):  # PyTorch tensor
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)
    return mx.array(data)

def from_mlx(array):
    """Convert MLX array to numpy array"""
    if array is None:
        return None
    # Convert to float32 and return numpy array
    return array.astype(mx.float32).numpy()

@dataclass
class HunyuanVideoPipelineOutput:
    """Pipeline output for video generation"""
    videos: np.ndarray
    seeds: List[int]
    prompts: List[str]

# Alias for backward compatibility
Output = HunyuanVideoPipelineOutput

class PatchEmbed(nn.Module):
    """Patch embedding layer for video"""
    def __init__(self, patch_size, in_channels, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * np.prod(patch_size), hidden_size)
        
    def __call__(self, x):
        # Handle 3D tensor reshaping for video
        B, C, T, H, W = x.shape
        P_t, P_h, P_w = self.patch_size
        
        # First reshape to separate patches in each dimension
        x = x.reshape(B, C, T//P_t, P_t, H//P_h, P_h, W//P_w, P_w)
        
        # Rearrange dimensions to group patches
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        
        # Flatten patches and combine channels
        x = x.reshape(B, (T//P_t) * (H//P_h) * (W//P_w), C * P_t * P_h * P_w)
        
        # Project to embedding dimension
        return self.proj(x)

class TextProjection(nn.Module):
    """Text projection layer"""
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
    def __call__(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class TimestepEmbedder(nn.Module):
    """Timestep embedding layer"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
    def __call__(self, timesteps):
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None]  # Add batch dimension
            
        # Get sinusoidal embedding
        emb = get_timestep_embedding(timesteps, self.hidden_size)
        
        # Project through MLP
        x = self.linear1(emb)
        x = self.act(x)
        x = self.linear2(x)
        return x

class MLPEmbedder(nn.Module):
    """MLP embedding layer"""
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
    def __call__(self, x):
        if x is None:
            # Return zero embedding if input is None
            return mx.zeros((1, self.hidden_size))
            
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class DoubleStreamBlock(nn.Module):
    """Double stream attention block for video generation"""
    def __init__(self, hidden_size, heads_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.head_dim = hidden_size // heads_num
        self.scale = self.head_dim ** -0.5
        
        # Image stream
        self.img_norm1 = nn.LayerNorm(hidden_size)
        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size)
        self.img_norm2 = nn.LayerNorm(hidden_size)
        
        # Image MLP
        self.img_mlp_fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.img_mlp_act = get_mlx_activation("gelu")()
        self.img_mlp_fc2 = nn.Linear(hidden_size * 4, hidden_size)
        
        # Text stream
        self.txt_norm1 = nn.LayerNorm(hidden_size)
        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size)
        self.txt_norm2 = nn.LayerNorm(hidden_size)
        
        # Text MLP
        self.txt_mlp_fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.txt_mlp_act = get_mlx_activation("gelu")()
        self.txt_mlp_fc2 = nn.Linear(hidden_size * 4, hidden_size)
        
    def __call__(self, img, txt, vec, freqs_cis=None):
        # Image attention
        img_norm = self.img_norm1(img)
        img_qkv = self.img_attn_qkv(img_norm)
        img_q, img_k, img_v = mlx_rearrange_qkv(img_qkv, self.heads_num)
        
        # Text attention
        txt_norm = self.txt_norm1(txt)
        txt_qkv = self.txt_attn_qkv(txt_norm)
        txt_q, txt_k, txt_v = mlx_rearrange_qkv(txt_qkv, self.heads_num)
        
        # Apply rotary embeddings if provided
        if freqs_cis is not None:
            freqs_cos, freqs_sin = freqs_cis
            # Get shapes for debugging
            B, L, H, D = img_q.shape  # batch, seq_len, heads, dim
            logger.info(f"Query shape: B={B}, L={L}, H={H}, D={D}")
            
            # Ensure frequencies have correct shape
            if len(freqs_cos.shape) == 2:
                logger.info(f"Original freq shape: {freqs_cos.shape}")
                # Add missing dimensions
                freqs_cos = freqs_cos.reshape(1, -1, 1, freqs_cos.shape[1])
                freqs_sin = freqs_sin.reshape(1, -1, 1, freqs_sin.shape[1])
                logger.info(f"Reshaped freq shape: {freqs_cos.shape}")
            
            _, F, _, FD = freqs_cos.shape
            logger.info(f"Freq dimensions: F={F}, FD={FD}")
            
            # Ensure freq dimensions match
            if FD != D:
                # Pad or truncate frequency dimensions if needed
                if FD < D:
                    pad_size = D - FD
                    freqs_cos = mx.pad(freqs_cos, [(0, 0), (0, 0), (0, 0), (0, pad_size)])
                    freqs_sin = mx.pad(freqs_sin, [(0, 0), (0, 0), (0, 0), (0, pad_size)])
                else:
                    freqs_cos = freqs_cos[..., :D]
                    freqs_sin = freqs_sin[..., :D]
            
            # Pad frequencies to match sequence length if needed
            if F < L:
                logger.info(f"Padding frequencies from {F} to {L}")
                pad_size = L - F
                freqs_cos = mx.pad(freqs_cos, [(0, 0), (0, pad_size), (0, 0), (0, 0)])
                freqs_sin = mx.pad(freqs_sin, [(0, 0), (0, pad_size), (0, 0), (0, 0)])
            elif F > L:
                logger.info(f"Truncating frequencies from {F} to {L}")
                freqs_cos = freqs_cos[:, :L]
                freqs_sin = freqs_sin[:, :L]
            
            # Broadcast to match batch size and heads
            freqs_cos = mx.broadcast_to(freqs_cos, (B, L, H, D))
            freqs_sin = mx.broadcast_to(freqs_sin, (B, L, H, D))
            
            # Apply rotary embeddings
            img_q = img_q * freqs_cos - img_q * freqs_sin
            img_k = img_k * freqs_cos - img_k * freqs_sin
        
        # Combined attention
        q = mx.concatenate([img_q, txt_q], axis=1)
        k = mx.concatenate([img_k, txt_k], axis=1)
        v = mx.concatenate([img_v, txt_v], axis=1)
        
        # Compute attention
        attn = compute_attention(q, k, v, self.scale)
        
        # Split attention results
        img_attn, txt_attn = mx.split(attn, [img.shape[1]], axis=1)
        
        # Image residual
        img = img + self.img_attn_proj(img_attn)
        img_mlp = self.img_mlp_fc2(self.img_mlp_act(self.img_mlp_fc1(self.img_norm2(img))))
        img = img + img_mlp
        
        # Text residual
        txt = txt + self.txt_attn_proj(txt_attn)
        txt_mlp = self.txt_mlp_fc2(self.txt_mlp_act(self.txt_mlp_fc1(self.txt_norm2(txt))))
        txt = txt + txt_mlp
        
        return img, txt

class SingleStreamBlock(nn.Module):
    """Single stream attention block for video generation"""
    def __init__(self, hidden_size, heads_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.head_dim = hidden_size // heads_num
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV and MLP
        self.norm = nn.LayerNorm(hidden_size)
        self.qkv_mlp = nn.Linear(hidden_size, hidden_size * 3 + hidden_size * 4)
        self.proj = nn.Linear(hidden_size + hidden_size * 4, hidden_size)
        self.mlp_act = get_mlx_activation("gelu")()
        
    def __call__(self, x, vec, txt_len, freqs_cis=None):
        x_norm = self.norm(x)
        qkv_mlp = self.qkv_mlp(x_norm)
        
        # Split QKV and MLP
        qkv, mlp = mx.split(qkv_mlp, [self.hidden_size * 3], axis=-1)
        q, k, v = mlx_rearrange_qkv(qkv, self.heads_num)
        
        # Apply rotary embeddings to image portion
        if freqs_cis is not None:
            freqs_cos, freqs_sin = freqs_cis
            img_q, txt_q = mx.split(q, [q.shape[1] - txt_len], axis=1)
            img_k, txt_k = mx.split(k, [k.shape[1] - txt_len], axis=1)
            
            # Get shapes for debugging
            B, L, H, D = img_q.shape  # batch, seq_len, heads, dim
            logger.info(f"Single Stream Query shape: B={B}, L={L}, H={H}, D={D}")
            
            # Ensure frequencies have correct shape
            if len(freqs_cos.shape) == 2:
                logger.info(f"Single Stream Original freq shape: {freqs_cos.shape}")
                # Add missing dimensions
                freqs_cos = freqs_cos.reshape(1, -1, 1, freqs_cos.shape[1])
                freqs_sin = freqs_sin.reshape(1, -1, 1, freqs_sin.shape[1])
                logger.info(f"Single Stream Reshaped freq shape: {freqs_cos.shape}")
            
            _, F, _, FD = freqs_cos.shape
            logger.info(f"Single Stream Freq dimensions: F={F}, FD={FD}")
            
            # Ensure freq dimensions match
            if FD != D:
                # Pad or truncate frequency dimensions if needed
                if FD < D:
                    pad_size = D - FD
                    freqs_cos = mx.pad(freqs_cos, [(0, 0), (0, 0), (0, 0), (0, pad_size)])
                    freqs_sin = mx.pad(freqs_sin, [(0, 0), (0, 0), (0, 0), (0, pad_size)])
                else:
                    freqs_cos = freqs_cos[..., :D]
                    freqs_sin = freqs_sin[..., :D]
            
            # Pad frequencies to match sequence length if needed
            if F < L:
                logger.info(f"Single Stream: Padding frequencies from {F} to {L}")
                pad_size = L - F
                freqs_cos = mx.pad(freqs_cos, [(0, 0), (0, pad_size), (0, 0), (0, 0)])
                freqs_sin = mx.pad(freqs_sin, [(0, 0), (0, pad_size), (0, 0), (0, 0)])
            elif F > L:
                logger.info(f"Single Stream: Truncating frequencies from {F} to {L}")
                freqs_cos = freqs_cos[:, :L]
                freqs_sin = freqs_sin[:, :L]
            
            # Broadcast to match batch size and heads
            freqs_cos = mx.broadcast_to(freqs_cos, (B, L, H, D))
            freqs_sin = mx.broadcast_to(freqs_sin, (B, L, H, D))
            
            # Apply rotary embeddings
            img_q = img_q * freqs_cos - img_q * freqs_sin
            img_k = img_k * freqs_cos - img_k * freqs_sin
            
            q = mx.concatenate([img_q, txt_q], axis=1)
            k = mx.concatenate([img_k, txt_k], axis=1)
        
        # Compute attention
        attn = compute_attention(q, k, v, self.scale)
        
        # MLP
        mlp = self.mlp_act(mlp)
        
        # Combine attention and MLP
        output = self.proj(mx.concatenate([attn, mlp], axis=-1))
        return x + output

class HunyuanVideoPipeline:
    """Pipeline for text-to-video generation using HunyuanVideo.
    
    This pipeline uses a hybrid approach:
    1. Text Encoding: PyTorch (text_encoder) → MLX conversion
    2. Core Generation: Pure MLX for optimal Metal performance
    3. VAE Decoding: MLX → PyTorch → MLX for compatibility
    4. Post-processing: Pure MLX
    """
    
    def __init__(
        self,
        vae,
        text_encoder,
        transformer,
        scheduler,
        text_encoder_2=None,
        args=None,
        quantization_config=None,
        **kwargs
    ):
        logger.info("Initializing pipeline with quantization support...")
        
        # Setup quantization config
        self.quant_config = {**DEFAULT_QUANT_CONFIG, **(quantization_config or {})}
        
        # Store components
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.scheduler = scheduler
        self.text_encoder_2 = text_encoder_2
        self.args = args
        
        # Copy transformer config first
        logger.info("Setting up transformer configuration...")
        self.hidden_size = transformer.config.hidden_size
        self.heads_num = transformer.config.heads_num
        self.patch_size = transformer.config.patch_size
        
        # Add projection layer for noise prediction after hidden_size is set
        self.proj = nn.Linear(self.hidden_size, transformer.config.in_channels)
        
        # Convert VAE to MLX format
        logger.info("Converting VAE to MLX format...")
        self.vae = MLXVAE(vae)
        
        logger.info("Computing VAE scale factor...")
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    
        # Create embedders
        logger.info("Creating embedders...")
        try:
            self.img_in = PatchEmbed(
                self.patch_size,
                transformer.config.in_channels,
                self.hidden_size
            )
            clear_mlx_cache()
            
            self.txt_in = TextProjection(
                args.text_states_dim,
                self.hidden_size
            )
            clear_mlx_cache()
            
            self.time_in = TimestepEmbedder(self.hidden_size)
            clear_mlx_cache()
            
            self.vector_in = MLPEmbedder(args.text_states_dim_2, self.hidden_size)
            clear_mlx_cache()
            
            self.use_mlx = True
            self._setup_mlx_models()
            
            # Apply quantization if enabled
            if self.quant_config["enabled"]:
                logger.info(f"Applying {self.quant_config['bits']}-bit quantization...")
                
                # Quantize transformer blocks
                if "transformer" not in self.quant_config.get("exclude_modules", []):
                    for block in self.double_blocks:
                        quantize_model_weights(block, bits=self.quant_config["bits"])
                    for block in self.single_blocks:
                        quantize_model_weights(block, bits=self.quant_config["bits"])
                    logger.info("Transformer blocks quantized")
                
                # Quantize embedders if not excluded
                if "embedders" not in self.quant_config.get("exclude_modules", []):
                    quantize_model_weights(self.img_in, bits=self.quant_config["bits"])
                    quantize_model_weights(self.txt_in, bits=self.quant_config["bits"])
                    quantize_model_weights(self.time_in, bits=self.quant_config["bits"])
                    quantize_model_weights(self.vector_in, bits=self.quant_config["bits"])
                    logger.info("Embedders quantized")
                
                # Clear cache after quantization
                clear_mlx_cache()
        except Exception as e:
            logger.error(f"Error initializing embedders: {str(e)}")
            raise
        
    def _setup_mlx_models(self):
        """Convert key components to MLX format for faster inference"""
        # Create double stream blocks
        self.double_blocks = [
            DoubleStreamBlock(self.hidden_size, self.heads_num)
            for _ in range(self.transformer.config.mm_double_blocks_depth)
        ]
        
        # Create single stream blocks
        self.single_blocks = [
            SingleStreamBlock(self.hidden_size, self.heads_num)
            for _ in range(self.transformer.config.mm_single_blocks_depth)
        ]
        
        try:
            # Copy weights from PyTorch model to MLX with cleanup
            logger.info("Copying double stream block weights...")
            for i, (mlx_block, pt_block) in enumerate(zip(self.double_blocks, self.transformer.double_blocks)):
                try:
                    # Copy image stream weights in chunks
                    logger.info(f"Processing double stream block {i+1}/{len(self.double_blocks)}")
                    
                    # Image attention weights
                    mlx_block.img_attn_qkv.weight = to_mlx(pt_block.img_attn_qkv.weight.cpu())
                    clear_mlx_cache()
                    mlx_block.img_attn_proj.weight = to_mlx(pt_block.img_attn_proj.weight.cpu())
                    clear_mlx_cache()
                    
                    # Image MLP weights
                    mlx_block.img_mlp_fc1.weight = to_mlx(pt_block.img_mlp.fc1.weight.cpu())
                    clear_mlx_cache()
                    mlx_block.img_mlp_fc2.weight = to_mlx(pt_block.img_mlp.fc2.weight.cpu())
                    clear_mlx_cache()
                    
                    # Text attention weights
                    mlx_block.txt_attn_qkv.weight = to_mlx(pt_block.txt_attn_qkv.weight.cpu())
                    clear_mlx_cache()
                    mlx_block.txt_attn_proj.weight = to_mlx(pt_block.txt_attn_proj.weight.cpu())
                    clear_mlx_cache()
                    
                    # Text MLP weights
                    mlx_block.txt_mlp_fc1.weight = to_mlx(pt_block.txt_mlp.fc1.weight.cpu())
                    clear_mlx_cache()
                    mlx_block.txt_mlp_fc2.weight = to_mlx(pt_block.txt_mlp.fc2.weight.cpu())
                    clear_mlx_cache()
                    
                except Exception as e:
                    logger.error(f"Error copying double stream block {i} weights: {str(e)}")
                    raise
            
            logger.info("Copying single stream block weights...")
            for i, (mlx_block, pt_block) in enumerate(zip(self.single_blocks, self.transformer.single_blocks)):
                try:
                    logger.info(f"Processing single stream block {i+1}/{len(self.single_blocks)}")
                    mlx_block.qkv_mlp.weight = to_mlx(pt_block.linear1.weight.cpu())
                    clear_mlx_cache()
                    mlx_block.proj.weight = to_mlx(pt_block.linear2.weight.cpu())
                    clear_mlx_cache()
                except Exception as e:
                    logger.error(f"Error copying single stream block {i} weights: {str(e)}")
                    raise
                
        except Exception as e:
            print(f"Error copying weights: {str(e)}")
            raise e
        finally:
            clear_mlx_cache()
            
    def encode_prompt(self, prompt, num_videos_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None):
        """MLX-optimized prompt encoding with memory management"""
        try:
            with torch.no_grad():
                # Convert prompt to embeddings
                text_inputs = self.text_encoder.text2tokens(prompt)
                prompt_embeds_pt = self.text_encoder.encode(text_inputs).hidden_state
                prompt_embeds = to_mlx(prompt_embeds_pt)
                del prompt_embeds_pt  # Clear PyTorch memory immediately
                
                # Handle negative prompt
                if do_classifier_free_guidance:
                    uncond_tokens = "" if negative_prompt is None else negative_prompt
                    uncond_input = self.text_encoder.text2tokens([uncond_tokens])
                    negative_prompt_embeds_pt = self.text_encoder.encode(uncond_input).hidden_state
                    negative_prompt_embeds = to_mlx(negative_prompt_embeds_pt)
                    del negative_prompt_embeds_pt  # Clear PyTorch memory immediately
                    
                    # Concatenate embeddings in MLX
                    prompt_embeds = mx.concatenate([negative_prompt_embeds, prompt_embeds], axis=0)
                    del negative_prompt_embeds  # Clear MLX memory
                
                # Clear PyTorch memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                
                return prompt_embeds
                
        except Exception as e:
            print(f"Error encoding prompt: {str(e)}")
            raise e
        finally:
            # Clear MLX cache
            clear_mlx_cache()
        
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        latents: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        **kwargs
    ):
        """MLX-optimized generation pipeline."""
        try:
            # 1. Text Encoding (PyTorch → MLX)
            try:
                with torch.no_grad():
                    # Get text embeddings from PyTorch text encoder
                    text_inputs = self.text_encoder.text2tokens([prompt] if isinstance(prompt, str) else prompt)
                    prompt_embeds_pt = self.text_encoder.encode(text_inputs).hidden_state
                    
                    if guidance_scale > 1.0:
                        uncond_tokens = [""] if negative_prompt is None else [negative_prompt]
                        uncond_input = self.text_encoder.text2tokens(uncond_tokens)
                        negative_prompt_embeds_pt = self.text_encoder.encode(uncond_input).hidden_state
                        prompt_embeds_pt = torch.cat([negative_prompt_embeds_pt, prompt_embeds_pt])
                    
                    # Convert to MLX
                    prompt_embeds = to_mlx(prompt_embeds_pt)
            except Exception as e:
                print(f"Error in text encoding: {str(e)}")
                raise e
            finally:
                # Clear PyTorch memory
                if 'prompt_embeds_pt' in locals():
                    del prompt_embeds_pt
                if 'negative_prompt_embeds_pt' in locals():
                    del negative_prompt_embeds_pt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            # 2. Prepare timesteps in MLX format
            timesteps = mx.array(list(range(num_inference_steps)))
            
            # 3. Prepare latent variables in MLX format
            # Ensure dimensions are divisible by patch size
            P_t, P_h, P_w = self.patch_size
            
            # Adjust dimensions to be divisible by patch size
            adjusted_height = ((height // self.vae_scale_factor) // P_h) * P_h
            adjusted_width = ((width // self.vae_scale_factor) // P_w) * P_w
            adjusted_length = ((video_length) // P_t) * P_t
            
            logger.info(f"Adjusting dimensions to be divisible by patch size:")
            logger.info(f"Height: {height // self.vae_scale_factor} -> {adjusted_height}")
            logger.info(f"Width: {width // self.vae_scale_factor} -> {adjusted_width}")
            logger.info(f"Length: {video_length} -> {adjusted_length}")
            
            shape = (
                1,  # batch_size
                self.transformer.config.in_channels,
                adjusted_length,
                adjusted_height,
                adjusted_width,
            )
            
            if latents is None:
                # Generate random latents
                latents = mx.random.normal(shape=shape, loc=0.0, scale=1.0)
            else:
                latents = to_mlx(latents)
                
            # Scale latents
            latents = latents * timesteps[0]
            
            # 4. Denoising loop with MLX operations
            for i, t in enumerate(timesteps):
                try:
                    # Handle classifier free guidance
                    latent_model_input = mx.concatenate([latents, latents], axis=0) if guidance_scale > 1.0 else latents
                    
                    # Process in chunks to manage memory
                    chunk_size = 4  # Process 4 frames at a time
                    img_chunks = []
                    
                    for i in range(0, latent_model_input.shape[2], chunk_size):
                        try:
                            # Process smaller chunk
                            chunk = latent_model_input[:, :, i:i+chunk_size]
                            img_chunk = self.img_in(chunk)
                            img_chunks.append(img_chunk)
                            
                            # Aggressive cleanup after each chunk
                            del chunk
                            clear_mlx_cache()
                            
                        except Exception as e:
                            logger.error(f"Error processing chunk {i}: {str(e)}")
                            raise
                    
                    # Combine chunks
                    img = mx.concatenate(img_chunks, axis=1)
                    del img_chunks  # Free memory
                    clear_mlx_cache()
                    
                    # Process text and time embeddings
                    txt = self.txt_in(prompt_embeds)
                    vec = self.time_in(t)
                    vec = vec + self.vector_in(to_mlx(kwargs.get('text_states_2')))
                    clear_mlx_cache()
                    
                    # Double stream blocks
                    for block in self.double_blocks:
                        img, txt = block(img, txt, vec, kwargs.get('freqs_cis'))
                        
                    # Single stream blocks
                    x = mx.concatenate([img, txt], axis=1)
                    txt_len = txt.shape[1]
                    for block in self.single_blocks:
                        x = block(x, vec, txt_len, kwargs.get('freqs_cis'))
                        
                    # Get image portion and apply guidance
                    img = x[:, :img.shape[1]]
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = mx.split(img, 2, axis=0)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = img
                        
                    # Log shapes for debugging
                    logger.info(f"Before reshape - noise_pred: {noise_pred.shape}, latents: {latents.shape}")
                    
                    # Get latents shape
                    B, C, T, H, W = latents.shape
                    
                    # Project noise prediction to match VAE latent channels
                    logger.info(f"Initial noise_pred shape: {noise_pred.shape}")
                    noise_pred = noise_pred.reshape(-1, self.hidden_size)  # Flatten to 2D
                    logger.info(f"After flatten shape: {noise_pred.shape}")
                    noise_pred = self.proj(noise_pred)  # Project to correct channel dim
                    logger.info(f"After projection shape: {noise_pred.shape}")
                    
                    # Calculate spatial dimensions
                    total_spatial = T * H * W
                    spatial_per_batch = noise_pred.shape[0] // B
                    
                    # Ensure dimensions match
                    if spatial_per_batch != total_spatial:
                        logger.warning(f"Spatial dimensions mismatch: got {spatial_per_batch}, expected {total_spatial}")
                        # Interpolate noise_pred to match expected dimensions
                        noise_pred = noise_pred.reshape(B, -1, C)  # First reshape to batch
                        # Resize to match spatial dimensions
                        noise_pred = mx.repeat(noise_pred, total_spatial // noise_pred.shape[1], axis=1)
                        # Truncate any extra elements
                        noise_pred = noise_pred[:, :total_spatial, :]
                    else:
                        noise_pred = noise_pred.reshape(B, total_spatial, C)
                    
                    logger.info(f"After spatial adjustment: {noise_pred.shape}")
                    noise_pred = noise_pred.transpose(0, 2, 1)  # B, C, THW
                    logger.info(f"After transpose: {noise_pred.shape}")
                    noise_pred = noise_pred.reshape(B, C, T, H, W)  # Final reshape to match latents
                    logger.info(f"Final shape: {noise_pred.shape}")
                    logger.info(f"Target latents shape: {latents.shape}")
                    
                    # Update latents with broadcasting
                    alpha = 1.0 / ((t + 1) ** 0.5)
                    alpha = mx.array(alpha, dtype=latents.dtype)
                    latents = (latents - (1 - alpha) * noise_pred) / alpha
                    
                    # Clear MLX cache periodically
                    if i % 5 == 0:
                        clear_mlx_cache()
                        
                except Exception as e:
                    print(f"Error in denoising step {i}: {str(e)}")
                    raise e
            
            # 5. Post-processing with MLX
            try:
                if output_type == "latent":
                    videos = from_mlx(latents)
                else:
                    # VAE Decoding with MLX
                    try:
                        # Always process in smaller chunks for VAE
                        chunk_size = 8  # Smaller chunk size for VAE
                        chunks = []
                        
                        for i in range(0, latents.shape[2], chunk_size):
                            try:
                                # Process chunk
                                chunk = latents[:, :, i:i+chunk_size]
                                chunk_decoded = self.vae.decode(chunk)["sample"]
                                chunks.append(chunk_decoded)
                                
                                # Aggressive cleanup
                                del chunk
                                clear_mlx_cache()
                                
                            except Exception as e:
                                logger.error(f"Error in VAE decoding chunk {i}: {str(e)}")
                                raise
                        
                        # Combine chunks with cleanup
                        videos = mx.concatenate(chunks, axis=2)
                        del chunks
                        clear_mlx_cache()
                        
                        # Convert to numpy
                        videos = from_mlx(videos)
                            
                    except Exception as e:
                        print(f"Error in VAE decoding: {str(e)}")
                        raise e
                
                # Convert to numpy array
                videos = np.array(videos)
            except Exception as e:
                print(f"Error in post-processing: {str(e)}")
                raise e
            
            # Convert videos to PyTorch tensor if needed
            if output_type == "pil":
                videos = torch.from_numpy(videos)
                if torch.backends.mps.is_available():
                    videos = videos.to("mps")
            
            # Return with metadata
            return HunyuanVideoPipelineOutput(
                videos=videos,
                seeds=[kwargs.get('seed', -1)],  # Use provided seed or -1
                prompts=[prompt] if isinstance(prompt, str) else prompt
            )
            
        except Exception as e:
            print(f"Error in generation: {str(e)}")
            raise e
        finally:
            # Final cleanup
            clear_mlx_cache()
