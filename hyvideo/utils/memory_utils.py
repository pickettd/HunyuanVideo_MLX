import os
import gc
import torch
import psutil
from loguru import logger
from typing import Optional, Dict, Any
import numpy as np

def clear_memory(device: Optional[torch.device] = None):
    """Aggressively clear memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    
    if device is not None and str(device).startswith('mps'):
        torch.mps.synchronize()
    
    gc.collect()
    gc.collect()

def get_memory_status() -> Dict[str, float]:
    """Get detailed memory usage information"""
    process = psutil.Process()
    mem_info = process.memory_info()
    
    return {
        'process_rss': mem_info.rss / (1024 * 1024 * 1024),  # GB
        'process_vms': mem_info.vms / (1024 * 1024 * 1024),  # GB
        'system_available': psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
        'system_total': psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
    }

def log_memory_status(prefix: str = ""):
    """Log current memory status"""
    mem_status = get_memory_status()
    logger.info(f"{prefix} Memory Status:")
    logger.info(f"  Process RSS: {mem_status['process_rss']:.2f}GB")
    logger.info(f"  Process VMS: {mem_status['process_vms']:.2f}GB")
    logger.info(f"  System Available: {mem_status['system_available']:.2f}GB")
    logger.info(f"  System Total: {mem_status['system_total']:.2f}GB")

def shard_tensor(tensor: torch.Tensor, num_shards: int = 2) -> list:
    """Split a tensor into shards for memory efficiency"""
    return torch.chunk(tensor, num_shards)

def load_sharded_tensor(filename: str, device: torch.device, num_shards: int = 2) -> torch.Tensor:
    """Load a large tensor in shards"""
    shards = []
    for i in range(num_shards):
        shard = torch.load(f"{filename}.{i}", map_location='cpu')
        shards.append(shard.to(device))
        clear_memory(device)
    
    return torch.cat(shards, dim=0)

def optimize_memory_settings():
    """Set optimal memory management settings"""
    # Conservative memory settings
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.15'
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.1'
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
    os.environ['MPS_USE_GUARD_MODE'] = '1'
    os.environ['MPS_ENABLE_MEMORY_GUARD'] = '1'
    os.environ['PYTORCH_MPS_SYNC_OPERATIONS'] = '1'
    os.environ['PYTORCH_MPS_AGGRESSIVE_MEMORY_CLEANUP'] = '1'
    
    # Log current settings
    logger.info("Memory optimization settings:")
    for key in ['PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO']:
        logger.info(f"  {key}: {os.getenv(key)}")

def estimate_memory_requirements(height: int, width: int, video_length: int, batch_size: int = 1) -> Dict[str, float]:
    """Estimate memory requirements for video generation"""
    # Basic memory estimations in GB
    vae_memory = (height * width * video_length * batch_size * 4 * 4) / (1024 * 1024 * 1024)  # 4 channels, 4 bytes per float
    transformer_memory = (height * width * video_length * batch_size * 16) / (1024 * 1024 * 1024)  # Rough transformer estimate
    
    total_estimate = vae_memory + transformer_memory
    
    return {
        'vae_memory': vae_memory,
        'transformer_memory': transformer_memory,
        'total_estimate': total_estimate
    }

def check_memory_feasibility(height: int, width: int, video_length: int) -> tuple[bool, str]:
    """Check if video generation is feasible with current memory"""
    mem_status = get_memory_status()
    mem_estimate = estimate_memory_requirements(height, width, video_length)
    
    available_memory = mem_status['system_available']
    required_memory = mem_estimate['total_estimate']
    
    if required_memory > available_memory * 0.8:  # Leave 20% buffer
        return False, f"Insufficient memory. Need {required_memory:.1f}GB, have {available_memory:.1f}GB available"
    
    return True, "Memory requirements look feasible"

def suggest_optimal_settings() -> Dict[str, Any]:
    """Suggest optimal settings based on available memory"""
    mem_status = get_memory_status()
    available_memory = mem_status['system_available']
    
    # Conservative settings for different memory ranges
    if available_memory >= 48:  # High memory
        return {
            'resolution': (544, 960),
            'video_length': 65,
            'chunk_size': 16,
            'batch_size': 1,
            'precision': 'fp16'
        }
    elif available_memory >= 32:  # Medium memory
        return {
            'resolution': (384, 640),
            'video_length': 33,
            'chunk_size': 8,
            'batch_size': 1,
            'precision': 'fp16'
        }
    else:  # Low memory
        return {
            'resolution': (256, 384),
            'video_length': 33,
            'chunk_size': 4,
            'batch_size': 1,
            'precision': 'fp16'
        }
