import torch
import gc
import os
import psutil
from typing import Dict, Any, Optional
from loguru import logger

def clear_memory():
    """Aggressively clear memory with multiple cycles"""
    if torch.backends.mps.is_available():
        # Force multiple synchronization cycles
        for _ in range(3):
            torch.mps.synchronize()
            torch.mps.empty_cache()
            torch.mps.synchronize()
    
    # Multiple garbage collection cycles
    for _ in range(3):
        gc.collect()

def generate_video_chunks(
    model,
    prompt: str,
    height: int,
    width: int,
    video_length: int,
    chunk_size: int = 8,  # Use 8-frame chunks (>= 4 and multiple of 4)
    overlap: int = 2,     # Use 2 frame overlap for 8-frame chunks
    **kwargs
) -> Dict[str, Any]:
    """
    Generate video in chunks with enhanced memory management.
    
    Args:
        model: HunyuanVideoSampler instance
        prompt: Text prompt for generation
        height: Video height
        width: Video width
        video_length: Total number of frames
        chunk_size: Number of frames to process at once
        overlap: Number of overlapping frames between chunks
        **kwargs: Additional arguments passed to model.predict
    
    Returns:
        Dict containing generated video samples and metadata
    """
    # Force chunk size to be 8 (>= 4 and multiple of 4)
    chunk_size = 8
    overlap = min(2, overlap)  # Limit overlap to 2 for 8-frame chunks
    
    logger.info(f"Generating video in chunks: {video_length} frames total, {chunk_size} frames per chunk")
    
    # Calculate number of chunks needed
    effective_chunk_size = chunk_size - overlap
    num_chunks = (video_length - overlap) // effective_chunk_size
    if (video_length - overlap) % effective_chunk_size != 0:
        num_chunks += 1
    
    # Initialize list to store chunk outputs
    chunk_samples = []
    all_seeds = []
    
    # Store original environment variables
    original_high = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
    original_low = os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.0')
    
    try:
        for i in range(num_chunks):
            logger.info(f"Processing chunk {i+1}/{num_chunks}")
            
            # Calculate chunk start and end frames
            start_frame = i * effective_chunk_size
            end_frame = min(start_frame + chunk_size, video_length)
            current_chunk_size = end_frame - start_frame
            
            # Ensure chunk size is exactly 8 frames
            if current_chunk_size != 8:
                current_chunk_size = 8
            
            # Modify prompt for temporal context
            if i > 0:
                chunk_prompt = f"{prompt} (continuing from previous segment)"
            else:
                chunk_prompt = prompt
            
            # Clear memory before chunk processing
            clear_memory()
                
            try:
                # Set conservative memory limits for chunk processing
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable limits for generation
                os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
                
                # Generate chunk
                chunk_output = model.predict(
                    prompt=chunk_prompt,
                    height=height,
                    width=width,
                    video_length=current_chunk_size,
                    **kwargs
                )
                
                chunk_samples.append(chunk_output['samples'][0])
                all_seeds.extend(chunk_output['seeds'])
                
                # Clear memory after chunk processing
                clear_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM error in chunk {i+1}. Attempting recovery...")
                    
                    # Aggressive memory cleanup
                    clear_memory()
                    
                    # Cannot reduce chunk size further since we need 8 frames
                    raise RuntimeError("Cannot reduce chunk size further. Please try with smaller resolution.")
                else:
                    raise
    
    finally:
        # Restore original environment variables
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = original_high
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = original_low
    
    # Blend overlapping frames between chunks
    final_samples = []
    for i in range(len(chunk_samples) - 1):
        current_chunk = chunk_samples[i]
        next_chunk = chunk_samples[i + 1]
        
        # Keep all frames except overlap region from current chunk
        final_samples.append(current_chunk[:-overlap])
        
        # Blend overlapping frames with linear interpolation
        overlap_start = -overlap
        for j in range(overlap):
            alpha = j / overlap
            blended_frame = (1 - alpha) * current_chunk[overlap_start + j] + alpha * next_chunk[j]
            final_samples.append(blended_frame.unsqueeze(0))
        
        # Clear intermediate tensors
        del current_chunk
        clear_memory()
    
    # Add remaining frames from last chunk
    final_samples.append(chunk_samples[-1][overlap:])
    
    # Clear chunk samples to free memory
    del chunk_samples
    clear_memory()
    
    # Concatenate all frames
    final_video = torch.cat(final_samples, dim=0)
    
    return {
        'samples': [final_video],
        'seeds': all_seeds,
        'prompts': [prompt]
    }
