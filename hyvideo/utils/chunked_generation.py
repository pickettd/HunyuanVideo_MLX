import torch
import gc
from typing import Dict, Any, Optional
from loguru import logger

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

def generate_video_chunks(
    model,
    prompt: str,
    height: int,
    width: int,
    video_length: int,
    chunk_size: int = 16,  # Process 16 frames at a time
    overlap: int = 4,      # 4 frame overlap for smooth transitions
    **kwargs
) -> Dict[str, Any]:
    """
    Generate video in chunks to reduce memory usage.
    
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
    logger.info(f"Generating video in chunks: {video_length} frames total, {chunk_size} frames per chunk")
    
    # Calculate number of chunks needed
    effective_chunk_size = chunk_size - overlap
    num_chunks = (video_length - overlap) // effective_chunk_size
    if (video_length - overlap) % effective_chunk_size != 0:
        num_chunks += 1
    
    # Initialize list to store chunk outputs
    chunk_samples = []
    all_seeds = []
    
    for i in range(num_chunks):
        logger.info(f"Processing chunk {i+1}/{num_chunks}")
        
        # Calculate chunk start and end frames
        start_frame = i * effective_chunk_size
        end_frame = min(start_frame + chunk_size, video_length)
        current_chunk_size = end_frame - start_frame
        
        # Modify prompt for temporal context
        if i > 0:
            chunk_prompt = f"{prompt} (continuing from previous segment)"
        else:
            chunk_prompt = prompt
            
        # Generate chunk with lower memory settings
        try:
            # Temporarily lower watermark ratios
            original_high = torch.mps.get_mem_high_watermark_ratio()
            original_low = torch.mps.get_mem_low_watermark_ratio()
            
            torch.mps.set_mem_high_watermark_ratio(0.3)  # Conservative setting for chunk processing
            torch.mps.set_mem_low_watermark_ratio(0.2)
            
            # Generate chunk
            chunk_output = model.predict(
                prompt=chunk_prompt,
                height=height,
                width=width,
                video_length=current_chunk_size,
                **kwargs
            )
            
            # Restore original watermark ratios
            torch.mps.set_mem_high_watermark_ratio(original_high)
            torch.mps.set_mem_low_watermark_ratio(original_low)
            
            chunk_samples.append(chunk_output['samples'][0])
            all_seeds.extend(chunk_output['seeds'])
            
            # Clear memory after each chunk
            clear_memory()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM error in chunk {i+1}. Trying with smaller chunk...")
                # Could implement recursive chunk size reduction here
                raise
            else:
                raise
    
    # Blend overlapping frames between chunks
    final_samples = []
    for i in range(len(chunk_samples) - 1):
        current_chunk = chunk_samples[i]
        next_chunk = chunk_samples[i + 1]
        
        # Keep all frames except overlap region from current chunk
        final_samples.append(current_chunk[:-overlap])
        
        # Blend overlapping frames
        overlap_start = -overlap
        for j in range(overlap):
            alpha = j / overlap
            blended_frame = (1 - alpha) * current_chunk[overlap_start + j] + alpha * next_chunk[j]
            final_samples.append(blended_frame.unsqueeze(0))
    
    # Add remaining frames from last chunk
    final_samples.append(chunk_samples[-1][overlap:])
    
    # Concatenate all frames
    final_video = torch.cat(final_samples, dim=0)
    
    return {
        'samples': [final_video],
        'seeds': all_seeds,
        'prompts': [prompt]
    }
