import time
import torch
import numpy as np
from loguru import logger

from hyvideo.inference import HunyuanVideo
from hyvideo.utils.memory_utils import print_memory_usage

def benchmark_generation(
    model,
    prompt="A beautiful sunset over the ocean",
    height=540,
    width=960,
    video_length=16,
    num_runs=3,
    guidance_scale=7.0,
    num_inference_steps=25,
    seed=42
):
    """Benchmark video generation performance"""
    generation_times = []
    memory_usages = []
    
    for i in range(num_runs):
        logger.info(f"\nRun {i+1}/{num_runs}")
        
        # Clear memory before each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        # Record initial memory
        initial_memory = print_memory_usage()
        
        # Generate video and time it
        start_time = time.time()
        outputs = model.predict(
            prompt=prompt,
            height=height,
            width=width,
            video_length=video_length,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed
        )
        end_time = time.time()
        
        # Record final memory
        final_memory = print_memory_usage()
        
        # Calculate metrics
        generation_time = end_time - start_time
        memory_used = final_memory - initial_memory
        
        generation_times.append(generation_time)
        memory_usages.append(memory_used)
        
        logger.info(f"Generation time: {generation_time:.2f}s")
        logger.info(f"Memory used: {memory_used:.2f}GB")
    
    # Calculate statistics
    avg_time = np.mean(generation_times)
    std_time = np.std(generation_times)
    avg_memory = np.mean(memory_usages)
    std_memory = np.std(memory_usages)
    
    logger.info("\nBenchmark Results:")
    logger.info(f"Average generation time: {avg_time:.2f}s ± {std_time:.2f}s")
    logger.info(f"Average memory usage: {avg_memory:.2f}GB ± {std_memory:.2f}GB")
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'avg_memory': avg_memory,
        'std_memory': std_memory,
        'times': generation_times,
        'memories': memory_usages
    }

def main():
    # Load model
    logger.info("Loading model...")
    model = HunyuanVideo.from_pretrained("ckpts")
    
    # Run benchmark
    results = benchmark_generation(model)
    
    # Print detailed results
    logger.info("\nDetailed Results:")
    logger.info(f"Generation times: {[f'{t:.2f}s' for t in results['times']]}")
    logger.info(f"Memory usages: {[f'{m:.2f}GB' for m in results['memories']]}")

if __name__ == "__main__":
    main()
