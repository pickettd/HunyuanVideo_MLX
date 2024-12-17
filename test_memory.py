import os
import torch
from loguru import logger
from hyvideo.utils.memory_utils import (
    optimize_memory_settings,
    log_memory_status,
    check_memory_feasibility,
    suggest_optimal_settings,
    clear_memory
)

def test_memory_configuration():
    """Test memory configuration and suggest optimal settings"""
    logger.info("Testing memory configuration...")
    
    # 1. Optimize memory settings
    optimize_memory_settings()
    
    # 2. Clear existing memory
    clear_memory()
    
    # 3. Log initial memory status
    logger.info("\nInitial memory status:")
    log_memory_status()
    
    # 4. Get optimal settings
    settings = suggest_optimal_settings()
    logger.info("\nRecommended settings for your system:")
    logger.info(f"Resolution: {settings['resolution']}")
    logger.info(f"Video Length: {settings['video_length']}")
    logger.info(f"Chunk Size: {settings['chunk_size']}")
    logger.info(f"Batch Size: {settings['batch_size']}")
    logger.info(f"Precision: {settings['precision']}")
    
    # 5. Test memory allocation
    logger.info("\nTesting memory allocation...")
    height, width = settings['resolution']
    feasible, message = check_memory_feasibility(height, width, settings['video_length'])
    
    if feasible:
        logger.info("Memory test passed! You can proceed with video generation using these settings.")
        
        # Test small tensor allocation
        try:
            logger.info("Testing tensor allocation...")
            test_size = (1, 4, settings['video_length'], height//8, width//8)  # Smaller test tensor
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
            # Try allocating a test tensor
            test_tensor = torch.zeros(test_size, device=device)
            del test_tensor
            clear_memory()
            
            logger.info("Tensor allocation test passed!")
            
            return True, settings
            
        except RuntimeError as e:
            logger.error(f"Tensor allocation failed: {str(e)}")
            logger.info("Suggesting even more conservative settings...")
            
            # Suggest more conservative settings
            settings['resolution'] = (256, 384)
            settings['video_length'] = 33
            settings['chunk_size'] = 4
            
            return False, settings
    else:
        logger.warning(f"Memory test failed: {message}")
        logger.info("Suggesting minimum viable settings...")
        
        # Suggest minimum settings
        settings['resolution'] = (256, 384)
        settings['video_length'] = 33
        settings['chunk_size'] = 4
        
        return False, settings

if __name__ == "__main__":
    success, recommended_settings = test_memory_configuration()
    
    if success:
        logger.info("\nRecommended command:")
        cmd = f"""python sample_video_mps.py \\
    --mmgp-mode \\
    --mmgp-config configs/mmgp_mlx.json \\
    --video-size {recommended_settings['resolution'][0]} {recommended_settings['resolution'][1]} \\
    --video-length {recommended_settings['video_length']} \\
    --precision {recommended_settings['precision']} \\
    --prompt "your prompt here"
"""
        logger.info(cmd)
    else:
        logger.warning("\nPlease try with minimum settings or free up more memory.")
