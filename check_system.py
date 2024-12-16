import torch
import platform
import psutil
import os
from pathlib import Path

def check_mps():
    """Check MPS availability and configuration"""
    print("\n=== MPS Configuration ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("\n⚠️  MPS not available because PyTorch was not built with MPS enabled")
        else:
            print("\n⚠️  MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device")

def check_system():
    """Check system specifications"""
    print("\n=== System Information ===")
    print(f"OS: {platform.system()} {platform.mac_ver()[0]}")
    print(f"Processor: {platform.processor()}")
    
    # Check if running on Apple Silicon
    is_arm = platform.processor() == 'arm'
    print(f"Apple Silicon: {'Yes' if is_arm else 'No'}")
    
    # Memory information
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024 ** 3)
    print(f"Total RAM: {memory_gb:.1f} GB")
    print(f"Available RAM: {memory.available / (1024 ** 3):.1f} GB")
    
    if memory_gb < 32:
        print("\n⚠️  Warning: Less than 32GB RAM detected. This may impact performance.")
        print("    Recommended: 32GB minimum, 64GB for higher resolutions")

def check_model_weights():
    """Check if model weights are downloaded"""
    print("\n=== Model Weights ===")
    ckpts_dir = Path("ckpts")
    
    required_files = [
        "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        "vae/884-16c-hy.pt",
        "text_encoder/llm.pt"
    ]
    
    missing_files = []
    for file in required_files:
        if not (ckpts_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("\n⚠️  Missing model weights:")
        for file in missing_files:
            print(f"    - {file}")
        print("\nRun 'python download_weights.py' to download the required model weights")
    else:
        print("✓ All required model weights are present")

def main():
    print("=== HunyuanVideo System Check ===")
    
    check_mps()
    check_system()
    check_model_weights()
    
    print("\n=== Environment Variables ===")
    mps_ratio = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.5')
    print(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO: {mps_ratio}")
    
    if float(mps_ratio) < 0.7:
        print("\nTip: For better performance, you may want to set:")
        print("export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7")

if __name__ == "__main__":
    main()
