import os
import torch
import psutil
import sys
from pathlib import Path
from loguru import logger

def check_python_version():
    print("\n=== Python Version Check ===")
    required = (3, 10)
    current = sys.version_info[:2]
    if current >= required:
        print(f"✓ Python version {'.'.join(map(str, current))} meets requirement")
    else:
        print(f"✗ Python version {'.'.join(map(str, current))} does not meet minimum requirement of {'.'.join(map(str, required))}")

def check_torch_mps():
    print("\n=== PyTorch MPS Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        # Try to create a small tensor on MPS
        try:
            device = torch.device("mps")
            x = torch.ones(1, device=device)
            print("✓ Successfully created tensor on MPS device")
        except Exception as e:
            print(f"✗ Error creating tensor on MPS device: {str(e)}")
    else:
        print("✗ MPS is not available")

def check_memory():
    print("\n=== Memory Check ===")
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024 ** 3)
    available_gb = memory.available / (1024 ** 3)
    
    print(f"Total RAM: {total_gb:.1f} GB")
    print(f"Available RAM: {available_gb:.1f} GB")
    
    if total_gb >= 64:
        print("✓ Memory meets recommended requirement (64GB+)")
    elif total_gb >= 32:
        print("! Memory meets minimum requirement (32GB)")
    else:
        print("✗ Memory below minimum requirement (32GB)")

def check_model_weights():
    print("\n=== Model Weights Check ===")
    ckpts_dir = Path("ckpts")
    
    required_files = [
        "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        "vae/884-16c-hy.pt",
        "text_encoder/llm.pt"
    ]
    
    all_present = True
    for file in required_files:
        path = ckpts_dir / file
        if path.exists():
            size_gb = path.stat().st_size / (1024 ** 3)
            print(f"✓ Found {file} ({size_gb:.1f} GB)")
        else:
            print(f"✗ Missing {file}")
            all_present = False
    
    if not all_present:
        print("\nRun 'python download_weights.py' to download missing weights")

def check_environment_variables():
    print("\n=== Environment Variables Check ===")
    mps_ratio = os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.5')
    print(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO: {mps_ratio}")
    
    if float(mps_ratio) < 0.7:
        print("! Consider setting PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 for better performance")

def test_small_inference():
    print("\n=== Testing Small Inference ===")
    try:
        # Create a small random tensor
        device = torch.device("mps")
        x = torch.randn(1, 3, 32, 32, device=device)
        
        # Simple forward pass
        y = torch.nn.functional.interpolate(x, size=(64, 64))
        
        print("✓ Successfully ran small inference test")
    except Exception as e:
        print(f"✗ Error during inference test: {str(e)}")

def main():
    print("=== HunyuanVideo Setup Test ===")
    
    check_python_version()
    check_torch_mps()
    check_memory()
    check_model_weights()
    check_environment_variables()
    test_small_inference()
    
    print("\n=== Test Complete ===")
    print("For detailed system monitoring, run: python monitor_resources.py")
    print("To clean up temporary files, run: ./cleanup.sh")
    print("To start video generation, see examples in QUICKSTART.md")

if __name__ == "__main__":
    main()
