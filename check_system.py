import mlx.core as mx
import platform
import psutil
import os

def check_system():
    """Check system requirements for running HunyuanVideo."""
    print("\nChecking system requirements...")
    
    # Check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    
    # Check MLX version
    mlx_version = mx.__version__
    print(f"MLX version: {mlx_version}")
    
    # Check MLX device
    device = mx.default_device()
    print(f"MLX device: {device}")
    
    # Check Metal availability (Apple Silicon)
    metal_available = mx.metal.is_available()
    print(f"Metal available: {metal_available}")
    
    # Check system memory
    memory = psutil.virtual_memory()
    print(f"\nSystem memory:")
    print(f"  Total: {memory.total / 1024**3:.1f} GB")
    print(f"  Available: {memory.available / 1024**3:.1f} GB")
    print(f"  Used: {memory.used / 1024**3:.1f} GB ({memory.percent}%)")
    
    # Check disk space
    disk = psutil.disk_usage(os.getcwd())
    print(f"\nDisk space:")
    print(f"  Total: {disk.total / 1024**3:.1f} GB")
    print(f"  Free: {disk.free / 1024**3:.1f} GB")
    print(f"  Used: {disk.used / 1024**3:.1f} GB ({disk.percent}%)")

if __name__ == "__main__":
    check_system()
