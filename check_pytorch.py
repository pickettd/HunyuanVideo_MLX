import torch
import os

print("Environment variables:")
print(f"PYTORCH_MPS_HIGH_WATERMARK_RATIO: {os.getenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO')}")
print(f"PYTORCH_MPS_LOW_WATERMARK_RATIO: {os.getenv('PYTORCH_MPS_LOW_WATERMARK_RATIO')}")

print("\nPyTorch configuration:")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if hasattr(torch.backends.mps, 'get_watermark_ratios'):
    high, low = torch.backends.mps.get_watermark_ratios()
    print(f"\nCurrent watermark ratios:")
    print(f"High: {high}")
    print(f"Low: {low}")
