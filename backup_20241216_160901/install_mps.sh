#!/bin/bash

# Create and activate conda environment
echo "Creating conda environment..."
conda create -n HunyuanVideo-MPS python=3.10.9 -y

# Activate environment (note: in a script this won't persist, user needs to activate manually after)
echo "Please run: conda activate HunyuanVideo-MPS"

# Install PyTorch with MPS support
echo "Installing PyTorch with MPS support..."
conda install pytorch torchvision torchaudio -c pytorch-nightly

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements_mps.txt

# Install flash attention v2 (CPU version since we're not using CUDA)
echo "Installing flash-attention..."
pip install ninja
pip install flash-attention --no-build-isolation

echo "Installation complete! Please activate the environment with: conda activate HunyuanVideo-MPS"
echo "Then run: python sample_video_mps.py to generate videos"
