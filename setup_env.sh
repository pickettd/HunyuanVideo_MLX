#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up environment for HunyuanVideo...${NC}"

# Set MPS high watermark ratio for better memory management
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
echo -e "${GREEN}✓${NC} Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7"

# Enable MPS debug mode if needed
# export PYTORCH_MPS_DEBUG=1
# echo -e "${GREEN}✓${NC} Enabled MPS debug mode"

# Set maximum copy size (in bytes) - 1GB
export PYTORCH_MPS_MAX_COPY_SIZE=1000000000
echo -e "${GREEN}✓${NC} Set PYTORCH_MPS_MAX_COPY_SIZE=1GB"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓${NC} Activated virtual environment"
else
    echo -e "${YELLOW}⚠️  Virtual environment not found${NC}"
    echo "Run these commands to set up the environment:"
    echo "python3 -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu"
    echo "pip install -r requirements_mps.txt"
    echo "pip install ninja flash-attention --no-build-isolation"
fi

# Check if model weights exist
if [ -d "ckpts" ]; then
    echo -e "${GREEN}✓${NC} Model weights directory found"
else
    echo -e "${YELLOW}⚠️  Model weights not found${NC}"
    echo "Run: python download_weights.py"
fi

# Print current configuration
echo -e "\n${GREEN}Current Environment Configuration:${NC}"
echo "PYTORCH_MPS_HIGH_WATERMARK_RATIO=$PYTORCH_MPS_HIGH_WATERMARK_RATIO"
echo "PYTORCH_MPS_MAX_COPY_SIZE=$PYTORCH_MPS_MAX_COPY_SIZE"

echo -e "\n${GREEN}Environment setup complete!${NC}"
echo "Run 'python check_system.py' to verify your system configuration"
echo "Run 'python sample_video_mps.py --help' to see available options for video generation"
