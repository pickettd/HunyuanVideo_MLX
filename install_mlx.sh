#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${RED}Error: This script is only for Apple Silicon Macs${NC}"
    exit 1
fi

# Check minimum macOS version (12.3+)
if [[ $(sw_vers -productVersion | cut -d. -f1) -lt 12 ]] || [[ $(sw_vers -productVersion | cut -d. -f1) -eq 12 && $(sw_vers -productVersion | cut -d. -f2) -lt 3 ]]; then
    echo -e "${RED}Error: macOS 12.3 or later is required${NC}"
    exit 1
fi

# Check for Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo -e "${RED}Error: Python 3.10 is required but not found${NC}"
    echo -e "${YELLOW}Please install Python 3.10:${NC}"
    echo -e "brew install python@3.10"
    exit 1
fi

echo -e "${GREEN}Setting up HunyuanVideo MLX environment...${NC}"

# Create Python virtual environment
echo -e "\n${GREEN}Creating Python virtual environment...${NC}"
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "\n${GREEN}Installing dependencies...${NC}"
python -m pip install --upgrade pip
python -m pip install -r requirements_mps.txt

# Create directory structure
echo -e "\n${GREEN}Creating directory structure...${NC}"
mkdir -p ckpts/hunyuan-video-t2v-720p/{transformers,vae}
mkdir -p ckpts/text_encoder
mkdir -p ckpts/text_encoder_2
mkdir -p results

# Create .env file template if it doesn't exist
if [ ! -f .env ]; then
    echo -e "\n${GREEN}Creating .env file template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env and add your Hugging Face token${NC}"
fi

# Set environment variables
echo -e "\n${GREEN}Setting up environment variables...${NC}"
VENV_ACTIVATE_SCRIPT="venv/bin/activate"

# Remove any existing PyTorch MPS settings
sed -i '' '/export PYTORCH_MPS_.*_WATERMARK_RATIO/d' "$VENV_ACTIVATE_SCRIPT"

# Add the correct watermark ratios
echo 'export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7' >> "$VENV_ACTIVATE_SCRIPT"
echo 'export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5' >> "$VENV_ACTIVATE_SCRIPT"

# Also set them in the current session
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5

# Verify environment variables
if [ "$PYTORCH_MPS_HIGH_WATERMARK_RATIO" != "0.7" ] || [ "$PYTORCH_MPS_LOW_WATERMARK_RATIO" != "0.5" ]; then
    echo -e "${RED}Error: Failed to set PyTorch MPS watermark ratios${NC}"
    exit 1
fi

# Download models
echo -e "\n${GREEN}Downloading model weights...${NC}"
if [ -f .env ]; then
    python download_weights.py
else
    echo -e "${RED}Error: .env file not found. Please create it with your Hugging Face token${NC}"
    exit 1
fi

# Verify installation
echo -e "\n${GREEN}Verifying installation...${NC}"
python check_system.py

echo -e "\n${GREEN}Installation complete!${NC}"
echo -e "\nNext steps:"
echo -e "1. Run: ${YELLOW}source venv/bin/activate${NC}"
echo -e "2. Verify environment variables:"
echo -e "   ${YELLOW}echo \$PYTORCH_MPS_HIGH_WATERMARK_RATIO  # Should be 0.7${NC}"
echo -e "   ${YELLOW}echo \$PYTORCH_MPS_LOW_WATERMARK_RATIO   # Should be 0.5${NC}"
echo -e "3. Generate a video: ${YELLOW}python sample_video_mps.py \\"
echo "    --video-size 544 960 \\"
echo "    --video-length 129 \\"
echo "    --infer-steps 30 \\"
echo "    --prompt \"a cat is running, realistic.\" \\"
echo "    --flow-reverse \\"
echo "    --save-path ./results${NC}"

# Print system info
echo -e "\n${GREEN}System Information:${NC}"
echo "Python version: $(python --version)"
echo "MLX version: $(pip show mlx | grep Version)"
echo "macOS version: $(sw_vers -productVersion)"
echo "Architecture: $(uname -m)"
echo "MPS High Watermark Ratio: $PYTORCH_MPS_HIGH_WATERMARK_RATIO"
echo "MPS Low Watermark Ratio: $PYTORCH_MPS_LOW_WATERMARK_RATIO"

# Deactivate virtual environment
deactivate

# Final reminder
echo -e "\n${YELLOW}Important: After activating the virtual environment, verify the MPS settings:${NC}"
echo -e "${YELLOW}  PYTORCH_MPS_HIGH_WATERMARK_RATIO should be 0.7${NC}"
echo -e "${YELLOW}  PYTORCH_MPS_LOW_WATERMARK_RATIO should be 0.5${NC}"
