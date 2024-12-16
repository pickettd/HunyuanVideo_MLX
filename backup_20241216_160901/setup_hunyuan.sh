#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up HunyuanVideo environment...${NC}"

# Create conda environment
echo -e "\n${GREEN}Creating conda environment...${NC}"
conda create -n HunyuanVideo python=3.10 -y
conda activate HunyuanVideo

# Install PyTorch with MPS support
echo -e "\n${GREEN}Installing PyTorch with MPS support...${NC}"
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install dependencies
echo -e "\n${GREEN}Installing dependencies...${NC}"
pip install -r requirements.txt
pip install ninja flash-attention --no-build-isolation

# Create directory structure
echo -e "\n${GREEN}Creating directory structure...${NC}"
mkdir -p ckpts/hunyuan-video-t2v-720p/transformers
mkdir -p ckpts/vae
mkdir -p ckpts/text_encoder

# Set environment variables
echo -e "\n${GREEN}Setting environment variables...${NC}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7

# Download model weights
echo -e "\n${GREEN}Downloading model weights...${NC}"
python download_weights.py

# Verify setup
echo -e "\n${GREEN}Verifying setup...${NC}"
python check_system.py

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "Next steps:"
echo -e "1. Run: ${YELLOW}conda activate HunyuanVideo${NC}"
echo -e "2. Generate video: ${YELLOW}python sample_video_mps.py \\"
echo "    --video-size 544 960 \\"
echo "    --video-length 129 \\"
echo "    --infer-steps 30 \\"
echo "    --prompt \"An Australian Shepherd dog catching a frisbee in mid-air, slow motion, cinematic style\" \\"
echo "    --flow-reverse \\"
echo "    --save-path ./results${NC}"

# Create results directory
mkdir -p results
