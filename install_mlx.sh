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

echo -e "${GREEN}Setting up HunyuanVideo MLX environment...${NC}"

# Create conda environment
echo -e "\n${GREEN}Creating conda environment...${NC}"
conda create -n hunyuan-mlx python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate hunyuan-mlx

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

# Download models
echo -e "\n${GREEN}Downloading model weights...${NC}"
python download_weights.py

# Set environment variables
echo -e "\n${GREEN}Setting up environment variables...${NC}"
CONDA_ENV_PATH=$(conda env list | grep hunyuan-mlx | awk '{print $2}')
CONDA_ACTIVATE_SCRIPT="$CONDA_ENV_PATH/etc/conda/activate.d/env_vars.sh"
mkdir -p "$(dirname "$CONDA_ACTIVATE_SCRIPT")"
echo '#!/bin/bash' > "$CONDA_ACTIVATE_SCRIPT"
echo 'export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7' >> "$CONDA_ACTIVATE_SCRIPT"
chmod +x "$CONDA_ACTIVATE_SCRIPT"

# Verify installation
echo -e "\n${GREEN}Verifying installation...${NC}"
python check_system.py

echo -e "\n${GREEN}Installation complete!${NC}"
echo -e "\nNext steps:"
echo -e "1. Run: ${YELLOW}conda activate hunyuan-mlx${NC}"
echo -e "2. Generate a video: ${YELLOW}python sample_video_mps.py \\"
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
