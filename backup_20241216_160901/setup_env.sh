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

# Install dependencies
echo -e "\n${GREEN}Installing dependencies...${NC}"
python -m pip install -r requirements.txt

# Install flash attention
echo -e "\n${GREEN}Installing flash attention...${NC}"
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.9.post1

# Install huggingface-cli
echo -e "\n${GREEN}Installing Hugging Face CLI...${NC}"
python -m pip install "huggingface_hub[cli]"

# Create directory structure
echo -e "\n${GREEN}Creating directory structure...${NC}"
mkdir -p ckpts/hunyuan-video-t2v-720p/{transformers,vae}
mkdir -p ckpts/text_encoder
mkdir -p ckpts/text_encoder_2

# Download models
echo -e "\n${GREEN}Downloading HunyuanVideo model...${NC}"
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts

echo -e "\n${GREEN}Downloading text encoder (llava-llama-3-8b)...${NC}"
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava-llama-3-8b-v1_1-transformers

echo -e "\n${GREEN}Processing text encoder...${NC}"
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
    --input_dir ckpts/llava-llama-3-8b-v1_1-transformers \
    --output_dir ckpts/text_encoder

echo -e "\n${GREEN}Downloading CLIP model...${NC}"
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2

# Verify setup
echo -e "\n${GREEN}Verifying setup...${NC}"
python check_system.py

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "Next steps:"
echo -e "1. Run: ${YELLOW}conda activate HunyuanVideo${NC}"
echo -e "2. Generate video: ${YELLOW}python sample_video.py \\"
echo "    --video-size 544 960 \\"
echo "    --video-length 129 \\"
echo "    --infer-steps 30 \\"
echo "    --prompt \"a cat is running, realistic.\" \\"
echo "    --flow-reverse \\"
echo "    --save-path ./results${NC}"

# Create results directory
mkdir -p results
