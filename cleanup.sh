#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to confirm action
confirm() {
    read -p "$(echo -e "${YELLOW}$1 [y/N]${NC}") " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            true
            ;;
        *)
            false
            ;;
    esac
}

echo -e "${GREEN}HunyuanVideo MLX Cleanup Utility${NC}"
echo "This script helps clean up temporary files and cached data."

# Clean Metal shader cache
if confirm "Clean Metal shader cache? This may cause slower initial startup next time:"; then
    echo -e "\n${GREEN}Cleaning Metal shader cache...${NC}"
    rm -rf ~/Library/Caches/com.apple.metal/*/
    echo "Metal shader cache cleaned"
fi

# Clean model weights
if confirm "Remove downloaded model weights? You'll need to download them again:"; then
    echo -e "\n${GREEN}Removing model weights...${NC}"
    rm -rf ckpts/*
    echo "Model weights removed"
fi

# Clean results directory
if confirm "Clean results directory? This will remove all generated videos:"; then
    echo -e "\n${GREEN}Cleaning results directory...${NC}"
    rm -rf results/*
    echo "Results directory cleaned"
fi

# Clean Python cache
if confirm "Clean Python cache files?"; then
    echo -e "\n${GREEN}Cleaning Python cache...${NC}"
    find . -type d -name "__pycache__" -exec rm -r {} +
    find . -type f -name "*.pyc" -delete
    echo "Python cache cleaned"
fi

# Clean temporary files
if confirm "Clean temporary files?"; then
    echo -e "\n${GREEN}Cleaning temporary files...${NC}"
    rm -f *.tmp
    rm -f *.log
    echo "Temporary files cleaned"
fi

echo -e "\n${GREEN}Cleanup complete!${NC}"

# Recreate necessary directories
echo -e "\n${GREEN}Recreating necessary directories...${NC}"
mkdir -p ckpts/hunyuan-video-t2v-720p/{transformers,vae}
mkdir -p ckpts/text_encoder
mkdir -p ckpts/text_encoder_2
mkdir -p results

echo -e "\nNext steps:"
echo -e "1. Run ${YELLOW}python check_system.py${NC} to verify system status"
echo -e "2. Run ${YELLOW}python download_weights.py${NC} if you removed model weights"
