#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting cleanup process...${NC}"

# Function to format size in human-readable format
format_size() {
    local size=$1
    local units=("B" "KB" "MB" "GB" "TB")
    local unit=0
    
    while ((size > 1024)); do
        size=$(($size / 1024))
        unit=$((unit + 1))
    done
    
    echo "$size${units[$unit]}"
}

# Function to calculate directory size
get_dir_size() {
    local dir=$1
    if [ -d "$dir" ]; then
        local size=$(du -sb "$dir" | cut -f1)
        format_size $size
    else
        echo "0B"
    fi
}

# Print initial disk usage
echo -e "\n${YELLOW}Current disk usage:${NC}"
df -h .

# Clean Python cache files
echo -e "\n${GREEN}Cleaning Python cache files...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

# Clean temporary files
echo -e "\n${GREEN}Cleaning temporary files...${NC}"
find . -type f -name "*.tmp" -delete
find . -type f -name "*.temp" -delete
find . -type f -name ".DS_Store" -delete

# Clean results directory (optional)
if [ -d "results" ]; then
    size=$(get_dir_size "results")
    echo -e "\n${YELLOW}Results directory size: $size${NC}"
    read -p "Do you want to clean the results directory? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf results/*
        echo -e "${GREEN}Results directory cleaned${NC}"
    fi
fi

# Report model weights status
echo -e "\n${GREEN}Checking model weights...${NC}"
if [ -d "ckpts" ]; then
    size=$(get_dir_size "ckpts")
    echo -e "Model weights size: $size"
    
    # Check for incomplete downloads
    find ckpts -type f -name "*.tmp" -o -name "*.part" | while read -r file; do
        echo -e "${YELLOW}Warning: Incomplete download found: $file${NC}"
    done
else
    echo -e "${YELLOW}No model weights found${NC}"
fi

# Clean virtual environment (optional)
if [ -d "venv" ]; then
    size=$(get_dir_size "venv")
    echo -e "\n${YELLOW}Virtual environment size: $size${NC}"
    read -p "Do you want to clean and rebuild the virtual environment? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        source venv/bin/activate
        pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
        pip install -r requirements_mps.txt
        pip install ninja flash-attention --no-build-isolation
        echo -e "${GREEN}Virtual environment rebuilt${NC}"
    fi
fi

# Print final disk usage
echo -e "\n${YELLOW}Final disk usage:${NC}"
df -h .

echo -e "\n${GREEN}Cleanup complete!${NC}"
echo "Run './setup_env.sh' to ensure environment variables are set correctly"
