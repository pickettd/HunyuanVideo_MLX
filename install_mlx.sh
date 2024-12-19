#!/bin/bash

# Function to print error and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    error_exit "This script requires macOS. Current OS: $(uname)"
fi

# Check if running on Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    error_exit "This script requires Apple Silicon (M1/M2/M3). Current architecture: $(uname -m)"
fi

# Ensure we're using Python 3.11
if ! command -v python3.11 &> /dev/null; then
    error_exit "Python 3.11 is required but not found. Please install Python 3.11."
fi

# Force use of Python 3.11
PYTHON_CMD="python3.11"
python_version=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$python_version" != "3.11" ]; then
    error_exit "Python 3.11 is required. Current version: $python_version"
fi

echo "System checks passed:"
echo "- macOS detected: $(sw_vers -productVersion)"
echo "- Apple Silicon detected: $(uname -m)"
echo "- Python version: $python_version"

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create new virtual environment with Python 3.11
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv || error_exit "Failed to create virtual environment"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || error_exit "Failed to activate virtual environment"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip || error_exit "Failed to upgrade pip"

# Install MLX and other requirements
echo "Installing MLX and other requirements..."
pip install --upgrade pip || error_exit "Failed to upgrade pip"
pip install "mlx>=0.0.10" || error_exit "Failed to install MLX"
pip install -r requirements.txt || error_exit "Failed to install requirements"

# Set up environment variables
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env || error_exit "Failed to create .env file"
    echo "Created .env file from template. Please edit it to add your Hugging Face token."
fi

# Export environment variables for Metal optimization
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection
export MPS_USE_GUARD_MODE=1
export MPS_ENABLE_MEMORY_GUARD=1
export PYTORCH_MPS_SYNC_OPERATIONS=1
export PYTORCH_MPS_AGGRESSIVE_MEMORY_CLEANUP=1

# Optional: Install development dependencies
if [[ "$1" == "--with-dev" ]]; then
    echo "Installing development dependencies..."
    pip install pytest pytest-cov black isort flake8 || error_exit "Failed to install development dependencies"
fi

echo "✅ Installation complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To verify MLX installation:"
echo "  python3 -c 'import mlx; print(f\"MLX version: {mlx.__version__}\")'"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and add your Hugging Face token"
echo "2. Run python download_weights.py to download model weights"
echo "3. Start generating videos with sample_video_mps.py"
