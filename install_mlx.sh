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

# Check Python version (3.10 or higher required)
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.10" | bc -l) )); then
    error_exit "Python 3.10 or higher is required. Current version: $python_version"
fi

echo "System checks passed:"
echo "- macOS detected: $(sw_vers -productVersion)"
echo "- Apple Silicon detected: $(uname -m)"
echo "- Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv || error_exit "Failed to create virtual environment"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || error_exit "Failed to activate virtual environment"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip || error_exit "Failed to upgrade pip"

# Install MLX
echo "Installing MLX..."
pip install "mlx>=0.0.10" || error_exit "Failed to install MLX"

# Install other requirements
echo "Installing project requirements..."
pip install -r requirements.txt || error_exit "Failed to install requirements"

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
