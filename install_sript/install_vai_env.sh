#!/usr/bin/env bash
set -e
set -o pipefail

# ==============================
# Fixed configuration
# ==============================
# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VAI_ROOT=""
YML_FILE="$SCRIPT_DIR/vitis-ai-pytorch.yml"

# Vitis AI conda channel download URL
VAI_CONDA_CHANNEL_URL="https://www.xilinx.com/bin/public/openDownload?filename=conda-channel-3.5.0.tar.gz"

# ==============================
# Help message
# ==============================
usage() {
    echo "Usage:"
    echo "  $0 --env-name <env_name>"
    echo ""
    echo "Options:"
    echo "  --env-name       Conda environment name (e.g. vitis-ai-pytorch)"
    echo "  -h, --help       Show this help message"
    exit 1
}

# ==============================
# Parse arguments
# ==============================
ENV_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# ==============================
# Validate arguments
# ==============================
if [[ -z "$ENV_NAME" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# ==============================
# Load conda
# ==============================
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "====================================="
echo "ENV_NAME : $ENV_NAME"
echo "VAI_ROOT : $VAI_ROOT"
echo "====================================="

# =========================================================
# Download Vitis AI offline conda channel
# =========================================================
sudo mkdir -p /scratch
cd /scratch

# ==============================
# Check if conda channel exists
# ==============================
if [[ -d "/scratch/conda-channel" ]]; then
    echo "====================================="
    echo "Conda channel already exists, skipping download."
    echo "====================================="
else
    echo "====================================="
    echo "Downloading Vitis AI conda channel..."
    echo "====================================="

    sudo wget -O conda-channel.tar.gz --progress=dot:mega "$VAI_CONDA_CHANNEL_URL"

    echo "Extracting conda channel..."
    sudo tar -xzvf conda-channel.tar.gz
fi

export VAI_CONDA_CHANNEL="file:///scratch/conda-channel"

echo "Local conda channel: $VAI_CONDA_CHANNEL"

# ==============================
# Create compiler directory
# ==============================
sudo mkdir -p "$VAI_ROOT/compiler"

# ==============================
# Configure conda channel
# ==============================
conda config --remove-key channels || true
conda config --append channels "$VAI_CONDA_CHANNEL"
conda config --show channels

# ==============================
# Create conda environment
# ==============================
conda env create -v -f "$YML_FILE" -n "$ENV_NAME"

# ==============================
# Activate environment
# ==============================
conda activate "$ENV_NAME"

conda install xnnc -y

# ==============================
# Install PyTorch & YOLO
# ==============================
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# ==============================
# Copy Vitis AI arch
# ==============================
sudo cp -r "$CONDA_PREFIX/lib/python3.8/site-packages/vaic/arch" \
    "$VAI_ROOT/compiler/arch"

# ==============================
# Clean conda cache
# ==============================
conda clean -y --force-pkgs-dirs
conda clean --all -y

# ==============================
# Cleanup
# ==============================
conda deactivate
conda config --append channels defaults
rm -rf ~/.cache

echo "====================================="
echo "Vitis AI environment setup completed"
echo "ENV_NAME: $ENV_NAME"
echo "====================================="
