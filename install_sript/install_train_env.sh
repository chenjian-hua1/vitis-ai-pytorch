#!/bin/bash

# =========================
# Argument parsing
# =========================
while getopts "n:" opt; do
  case $opt in
    n)
      ENV_NAME="$OPTARG"
      ;;
    *)
      echo "Usage: $0 -n <env_name>"
      exit 1
      ;;
  esac
done

# Check if env name is provided
if [ -z "$ENV_NAME" ]; then
  echo "Error: Environment name is required"
  echo "Usage: $0 -n <env_name>"
  exit 1
fi

# =========================
# Get absolute path of script directory
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_PATH="${SCRIPT_DIR}/train_env.yaml"

# =========================
# Check if conda exists
# =========================
if ! command -v conda &> /dev/null; then
  echo "Error: conda is not installed or not in PATH"
  exit 1
fi

# =========================
# Initialize conda
# =========================
source "$(conda info --base)/etc/profile.d/conda.sh"

# =========================
# Check YAML file
# =========================
if [ ! -f "$YAML_PATH" ]; then
  echo "Error: Cannot find $YAML_PATH"
  exit 1
fi

echo "Using YAML file: $YAML_PATH"
echo "Creating environment: $ENV_NAME"

# =========================
# Remove existing environment
# =========================
if conda env list | grep -q "^$ENV_NAME "; then
  echo "Warning: Environment already exists. Removing..."
  conda env remove -n "$ENV_NAME" -y
fi

# =========================
# Create environment
# =========================
conda env create -n "$ENV_NAME" -f "$YAML_PATH"

# =========================
# Activate environment
# =========================
echo "Activating environment..."
conda activate "$ENV_NAME"

echo "Install TrackEval , OpenCV , numpy"
pip install trackeval==0.1.5 -y
pip install opencv-python==4.11.0.86 -y
pip install opencv-python-headless==4.11.0.86 -y
pip install numpy==1.26 -y

echo "Done! Environment '$ENV_NAME' is ready."
