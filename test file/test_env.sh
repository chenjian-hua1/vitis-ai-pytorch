#!/usr/bin/env bash
set -e
set -o pipefail

# ==============================
# Fixed parameters
# ==============================
DATA_DIR="dataset/tiny-imagenet-200/"
PRETRAINED_MODEL="model/resnet18.pth"
MODEL_DIR="model"
XMODEL_PATH="quantize_result/ResNet_int.xmodel"
BOARD="/compiler/arch/DPUCZDX8G/KV260"
OUTPUT_PATH="kv26_dpu"
NETNAME="kv26_resnet18"

# ==============================
# Logging helpers
# ==============================
stage() {
    echo ""
    echo "===================================================="
    echo "[STAGE] $1"
    echo "===================================================="
}

info() {
    echo "[INFO] $1"
}

# ==============================
# Help message
# ==============================
usage() {
    echo "Usage:"
    echo "  $0 --env-name <vai_pytorch_env> --dataset-dir <dataset_path>"
    echo ""
    echo "Options:"
    echo "  --env-name        Conda environment name for Vitis AI"
    echo "  --dataset-dir     Dataset directory path"
    echo "  -h, --help        Show this help message"
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
        --dataset-dir)
            DATA_DIR="$2"
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
    echo "Error: Missing required argument --env-name"
    usage
fi

DATA_DIR=$(realpath "$DATA_DIR")

# ==============================
# Load conda
# ==============================
stage "Loading Conda environment support"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found. Please check your Conda installation."
    exit 1
fi

info "Conda shell support loaded."

# ==============================
# Activate conda environment
# ==============================
stage "Activating Conda environment"

conda activate "$ENV_NAME"
info "Activated environment: $ENV_NAME"

# ==============================
# Show configuration
# ==============================
stage "Configuration Summary"

echo "ENV_NAME         : $ENV_NAME"
echo "DATA_DIR         : $DATA_DIR"
echo "PRETRAINED_MODEL : $PRETRAINED_MODEL"
echo "MODEL_DIR        : $MODEL_DIR"
echo "XMODEL_PATH      : $XMODEL_PATH"
echo "BOARD            : $BOARD"
echo "OUTPUT_PATH      : $OUTPUT_PATH"
echo "NETNAME          : $NETNAME"

# ==============================
# Check required files/directories
# ==============================
stage "Checking required files and directories"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: DATA_DIR not found: $DATA_DIR"
    exit 1
fi
info "Found DATA_DIR: $DATA_DIR"

if [[ ! -f "$PRETRAINED_MODEL" ]]; then
    echo "Error: PRETRAINED_MODEL not found: $PRETRAINED_MODEL"
    exit 1
fi
info "Found PRETRAINED_MODEL: $PRETRAINED_MODEL"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: MODEL_DIR not found: $MODEL_DIR"
    exit 1
fi
info "Found MODEL_DIR: $MODEL_DIR"

if [[ ! -d "$BOARD" ]]; then
    echo "Error: BOARD directory not found: $BOARD"
    exit 1
fi
info "Found BOARD directory: $BOARD"

if [[ ! -f "$BOARD/arch.json" ]]; then
    echo "Error: arch.json not found: $BOARD/arch.json"
    exit 1
fi
info "Found arch.json: $BOARD/arch.json"

# ==============================
# Step 1: Test pruning
# ==============================
stage "Step 1: Testing pruning"

info "Running pruning script..."
python resnet18_pruning.py \
    --pretrained "$PRETRAINED_MODEL" \
    --data_dir "$DATA_DIR" \
    --method iterative

info "Pruning step completed."

# ==============================
# Step 2: Inspect DPU compatibility
# ==============================
stage "Step 2: Inspecting DPU compatibility"

info "Checking model compatibility with DPU..."
python resnet18_quant.py \
    --quant_mode float \
    --inspect \
    --model_dir "$MODEL_DIR" \
    --target DPUCZDX8G_ISA1_B4096

info "DPU compatibility inspection completed."

# ==============================
# Step 3: Calibration
# ==============================
stage "Step 3: Quantization calibration"

info "Running calibration..."
python resnet18_quant.py \
    --quant_mode calib \
    --inspect \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_DIR"

info "Calibration finished."

# ==============================
# Step 4: Evaluate quantized model
# ==============================
stage "Step 4: Testing quantized model"

info "Evaluating quantized model..."
python resnet18_quant.py \
    --quant_mode test \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_DIR"

info "Quantized model evaluation completed."

# ==============================
# Step 5: Generate deploy xmodel
# ==============================
stage "Step 5: Exporting xmodel"

info "Generating deployable xmodel..."
python resnet18_quant.py \
    --quant_mode test \
    --model_dir "$MODEL_DIR" \
    --data_dir "$DATA_DIR" \
    --subset_len 1 \
    --batch_size 1 \
    --deploy

info "xmodel export completed."

# ==============================
# Step 6: Compile xmodel for DPU
# ==============================
stage "Step 6: Compiling xmodel for DPU"

info "Creating output directory..."
mkdir -p "$OUTPUT_PATH"

info "Compiling xmodel using vai_c_xir..."
vai_c_xir \
    -x "$XMODEL_PATH" \
    -a "$BOARD/arch.json" \
    -o "$OUTPUT_PATH" \
    -n "$NETNAME"

info "Compilation finished."

# ==============================
# Finish
# ==============================
stage "All steps completed successfully"

echo "Compiled model output directory: $OUTPUT_PATH"
