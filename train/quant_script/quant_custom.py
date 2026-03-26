# ============================================================
# Reference:
#   https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/
#   vai_q_pytorch/example/resnet18_quant.py
#
# Usage:
#   Dataloader  → create_dataloader() from utils/dataset.py
#   Evaluation  → test()             from yolo_train.py
#
#   Example commands:
#      # Step 1 — calibration
#      python quant_script/quant_custom.py \
#        --model_path weights/best.pt \
#        --data_dir   /path/to/dataset \
#        --quant_mode calib \
#        --input_size 640 \
#        --num_classes 10 \
#        --extra_path ./arch ./utils
#
#      # Step 2 — evaluate mAP
#      python quant_script/quant_custom.py \
#        --model_path weights/best.pt \
#        --data_dir   /path/to/dataset \
#        --quant_mode test \
#        --input_size 640 \
#        --num_classes 10 \
#        --extra_path ./arch ./utils
#
#      # Step 3 — export for deployment
#      python quant_script/quant_custom.py \
#        --model_path weights/best.pt \
#        --data_dir   /path/to/dataset \
#        --quant_mode test --deploy \
#        --input_size 640 \
#        --num_classes 10 \
#        --extra_path ./arch ./utils \
#        --target DPUCZDX8G_ISA1_B4096
# ============================================================

import os
import sys
import importlib
import argparse
import torch
from tqdm import tqdm


YELLOW = "\033[33m"
RESET  = "\033[0m"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="YOLO PTQ quantization with Vitis-AI pytorch_nndct"
)
parser.add_argument(
    '--model_path',
    required=True,
    help='Path to your .pt checkpoint file')
parser.add_argument(
    '--model_class',
    default=None,
    help=(
        'Dotted import path to the model factory (needed only for plain state_dict). '
        'Example: "arch.yolo.yolo_v11_n"'
    ))
parser.add_argument(
    '--extra_path',
    nargs='+',
    default=[],
    metavar='PATH',
    help=(
        'Directories to prepend to sys.path so imports resolve correctly. '
        'Example: --extra_path ./arch ./utils'
    ))
parser.add_argument(
    '--data_dir',
    required=True,
    help='Dataset root directory passed to create_dataloader and test')
parser.add_argument(
    '--config_file',
    default=None,
    help='Quantization config file (optional)')
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='Number of calibration images; uses full val set if not set')
parser.add_argument(
    '--batch_size',
    default=16,
    type=int,
    help='Batch size (default: 16)')
parser.add_argument(
    '--quant_mode',
    default='calib',
    choices=['float', 'calib', 'test'],
    help='float=evaluate float model, calib=calibrate, test=evaluate quantized model')
parser.add_argument(
    '--fast_finetune',
    dest='fast_finetune',
    action='store_true',
    help='Enable fast finetune before calibration')
parser.add_argument(
    '--deploy',
    dest='deploy',
    action='store_true',
    help='Export xmodel / onnx / torchscript for deployment')
parser.add_argument(
    '--inspect',
    dest='inspect',
    action='store_true',
    help='Inspect model for hardware compatibility')
parser.add_argument(
    '--target',
    dest='target',
    nargs='?',
    const='',
    help='Target hardware device name (used with --inspect or --deploy)')
parser.add_argument(
    '--input_size',
    default=640,
    type=int,
    help='Input image size (default: 640), used as (3, input_size, input_size)')
parser.add_argument(
    '--num_classes',
    default=80,
    type=int,
    help='Number of detection classes (default: 80)')
parser.add_argument(
    '--workers',
    default=8,
    type=int,
    help='DataLoader num_workers (default: 4)')
parser.add_argument(
    '--params',
    default=None,
    metavar='JSON_PATH',
    help='Path to hyp_params JSON file, passed to test() as params')

args, _ = parser.parse_known_args()
INPUT_SIZE = args.input_size

# ─────────────────────────────────────────────
# Load hyp_params from JSON if provided
# ─────────────────────────────────────────────
import json
if args.params:
    with open(args.params, 'r') as f:
        hyp_params = json.load(f)
    print(f"[INFO] Loaded params from: {args.params}")
else:
    hyp_params = None

# ─────────────────────────────────────────────
# Inject extra sys.path entries FIRST so all
# subsequent imports (create_dataloader, test,
# model definitions) can be resolved.
# ─────────────────────────────────────────────
for p in args.extra_path:
    abs_p = os.path.abspath(p)
    if abs_p not in sys.path:
        sys.path.insert(0, abs_p)
        print(f"[INFO] Added to sys.path: {abs_p}")

# ─────────────────────────────────────────────
# Lazy imports from training codebase
#   create_dataloader : utils/dataset.py
#   test              : yolo_train.py
# These are imported after sys.path is set up.
# ─────────────────────────────────────────────
try:
    from utils.dataset import create_dataloader
except ImportError as e:
    raise ImportError(
        "Cannot import create_dataloader from utils.dataset. "
        "Make sure utils/ is reachable via --extra_path."
    ) from e

try:
    from yolo_train import test
except ImportError as e:
    raise ImportError(
        "Cannot import test from yolo_train. "
        "Make sure yolo_train.py is reachable via --extra_path."
    ) from e


# ─────────────────────────────────────────────
# Dynamic model factory
# ─────────────────────────────────────────────
def get_custom_model(model_class: str, num_classes=None):
    """
    Dynamically instantiate a model from a dotted path.
    Example: 'arch.yolo.yolo_v11_n'
    """
    parts = model_class.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(
            f"--model_class must be a dotted path, e.g. 'arch.yolo.yolo_v11_n'. "
            f"Got: '{model_class}'"
        )
    module_path, attr_name = parts
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Cannot import '{module_path}'. "
            "Make sure the directory is in --extra_path."
        ) from e
    if not hasattr(module, attr_name):
        raise AttributeError(f"'{module_path}' has no attribute '{attr_name}'.")
    factory = getattr(module, attr_name)
    try:
        model = factory(num_classes) if num_classes is not None else factory()
    except TypeError:
        model = factory()
    print(f"[INFO] Model architecture: {model_class}")
    return model


# ─────────────────────────────────────────────
# Model loading
#   Priority:
#     1. Full nn.Module                      → weights + arch read directly
#     2. dict['model'] is nn.Module          → your training format
#                                              {"epoch": N, "model": <nn.Module>}
#     3. dict state_dict                     → requires --model_class
# ─────────────────────────────────────────────
def load_model(model_path, model_class=None, num_classes=None):
    print(f"[INFO] Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')

    # Case 1: raw nn.Module
    if isinstance(ckpt, torch.nn.Module):
        print("[INFO] Loaded as full nn.Module.")
        return ckpt.float().cpu()

    # Case 2: training format {"epoch": N, "model": <nn.Module>}
    if isinstance(ckpt, dict) and 'model' in ckpt \
            and isinstance(ckpt['model'], torch.nn.Module):
        print(f"[INFO] Loaded training checkpoint (epoch {ckpt.get('epoch', '?')}).")
        return ckpt['model'].float().cpu()

    # Case 3: plain state_dict
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        else:
            raise ValueError(
                "Unrecognized checkpoint format. "
                "Expected nn.Module, {'model': nn.Module, ...}, or a state_dict."
            )
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        if model_class is None:
            raise ValueError(
                "\n[ERROR] Checkpoint is a plain state_dict — architecture unknown.\n"
                "        Provide --model_class, e.g.:\n"
                "          --model_class arch.yolo.yolo_v11_n"
            )
        model = get_custom_model(model_class, num_classes=num_classes)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"{YELLOW}[WARNING] Missing keys:             {missing}{RESET}")
        if unexpected:
            print(f"{YELLOW}[WARNING] Unexpected keys (ignored): {unexpected}{RESET}")
        print("[INFO] Loaded as state_dict.")
        return model.float().cpu()

    raise ValueError(
        f"Unrecognized checkpoint type: {type(ckpt)}. "
        "Expected nn.Module or dict."
    )


# ─────────────────────────────────────────────
# Calibration forward loop
# Uses create_dataloader — same pipeline as training,
# no labels needed during calibration.
# ─────────────────────────────────────────────
def forward_loop(model, val_loader):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for samples, _ in tqdm(val_loader, desc="Calibrating"):
            samples = samples.to(device).float() / 255.0
            model(samples)


# ─────────────────────────────────────────────
# Main quantization pipeline
# ─────────────────────────────────────────────
def quantization():
    from pytorch_nndct.apis import torch_quantizer

    quant_mode  = args.quant_mode
    finetune    = args.fast_finetune
    deploy      = args.deploy
    batch_size  = args.batch_size
    subset_len  = args.subset_len
    inspect     = args.inspect
    config_file = args.config_file
    target      = args.target
    data_dir    = args.data_dir
    num_classes = args.num_classes
    workers     = args.workers

    # deploy constraints
    if quant_mode != 'test' and deploy:
        deploy = False
        print(f"{YELLOW}[WARNING] Exporting xmodel only works in test mode. deploy disabled.{RESET}")
    if deploy and (batch_size != 1 or subset_len != 1):
        print(f"{YELLOW}[WARNING] Exporting xmodel requires batch_size=1, subset_len=1. Adjusting.{RESET}")
        args.batch_size = 1
        args.subset_len = 1
        batch_size = 1
        subset_len = 1

    # ── Load model ───────────────────────────────────────────
    model = load_model(
        args.model_path,
        model_class=args.model_class,
        num_classes=num_classes,
    )
    dummy_input = torch.randn([batch_size, 3, INPUT_SIZE, INPUT_SIZE])

    # ── Quantizer setup ──────────────────────────────────────
    if quant_mode == 'float':
        quant_model = model
        if inspect:
            if not target:
                raise RuntimeError("--target is required for --inspect.")
            from pytorch_nndct.apis import Inspector
            Inspector(target).inspect(quant_model, (dummy_input,), device=device)
            sys.exit()
    else:
        print(f"[INFO] Creating quantizer, mode: {quant_mode}")
        quantizer = torch_quantizer(
            quant_mode, model, (dummy_input,),
            device=device,
            quant_config_file=config_file,
            target=target,
        )
        quant_model = quantizer.quant_model

    # ── DataLoader via create_dataloader ─────────────────────
    val_img_dir = os.path.join(data_dir, 'images', 'val')
    val_loader = create_dataloader(
        img_folder=val_img_dir,
        input_size=INPUT_SIZE,
        batch_size=batch_size,
        workers=workers,
        augment=False,
        shuffle=False,
    )

    # ── Fast finetune ────────────────────────────────────────
    if finetune:
        ft_loader = create_dataloader(
            img_folder=val_img_dir,
            input_size=INPUT_SIZE,
            batch_size=batch_size,
            workers=workers,
            augment=False,
            shuffle=True,
        )
        if quant_mode == 'calib':
            quantizer.fast_finetune(forward_loop, (quant_model, ft_loader))
        elif quant_mode == 'test':
            quantizer.load_ft_param()

    # ── Calibration or evaluation ────────────────────────────
    if quant_mode == 'calib':
        forward_loop(quant_model, val_loader)
        quantizer.export_quant_config()
        print("[INFO] Calibration complete. Quantization config exported.")

    else:
        # Reuse test() from yolo_train.py directly
        # test() returns (mAP, mAP@50, Recall, Precision)
        from argparse import Namespace
        test_args = Namespace(
            input_size=INPUT_SIZE,
            batch_size=batch_size,
            workers=workers,
        )
        dataset_config = {
            "dataset_dir": data_dir,
            "val_dir":     os.path.join("images", "val"),
            "num_classes": num_classes,
        }
        quant_model.to(device)
        results = test(test_args, dataset_config, model=quant_model, params=hyp_params)
        mAP, mAP50, recall, precision = results
        print(f"\n[RESULT] mAP:       {mAP:.4f}")
        print(f"[RESULT] mAP@50:    {mAP50:.4f}")
        print(f"[RESULT] Recall:    {recall:.4f}")
        print(f"[RESULT] Precision: {precision:.4f}")

    # ── Deployment export ────────────────────────────────────
    if quant_mode == 'test' and deploy:
        print("[INFO] Exporting deployment artifacts...")
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel()
        print("[INFO] Export complete: torchscript / onnx / xmodel")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print(f"  Model:      {args.model_path}")
    print(f"  Mode:       {args.quant_mode}")
    print(f"  Input size: (3, {INPUT_SIZE}, {INPUT_SIZE})")
    print(f"  Classes:    {args.num_classes}")
    print(f"  Device:     {device}")
    print("=" * 60)
    quantization()
    print("=" * 60)
    print("  Done.")
    print("=" * 60)