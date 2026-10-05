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
#
#   輸出模式：
#      預設            → 6 輸出 (box / cls 分開)
#      加 --no_split   → 3 輸出 (原本的 cat(box, cls))
#      注意：calib / test / deploy 三個步驟要用同一種模式
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
    '--no_split',
    action='store_true',
    help='不加：Head 輸出 6 張圖 (box/cls 分開，預設)；加上：維持原本 3 張 cat(box, cls)')
parser.add_argument(
    '--params',
    default=None,
    metavar='JSON_PATH',
    help='Path to hyp_params JSON file, passed to test() as params')

args, _ = parser.parse_known_args()
INPUT_SIZE = args.input_size

# ── 輸出模式開關 ─────────────────────────────────
#   預設（沒加 --no_split）: SPLIT_OUTPUT = True  → 6 輸出 (box0, cls0, box1, cls1, box2, cls2)
#   加上 --no_split       : SPLIT_OUTPUT = False → 3 輸出 cat(box, cls)
SPLIT_OUTPUT = not args.no_split

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
    from utils.util import norm
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
# Split output helpers
# ─────────────────────────────────────────────
def merge_split_outputs(outs):
    """(box0, cls0, box1, cls1, box2, cls2) → 3 張 cat(box, cls)；本來就是 3 張則原樣回傳"""
    outs = list(outs)
    if len(outs) == 6:
        outs = [torch.cat((outs[i], outs[i + 1]), dim=1) for i in range(0, 6, 2)]
    return outs


class MergedOutputModel(torch.nn.Module):
    """
    評估 mAP 用的包裝：quant_model 輸出 6 張圖時，先合併回 3 張再交給 test() 的後處理。
    只用於評估，不影響 quantizer 匯出的 xmodel（xmodel 由 quantizer 內部的計算圖決定）。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return merge_split_outputs(self.model(x))


def set_split_output(model, flag):
    if not hasattr(model, 'head'):
        print(f"{YELLOW}[WARNING] model 沒有 head 屬性，無法設定 split_output{RESET}")
        return model
    model.head.split_output = flag
    with torch.no_grad():
        n_out = len(model.eval()(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)))
    print(f"[INFO] split_output = {flag}  → model 輸出 {n_out} 張圖")
    expected = 6 if flag else 3
    if n_out != expected:
        raise RuntimeError(
            f"預期 {expected} 個輸出，實際 {n_out} 個。"
            "請確認載入的 arch/yolo11.py 是含 split_output 的新版本。")
    return model


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
            # ✅ 修正：dataloader 輸出 uint8 [0,255]，直接交給 norm()（走 uint8 分支）
            #    原本先 .float() 再 norm() 會走 float 分支（預期 [0,1]），之後又 /255，
            #    校正時的輸入分布和訓練 / 推論不同，量化 scale 會算錯
            samples = norm(samples.to(device))
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
    # ✅ 必須在建立 torch_quantizer 之前設定，quantizer 會 trace 當下的計算圖
    model = set_split_output(model, SPLIT_OUTPUT)
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

    elif deploy:
        # ✅ deploy 不跑完整 mAP 評估：匯出前只需要一次 forward（batch_size=1）
        #    原本流程先跑 test()，test() 出錯時 export_xmodel() 根本不會執行，
        #    quantize_result/ 裡留下的會是「上一次」的舊 xmodel
        quant_model.eval().to(device)
        with torch.no_grad():
            samples, _ = next(iter(val_loader))
            outs = quant_model(norm(samples.to(device)))
        print(f"[INFO] Deploy forward 完成，quant_model 輸出 {len(outs)} 張圖：")
        for o in outs:
            print(f"         {tuple(o.shape)}")

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
        # ✅ 6 輸出先合併成 3 張，再交給 test() 的 YOLOPostProcessor
        eval_model = MergedOutputModel(quant_model).to(device)
        results = test(test_args, dataset_config, model=eval_model, params=hyp_params)
        mAP, mAP50, recall, precision = results
        print(f"\n[RESULT] mAP:       {mAP:.4f}")
        print(f"[RESULT] mAP@50:    {mAP50:.4f}")
        print(f"[RESULT] Recall:    {recall:.4f}")
        print(f"[RESULT] Precision: {precision:.4f}")

    # ── Deployment export ────────────────────────────────────
    if quant_mode == 'test' and deploy:
        out_dir = 'quantize_result'
        # ✅ 先刪除舊的 xmodel，避免匯出失敗時誤用上一次的檔案
        for f in os.listdir(out_dir) if os.path.isdir(out_dir) else []:
            if f.endswith('.xmodel'):
                os.remove(os.path.join(out_dir, f))
                print(f"[INFO] 刪除舊檔: {os.path.join(out_dir, f)}")

        print("[INFO] Exporting deployment artifacts...")
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel()
        print("[INFO] Export complete: torchscript / onnx / xmodel")

        # ✅ 檢查匯出的 xmodel 輸出數量
        try:
            import xir
            for f in os.listdir(out_dir):
                if f.endswith('_int.xmodel'):
                    g = xir.Graph.deserialize(os.path.join(out_dir, f))
                    outs = [op for op in g.get_ops() if op.get_type() == 'return']
                    ins = outs[0].get_input_ops() if outs else {}
                    n = sum(len(v) for v in ins.values()) if isinstance(ins, dict) else len(ins)
                    print(f"[CHECK] {f}: return 節點有 {n} 個輸入")
        except Exception as e:
            print(f"{YELLOW}[WARNING] 無法檢查 xmodel 輸出：{e}{RESET}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print(f"  Model:      {args.model_path}")
    print(f"  Mode:       {args.quant_mode}")
    print(f"  Input size: (3, {INPUT_SIZE}, {INPUT_SIZE})")
    print(f"  Classes:    {args.num_classes}")
    print(f"  Outputs:    {'6 (box / cls split)' if SPLIT_OUTPUT else '3 (box + cls concat)'}")
    print(f"  Device:     {device}")
    print("=" * 60)
    quantization()
    print("=" * 60)
    print("  Done.")
    print("=" * 60)