import os
import re
import sys
import argparse
import time
import random
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# ============================================================
# Reference:
#   https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/example/resnet18_quant.py
#
# Usage:
#   Pass --model_class to specify the model architecture.
#   Three supported formats:
#
#   1) torchvision built-in:
#      --model_class torchvision.models.resnet18
#
#   2) Local module (file must be on PYTHONPATH or same directory):
#      --model_class my_module.MyNet
#
#   3) Full model saved with torch.save(model, ...):
#      (no --model_class needed; detected automatically)
#
#   Example commands:
#      python quantize_custom.py \
#        --model_path ./your_model.pt \
#        --model_class torchvision.models.resnet18 \
#        --data_dir /path/to/imagenet \
#        --quant_mode calib \
#        --input_size 3 224 224
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Dynamic model factory
# ─────────────────────────────────────────────
def get_custom_model(model_class: str, num_classes=None):
    """
    Dynamically instantiate a model from a dotted class/function path.

    model_class examples:
        'torchvision.models.resnet18'          # callable that returns a model
        'torchvision.models.ResNet'            # class (num_classes forwarded if given)
        'my_package.my_module.MyNet'           # any importable path
    """
    parts = model_class.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(
            f"--model_class must be a dotted path, e.g. 'torchvision.models.resnet18'. Got: '{model_class}'"
        )
    module_path, attr_name = parts

    import importlib
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Cannot import module '{module_path}'. "
            "Make sure the file is on your PYTHONPATH or in the current directory."
        )

    if not hasattr(module, attr_name):
        raise AttributeError(f"'{module_path}' has no attribute '{attr_name}'.")

    factory = getattr(module, attr_name)

    # Try to pass num_classes if the user supplied it
    try:
        if num_classes is not None:
            model = factory(num_classes=num_classes)
        else:
            model = factory()
    except TypeError:
        # Factory doesn't accept num_classes — fall back to no-arg call
        model = factory()

    print(f"[INFO] Model architecture: {model_class}")
    return model

# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Custom model PTQ quantization with Vitis-AI pytorch_nndct")

parser.add_argument(
    '--model_path',
    required=True,
    help='Path to your .pt model file (state_dict or full model)')
parser.add_argument(
    '--model_class',
    default=None,
    help=(
        'Dotted import path to the model class or factory function. '
        'Required when the .pt file is a state_dict. '
        'Examples: "torchvision.models.resnet18", "my_module.MyNet"'
    ))
parser.add_argument(
    '--extra_path',
    nargs='+',
    default=[],
    metavar='PATH',
    help=(
        'Extra directories to prepend to sys.path before loading the model. '
        'Use this when model definition files are in a separate folder. '
        'Example: --extra_path ../arch  or  --extra_path . ../arch'
    ))
parser.add_argument(
    '--data_dir',
    default="dataset/imagenet",
    help='ImageNet dataset root directory (must contain train/ and val/ subdirectories)')
parser.add_argument(
    '--config_file',
    default=None,
    help='Path to quantization config file (optional)')
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='Number of validation samples to use; uses the full dataset if not set')
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='Batch size for inference')
parser.add_argument(
    '--quant_mode',
    default='calib',
    choices=['float', 'calib', 'test'],
    help='Run mode: float=evaluate float model, calib=quantization calibration, test=evaluate quantized model')
parser.add_argument(
    '--fast_finetune',
    dest='fast_finetune',
    action='store_true',
    help='Enable fast finetune before calibration to improve accuracy')
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
    nargs="?",
    const="",
    help='Target hardware device name (used with --inspect or --deploy)')
parser.add_argument(
    '--input_size',
    default=224,
    type=int,
    help='Model input image size (default: 224), used as (3, input_size, input_size)')
parser.add_argument(
    '--num_classes',
    default=None,
    type=int,
    help='Number of output classes (required by some models)')

args, _ = parser.parse_known_args()

INPUT_SIZE = args.input_size

# Inject extra paths into sys.path so torch.load can find model definition files
if args.extra_path:
    for p in args.extra_path:
        abs_p = os.path.abspath(p)
        if abs_p not in sys.path:
            sys.path.insert(0, abs_p)
            print(f"[INFO] Added to sys.path: {abs_p}")

# ─────────────────────────────────────────────
# Model loading
#   Priority:
#     1. Full nn.Module            → architecture + weights read directly
#     2. dict with nn.Module model → YOLO-style checkpoint (extracts 'model' key)
#     3. dict with state_dict      → requires --model_class to rebuild architecture
# ─────────────────────────────────────────────
def load_model(model_path, model_class=None, num_classes=None):
    print(f"[INFO] Loading: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    # ── Case 1: full model (torch.save(model, path)) ──────────────────────
    # Check nn.Module FIRST — some model objects (e.g. YOLO) may also be
    # iterable or dict-like, so this must come before the dict check.
    if isinstance(checkpoint, torch.nn.Module):
        print("[INFO] Detected full nn.Module — architecture and weights loaded directly.")
        return checkpoint.cpu()

    # ── Case 2: dict → YOLO-style checkpoint or plain state_dict ──────────
    if isinstance(checkpoint, dict):
        # YOLO-style: {'model': <nn.Module>, 'epoch': ..., ...}
        if 'model' in checkpoint and isinstance(checkpoint['model'], torch.nn.Module):
            model = checkpoint['model']
            print("[INFO] Detected YOLO-style checkpoint — extracting nn.Module from 'model' key.")
            return model.float().cpu()

        # Unwrap common state_dict wrapper formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']  # assume state_dict at this point
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise ValueError(
                "The .pt file is a dict but does not look like a state_dict "
                "(mixed non-Tensor values). Cannot determine format."
            )

        # Strip DataParallel 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        if model_class is None:
            raise ValueError(
                "\n[ERROR] The .pt file contains only a state_dict (weights only).\n"
                "        Architecture cannot be inferred automatically.\n"
                "        Please provide --model_class, e.g.:\n"
                "          --model_class torchvision.models.resnet18\n"
                "          --model_class my_package.my_module.MyNet\n"
                "        If you want architecture + weights in one file, save with:\n"
                "          torch.save(model, 'model.pt')  # instead of model.state_dict()"
            )

        model = get_custom_model(model_class, num_classes=num_classes)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARNING] Missing keys:            {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys (ignored): {unexpected}")
        print("[INFO] Loaded as state_dict.")
        return model.cpu()

    raise ValueError(
        f"Unrecognized .pt file contents (type: {type(checkpoint)}). "
        "Expected nn.Module or a state_dict."
    )

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_data(train=True,
              data_dir='dataset/imagenet',
              batch_size=128,
              subset_len=None,
              sample_method='random',
              distributed=False,
              input_h=224,
              input_w=224,
              **kwargs):

    traindir = os.path.join(data_dir, 'train')
    valdir   = os.path.join(data_dir, 'val')
    train_sampler = None

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    resize = int(input_h * (256 / 224))  # proportional resize

    if train:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop((input_h, input_w)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if subset_len:
            assert subset_len <= len(dataset)
            idx = random.sample(range(len(dataset)), subset_len) \
                  if sample_method == 'random' else list(range(subset_len))
            dataset = torch.utils.data.Subset(dataset, idx)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            shuffle=(train_sampler is None), sampler=train_sampler, **kwargs)
    else:
        dataset = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop((input_h, input_w)),
                transforms.ToTensor(),
                normalize,
            ]))
        if subset_len:
            assert subset_len <= len(dataset)
            idx = random.sample(range(len(dataset)), subset_len) \
                  if sample_method == 'random' else list(range(subset_len))
            dataset = torch.utils.data.Subset(dataset, idx)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return data_loader, train_sampler

# ─────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, val_loader, loss_fn):
    model.eval()
    model = model.to(device)
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total = 0
    total_loss = 0
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        total += images.size(0)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
    return top1.avg, top5.avg, total_loss / len(val_loader)


def forward_loop(model, val_loader):
    model.eval()
    model = model.to(device)
    for images, _ in tqdm(val_loader, desc="Calibrating"):
        images = images.to(device)
        model(images)

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

    # deploy constraints
    if quant_mode != 'test' and deploy:
        deploy = False
        print("[WARNING] Exporting xmodel is only supported in test mode. deploy has been disabled.")
    if deploy and (batch_size != 1 or subset_len != 1):
        print("[WARNING] Exporting xmodel requires batch_size=1 and subset_len=1. Adjusting automatically.")
        batch_size = 1
        subset_len = 1

    # Load model
    model = load_model(args.model_path, model_class=args.model_class, num_classes=args.num_classes)
    dummy_input = torch.randn([batch_size, 3, INPUT_SIZE, INPUT_SIZE])

    if quant_mode == 'float':
        quant_model = model
        if inspect:
            if not target:
                raise RuntimeError("--target must be specified when using --inspect.")
            from pytorch_nndct.apis import Inspector
            inspector = Inspector(target)
            inspector.inspect(quant_model, (dummy_input,), device=device)
            sys.exit()
    else:
        print(f"[INFO] Creating quantizer, mode: {quant_mode}")
        quantizer = torch_quantizer(
            quant_mode, model, (dummy_input,),
            device=device,
            quant_config_file=config_file,
            target=target)
        quant_model = quantizer.quant_model

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    val_loader, _ = load_data(
        train=False,
        data_dir=data_dir,
        batch_size=batch_size,
        subset_len=subset_len,
        sample_method='random',
        input_h=INPUT_SIZE,
        input_w=INPUT_SIZE)

    # Fast finetune
    if finetune:
        ft_loader, _ = load_data(
            train=False,
            data_dir=data_dir,
            batch_size=batch_size,
            subset_len=5120,
            sample_method='random',
            input_h=INPUT_SIZE,
            input_w=INPUT_SIZE)
        if quant_mode == 'calib':
            quantizer.fast_finetune(forward_loop, (quant_model, ft_loader))
        elif quant_mode == 'test':
            quantizer.load_ft_param()

    # Run calibration or evaluation
    if quant_mode == 'calib':
        forward_loop(quant_model, val_loader)
        quantizer.export_quant_config()
        print("[INFO] Calibration complete. Quantization config exported.")
    else:
        acc1, acc5, loss = evaluate(quant_model, val_loader, loss_fn)
        print(f"\n[RESULT] Loss:      {loss:.4f}")
        print(f"[RESULT] Top-1 Acc: {acc1:.2f}%")
        print(f"[RESULT] Top-5 Acc: {acc5:.2f}%")

    # Export for deployment
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
    print(f"  Device:     {device}")
    print("=" * 60)
    quantization()
    print("=" * 60)
    print("  Done.")
    print("=" * 60)