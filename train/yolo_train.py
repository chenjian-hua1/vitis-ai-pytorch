"""
YOLOv11 Custom Training Script
Based on: https://github.com/jahongir7174/YOLOv11-pt

新增對齊 Ultralytics 的優化：
  1. copy_paste 增強（dataset.py 已整合）
  2. 多尺度訓練（每個 batch 隨機縮放輸入大小）
  3. warmup 同時 warm up momentum
  4. top_k 從 10 → 13
"""

import copy
import csv
import os
import glob
import yaml
import json
from types import SimpleNamespace
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import tqdm

from arch.yolo11 import yolo_v11_n, YOLOPostProcessor
from utils import util
from utils.dataset import create_dataloader, Dataset

# ─────────────────────────────────────────────────────────────
# Load Config
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training Configuration")

    parser.add_argument("--data_yaml",   type=str, default="config/data.yaml")
    parser.add_argument("--params_json", type=str, default="config/params.json")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--input_size",  type=int, default=640)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--workers",     type=int, default=8)

    return parser.parse_args()


def print_config(args, params, dataset_config):
    print("\n" + "=" * 50)
    print("  Training Configuration")
    print("=" * 50)

    print("\n[Args]")
    for k, v in vars(args).items():
        print(f"  {k:<15} = {v}")

    print("\n[Dataset]")
    print(f"  {'dataset_dir':<15} = {dataset_config['dataset_dir']}")
    print(f"  {'train_dir':<15} = {dataset_config['train_dir']}")
    print(f"  {'val_dir':<15} = {dataset_config['val_dir']}")
    print(f"  {'num_classes':<15} = {dataset_config['num_classes']}")
    print(f"  {'class_names':<15} = {dataset_config['class_names']}")

    print("\n[Params]")
    for k, v in params.items():
        print(f"  {k:<20} = {v}")

    print("=" * 50 + "\n")


def load_data_conf(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data yaml not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}")

    for field in ("path", "train", "val", "nc", "names"):
        if field not in data:
            raise KeyError(f"Missing required field in data.yaml: '{field}'")

    dataset_dir = data["path"]
    train_dir   = os.path.join(dataset_dir, data["train"])
    val_dir     = os.path.join(dataset_dir, data["val"])

    for label, path in [("dataset_dir", dataset_dir),
                        ("train_dir",   train_dir),
                        ("val_dir",     val_dir)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} does not exist: {path}")

    class_names = data["names"]
    if isinstance(class_names, dict):
        class_names = list(class_names.values())

    return {
        "dataset_dir": dataset_dir,
        "train_dir":   train_dir,
        "val_dir":     val_dir,
        "num_classes": data["nc"],
        "class_names": class_names,
    }


def load_params(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"params json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")


def load_config(print_flag:bool=True):
    args = parse_args()

    try:
        params         = load_params(args.params_json)
        dataset_config = load_data_conf(args.data_yaml)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[ERROR] {e}")
        return None, None, None

    if print_flag: 
        print_config(args, params, dataset_config)

    return args, params, dataset_config

# ─────────────────────────────────────────────────────────────
# Warmup Scheduler（同時 warm up lr 和 momentum）
# ─────────────────────────────────────────────────────────────

class WarmupScheduler:
    """
    Warmup 階段：lr 從 min_lr → max_lr，momentum 從 warmup_momentum → momentum
    Warmup 結束後：接原本的 LinearLR / CosineLR
    """

    def __init__(self, args, params, num_steps):
        self.warmup_steps    = int(max(params['warmup_epochs'] * num_steps, 100))
        self.min_lr          = params['min_lr']
        self.max_lr          = params['max_lr']
        self.momentum        = params['momentum']
        self.warmup_momentum = params.get('warmup_momentum', 0.8)

        # warmup 結束後接 LinearLR
        self._base = util.LinearLR(args, params, num_steps)

    def step(self, step, optimizer):
        if step < self.warmup_steps:
            # warmup：線性增加 lr，線性增加 momentum
            ratio = step / max(self.warmup_steps, 1)
            lr    = self.min_lr + (self.max_lr - self.min_lr) * ratio
            mom   = self.warmup_momentum + (self.momentum - self.warmup_momentum) * ratio
            for pg in optimizer.param_groups:
                pg['lr'] = lr
                if 'momentum' in pg:
                    pg['momentum'] = mom
        else:
            # warmup 結束後交給原本的 scheduler
            self._base.step(step, optimizer)


# ─────────────────────────────────────────────────────────────
# 訓練
# ─────────────────────────────────────────────────────────────

def train(args, params, dataset_config):
    # ── Load Config ────────────────────────────────────────
    # Args
    WEIGHTS_DIR = args.weights_dir
    INPUT_SIZE  = args.input_size
    BATCH_SIZE  = args.batch_size
    EPOCHS      = args.epochs
    WORKERS     = args.workers

    # Dataset
    DATASET_DIR  = dataset_config["dataset_dir"]
    TRAIN_DIR    = dataset_config["train_dir"]
    VAL_DIR      = dataset_config["val_dir"]
    TRAIN_IMG_DIR = os.path.join(DATASET_DIR, TRAIN_DIR)
    VAL_IMG_DIR = os.path.join(DATASET_DIR, VAL_DIR)
    NUM_CLASSES  = dataset_config["num_classes"]
    # CLASS_NAMES  = dataset_config["class_names"]

    # ── Create Weight Dir ──────────────────────────────────
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ── DATASET ────────────────────────────────────────────
    print(f"[Train] {len(os.listdir(TRAIN_IMG_DIR))} Images")
    print(f"[VAL] {len(os.listdir(VAL_IMG_DIR))} Images")

    if len(os.listdir(TRAIN_IMG_DIR))==0 or len(os.listdir(VAL_IMG_DIR))==0:
        print("Error : Can't Found Dataset")
        return None

    # ── Model ──────────────────────────────────────────────
    model = yolo_v11_n(NUM_CLASSES)
    model.cuda()

    # ── Optimizer ──────────────────────────────────────────
    accumulate = 1
    params = params.copy()
    params["weight_decay"] *= BATCH_SIZE * accumulate / 64

    # optimizer = torch.optim.AdamW(
    #     util.set_params(model, params["weight_decay"]),
    #     lr    = params["max_lr"],
    #     betas = (0.9, 0.999),
    #     eps   = 1e-8,
    # )

    # ✅ SGD
    optimizer = torch.optim.SGD(
        util.set_params(model, params["weight_decay"]),
        lr       = params["min_lr"],   # warmup 起點，scheduler 會推到 max_lr
        momentum = params["momentum"],
        nesterov = True,
    )

    # ── EMA ────────────────────────────────────────────────
    ema = util.EMA(model)

    # ── DataLoader ─────────────────────────────────────────
    train_loader = create_dataloader(
        img_folder = TRAIN_IMG_DIR,
        input_size = INPUT_SIZE,
        batch_size = BATCH_SIZE,
        workers    = WORKERS,
        augment    = True,
        shuffle    = True,
        hyp_params = params,
    )
    num_steps = len(train_loader)

    # ── Scheduler ──────────────────────────────────────────
    warmup_args = SimpleNamespace(epochs=EPOCHS, local_rank=0, world_size=1)
    scheduler = WarmupScheduler(warmup_args, params, num_steps)

    # ── Loss ───────────────────────────────────────────────
    criterion = util.ComputeLoss(model, params)

    # ── Training Loop ──────────────────────────────────────
    best = 0.0
    multi_scale = params.get("multi_scale", False)

    save = {"epoch": 0, "model": copy.deepcopy(ema.ema)}
    torch.save(save, f"{WEIGHTS_DIR}/last.pt")

    with open(f"{WEIGHTS_DIR}/step.csv", "w") as log:
        logger = csv.DictWriter(log, fieldnames=["epoch", "box", "cls", "dfl",
                                                  "Recall", "Precision", "mAP@50", "mAP"])
        logger.writeheader()

        for epoch in range(EPOCHS):
            model.train()

            # 最後 30 epoch 關閉 mosaic 和 copy_paste
            if EPOCHS - epoch == 30:
                train_loader.dataset.mosaic = False
                params["copy_paste"] = 0.0

            avg_box = util.AverageMeter()
            avg_cls = util.AverageMeter()
            avg_dfl = util.AverageMeter()

            print(("\n" + "%10s" * 5) % ("epoch", "memory", "box", "cls", "dfl"))
            p_bar = tqdm.tqdm(enumerate(train_loader), total=num_steps)

            for i, (samples, targets) in p_bar:
                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                # samples = samples.cuda().float() / 255.0
                samples = util.norm(samples.cuda())

                # ── [NEW] 多尺度訓練 ───────────────────────
                # if multi_scale and epoch < EPOCHS - 30:
                if multi_scale and epoch < 10:
                    # 隨機選 size，範圍 INPUT_SIZE ± 50%，步長 32
                    sz = random.randrange(
                        int(INPUT_SIZE * 0.5),
                        int(INPUT_SIZE * 1.5) + 32,
                    ) // 32 * 32
                    if sz != INPUT_SIZE:
                        samples = F.interpolate(
                            samples, size=sz,
                            mode='bilinear', align_corners=False,
                        )

                optimizer.zero_grad()
                outputs = model(samples)
                loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box.update(loss_box.item(), samples.size(0))
                avg_cls.update(loss_cls.item(), samples.size(0))
                avg_dfl.update(loss_dfl.item(), samples.size(0))

                (loss_box + loss_cls + loss_dfl).backward()
                optimizer.step()
                ema.update(model)

                torch.cuda.synchronize()

                mem = f"{torch.cuda.memory_reserved() / 1e9:.3g}G"
                s   = ("%10s" * 2 + "%10.4g" * 3) % (
                    f"{epoch + 1}/{EPOCHS}", mem,
                    avg_box.avg, avg_cls.avg, avg_dfl.avg,
                )
                p_bar.set_description(s)

            # ── Validation ─────────────────────────────────
            last = test(args, dataset_config, params, model=ema.ema)

            logger.writerow({
                "epoch":     str(epoch + 1).zfill(3),
                "box":       f"{avg_box.avg:.4f}",
                "cls":       f"{avg_cls.avg:.4f}",
                "dfl":       f"{avg_dfl.avg:.4f}",
                "mAP":       f"{last[0]:.3f}",
                "mAP@50":    f"{last[1]:.3f}",
                "Recall":    f"{last[2]:.3f}",
                "Precision": f"{last[3]:.3f}",
            })
            log.flush()

            save = {"epoch": epoch + 1, "model": copy.deepcopy(ema.ema)}
            torch.save(save, f"{WEIGHTS_DIR}/last.pt")
            if last[0] > best:
                best = last[0]
                torch.save(save, f"{WEIGHTS_DIR}/best.pt")
            del save

    util.strip_optimizer(f"{WEIGHTS_DIR}/best.pt")
    util.strip_optimizer(f"{WEIGHTS_DIR}/last.pt")


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def test(args, dataset_config, params, model=None):
    # ── Load Config ────────────────────────────────────────
    # Args
    WEIGHTS_DIR = "weights"
    INPUT_SIZE  = args.input_size
    BATCH_SIZE  = args.batch_size
    # EPOCHS      = args.epochs
    WORKERS     = args.workers

    # Dataset
    DATASET_DIR  = dataset_config["dataset_dir"]
    VAL_DIR      = dataset_config["val_dir"]
    VAL_IMG_DIR = os.path.join(DATASET_DIR, VAL_DIR)
    # TRAIN_DIR    = dataset_config["train_dir"]
    # TRAIN_IMG_DIR = os.path.join(DATASET_DIR, TRAIN_DIR)
    NUM_CLASSES  = dataset_config["num_classes"]

    # ── Create Val Loader ──────────────────────────────────
    val_loader = create_dataloader(
        img_folder = VAL_IMG_DIR,
        input_size = INPUT_SIZE,
        batch_size = BATCH_SIZE,
        workers=WORKERS,
        augment    = False,
        shuffle    = False,
        hyp_params = params,
    )

    # ── Inference ──────────────────────────────────────────
    # 實例化後處理器 (放在 CPU 或 GPU 都可以，通常驗證時放 GPU 比較快)
    post_processor = YOLOPostProcessor(nc=NUM_CLASSES).to("cuda")

    plot = False
    if model is None:
        plot  = True
        ckpt  = torch.load(f"{WEIGHTS_DIR}/best.pt", map_location="cuda")
        model = ckpt["model"].float().fuse()

    model.half()
    model.eval()

    iou_v   = torch.linspace(0.5, 0.95, 10).cuda()
    n_iou   = iou_v.numel()
    m_pre   = m_rec = map50 = mean_ap = 0
    metrics = []

    p_bar = tqdm.tqdm(val_loader,
                      desc=("%10s" * 5) % ("", "precision", "recall", "mAP50", "mAP"))

    for samples, targets in p_bar:
        # samples = samples.cuda().half() / 255.0
        samples = util.norm(samples.cuda().half() / 255.0)
        _, _, h, w = samples.shape
        scale = torch.tensor((w, h, w, h)).cuda()

        outputs = post_processor(model(samples))
        outputs = util.non_max_suppression(outputs)

        for i, output in enumerate(outputs):
            idx = targets["idx"] == i
            cls = targets["cls"][idx].cuda()
            box = targets["box"][idx].cuda()
            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue

            if cls.shape[0]:
                target = torch.cat((cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)

            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(
            *metrics, plot=plot, names=params["names"]
        )

    print(("%10s" + "%10.3g" * 4) % ("", m_pre, m_rec, map50, mean_ap))
    model.float()

    return mean_ap, map50, m_rec, m_pre


# ─────────────────────────────────────────────────────────────

import random   # noqa（放這裡避免影響上方 import 順序）

if __name__ == "__main__":
    args, params, dataset_config = load_config()

    for cache_file in glob.glob(os.path.join(dataset_config["dataset_dir"], "labels", "*.cache")):
        os.remove(cache_file)

    util.setup_seed()
    util.setup_multi_processes()
    train(args, params, dataset_config)