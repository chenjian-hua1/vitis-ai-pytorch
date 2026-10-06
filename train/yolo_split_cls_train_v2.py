"""
YOLOv11 Custom Training Script
Based on: https://github.com/jahongir7174/YOLOv11-pt

本版修改（相對於原版）：
  [CLEAN] 訓練 / 驗證前處理統一為 preprocess()（原版兩邊其實都正確，util.norm 會依 dtype 處理 /255）
  [FIX]   close mosaic 判斷改為 >=，並在 epochs < close_mosaic 時從第 0 epoch 起關閉
  [FIX]   close mosaic 時重建 train_loader，確保 worker 行程拿到新設定
          （同時關閉 mosaic / copy_paste / mixup，對齊 Ultralytics）
  [FIX]   多尺度尺寸保證為 stride 倍數，random 移到檔案頂端
  [FIX]   TRAIN/VAL 路徑不再重複 join dataset_dir
  [FIX]   test() 的類別名稱改用 dataset_config["class_names"]
  [SPEED] val_loader 只建立一次
  [SPEED] 移除每步 cuda.synchronize / .item()，loss 在 GPU 上累加，每 LOG_INTERVAL 步同步一次
  [SPEED] uint8 non_blocking 傳到 GPU 後才轉 float
  [SPEED] 可選 AMP（params.json 設 "amp": true），搭配梯度裁剪（"clip_grad"，預設 10.0）
  [FIX]   驗證改用 autocast，不再 model.half() → float() 來回轉換 EMA 權重（每個 epoch 都會損失精度）
  [FIX]   torch.load 加 weights_only=False（torch>=2.6 預設 True，無法載入整個模型）
  [SPEED] setup_seed 可設定 deterministic / benchmark（params: "deterministic"）
"""

import copy
import csv
import os
import glob
import random
import yaml
import json
from types import SimpleNamespace
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import tqdm

from arch.yolo11_split_cls import yolo_v11_n, YOLOPostProcessor, verify_split_output, save_split_checkpoint
from utils import util_v2 as util
from utils.dataset_v2 import create_dataloader, make_loader, Dataset

LOG_INTERVAL = 10   # [SPEED] 每幾步同步一次 GPU 並更新進度條
MAX_STRIDE   = 32


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
    for k in ("dataset_dir", "train_dir", "val_dir", "num_classes", "class_names"):
        print(f"  {k:<15} = {dataset_config[k]}")

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
        "train_dir":   train_dir,    # 已包含 dataset_dir
        "val_dir":     val_dir,      # 已包含 dataset_dir
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


def load_config(print_flag: bool = True):
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
# [FIX] 統一前處理：訓練與驗證必須完全相同
# ─────────────────────────────────────────────────────────────

def preprocess(samples):
    """
    uint8 (B,3,H,W) 直接傳到 GPU（傳輸量是 float32 的 1/4），
    util.norm 對 uint8 輸入會自行做 /255 與 ImageNet mean/std 正規化，回傳 float32。
    """
    return util.norm(samples.cuda(non_blocking=True))


# ─────────────────────────────────────────────────────────────
# DataLoader builders
# ─────────────────────────────────────────────────────────────

def build_train_loader(img_dir, input_size, batch_size, workers, hyp, mosaic=True):
    # [FIX] create_dataloader 的 mosaic 預設是 False，原版從來沒有傳入
    #       → 原本的訓練完全沒有用到 mosaic / mix_up / copy_paste
    return create_dataloader(
        img_folder = img_dir,
        input_size = input_size,
        batch_size = batch_size,
        workers    = workers,
        augment    = True,
        shuffle    = True,
        mosaic     = mosaic,
        hyp_params = hyp,
        cache      = hyp.get("cache"),   # None / "ram" / "disk"
    )


def build_val_loader(img_dir, input_size, batch_size, workers, hyp):
    return create_dataloader(
        img_folder = img_dir,
        input_size = input_size,
        batch_size = batch_size,
        workers    = workers,
        augment    = False,
        shuffle    = False,
        mosaic     = False,
        hyp_params = hyp,
        cache      = hyp.get("cache"),
    )


# ─────────────────────────────────────────────────────────────
# Warmup Scheduler（同時 warm up lr 和 momentum）
# ─────────────────────────────────────────────────────────────

class WarmupScheduler:
    """
    Warmup 階段：lr 從 min_lr → max_lr，momentum 從 warmup_momentum → momentum
    Warmup 結束後：接原本的 LinearLR
    """

    def __init__(self, args, params, num_steps):
        self.warmup_steps    = int(max(params['warmup_epochs'] * num_steps, 100))
        self.min_lr          = params['min_lr']
        self.max_lr          = params['max_lr']
        self.momentum        = params['momentum']
        self.warmup_momentum = params.get('warmup_momentum', 0.8)
        self._base = util.LinearLR(args, params, num_steps)

    def step(self, step, optimizer):
        if step < self.warmup_steps:
            ratio = step / max(self.warmup_steps, 1)
            lr    = self.min_lr + (self.max_lr - self.min_lr) * ratio
            mom   = self.warmup_momentum + (self.momentum - self.warmup_momentum) * ratio
            for pg in optimizer.param_groups:
                pg['lr'] = lr
                if 'momentum' in pg:
                    pg['momentum'] = mom
        else:
            self._base.step(step, optimizer)


# ─────────────────────────────────────────────────────────────
# 訓練
# ─────────────────────────────────────────────────────────────

def train(args, params, dataset_config):
    WEIGHTS_DIR = args.weights_dir
    INPUT_SIZE  = args.input_size
    BATCH_SIZE  = args.batch_size
    EPOCHS      = args.epochs
    WORKERS     = args.workers

    # [FIX] train_dir / val_dir 已包含 dataset_dir，不要再 join 一次
    TRAIN_IMG_DIR = dataset_config["train_dir"]
    VAL_IMG_DIR   = dataset_config["val_dir"]
    NUM_CLASSES   = dataset_config["num_classes"]

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    n_train, n_val = len(os.listdir(TRAIN_IMG_DIR)), len(os.listdir(VAL_IMG_DIR))
    print(f"[Train] {n_train} Images")
    print(f"[VAL] {n_val} Images")
    if n_train == 0 or n_val == 0:
        print("Error : Can't Found Dataset")
        return None

    # ── Model ──────────────────────────────────────────────
    model = yolo_v11_n(NUM_CLASSES)
    model.cuda()
    model.set_split_output(False)   # 訓練一律使用合併輸出

    ok, max_err = verify_split_output(model, INPUT_SIZE)
    print(f"[Split output check] {'PASS' if ok else 'FAIL'}  (max abs diff = {max_err:.3e})")
    if not ok:
        print("Error : split_output 與合併輸出不一致，請檢查 Head.forward")
        return None

    # ── Optimizer ──────────────────────────────────────────
    accumulate = 1
    params = params.copy()
    params["weight_decay"] *= BATCH_SIZE * accumulate / 64

    optimizer = torch.optim.SGD(
        util.set_params(model, params["weight_decay"]),
        lr       = params["min_lr"],
        momentum = params["momentum"],
        nesterov = True,
    )

    ema = util.EMA(model)

    # ── [SPEED] AMP（可選）──────────────────────────────────
    use_amp = bool(params.get("amp", False))
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    clip_grad = float(params.get("clip_grad", 10.0))   # Ultralytics / 原 repo 皆為 10；設 0 關閉

    # ── DataLoader ─────────────────────────────────────────
    close_mosaic = int(params.get("close_mosaic", 30))
    close_epoch  = max(EPOCHS - close_mosaic, 0) if close_mosaic > 0 else EPOCHS
    mosaic_on    = close_epoch > 0   # [FIX] epochs <= close_mosaic 時一開始就不開 mosaic

    def closed_hyp(hyp):
        hyp = dict(hyp)
        for key in ("copy_paste", "mix_up", "mosaic"):   # dataset.py 用的是 "mix_up"
            if key in hyp:
                hyp[key] = 0.0
        return hyp

    train_hyp = params if mosaic_on else closed_hyp(params)
    train_loader = build_train_loader(TRAIN_IMG_DIR, INPUT_SIZE, BATCH_SIZE, WORKERS,
                                      train_hyp, mosaic=mosaic_on)
    val_loader = build_val_loader(VAL_IMG_DIR, INPUT_SIZE, BATCH_SIZE, WORKERS, params)   # [SPEED] 只建一次
    num_steps = len(train_loader)

    # ── Scheduler / Loss ───────────────────────────────────
    warmup_args = SimpleNamespace(epochs=EPOCHS, local_rank=0, world_size=1)
    scheduler = WarmupScheduler(warmup_args, params, num_steps)
    criterion = util.ComputeLoss(model, params)

    # ── Multi-scale 設定 ───────────────────────────────────
    multi_scale        = params.get("multi_scale", False)
    multi_scale_epochs = int(params.get("multi_scale_epochs", 10))   # 原本寫死 10，改為可設定
    ms_min = max(int(INPUT_SIZE * 0.5) // MAX_STRIDE * MAX_STRIDE, MAX_STRIDE)
    ms_max = int(INPUT_SIZE * 1.5) // MAX_STRIDE * MAX_STRIDE

    best = 0.0
    save = {"epoch": 0, "model": copy.deepcopy(ema.ema)}
    torch.save(save, f"{WEIGHTS_DIR}/last.pt")

    with open(f"{WEIGHTS_DIR}/step.csv", "w") as log:
        logger = csv.DictWriter(log, fieldnames=["epoch", "box", "cls", "dfl",
                                                  "Recall", "Precision", "mAP@50", "mAP"])
        logger.writeheader()

        for epoch in range(EPOCHS):
            model.train()

            # ── [FIX] close mosaic：改 dataset 後重建 DataLoader，新 worker 才拿得到新設定 ──
            #    （重用同一個 dataset → RAM 快取與 label 不必重新載入）
            if mosaic_on and epoch >= close_epoch:
                print(f"[Aug] epoch {epoch + 1}: 關閉 mosaic / copy_paste / mix_up")
                train_ds = train_loader.dataset
                train_ds.mosaic = False
                train_ds.params = closed_hyp(train_ds.params)
                p_bar = None
                del train_loader                      # 關閉舊的 persistent workers
                train_loader = make_loader(train_ds, BATCH_SIZE, WORKERS, shuffle=True)
                mosaic_on = False

            # [SPEED] loss 在 GPU 上累加，避免每步 .item() 造成同步
            loss_sum = torch.zeros(3, device="cuda")
            n_seen   = 0
            avg      = [0.0, 0.0, 0.0]

            print(("\n" + "%10s" * 5) % ("epoch", "memory", "box", "cls", "dfl"))
            p_bar = tqdm.tqdm(enumerate(train_loader), total=num_steps)

            for i, (samples, targets) in p_bar:
                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = preprocess(samples)   # [FIX] 與 test 相同的前處理

                # ── 多尺度訓練（在 float tensor 上插值）──────
                if multi_scale and epoch < multi_scale_epochs:
                    sz = random.randrange(ms_min, ms_max + MAX_STRIDE, MAX_STRIDE)
                    if sz != samples.shape[-1]:
                        samples = F.interpolate(samples, size=(sz, sz),
                                                mode='bilinear', align_corners=False)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=use_amp):
                    outputs = model(samples)   # split_output=False → 3 張合併輸出
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                scaler.scale(loss_box + loss_cls + loss_dfl).backward()
                if clip_grad > 0:
                    scaler.unscale_(optimizer)               # 先還原梯度尺度再裁剪
                    util.clip_gradients(model, max_norm=clip_grad)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)

                bs = samples.size(0)
                loss_sum += torch.cat([l.detach().float().reshape(1)
                                       for l in (loss_box, loss_cls, loss_dfl)]) * bs
                n_seen   += bs

                if (i + 1) % LOG_INTERVAL == 0 or i + 1 == num_steps:
                    avg = (loss_sum / n_seen).tolist()   # 唯一的同步點
                    mem = f"{torch.cuda.memory_reserved() / 1e9:.3g}G"
                    p_bar.set_description(("%10s" * 2 + "%10.4g" * 3) % (
                        f"{epoch + 1}/{EPOCHS}", mem, *avg))

            # ── Validation ─────────────────────────────────
            last = test(args, dataset_config, params, model=ema.ema, val_loader=val_loader)

            logger.writerow({
                "epoch":     str(epoch + 1).zfill(3),
                "box":       f"{avg[0]:.4f}",
                "cls":       f"{avg[1]:.4f}",
                "dfl":       f"{avg[2]:.4f}",
                "mAP":       f"{last[0]:.3f}",
                "mAP@50":    f"{last[1]:.3f}",
                "Recall":    f"{last[2]:.3f}",
                "Precision": f"{last[3]:.3f}",
            })
            log.flush()

            # ── 存檔 ───────────────────────────────────────
            save = {"epoch": epoch + 1, "model": copy.deepcopy(ema.ema)}
            torch.save(save, f"{WEIGHTS_DIR}/last.pt")
            save_split_checkpoint(ema.ema, f"{WEIGHTS_DIR}/last_split.pt",
                                  epoch=epoch + 1, input_size=INPUT_SIZE)

            if last[0] > best:
                best = last[0]
                torch.save(save, f"{WEIGHTS_DIR}/best.pt")
                save_split_checkpoint(ema.ema, f"{WEIGHTS_DIR}/best_split.pt",
                                      epoch=epoch + 1, input_size=INPUT_SIZE)
                print(f"[Save] best.pt / best_split.pt  (mAP = {best:.3f})")
            del save

    for name in ("best.pt", "last.pt", "best_split.pt", "last_split.pt"):
        util.strip_optimizer(f"{WEIGHTS_DIR}/{name}")


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def test(args, dataset_config, params, model=None, split_output=False, val_loader=None):
    """
    split_output : True 時以 6 張分開輸出跑驗證（用來確認拆開後 mAP 不變）
    val_loader   : 訓練中傳入同一個 loader，避免每個 epoch 重建
    """
    WEIGHTS_DIR = args.weights_dir
    NUM_CLASSES = dataset_config["num_classes"]

    if val_loader is None:
        val_loader = build_val_loader(dataset_config["val_dir"], args.input_size,
                                      args.batch_size, args.workers, params)

    post_processor = YOLOPostProcessor(nc=NUM_CLASSES).to("cuda")

    plot = False
    if model is None:
        plot  = True
        ckpt  = torch.load(f"{WEIGHTS_DIR}/best.pt", map_location="cuda", weights_only=False)
        model = ckpt["model"].float().fuse()

    prev_flag = getattr(model.head, 'split_output', False)
    model.head.split_output = split_output

    # [FIX] 不再 model.half()：對 ema.ema 做 half() 再 float() 會把 EMA 權重每個 epoch
    #       截斷成 fp16 精度一次。改用 autocast，權重維持 fp32，速度相同。
    was_training = model.training
    model.eval()

    iou_v   = torch.linspace(0.5, 0.95, 10).cuda()
    n_iou   = iou_v.numel()
    m_pre   = m_rec = map50 = mean_ap = 0
    metrics = []

    p_bar = tqdm.tqdm(val_loader,
                      desc=("%10s" * 5) % ("", "precision", "recall", "mAP50", "mAP"))

    for samples, targets in p_bar:
        samples = preprocess(samples)   # 與訓練完全相同的前處理
        _, _, h, w = samples.shape
        scale = torch.tensor((w, h, w, h), device="cuda")

        with torch.autocast("cuda", dtype=torch.float16):
            outputs = post_processor(model(samples))
        outputs = outputs.float()
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
            *metrics, plot=plot, names=dataset_config["class_names"]   # [FIX]
        )

    print(("%10s" + "%10.3g" * 4) % ("", m_pre, m_rec, map50, mean_ap))
    model.train(was_training)
    model.head.split_output = prev_flag

    return mean_ap, map50, m_rec, m_pre


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args, params, dataset_config = load_config()
    if args is None:
        raise SystemExit(1)

    # deterministic=True 可逐位元重現但較慢；沒開 multi_scale 時讓 cuDNN benchmark 挑最快演算法
    deterministic = bool(params.get("deterministic", False))
    util.setup_seed(deterministic=deterministic,
                    benchmark=not params.get("multi_scale", False))
    util.setup_multi_processes()
    train(args, params, dataset_config)