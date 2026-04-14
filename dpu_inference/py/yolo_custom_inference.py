#!/usr/bin/env python3
"""
Custom Anchor-free DFL YOLO Inference — Vitis AI VART
======================================================
Architecture:
  - Head output : (B, H, W, 68)  NHWC  = box(64) + cls(4)
  - 3 scales    : 80x80 / 40x40 / 20x20  (stride 8 / 16 / 32)
  - DFL         : ch=16,  4x16=64 channels for box regression
  - num_classes : 4
  - Activation  : ReLU

Backend selection (--device flag):
  dpu  — run on FPGA DPU via VART "run"       (board only)
  cpu  — run on x86/ARM CPU via VART "run_sim" (PC testing, slow)

Run:
  # On FPGA board (default)
  python3 yolo_custom_inference.py --source dog.jpg
  python3 yolo_custom_inference.py --source 0 --device dpu    # webcam
  python3 yolo_custom_inference.py --source video.mp4

  # On PC for logic validation (CPU simulation)
  python3 yolo_custom_inference.py --source dog.jpg --device cpu
  python3 yolo_custom_inference.py --source dog.jpg --device cpu --bench 10
"""

import argparse
import time
from enum import Enum
import numpy as np
import cv2
import xir
import vart

from utils.util import non_max_suppression
from utils.post_process import YOLOPostProcessor

# ─────────────────────────────────────────────
#  User configuration (edit here)
# ─────────────────────────────────────────────
XMODEL_PATH = "your_model.xmodel"
INPUT_W     = 640
INPUT_H     = 640
NC          = 4       # num_classes
DFL_CH      = 16      # Head self.ch
STRIDES     = [8, 16, 32]
CONF_THRESH = 0.25
IOU_THRESH  = 0.45

# Class names — replace with your actual label names
CLASS_NAMES = [f"class{i}" for i in range(NC)]

np.random.seed(0)
PALETTE = np.random.randint(0, 230, (NC, 3), dtype=np.uint8)


# ─────────────────────────────────────────────
#  Backend selector
# ─────────────────────────────────────────────
class RunDevice(Enum):
    DPU = "run"       # FPGA DPU  — VART runner key: "run"
    CPU = "run_sim"   # CPU sim   — VART runner key: "run_sim"  (PC testing)

def parse_device(s: str) -> RunDevice:
    """Parse 'dpu' or 'cpu' string (case-insensitive) to RunDevice."""
    s = s.lower()
    if s == "dpu":
        return RunDevice.DPU
    if s == "cpu":
        return RunDevice.CPU
    raise argparse.ArgumentTypeError(
        f"Unknown device: '{s}'. Valid options: dpu, cpu"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  DFL decode utilities
# ═══════════════════════════════════════════════════════════════════════════

# def make_anchors(feat_h: int, feat_w: int, stride: int,
#                  offset: float = 0.5) -> np.ndarray:
#     """
#     Generate anchor grid center points (in pixel space).
#     Returns shape: (feat_h * feat_w, 2)  [cx, cy]
#     """
#     sx = (np.arange(feat_w) + offset) * stride
#     sy = (np.arange(feat_h) + offset) * stride
#     cx, cy = np.meshgrid(sx, sy)
#     return np.stack([cx, cy], axis=-1).reshape(-1, 2).astype(np.float32)


# def dfl_decode(box_raw: np.ndarray, ch: int = DFL_CH) -> np.ndarray:
#     """
#     DFL softmax decode.
#     Input  : (N, 4*ch)  — raw box output per anchor
#     Output : (N, 4)     — [l, t, r, b] distances
#     """
#     N   = box_raw.shape[0]
#     box = box_raw.reshape(N, 4, ch)
#     # Numerically stable softmax
#     box = box - box.max(axis=-1, keepdims=True)
#     box = np.exp(box)
#     box = box / box.sum(axis=-1, keepdims=True)
#     # Expected value: sum_k( k * p_k )
#     proj = np.arange(ch, dtype=np.float32)
#     return (box * proj).sum(axis=-1)   # (N, 4)

def norm(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: RGB array, shape (B, 3, H, W), dtype uint8 [0, 255]
           or float32 [0.0, 1.0]
    Returns:
        normalized array, shape (B, 3, H, W), dtype float32
    """
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    # std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    if x.dtype == np.uint8:
        scale = 1.0 / (std * 255.0)
        mean  = mean * 255.0
        return (x.astype(np.float32) - mean) * scale
    else:  # float32 or float16
        scale = 1.0 / std
        return ((x.astype(np.float32) - mean) * scale).astype(x.dtype)



# def nms(dets: list, iou_thresh: float) -> list:
#     """Non-maximum suppression via OpenCV NMSBoxes."""
#     if not dets:
#         return []
#     boxes  = np.array([[d[0], d[1], d[2], d[3]] for d in dets], np.float32)
#     scores = np.array([d[4] for d in dets], np.float32)
#     # NMSBoxes expects xywh
#     boxes_xywh      = boxes.copy()
#     boxes_xywh[:, 2] -= boxes_xywh[:, 0]
#     boxes_xywh[:, 3] -= boxes_xywh[:, 1]
#     idx = cv2.dnn.NMSBoxes(
#         boxes_xywh.tolist(), scores.tolist(),
#         CONF_THRESH, iou_thresh)
#     if len(idx) == 0:
#         return []
#     return [dets[i] for i in idx.flatten()]


# ═══════════════════════════════════════════════════════════════════════════
#  VART Runner
# ═══════════════════════════════════════════════════════════════════════════

class YOLORunner:
    def __init__(self, xmodel_path: str, device: RunDevice = RunDevice.DPU):
        self.device    = device
        runner_key     = device.value   # "run" or "run_sim"

        print(f"[INFO] Loading xmodel  : {xmodel_path}")
        print(f"[INFO] Backend         : "
              f"{'DPU (FPGA)' if device == RunDevice.DPU else 'CPU (simulation)'}"
              f"  [runner key: '{runner_key}']")

        self.graph    = xir.Graph.deserialize(xmodel_path)
        self.subgraph = self._get_subgraph()
        self.runner   = vart.Runner.create_runner(self.subgraph, runner_key)

        self.in_tensors  = self.runner.get_input_tensors()
        self.out_tensors = self.runner.get_output_tensors()

        in_shape     = self.in_tensors[0].dims   # [B, H, W, C]
        self.input_h = in_shape[1]
        self.input_w = in_shape[2]
        print(f"[INFO] Input size      : {self.input_w}x{self.input_h}")
        self.__input_scale = 2**int(self.in_tensors[0].get_attr("fix_point")) if self.in_tensors[0].has_attr("fix_point") else 1.0

        print("[INFO] Output tensors  :")
        for t in self.out_tensors:
            fp = t.get_attr("fix_point") if t.has_attr("fix_point") else "N/A"
            print(f"       {t.name}  dims={t.dims}  fix_point={fp}")

        # Pre-compute dequantization scale for each output tensor.
        # DPU outputs int8; float_value = int8_value * 2^(-fix_point)
        self._out_scales = []
        for t in self.out_tensors:
            if t.has_attr("fix_point"):
                self._out_scales.append(2.0 ** -t.get_attr("fix_point"))
            else:
                self._out_scales.append(1.0)   # float model, no scaling

        # Sort outputs by H*W descending: 80x80 -> 40x40 -> 20x20
        self._sorted_idx = sorted(
            range(len(self.out_tensors)),
            key=lambda i: self.out_tensors[i].dims[1] *
                          self.out_tensors[i].dims[2],
            reverse=True
        )

        self.__post_processor = YOLOPostProcessor(nc=NC, ch=DFL_CH, strides=STRIDES)
        
        ### test
        # self.__input_scale = 2**6
        # self.input_h, self.input_w = 640, 640
        # self.__post_processor = YOLOPostProcessor(nc=NC, ch=DFL_CH, strides=STRIDES)

    # ── subgraph lookup ───────────────────────────────────────────────────
    def _get_subgraph(self):
        root     = self.graph.get_root_subgraph()
        children = root.toposort_child_subgraph()

        # DPU backend -> find subgraph with device="DPU"
        # CPU backend -> find subgraph with device="CPU" (run_sim)
        target = "DPU" if self.device == RunDevice.DPU else "CPU"
        matches = [
            c for c in children
            if c.has_attr("device") and
               c.get_attr("device").upper() == target
        ]

        # CPU fallback: some xmodels label the subgraph differently
        if not matches and self.device == RunDevice.CPU:
            matches = [c for c in children if c.has_attr("device")]

        assert matches, (
            f"No subgraph found for device='{target}'. "
            "Verify the .xmodel was compiled correctly."
        )
        return matches[0]

    # ── letterbox preprocess ──────────────────────────────────────────────
    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """
        BGR -> RGB -> letterbox (gray pad 114) -> xint8 (1, H, W, 3)
        DPU input dtype is xint8 (signed int8).
        Conversion: uint8 [0,255] -> int8 [-128,127]  by casting to int16
        then subtracting 128, then casting to int8.
        """
        h0, w0  = bgr.shape[:2]
        scale   = min(self.input_h / h0, self.input_w / w0)
        nh, nw  = int(h0 * scale), int(w0 * scale)

        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Gray pad value 114 in uint8 -> 114 - 128 = -14 in int8
        canvas   = np.full((self.input_h, self.input_w, 3), -14, dtype=np.int8)
        pad_top  = (self.input_h - nh) // 2
        pad_left = (self.input_w - nw) // 2
        # uint8 -> int8: cast via int16 to avoid overflow, then subtract 128
        # canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = \
        #     (resized.astype(np.int16) - 128).astype(np.int8)
        
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = \
            (norm(resized)*self.__input_scale).astype(np.int8)
            # (resized/255.*(2**self.__input_fp)).astype(np.int8)
            

        # Store letterbox params for coordinate restoration
        self._scale    = scale
        self._pad_top  = pad_top
        self._pad_left = pad_left
        self._orig_hw  = (h0, w0)

        return np.expand_dims(canvas, axis=0)   # (1, H, W, 3) int8

    # ── coordinate restoration ────────────────────────────────────────────
    # def _restore_coords(self, dets: list) -> list:
    #     """Map detections from letterbox space back to original image space."""
    #     h0, w0 = self._orig_hw
    #     result = []
    #     for (x1, y1, x2, y2, conf, cls_id) in dets:
    #         x1 = float(np.clip((x1 - self._pad_left) / self._scale, 0, w0))
    #         y1 = float(np.clip((y1 - self._pad_top)  / self._scale, 0, h0))
    #         x2 = float(np.clip((x2 - self._pad_left) / self._scale, 0, w0))
    #         y2 = float(np.clip((y2 - self._pad_top)  / self._scale, 0, h0))
    #         result.append([x1, y1, x2, y2, conf, cls_id])
    #     return result

    def _restore_coords(self, dets: list) -> list:
        """Map detections from letterbox space back to original image space for a batch."""
        batch_result = []
        
        # Iterate through the predictions of each image in the batch 
        # (img_dets is an array of shape (N, 6))
        for batch_index, img_dets in enumerate(dets):
            img_result = []
            
            # If nothing is detected in this image, append an empty list
            if len(img_dets) == 0:
                batch_result.append(img_result)
                continue
            
            # ⚠️ NOTE: If your batch contains images of different original sizes,
            # _orig_hw, _pad_left, _pad_top, and _scale must be Lists,
            # and you need to access the corresponding values using batch_index, for example:
            # h0, w0 = self._orig_hw[batch_index]
            # pad_l = self._pad_left[batch_index]
            # scale = self._scale[batch_index]
            # (The following assumes all images in the batch share the same 
            # preprocessing parameters for now)
            
            h0, w0 = self._orig_hw
            pad_l = self._pad_left
            pad_t = self._pad_top
            scale = self._scale
            
            # Restore coordinates for all bounding boxes in a single image
            for (x1, y1, x2, y2, conf, cls_id) in img_dets:
                x1 = float(np.clip((x1 - pad_l) / scale, 0, w0))
                y1 = float(np.clip((y1 - pad_t) / scale, 0, h0))
                x2 = float(np.clip((x2 - pad_l) / scale, 0, w0))
                y2 = float(np.clip((y2 - pad_t) / scale, 0, h0))
                img_result.append([x1, y1, x2, y2, conf, cls_id])
                
            batch_result.append(img_result)
            
        return batch_result

    # ── full inference pipeline ───────────────────────────────────────────
    def run(self, bgr: np.ndarray) -> list:
        """
        Full inference pipeline:
          1. Letterbox preprocess   (CPU)
          2. VART execute           (DPU or CPU sim)
          3. Decode + NMS           (CPU)
          4. Restore coordinates    (CPU)

        Timing for each stage is stored in self.last_timing (ms) after
        every call, so run_bench() can accumulate and report them separately.
        """
        t0 = time.perf_counter()

        # 1. Preprocess (CPU)
        inp = self._preprocess(bgr)
        t1 = time.perf_counter()
        # return []

        # 2. DPU / CPU-sim execute
        out_bufs = [
            np.empty(t.dims, dtype=np.int8)
            for t in self.out_tensors
        ]
        job_id = self.runner.execute_async([inp], out_bufs)
        self.runner.wait(job_id)
        t2 = time.perf_counter()

        # 3. Dequantize + decode all scales + NMS (CPU)
        all_dets = []
        for rank, orig_idx in enumerate(self._sorted_idx):
            # 1. Remove [0] to keep the Batch dimension. 
            # The shape of feat is now (1, H, W, C)
            # For example: (1, 80, 80, 68)
            feat = out_bufs[orig_idx].astype(np.float32) * self._out_scales[orig_idx]
            
            # 2. Convert NHWC format to the NCHW format expected by PyTorch -> (1, C, H, W)
            # For example: from (1, 80, 80, 68) to (1, 68, 80, 80)
            feat = np.transpose(feat, (0, 3, 1, 2))
            
            # 3. Use append() to add the feature map as a 'whole' to the list, 
            # instead of unrolling it with extend()
            all_dets.append(feat)
        
        all_dets = self.__post_processor(all_dets)
        all_dets = non_max_suppression(all_dets, iou_threshold=IOU_THRESH)
        t3 = time.perf_counter()

        # 4. Restore coordinates (CPU)
        result = self._restore_coords(all_dets)
        t4 = time.perf_counter()

        # Store per-stage timing (ms) for run_bench()
        self.last_timing = {
            "preprocess" : (t1 - t0) * 1000,
            "dpu"        : (t2 - t1) * 1000,
            "postprocess": (t3 - t2) * 1000,
            "restore"    : (t4 - t3) * 1000,
            "total"      : (t4 - t0) * 1000,
        }
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════

def draw(frame: np.ndarray, dets: list) -> np.ndarray:
    out = frame.copy()
    for (x1, y1, x2, y2, conf, cls_id) in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = PALETTE[cls_id % NC].tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name  = CLASS_NAMES[cls_id] if cls_id < NC else f"cls{cls_id}"
        label = f"{name} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw, ty + bl), color, -1)
        cv2.putText(out, label, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Run modes
# ═══════════════════════════════════════════════════════════════════════════

def run_image(model: YOLORunner, src: str, out: str = "result.jpg"):
    frame = cv2.imread(src)
    assert frame is not None, f"[ERROR] Cannot read image: {src}"

    t0   = time.perf_counter()
    dets = model.run(frame)[0]
    ms   = (time.perf_counter() - t0) * 1000

    print(f"[INFO] Inference: {ms:.1f} ms  |  {len(dets)} object(s) detected")
    for d in dets:
        name = CLASS_NAMES[d[5]] if d[5] < NC else f"cls{d[5]}"
        print(f"       {name:<12s} conf={d[4]:.3f}  "
              f"xyxy=[{d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f}]")

    cv2.imwrite(out, draw(frame, dets))
    print(f"[INFO] Result saved to: {out}")


def run_video(model: YOLORunner, src, out: str = "result.avi"):
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    assert cap.isOpened(), f"[ERROR] Cannot open source: {src}"

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[INFO] Source: {W}x{H} @ {fps:.1f} fps")

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))

    cnt, total_ms = 0, 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        dets = model.run(frame)[0]
        total_ms += (time.perf_counter() - t0) * 1000
        cnt += 1

        vis     = draw(frame, dets)
        avg_fps = 1000.0 / (total_ms / cnt)
        cv2.putText(vis, f"FPS:{avg_fps:.1f}", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        writer.write(vis)
        if cnt % 30 == 0:
            print(f"[INFO] frame={cnt}  avg_fps={avg_fps:.1f}")

    cap.release()
    writer.release()
    print(f"[INFO] Processed {cnt} frames. Result saved to: {out}")


def run_video_batched(model: YOLORunner, src, out: str = "result.avi", batch_size: int = 4):
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    assert cap.isOpened(), f"[ERROR] Cannot open source: {src}"

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[INFO] Source: {W}x{H} @ {fps:.1f} fps | Batch Size: {batch_size}")

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))

    cnt, total_ms = 0, 0.0
    frames_buffer = [] # Buffer to collect frames

    while True:
        ret, frame = cap.read()
        
        # 1. Collect frames
        if ret:
            frames_buffer.append(frame)

        # 2. Perform inference when the buffer reaches batch_size, or when the video ends (remaining frames)
        if len(frames_buffer) == batch_size or (not ret and len(frames_buffer) > 0):
            t0 = time.perf_counter()
            
            # Send the entire batch to the model. 'dets_batch' will be a List of length len(frames_buffer)
            # For example: dets_batch = [ [box...], [box...], [box...], [box...] ]
            dets_batch = model.run(frames_buffer) 
            
            total_ms += (time.perf_counter() - t0) * 1000
            
            # 3. Iterate through the batch results to draw bounding boxes and write to the video
            for i, (f, dets) in enumerate(zip(frames_buffer, dets_batch)):
                cnt += 1
                vis = draw(f, dets)
                
                # Calculate and display average FPS
                avg_fps = 1000.0 / (total_ms / cnt) if cnt > 0 else 0
                cv2.putText(vis, f"FPS:{avg_fps:.1f}", (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                writer.write(vis)
                
                if cnt % 30 == 0:
                    print(f"[INFO] frame={cnt}  avg_fps={avg_fps:.1f}")
            
            # 4. Clear the buffer to prepare for the next batch
            frames_buffer.clear()

        # If the video is finished, break out of the loop
        if not ret:
            break

    cap.release()
    writer.release()
    print(f"[INFO] Processed {cnt} frames. Result saved to: {out}")


def run_bench(model: YOLORunner, src: str, repeat: int = 100):
    frame = cv2.imread(src)
    assert frame is not None, f"[ERROR] Cannot read image: {src}"

    backend = "DPU (FPGA)" if model.device == RunDevice.DPU else "CPU (simulation)"
    print(f"[BENCH] Backend  : {backend}")
    print(f"[BENCH] Warming up (10 runs)...")
    for _ in range(10):
        model.run(frame)

    print(f"[BENCH] Running benchmark ({repeat} iterations)...")

    # Accumulators for each stage (ms)
    acc = {"preprocess": 0.0, "dpu": 0.0, "postprocess": 0.0,
           "restore": 0.0, "total": 0.0}

    for _ in range(repeat):
        model.run(frame)
        for k in acc:
            acc[k] += model.last_timing[k]

    avg = {k: v / repeat for k, v in acc.items()}

    # ── DPU label varies by device ────────────────────────────────────────
    dpu_label = "DPU inference" if model.device == RunDevice.DPU \
                else "CPU sim inference"

    print()
    print(f"[BENCH] ┌─────────────────────────────────────────┐")
    print(f"[BENCH] │  Stage breakdown  ({repeat} iterations avg)  │")
    print(f"[BENCH] ├──────────────────────┬──────────┬───────┤")
    print(f"[BENCH] │ Stage                │  Avg(ms) │   %   │")
    print(f"[BENCH] ├──────────────────────┼──────────┼───────┤")
    stages = [
        ("Preprocess (CPU)",  "preprocess"),
        (f"{dpu_label}",      "dpu"),
        ("Decode + NMS (CPU)","postprocess"),
        ("Coord restore (CPU)","restore"),
    ]
    for label, key in stages:
        pct = avg[key] / avg["total"] * 100 if avg["total"] > 0 else 0
        print(f"[BENCH] │ {label:<20s} │ {avg[key]:>8.2f} │ {pct:>5.1f} │")
    print(f"[BENCH] ├──────────────────────┼──────────┼───────┤")
    print(f"[BENCH] │ {'Total':<20s} │ {avg['total']:>8.2f} │ 100.0 │")
    print(f"[BENCH] └──────────────────────┴──────────┴───────┘")
    print()
    print(f"[BENCH] Throughput  : {1000.0 / avg['total']:.1f} FPS  "
          f"(end-to-end)")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Custom anchor-free DFL YOLO inference (Vitis AI VART)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  On FPGA board : python3 %(prog)s --source dog.jpg\n"
            "  On PC (test)  : python3 %(prog)s --source dog.jpg --device cpu\n"
            "  Benchmark     : python3 %(prog)s --source dog.jpg --device cpu --bench 10\n"
        )
    )
    p.add_argument("--xmodel", default=XMODEL_PATH,
                   help="Path to .xmodel file  (default: %(default)s)")
    p.add_argument("--source", default="test.jpg",
                   help="Image path / video path / camera index  (default: %(default)s)")
    p.add_argument("--output", default="result.jpg",
                   help="Output path  (default: %(default)s)")
    p.add_argument("--conf",   type=float, default=CONF_THRESH,
                   help="Confidence threshold  (default: %(default)s)")
    p.add_argument("--iou",    type=float, default=IOU_THRESH,
                   help="NMS IoU threshold  (default: %(default)s)")
    p.add_argument("--bench",  type=int,   default=0,
                   help="Benchmark iterations, 0=off  (default: %(default)s)")
    p.add_argument("--device", type=parse_device, default=RunDevice.DPU,
                   metavar="dpu|cpu",
                   help="Backend: dpu=FPGA, cpu=simulation  (default: dpu)")
    return p.parse_args()


def main():
    global CONF_THRESH, IOU_THRESH
    args = parse_args()
    CONF_THRESH = args.conf
    IOU_THRESH  = args.iou

    model = YOLORunner(args.xmodel, device=args.device)

    if args.bench > 0:
        run_bench(model, args.source, args.bench)
    elif str(args.source).isdigit() or args.source.endswith(
            (".mp4", ".avi", ".mov", ".mkv")):
        run_video(model, args.source, args.output)
    else:
        run_image(model, args.source, args.output)


if __name__ == "__main__":
    main()