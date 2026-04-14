import xir
import vart
import argparse
import time

from utils import util
import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  Argument Parser
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="YOLO DPU Inference")
    parser.add_argument("--source",      type=str,   default="../2308.jpg",       help="Path to input image")
    parser.add_argument("--xmodel",      type=str,   default="your_model.xmodel", help="Path to compiled .xmodel file")
    parser.add_argument("--nc",          type=int,   default=4,                   help="Number of classes")
    parser.add_argument("--conf_thresh", type=float, default=0.25,                help="Confidence threshold for NMS")
    parser.add_argument("--iou_thresh",  type=float, default=0.45,                help="IoU threshold for NMS")
    parser.add_argument("--no_draw",     action="store_true",                     help="Disable drawing bounding boxes")
    parser.add_argument("--benchmark",   action="store_true",                     help="Run benchmark and report FPS")
    parser.add_argument("--bench_runs",  type=int,   default=100,                 help="Number of iterations for benchmark")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Constants (not exposed to CLI)
# ═══════════════════════════════════════════════════════════════════════════
DFL_CH  = 16
STRIDES = [8, 16, 32]

# ═══════════════════════════════════════════════════════════════════════════
#  Global State
# ═══════════════════════════════════════════════════════════════════════════
_runner      : vart.Runner = None
_graph                     = None
_subgraph                  = None
_in_shapes   : list        = None
_out_shapes  : list        = None
_in_fps      : list        = None
_out_fps     : list        = None
_post_processor            = None
CLASS_NAMES  : list        = None


# ═══════════════════════════════════════════════════════════════════════════
#  Preprocess
# ═══════════════════════════════════════════════════════════════════════════
def preprocess_image(img: np.ndarray, input_size: int, channel_format: str = 'HWC') -> tuple:
    """
    Preprocess an input image including resizing, normalization, and format conversion.

    Args:
        img (np.ndarray): Input image as a NumPy array in BGR format.
        input_size (int): Target input size (square side length); the image will be resized accordingly.
        channel_format (str): Channel layout format of the output. Defaults to 'HWC'.
            - 'CHW': PyTorch format, output shape is (1, 3, H, W)
            - 'HWC': xmodel format, output shape is (1, H, W, 3)

    Returns:
        tuple: (norm_img, ratio, pad)
            - norm_img (np.ndarray): Normalized float32 image with batch dimension added.
            - ratio (float): Scaling ratio applied during resize.
            - pad (tuple): Padding applied as (pad_w, pad_h).
        None: If img is None or channel_format is invalid.
    """
    if img is None:
        print("[ERROR]  img is None, please check the input")
        return None

    resized_img, ratio, pad = util.resize(img, input_size)
    norm_img = util.norm(resized_img)

    if channel_format == 'CHW':
        norm_img = norm_img.transpose(2, 0, 1)
        norm_img = np.expand_dims(norm_img, axis=0)
    elif channel_format == 'HWC':
        norm_img = np.expand_dims(norm_img, axis=0)
    else:
        print(f"[ERROR]  Invalid channel_format '{channel_format}', use 'CHW' or 'HWC'")
        return None

    return norm_img, ratio, pad


# ═══════════════════════════════════════════════════════════════════════════
#  DPU Inference
# ═══════════════════════════════════════════════════════════════════════════
def get_tensor_info() -> tuple[list, list, list, list]:
    """
    Print a summary of input/output tensors and return their shapes and fix_points.

    Returns:
        tuple:
            - in_shapes  (list): Input tensor dimensions,  e.g. [[1, H, W, C]].
            - out_shapes (list): Output tensor dimensions, e.g. [[1, H, W, C], ...].
            - in_fps     (list): Input tensor fix_points,  e.g. [8].
            - out_fps    (list): Output tensor fix_points, e.g. [4, 4, 4].
                                 None if fix_point attribute is not available.
    """
    in_tensors  = _runner.get_input_tensors()
    out_tensors = _runner.get_output_tensors()

    print("[INFO]   Input tensors :")
    for t in in_tensors:
        fp = t.get_attr("fix_point") if t.has_attr("fix_point") else "N/A"
        print(f"[INFO]     {t.name}  dims={t.dims}  fix_point={fp}")

    print("[INFO]   Output tensors :")
    for t in out_tensors:
        fp = t.get_attr("fix_point") if t.has_attr("fix_point") else "N/A"
        print(f"[INFO]     {t.name}  dims={t.dims}  fix_point={fp}")

    in_shapes  = [t.dims for t in in_tensors]
    out_shapes = [t.dims for t in out_tensors]
    in_fps     = [t.get_attr("fix_point") if t.has_attr("fix_point") else None for t in in_tensors]
    out_fps    = [t.get_attr("fix_point") if t.has_attr("fix_point") else None for t in out_tensors]

    return in_shapes, out_shapes, in_fps, out_fps


def load_xmodel(xmodel_path: str) -> None:
    """
    Load an xmodel file, print a summary, and initialize the global DPU runner.

    Args:
        xmodel_path (str): Path to the compiled .xmodel file.

    Raises:
        AssertionError: If no DPU subgraph is found in the xmodel.
    """
    global _runner, _graph, _subgraph, _in_shapes, _out_shapes, _in_fps, _out_fps

    print("[INFO] ── Load xmodel ────────────────────────────────")
    print(f"[INFO]   xmodel path  : {xmodel_path}")

    _graph       = xir.Graph.deserialize(xmodel_path)
    root         = _graph.get_root_subgraph()
    children     = root.toposort_child_subgraph()

    matches = [
        c for c in children
        if c.has_attr("device") and
           c.get_attr("device").upper() == "DPU"
    ]

    assert matches, (
        "No DPU subgraph found. "
        "Verify the .xmodel was compiled correctly."
    )

    _subgraph = matches[0]
    _runner   = vart.Runner.create_runner(_subgraph, "run")

    print("[INFO] ── xmodel summary ─────────────────────────────")
    print(f"[INFO]   Graph name   : {_graph.get_name()}")
    print(f"[INFO]   DPU subgraph : {_subgraph.get_name()}")
    _in_shapes, _out_shapes, _in_fps, _out_fps = get_tensor_info()
    print("[INFO]   Runner created successfully")
    print("[INFO] ───────────────────────────────────────────────")


def dpu_inference(inp: np.ndarray) -> list[np.ndarray]:
    """
    Execute DPU inference on a preprocessed input tensor.

    Args:
        inp (np.ndarray): Preprocessed input tensor with shape (1, H, W, C) in HWC format.

    Returns:
        list[np.ndarray]: Raw int8 output buffers, one per output tensor.
    """
    out_bufs = [
        np.zeros(t.dims, dtype=np.int8)
        for t in _runner.get_output_tensors()
    ]

    job_id = _runner.execute_async([inp], out_bufs)
    _runner.wait(job_id)

    return out_bufs


# ═══════════════════════════════════════════════════════════════════════════
#  Post Process
# ═══════════════════════════════════════════════════════════════════════════
def post_process(outputs: list[np.ndarray], conf_thresh: float = 0.2) -> list[np.ndarray]:
    """
    Run post-processing and NMS on raw DPU output buffers.

    Args:
        outputs (list[np.ndarray]): Raw dequantized feature maps from DPU,
                                    each with shape (1, C, H, W).
        conf_thresh (float): Confidence threshold for NMS. Defaults to 0.2.

    Returns:
        list[np.ndarray]: Detected boxes for each image in the batch.
                          Each element has shape (N, 6) where N is the number
                          of detections and 6 = [x1, y1, x2, y2, score, class].
    """
    post_out = _post_processor(outputs)
    nms_out  = util.non_max_suppression(post_out, confidence_threshold=conf_thresh)
    return nms_out


# ═══════════════════════════════════════════════════════════════════════════
#  Run Image
# ═══════════════════════════════════════════════════════════════════════════
def run_image(img: np.ndarray, draw_flag: bool = True) -> list[np.ndarray]:
    """
    Run full inference pipeline on a single cv2 image.

    Args:
        img (np.ndarray): Input image in BGR format.
        draw_flag (bool): If True, draw bounding boxes and save to 'output.jpg'. Defaults to True.

    Returns:
        list[np.ndarray]: Detection results with shape (N, 6) as [x1, y1, x2, y2, score, class].
        Returns None if img is None.
    """
    print("[INFO] ══ Run Image ══════════════════════════════════")
    if img is None:
        print("[ERROR]  img is None, please check the input")
        return None

    orig_shape = img.shape[:2]
    print(f"[INFO]   orig_shape={orig_shape}")
    input_size = _in_shapes[0][1]

    print("[INFO] ── Preprocess ─────────────────────────────────")
    inp, ratio, pad = preprocess_image(img=img, input_size=input_size, channel_format="HWC")
    inp = util.float2fix(inp, _in_fps[0])
    print(f"[INFO]   output={inp.shape}  dtype={inp.dtype}")

    print("[INFO] ── DPU Inference ──────────────────────────────")
    out_bufs = dpu_inference(inp)
    print("[INFO]   DPU inference done")

    print("[INFO] ── Dequantize ─────────────────────────────────")
    for idx, fp in enumerate(_out_fps):
        buf = util.fix2float(out_bufs[idx], fp)
        buf = buf.transpose(0, 3, 1, 2)
        out_bufs[idx] = buf
        print(f"[INFO]   out_bufs[{idx}]  fix_point={fp}  shape={out_bufs[idx].shape}")

    print("[INFO] ── Post Process ───────────────────────────────")
    result = post_process(out_bufs)[0]
    print(f"[INFO]   detections={len(result)}")

    if draw_flag:
        print("[INFO] ── Draw & Save ────────────────────────────────")
        scale_result        = np.zeros_like(result)
        scale_result[:, 4:] = result[:, 4:]
        scale_result[:, :4] = util.scale_boxes(result[:, :4], ratio, pad, orig_shape)
        output_img          = util.draw_boxes(img, scale_result, CLASS_NAMES)
        cv2.imwrite("output.jpg", output_img)
        print("[INFO]   Saved to output.jpg")

    print(f"[INFO] ══ Done  (detections={len(result)}) ══════════════════════")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark
# ═══════════════════════════════════════════════════════════════════════════
def run_benchmark(img: np.ndarray, num_runs: int = 100) -> None:
    """
    Run the full inference pipeline repeatedly and report per-stage timing and FPS.

    Args:
        img (np.ndarray): Input image in BGR format.
        num_runs (int): Number of iterations. Defaults to 100.
    """
    print(f"[INFO] ══ Benchmark  (runs={num_runs}) ══════════════════════")
    if img is None:
        print("[ERROR]  img is None, please check the input")
        return

    input_size = _in_shapes[0][1]

    # Warmup
    print("[INFO]   Warming up (5 runs)...")
    for _ in range(5):
        resized_img, _, _ = util.resize(img, input_size)
        norm_img          = util.norm(resized_img)
        norm_img          = np.expand_dims(norm_img, axis=0)
        inp               = util.float2fix(norm_img, _in_fps[0])
        out_bufs          = dpu_inference(inp)
        for idx, fp in enumerate(_out_fps):
            buf = util.fix2float(out_bufs[idx], fp)
            out_bufs[idx] = buf.transpose(0, 3, 1, 2)
        post_process(out_bufs)

    # Timing accumulators
    t_resize    = 0.0
    t_norm      = 0.0
    t_fix       = 0.0
    t_dpu       = 0.0
    t_deq       = 0.0
    t_transpose = 0.0
    t_decode    = 0.0   # 新增
    t_nms       = 0.0   # 新增
    t_total     = 0.0

    print(f"[INFO]   Running {num_runs} iterations...")
    for _ in range(num_runs):
        t0 = time.perf_counter()

        # Resize
        ta = time.perf_counter()
        resized_img, ratio, pad = util.resize(img, input_size)
        t_resize += (time.perf_counter() - ta) * 1000

        # Normalize
        ta = time.perf_counter()
        norm_img = util.norm(resized_img)
        norm_img = np.expand_dims(norm_img, axis=0)
        t_norm += (time.perf_counter() - ta) * 1000

        # Float2Fix
        ta = time.perf_counter()
        inp = util.float2fix(norm_img, _in_fps[0])
        t_fix += (time.perf_counter() - ta) * 1000

        # DPU
        ta = time.perf_counter()
        out_bufs = dpu_inference(inp)
        t_dpu += (time.perf_counter() - ta) * 1000

        # Dequantize
        ta = time.perf_counter()
        for idx, fp in enumerate(_out_fps):
            out_bufs[idx] = util.fix2float(out_bufs[idx], fp)
        t_deq += (time.perf_counter() - ta) * 1000

        # Transpose
        ta = time.perf_counter()
        for idx in range(len(out_bufs)):
            out_bufs[idx] = out_bufs[idx].transpose(0, 3, 1, 2)
        t_transpose += (time.perf_counter() - ta) * 1000

        # Decode (DFL)
        ta = time.perf_counter()
        post_out = _post_processor(out_bufs)
        t_decode += (time.perf_counter() - ta) * 1000

        # NMS
        ta = time.perf_counter()
        nms_out = util.non_max_suppression(post_out, confidence_threshold=CONF_THRESH)
        t_nms += (time.perf_counter() - ta) * 1000

        t_total += (time.perf_counter() - t0) * 1000

    # Average
    r             = num_runs
    avg_resize    = t_resize    / r
    avg_norm      = t_norm      / r
    avg_fix       = t_fix       / r
    avg_dpu       = t_dpu       / r
    avg_deq       = t_deq       / r
    avg_transpose = t_transpose / r
    avg_decode    = t_decode    / r   # 新增
    avg_nms       = t_nms       / r   # 新增
    avg_total     = t_total     / r
    fps           = 1000.0      / avg_total

    print("[INFO] ── Benchmark Result ───────────────────────────")
    print(f"[INFO]   Runs                     : {num_runs}")
    print(f"[INFO]   Resize                   : {avg_resize   :.2f} ms")
    print(f"[INFO]   Normalize                : {avg_norm     :.2f} ms")
    print(f"[INFO]   Float2Fix                : {avg_fix      :.2f} ms")
    print(f"[INFO]   DPU Inference            : {avg_dpu      :.2f} ms")
    print(f"[INFO]   Dequantize (Fix2Float)   : {avg_deq      :.2f} ms")
    print(f"[INFO]   Transpose NHWC->NCHW     : {avg_transpose:.2f} ms")
    print(f"[INFO]   DFL Decode               : {avg_decode   :.2f} ms")
    print(f"[INFO]   NMS                      : {avg_nms      :.2f} ms")
    print(f"[INFO]   ──────────────────────────────────────────")
    print(f"[INFO]   Total                    : {avg_total    :.2f} ms")
    print(f"[INFO]   FPS                      : {fps          :.2f}")
    print("[INFO] ───────────────────────────────────────────────")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()

    NC          = args.nc
    CONF_THRESH = args.conf_thresh
    IOU_THRESH  = args.iou_thresh

    print("[INFO] ══ Initializing ═══════════════════════════════")
    print(f"[INFO]   source      : {args.source}")
    print(f"[INFO]   xmodel      : {args.xmodel}")
    print(f"[INFO]   nc          : {NC}")
    print(f"[INFO]   conf_thresh : {CONF_THRESH}")
    print(f"[INFO]   iou_thresh  : {IOU_THRESH}")
    print(f"[INFO]   draw        : {not args.no_draw}")
    print(f"[INFO]   benchmark   : {args.benchmark}  runs={args.bench_runs}")

    _post_processor = util.YOLOPostProcessor(NC, DFL_CH, STRIDES)
    print("[INFO]   YOLOPostProcessor initialized")

    CLASS_NAMES = [f"class{i}" for i in range(NC)]

    np.random.seed(0)
    PALETTE = np.random.randint(0, 230, (NC, 3), dtype=np.uint8)

    load_xmodel(args.xmodel)
    img = cv2.imread(args.source)

    if args.benchmark:
        run_benchmark(img, num_runs=args.bench_runs)
    else:
        run_image(img, draw_flag=not args.no_draw)