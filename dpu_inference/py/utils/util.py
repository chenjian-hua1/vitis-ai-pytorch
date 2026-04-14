from typing import List
import cv2
import numpy as np
import time

# ---------------------------- Fix & Float cvt ---------------------------------- #
def fix2float(data: np.ndarray, fix_point: int) -> np.ndarray:
    """
    Convert fixed-point integer to float.

    Args:
        data      : numpy array of int8 fixed-point values
        fix_point : fix_point exponent from tensor attribute
    Returns:
        float32 numpy array
    """
    # scale = np.exp2(-1.0 * fix_point)  # 2^(-fix_point)
    data = data.astype(np.float32)
    data *= np.exp2(-fix_point, dtype=np.float32)
    return data


def float2fix(data: np.ndarray, fix_point: int) -> np.ndarray:
    """
    Convert float to fixed-point integer (int8).

    Args:
        data      : numpy array of float32 values
        fix_point : fix_point exponent from tensor attribute
    Returns:
        int8 numpy array
    """
    data *= (2**fix_point)

    # Clip to int8 range [-128, 127] and cast
    data  = np.clip(data, -128.0, 127.0)

    return data.astype(np.int8)


# ---------------------------- Base Function ------------------------------------ #
def wh2xy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.
    x : (N, 4+)  center-x, center-y, width, height in the first 4 columns
    """
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top-left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top-left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom-right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom-right y
    return y

def xyxy2xywh(box: np.ndarray) -> np.ndarray:
    """(x1, y1, x2, y2) → (x, y, w, h)，cv2.dnn.NMSBoxes 需要此格式"""
    out = np.empty_like(box)
    out[:, 0] = box[:, 0]                   # x
    out[:, 1] = box[:, 1]                   # y
    out[:, 2] = box[:, 2] - box[:, 0]      # w
    out[:, 3] = box[:, 3] - box[:, 1]      # h
    return out

def make_anchors(x: List[np.ndarray], strides: List[int],
                 offset: float = 0.5):
    """
    Generate anchor points for each feature map scale.

    x       : list of feature maps, each with shape (B, C, H, W)
    strides : downsampling factors, e.g. [8, 16, 32]
    offset  : anchor offset from grid origin (default 0.5 = cell center)

    Returns
    -------
    anchors      : (A, 2)   anchor (x, y) coordinates in feature-map space
    stride_tensor: (A, 1)   corresponding stride value for each anchor
    """
    anchor_list, stride_list = [], []

    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape

        # Build grid coordinates for the current scale
        sx = np.arange(w, dtype=np.float32) + offset   # (W,)
        sy = np.arange(h, dtype=np.float32) + offset   # (H,)
        grid_y, grid_x = np.meshgrid(sy, sx, indexing='ij')  # (H, W)

        # Flatten and stack into (H*W, 2)
        anchors = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        anchor_list.append(anchors)
        stride_list.append(np.full((h * w, 1), stride, dtype=np.float32))

    return (
        np.concatenate(anchor_list, axis=0),   # (A, 2)
        np.concatenate(stride_list, axis=0),   # (A, 1)
    )


# ----------------------------------- Pre Process ------------------------------------------- #
float_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # shape (3,)
float_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # shape (3,)
float_scale = 1/float_std

uint8_mean = float_mean * 255.
uint8_scale = 1 / (float_std * 255.0)
bias = float_mean*float_scale

def norm(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: RGB numpy array from cv2, shape (H, W, 3)
           dtype uint8 [0, 255] or float32 [0.0, 1.0]
    Returns:
        normalized numpy array, shape (H, W, 3), dtype float32
    """
    if x.dtype == np.uint8:
        # scale = 1 / (std * 255.0)
        # mean  = mean * 255.0
        # return (x.astype(np.float32) - uint8_mean) * uint8_scale

        x = x.astype(np.float32)
        x *= uint8_scale
        x += bias
        return x
        # return x.astype(np.float32)*uint8_scale + bias
    else:  # float32
        return (x.astype(np.float32) - float_mean) * float_scale

## fma cpu 加速
import numexpr as ne
def norm_fma(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        x = x.astype(np.float32)
        return ne.evaluate('x * uint8_scale + bias')
    else:
        x = x.astype(np.float32)
        return ne.evaluate('(x - float_mean) * float_scale')

def resize(img:np.ndarray, input_size:int):
    shape = img.shape[:2]
    r = min(input_size / shape[0], input_size / shape[1])
    r = min(r, 1.0)

    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2
    if shape[::-1] != pad:
        img = cv2.resize(img, dsize=pad,
                           interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return img, (r, r), (w, h)


# ----------------------------------- Post Process ------------------------------------------ #
def non_max_suppression(outputs: np.ndarray, confidence_threshold=0.001, iou_threshold=0.65):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]       # batch size
    nc = outputs.shape[1] - 4  # number of classes

    # outputs shape: (B, 8, 8400), take class scores along axis=1
    xc = outputs[:, 4:4 + nc].max(axis=1) > confidence_threshold  # (B, 8400)

    start = time.time()
    limit = 0.5 + 0.05 * bs
    output = [np.zeros((0, 6))] * bs

    for index, x in enumerate(outputs):  # x shape: (8, 8400)
        x = x.T[xc[index]]              # (8, 8400) -> (8400, 8) -> filter by confidence

        if x.shape[0] == 0:
            continue

        box, cls = x[:, :4], x[:, 4:4 + nc]  # (N, 4), (N, nc)
        box = wh2xy(box)                       # (cx, cy, w, h) -> (x1, y1, x2, y2)

        if nc > 1:
            i, j = np.where(cls > confidence_threshold)
            x = np.concatenate([
                box[i],
                x[i, 4 + j][:, None],
                j[:, None].astype(np.float32)
            ], axis=1)
        else:  # best class only
            conf = cls.max(axis=1, keepdims=True)
            j    = cls.argmax(axis=1, keepdims=True)
            x    = np.concatenate([box, conf, j.astype(np.float32)], axis=1)
            x    = x[conf.flatten() > confidence_threshold]

        if x.shape[0] == 0:
            continue

        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort()[::-1][:max_nms]]

        # Batched NMS: offset boxes by class to separate different classes
        c             = x[:, 5:6] * max_wh
        boxes_offset  = x[:, :4] + c   # (x1, y1, x2, y2)
        scores        = x[:, 4]

        # cv2.dnn.NMSBoxes requires (x, y, w, h)
        boxes_xywh = xyxy2xywh(boxes_offset)
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            confidence_threshold,
            iou_threshold
        )

        if len(indices) == 0:
            continue

        indices        = np.array(indices).flatten()[:max_det]
        output[index]  = x[indices]

        if (time.time() - start) > limit:
            break

    return output

def _dfl_decode(x: np.ndarray, ch: int = 16) -> np.ndarray:
    """
    Generalized Focal Loss DFL decoding (numpy equivalent of DFL module).
    Replaces the Conv2d weight-sum with an explicit softmax + weighted sum.

    x      : (B, 4*ch, A)
    ch     : number of DFL bins (default 16)
    Returns: (B, 4, A)
    """
    B, _, A = x.shape
    # Reshape to (B, 4, ch, A) to apply softmax over the ch dimension
    x = x.reshape(B, 4, ch, A)

    # Softmax over ch axis for numerical stability
    x = x - x.max(axis=2, keepdims=True)
    e = np.exp(x)
    s = e / e.sum(axis=2, keepdims=True)            # (B, 4, ch, A)

    # Weighted sum: equivalent to Conv2d with weights [0, 1, ..., ch-1]
    arange = np.arange(ch, dtype=np.float32).reshape(1, 1, ch, 1)  # (1, 1, ch, 1)
    return (s * arange).sum(axis=2)                 # (B, 4, A)


class YOLOPostProcessor:
    """
    [call function] Decode raw YOLO multi-scale feature maps into box + class predictions.

    x           : list of feature maps, each (B, no, H, W)  float32
    conf_thresh : confidence threshold (unused here, applied in NMS)

    Returns
    -------
    output : (B, 4+nc, Anchors)
                axis-1 layout: [cx, cy, w, h, cls_score_0, ..., cls_score_{nc-1}]
    """
    
    def __init__(self, nc: int = 80, ch: int = 16, strides: List[int] = [8, 16, 32]):
        """
        Args:
            nc      (int)       : Number of classes (e.g. 80 for COCO, 4 for custom). Default: 80
            ch      (int)       : Number of DFL bins per coordinate. Default: 16
            strides (List[int]) : Downsampling strides for each detection scale.
                                8  -> large feature map  (80×80 for 640 input)
                                16 -> medium feature map (40×40 for 640 input)
                                32 -> small feature map  (20×20 for 640 input)
                                Default: [8, 16, 32]
        """
        self.nc      = nc
        self.ch      = ch
        self.no      = nc + ch * 4
        self.strides = strides

    def __call__(self, x: List[np.ndarray],
                 conf_thresh: float = 0.25) -> np.ndarray:
        """
        Decode raw YOLO multi-scale feature maps into box + class predictions.

        x           : list of feature maps, each (B, no, H, W)  float32
        conf_thresh : confidence threshold (unused here, applied in NMS)

        Returns
        -------
        output : (B, 4+nc, Anchors)
                 axis-1 layout: [cx, cy, w, h, cls_score_0, ..., cls_score_{nc-1}]
        """
        B = x[0].shape[0]

        # ── 1. Generate anchors and per-anchor stride values ─────────────────
        # anchors : (2, A),  stride_vals : (1, A)
        anchors, stride_vals = make_anchors(x, self.strides)

        # Add the missing transpose operation to match PyTorch's i.transpose(0, 1)
        # Convert anchors shape from (A, 2) to (2, A)
        anchors = anchors.T 
        
        # stride_vals might be (A, 1) or (A,) depending on the make_anchors implementation
        # Ensure it is (1, A) or (A,) for proper broadcasting later
        if stride_vals.ndim == 2:
            stride_vals = stride_vals.T

        # ── 2. Concatenate all scales → (B, no, A) ───────────────────────────
        x_cat = np.concatenate(
            [xi.reshape(B, self.no, -1) for xi in x], axis=2
        )

        # ── 3. Split into box and class branches ─────────────────────────────
        split  = 4 * self.ch
        box_raw = x_cat[:, :split, :]    # (B, 4*ch, A)
        cls_raw = x_cat[:, split:, :]    # (B, nc,   A)

        # ── 4. DFL decoding: distribution → distances ─────────────────────────
        # Equivalent to DFL module forward pass
        dfl_out = _dfl_decode(box_raw, self.ch)   # (B, 4, A)

        # ── 5. dist2bbox: anchor ± offset → cxcywh, then scale to image space ─
        # Mirrors narrow(1,0,2) and narrow(1,2,2) from the PyTorch version
        a = anchors[np.newaxis] - dfl_out[:, :2, :]   # (B, 2, A)  left-top
        b = anchors[np.newaxis] + dfl_out[:, 2:, :]   # (B, 2, A)  right-bottom

        cx_cy = (a + b) / 2                            # (B, 2, A)
        wh    = b - a                                  # (B, 2, A)
        box   = np.concatenate([cx_cy, wh], axis=1) * stride_vals  # (B, 4, A)

        # ── 6. Class sigmoid ─────────────────────────────────────────────────
        # cls_raw  = cls_raw - cls_raw.max(axis=1, keepdims=True)  # numerical stability
        cls_prob = 1.0 / (1.0 + np.exp(-cls_raw))               # (B, nc, A)

        # ── 7. Concatenate box + cls → (Batch, 4+nc, Anchor) ─────────────────────────
        return np.concatenate([box, cls_prob], axis=1)


# -------------------------------- Visualize ----------------------------------------------------
def scale_boxes(boxes: np.ndarray, ratio: tuple, pad: tuple, orig_shape: tuple) -> np.ndarray:
    """
    Scale boxes from resized/padded image back to original image coordinates.

    Args:
        boxes      : (N, 4) array of boxes in (x1, y1, x2, y2) format
        ratio      : (r, r) scale ratio from resize()
        pad        : (w, h) padding from resize()
        orig_shape : (H, W) original image shape
    Returns:
        boxes      : (N, 4) boxes in original image coordinates
    """
    boxes = boxes.copy().astype(np.float32)

    # Remove padding
    boxes[:, 0] -= pad[0]  # x1 - left padding
    boxes[:, 1] -= pad[1]  # y1 - top padding
    boxes[:, 2] -= pad[0]  # x2 - left padding
    boxes[:, 3] -= pad[1]  # y2 - top padding

    # Scale back to original size
    boxes[:, 0] /= ratio[0]  # x1
    boxes[:, 1] /= ratio[1]  # y1
    boxes[:, 2] /= ratio[0]  # x2
    boxes[:, 3] /= ratio[1]  # y2

    # Clip to original image boundary
    boxes[:, 0] = boxes[:, 0].clip(0, orig_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clip(0, orig_shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clip(0, orig_shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clip(0, orig_shape[0])  # y2

    return boxes

def draw_boxes(img: np.ndarray, boxes: np.ndarray, class_names: list = None) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        img         : original image (H, W, 3) BGR
        boxes       : (N, 6) array of (x1, y1, x2, y2, score, class_id)
        class_names : list of class names, e.g. ['cat', 'dog', 'car']
    Returns:
        img         : image with bounding boxes drawn
    """
    img = img.copy()

    for det in boxes:
        x1, y1, x2, y2, score, class_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)

        # Generate color per class
        color = tuple(int(c) for c in np.array([
            (class_id * 67 + 100) % 255,
            (class_id * 113 + 50) % 255,
            (class_id * 179 + 150) % 255
        ]))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

        # Build label
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {score:.2f}"
        else:
            label = f"Class {class_id}: {score:.2f}"

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, thickness=-1)

        # Draw label text
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

    return img