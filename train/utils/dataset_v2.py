"""
Dataset / DataLoader

本版修改（相對於原版）：
  [FIX]   copy_paste 改用 load_image()：來源圖先縮到 input_size 尺度，物件大小才和目標圖一致
          （原本直接讀原圖，大圖的物件縮放後常超出畫面被跳過，或大到不合理）
  [FIX]   copy_paste 加入遮擋檢查（IoA），不再把物件貼在既有物件上卻保留被蓋掉的 label
  [FIX]   label cache 真的會被讀取（以檔案大小 + mtime 做 hash 驗證），且改用 pickle
          （torch>=2.6 的 torch.load 預設 weights_only=True，讀不了 numpy dict）
  [FIX]   找圖片改為不分大小寫（Linux 上 *.JPG / *.PNG 原本會被漏掉），並排序確保可重現
  [FIX]   損壞圖片（PIL 無法開啟）不再讓整個訓練 crash，改為略過並統計數量
  [FIX]   Albumentations 相容 1.x / 2.x 參數名稱；例外不再靜默吞掉（每個 worker 提示一次）
  [FIX]   mix_up 參數讀取改用 .get，缺 key 時不會 KeyError
  [SPEED] 影像快取：cache='ram'（記憶體）或 'disk'（.npy），存的是已縮到 input_size 的圖
  [SPEED] 無快取時，大張 JPEG 用 IMREAD_REDUCED_COLOR_2/4/8 直接以低解析度解碼
  [SPEED] label 驗證改為多執行緒，且有 cache 時直接跳過
  [SPEED] mix_up 改用 cv2.addWeighted（不再產生 float64 中間陣列）
  [SPEED] DataLoader：persistent_workers、prefetch_factor、worker 內 cv2.setNumThreads(0)
"""

import hashlib
import math
import os
import pickle
import random
from multiprocessing.pool import ThreadPool

import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader

FORMATS       = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')
CACHE_VERSION = 2
NUM_THREADS   = min(8, max(1, os.cpu_count() or 1))


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment, mosaic, cache=None):
        """
        cache : None / 'ram' / 'disk'
          'ram'  → 縮放後的影像放在記憶體（最快；Linux fork 下 worker 共用同一份）
          'disk' → 縮放後的影像存成 <原檔名>.<input_size>.npy（Windows 或記憶體不足時用）
        """
        self.params     = params
        self.mosaic     = mosaic
        self.augment    = augment
        self.input_size = input_size

        labels, shapes  = self.load_label(filenames)
        self.filenames  = list(labels.keys())
        self.labels     = list(labels.values())
        self.shapes     = [shapes[f] for f in self.filenames]   # 原圖 (h, w)
        self.n          = len(self.filenames)
        self.indices    = range(self.n)

        self.cache  = cache
        self.images = [None] * self.n
        if cache in ('ram', 'disk'):
            self.cache_images(cache)

        self.albumentations = Albumentations() if augment else None

    # ─────────────────────────────────────────────────────────
    def __getitem__(self, index):
        index = self.indices[index]

        if self.mosaic and random.random() < self.params.get('mosaic', 1.0):
            image, label = self.load_mosaic(index, self.params)

            # MixUp
            if random.random() < self.params.get('mix_up', 0.0):
                index2 = random.choice(self.indices)
                image2, label2 = self.load_mosaic(index2, self.params)
                image, label = mix_up(image, label, image2, label2)

            # Copy-Paste
            if random.random() < self.params.get('copy_paste', 0.0):
                image, label = copy_paste(image, label, self,
                                          n_paste=int(self.params.get('copy_paste_n', 3)))
        else:
            image, _ = self.load_image(index)
            h, w = image.shape[:2]
            image, ratio, pad = resize(image, self.input_size, self.augment)

            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            if self.augment:
                image, label = random_perspective(image, label, self.params)

        nl = len(label)
        h, w = image.shape[:2]
        cls = label[:, 0:1]
        box = xy2wh(label[:, 1:5], w, h)

        if self.augment:
            image, box, cls = self.albumentations(image, box, cls)
            cls = numpy.asarray(cls, dtype=numpy.float32).reshape(-1, 1)
            box = numpy.asarray(box, dtype=numpy.float32).reshape(-1, 4)
            nl = len(box)

            augment_hsv(image, self.params)

            if random.random() < self.params['flip_ud']:
                image = numpy.flipud(image)
                if nl:
                    box[:, 1] = 1 - box[:, 1]

            if random.random() < self.params['flip_lr']:
                image = numpy.fliplr(image)
                if nl:
                    box[:, 0] = 1 - box[:, 0]

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 4))
        if nl:
            target_cls = torch.from_numpy(numpy.ascontiguousarray(cls, dtype=numpy.float32))
            target_box = torch.from_numpy(numpy.ascontiguousarray(box, dtype=numpy.float32))

        sample = numpy.ascontiguousarray(image.transpose((2, 0, 1))[::-1])   # HWC BGR → CHW RGB
        return torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl)

    def __len__(self):
        return self.n

    # ─────────────────────────────────────────────────────────
    # 影像讀取 / 快取
    # ─────────────────────────────────────────────────────────
    def npy_path(self, i):
        return os.path.splitext(self.filenames[i])[0] + f'.{self.input_size}.npy'

    def _read_resized(self, i, interp=None):
        """讀圖並把長邊縮到 input_size。大張 JPEG 直接以 1/2、1/4、1/8 解析度解碼。"""
        path   = self.filenames[i]
        h0, w0 = self.shapes[i]
        flag   = cv2.IMREAD_COLOR
        if path.lower().endswith(('.jpg', '.jpeg')):
            ratio = max(h0, w0) / self.input_size
            if ratio >= 8:
                flag = cv2.IMREAD_REDUCED_COLOR_8
            elif ratio >= 4:
                flag = cv2.IMREAD_REDUCED_COLOR_4
            elif ratio >= 2:
                flag = cv2.IMREAD_REDUCED_COLOR_2

        image = cv2.imread(path, flag)
        if image is None:
            raise FileNotFoundError(f'Image Not Found: {path}')

        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            if interp is None:
                interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            image = cv2.resize(image, (max(1, round(w * r)), max(1, round(h * r))),
                               interpolation=interp)
        return image

    def load_image(self, i):
        """
        回傳 (長邊 = input_size 的影像, 原圖 (h, w))。
        RAM 快取的陣列是唯讀的：任何 in-place 修改都會直接報錯，而不是默默污染快取。
        """
        cached = self.images[i]
        if cached is not None:
            return cached, self.shapes[i]

        if self.cache == 'disk':
            npy = self.npy_path(i)
            if os.path.exists(npy):
                try:
                    return numpy.load(npy), self.shapes[i]
                except (OSError, ValueError):
                    pass   # 損壞的 .npy → 退回直接讀圖

        interp = resample() if self.augment else None
        return self._read_resized(i, interp), self.shapes[i]

    def cache_images(self, mode):
        def work(i):
            if mode == 'disk':
                npy = self.npy_path(i)
                if not os.path.exists(npy):
                    tmp = npy[:-4] + '.tmp.npy'
                    numpy.save(tmp, self._read_resized(i))
                    os.replace(tmp, npy)   # 原子寫入，避免中斷時留下半個檔案
                return i, None, os.path.getsize(npy)
            image = self._read_resized(i)
            image.flags.writeable = False
            return i, image, image.nbytes

        total = 0
        with ThreadPool(NUM_THREADS) as pool:
            for k, (i, image, nbytes) in enumerate(pool.imap_unordered(work, range(self.n))):
                if image is not None:
                    self.images[i] = image
                total += nbytes
                if (k + 1) % 500 == 0 or k + 1 == self.n:
                    print(f'\r[Cache:{mode}] {k + 1}/{self.n}  ({total / 1e9:.2f} GB)', end='')
        print()

    def load_mosaic(self, index, params):
        label4 = []
        s      = self.input_size
        border = [-s // 2, -s // 2]
        image4 = numpy.zeros((s * 2, s * 2, 3), dtype=numpy.uint8)

        xc = int(random.uniform(-border[0], 2 * s + border[1]))
        yc = int(random.uniform(-border[0], 2 * s + border[1]))

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            image, _ = self.load_image(index)
            shape = image.shape
            if i == 0:
                x1a = max(xc - shape[1], 0);  y1a = max(yc - shape[0], 0)
                x2a = xc;                      y2a = yc
                x1b = shape[1] - (x2a - x1a); y1b = shape[0] - (y2a - y1a)
                x2b = shape[1];                y2b = shape[0]
            elif i == 1:
                x1a = xc;                      y1a = max(yc - shape[0], 0)
                x2a = min(xc + shape[1], s * 2); y2a = yc
                x1b = 0;                       y1b = shape[0] - (y2a - y1a)
                x2b = min(shape[1], x2a - x1a); y2b = shape[0]
            elif i == 2:
                x1a = max(xc - shape[1], 0);  y1a = yc
                x2a = xc;                      y2a = min(s * 2, yc + shape[0])
                x1b = shape[1] - (x2a - x1a); y1b = 0
                x2b = shape[1];                y2b = min(y2a - y1a, shape[0])
            else:
                x1a = xc;                      y1a = yc
                x2a = min(xc + shape[1], s * 2)
                y2a = min(s * 2, yc + shape[0])
                x1b = 0;                       y1b = 0
                x2b = min(shape[1], x2a - x1a); y2b = min(y2a - y1a, shape[0])

            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            label = self.labels[index].copy()
            if len(label):
                label[:, 1:] = wh2xy(label[:, 1:], shape[1], shape[0], x1a - x1b, y1a - y1b)
            label4.append(label)

        label4 = numpy.concatenate(label4, 0)
        numpy.clip(label4[:, 1:], 0, 2 * s, out=label4[:, 1:])

        return random_perspective(image4, label4, params, border)

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = zip(*batch)
        indices = torch.cat([idx + i for i, idx in enumerate(indices)], dim=0)
        targets = {'cls': torch.cat(cls, dim=0),
                   'box': torch.cat(box, dim=0),
                   'idx': indices}
        return torch.stack(samples, dim=0), targets

    # ─────────────────────────────────────────────────────────
    # Label 讀取 + 快取
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def files_hash(filenames):
        h = hashlib.md5()
        for f in filenames:
            for p in (f, label_path(f)):
                try:
                    st = os.stat(p)
                    h.update(f'{p}|{st.st_size}|{st.st_mtime_ns};'.encode())
                except OSError:
                    h.update(f'{p}|missing;'.encode())
        return h.hexdigest()

    @staticmethod
    def load_label(filenames):
        cache_path = f'{os.path.dirname(filenames[0])}.cache'
        key = Dataset.files_hash(filenames)

        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if cache.get('version') == CACHE_VERSION and cache.get('hash') == key:
                return cache['labels'], cache['shapes']
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, KeyError):
            pass

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.map(verify_image_label, filenames)

        labels, shapes, bad = {}, {}, []
        for filename, label, shape, msg in results:
            if label is None:
                bad.append(msg)
                continue
            labels[filename] = label
            shapes[filename] = shape

        if bad:
            print(f'[Warning] 略過 {len(bad)} 張圖片（損壞或標註格式錯誤），例如：')
            for msg in bad[:5]:
                print(f'          {msg}')

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({'version': CACHE_VERSION, 'hash': key,
                             'labels': labels, 'shapes': shapes}, f)
        except OSError as e:
            print(f'[Warning] 無法寫入 label cache {cache_path}: {e}')

        return labels, shapes


def label_path(img_path):
    a = f'{os.sep}images{os.sep}'
    b = f'{os.sep}labels{os.sep}'
    return b.join(img_path.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'


def verify_image_label(filename):
    """回傳 (filename, label 或 None, (h, w), 錯誤訊息)"""
    try:
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image.verify()
        w, h = image.size
        assert w > 9 and h > 9, f'image too small {w}x{h}'
        assert image.format.lower() in FORMATS, f'unsupported format {image.format}'

        lbl = label_path(filename)
        label = numpy.zeros((0, 5), dtype=numpy.float32)
        if os.path.isfile(lbl):
            with open(lbl) as f:
                rows = [x.split() for x in f.read().strip().splitlines() if len(x)]
            if rows:
                label = numpy.array(rows, dtype=numpy.float32)
                assert label.shape[1] == 5, f'label needs 5 columns, got {label.shape[1]}'
                assert (label >= 0).all(), 'negative label values'
                assert (label[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinates'
                _, i = numpy.unique(label, axis=0, return_index=True)
                if len(i) < len(label):
                    label = label[numpy.sort(i)]
        return filename, label, (h, w), ''
    except Exception as e:
        return filename, None, None, f'{filename}: {e}'


# ─────────────────────────────────────────────────────────────
# Copy-Paste（bbox 版本）
# ─────────────────────────────────────────────────────────────

def bbox_ioa(box, boxes, eps=1e-7):
    """box (4,) 與 boxes (N,4) 的交集 / boxes 面積，用來判斷既有物件被蓋住多少。"""
    ix1 = numpy.maximum(box[0], boxes[:, 0])
    iy1 = numpy.maximum(box[1], boxes[:, 1])
    ix2 = numpy.minimum(box[2], boxes[:, 2])
    iy2 = numpy.minimum(box[3], boxes[:, 3])
    inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
    area  = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + eps
    return inter / area


def copy_paste(image, label, dataset, n_paste=3, ioa_thresh=0.3, max_tries=3,
               scale_range=(0.5, 1.5)):
    """
    從資料集隨機挑物件，裁出 bbox 區域貼到 image 上。

    image : mosaic 後的影像（可寫入）
    label : (N, 5) [cls, x1, y1, x2, y2] pixel xyxy

    來源圖透過 dataset.load_image() 取得，已縮到 input_size 尺度（和 mosaic 一致），
    有快取時不需重新解碼。貼上位置若會遮住任何既有物件超過 ioa_thresh 就換位置重試。
    """
    h, w   = image.shape[:2]
    boxes  = label[:, 1:5].astype(numpy.float32) if len(label) else numpy.zeros((0, 4), numpy.float32)
    pasted = []

    for _ in range(n_paste):
        src_idx = random.randrange(dataset.n)
        src_lbs = dataset.labels[src_idx]          # YOLO 正規化 (M, 5)
        if len(src_lbs) == 0:
            continue

        src_img, _ = dataset.load_image(src_idx)
        sh, sw = src_img.shape[:2]

        cls_id, cx, cy, bw, bh = (float(v) for v in src_lbs[random.randrange(len(src_lbs))])
        x1 = max(0, int((cx - bw / 2) * sw))
        y1 = max(0, int((cy - bh / 2) * sh))
        x2 = min(sw, int((cx + bw / 2) * sw))
        y2 = min(sh, int((cy + bh / 2) * sh))
        obj_w, obj_h = x2 - x1, y2 - y1
        if obj_w < 4 or obj_h < 4:
            continue

        scale  = random.uniform(*scale_range)
        new_ow = max(4, int(obj_w * scale))
        new_oh = max(4, int(obj_h * scale))
        if new_ow >= w or new_oh >= h:
            continue

        for _ in range(max_tries):
            px = random.randint(0, w - new_ow)
            py = random.randint(0, h - new_oh)
            new_box = numpy.array([px, py, px + new_ow, py + new_oh], dtype=numpy.float32)
            if len(boxes) and (bbox_ioa(new_box, boxes) > ioa_thresh).any():
                continue

            interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
            image[py:py + new_oh, px:px + new_ow] = cv2.resize(
                src_img[y1:y2, x1:x2], (new_ow, new_oh), interpolation=interp)

            boxes = numpy.vstack((boxes, new_box[None]))   # 之後貼的物件也不能蓋住這個
            pasted.append([cls_id, *new_box.tolist()])
            break

    if pasted:
        label = numpy.concatenate(
            (label.reshape(-1, 5), numpy.array(pasted, dtype=numpy.float32)), 0)
    return image, label.astype(numpy.float32).reshape(-1, 5)


# ─────────────────────────────────────────────────────────────
# 幾何 / 色彩增強
# ─────────────────────────────────────────────────────────────

def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h
    return y


def xy2wh(x, w, h):
    x = numpy.copy(x)
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)
    y = numpy.empty_like(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y


def resample():
    choices = (cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR,
               cv2.INTER_NEAREST, cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image, params):
    """in-place；image 必須可寫入（RAM 快取的原圖是唯讀的，這裡只會收到增強後的新陣列）"""
    r = numpy.random.uniform(-1, 1, 3) * [params['hsv_h'], params['hsv_s'], params['hsv_v']] + 1
    hh, ss, vv = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')
    hsv = cv2.merge((cv2.LUT(hh, lut_h), cv2.LUT(ss, lut_s), cv2.LUT(vv, lut_v)))
    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)


def resize(image, input_size, augment):
    shape = image.shape[:2]
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:
        r = min(r, 1.0)
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2
    if shape[::-1] != pad:
        image = cv2.resize(image, dsize=pad,
                           interpolation=resample() if augment else
                           (cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA))
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)   # 永遠回傳新陣列
    return image, (r, r), (w, h)


def candidates(box1, box2):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)


def random_perspective(image, label, params, border=(0, 0)):
    h = image.shape[0] + border[0] * 2
    w = image.shape[1] + border[1] * 2

    center = numpy.eye(3)
    center[0, 2] = -image.shape[1] / 2
    center[1, 2] = -image.shape[0] / 2

    rotate = numpy.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

    translate = numpy.eye(3)
    translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
    translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h

    matrix = translate @ shear @ rotate @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():
        image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    n = len(label)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = (xy @ matrix.T)[:, :2].reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        box = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        box[:, [0, 2]] = box[:, [0, 2]].clip(0, w)
        box[:, [1, 3]] = box[:, [1, 3]].clip(0, h)
        indices = candidates(box1=label[:, 1:5].T * s, box2=box.T)
        label = label[indices]
        label[:, 1:5] = box[indices]

    return image, label


def mix_up(image1, label1, image2, label2):
    alpha = numpy.random.beta(a=32.0, b=32.0)
    image = cv2.addWeighted(image1, alpha, image2, 1.0 - alpha, 0.0)   # [SPEED] 不產生 float64 中間陣列
    return image, numpy.concatenate((label1, label2), 0)


# ─────────────────────────────────────────────────────────────
# Albumentations（相容 1.x / 2.x）
# ─────────────────────────────────────────────────────────────

class Albumentations:
    def __init__(self):
        self.transform = None
        self._warned   = False
        try:
            import albumentations as A
            major = int(A.__version__.split('.')[0])
            if major >= 2:
                noise   = A.GaussNoise(std_range=(0.0124, 0.0277), p=0.1)   # ≈ var 10~50 (uint8)
                dropout = A.CoarseDropout(num_holes_range=(2, 8),
                                          hole_height_range=(8, 32),
                                          hole_width_range=(8, 32),
                                          fill=0, p=0.2)
            else:
                noise   = A.GaussNoise(var_limit=(10.0, 50.0), p=0.1)
                dropout = A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                                          min_holes=2, min_height=8, min_width=8,
                                          fill_value=0, p=0.2)
            self.transform = A.Compose(
                [
                    A.Blur(p=0.05),
                    A.MedianBlur(p=0.05),
                    A.MotionBlur(p=0.05),
                    noise,
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.1),
                    A.ToGray(p=0.05),
                    dropout,
                ],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
                                         min_area=16, min_visibility=0.2),
            )
        except ImportError:
            print("[Warning] albumentations is not installed. Advanced data augmentation will be skipped.")

    def __call__(self, image, box, cls):
        if self.transform is None or len(box) == 0:
            return image, box, cls
        try:
            x = self.transform(image=image, bboxes=box, class_labels=cls.reshape(-1))
            image = x['image']
            box = numpy.array(x['bboxes'], dtype=numpy.float32).reshape(-1, 4)
            cls = numpy.array(x['class_labels'], dtype=numpy.float32).reshape(-1, 1)
        except Exception as e:
            if not self._warned:
                print(f'\n[Warning] albumentations 失敗，此樣本略過進階增強: {e!r}（每個 worker 只提示一次）')
                self._warned = True
        return image, box, cls


# ─────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────

def _worker_init(_):
    # 每個 worker 只用單執行緒的 OpenCV，避免 workers × cv2 threads 搶 CPU
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def make_loader(dataset, batch_size, workers, shuffle,
                persistent_workers=True, prefetch_factor=4):
    """用既有的 dataset 建 DataLoader（close mosaic 時可重用 RAM 快取，不必重讀圖片）"""
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                  pin_memory=True, collate_fn=Dataset.collate_fn)
    if workers > 0:
        kwargs.update(persistent_workers=persistent_workers,
                      prefetch_factor=prefetch_factor,
                      worker_init_fn=_worker_init)
    return DataLoader(dataset, **kwargs)


def create_dataloader(
    img_folder,
    input_size=640,
    batch_size=8,
    workers=8,
    augment=True,
    shuffle=True,
    mosaic=False,
    hyp_params=None,
    cache=None,
    persistent_workers=True,
    prefetch_factor=4,
):
    filenames = sorted(
        os.path.join(img_folder, f) for f in os.listdir(img_folder)
        if f.rsplit('.', 1)[-1].lower() in FORMATS
    )
    if not filenames:
        print(f"[Warning] No images found in {img_folder}!")
        return None

    dataset = Dataset(filenames=filenames, input_size=input_size, params=hyp_params,
                      augment=augment, mosaic=mosaic, cache=cache)
    return make_loader(dataset, batch_size, workers, shuffle,
                       persistent_workers=persistent_workers,
                       prefetch_factor=prefetch_factor)