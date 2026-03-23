import math
import os
import random
import glob

import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment, mosaic):
        self.params = params
        self.mosaic = mosaic
        self.augment = augment
        self.input_size = input_size

        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())
        self.n = len(self.filenames)
        self.indices = range(self.n)

        self.albumentations = Albumentations()

    def __getitem__(self, index):
        index = self.indices[index]

        if self.mosaic and random.random() < self.params['mosaic']:
            image, label = self.load_mosaic(index, self.params)

            # MixUp
            if random.random() < self.params['mix_up']:
                index2 = random.choice(self.indices)
                image2, label2 = self.load_mosaic(index2, self.params)
                image, label = mix_up(image, label, image2, label2)

            # ── [NEW] Copy-Paste ──────────────────────────────
            if random.random() < self.params.get('copy_paste', 0.0):
                image, label = copy_paste(image, label, self.labels, self.filenames,
                                          self.input_size)
        else:
            image, shape = self.load_image(index)
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
        box = label[:, 1:5]
        box = xy2wh(box, w, h)

        if self.augment:
            image, box, cls = self.albumentations(image, box, cls)
            cls = numpy.array(cls, dtype=numpy.float32).reshape(-1, 1)  # 永遠是 (N, 1)
            box = numpy.array(box, dtype=numpy.float32).reshape(-1, 4)  # 永遠是 (N, 4)
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
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)

        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)

        return torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl)

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.filenames[i])
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(image,
                               dsize=(int(w * r), int(h * r)),
                               interpolation=resample() if self.augment else cv2.INTER_LINEAR)
        return image, (h, w)

    def load_mosaic(self, index, params):
        label4 = []
        border = [-self.input_size // 2, -self.input_size // 2]
        image4 = numpy.full((self.input_size * 2, self.input_size * 2, 3), 0, dtype=numpy.uint8)
        y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = (None,) * 8

        xc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))

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
            if i == 1:
                x1a = xc;                      y1a = max(yc - shape[0], 0)
                x2a = min(xc + shape[1], self.input_size * 2); y2a = yc
                x1b = 0;                       y1b = shape[0] - (y2a - y1a)
                x2b = min(shape[1], x2a - x1a); y2b = shape[0]
            if i == 2:
                x1a = max(xc - shape[1], 0);  y1a = yc
                x2a = xc;                      y2a = min(self.input_size * 2, yc + shape[0])
                x1b = shape[1] - (x2a - x1a); y1b = 0
                x2b = shape[1];                y2b = min(y2a - y1a, shape[0])
            if i == 3:
                x1a = xc;                      y1a = yc
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = 0;                       y1b = 0
                x2b = min(shape[1], x2a - x1a); y2b = min(y2a - y1a, shape[0])

            pad_w = x1a - x1b
            pad_h = y1a - y1b
            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            label = self.labels[index].copy()
            if len(label):
                label[:, 1:] = wh2xy(label[:, 1:], shape[1], shape[0], pad_w, pad_h)
            label4.append(label)

        label4 = numpy.concatenate(label4, 0)
        for x in label4[:, 1:]:
            numpy.clip(x, 0, 2 * self.input_size, out=x)

        image4, label4 = random_perspective(image4, label4, params, border)
        return image4, label4

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = zip(*batch)
        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)
        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)
        targets = {'cls': cls, 'box': box, 'idx': indices}
        return torch.stack(samples, dim=0), targets

    @staticmethod
    def load_label(filenames):
        path = f'{os.path.dirname(filenames[0])}.cache'
        if os.path.exists(path):
            os.remove(path)
            print(f"\n[提示] 已自動刪除舊的快取檔案: {path}")

        x = {}
        for filename in filenames:
            try:
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()
                shape = image.size
                assert (shape[0] > 9) & (shape[1] > 9)
                assert image.format.lower() in FORMATS

                a = f'{os.sep}images{os.sep}'
                b = f'{os.sep}labels{os.sep}'
                lbl_path = b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'
                if os.path.isfile(lbl_path):
                    with open(lbl_path) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        assert (label >= 0).all()
                        assert label.shape[1] == 5
                        assert (label[:, 1:] <= 1).all()
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:
                            label = label[i]
                    else:
                        label = numpy.zeros((0, 5), dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 5), dtype=numpy.float32)
            except FileNotFoundError:
                label = numpy.zeros((0, 5), dtype=numpy.float32)
            except AssertionError:
                continue
            x[filename] = label

        torch.save(x, path)
        return x


# ─────────────────────────────────────────────────────────────
# [NEW] Copy-Paste 增強
# 從其他圖片隨機挑幾個物件，直接貼到當前圖上
# 對小物件和密集場景效果顯著
# ─────────────────────────────────────────────────────────────

def copy_paste(image, label, all_labels, all_filenames, input_size, n_paste=3):
    """
    隨機從資料集中挑 n_paste 張圖，把其中的物件裁出來貼到 image 上。

    label 格式與 __getitem__ 一致：(N, 5)  [cls, x1, y1, x2, y2]  pixel xyxy
    all_labels 裡存的是 YOLO 格式 (cx cy w h 正規化)，需要先轉換。
    """
    h, w = image.shape[:2]
    # 確保 new_label 是 list of list，shape 統一為 (*, 5)
    new_label = label.tolist() if len(label) else []

    for _ in range(n_paste):
        src_idx = random.randrange(len(all_filenames))
        src_lbs = all_labels[src_idx]          # YOLO 格式 (M, 5)
        if len(src_lbs) == 0:
            continue

        src_img = cv2.imread(all_filenames[src_idx])
        if src_img is None:
            continue
        sh, sw = src_img.shape[:2]

        # 隨機選一個物件
        lb = src_lbs[random.randrange(len(src_lbs))]  # [cls, cx, cy, bw, bh]
        cls_id        = float(lb[0])
        cx, cy, bw, bh = float(lb[1]), float(lb[2]), float(lb[3]), float(lb[4])

        # YOLO 正規化 → pixel xyxy
        x1 = int((cx - bw / 2) * sw)
        y1 = int((cy - bh / 2) * sh)
        x2 = int((cx + bw / 2) * sw)
        y2 = int((cy + bh / 2) * sh)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(sw, x2), min(sh, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        obj = src_img[y1:y2, x1:x2]
        obj_h, obj_w = obj.shape[:2]
        if obj_h < 4 or obj_w < 4:
            continue

        # 隨機縮放
        scale  = random.uniform(0.3, 1.5)
        new_ow = max(4, int(obj_w * scale))
        new_oh = max(4, int(obj_h * scale))
        if new_ow > w or new_oh > h:
            continue
        obj = cv2.resize(obj, (new_ow, new_oh), interpolation=cv2.INTER_LINEAR)

        # 隨機貼上位置
        px = random.randint(0, w - new_ow)
        py = random.randint(0, h - new_oh)
        image[py:py + new_oh, px:px + new_ow] = obj

        # ── 新標籤與原始 label 格式相同：[cls, x1, y1, x2, y2] pixel xyxy ──
        new_label.append([cls_id,
                          float(px),          float(py),
                          float(px + new_ow), float(py + new_oh)])

    # 統一轉成 (N, 5) float32，保證後續 label[:, 0:1] / label[:, 1:5] 正確
    if new_label:
        label = numpy.array(new_label, dtype=numpy.float32).reshape(-1, 5)
    else:
        label = numpy.zeros((0, 5), dtype=numpy.float32)

    return image, label


# ─────────────────────────────────────────────────────────────
# 原有函數（保持不變）
# ─────────────────────────────────────────────────────────────

def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h
    return y


def xy2wh(x, w, h):
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)
    y = numpy.copy(x)
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
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']
    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
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
                           interpolation=resample() if augment else cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
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

    perspective = numpy.eye(3)

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

    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():
        image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    n = len(label)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ matrix.T
        xy = xy[:, :2].reshape(n, 8)
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
    image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label


class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            transforms = [
                A.Blur(p=0.05),
                A.MedianBlur(p=0.05),
                A.MotionBlur(p=0.05),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.1),
                A.ToGray(p=0.05),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                                min_holes=2, min_height=8, min_width=8,
                                fill_value=0, p=0.2),
            ]
            self.transform = A.Compose(
                transforms,
                bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels'],
                    min_area=16,
                    min_visibility=0.2,
                )
            )
        except ImportError:
            print("警告: 尚未安裝 albumentations，將跳過進階數據增強。")

    def __call__(self, image, box, cls):
        if self.transform and len(box) > 0:
            try:
                x = self.transform(image=image, bboxes=box, class_labels=cls)
                image = x['image']
                box = numpy.array(x['bboxes']) if len(x['bboxes']) > 0 else numpy.zeros((0, 4))
                cls = numpy.array(x['class_labels']) if len(x['class_labels']) > 0 else numpy.zeros((0, 1))
            except Exception:
                pass
        return image, box, cls


def create_dataloader(img_folder, input_size=640, batch_size=8,
                      augment=True, shuffle=True, mosaic=False, hyp_params=None):
    filenames = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        filenames.extend(glob.glob(os.path.join(img_folder, ext)))

    if not filenames:
        print(f"警告：在 {img_folder} 找不到任何圖片！")
        return None

    # print(f"成功找到 {len(filenames)} 張圖片。")

    custom_dataset = Dataset(
        filenames  = filenames,
        input_size = input_size,
        params     = hyp_params,
        augment    = augment,
        mosaic=mosaic
    )

    dataloader = DataLoader(
        custom_dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = Dataset.collate_fn,
    )

    return dataloader