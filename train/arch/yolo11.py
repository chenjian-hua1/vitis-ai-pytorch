# reference : https://github.com/jahongir7174/YOLOv11-pt/blob/master/nets/nn.py

import math

import torch

from utils.util import make_anchors


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.ReLU = activation

    def forward(self, x):
        return self.ReLU(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.ReLU(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), torch.nn.ReLU(), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, torch.nn.ReLU(), k=3, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, torch.nn.ReLU())
        self.conv2 = Conv(in_ch, out_ch // 2, torch.nn.ReLU())
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, torch.nn.ReLU())
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), torch.nn.ReLU())
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, torch.nn.ReLU())

        # 🛑 關鍵：在 __init__ 先算好切片的通道數，不要在 forward 裡面算
        self.half_ch = out_ch // r

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(CSPModule(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        # 1. 經過第一層卷積
        out = self.conv1(x)
        # c = out.shape[1] // 2
        
        # 2. 🛑 巧妙的邏輯：out 本身就已經是 [y1, y2] 拼接好的狀態了！
        # 我們直接把 out 當作拼接的基底 (y_cat)
        y_cat = out 
        
        # 只有後半段 (y2) 會進入後續的 res_m 模組
        last_feat = out[:, self.half_ch:, :, :]
        
        # 3. 🛑 滾動式拼接 (Rolling Concat)：徹底捨棄所有的 List 和 Tuple
        for m in self.res_m:
            last_feat = m(last_feat)
            # 每次產生新的特徵圖，就直接 concat 到 y_cat 的後面
            y_cat = torch.cat((y_cat, last_feat), dim=1)
            
        # 4. 迴圈結束後，y_cat 已經是一個包含所有特徵的純 Tensor！直接輸出
        return self.conv2(y_cat)


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, torch.nn.ReLU())
        self.conv2 = Conv(in_ch * 2, out_ch, torch.nn.ReLU())
        self.res_m = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        # 🛑 關鍵：先把 y3 算出來！
        y3 = self.res_m(y2)
        return self.conv2(torch.cat(tensors=(x, y1, y2, y3), dim=1))

class Attention(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5
        self.layer_param0 = self.dim_key * 2 + self.dim_head
        self.layer_param1 = self.dim_key * 2

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, torch.nn.Identity())

        self.conv1 = Conv(ch, ch, torch.nn.Identity(), k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, torch.nn.Identity())

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)

        # ❌ h * w 是 Python int，某些版本會產生 int64 中間值
        # qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)
        # ✅ 改用 -1 讓 PyTorch 自動推斷
        qkv = qkv.view(b, self.num_head, self.layer_param, -1)


        # ❌ 原本：split 產生 nndct_stack
        # q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)
        # ✅ 改成 chunk 或 narrow，量化器更好追蹤
        q = qkv[:, :, :self.dim_key, :]
        k = qkv[:, :, self.dim_key:self.layer_param1, :]
        v = qkv[:, :, self.layer_param1:, :]

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        # ❌ 原本
        # x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        # ✅ reshape 也用 -1
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, -1).reshape(b, c, h, w))

        return self.conv2(x)


class PSABlock(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = torch.nn.Sequential(Conv(ch, ch * 2, torch.nn.ReLU()),
                                         Conv(ch * 2, ch, torch.nn.Identity()))

    def forward(self, x):
        # x = x + self.conv1(x)
        return x + self.conv2(x)


class PSA(torch.nn.Module):
    def __init__(self, ch, n):
        super().__init__()
        self.conv1 = Conv(ch, 2 * (ch // 2), torch.nn.ReLU())
        self.conv2 = Conv(2 * (ch // 2), ch, torch.nn.ReLU())
        # 🛑 關鍵：在 __init__ 算好通道數
        self.half_ch = ch // 2
        self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

    def forward(self, x):
        # 原始寫法：
        # x, y = self.conv1(x).chunk(2, 1)
        # 修改為切片寫法：
        out = self.conv1(x)
        # c = out.shape[1] // 2
        x, y = out[:, :self.half_ch, :, :], out[:, self.half_ch:, :, :]
        
        return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.ReLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], torch.nn.ReLU(), k=3, s=2, p=1))
        self.p2.append(CSP(width[2], width[3], depth[0], csp[0], r=4))
        # p3/8
        self.p3.append(Conv(width[3], width[3], torch.nn.ReLU(), k=3, s=2, p=1))
        self.p3.append(CSP(width[3], width[4], depth[1], csp[0], r=4))
        # p4/16
        self.p4.append(Conv(width[4], width[4], torch.nn.ReLU(), k=3, s=2, p=1))
        self.p4.append(CSP(width[4], width[4], depth[2], csp[1], r=2))
        # p5/32
        self.p5.append(Conv(width[4], width[5], torch.nn.ReLU(), k=3, s=2, p=1))
        self.p5.append(CSP(width[5], width[5], depth[3], csp[1], r=2))
        self.p5.append(SPP(width[5], width[5]))
        self.p5.append(PSA(width[5], depth[4]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp[0], r=2)
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp[0], r=2)
        self.h3 = Conv(width[3], width[3], torch.nn.ReLU(), k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp[0], r=2)
        self.h5 = Conv(width[4], width[4], torch.nn.ReLU(), k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp[1], r=2)
    
    def forward(self, p3, p4, p5):
        # 🛑 關鍵：把每一個運算都獨立成一行，清清楚楚，不要有任何巢狀函數
        
        up_p5 = self.up(p5)
        cat_p4 = torch.cat((up_p5, p4), dim=1)
        p4 = self.h1(cat_p4)

        up_p4 = self.up(p4)
        cat_p3 = torch.cat((up_p4, p3), dim=1)
        p3 = self.h2(cat_p3)

        h3_p3 = self.h3(p3)
        cat_p4_2 = torch.cat((h3_p3, p4), dim=1)
        p4 = self.h4(cat_p4_2)

        h5_p4 = self.h5(p4)
        cat_p5 = torch.cat((h5_p4, p5), dim=1)
        p5 = self.h6(cat_p5)

        return p3, p4, p5


class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)

        return self.conv(x.softmax(1)).view(b, 4, a)


class YOLOPostProcessor(torch.nn.Module):
    """
    [call function] Decode raw YOLO multi-scale feature maps into box + class predictions.

    x           : list of feature maps, each (B, no, H, W)  float32
    conf_thresh : confidence threshold (unused here, applied in NMS)

    Returns
    -------
    output : (B, 4+nc, Anchors)
                axis-1 layout: [cx, cy, w, h, cls_score_0, ..., cls_score_{nc-1}]
    """
    def __init__(self, nc=80, ch=16, strides=[8, 16, 32]):
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
        super().__init__()
        self.nc = nc
        self.ch = ch
        self.no = nc + ch * 4
        self.stride = torch.tensor(strides) # YOLOv11 預設的降採樣倍率
        self.dfl = DFL(ch) # 實例化 DFL 模組

    def forward(self, x):
        """
        Decode raw YOLO multi-scale feature maps into box + class predictions.

        x           : list of feature maps, each (B, no, H, W)  float32
        conf_thresh : confidence threshold (unused here, applied in NMS)

        Returns
        -------
        output : (B, 4+nc, Anchors)
                 axis-1 layout: [cx, cy, w, h, cls_score_0, ..., cls_score_{nc-1}]
        """

        # ── 修正：明確指定 dtype=float32，避免 int64 ──────────
        self.anchors, self.strides = (
            i.transpose(0, 1) for i in make_anchors(x, self.stride.float())
        )

        # ✅ 新增這行：將來自模型的 FP16 特徵圖強制轉回 FP32
        x = [i.float() for i in x]

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        # ── 修正：用 narrow 取代 chunk，量化器追蹤更穩定 ──────
        dfl_out = self.dfl(box)
        a = dfl_out.narrow(1, 0, 2)          # 取前兩個 channel
        b = dfl_out.narrow(1, 2, 2)          # 取後兩個 channel

        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

        
    

class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        # self.dfl = DFL(self.ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box,torch.nn.ReLU(), k=3, p=1),
                                                           Conv(box, box,torch.nn.ReLU(), k=3, p=1),
                                                           torch.nn.Conv2d(box, out_channels=4 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, torch.nn.ReLU(), k=3, p=1, g=x),
                                                           Conv(x, cls, torch.nn.ReLU()),
                                                           Conv(cls, cls, torch.nn.ReLU(), k=3, p=1, g=cls),
                                                           Conv(cls, cls, torch.nn.ReLU()),
                                                           torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)
    
    # 🛑 關鍵：參數改為接收 p3, p4, p5，而不是單一的 x
    def forward(self, p3, p4, p5):
        # 第一層 (P3)
        box0 = self.box[0](p3)
        cls0 = self.cls[0](p3)
        out0 = torch.cat((box0, cls0), dim=1)
        
        # 第二層 (P4)
        box1 = self.box[1](p4)
        cls1 = self.cls[1](p4)
        out1 = torch.cat((box1, cls1), dim=1)
        
        # 第三層 (P5)
        box2 = self.box[2](p5)
        cls2 = self.cls[2](p5)
        out2 = torch.cat((box2, cls2), dim=1)
        
        return out0, out1, out2

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        p3, p4, p5 = self.net(x)  # 從 Backbone 取得 3 個獨立的 Tensor
        
        # 🛑 關鍵：不要傳 tuple(x)，而是把 3 個 Tensor 獨立傳進去！
        p3, p4, p5 = self.fpn(p3, p4, p5) 
        
        # 🛑 關鍵：Head 也接收 3 個獨立的 Tensor
        return self.head(p3, p4, p5)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v11_n(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_t(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 24, 48, 96, 192, 384]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_s(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_m(num_classes: int = 80):
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_l(num_classes: int = 80):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_x(num_classes: int = 80):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return YOLO(width, depth, csp, num_classes)
