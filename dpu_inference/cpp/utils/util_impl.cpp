#include "util.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <numeric>
#include <stdexcept>

// ============================================================================
//  編譯器提示巨集（SIMD / 向量化友善）
// ============================================================================
#if defined(__GNUC__) || defined(__clang__)
#  define RESTRICT __restrict__
#else
#  define RESTRICT
#endif

// ============================================================================
//  ImageNet normalisation constants (mirrors Python globals)
// ============================================================================
namespace {

constexpr float kMeanR = 0.485f, kMeanG = 0.456f, kMeanB = 0.406f;
constexpr float kStdR  = 0.229f, kStdG  = 0.224f, kStdB  = 0.225f;

// uint8 path: dst = src * scale + bias
//   scale = 1/(std*255)
//   bias  = -(mean/std)
constexpr float kU8ScaleR = 1.f/(kStdR*255.f);
constexpr float kU8ScaleG = 1.f/(kStdG*255.f);
constexpr float kU8ScaleB = 1.f/(kStdB*255.f);
constexpr float kU8BiasR  = -kMeanR/kStdR;
constexpr float kU8BiasG  = -kMeanG/kStdG;
constexpr float kU8BiasB  = -kMeanB/kStdB;

// float path: dst = src * scale + bias
//   scale = 1/std
//   bias  = -(mean/std)
constexpr float kF32ScaleR = 1.f/kStdR;
constexpr float kF32ScaleG = 1.f/kStdG;
constexpr float kF32ScaleB = 1.f/kStdB;
// bias 與 uint8 path 相同，共用 kU8Bias*

} // anonymous namespace

// ============================================================================
//  Fix / Float Conversion
// ============================================================================

#include <cmath>

void fix2float(const cv::Mat& data, int fix_point, cv::Mat& out)
{
    // std::exp2f(-fix_point) 直接算出 2^(-fix_point)
    // 既安全，編譯器也能在編譯期高度最佳化
    float scale = std::exp2f(-static_cast<float>(fix_point));
    data.convertTo(out, CV_32FC3, scale);
}

void float2fix(const cv::Mat& data, int fix_point, cv::Mat& out)
{
    // std::exp2f(fix_point) 直接算出 2^(fix_point)
    float scale = std::exp2f(static_cast<float>(fix_point));
    data.convertTo(out, CV_8SC3, scale);
}

// ============================================================================
//  Bounding Box Utilities
// ============================================================================

cv::Mat wh2xy(const cv::Mat& x)
{
    // x : (N, 4+)  columns: cx, cy, w, h, ...
    assert(x.cols >= 4 && x.type() == CV_32F);
    cv::Mat y = x.clone();

    const int N    = x.rows;
    const int cols = x.cols;
    const float* RESTRICT src = x.ptr<float>();
    float*       RESTRICT dst = y.ptr<float>();

    #pragma omp simd
    for (int i = 0; i < N; ++i) {
        const float cx = src[i * cols + 0];
        const float cy = src[i * cols + 1];
        const float w  = src[i * cols + 2];
        const float h  = src[i * cols + 3];
        dst[i * cols + 0] = cx - w * 0.5f;   // x1
        dst[i * cols + 1] = cy - h * 0.5f;   // y1
        dst[i * cols + 2] = cx + w * 0.5f;   // x2
        dst[i * cols + 3] = cy + h * 0.5f;   // y2
    }
    return y;
}

cv::Mat xyxy2xywh(const cv::Mat& box)
{
    // box : (N, 4)  x1, y1, x2, y2
    assert(box.cols == 4 && box.type() == CV_32F);
    cv::Mat out = box.clone();

    const int N = box.rows;
    const float* RESTRICT src = box.ptr<float>();
    float*       RESTRICT dst = out.ptr<float>();

    #pragma omp simd
    for (int i = 0; i < N; ++i) {
        dst[i * 4 + 2] = src[i * 4 + 2] - src[i * 4 + 0]; // w
        dst[i * 4 + 3] = src[i * 4 + 3] - src[i * 4 + 1]; // h
    }
    return out;
}

// ============================================================================
//  Anchor Generation
// ============================================================================

AnchorResult make_anchors(const std::vector<cv::Mat>& feature_maps,
                            const std::vector<int>&     strides,
                            float                       offset)
{
    // feature_maps[i] is expected to be a 4-dim Mat with size {B, C, H, W}
    // We read H = dims[2], W = dims[3].

    std::vector<cv::Mat> anchor_list, stride_list;

    for (size_t i = 0; i < strides.size(); ++i) {
        const cv::Mat& fm = feature_maps[i];
        int H = fm.size[2];
        int W = fm.size[3];
        int stride = strides[i];

        cv::Mat anchors(H * W, 2, CV_32F);
        cv::Mat strides_mat(H * W, 1, CV_32F, cv::Scalar(static_cast<float>(stride)));

        int idx = 0;
        for (int gy = 0; gy < H; ++gy) {
            for (int gx = 0; gx < W; ++gx, ++idx) {
                anchors.at<float>(idx, 0) = static_cast<float>(gx) + offset; // x
                anchors.at<float>(idx, 1) = static_cast<float>(gy) + offset; // y
            }
        }
        anchor_list.push_back(anchors);
        stride_list.push_back(strides_mat);
    }

    AnchorResult res;
    cv::vconcat(anchor_list, res.anchors);       // (A, 2)
    cv::vconcat(stride_list, res.stride_tensor); // (A, 1)
    return res;
}

// ============================================================================
//  Pre-Processing
// ============================================================================

void norm(const cv::Mat& x, cv::Mat& out)
{
    CV_Assert(x.type() == CV_8UC3);

    out.create(x.rows, x.cols, CV_32FC3);

    const uchar* RESTRICT src = x.ptr<uchar>();
    float*       RESTRICT dst = out.ptr<float>();

    const int total = x.rows * x.cols;

    // interleaved BGR(x3)，stride 3 的 layout 編譯器通常需要 pragma 提示
    // 才會生成 NEON ld3/st3 指令。
    #pragma omp simd
    for (int i = 0; i < total; i++) {
        dst[3*i + 0] = src[3*i + 0] * kU8ScaleR + kU8BiasR;
        dst[3*i + 1] = src[3*i + 1] * kU8ScaleG + kU8BiasG;
        dst[3*i + 2] = src[3*i + 2] * kU8ScaleB + kU8BiasB;
    }
}

void resize(const cv::Mat& img, int input_size, ResizeResult& res)
{
    const int orig_h = img.rows, orig_w = img.cols;

    float r = std::min(static_cast<float>(input_size) / orig_h,
                    static_cast<float>(input_size) / orig_w);
    r = std::min(r, 1.0f);

    int pad_w = static_cast<int>(std::round(orig_w * r));
    int pad_h = static_cast<int>(std::round(orig_h * r));

    int dw = (input_size - pad_w) / 2.0f;
    int dh = (input_size - pad_h) / 2.0f;

    int top    = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left   = static_cast<int>(std::round(dw - 0.1f));
    int right  = static_cast<int>(std::round(dw + 0.1f));

    res.ratio = {r,  r};
    res.pad   = {dw, dh};

    // 🔥 一次配置
    res.img.create(input_size, input_size, img.type());
    res.img.setTo(cv::Scalar(0,0,0));  // padding

    // 🔥 ROI 直接寫入
    cv::Mat roi = res.img(cv::Rect(left, top, pad_w, pad_h));

    cv::resize(img, roi, roi.size(), 0, 0, cv::INTER_LINEAR);
}


// ============================================================================
//  YOLOPostProcessor
// ============================================================================

// ═══════════════ ctor ═══════════════

YOLOPostProcessor::YOLOPostProcessor(int batch,
                                      int input_h,
                                      int input_w,
                                      int nc,
                                      int ch,
                                      std::vector<int> strides,
                                      int max_nms,
                                      int max_det)
    : B_(batch),
      input_h_(input_h),
      input_w_(input_w),
      nc_(nc),
      ch_(ch),
      no_(nc + ch * 4),
      strides_(std::move(strides)),
      max_nms_(max_nms),
      max_det_(max_det)
{
    precompute_anchors();    // anchors_T_ / stride_T_ / scale_offsets_
    allocate_buffers();      // x_cat_ / dfl_out_ / output_ / NMS buffer / detections_
    bind_views();            // box_raw_view_ / cls_raw_view_
    cache_pointers();        // ①② 所有 row pointer + anchors / stride 指標
    precompute_tables();     // ④ class_offsets_ ⑤ dfl_arange_
}


// ═══════════════ 初始化 ═══════════════

void YOLOPostProcessor::precompute_anchors()
{
    constexpr float offset = 0.5f;

    hw_per_scale_.clear();
    scale_offsets_.clear();
    std::vector<std::pair<int,int>> hw_list;
    A_ = 0;
    int acc_offset = 0;
    for (int stride : strides_) {
        if (input_h_ % stride != 0 || input_w_ % stride != 0)
            throw std::runtime_error("input size not divisible by stride");
        int H = input_h_ / stride;
        int W = input_w_ / stride;
        hw_list.emplace_back(H, W);
        hw_per_scale_.push_back(H * W);
        scale_offsets_.push_back(acc_offset);  // ③
        acc_offset += H * W;
        A_ += H * W;
    }

    anchors_T_ = cv::Mat(2, A_, CV_32F);
    stride_T_  = cv::Mat(1, A_, CV_32F);

    int col = 0;
    for (size_t i = 0; i < strides_.size(); ++i) {
        int H = hw_list[i].first;
        int W = hw_list[i].second;
        float s = static_cast<float>(strides_[i]);
        for (int gy = 0; gy < H; ++gy)
            for (int gx = 0; gx < W; ++gx, ++col) {
                anchors_T_.at<float>(0, col) = static_cast<float>(gx) + offset;
                anchors_T_.at<float>(1, col) = static_cast<float>(gy) + offset;
                stride_T_.at<float>(0, col)  = s;
            }
    }
}


void YOLOPostProcessor::allocate_buffers()
{
    // cv::Mat 的 data pointer 預設對齊到 CV_MALLOC_ALIGN（通常 64 byte），
    // 對 NEON（16 byte）的對齊需求綽綽有餘。
    x_cat_   = cv::Mat(std::vector<int>{B_, no_,     A_}, CV_32F);
    dfl_out_ = cv::Mat(std::vector<int>{B_, 4,       A_}, CV_32F);
    output_  = cv::Mat(std::vector<int>{B_, 4 + nc_, A_}, CV_32F);

    candidates_.resize(max_nms_);
    boxes_cv_.resize(max_nms_);
    scores_cv_.resize(max_nms_);
    indices_.reserve(max_nms_);

    detections_.resize(B_);
    for (auto& db : detections_) {
        db.data.resize(max_det_);
        db.count = 0;
    }

    // ── active anchor 相關 buffer（上限為 A_，實際長度由 vector 自己管）──
    active_indices_.assign(B_, {});
    active_max_score_.assign(B_, {});
    active_max_cls_.assign(B_, {});
    active_cls_scores_.assign(B_, {});
    for (int b = 0; b < B_; ++b) {
        active_indices_[b].reserve(A_);
        active_max_score_[b].reserve(A_);
        active_max_cls_[b].reserve(A_);
        active_cls_scores_[b].reserve(A_ * nc_);  // 上界，實際用多少看 active 數
    }
}


void YOLOPostProcessor::bind_views()
{
    const int split = 4 * ch_;
    cv::Range box_r[] = { cv::Range::all(), cv::Range(0, split),   cv::Range::all() };
    cv::Range cls_r[] = { cv::Range::all(), cv::Range(split, no_), cv::Range::all() };
    box_raw_view_ = x_cat_(box_r);
    cls_raw_view_ = x_cat_(cls_r);
}


void YOLOPostProcessor::cache_pointers()
{
    // anchors / stride
    ax_ = anchors_T_.ptr<float>(0);
    ay_ = anchors_T_.ptr<float>(1);
    sv_ = stride_T_.ptr<float>(0);

    // ── ① output 的 row pointer ──
    output_cx_rows_.resize(B_);
    output_cy_rows_.resize(B_);
    output_w_rows_.resize(B_);
    output_h_rows_.resize(B_);
    output_cls_rows_.assign(B_, std::vector<float*>(nc_, nullptr));

    for (int b = 0; b < B_; ++b) {
        output_cx_rows_[b] = output_.ptr<float>(b, 0);
        output_cy_rows_[b] = output_.ptr<float>(b, 1);
        output_w_rows_[b]  = output_.ptr<float>(b, 2);
        output_h_rows_[b]  = output_.ptr<float>(b, 3);
        for (int c = 0; c < nc_; ++c)
            output_cls_rows_[b][c] = output_.ptr<float>(b, 4 + c);
    }

    // ── ② dfl_out 的 row pointer ──
    dfl_rows_.resize(B_);
    for (int b = 0; b < B_; ++b)
        for (int coord = 0; coord < 4; ++coord)
            dfl_rows_[b][coord] = dfl_out_.ptr<float>(b, coord);

    // ── ② box_raw / cls_raw 的 row pointer（view 的 ptr 指向 x_cat_ 的記憶體）──
    const int split = 4 * ch_;
    box_raw_rows_.assign(B_, std::vector<const float*>(split, nullptr));
    cls_raw_rows_.assign(B_, std::vector<const float*>(nc_,   nullptr));
    for (int b = 0; b < B_; ++b) {
        for (int c = 0; c < split; ++c)
            box_raw_rows_[b][c] = box_raw_view_.ptr<float>(b, c);
        for (int c = 0; c < nc_; ++c)
            cls_raw_rows_[b][c] = cls_raw_view_.ptr<float>(b, c);
    }

    // ── x_cat_ 的 row pointer（concat 寫入用）──
    xcat_rows_.assign(B_, std::vector<float*>(no_, nullptr));
    for (int b = 0; b < B_; ++b)
        for (int c = 0; c < no_; ++c)
            xcat_rows_[b][c] = x_cat_.ptr<float>(b, c);
}


void YOLOPostProcessor::precompute_tables()
{
    // ── ④ class_id × max_wh ──
    constexpr float max_wh = 7680.f;
    class_offsets_.resize(nc_);
    for (int c = 0; c < nc_; ++c)
        class_offsets_[c] = static_cast<float>(c) * max_wh;

    // ── ⑤ DFL arange [0, 1, ..., ch-1] ──
    dfl_arange_.resize(ch_);
    std::iota(dfl_arange_.begin(), dfl_arange_.end(), 0.f);
}




// ═══════════════ decode pipeline（全稀疏版本）═══════════════
//
//  設計目標：decode 時就跳過低分 anchor，不做任何 dense 計算。
//
//  Pipeline：
//    1. concat feature maps → x_cat_（仍是 dense，memcpy 本身不是瓶頸）
//    2. classify_and_build_mask(conf_thresh):
//         - 對每個 anchor 在 **logit 空間** 找 max 與 argmax（純比較，無 expf）
//         - max_logit > logit(conf_thresh) → active，放入 active_indices_
//         - active anchor 才算 sigmoid：先算 argmax 的分數（active_max_score_），
//           若 nc > 1 則再逐 class 算 sigmoid 放入 active_cls_scores_，
//           給 NMS 做 multi-label 判斷用
//    3. dfl_decode_masked(): 只處理 active anchor 的 DFL softmax
//    4. dist2bbox_masked(): 只處理 active anchor 的座標轉換
//
//  output_ 語意：decode 後只有 active anchor 的位置被寫入，其餘位置值未定義。
//  raw_output() 在此設計下不再有「dense tensor」的意義，若外部需要完整 tensor
//  請呼叫端自行初始化 output_ 為零或改用 detections()。
//
//  備註：YOLO v8/v11 分類分支是 multi-label sigmoid。比較門檻時利用
//        sigmoid 的單調性，在 logit 空間用 x > log(t/(1-t)) 代替，免去 expf。


void YOLOPostProcessor::classify_and_build_mask(float conf_thresh)
{
    const int A = A_;

    // sigmoid 嚴格單調：sigmoid(x) > t  ⇔  x > logit(t)
    // 預先計算 logit(t)，後續逐 anchor 的比較完全不碰 expf。
    const float logit_thresh = (conf_thresh > 0.f && conf_thresh < 1.f)
        ? std::log(conf_thresh / (1.f - conf_thresh))
        : -FLT_MAX;

    for (int b = 0; b < B_; ++b) {
        active_indices_[b].clear();
        active_max_score_[b].clear();
        active_max_cls_[b].clear();
        active_cls_scores_[b].clear();

        const std::vector<const float*>& cls_raw = cls_raw_rows_[b];

        // ── Step A: 逐 anchor 在 logit 空間找 max + argmax，建 active list。
        //
        // 這個 loop 是每 anchor O(nc) 次比較，沒有 expf、沒有 memory write。
        // 大約是 A * nc 次 load + compare — 遠比原本 dense sigmoid 的
        // A * nc 次 expf 便宜（expf 約 8-20 個 cycle，純比較 1 cycle 內）。
        //
        // 若 nc 很大且分佈偏極端（例如背景 logit << 0），這裡其實還可以做
        // "early break on first positive"，但會破壞 argmax 語意。保持完整
        // 掃描換取正確的 argmax。
        //
        // 註：這個 loop 不向量化（因為有 conditional push_back），但大部分
        //    anchor 會 fail 過濾條件而直接跳過後續工作，實際成本主要就是
        //    nc 次 load + max reduction。
        for (int a = 0; a < A; ++a) {
            // 找 max logit + argmax
            float max_l  = cls_raw[0][a];
            int   max_id = 0;
            for (int c = 1; c < nc_; ++c) {
                float v = cls_raw[c][a];
                if (v > max_l) { max_l = v; max_id = c; }
            }

            if (max_l <= logit_thresh) continue;   // 低分 anchor，完全略過

            active_indices_[b].push_back(a);
            // 只對 argmax 算 sigmoid（比算完整 nc 條便宜 nc 倍）
            float max_s = 1.f / (1.f + expf(-max_l));
            active_max_score_[b].push_back(max_s);
            active_max_cls_[b].push_back(max_id);
        }

        const int n_active = static_cast<int>(active_indices_[b].size());

        // ── Step B（僅 nc > 1）: 對 active anchor 計算完整的 nc 個 sigmoid，
        //    供 NMS multi-label 分支使用（同一 anchor 可能多個 class 過門檻）。
        //
        // 成本：n_active * nc 次 expf，遠小於原版 dense 的 A * nc。
        // layout: active_cls_scores_[b][i * nc + c] = sigmoid(cls_raw[c][act[i]])
        if (nc_ > 1 && n_active > 0) {
            auto& scores = active_cls_scores_[b];
            scores.resize(static_cast<size_t>(n_active) * nc_);

            const int* RESTRICT act = active_indices_[b].data();
            float* RESTRICT dst = scores.data();

            for (int i = 0; i < n_active; ++i) {
                const int a = act[i];
                float* RESTRICT row = dst + static_cast<size_t>(i) * nc_;
                for (int c = 0; c < nc_; ++c)
                    row[c] = 1.f / (1.f + expf(-cls_raw[c][a]));
            }
        }
    }
}


void YOLOPostProcessor::dfl_decode_masked()
{
    const int ch = ch_;

    for (int b = 0; b < B_; ++b) {
        const std::vector<int>& act = active_indices_[b];
        const int N = static_cast<int>(act.size());
        if (N == 0) continue;

        for (int coord = 0; coord < 4; ++coord) {
            float* RESTRICT out_row = dfl_rows_[b][coord];
            const int base = coord * ch;

            // 把 ch 條 row pointer 拉到 local。ch 典型 16，上限 32。
            const float* RESTRICT ch_rows[32];
            for (int c = 0; c < ch; ++c)
                ch_rows[c] = box_raw_rows_[b][base + c];

            // 只迭代 active anchor。存取是 gather（非連續），但 N << A
            // 時省下的 expf 遠大於失去的連續存取收益。
            for (int i = 0; i < N; ++i) {
                const int a = act[i];

                // ch 個 logit 的 numerically-stable softmax + 期望值
                float max_l = ch_rows[0][a];
                for (int c = 1; c < ch; ++c) {
                    float v = ch_rows[c][a];
                    if (v > max_l) max_l = v;
                }

                float sum_e = 0.f, wsum = 0.f;
                for (int c = 0; c < ch; ++c) {
                    float e = expf(ch_rows[c][a] - max_l);
                    sum_e += e;
                    wsum  += e * dfl_arange_[c];
                }
                out_row[a] = wsum / sum_e;
            }
        }
    }
}


void YOLOPostProcessor::dist2bbox_masked()
{
    for (int b = 0; b < B_; ++b) {
        const std::vector<int>& act = active_indices_[b];
        const int N = static_cast<int>(act.size());
        if (N == 0) continue;

        float* RESTRICT cx = output_cx_rows_[b];
        float* RESTRICT cy = output_cy_rows_[b];
        float* RESTRICT ww = output_w_rows_[b];
        float* RESTRICT hh = output_h_rows_[b];
        const float* RESTRICT dl = dfl_rows_[b][0];
        const float* RESTRICT dt = dfl_rows_[b][1];
        const float* RESTRICT dr = dfl_rows_[b][2];
        const float* RESTRICT db = dfl_rows_[b][3];
        const float* RESTRICT ax = ax_;
        const float* RESTRICT ay = ay_;
        const float* RESTRICT sv = sv_;

        for (int i = 0; i < N; ++i) {
            const int a = act[i];
            float lt_x = ax[a] - dl[a];
            float lt_y = ay[a] - dt[a];
            float rb_x = ax[a] + dr[a];
            float rb_y = ay[a] + db[a];
            float s    = sv[a];

            cx[a] = (lt_x + rb_x) * 0.5f * s;
            cy[a] = (lt_y + rb_y) * 0.5f * s;
            ww[a] = (rb_x - lt_x) * s;
            hh[a] = (rb_y - lt_y) * s;
        }
    }
}


const cv::Mat& YOLOPostProcessor::decode(const std::vector<cv::Mat>& x,
                                         float conf_thresh)
{
    if (x[0].size[0] != B_)
        throw std::runtime_error("feature map batch mismatch");

    const int B = B_;

    // ── 1. concat → x_cat_（dense memcpy）──
    for (size_t i = 0; i < x.size(); ++i) {
        const auto& xi = x[i];
        const int hw = xi.size[2] * xi.size[3];
        const int col_offset = scale_offsets_[i];

        for (int b = 0; b < B; ++b)
            for (int c = 0; c < no_; ++c) {
                const float* src = xi.ptr<float>(b, c);
                float*       dst = xcat_rows_[b][c] + col_offset;
                std::memcpy(dst, src, hw * sizeof(float));
            }
    }

    conf_thresh_cached_ = conf_thresh;

    // ── 2. classify + mask（logit-space 比較，僅 active anchor 算 sigmoid）──
    classify_and_build_mask(conf_thresh);

    // ── 3. DFL decode（僅 active anchor）──
    dfl_decode_masked();

    // ── 4. dist2bbox（僅 active anchor）──
    dist2bbox_masked();

    return output_;
}


// ═══════════════ NMS ═══════════════

void YOLOPostProcessor::nms_single_batch(int b,
                                          float conf_thresh,
                                          float iou_thresh)
{
    cand_count_ = 0;

    // decode 階段用的 conf_thresh 必須跟 NMS 一致；否則 active list 的語意
    // 錯配。這是新設計的硬性前提（不再 fallback slow path）。
    if (conf_thresh_cached_ != conf_thresh) {
        throw std::runtime_error(
            "nms conf_thresh must match the conf_thresh used in decode()");
    }

    const float* cx_row = output_cx_rows_[b];
    const float* cy_row = output_cy_rows_[b];
    const float* ww_row = output_w_rows_[b];
    const float* hh_row = output_h_rows_[b];

    const std::vector<int>&   act         = active_indices_[b];
    const std::vector<float>& max_score   = active_max_score_[b];
    const std::vector<int>&   max_cls     = active_max_cls_[b];
    const std::vector<float>& cls_scores  = active_cls_scores_[b];   // 僅 nc>1 時有效
    const int N = static_cast<int>(act.size());

    for (int i = 0; i < N; ++i) {
        const int a = act[i];

        // decode 時已用 max_logit > logit_thresh 濾過，max_score[i] > conf_thresh
        // 嚴格成立；不用再檢查。
        float cx = cx_row[a], cy = cy_row[a];
        float w  = ww_row[a], h  = hh_row[a];
        float x1 = cx - w * 0.5f, y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f, y2 = cy + h * 0.5f;

        if (nc_ > 1) {
            // multi-label：同一 anchor 可能有多個 class 過門檻。讀預算好的
            // active_cls_scores_[b] 的第 i 列（長度 nc）。
            const float* RESTRICT row = cls_scores.data()
                                      + static_cast<size_t>(i) * nc_;
            for (int c = 0; c < nc_; ++c) {
                float score = row[c];
                if (score > conf_thresh && cand_count_ < max_nms_) {
                    candidates_[cand_count_++] = {
                        x1, y1, x2, y2, score, static_cast<float>(c)
                    };
                }
            }
        } else {
            // nc == 1：max_score[i] 就是唯一的 class score。
            (void)max_cls;
            float score = max_score[i];
            if (cand_count_ < max_nms_) {
                candidates_[cand_count_++] = {x1, y1, x2, y2, score, 0.f};
            }
        }

        if (cand_count_ >= max_nms_) break;
    }

    if (cand_count_ == 0) {
        detections_[b].count = 0;
        return;
    }

    std::sort(candidates_.begin(),
              candidates_.begin() + cand_count_,
              [](const auto& a, const auto& b){ return a[4] > b[4]; });

    // ── 組 offset 框，用 class_offsets_ 查表 ──
    for (int i = 0; i < cand_count_; ++i) {
        const auto& c = candidates_[i];
        int   cls_id = static_cast<int>(c[5]);
        float off    = class_offsets_[cls_id];
        boxes_cv_[i]  = cv::Rect2d(c[0] + off, c[1] + off,
                                    c[2] - c[0], c[3] - c[1]);
        scores_cv_[i] = c[4];
    }

    boxes_cv_.resize(cand_count_);
    scores_cv_.resize(cand_count_);

    cv::dnn::NMSBoxes(boxes_cv_, scores_cv_,
                      conf_thresh, iou_thresh, indices_);

    boxes_cv_.resize(max_nms_);
    scores_cv_.resize(max_nms_);

    int keep = std::min(static_cast<int>(indices_.size()), max_det_);
    auto& db = detections_[b];
    for (int k = 0; k < keep; ++k) {
        const auto& row = candidates_[indices_[k]];
        db.data[k] = Detection{
            row[0], row[1], row[2], row[3],
            row[4], static_cast<int>(row[5])
        };
    }
    db.count = keep;
}


void YOLOPostProcessor::nms(float conf_thresh, float iou_thresh)
{
    for (int b = 0; b < B_; ++b)
        nms_single_batch(b, conf_thresh, iou_thresh);
}


const std::vector<DetectionBatch>& YOLOPostProcessor::process(
    const std::vector<cv::Mat>& feature_maps,
    float conf_thresh,
    float iou_thresh)
{
    // 把 conf_thresh 一併帶進 decode，啟用「先 classify → mask → 只 decode
    // active anchor」的 fast path。
    decode(feature_maps, conf_thresh);
    nms(conf_thresh, iou_thresh);
    return detections_;
}

// ============================================================================
//  Visualization
// ============================================================================

cv::Mat scale_boxes(const cv::Mat&        boxes,
                    std::pair<float,float> ratio,
                    std::pair<float,float> pad,
                    cv::Size               orig_shape)
{
    cv::Mat out;
    boxes.convertTo(out, CV_32F);

    float w_max = static_cast<float>(orig_shape.width);
    float h_max = static_cast<float>(orig_shape.height);

    // x1
    {
        cv::Mat col = (out.col(0) - pad.first) / ratio.first;
        cv::min(col, w_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(0));
    }

    // y1
    {
        cv::Mat col = (out.col(1) - pad.second) / ratio.second;
        cv::min(col, h_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(1));
    }

    // x2
    {
        cv::Mat col = (out.col(2) - pad.first) / ratio.first;
        cv::min(col, w_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(2));
    }

    // y2
    {
        cv::Mat col = (out.col(3) - pad.second) / ratio.second;
        cv::min(col, h_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(3));
    }

    return out;
}

cv::Mat draw_boxes(const cv::Mat&                  img,
                   const std::vector<Detection>&   detections,
                   const std::vector<std::string>& class_names)
{
    cv::Mat out = img.clone();

    for (const Detection& det : detections) {
        int x1 = static_cast<int>(det.x1), y1 = static_cast<int>(det.y1);
        int x2 = static_cast<int>(det.x2), y2 = static_cast<int>(det.y2);
        int id = det.class_id;

        // Deterministic per-class colour
        cv::Scalar color(
            (id * 67  + 100) % 255,
            (id * 113 +  50) % 255,
            (id * 179 + 150) % 255
        );

        cv::rectangle(out, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // Label
        std::string label = (class_names.size() > static_cast<size_t>(id))
            ? class_names[id] + ": " + std::to_string(det.score).substr(0, 4)
            : "Class " + std::to_string(id) + ": " +
              std::to_string(det.score).substr(0, 4);

        int      baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                       0.5, 1, &baseline);

        // Label background
        cv::rectangle(out,
                       cv::Point(x1, y1 - ts.height - 6),
                       cv::Point(x1 + ts.width, y1),
                       color, cv::FILLED);

        // Label text
        cv::putText(out, label, cv::Point(x1, y1 - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }

    return out;
}


#ifdef ONNX_MODE

// ============================================================================
//  OnnxInferenceEngine
// ============================================================================
//
//  設計要點：
//
//  (1) 所有 buffer 在 ctor 配好，run() 完全不做 heap allocation。
//      - input_tensor_values_ : N*C*H*W 個 float
//      - outputs_             : 每個輸出對應一個 4D cv::Mat
//
//  (2) 輸出不 copy。ORT 的 Value::CreateTensor<float>(..., user_data, ...)
//      會讓 ORT 把結果直接寫進我們提供的 buffer（也就是 cv::Mat 的 data）。
//      推理結束後 outputs_[i] 即可直接使用。
//
//  (3) 輸出 shape 的動態維度（-1）被當作 1 處理，與原 onnx_test.cpp 一致；
//      這對常見 YOLO 系列模型（固定 input → 固定 output）足夠。若有真正動態
//      shape 需求，需要改為每次 Run 後讀 tensor_info 再重新包 cv::Mat。
//
//  (4) 名稱字串生命週期：GetInputNameAllocated 回傳的 AllocatedStringPtr
//      擁有 char* 的所有權，我們把 ptr 存進 allocated_strings_、把 raw char*
//      push 進 input_names_/output_names_，這樣 ORT Run 時指標仍有效。
//

OnnxInferenceEngine::OnnxInferenceEngine(const std::string& model_path,
                                         ConfigureOptionsFn configure_options)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxInferenceEngine")
{
    // SessionOptions 不可複製（RAII wrapper 包 C handle），所以在 ctor 內
    // 就地建好、讓呼叫端透過 callback 修改、然後傳給 Session ctor。
    // Session ctor 會把 options 的內容 copy 到 session 內部，之後 opts
    // 可以安心被解構。
    Ort::SessionOptions opts;
    if (configure_options) {
        configure_options(opts);
    }

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
    initialize_model_info();
}


void OnnxInferenceEngine::initialize_model_info()
{
    // ── 輸入 ──
    {
        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator_);
        input_names_.push_back(input_name_ptr.get());
        allocated_strings_.push_back(std::move(input_name_ptr));

        auto in_shape = session_->GetInputTypeInfo(0)
                                 .GetTensorTypeAndShapeInfo()
                                 .GetShape();
        if (in_shape.size() != 4) {
            throw std::runtime_error("OnnxInferenceEngine: input must be 4D (NCHW).");
        }
        // NCHW；動態 batch 視為 1
        ch_   = (in_shape[1] < 0) ? 3 : in_shape[1];
        in_h_ = (in_shape[2] < 0) ? 1 : in_shape[2];
        in_w_ = (in_shape[3] < 0) ? 1 : in_shape[3];

        if (ch_ != 3) {
            throw std::runtime_error(
                "OnnxInferenceEngine: only 3-channel input is supported.");
        }

        input_shape_ = {1, ch_, in_h_, in_w_};
        input_tensor_values_.assign(
            static_cast<size_t>(ch_ * in_h_ * in_w_), 0.0f);
    }

    // ── 輸出 ──
    const size_t num_outputs = session_->GetOutputCount();
    outputs_.clear();
    outputs_.reserve(num_outputs);
    output_shapes_.clear();
    output_shapes_.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {
        auto out_name_ptr = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.push_back(out_name_ptr.get());
        allocated_strings_.push_back(std::move(out_name_ptr));

        auto out_shape = session_->GetOutputTypeInfo(i)
                                  .GetTensorTypeAndShapeInfo()
                                  .GetShape();
        // 把動態維度 (-1) 視為 1
        for (auto& d : out_shape) if (d < 0) d = 1;

        if (out_shape.size() != 4) {
            throw std::runtime_error(
                "OnnxInferenceEngine: only 4D outputs are supported (got shape size "
                + std::to_string(out_shape.size()) + ").");
        }

        // 用 cv::Mat 的 4D 建構：sizes = {N, C, H, W}
        std::vector<int> sizes_int(out_shape.size());
        for (size_t k = 0; k < out_shape.size(); ++k)
            sizes_int[k] = static_cast<int>(out_shape[k]);

        outputs_.emplace_back(static_cast<int>(sizes_int.size()),
                              sizes_int.data(), CV_32F);
        output_shapes_.push_back(std::move(out_shape));
    }
}


void OnnxInferenceEngine::hwc_to_nchw(const cv::Mat& src)
{
    // src: CV_32FC3, 尺寸 (in_h_, in_w_)（由 run() 的 assert 保證）
    // dst buffer 已經在 ctor 配好，這裡只是把 3 個通道 split 到連續的 plane
    const int H = static_cast<int>(in_h_);
    const int W = static_cast<int>(in_w_);
    const size_t plane = static_cast<size_t>(H) * W;

    std::vector<cv::Mat> ch_planes;
    ch_planes.reserve(3);
    for (int c = 0; c < 3; ++c) {
        ch_planes.emplace_back(H, W, CV_32FC1,
                               input_tensor_values_.data() + c * plane);
    }
    cv::split(src, ch_planes);
}


void OnnxInferenceEngine::run(const cv::Mat& input_img)
{
    // ── 防呆：型別與尺寸 ──
    //
    // 要求嚴格的 CV_32FC3 + (in_h_, in_w_)。前處理端（fix2float）現在會
    // 保留 3-channel header，所以這個 assert 能通過。若你之後又看到這個
    // assert 爆掉，第一嫌疑仍是「有人用 cv::Mat_<float> 當 convertTo 的
    // 輸出」——Mat_<T> 會強制單通道 reshape，請改回 cv::Mat。
    CV_Assert(input_img.type() == CV_32FC3);
    CV_Assert(input_img.cols == in_w_ && input_img.rows == in_h_);

    // ── HWC → NCHW ──
    hwc_to_nchw(input_img);

    // ── 建立 ORT Value：輸入 ──
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(1);
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values_.data(),
        input_tensor_values_.size(),
        input_shape_.data(),
        input_shape_.size()));

    // ── 建立 ORT Value：輸出（綁到預配置的 cv::Mat buffer）──
    std::vector<Ort::Value> output_tensors;
    output_tensors.reserve(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i) {
        cv::Mat& m = outputs_[i];
        const size_t n_elem = static_cast<size_t>(m.total());
        output_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            m.ptr<float>(),         // ORT 會直接寫入這塊記憶體
            n_elem,
            output_shapes_[i].data(),
            output_shapes_[i].size()));
    }

    // ── Run ──
    session_->Run(Ort::RunOptions{nullptr},
                  input_names_.data(),  input_tensors.data(),  input_tensors.size(),
                  output_names_.data(), output_tensors.data(), output_tensors.size());
}

#endif

 
#ifdef XMODEL_MODE

// ============================================================================
//  XmodelInferenceEngine 實作
//
//  連結需求（典型最小集合）:
//      -lvart-runner -lxir -lglog
//  若 link error 抱怨缺 symbol，依訊息補:
//      -lvart-util -lunilog ...
// ============================================================================

// 加上這些 VART / XIR 的標頭檔！
#include <xir/graph/graph.hpp>
#include <xir/attrs/attrs.hpp>
#include <xir/tensor/tensor.hpp>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>

// ── 匿名 helper：從 XIR Tensor 抓取 fix_point 並轉為 scale ──
namespace {

// input  scale = 2^(+fix_point)
inline float get_input_scale(const xir::Tensor* t) {
    int fp = t->template get_attr<int>("fix_point");
    return std::exp2f(static_cast<float>(fp));
}

// output scale = 2^(-fix_point)
inline float get_output_scale(const xir::Tensor* t) {
    int fp = t->template get_attr<int>("fix_point");
    return std::exp2f(-static_cast<float>(fp));
}

} // namespace

// ── 建構子 / 解構子 ─────────────────────────────────────────────────────────

XmodelInferenceEngine::XmodelInferenceEngine(const std::string& xmodel_path)
{
    // 建構時直接呼叫初始化函數
    initialize_model_info(xmodel_path);
}

// 解構子必須在 .cpp 中定義，因為 std::unique_ptr 需要知道 xir/vart 類別的完整大小才能釋放
XmodelInferenceEngine::~XmodelInferenceEngine() = default;


// ── 實作區 ─────────────────────────────────────────────────────────────────

void XmodelInferenceEngine::initialize_model_info(const std::string& xmodel_path)
{
    // 1. Deserialize xmodel & 挑出 DPU subgraph
    graph_ = xir::Graph::deserialize(xmodel_path);
    const auto* root = graph_->get_root_subgraph();
 
    // 👇 這裡加上 const！
    const xir::Subgraph* dpu_subgraph = nullptr; 
    
    for (auto* c : root->children_topological_sort()) {
        if (c->has_attr("device") && c->get_attr<std::string>("device") == "DPU") {
            dpu_subgraph = c;
            break;
        }
    }
    
    if (!dpu_subgraph) {
        throw std::runtime_error("XmodelInferenceEngine: no DPU subgraph found in " + xmodel_path);
    }
 
    // 2. 建 runner
    attrs_  = xir::Attrs::create();
    runner_ = vart::RunnerExt::create_runner(dpu_subgraph, attrs_.get());
 
    // 3. 抓 input / output tensor buffers
    input_tensor_buffers_  = runner_->get_inputs();
    output_tensor_buffers_ = runner_->get_outputs();

    // 4. Input meta & 建立輸入替身 (input_mat_)
    {
        const auto* in_t  = input_tensor_buffers_[0]->get_tensor();
        const auto  shape = in_t->get_shape();
        in_h_ = shape[1];
        in_w_ = shape[2];
        in_c_ = shape[3];
        input_scale_ = get_input_scale(in_t);

        // 取得 DPU Input Buffer 實體位址並封裝成 cv::Mat
        uint64_t data_addr = 0u;
        size_t   size_bytes = 0u;
        std::tie(data_addr, size_bytes) = input_tensor_buffers_[0]->data({0, 0, 0, 0});
        
        input_mat_ = cv::Mat(in_h_, in_w_, CV_8SC3, reinterpret_cast<void*>(data_addr));
    }

    // 5. Output meta & 建立輸出替身
    const size_t num_outs = output_tensor_buffers_.size();
    outputs_.clear();        outputs_.reserve(num_outs);
    outputs_nchw_.clear();   outputs_nchw_.reserve(num_outs); // 新增
    output_scales_.clear();  output_scales_.reserve(num_outs);
 
    for (size_t i = 0; i < num_outs; ++i) {
        const auto* out_t = output_tensor_buffers_[i]->get_tensor();
        const auto  shape = out_t->get_shape();
        output_scales_.push_back(get_output_scale(out_t));
 
        // DPU 原生的 NHWC shape
        std::vector<int> sizes_int(shape.size());
        for (size_t k = 0; k < shape.size(); ++k) sizes_int[k] = static_cast<int>(shape[k]);
 
        // 取得 DPU 實體位址並封裝成 cv::Mat (NHWC)
        std::vector<int> idx(shape.size(), 0);
        uint64_t data_addr = 0u; size_t size_bytes = 0u;
        std::tie(data_addr, size_bytes) = output_tensor_buffers_[i]->data(idx);
        
        outputs_.emplace_back(static_cast<int>(sizes_int.size()),
                              sizes_int.data(), CV_8S, reinterpret_cast<void*>(data_addr));

        // 🔥 [新增] 預先配置一塊獨立的 CPU 記憶體，用來放 NCHW 的 int8 資料
        int C = sizes_int[sizes_int.size() - 1];
        int W = sizes_int[sizes_int.size() - 2];
        int H = sizes_int[sizes_int.size() - 3];
        int sizes_nchw[] = {1, C, H, W};
        outputs_nchw_.emplace_back(4, sizes_nchw, CV_8S); 
    }
}

// ── 執行推理：極簡化，只做 sync 與 execute ─────────────────────────────────
void XmodelInferenceEngine::run()
{
    // 1. flush cache
    for (auto* inp : input_tensor_buffers_) {
        const auto* t = inp->get_tensor();
        inp->sync_for_write(0, t->get_data_size() / t->get_shape()[0]);
    }
 
    // 2. 跑 DPU
    auto v = runner_->execute_async(input_tensor_buffers_, output_tensor_buffers_);
    const int status = runner_->wait(static_cast<int>(v.first), -1);
    // ... 錯誤檢查 ...
 
    // 3. invalidate cache (讓 CPU 讀到最新 DPU 結果)
    for (auto* out : output_tensor_buffers_) {
        const auto* t = out->get_tensor();
        out->sync_for_read(0, t->get_data_size() / t->get_shape()[0]);
    }

    // 4. 🔥 [新增] 把 DPU 的 NHWC int8 資料，搬移到預建的 NCHW int8 替身中
    for (size_t i = 0; i < output_tensor_buffers_.size(); ++i) {
        const cv::Mat& nhwc_mat = outputs_[i];
        cv::Mat&       nchw_mat = outputs_nchw_[i];

        int ndims = nhwc_mat.dims;
        int H = nhwc_mat.size[ndims - 3];
        int W = nhwc_mat.size[ndims - 2];
        int C = nhwc_mat.size[ndims - 1];

        const int8_t* src = nhwc_mat.ptr<int8_t>();
        int8_t* dst = nchw_mat.ptr<int8_t>();

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    dst[c * H * W + h * W + w] = src[h * W * C + w * C + c];
                }
            }
        }
    }
}

#endif