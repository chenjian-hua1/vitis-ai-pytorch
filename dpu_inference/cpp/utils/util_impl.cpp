#include "util.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstring>
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
//  ImageNet normalisation constants
// ============================================================================
namespace {

constexpr float kMeanR = 0.485f, kMeanG = 0.456f, kMeanB = 0.406f;
constexpr float kStdR  = 0.229f, kStdG  = 0.224f, kStdB  = 0.225f;

constexpr float kU8ScaleR = 1.f/(kStdR*255.f);
constexpr float kU8ScaleG = 1.f/(kStdG*255.f);
constexpr float kU8ScaleB = 1.f/(kStdB*255.f);
constexpr float kU8BiasR  = -kMeanR/kStdR;
constexpr float kU8BiasG  = -kMeanG/kStdG;
constexpr float kU8BiasB  = -kMeanB/kStdB;

constexpr float kF32ScaleR = 1.f/kStdR;
constexpr float kF32ScaleG = 1.f/kStdG;
constexpr float kF32ScaleB = 1.f/kStdB;

} // anonymous namespace

// ============================================================================
//  Fix / Float Conversion
// ============================================================================

void fix2float(const cv::Mat& data, int fix_point, cv::Mat& out)
{
    float scale = std::exp2f(-static_cast<float>(fix_point));
    data.convertTo(out, CV_32FC3, scale);
}

void float2fix(const cv::Mat& data, int fix_point, cv::Mat& out)
{
    float scale = std::exp2f(static_cast<float>(fix_point));
    data.convertTo(out, CV_8SC3, scale);
}

// ============================================================================
//  Bounding Box Utilities
// ============================================================================

cv::Mat wh2xy(const cv::Mat& x)
{
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
        dst[i * cols + 0] = cx - w * 0.5f;
        dst[i * cols + 1] = cy - h * 0.5f;
        dst[i * cols + 2] = cx + w * 0.5f;
        dst[i * cols + 3] = cy + h * 0.5f;
    }
    return y;
}

cv::Mat xyxy2xywh(const cv::Mat& box)
{
    assert(box.cols == 4 && box.type() == CV_32F);
    cv::Mat out = box.clone();

    const int N = box.rows;
    const float* RESTRICT src = box.ptr<float>();
    float*       RESTRICT dst = out.ptr<float>();

    #pragma omp simd
    for (int i = 0; i < N; ++i) {
        dst[i * 4 + 2] = src[i * 4 + 2] - src[i * 4 + 0];
        dst[i * 4 + 3] = src[i * 4 + 3] - src[i * 4 + 1];
    }
    return out;
}

// ============================================================================
//  Anchor Generation (free function, unchanged interface)
// ============================================================================
//
//  注意：這個 free function 仍假設 feature map 為 4D，並從 size[2], size[3]
//  讀 H, W。若呼叫端餵 NHWC，會讀錯維度。本檔案內 YOLOPostProcessor 不再
//  使用這個 free function（它有自己的 precompute_anchors() 從 strides 推算
//  H, W），所以此函數保持原樣不會影響新 pipeline。若你別處有用，需要呼叫端
//  自己注意 layout。

AnchorResult make_anchors(const std::vector<cv::Mat>& feature_maps,
                            const std::vector<int>&     strides,
                            float                       offset)
{
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
                anchors.at<float>(idx, 0) = static_cast<float>(gx) + offset;
                anchors.at<float>(idx, 1) = static_cast<float>(gy) + offset;
            }
        }
        anchor_list.push_back(anchors);
        stride_list.push_back(strides_mat);
    }

    AnchorResult res;
    cv::vconcat(anchor_list, res.anchors);
    cv::vconcat(stride_list, res.stride_tensor);
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

    res.img.create(input_size, input_size, img.type());
    res.img.setTo(cv::Scalar(0,0,0));

    cv::Mat roi = res.img(cv::Rect(left, top, pad_w, pad_h));
    cv::resize(img, roi, roi.size(), 0, 0, cv::INTER_LINEAR);
}


// ============================================================================
//  YOLOPostProcessor (NHWC input)
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
    precompute_anchors();
    allocate_buffers();
    cache_pointers();
    precompute_tables();
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
        scale_offsets_.push_back(acc_offset);
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
    // ⭐ x_cat_ 改成 NHWC：(B, A, no_)
    //    同一個 anchor 的 no_ 個 channel 連續，inner loop sequential。
    x_cat_   = cv::Mat(std::vector<int>{B_, A_,      no_}, CV_32F);

    // output_ / dfl_out_ 維持 NCHW（dist2bbox / NMS 的 row-wise 寫法不變）。
    dfl_out_ = cv::Mat(std::vector<int>{B_, 4,       A_},  CV_32F);
    output_  = cv::Mat(std::vector<int>{B_, 4 + nc_, A_},  CV_32F);

    candidates_.resize(max_nms_);
    boxes_cv_.resize(max_nms_);
    scores_cv_.resize(max_nms_);
    indices_.reserve(max_nms_);

    detections_.resize(B_);
    for (auto& db : detections_) {
        db.data.resize(max_det_);
        db.count = 0;
    }

    active_indices_.assign(B_, {});
    active_max_score_.assign(B_, {});
    active_max_cls_.assign(B_, {});
    active_cls_scores_.assign(B_, {});
    for (int b = 0; b < B_; ++b) {
        active_indices_[b].reserve(A_);
        active_max_score_[b].reserve(A_);
        active_max_cls_[b].reserve(A_);
        active_cls_scores_[b].reserve(A_ * nc_);
    }
}


void YOLOPostProcessor::cache_pointers()
{
    // anchors / stride
    ax_ = anchors_T_.ptr<float>(0);
    ay_ = anchors_T_.ptr<float>(1);
    sv_ = stride_T_.ptr<float>(0);

    // ── ① output_ 各 channel 的 row pointer（NCHW）──
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

    // ── ② dfl_out_ 各方向的 row pointer（NCHW）──
    dfl_rows_.resize(B_);
    for (int b = 0; b < B_; ++b)
        for (int coord = 0; coord < 4; ++coord)
            dfl_rows_[b][coord] = dfl_out_.ptr<float>(b, coord);

    // ── ③ x_cat_ 各 batch 的起點 pointer（NHWC）──
    xcat_base_.assign(B_, nullptr);
    for (int b = 0; b < B_; ++b)
        xcat_base_[b] = x_cat_.ptr<float>(b);
}


void YOLOPostProcessor::precompute_tables()
{
    constexpr float max_wh = 7680.f;
    class_offsets_.resize(nc_);
    for (int c = 0; c < nc_; ++c)
        class_offsets_[c] = static_cast<float>(c) * max_wh;

    dfl_arange_.resize(ch_);
    std::iota(dfl_arange_.begin(), dfl_arange_.end(), 0.f);
}


// ═══════════════ decode pipeline (NHWC) ═══════════════
//
//  Pipeline：
//    1. concat NHWC feature maps → x_cat_(NHWC)，大塊 memcpy
//    2. classify_and_build_mask: 對每個 anchor 在 logit 空間找 max,
//       建 active list；只對 active anchor 算 sigmoid
//    3. dfl_decode_masked: 只處理 active anchor 的 DFL softmax
//    4. dist2bbox_masked: 只處理 active anchor 的座標轉換
//
//  NHWC 紅利：每個 anchor 的 no_ 個 channel 在記憶體中完全連續，
//  classify 與 DFL 的 inner loop 變成 sequential read，
//  編譯器可向量化、cache 友善。
//

void YOLOPostProcessor::classify_and_build_mask(float conf_thresh)
{
    const int A = A_;
    const int CLS_OFFSET = 4 * ch_;   // 一個 anchor 內 class logit 的起始 offset

    const float logit_thresh = (conf_thresh > 0.f && conf_thresh < 1.f)
        ? std::log(conf_thresh / (1.f - conf_thresh))
        : -FLT_MAX;

    for (int b = 0; b < B_; ++b) {
        active_indices_[b].clear();
        active_max_score_[b].clear();
        active_max_cls_[b].clear();
        active_cls_scores_[b].clear();

        const float* RESTRICT xcat = xcat_base_[b];  // (A, no_) NHWC

        // ── Step A: 逐 anchor 在 logit 空間找 max + argmax ──
        //
        // NHWC 下，cls = xcat + a * no_ + CLS_OFFSET 指向該 anchor 的 nc 個
        // class logit，這 nc 個值在記憶體中完全連續。inner max-reduction
        // loop 為 sequential read，編譯器可向量化。
        for (int a = 0; a < A; ++a) {
            // achor_base_addr = achor_idx*no (ch*4+nc)
            // achor_cls_base_addr = achor_base_addr + ch*4
            // scan nc times cls logit [achor_cls_base_addr:achor_cls_base_addr+nc]
            const float* RESTRICT cls = xcat + static_cast<size_t>(a) * no_ + CLS_OFFSET;

            // max(cls[0:nc])
            float max_l  = cls[0];
            int   max_id = 0;
            for (int c = 1; c < nc_; ++c) {
                float v = cls[c];
                if (v > max_l) { max_l = v; max_id = c; }
            }

            // max_logit < conf  →  jump calculate sigmoid prob 
            if (max_l <= logit_thresh) continue;

            // calculate cls prob (sigmoid)
            active_indices_[b].push_back(a);
            float max_s = 1.f / (1.f + expf(-max_l));
            active_max_score_[b].push_back(max_s);
            active_max_cls_[b].push_back(max_id);
        }

        const int n_active = static_cast<int>(active_indices_[b].size());

        // ── Step B（僅 nc > 1）：對 active anchor 算完整 nc 個 sigmoid，
        //    供 NMS multi-label 分支使用。layout: scores[i*nc + c].
        if (nc_ > 1 && n_active > 0) {
            auto& scores = active_cls_scores_[b];
            scores.resize(static_cast<size_t>(n_active) * nc_);

            const int* RESTRICT act = active_indices_[b].data();
            float* RESTRICT dst = scores.data();

            for (int i = 0; i < n_active; ++i) {
                const int a = act[i];
                const float* RESTRICT cls = xcat + static_cast<size_t>(a) * no_ + CLS_OFFSET;
                float* RESTRICT row = dst + static_cast<size_t>(i) * nc_;
                for (int c = 0; c < nc_; ++c)
                    row[c] = 1.f / (1.f + expf(-cls[c]));
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

        const float* RESTRICT xcat = xcat_base_[b];  // (A, no_) NHWC

        // ⭐ NHWC 下，一個 anchor 的 4*ch 個 box logit 完全連續。
        //    不再需要 4 個外層 coord loop + ch 條 row pointer array；
        //    直接對每個 active anchor 一氣呵成跑 4 個 coord 的 softmax。
        for (int i = 0; i < N; ++i) {
            const int a = act[i];
            const float* RESTRICT box = xcat + static_cast<size_t>(a) * no_;

            for (int coord = 0; coord < 4; ++coord) {
                const float* RESTRICT logits = box + coord * ch;

                // numerically-stable softmax + 期望值
                float max_l = logits[0];
                for (int c = 1; c < ch; ++c)
                    if (logits[c] > max_l) max_l = logits[c];

                float sum_e = 0.f, wsum = 0.f;
                for (int c = 0; c < ch; ++c) {
                    float e = expf(logits[c] - max_l);
                    sum_e += e;
                    wsum  += e * dfl_arange_[c];
                }
                dfl_rows_[b][coord][a] = wsum / sum_e;
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

    // ── 1. concat NHWC feature maps → x_cat_(NHWC) ──
    //
    // 輸入: x[i] shape = (B, H_i, W_i, no_)，每個 batch 內 (H*W*no_) 連續
    // 輸出: x_cat_ shape = (B, A, no_)
    //
    // 由於兩端 layout 完全一致（每個 anchor 的 no_ channel 都連續），
    // 只需把每個 batch、每個 scale 的整塊資料 memcpy 到對應 anchor 區間。
    for (size_t i = 0; i < x.size(); ++i) {
        const cv::Mat& xi = x[i];

        // NHWC: shape = (B, H, W, C)
        if (xi.dims != 4)
            throw std::runtime_error("feature map must be 4D (NHWC)");
        if (xi.size[3] != no_)
            throw std::runtime_error("feature map channel count mismatch (expect NHWC: B,H,W,no_)");

        const int H_i = xi.size[1];
        const int W_i = xi.size[2];
        const int hw  = H_i * W_i;

        if (hw != hw_per_scale_[i])
            throw std::runtime_error("feature map HW does not match precomputed anchor grid");

        const size_t chunk_floats = static_cast<size_t>(hw) * no_;
        const size_t anchor_offset = static_cast<size_t>(scale_offsets_[i]) * no_;

        for (int b = 0; b < B; ++b) {
            const float* src = xi.ptr<float>(b);
            float*       dst = xcat_base_[b] + anchor_offset;
            std::memcpy(dst, src, chunk_floats * sizeof(float));
        }
    }

    conf_thresh_cached_ = conf_thresh;

    classify_and_build_mask(conf_thresh);
    dfl_decode_masked();
    dist2bbox_masked();

    return output_;
}


// ═══════════════ NMS ═══════════════

void YOLOPostProcessor::nms_single_batch(int b,
                                          float conf_thresh,
                                          float iou_thresh)
{
    cand_count_ = 0;

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
    const std::vector<float>& cls_scores  = active_cls_scores_[b];
    const int N = static_cast<int>(act.size());

    for (int i = 0; i < N; ++i) {
        const int a = act[i];

        float cx = cx_row[a], cy = cy_row[a];
        float w  = ww_row[a], h  = hh_row[a];
        float x1 = cx - w * 0.5f, y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f, y2 = cy + h * 0.5f;

        if (nc_ > 1) {
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

    {
        cv::Mat col = (out.col(0) - pad.first) / ratio.first;
        cv::min(col, w_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(0));
    }
    {
        cv::Mat col = (out.col(1) - pad.second) / ratio.second;
        cv::min(col, h_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(1));
    }
    {
        cv::Mat col = (out.col(2) - pad.first) / ratio.first;
        cv::min(col, w_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(2));
    }
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

        cv::Scalar color(
            (id * 67  + 100) % 255,
            (id * 113 +  50) % 255,
            (id * 179 + 150) % 255
        );

        cv::rectangle(out, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        std::string label = (class_names.size() > static_cast<size_t>(id))
            ? class_names[id] + ": " + std::to_string(det.score).substr(0, 4)
            : "Class " + std::to_string(id) + ": " +
              std::to_string(det.score).substr(0, 4);

        int      baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                       0.5, 1, &baseline);

        cv::rectangle(out,
                       cv::Point(x1, y1 - ts.height - 6),
                       cv::Point(x1 + ts.width, y1),
                       color, cv::FILLED);

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
//  本引擎對 output shape 採「任何 4D 都接受」的態度（assert size == 4）。
//  若 ONNX 模型 export 為 NHWC 輸出 (1, H, W, C)，這裡會用該 shape 配
//  cv::Mat，PostProcessor 也以 NHWC 接受，整條鏈路沒有任何 layout 轉置。
//

OnnxInferenceEngine::OnnxInferenceEngine(const std::string& model_path,
                                         ConfigureOptionsFn configure_options)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxInferenceEngine")
{
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
        for (auto& d : out_shape) if (d < 0) d = 1;

        if (out_shape.size() != 4) {
            throw std::runtime_error(
                "OnnxInferenceEngine: only 4D outputs are supported (got shape size "
                + std::to_string(out_shape.size()) + ").");
        }

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
    CV_Assert(input_img.type() == CV_32FC3);
    CV_Assert(input_img.cols == in_w_ && input_img.rows == in_h_);

    hwc_to_nchw(input_img);

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

    std::vector<Ort::Value> output_tensors;
    output_tensors.reserve(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i) {
        cv::Mat& m = outputs_[i];
        const size_t n_elem = static_cast<size_t>(m.total());
        output_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            m.ptr<float>(),
            n_elem,
            output_shapes_[i].data(),
            output_shapes_[i].size()));
    }

    session_->Run(Ort::RunOptions{nullptr},
                  input_names_.data(),  input_tensors.data(),  input_tensors.size(),
                  output_names_.data(), output_tensors.data(), output_tensors.size());
}

#endif


#ifdef XMODEL_MODE

// ============================================================================
//  XmodelInferenceEngine (NHWC native)
// ============================================================================
//
//  DPU 原生輸出為 NHWC；PostProcessor 也接受 NHWC，因此本引擎不再做任何
//  NHWC→NCHW 轉置，直接把 DPU 記憶體上的 NHWC int8 buffer 包成 cv::Mat
//  暴露給呼叫端。
//
//  注意：輸出為 int8 (CV_8S)，PostProcessor 需要 float32 (CV_32F)。
//  呼叫端需用 fix2float() 反量化（element-wise，不影響 layout）。
// ============================================================================

#include <xir/graph/graph.hpp>
#include <xir/attrs/attrs.hpp>
#include <xir/tensor/tensor.hpp>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>

namespace {

inline float get_input_scale(const xir::Tensor* t) {
    int fp = t->template get_attr<int>("fix_point");
    return std::exp2f(static_cast<float>(fp));
}

inline float get_output_scale(const xir::Tensor* t) {
    int fp = t->template get_attr<int>("fix_point");
    return std::exp2f(-static_cast<float>(fp));
}

} // namespace


XmodelInferenceEngine::XmodelInferenceEngine(const std::string& xmodel_path)
{
    initialize_model_info(xmodel_path);
}

XmodelInferenceEngine::~XmodelInferenceEngine() = default;


void XmodelInferenceEngine::initialize_model_info(const std::string& xmodel_path)
{
    // 1. Deserialize xmodel & 挑出 DPU subgraph
    graph_ = xir::Graph::deserialize(xmodel_path);
    const auto* root = graph_->get_root_subgraph();

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

    // 4. Input meta & 建立輸入替身
    {
        const auto* in_t  = input_tensor_buffers_[0]->get_tensor();
        const auto  shape = in_t->get_shape();
        in_h_ = shape[1];
        in_w_ = shape[2];
        in_c_ = shape[3];
        input_scale_ = get_input_scale(in_t);

        uint64_t data_addr = 0u;
        size_t   size_bytes = 0u;
        std::tie(data_addr, size_bytes) = input_tensor_buffers_[0]->data({0, 0, 0, 0});

        input_mat_ = cv::Mat(in_h_, in_w_, CV_8SC3, reinterpret_cast<void*>(data_addr));
    }

    // 5. Output meta — DPU 原生 NHWC，直接暴露給 PostProcessor
    const size_t num_outs = output_tensor_buffers_.size();
    outputs_.clear();        outputs_.reserve(num_outs);
    output_scales_.clear();  output_scales_.reserve(num_outs);

    for (size_t i = 0; i < num_outs; ++i) {
        const auto* out_t = output_tensor_buffers_[i]->get_tensor();
        const auto  shape = out_t->get_shape();
        output_scales_.push_back(get_output_scale(out_t));

        std::vector<int> sizes_int(shape.size());
        for (size_t k = 0; k < shape.size(); ++k) sizes_int[k] = static_cast<int>(shape[k]);

        std::vector<int> idx(shape.size(), 0);
        uint64_t data_addr = 0u; size_t size_bytes = 0u;
        std::tie(data_addr, size_bytes) = output_tensor_buffers_[i]->data(idx);

        // NHWC int8，PostProcessor 端會用 fix2float 反量化為 float32 NHWC
        outputs_.emplace_back(static_cast<int>(sizes_int.size()),
                              sizes_int.data(), CV_8S, reinterpret_cast<void*>(data_addr));
    }
}


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
    (void)status;  // 若有錯誤檢查需求，這裡可加 throw

    // 3. invalidate cache (讓 CPU 讀到最新 DPU 結果)
    for (auto* out : output_tensor_buffers_) {
        const auto* t = out->get_tensor();
        out->sync_for_read(0, t->get_data_size() / t->get_shape()[0]);
    }

    // ⭐ 完成。DPU 原生 NHWC，無需任何 layout 轉置。
}

#endif