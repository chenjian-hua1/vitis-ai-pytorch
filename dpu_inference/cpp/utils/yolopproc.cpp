#include <yolopproc.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <numeric>
#include <stdexcept>

// ============================================================================
//  YOLOPostProcessor (NCHW input)
// ============================================================================

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
    bind_views();
    cache_pointers();
    precompute_tables();
}


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

    active_indices_.assign(B_, {});
    active_cls_scores_.assign(B_, {});
    for (int b = 0; b < B_; ++b) {
        active_indices_[b].reserve(A_);
        active_cls_scores_[b].reserve(A_ * nc_);
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
    ax_ = anchors_T_.ptr<float>(0);
    ay_ = anchors_T_.ptr<float>(1);
    sv_ = stride_T_.ptr<float>(0);

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

    dfl_rows_.resize(B_);
    for (int b = 0; b < B_; ++b)
        for (int coord = 0; coord < 4; ++coord)
            dfl_rows_[b][coord] = dfl_out_.ptr<float>(b, coord);

    const int split = 4 * ch_;
    box_raw_rows_.assign(B_, std::vector<const float*>(split, nullptr));
    cls_raw_rows_.assign(B_, std::vector<const float*>(nc_,   nullptr));
    for (int b = 0; b < B_; ++b) {
        for (int c = 0; c < split; ++c)
            box_raw_rows_[b][c] = box_raw_view_.ptr<float>(b, c);
        for (int c = 0; c < nc_; ++c)
            cls_raw_rows_[b][c] = cls_raw_view_.ptr<float>(b, c);
    }

    xcat_rows_.assign(B_, std::vector<float*>(no_, nullptr));
    for (int b = 0; b < B_; ++b)
        for (int c = 0; c < no_; ++c)
            xcat_rows_[b][c] = x_cat_.ptr<float>(b, c);
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


// ═══════════════ decode pipeline ═══════════════

void YOLOPostProcessor::classify_and_build_mask(float conf_thresh)
{
    const int A = A_;

    const float logit_thresh = (conf_thresh > 0.f && conf_thresh < 1.f)
        ? std::log(conf_thresh / (1.f - conf_thresh))
        : -FLT_MAX;

    for (int b = 0; b < B_; ++b) {
        active_indices_[b].clear();

        const std::vector<const float*>& cls_raw = cls_raw_rows_[b];

        // Scan Anthor
        for (int a = 0; a < A; ++a) {
            for (int c = 0; c < nc_; ++c) {
                if (cls_raw[c][a] > logit_thresh) 
                {
                    // confidence > th : record position, score, cls
                    active_indices_[b].push_back(a);
                    break; 
                }
            }
        }

        const int n_active = static_cast<int>(active_indices_[b].size());

        if (nc_ > 1 && n_active > 0) {
            auto& scores = active_cls_scores_[b];
            scores.resize(static_cast<size_t>(n_active) * nc_);

            const int* RESTRICT act = active_indices_[b].data();
            float* RESTRICT dst = scores.data();

            // 掃描全部 Activate Anchor (conf>th部份)
            for (int i = 0; i < n_active; ++i) {
                const int a = act[i];
                float* RESTRICT row = dst + static_cast<size_t>(i) * nc_;
                // 計算每個 cls 的機率 (sigmoid) 
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

            const float* RESTRICT ch_rows[32];
            for (int c = 0; c < ch; ++c)
                ch_rows[c] = box_raw_rows_[b][base + c];

            for (int i = 0; i < N; ++i) {
                // softmax
                const int a = act[i];

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

    // NCHW concat: 一個 channel 一次 memcpy (H*W floats)
    for (size_t i = 0; i < x.size(); ++i) {
        const cv::Mat& xi = x[i];
        const int hw = xi.size[2] * xi.size[3];
        const int col_offset = scale_offsets_[i];

        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < no_; ++c) {
                const float* src = xi.ptr<float>(b, c);
                float*       dst = xcat_rows_[b][c] + col_offset;
                std::memcpy(dst, src, hw * sizeof(float));
            }
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
            // nc_ == 1:直接對那唯一一類的 logit 取 sigmoid
            float score = 1.f / (1.f + expf(-cls_raw_rows_[b][0][a]));
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