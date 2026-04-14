#include "util.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

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

cv::Mat fix2float(const cv::Mat& data, int fix_point)
{
    // 確保輸入是 8-bit 有號整數 (INT8)，這是 DPU 常見的輸出型別
    CV_Assert(data.depth() == CV_8S);

    cv::Mat out;
    // 使用 double 計算 scale 以維持精度，避免 1 << fix_point 在大位數時溢位
    double scale = 1.0 / static_cast<double>(1 << fix_point);

    // 直接完成：[int8] -> [乘上 scale] -> [轉成 float32]
    data.convertTo(out, CV_32F, scale);

    return out;
}

cv::Mat float2fix(const cv::Mat& data, int fix_point)
{
    // 改用 .depth() 檢查，這只會檢查資料型別(Float32)，不論通道數
    CV_Assert(data.depth() == CV_32F);

    float scale = static_cast<float>(1 << fix_point);

    cv::Mat out;
    data.convertTo(out, CV_8S, scale);  // float → int8 (Q format)

    // float scale = static_cast<float>(1 << fix_point); // 2^fix_point (bit shift)
    // out *= scale;

    // out = cv::max(out, -128.f);
    // out = cv::min(out,  127.f);

    // out.convertTo(out, CV_8S);

    return out;
}

// ============================================================================
//  Bounding Box Utilities
// ============================================================================

cv::Mat wh2xy(const cv::Mat& x)
{
    // x : (N, 4+)  columns: cx, cy, w, h, ...
    assert(x.cols >= 4 && x.type() == CV_32F);
    cv::Mat y = x.clone();

    for (int i = 0; i < x.rows; ++i) {
        const float cx = x.at<float>(i, 0);
        const float cy = x.at<float>(i, 1);
        const float w  = x.at<float>(i, 2);
        const float h  = x.at<float>(i, 3);
        y.at<float>(i, 0) = cx - w * 0.5f;   // x1
        y.at<float>(i, 1) = cy - h * 0.5f;   // y1
        y.at<float>(i, 2) = cx + w * 0.5f;   // x2
        y.at<float>(i, 3) = cy + h * 0.5f;   // y2
    }
    return y;
}

cv::Mat xyxy2xywh(const cv::Mat& box)
{
    // box : (N, 4)  x1, y1, x2, y2
    assert(box.cols == 4 && box.type() == CV_32F);
    cv::Mat out = box.clone();

    for (int i = 0; i < box.rows; ++i) {
        out.at<float>(i, 2) = box.at<float>(i, 2) - box.at<float>(i, 0); // w
        out.at<float>(i, 3) = box.at<float>(i, 3) - box.at<float>(i, 1); // h
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

cv::Mat norm(const cv::Mat& x)
{
    // Select per-channel scale based on input depth:
    //   uint8  path: scale = 1 / (std * 255)
    //   float32 path: scale = 1 / std
    // bias = -(mean / std) is identical for both paths (precomputed at compile time)
    const bool is_uint8 = (x.depth() == CV_8U);
    const float scaleR = is_uint8 ? kU8ScaleR : kF32ScaleR;
    const float scaleG = is_uint8 ? kU8ScaleG : kF32ScaleG;
    const float scaleB = is_uint8 ? kU8ScaleB : kF32ScaleB;

    std::vector<cv::Mat> ch(3);
    cv::split(x, ch);

    // convertTo(dst, type, alpha, beta) computes: dst = src * alpha + beta
    // which maps directly to a single FMA instruction (fused multiply-add).
    // Both alpha (scale) and beta (bias) are compile-time constants,
    // so no runtime arithmetic is performed outside the FMA itself.
    //   ch[c] = fma(src[c], scale[c], bias[c])
    //         = src[c] * scale[c] + bias[c]
    ch[0].convertTo(ch[0], CV_32F, scaleR, kU8BiasR);
    ch[1].convertTo(ch[1], CV_32F, scaleG, kU8BiasG);
    ch[2].convertTo(ch[2], CV_32F, scaleB, kU8BiasB);

    cv::Mat out;
    cv::merge(ch, out);
    return out;
}

ResizeResult resize(const cv::Mat& img, int input_size)
{
    int orig_h = img.rows, orig_w = img.cols;
    float r = std::min(static_cast<float>(input_size) / orig_h,
                       static_cast<float>(input_size) / orig_w);
    r = std::min(r, 1.0f);  // never upscale

    int pad_w = static_cast<int>(std::round(orig_w * r));
    int pad_h = static_cast<int>(std::round(orig_h * r));

    float dw = (input_size - pad_w) / 2.0f;
    float dh = (input_size - pad_h) / 2.0f;

    cv::Mat resized;
    if (img.cols != pad_w || img.rows != pad_h) {
        cv::resize(img, resized, cv::Size(pad_w, pad_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = img.clone();
    }

    int top    = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left   = static_cast<int>(std::round(dw - 0.1f));
    int right  = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    ResizeResult res;
    res.img   = out;
    res.ratio = {r, r};
    res.pad   = {dw, dh};
    return res;
}

ResizeResult resize_zero_copy(const cv::Mat& img, int input_size)
{
    int orig_h = img.rows;
    int orig_w = img.cols;

    float r = std::min((float)input_size / orig_h,
                       (float)input_size / orig_w);
    r = std::min(r, 1.0f);

    int new_w = static_cast<int>(std::round(orig_w * r));
    int new_h = static_cast<int>(std::round(orig_h * r));

    float dw = (input_size - new_w) / 2.0f;
    float dh = (input_size - new_h) / 2.0f;

    int top  = static_cast<int>(std::round(dh - 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));

    // 🔥 只分配一次輸出
    cv::Mat out(input_size, input_size, img.type(), cv::Scalar(0,0,0));

    // 🔥 ROI：以 top/left 為準，寬高用實際剩餘空間 clamp，避免越界
    int roi_w = std::min(new_w, input_size - left);
    int roi_h = std::min(new_h, input_size - top);
    cv::Rect roi(left, top, roi_w, roi_h);
    cv::Mat dst_roi = out(roi);

    // 🔥 直接 resize 到 ROI（沒有 resized 中間變數）
    if (new_w != orig_w || new_h != orig_h) {
        cv::resize(img, dst_roi, dst_roi.size(), 0, 0, cv::INTER_LINEAR);
    } else {
        // ⚠️ 這裡仍然會 copy（無法完全避免，因為位置不同）
        img.copyTo(dst_roi);
    }

    ResizeResult res;
    res.img   = out;
    res.ratio = {r, r};
    res.pad   = {dw, dh};

    return res;
}

// ============================================================================
//  Post-Processing — DFL
// ============================================================================

cv::Mat dfl_decode(const cv::Mat& x, int ch)
{
    // x : (B, 4*ch, A)  stored as a 3-dim Mat
    int B = x.size[0];
    int A = x.size[2];

    cv::Mat out(std::vector<int>{B, 4, A}, CV_32F);

    // arange weights [0, 1, ..., ch-1]
    std::vector<float> arange(ch);
    std::iota(arange.begin(), arange.end(), 0.f);

    for (int b = 0; b < B; ++b) {
        for (int coord = 0; coord < 4; ++coord) {
            for (int a = 0; a < A; ++a) {
                // Gather the ch logits for this (b, coord, a)
                std::vector<float> logits(ch);
                for (int c = 0; c < ch; ++c) {
                    // x[b, coord*ch + c, a]
                    logits[c] = x.at<float>(
                        std::vector<int>{b, coord * ch + c, a}.data());
                }

                // Numerically stable softmax
                float max_l = *std::max_element(logits.begin(), logits.end());
                float sum_e = 0.f;
                for (int c = 0; c < ch; ++c) {
                    logits[c] = std::exp(logits[c] - max_l);
                    sum_e += logits[c];
                }

                // Weighted sum (DFL expectation)
                float val = 0.f;
                for (int c = 0; c < ch; ++c) {
                    val += (logits[c] / sum_e) * arange[c];
                }
                out.at<float>(std::vector<int>{b, coord, a}.data()) = val;
            }
        }
    }
    return out;
}

// ============================================================================
//  Post-Processing — NMS
// ============================================================================

std::vector<std::vector<Detection>> non_max_suppression(
    const cv::Mat& outputs,
    float          confidence_threshold,
    float          iou_threshold)
{
    constexpr float max_wh  = 7680.f;
    constexpr int   max_det = 300;
    constexpr int   max_nms = 30000;

    int B  = outputs.size[0];
    int nc = outputs.size[1] - 4;
    int A  = outputs.size[2];

    std::vector<std::vector<Detection>> result(B);

    auto t_start = std::chrono::steady_clock::now();
    float limit  = 0.5f + 0.05f * B;

    for (int b = 0; b < B; ++b) {
        // Collect candidate rows
        // Each candidate: [x1, y1, x2, y2, score, class_id]
        std::vector<std::array<float,6>> candidates;
        candidates.reserve(512);

        for (int a = 0; a < A; ++a) {
            // Find max class score across nc classes
            float max_cls = -1e9f;
            for (int c = 0; c < nc; ++c) {
                float v = outputs.at<float>(
                    std::vector<int>{b, 4 + c, a}.data());
                if (v > max_cls) max_cls = v;
            }
            if (max_cls <= confidence_threshold) continue;

            // Decode box (cx,cy,w,h → x1,y1,x2,y2)
            float cx = outputs.at<float>(std::vector<int>{b,0,a}.data());
            float cy = outputs.at<float>(std::vector<int>{b,1,a}.data());
            float w  = outputs.at<float>(std::vector<int>{b,2,a}.data());
            float h  = outputs.at<float>(std::vector<int>{b,3,a}.data());
            float x1 = cx - w * 0.5f, y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f, y2 = cy + h * 0.5f;

            if (nc > 1) {
                for (int c = 0; c < nc; ++c) {
                    float score = outputs.at<float>(
                        std::vector<int>{b, 4 + c, a}.data());
                    if (score > confidence_threshold) {
                        candidates.push_back({x1, y1, x2, y2, score,
                                              static_cast<float>(c)});
                    }
                }
            } else {
                float score = outputs.at<float>(std::vector<int>{b,4,a}.data());
                if (score > confidence_threshold) {
                    candidates.push_back({x1, y1, x2, y2, score, 0.f});
                }
            }
        }

        if (candidates.empty()) continue;

        // Sort by score descending, keep top max_nms
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b){ return a[4] > b[4]; });
        if ((int)candidates.size() > max_nms)
            candidates.resize(max_nms);

        // Build offset boxes for batched NMS
        std::vector<cv::Rect2d> boxes_cv;
        std::vector<float>      scores_cv;
        boxes_cv.reserve(candidates.size());
        scores_cv.reserve(candidates.size());

        for (const auto& c : candidates) {
            float off = c[5] * max_wh;
            float ox1 = c[0] + off, oy1 = c[1] + off;
            float ow  = c[2] - c[0], oh = c[3] - c[1];
            boxes_cv.emplace_back(ox1, oy1, ow, oh);
            scores_cv.push_back(c[4]);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes_cv, scores_cv,
                          confidence_threshold, iou_threshold, indices);

        int keep = std::min((int)indices.size(), max_det);
        result[b].reserve(keep);
        for (int k = 0; k < keep; ++k) {
            const auto& row = candidates[indices[k]];
            result[b].push_back({row[0], row[1], row[2], row[3],
                                  row[4], static_cast<int>(row[5])});
        }

        // Time guard
        auto elapsed = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - t_start).count();
        if (elapsed > limit) break;
    }

    return result;
}

// ============================================================================
//  YOLOPostProcessor
// ============================================================================

YOLOPostProcessor::YOLOPostProcessor(int nc, int ch, std::vector<int> strides)
    : nc_(nc), ch_(ch), no_(nc + ch * 4), strides_(std::move(strides))
{}

cv::Mat YOLOPostProcessor::operator()(const std::vector<cv::Mat>& x,
                                       float /*conf_thresh*/) const
{
    int B = x[0].size[0];

    // 1. Generate anchors (A, 2) and stride_tensor (A, 1)
    auto [anchors_mat, stride_vals] = make_anchors(x, strides_);
    // Transpose anchors → (2, A)
    cv::Mat anchors_T;
    cv::transpose(anchors_mat, anchors_T); // (2, A)
    // stride_vals (A, 1) → (1, A)
    cv::Mat sv_T;
    cv::transpose(stride_vals, sv_T); // (1, A)

    // 2. Concatenate all scales → (B, no, A)
    int A = anchors_T.cols;
    cv::Mat x_cat(std::vector<int>{B, no_, A}, CV_32F);
    int col_offset = 0;
    for (const auto& xi : x) {
        int hw = xi.size[2] * xi.size[3];
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < no_; ++c) {
                int h_dim = xi.size[2];
                int w_dim = xi.size[3];
                for (int gy = 0; gy < h_dim; ++gy) {
                    for (int gx = 0; gx < w_dim; ++gx) {
                        int a = gy * w_dim + gx;
                        x_cat.at<float>(std::vector<int>{b, c, col_offset + a}.data())
                            = xi.at<float>(std::vector<int>{b, c, gy, gx}.data());
                    }
                }
            }
        }
        col_offset += hw;
    }

    // 3. Split into box_raw (B, 4*ch, A) and cls_raw (B, nc, A)
    int split = 4 * ch_;
    cv::Mat box_raw(std::vector<int>{B, split, A}, CV_32F);
    cv::Mat cls_raw(std::vector<int>{B, nc_,   A}, CV_32F);

    for (int b = 0; b < B; ++b)
        for (int a = 0; a < A; ++a) {
            for (int c = 0; c < split; ++c)
                box_raw.at<float>(std::vector<int>{b,c,a}.data())
                    = x_cat.at<float>(std::vector<int>{b,c,a}.data());
            for (int c = 0; c < nc_; ++c)
                cls_raw.at<float>(std::vector<int>{b,c,a}.data())
                    = x_cat.at<float>(std::vector<int>{b, split+c, a}.data());
        }

    // 4. DFL decoding → (B, 4, A)
    cv::Mat dfl_out = dfl_decode(box_raw, ch_);

    // 5. dist2bbox → cxcywh, scaled to image space
    //    a = anchors[np.newaxis] - dfl_out[:, :2, :]
    //    b = anchors[np.newaxis] + dfl_out[:, 2:, :]
    cv::Mat box(std::vector<int>{B, 4, A}, CV_32F);
    for (int b = 0; b < B; ++b)
        for (int a = 0; a < A; ++a) {
            float lt_x = anchors_T.at<float>(0, a)
                         - dfl_out.at<float>(std::vector<int>{b,0,a}.data());
            float lt_y = anchors_T.at<float>(1, a)
                         - dfl_out.at<float>(std::vector<int>{b,1,a}.data());
            float rb_x = anchors_T.at<float>(0, a)
                         + dfl_out.at<float>(std::vector<int>{b,2,a}.data());
            float rb_y = anchors_T.at<float>(1, a)
                         + dfl_out.at<float>(std::vector<int>{b,3,a}.data());
            float stride = sv_T.at<float>(0, a);
            box.at<float>(std::vector<int>{b,0,a}.data()) = (lt_x + rb_x) * 0.5f * stride; // cx
            box.at<float>(std::vector<int>{b,1,a}.data()) = (lt_y + rb_y) * 0.5f * stride; // cy
            box.at<float>(std::vector<int>{b,2,a}.data()) = (rb_x - lt_x) * stride;         // w
            box.at<float>(std::vector<int>{b,3,a}.data()) = (rb_y - lt_y) * stride;         // h
        }

    // 6. Class sigmoid
    cv::Mat cls_prob(std::vector<int>{B, nc_, A}, CV_32F);
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < nc_; ++c)
            for (int a = 0; a < A; ++a) {
                float v = cls_raw.at<float>(std::vector<int>{b,c,a}.data());
                cls_prob.at<float>(std::vector<int>{b,c,a}.data())
                    = 1.f / (1.f + std::exp(-v));
            }

    // 7. Concatenate box + cls → (B, 4+nc, A)
    cv::Mat output(std::vector<int>{B, 4 + nc_, A}, CV_32F);
    for (int b = 0; b < B; ++b)
        for (int a = 0; a < A; ++a) {
            for (int c = 0; c < 4; ++c)
                output.at<float>(std::vector<int>{b,c,a}.data())
                    = box.at<float>(std::vector<int>{b,c,a}.data());
            for (int c = 0; c < nc_; ++c)
                output.at<float>(std::vector<int>{b, 4+c, a}.data())
                    = cls_prob.at<float>(std::vector<int>{b,c,a}.data());
        }

    return output;
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