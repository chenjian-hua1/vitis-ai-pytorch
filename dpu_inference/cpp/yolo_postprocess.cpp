/*
 * yolo_postprocess.cpp
 * ───────────────────────────────────────────────────────────────────────────
 * Post-processing implementation:
 *   letterbox / DFL decode / dist2bbox / NMS / visualization
 */

#include "yolo_custom.hpp"

#include <numeric>
#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>      // NMSBoxes


// ═══════════════════════════════════════════════════════════════════════════
//  Letterbox
// ═══════════════════════════════════════════════════════════════════════════
cv::Mat letterbox(const cv::Mat& bgr, int target_w, int target_h,
                  LetterboxInfo& info)
{
    info.orig_w = bgr.cols;
    info.orig_h = bgr.rows;
    info.scale  = std::min(static_cast<float>(target_h) / info.orig_h,
                           static_cast<float>(target_w) / info.orig_w);

    int new_w = static_cast<int>(info.orig_w * info.scale);
    int new_h = static_cast<int>(info.orig_h * info.scale);

    info.pad_left = (target_w - new_w) / 2;
    info.pad_top  = (target_h - new_h) / 2;

    // BGR -> RGB, resize
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, {new_w, new_h}, 0, 0, cv::INTER_LINEAR);

    // Gray canvas (114)
    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    rgb.copyTo(canvas(cv::Rect(info.pad_left, info.pad_top, new_w, new_h)));
    return canvas;   // RGB uint8, HWC
}


// ═══════════════════════════════════════════════════════════════════════════
//  DFL Softmax Decode
//  box_ptr  : float[4 * DFL_CH]  — raw box output for a single anchor
//  dist_out : float[4]           — output [l, t, r, b] distances
// ═══════════════════════════════════════════════════════════════════════════
static void dfl_decode_one(const float* box_ptr,
                            int dfl_ch,
                            float* dist_out)
{
    // box_ptr layout: [d0_0..d0_{ch-1}, d1_0..d1_{ch-1}, d2_0..d2_{ch-1}, d3_0..d3_{ch-1}]
    for (int side = 0; side < 4; ++side) {
        const float* src = box_ptr + side * dfl_ch;

        // Numerically stable softmax
        float max_val = *std::max_element(src, src + dfl_ch);
        float sum = 0.f;
        std::vector<float> e(dfl_ch);
        for (int k = 0; k < dfl_ch; ++k) {
            e[k] = std::exp(src[k] - max_val);
            sum += e[k];
        }
        // Expected value: sum_k( k * p_k )
        float dist = 0.f;
        for (int k = 0; k < dfl_ch; ++k)
            dist += k * (e[k] / sum);

        dist_out[side] = dist;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Single scale decode
//  feat_ptr : (feat_h, feat_w, OUT_CH) — float, NHWC without batch dim
// ═══════════════════════════════════════════════════════════════════════════
void decode_scale(const float* feat_ptr,
                  int feat_h, int feat_w,
                  int stride,
                  float conf_thresh,
                  std::vector<Detection>& out_dets)
{
    const int dfl_ch = cfg::DFL_CH;
    const int box_ch = dfl_ch * 4;   // 64
    const int nc     = cfg::NC;      // 4
    const int out_ch = cfg::OUT_CH;  // 68
    const int n      = feat_h * feat_w;

    for (int idx = 0; idx < n; ++idx) {
        const float* cell = feat_ptr + idx * out_ch;

        // ── Class sigmoid + argmax ────────────────────────────────────────
        const float* cls_ptr = cell + box_ch;
        float max_logit = cls_ptr[0];
        int   cls_id    = 0;
        for (int c = 1; c < nc; ++c) {
            if (cls_ptr[c] > max_logit) {
                max_logit = cls_ptr[c];
                cls_id    = c;
            }
        }
        float conf = 1.f / (1.f + std::exp(-max_logit));  // sigmoid
        if (conf < conf_thresh) continue;

        // ── Anchor center point (pixel space) ────────────────────────────
        int   gy = idx / feat_w;
        int   gx = idx % feat_w;
        float cx = (gx + 0.5f) * stride;
        float cy = (gy + 0.5f) * stride;

        // ── DFL decode -> ltrb ────────────────────────────────────────────
        float dist[4];
        dfl_decode_one(cell, dfl_ch, dist);   // dist in grid units
        // Scale to pixel distances
        float l = dist[0] * stride;
        float t = dist[1] * stride;
        float r = dist[2] * stride;
        float b = dist[3] * stride;

        // ── dist2bbox -> xyxy ─────────────────────────────────────────────
        Detection det;
        det.x1     = cx - l;
        det.y1     = cy - t;
        det.x2     = cx + r;
        det.y2     = cy + b;
        det.conf   = conf;
        det.cls_id = cls_id;
        out_dets.push_back(det);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  NMS  (OpenCV NMSBoxes)
// ═══════════════════════════════════════════════════════════════════════════
std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thresh)
{
    if (dets.empty()) return {};

    std::vector<cv::Rect2d> boxes;
    std::vector<float>      scores;
    boxes.reserve(dets.size());
    scores.reserve(dets.size());

    for (const auto& d : dets) {
        // NMSBoxes expects xywh format
        boxes.emplace_back(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
        scores.push_back(d.conf);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores,
                      cfg::CONF_THRESH, iou_thresh,
                      indices);

    std::vector<Detection> result;
    result.reserve(indices.size());
    for (int i : indices)
        result.push_back(dets[i]);
    return result;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Restore coordinates from letterbox space to original image space
// ═══════════════════════════════════════════════════════════════════════════
void restore_coords(std::vector<Detection>& dets, const LetterboxInfo& info)
{
    for (auto& d : dets) {
        d.x1 = std::clamp((d.x1 - info.pad_left) / info.scale, 0.f, (float)info.orig_w);
        d.y1 = std::clamp((d.y1 - info.pad_top)  / info.scale, 0.f, (float)info.orig_h);
        d.x2 = std::clamp((d.x2 - info.pad_left) / info.scale, 0.f, (float)info.orig_w);
        d.y2 = std::clamp((d.y2 - info.pad_top)  / info.scale, 0.f, (float)info.orig_h);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  Draw detections
// ═══════════════════════════════════════════════════════════════════════════
void draw_detections(cv::Mat& frame, const std::vector<Detection>& dets)
{
    for (const auto& d : dets) {
        int x1 = static_cast<int>(d.x1);
        int y1 = static_cast<int>(d.y1);
        int x2 = static_cast<int>(d.x2);
        int y2 = static_cast<int>(d.y2);

        const cv::Scalar& color =
            cfg::PALETTE[d.cls_id % (int)cfg::PALETTE.size()];

        cv::rectangle(frame, {x1, y1}, {x2, y2}, color, 2);

        const std::string& name =
            (d.cls_id < (int)cfg::CLASS_NAMES.size())
            ? cfg::CLASS_NAMES[d.cls_id]
            : "cls" + std::to_string(d.cls_id);

        std::ostringstream oss;
        oss << name << " " << std::fixed << std::setprecision(2) << d.conf;
        std::string label = oss.str();

        int      baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                      0.55, 1, &baseline);
        int ty = std::max(y1 - 4, ts.height + 4);
        cv::rectangle(frame,
                      {x1, ty - ts.height - 3},
                      {x1 + ts.width, ty + baseline},
                      color, cv::FILLED);
        cv::putText(frame, label, {x1, ty},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}
