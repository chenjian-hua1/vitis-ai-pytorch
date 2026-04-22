#pragma once

#include <vector>
#include <string>
#include <tuple>

#include <opencv2/opencv.hpp>

// ============================================================================
//  Data Structures
// ============================================================================

/**
 * @brief Result of make_anchors(): anchor (x,y) coordinates and stride values.
 */
struct AnchorResult {
    cv::Mat anchors;       ///< (A, 2) float32 — anchor (x, y) in feature-map space
    cv::Mat stride_tensor; ///< (A, 1) float32 — stride per anchor
};

/**
 * @brief Result of resize(): resized image plus metadata.
 */
struct ResizeResult {
    cv::Mat img;           ///< Resized + padded image
    std::pair<float,float> ratio; ///< (r, r) scale ratio
    std::pair<float,float> pad;   ///< (w, h) padding in pixels
};

/**
 * @brief Single detection: (x1, y1, x2, y2, score, class_id).
 */
struct Detection {
    float x1, y1, x2, y2;
    float score;
    int   class_id;
};

// ============================================================================
//  Fix / Float Conversion
// ============================================================================

/**
 * @brief Convert int8 fixed-point array to float32.
 *
 * @param data      CV_8S Mat of fixed-point values
 * @param fix_point Exponent (scale = 2^(-fix_point))
 * @param out       CV_32F Mat reused across calls to avoid repeated allocation.
 */
void fix2float(const cv::Mat& data, int fix_point, cv::Mat& out);

/**
 * @brief Convert float32 array to int8 fixed-point.
 *
 * @param data      CV_32F Mat
 * @param fix_point Exponent (scale = 2^fix_point)
 * @param out       CV_8S Mat clipped to [-128, 127],
 *                  reused across calls to avoid repeated allocation.
 */
void float2fix(const cv::Mat& data, int fix_point, cv::Mat& out);

// ============================================================================
//  Bounding Box Utilities
// ============================================================================

/**
 * @brief Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2).
 *
 * @param x CV_32F Mat of shape (N, 4+)
 * @return  CV_32F Mat of same shape, first 4 columns converted
 */
cv::Mat wh2xy(const cv::Mat& x);

/**
 * @brief Convert boxes from (x1, y1, x2, y2) to (x, y, w, h).
 *
 * @param box CV_32F Mat of shape (N, 4)
 * @return    CV_32F Mat of shape (N, 4)
 */
cv::Mat xyxy2xywh(const cv::Mat& box);

// ============================================================================
//  Pre-Processing
// ============================================================================

/**
 * @brief Normalize an image using ImageNet mean/std statistics.
 *
 * Performs per-channel linear transformation:
 *   out[c] = (x[c] / (std[c] * 255)) - (mean[c] / std[c])
 *
 * ImageNet constants:
 *   mean = [0.485, 0.456, 0.406]  (R, G, B)
 *   std  = [0.229, 0.224, 0.225]  (R, G, B)
 *
 * @param x   Input image with the following constraints:
 *              - Shape  : (H, W, 3), must be continuous in memory
 *              - Depth  : CV_8U [0, 255]
 *              - Channel: 3 (BGR or RGB, channel order must match
 *                            the order used during model training)
 * @param out Output image reused across calls to avoid repeated allocation.
 *              - Depth  : CV_32F
 *              - Shape  : same as input (H, W, 3)
 *              - Range  : approximately [-2.5, 2.5] per channel
 *
 * @throws cv::Exception if x is not CV_8U, not 3-channel, or not continuous
 *
 * @note Channel order (BGR vs RGB) is NOT checked internally.
 *       Caller is responsible for ensuring correct channel ordering
 *       before passing to this function.
 */
void norm(const cv::Mat& x, cv::Mat& out);

/**
 * @brief Letterbox-resize an image to a square of side `input_size`.
 *
 * Only downscales (r <= 1.0). Pads with black borders to reach exact size.
 *
 * @param img        Input BGR image
 * @param input_size Target side length (e.g. 640)
 * @param res        ResizeResult reused across calls to avoid repeated allocation,
 *                   containing output image, ratio, and padding.
 */
void resize(const cv::Mat& img, int input_size, ResizeResult& res);

// ============================================================================
//  YOLO Post-Processor Class
// ============================================================================
 
struct DetectionBatch {
    std::vector<Detection> data;
    int                    count;
 
    Detection*       begin()       { return data.data(); }
    Detection*       end()         { return data.data() + count; }
    const Detection* begin() const { return data.data(); }
    const Detection* end()   const { return data.data() + count; }
    int              size()  const { return count; }
};
 
/**
 * @brief YOLO 後處理 + NMS 一體化管線。
 *
 * 建構時預算所有與模型尺寸相關的不變量：
 *   - anchors / stride 座標表
 *   - pipeline tensor buffer（x_cat, dfl_out, output）
 *   - NMS 暫存 buffer（固定大小，用 count 控制有效長度）
 *   - 所有 tensor 的 row pointer 快取（O(B·C) 個 float*）
 *   - concat 用的 scale offset 表
 *   - NMS 用的 class offset 表（避免重複 class_id × max_wh）
 *   - DFL 的 arange 權重
 *
 * 推理期 operator()/decode()/nms() 完全不動態配置記憶體。
 */
class YOLOPostProcessor {
public:
    YOLOPostProcessor(int batch,
                      int input_h,
                      int input_w,
                      int nc       = 80,
                      int ch       = 16,
                      std::vector<int> strides = {8, 16, 32},
                      int max_nms  = 5000,
                      int max_det  = 100);
 
    const std::vector<DetectionBatch>& process(
        const std::vector<cv::Mat>& feature_maps,
        float confidence_threshold = 0.25f,
        float iou_threshold        = 0.45f);

    // 早期過濾式 decode：
    //   先做 classify + mask（logit-space 比較），產生 active anchor table。
    //   DFL decode / dist2bbox 都只對 active anchor 計算，低分框完全略過。
    //
    // conf_thresh 必須 > 0；使用 conf_thresh=0 會讓所有 anchor 通過過濾，
    // 等同退化為全量 decode（不建議這樣用，失去本設計初衷）。
    const cv::Mat& decode(const std::vector<cv::Mat>& feature_maps,
                          float conf_thresh);
    void           nms(float confidence_threshold, float iou_threshold);
 
    const std::vector<DetectionBatch>& detections() const { return detections_; }
    const cv::Mat& raw_output()                    const { return output_; }
    int            total_anchors()                 const { return A_; }
 
private:
    // ── 固定參數 ──
    int B_, input_h_, input_w_, nc_, ch_, no_;
    std::vector<int> strides_;
    int max_nms_, max_det_;
 
    // ── anchor 表（一次算，終身用）──
    int     A_ = 0;
    cv::Mat anchors_T_;                 // (2, A)
    cv::Mat stride_T_;                  // (1, A)
    std::vector<int> hw_per_scale_;     // 每個 scale 的 H*W
    std::vector<int> scale_offsets_;    // ③ concat 起點 offset
 
    // ── pipeline buffer ──
    cv::Mat x_cat_, dfl_out_, output_;
    cv::Mat box_raw_view_, cls_raw_view_;
 
    // ── 預算的 raw pointer 表 ──
    // anchors / stride（1D）
    const float* ax_ = nullptr;
    const float* ay_ = nullptr;
    const float* sv_ = nullptr;
 
    // ① output 各 channel 在每個 batch 的 row pointer
    //     output_cx_rows_[b] 指向 output_[b, 0, :]
    std::vector<float*> output_cx_rows_;       // (B,)
    std::vector<float*> output_cy_rows_;
    std::vector<float*> output_w_rows_;
    std::vector<float*> output_h_rows_;
    std::vector<std::vector<float*>> output_cls_rows_;  // (B, nc)
 
    // ② dfl_out 各方向的 row pointer（(B, 4)）
    std::vector<std::array<float*, 4>> dfl_rows_;        // dfl_rows_[b][coord]
 
    // ② box_raw / cls_raw 的 row pointer
    //     box_raw_rows_[b][coord*ch + c]
    std::vector<std::vector<const float*>> box_raw_rows_;   // (B, 4*ch)
    std::vector<std::vector<const float*>> cls_raw_rows_;   // (B, nc)
 
    // x_cat 的 row pointer（concat 用）
    std::vector<std::vector<float*>> xcat_rows_;            // (B, no)
 
    // ── NMS 相關 ──
    std::vector<std::array<float, 6>> candidates_;
    int                               cand_count_ = 0;
    std::vector<cv::Rect2d>           boxes_cv_;
    std::vector<float>                scores_cv_;
    std::vector<int>                  indices_;
    std::vector<DetectionBatch>       detections_;

    // ── 早期過濾：classify → threshold → 產生 active anchor table ──
    // 大多數 anchor 的 max class logit 都會低於 conf_thresh（訓練後 YOLO
    // 的背景 anchor logits 通常 << 0），可跳過後續的 DFL decode /
    // dist2bbox / NMS 掃描，省下大量 expf 計算。
    float conf_thresh_cached_ = -1.f;       // decode() 時暫存當次門檻
    std::vector<std::vector<int>>    active_indices_;     // (B,) 每 batch 的 active anchor index
    std::vector<std::vector<float>>  active_max_score_;   // (B,) 對應 max cls score（sigmoid 後）
    std::vector<std::vector<int>>    active_max_cls_;     // (B,) 對應 argmax class id
    // 若 nc > 1 且要支援 multi-label NMS（同 anchor 多 class 都過門檻），
    // 則再用 per-active-anchor 的 sigmoid buffer。這裡 lazy-allocate。
    std::vector<std::vector<float>>  active_cls_scores_;  // (B,) flat, size = n_active * nc
 
    // ④ class_id × max_wh 的 offset 表
    std::vector<float> class_offsets_;   // (nc,)
 
    // ⑤ DFL 的 arange 權重（0..ch-1）
    std::vector<float> dfl_arange_;      // (ch,)
 
    // ── 初始化 ──
    void precompute_anchors();
    void precompute_tables();            // ③④⑤
    void allocate_buffers();
    void bind_views();
    void cache_pointers();               // ①②
    void cache_row_pointers_for_fmap(const std::vector<cv::Mat>& x);
 
    // ── pipeline ──
    void classify_and_build_mask(float conf_thresh);   // logit max → active list
    void dfl_decode_masked();                          // 只處理 active anchor
    void dist2bbox_masked();                           // 只處理 active anchor
    void nms_single_batch(int b, float conf_thresh, float iou_thresh);
};


// ============================================================================
//  Visualization
// ============================================================================

/**
 * @brief Scale boxes from resized/padded image back to original coordinates.
 *
 * @param boxes      (N, 4) CV_32F in (x1, y1, x2, y2)
 * @param ratio      (r, r) from ResizeResult
 * @param pad        (w, h) padding from ResizeResult
 * @param orig_shape Original image size as cv::Size (width, height)
 * @return           (N, 4) CV_32F boxes in original image coordinates
 */
cv::Mat scale_boxes(const cv::Mat&              boxes,
                    std::pair<float,float>       ratio,
                    std::pair<float,float>       pad,
                    cv::Size                     orig_shape);

/**
 * @brief Draw bounding boxes on an image.
 *
 * @param img          BGR image (H, W, 3) CV_8UC3
 * @param detections   Vector of Detection structs
 * @param class_names  Optional class name list
 * @return             Copy of image with boxes drawn
 */
cv::Mat draw_boxes(const cv::Mat&                    img,
                   const std::vector<Detection>&     detections,
                   const std::vector<std::string>&   class_names = {});