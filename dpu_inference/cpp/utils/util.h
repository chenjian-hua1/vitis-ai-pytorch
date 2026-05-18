#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <functional>
#include <array>

#include <opencv2/opencv.hpp>

// #define ONNX_MODE
#ifndef XMODEL_MODE
#define ONNX_MODE
#endif

// ============================================================================
//  Data Structures
// ============================================================================

struct AnchorResult {
    cv::Mat anchors;       ///< (A, 2) float32 — anchor (x, y) in feature-map space
    cv::Mat stride_tensor; ///< (A, 1) float32 — stride per anchor
};

struct ResizeResult {
    cv::Mat img;
    std::pair<float,float> ratio;
    std::pair<float,float> pad;
};

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int   class_id;
};

// ============================================================================
//  Fix / Float Conversion
// ============================================================================

void fix2float(const cv::Mat& data, int fix_point, cv::Mat& out);
void float2fix(const cv::Mat& data, int fix_point, cv::Mat& out);

// ============================================================================
//  Bounding Box Utilities
// ============================================================================

cv::Mat wh2xy(const cv::Mat& x);
cv::Mat xyxy2xywh(const cv::Mat& box);

// ============================================================================
//  Pre-Processing
// ============================================================================

void norm(const cv::Mat& x, cv::Mat& out);
void resize(const cv::Mat& img, int input_size, ResizeResult& res);
/**
 * @brief 將 uint8 BGR 影像做 ImageNet 正規化並量化為 int8 fix-point。
 *
 * 對每個 channel c 執行：
 *   dst[c] = clamp( src[c] * (kU8Scale[c] * 2^fix_point)
 *                           + kU8Bias[c]  * 2^fix_point,
 *                   -128, 127 )
 *
 * 等價於先做 norm（減 mean、除 std）再做 float2fix，
 * 但合併成單一 pass 避免中間 float buffer。
 *
 * @param x          輸入影像，必須為 CV_8UC3 (BGR)
 * @param fix_point  DPU input tensor 的 fix-point exponent
 *                   (從 xir::Tensor::get_attr<int>("fix_point") 取得)
 * @param out        輸出影像，CV_8SC3，與 x 同尺寸
 */
void norm_and_fix(const cv::Mat& x, int fix_point, cv::Mat& out);

// ============================================================================
//  YOLO Post-Processor Class (NCHW input)
// ============================================================================
//
//  Layout：本實作要求所有 feature_maps 為 NCHW (B, no_, H, W)。
//
//  歷史筆記：曾經實驗過 NHWC layout 想吃 cache locality 紅利（classify/DFL
//  的 inner loop 變 sequential），但實測在 ARM Cortex-A + DPU 場景下反而慢：
//    1) 早期過濾 (conf=0.1) 後 active anchor 比例極低，classify 的 outer
//       loop 雖然全跑但 inner 只是純比較、沒有 expf，本來就不是瓶頸；
//    2) PostProcessor 整體時間佔比小，比不過 DPU sync + dequantize，layout
//       優化的差距落在量測噪音之下；
//    3) NHWC 下 inner loop 從「預算 row pointer」變成「每 anchor 算 offset
//       (a * no_ + ...)」，多了乘加，吃掉了 sequential read 的便宜。
//  教訓：cache locality 紅利只有在 hot loop 真正 cache-bound 時才會兌現。
//

struct DetectionBatch {
    std::vector<Detection> data;
    int                    count;

    Detection*       begin()       { return data.data(); }
    Detection*       end()         { return data.data() + count; }
    const Detection* begin() const { return data.data(); }
    const Detection* end()   const { return data.data() + count; }
    int              size()  const { return count; }
};

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

    // ── anchor 表 ──
    int     A_ = 0;
    cv::Mat anchors_T_;                 // (2, A)
    cv::Mat stride_T_;                  // (1, A)
    std::vector<int> hw_per_scale_;
    std::vector<int> scale_offsets_;

    // ── pipeline buffer (NCHW) ──
    cv::Mat x_cat_, dfl_out_, output_;
    cv::Mat box_raw_view_, cls_raw_view_;

    // ── 預算的 raw pointer 表 ──
    const float* ax_ = nullptr;
    const float* ay_ = nullptr;
    const float* sv_ = nullptr;

    std::vector<float*> output_cx_rows_;
    std::vector<float*> output_cy_rows_;
    std::vector<float*> output_w_rows_;
    std::vector<float*> output_h_rows_;
    std::vector<std::vector<float*>> output_cls_rows_;

    std::vector<std::array<float*, 4>> dfl_rows_;

    std::vector<std::vector<const float*>> box_raw_rows_;
    std::vector<std::vector<const float*>> cls_raw_rows_;

    std::vector<std::vector<float*>> xcat_rows_;

    // ── NMS 相關 ──
    std::vector<std::array<float, 6>> candidates_;
    int                               cand_count_ = 0;
    std::vector<cv::Rect2d>           boxes_cv_;
    std::vector<float>                scores_cv_;
    std::vector<int>                  indices_;
    std::vector<DetectionBatch>       detections_;

    // ── 早期過濾 ──
    float conf_thresh_cached_ = -1.f;
    std::vector<std::vector<int>>    active_indices_;
    std::vector<std::vector<float>>  active_max_score_;
    std::vector<std::vector<int>>    active_max_cls_;
    std::vector<std::vector<float>>  active_cls_scores_;

    std::vector<float> class_offsets_;
    std::vector<float> dfl_arange_;

    void precompute_anchors();
    void precompute_tables();
    void allocate_buffers();
    void bind_views();
    void cache_pointers();

    void classify_and_build_mask(float conf_thresh);
    void dfl_decode_masked();
    void dist2bbox_masked();
    void nms_single_batch(int b, float conf_thresh, float iou_thresh);
};


// ============================================================================
//  Visualization
// ============================================================================

cv::Mat scale_boxes(const cv::Mat&              boxes,
                    std::pair<float,float>       ratio,
                    std::pair<float,float>       pad,
                    cv::Size                     orig_shape);

cv::Mat draw_boxes(const cv::Mat&                    img,
                   const std::vector<Detection>&     detections,
                   const std::vector<std::string>&   class_names = {});


#ifdef ONNX_MODE

#include <onnxruntime_cxx_api.h>

class OnnxInferenceEngine {
public:
    using ConfigureOptionsFn = std::function<void(Ort::SessionOptions&)>;

    explicit OnnxInferenceEngine(const std::string& model_path,
                                 ConfigureOptionsFn configure_options = {});

    void run(const cv::Mat& input_img);

    const std::vector<cv::Mat>& output_mats() const { return outputs_; }
    const cv::Mat& output_mat_nchw(size_t idx) const { return outputs_.at(idx); }

    int64_t in_c() const { return ch_; }
    int64_t in_h() const { return in_h_; }
    int64_t in_w() const { return in_w_; }
    size_t  num_outputs() const { return outputs_.size(); }

private:
    Ort::Env                          env_;
    std::unique_ptr<Ort::Session>     session_;
    Ort::AllocatorWithDefaultOptions  allocator_;

    std::vector<Ort::AllocatedStringPtr> allocated_strings_;
    std::vector<const char*>             input_names_;
    std::vector<const char*>             output_names_;

    int64_t            ch_   = 0;
    int64_t            in_h_ = 0;
    int64_t            in_w_ = 0;
    std::vector<float> input_tensor_values_;
    std::vector<int64_t> input_shape_;

    std::vector<cv::Mat>              outputs_;
    std::vector<std::vector<int64_t>> output_shapes_;

    void initialize_model_info();
    void hwc_to_nchw(const cv::Mat& src);
};

#endif


#ifdef XMODEL_MODE

namespace xir  { class Graph; class Attrs; class Tensor; }
namespace vart { class RunnerExt; class TensorBuffer; }

/**
 *  Xmodel 引擎：DPU 原生輸出為 NHWC int8。本引擎在 run() 結束時做一次
 *  NHWC → NCHW 轉置（int8 element-wise 搬移），把 NCHW int8 暴露給外部用。
 *  PostProcessor 需要 NCHW float32，呼叫端再用 fix2float() 反量化即可
 *  （fix2float 只做 element-wise 縮放，不改 layout）。
 */
class XmodelInferenceEngine {
public:
    explicit XmodelInferenceEngine(const std::string& xmodel_path);
    ~XmodelInferenceEngine();

    XmodelInferenceEngine(const XmodelInferenceEngine&)            = delete;
    XmodelInferenceEngine& operator=(const XmodelInferenceEngine&) = delete;

    void bind_input_mat(cv::Mat& ext_input) const { ext_input = input_mat_; }

    /**
     * @brief 取得轉置後的 NCHW int8 輸出（PostProcessor 友善 layout）。
     */
    const cv::Mat& output_mat_nchw(size_t idx) const { return outputs_nchw_.at(idx); }

    void run();

    int    in_c() const { return in_c_; }
    int    in_h() const { return in_h_; }
    int    in_w() const { return in_w_; }
    size_t num_outputs() const { return outputs_nchw_.size(); }
    float  input_scale() const { return input_scale_; }
    float  output_scale(size_t i) const { return output_scales_.at(i); }

private:
    std::unique_ptr<xir::Graph>       graph_;
    std::unique_ptr<xir::Attrs>       attrs_;
    std::unique_ptr<vart::RunnerExt>  runner_;

    std::vector<vart::TensorBuffer*>  input_tensor_buffers_;
    std::vector<vart::TensorBuffer*>  output_tensor_buffers_;
    std::vector<std::vector<int8_t>> output_cache_buffers_;

    int   in_c_ = 0, in_h_ = 0, in_w_ = 0;
    float input_scale_ = 1.0f;

    cv::Mat               input_mat_;
    std::vector<cv::Mat>  outputs_;        // DPU 實體記憶體替身 (NHWC int8)
    std::vector<cv::Mat>  outputs_nchw_;   // 預建 NCHW int8 buffer
    std::vector<float>    output_scales_;

    void initialize_model_info(const std::string& xmodel_path);
};

#endif