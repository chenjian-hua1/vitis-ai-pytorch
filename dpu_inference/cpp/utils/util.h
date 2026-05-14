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

// ============================================================================
//  YOLO Post-Processor Class (NHWC input)
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
 * @brief YOLO 後處理 + NMS 一體化管線（NHWC 輸入版本）。
 *
 *  輸入假設：feature_maps[i] 為 4D cv::Mat，shape = (B, H_i, W_i, no_)，
 *  其中 no_ = 4*ch_ + nc_。前 4*ch_ 個 channel 是 DFL box logits，
 *  後 nc_ 個 channel 是 class logits。
 *
 *  內部 x_cat_ 也是 NHWC: (B, A, no_)。同一個 anchor 的 no_ 個 channel
 *  記憶體完全連續，classify 與 DFL 的 inner loop 都是 sequential read，
 *  編譯器可向量化、cache 友善。
 *
 *  output_ 與 dfl_out_ 維持 NCHW，因為 dist2bbox / NMS 是逐 channel 寫/讀，
 *  NCHW 對它們較友善。這兩個 buffer 都只在 PostProcessor 內部流轉。
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
    std::vector<int> scale_offsets_;    // concat 起點（以 anchor 為單位）

    // ── pipeline buffer ──
    //  x_cat_   : (B, A, no_)        NHWC，concat 結果
    //  dfl_out_ : (B, 4, A)          NCHW，DFL 解碼後的距離值
    //  output_  : (B, 4 + nc_, A)    NCHW，最終 (cx, cy, w, h, scores...) — 後 nc_ 預留但目前未使用
    cv::Mat x_cat_, dfl_out_, output_;

    // ── 預算的 raw pointer 表 ──
    // anchors / stride（1D）
    const float* ax_ = nullptr;
    const float* ay_ = nullptr;
    const float* sv_ = nullptr;

    // ① output_ 各 channel 的 row pointer（NCHW，沿用舊邏輯）
    std::vector<float*> output_cx_rows_;
    std::vector<float*> output_cy_rows_;
    std::vector<float*> output_w_rows_;
    std::vector<float*> output_h_rows_;
    std::vector<std::vector<float*>> output_cls_rows_;

    // ② dfl_out_ 各方向的 row pointer（NCHW）
    std::vector<std::array<float*, 4>> dfl_rows_;

    // ③ x_cat_ 各 batch 的起點 pointer（NHWC）
    //    第 b batch 的第 a anchor 第 c channel 位於：
    //      xcat_base_[b] + a * no_ + c
    std::vector<float*> xcat_base_;

    // ── NMS 相關 ──
    std::vector<std::array<float, 6>> candidates_;
    int                               cand_count_ = 0;
    std::vector<cv::Rect2d>           boxes_cv_;
    std::vector<float>                scores_cv_;
    std::vector<int>                  indices_;
    std::vector<DetectionBatch>       detections_;

    // ── 早期過濾：classify → threshold → active anchor table ──
    float conf_thresh_cached_ = -1.f;
    std::vector<std::vector<int>>    active_indices_;
    std::vector<std::vector<float>>  active_max_score_;
    std::vector<std::vector<int>>    active_max_cls_;
    std::vector<std::vector<float>>  active_cls_scores_;  // flat: i * nc + c

    // ④ class_id × max_wh 的 offset 表
    std::vector<float> class_offsets_;

    // ⑤ DFL 的 arange 權重（0..ch-1）
    std::vector<float> dfl_arange_;

    // ── 初始化 ──
    void precompute_anchors();
    void precompute_tables();
    void allocate_buffers();
    void cache_pointers();

    // ── pipeline ──
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

// ============================================================================
//  ONNX Runtime Inference Engine
// ============================================================================
//
//  輸出 shape 接受任何 4D tensor。若 ONNX 模型 export 成 NHWC 輸出
//  (1, H, W, C)，這裡會直接以 NHWC 配 cv::Mat，PostProcessor 也接受 NHWC，
//  整條鏈路不再有任何 layout 轉置。
//
class OnnxInferenceEngine {
public:
    using ConfigureOptionsFn = std::function<void(Ort::SessionOptions&)>;

    explicit OnnxInferenceEngine(const std::string& model_path,
                                 ConfigureOptionsFn configure_options = {});

    void run(const cv::Mat& input_img);

    const std::vector<cv::Mat>& output_mats() const { return outputs_; }
    const cv::Mat& output_mat(size_t idx) const { return outputs_.at(idx); }

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
 *  Xmodel 引擎：DPU 原生輸出就是 NHWC，PostProcessor 也接受 NHWC，
 *  因此本引擎不再做任何 NHWC→NCHW 轉置，輸出直接是 DPU 記憶體上的 NHWC int8。
 *
 *  注意：輸出為 int8 (CV_8S)，PostProcessor 需要 float32 (CV_32F)。
 *  呼叫端需用 fix2float() 反量化（不改變 layout，只 element-wise 縮放）。
 */
class XmodelInferenceEngine {
public:
    explicit XmodelInferenceEngine(const std::string& xmodel_path);
    ~XmodelInferenceEngine();

    XmodelInferenceEngine(const XmodelInferenceEngine&)            = delete;
    XmodelInferenceEngine& operator=(const XmodelInferenceEngine&) = delete;

    void bind_input_mat(cv::Mat& ext_input) const { ext_input = input_mat_; }

    /**
     * @brief 取得 DPU 原生的 NHWC int8 輸出。Shape = (1, H, W, C)。
     */
    const cv::Mat& output_mat(size_t idx) const { return outputs_.at(idx); }

    void run();

    int    in_c() const { return in_c_; }
    int    in_h() const { return in_h_; }
    int    in_w() const { return in_w_; }
    size_t num_outputs() const { return outputs_.size(); }
    float  input_scale() const { return input_scale_; }
    float  output_scale(size_t i) const { return output_scales_.at(i); }

private:
    std::unique_ptr<xir::Graph>       graph_;
    std::unique_ptr<xir::Attrs>       attrs_;
    std::unique_ptr<vart::RunnerExt>  runner_;

    std::vector<vart::TensorBuffer*>  input_tensor_buffers_;
    std::vector<vart::TensorBuffer*>  output_tensor_buffers_;

    int   in_c_ = 0, in_h_ = 0, in_w_ = 0;
    float input_scale_ = 1.0f;

    cv::Mat               input_mat_;
    std::vector<cv::Mat>  outputs_;        // DPU 實體記憶體替身 (NHWC, int8)
    std::vector<float>    output_scales_;

    void initialize_model_info(const std::string& xmodel_path);
};

#endif