#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <functional>

#include <opencv2/opencv.hpp>

// #define ONNX_MODE
#define XMODEL_MODE

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
 * @brief Convert int8 fixed-point array to float32, preserving channel count.
 *
 * @param data      Fixed-point input (CV_8SC1 or CV_8SC3)
 * @param fix_point Exponent (scale = 2^(-fix_point))
 * @param out       Float32 output, same channel count as input,
 *                  reused across calls to avoid repeated allocation.
 *
 * @note  簽章刻意使用 cv::Mat&（不是 cv::Mat_<float>&）。Mat_<T> 是 OpenCV 的
 *        「強制單通道 typed view」，當 OutputArray 被要求 reshape 時會把 3-channel
 *        資料重新詮釋為 CV_32FC1 / cols*3。下游若要嚴格 assert CV_32FC3 就會失敗。
 *        使用 cv::Mat& 搭配完整型別 (CV_32FC3) 可以保留正確的 channel header。
 */
void fix2float(const cv::Mat& data, int fix_point, cv::Mat& out);

/**
 * @brief Convert float32 array to int8 fixed-point, preserving channel count.
 *
 * @param data      Float32 input (CV_32FC1 or CV_32FC3)
 * @param fix_point Exponent (scale = 2^fix_point)
 * @param out       Fixed-point output, same channel count as input,
 *                  clipped to [-128, 127], reused across calls.
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



#ifdef ONNX_MODE

#include <onnxruntime_cxx_api.h>

// ============================================================================
//  ONNX Runtime Inference Engine
// ============================================================================
//
//  一個 session 對應一顆模型。整個類別只配一次記憶體：
//    - input_tensor_values_ : NCHW float buffer（HWC→NCHW 轉換目的地）
//    - outputs_             : 每個輸出一個 cv::Mat (4D NCHW)，ORT tensor 直接
//                             綁在 Mat 的底層 buffer 上，推理完不用 copy。
//
//  呼叫方式：
//      OnnxInferenceEngine engine("model.onnx");
//      engine.run(float_img);                         // 跑一次
//      const auto& fmaps = engine.output_mats();      // 可直接丟 YOLO 後處理
//
//  假設：
//    - 模型為單一輸入，shape = (1, 3, H, W)，dtype = float32。
//    - 輸出為任意數量的 4D tensor（batch=1），dtype = float32。
//  若需要 int8 輸入或多輸入，請另行擴充。
//
class OnnxInferenceEngine {
public:
    /**
     * @brief 載入 ONNX 模型並配置所有推理用 buffer。
     *
     * @param model_path        ONNX 檔案路徑
     * @param configure_options 可選的設定 callback，會在 Session 建立前被呼叫，
     *                          傳入一個可寫的 SessionOptions 讓呼叫端調整
     *                          （例如 intra-op threads、graph optimization level）。
     *                          傳空的 lambda 或不傳即使用 ORT 預設值。
     *
     * @note  Ort::SessionOptions 不可複製（RAII wrapper 包 C handle），因此
     *        API 設計成「在 ctor 內就地建構並讓呼叫端 in-place 修改」的形式，
     *        而不是「呼叫端建好再傳入」。
     */
    using ConfigureOptionsFn = std::function<void(Ort::SessionOptions&)>;

    explicit OnnxInferenceEngine(const std::string& model_path,
                                 ConfigureOptionsFn configure_options = {});

    /**
     * @brief 執行一次推理。
     *
     * @param input_img  HWC CV_32FC3 圖片，尺寸 = (in_h, in_w)。
     *                   內部會做 HWC → NCHW 轉換寫入預先配置的 buffer。
     */
    void run(const cv::Mat& input_img);

    // ── 輸出存取（ORT 推理後直接可讀，資料存在內部 cv::Mat buffer 裡）──

    /// 所有輸出的 4D NCHW Mat（與模型輸出順序相同）。
    /// 可直接當作 YOLOPostProcessor::process() 的 feature_maps 輸入。
    const std::vector<cv::Mat>& output_mats() const { return outputs_; }

    /// 指定 index 的輸出 Mat。
    const cv::Mat& output_mat(size_t idx) const { return outputs_.at(idx); }

    // ── 模型資訊 ──
    int64_t in_c() const { return ch_; }
    int64_t in_h() const { return in_h_; }
    int64_t in_w() const { return in_w_; }
    size_t  num_outputs() const { return outputs_.size(); }

private:
    // ── ORT 物件（順序重要：env 要比 session 先死）──
    Ort::Env                          env_;
    std::unique_ptr<Ort::Session>     session_;
    Ort::AllocatorWithDefaultOptions  allocator_;

    // ── I/O 名稱（string 所有權留在 allocated_strings_，vector<const char*> 給 ORT）──
    std::vector<Ort::AllocatedStringPtr> allocated_strings_;
    std::vector<const char*>             input_names_;
    std::vector<const char*>             output_names_;

    // ── 輸入 ──
    int64_t            ch_   = 0;
    int64_t            in_h_ = 0;
    int64_t            in_w_ = 0;
    std::vector<float> input_tensor_values_;              // NCHW float buffer
    std::vector<int64_t> input_shape_;                    // {1, C, H, W}

    // ── 輸出 ──
    // 每個輸出一個 4D cv::Mat（float32），ORT tensor 綁在 Mat buffer 上。
    std::vector<cv::Mat>              outputs_;
    std::vector<std::vector<int64_t>> output_shapes_;     // 與 outputs_ 對應的 shape（int64 版本，給 ORT）

    // ── helper ──
    void initialize_model_info();
    void hwc_to_nchw(const cv::Mat& src);
};

#endif


#ifdef XMODEL_MODE

// 加上這兩行！告訴編譯器 xir 和 vart 是命名空間，裡面有這些 class
namespace xir  { class Graph; class Attrs; class Tensor; }
namespace vart { class RunnerExt; class TensorBuffer; }

class XmodelInferenceEngine {
public:
    explicit XmodelInferenceEngine(const std::string& xmodel_path);
    ~XmodelInferenceEngine();

    XmodelInferenceEngine(const XmodelInferenceEngine&)            = delete;
    XmodelInferenceEngine& operator=(const XmodelInferenceEngine&) = delete;

    // ── 綁定 API ──
    void bind_input_mat(cv::Mat& ext_input) const { ext_input = input_mat_; }

    /**
     * @brief 取得預先建好的 NCHW 排列 int8 輸出。
     * DPU 內部為 NHWC，引擎會在 run() 結束時自動排版至此 Mat 中。
     */
    const cv::Mat& output_mat_nchw(size_t idx) const { return outputs_nchw_.at(idx); }

    // ── 執行 ──
    /**
     * @brief 執行推理。
     * 呼叫前，請確保已透過綁定的 cv::Mat 寫入量化後的資料 (CV_8SC3)。
     * 內部純粹執行 DPU sync 與 wait，無任何拷貝。
     */
    void run();

    // ── 模型資訊 ──
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

    // 預建好的 DPU 記憶體替身
    cv::Mat               input_mat_;      
    std::vector<cv::Mat>  outputs_;        // DPU 實體記憶體替身 (NHWC)
    std::vector<cv::Mat>  outputs_nchw_;   // [新增] 預建好的 NCHW int8 緩衝區
    std::vector<float>    output_scales_;

    void initialize_model_info(const std::string& xmodel_path);
};

#endif