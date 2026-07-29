#include <data_struct.h>

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