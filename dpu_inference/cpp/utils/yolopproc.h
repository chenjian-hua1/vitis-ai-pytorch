#pragma once
#include <data_struct.h>
#include <unordered_map>

// ============================================================================
//  YOLO Post-Processor Class
// ============================================================================
//
//  兩種輸入介面：
//
//  (A) 原本的合併輸出（3 張圖）
//      process(feature_maps, conf, iou)
//      feature_maps[i] : NCHW float (B, 4*ch + nc, H, W)
//
//  (B) box / cls 分開輸出（6 張圖，split_output=True 的模型）
//      process(cls_maps, dist_maps, conf, iou, layout)                       ← float
//      process(cls_maps, dist_maps, cls_scales, dist_scales, conf, iou, layout) ← int8
//      cls_maps[i]  : (B, nc,   H, W) 或 (B, H, W, nc)
//      dist_maps[i] : (B, 4*ch, H, W) 或 (B, H, W, 4*ch)
//      i 的順序對應 strides（8, 16, 32）
//
//      流程：先掃 cls → 只有超過閾值的 anchor 才去讀它的 dist（4*ch 個值），
//            收集到連續的 vector 後再做 DFL decode → dist2bbox → NMS
//
//      int8 版直接吃 DPU 原始輸出，不需要先把整張圖反量化：
//        - cls 閾值換算成 int8 比較，不做浮點運算
//        - 只有通過的 anchor 才把 dist 反量化成 float
//        - cls_scales[i] / dist_scales[i] = 2^-fix_point（每個 tensor 各自的值）
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
    enum class Layout { NCHW, NHWC };

    // int8 cls 輸入的記憶體屬性（決定掃描前要不要先 memcpy 到暫存區）
    //   Auto         : 第一次遇到某個 buffer 時量測讀取速度自動判斷，結果依位址快取（預設）
    //   Cacheable    : 已是 cacheable（例如自己 malloc 的 buffer），直接掃描，不 memcpy
    //   NonCacheable : DPU non-cacheable buffer，一律先整塊 memcpy 再掃描
    enum class MemHint { Auto, Cacheable, NonCacheable };

    YOLOPostProcessor(int batch,
                      int input_h,
                      int input_w,
                      int nc       = 80,
                      int ch       = 16,
                      std::vector<int> strides = {8, 16, 32},
                      int max_nms  = 5000,
                      int max_det  = 100);

    // ── (A) 原本的合併輸出 ──
    const std::vector<DetectionBatch>& process(
        const std::vector<cv::Mat>& feature_maps,
        float confidence_threshold = 0.25f,
        float iou_threshold        = 0.45f);

    // ── (B-1) 分開輸出，float（CV_32F，已反量化） ──
    const std::vector<DetectionBatch>& process(
        const std::vector<cv::Mat>& cls_maps,
        const std::vector<cv::Mat>& dist_maps,
        float  confidence_threshold = 0.25f,
        float  iou_threshold        = 0.45f,
        Layout layout               = Layout::NHWC);

    // ── (B-2) 分開輸出，int8（CV_8S，DPU 原始輸出） ──
    const std::vector<DetectionBatch>& process(
        const std::vector<cv::Mat>& cls_maps,
        const std::vector<cv::Mat>& dist_maps,
        const std::vector<float>&   cls_scales,
        const std::vector<float>&   dist_scales,
        float  confidence_threshold = 0.25f,
        float  iou_threshold        = 0.45f,
        Layout layout               = Layout::NHWC);

    const cv::Mat& decode(const std::vector<cv::Mat>& feature_maps,
                          float conf_thresh);
    const cv::Mat& decode(const std::vector<cv::Mat>& cls_maps,
                          const std::vector<cv::Mat>& dist_maps,
                          float conf_thresh,
                          Layout layout = Layout::NHWC);
    const cv::Mat& decode(const std::vector<cv::Mat>& cls_maps,
                          const std::vector<cv::Mat>& dist_maps,
                          const std::vector<float>&   cls_scales,
                          const std::vector<float>&   dist_scales,
                          float conf_thresh,
                          Layout layout = Layout::NHWC);
    void           nms(float confidence_threshold, float iou_threshold);

    // ── int8 cls 記憶體屬性設定 ──
    void    set_cls_mem_hint(MemHint hint) { cls_mem_hint_ = hint; }
    MemHint cls_mem_hint() const           { return cls_mem_hint_; }
    void    reset_mem_probe()              { mem_probe_cache_.clear(); }   // buffer 重新配置後呼叫
    // 上一次 int8 decode 時，每個 head 的 cls 是否判定為 cacheable（true = 沒有 memcpy）
    const std::vector<bool>& last_cls_cacheable() const { return last_cls_cacheable_; }

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

    // ── 分開輸出 (B) 專用 ──
    bool split_mode_ = false;                       // 目前 active_* 是哪條路徑產生的
    std::vector<std::vector<float>>  active_dist_;  // [b] : N × (4*ch)，通過 anchor 的 dist（已轉 float）
    std::vector<int8_t>              cls_stage_;    // int8 cls 暫存區（從 DPU buffer 整塊 memcpy）

    // ── 記憶體屬性判斷 ──
    MemHint                                  cls_mem_hint_ = MemHint::Auto;
    std::unordered_map<const void*, bool>    mem_probe_cache_;     // buffer 位址 → 是否 cacheable
    std::vector<bool>                        last_cls_cacheable_;
    bool is_cacheable(const void* ptr, size_t bytes);
    static bool probe_cacheable(const void* ptr, size_t bytes);

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

    // ── 分開輸出 (B) 專用 ──
    void validate_split_inputs(const std::vector<cv::Mat>& cls_maps,
                               const std::vector<cv::Mat>& dist_maps,
                               int expected_type, Layout layout) const;
    template <typename T>
    void gather_split(const std::vector<cv::Mat>& cls_maps,
                      const std::vector<cv::Mat>& dist_maps,
                      const float* cls_scales,
                      const float* dist_scales,
                      float conf_thresh,
                      Layout layout);
    void dfl_decode_gathered();
};