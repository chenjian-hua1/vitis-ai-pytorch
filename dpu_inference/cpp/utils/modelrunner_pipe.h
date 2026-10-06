// modelrunner_pipe.h — 可流水線化的 Xmodel 引擎
//
//   submit(ctx)                  flush 輸入 + execute_async  -> 交給硬體,立即返回
//   wait_hw(ctx)                 等 DPU 完成 + invalidate    -> DPU 的時間軸
//   finish(ctx)                  memcpy 到 cacheable 暫存區  -> 只有 memcpy
//   output_float_nchw(ctx,i,dst) NHWC int8 -> NCHW float(轉置 + 反量化一次完成)
//   output_mat_nchw(ctx,i)       NHWC int8 -> NCHW int8(lazy,保留給除錯 / 舊程式)
//
// 編譯期開關(可用 -D 覆蓋):
//   DPU_SYNC_CACHE     1 = 做 sync_for_write / sync_for_read(cache flush / invalidate)
//   DPU_OUTPUT_STAGING 1 = finish() 先 memcpy 到 cacheable 暫存區,轉置讀暫存區
//                      0 = 轉置直接讀 DPU 輸出緩衝
//
//   緩衝屬性                    SYNC  STAGING
//   cacheable, non-coherent      1      0
//   uncached / write-combine     0      1
//   HPC coherent                 0      0
//   預設(與舊版行為相同)       1      1
//
// 注意:輸出相關函式必須在 finish(ctx) 之後、同一個 ctx 下一次 submit(ctx)
//       之前呼叫,且同一個 ctx 不要從多條執行緒同時呼叫。

#pragma once

#ifndef DPU_SYNC_CACHE
#define DPU_SYNC_CACHE 1
#endif
#ifndef DPU_OUTPUT_STAGING
#define DPU_OUTPUT_STAGING 1
#endif


#include <opencv2/opencv.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace xir  { class Graph; class Attrs; class Subgraph; }
namespace vart { class RunnerExt; class TensorBuffer; }

class XmodelPipelineEngine {
public:
    explicit XmodelPipelineEngine(const std::string& xmodel_path, int n_ctx = 3);
    ~XmodelPipelineEngine();

    XmodelPipelineEngine(const XmodelPipelineEngine&)            = delete;
    XmodelPipelineEngine& operator=(const XmodelPipelineEngine&) = delete;

    int    n_ctx()       const { return static_cast<int>(ctxs_.size()); }
    int    in_c()        const { return in_c_; }
    int    in_h()        const { return in_h_; }
    int    in_w()        const { return in_w_; }
    size_t num_outputs() const { return n_out_; }
    float  input_scale() const { return input_scale_; }
    float  output_scale(size_t i) const { return output_scales_.at(i); }

    // 只讀形狀,不會觸發轉置
    int output_channels(size_t i) const { return ctxs_.at(0).outputs_nchw.at(i).size[1]; }
    int output_h(size_t i)        const { return ctxs_.at(0).outputs_nchw.at(i).size[2]; }
    int output_w(size_t i)        const { return ctxs_.at(0).outputs_nchw.at(i).size[3]; }

    const cv::Mat& input_mat(int ctx) const { return ctxs_.at(ctx).input_mat; }

    // NHWC int8 -> NCHW float(1,C,H,W),乘上 output_scale(idx)。
    // 等同 output_mat_nchw() + fix2float(),但只走一次記憶體。
    // dst 形狀相同時不會重新配置。
    void output_float_nchw(int ctx, size_t idx, cv::Mat& dst);
    // NHWC int8 原始輸出（1, H, W, C），不轉置、不反量化。
    // 回傳的 Mat 只是 header，指向上一次 finish() 決定的資料位置：
    //   有複製 → cacheable 暫存區；沒複製 → DPU 輸出緩衝
    // 所以必須在 finish(ctx, ...) 之後取得。
    cv::Mat output_mat_nhwc(int ctx, size_t idx) const;    // (1, H, W, C) int8，不轉置、不反量化

    // NHWC int8 -> NCHW int8,lazy,同一幀內第二次呼叫直接回傳
    const cv::Mat& output_mat_nchw(int ctx, size_t idx);

    void submit(int ctx);
    void wait_hw(int ctx);
    // DPU 輸出整理（純 CPU）
    //   copy_to_staging = true  : memcpy 到 cacheable 暫存區，之後的輸出函式讀暫存區
    //   copy_to_staging = false : 不複製，之後的輸出函式直接讀 DPU 輸出緩衝
    // 必須在 wait_hw(ctx) 之後呼叫；每一幀都要呼叫（也會重設 lazy 轉置的狀態）
    void finish(int ctx, bool copy_to_staging = true);
    // 上一次 finish(ctx, ...) 是否有複製到暫存區
    bool output_staged(int ctx) const { return ctxs_.at(ctx).staged; }

private:
    struct Ctx {
        std::unique_ptr<vart::RunnerExt> runner;
        std::vector<vart::TensorBuffer*> in_tb, out_tb;
        cv::Mat              input_mat;
        std::vector<cv::Mat> outputs;        // NHWC int8,指向 DPU 記憶體
        std::vector<cv::Mat> outputs_nchw;   // CPU 端 NCHW int8
        std::vector<std::vector<int8_t>> cache_buf;
        std::vector<uint8_t> nchw_ready;
        std::pair<uint32_t, int> job{};
        bool staged = false;     // 上一次 finish() 是否有複製到 cache_buf
    };

    void build_ctx(const xir::Subgraph* sg, Ctx& c, bool first);
    const int8_t* nhwc_src(const Ctx& c, size_t i) const;

    std::unique_ptr<xir::Graph> graph_;
    std::unique_ptr<xir::Attrs> attrs_;
    std::vector<Ctx> ctxs_;

    int    in_c_ = 0, in_h_ = 0, in_w_ = 0;
    size_t n_out_ = 0;
    float  input_scale_ = 1.0f;
    std::vector<float> output_scales_;
};