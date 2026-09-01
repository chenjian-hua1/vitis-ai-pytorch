// modelrunner_pipe.h — 可流水線化的 Xmodel 引擎
//
// 與原本 XmodelInferenceEngine 的差別:
//
//   1. 多組 context。每個 context 有自己的 runner 與輸入/輸出張量緩衝,
//      所以第 N+1 幀的前處理可以和第 N 幀的 DPU 同時進行。
//      原本只有一組緩衝,前處理一定要等 DPU 讀完才能寫,無法重疊。
//
//   2. run() 拆成三段,對應三種不同的資源:
//        submit(ctx)   flush 輸入 + execute_async     -> 交給硬體,立即返回
//        wait_hw(ctx)  等 DPU 完成 + invalidate       -> DPU 的時間軸
//        finish(ctx)   memcpy + NHWC->NCHW 轉置       -> CPU 的時間軸
//      原本三段黏在一起,轉置那段 CPU 工作會卡住下一幀的 DPU 提交。
//
// 用法(每個 context 由一條 pipeline slot 專用):
//   engine.input_mat(ctx)              -> 前處理寫這裡
//   engine.submit(ctx);
//   engine.wait_hw(ctx);               -> 放在 DPU 專屬執行緒
//   engine.finish(ctx);                -> 放在 CPU 後處理執行緒
//   engine.output_mat_nchw(ctx, i)

#pragma once

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
    // n_ctx 通常等於 pipeline 的 slot 數。至少要 2 才能重疊。
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

    // 前處理把資料寫進這裡(CV_8SC3,指向該 context 的 DPU 輸入記憶體)
    const cv::Mat& input_mat(int ctx) const { return ctxs_.at(ctx).input_mat; }

    // finish() 之後才有效
    const cv::Mat& output_mat_nchw(int ctx, size_t idx) const {
        return ctxs_.at(ctx).outputs_nchw.at(idx);
    }

    void submit(int ctx);     // flush 輸入 + execute_async
    void wait_hw(int ctx);    // 等硬體 + invalidate 輸出
    void finish(int ctx);     // memcpy + NHWC -> NCHW 轉置(純 CPU)

private:
    struct Ctx {
        std::unique_ptr<vart::RunnerExt> runner;
        std::vector<vart::TensorBuffer*> in_tb, out_tb;
        cv::Mat              input_mat;
        std::vector<cv::Mat> outputs;        // NHWC int8,指向 DPU 記憶體
        std::vector<cv::Mat> outputs_nchw;   // CPU 端 NCHW int8
        std::vector<std::vector<int8_t>> cache_buf;
        std::pair<uint32_t, int> job{};      // execute_async 的回傳
    };

    void build_ctx(const xir::Subgraph* sg, Ctx& c, bool first);

    std::unique_ptr<xir::Graph> graph_;
    std::unique_ptr<xir::Attrs> attrs_;
    std::vector<Ctx> ctxs_;

    int    in_c_ = 0, in_h_ = 0, in_w_ = 0;
    size_t n_out_ = 0;
    float  input_scale_ = 1.0f;
    std::vector<float> output_scales_;
};