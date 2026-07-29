#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <functional>
#include <array>
#include <thread>
#include <mutex>
#include <atomic>

#include <opencv2/opencv.hpp>

// #define ONNX_MODE
#ifndef XMODEL_MODE
#define ONNX_MODE
#endif

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