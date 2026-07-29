#include <modelrunner.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef ONNX_MODE

OnnxInferenceEngine::OnnxInferenceEngine(const std::string& model_path,
                                         ConfigureOptionsFn configure_options)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OnnxInferenceEngine")
{
    Ort::SessionOptions opts;
    if (configure_options) {
        configure_options(opts);
    }

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
    initialize_model_info();
}


void OnnxInferenceEngine::initialize_model_info()
{
    {
        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator_);
        input_names_.push_back(input_name_ptr.get());
        allocated_strings_.push_back(std::move(input_name_ptr));

        auto in_shape = session_->GetInputTypeInfo(0)
                                 .GetTensorTypeAndShapeInfo()
                                 .GetShape();
        if (in_shape.size() != 4) {
            throw std::runtime_error("OnnxInferenceEngine: input must be 4D (NCHW).");
        }
        ch_   = (in_shape[1] < 0) ? 3 : in_shape[1];
        in_h_ = (in_shape[2] < 0) ? 1 : in_shape[2];
        in_w_ = (in_shape[3] < 0) ? 1 : in_shape[3];

        if (ch_ != 3) {
            throw std::runtime_error(
                "OnnxInferenceEngine: only 3-channel input is supported.");
        }

        input_shape_ = {1, ch_, in_h_, in_w_};
        input_tensor_values_.assign(
            static_cast<size_t>(ch_ * in_h_ * in_w_), 0.0f);
    }

    const size_t num_outputs = session_->GetOutputCount();
    outputs_.clear();
    outputs_.reserve(num_outputs);
    output_shapes_.clear();
    output_shapes_.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {
        auto out_name_ptr = session_->GetOutputNameAllocated(i, allocator_);
        output_names_.push_back(out_name_ptr.get());
        allocated_strings_.push_back(std::move(out_name_ptr));

        auto out_shape = session_->GetOutputTypeInfo(i)
                                  .GetTensorTypeAndShapeInfo()
                                  .GetShape();
        for (auto& d : out_shape) if (d < 0) d = 1;

        if (out_shape.size() != 4) {
            throw std::runtime_error(
                "OnnxInferenceEngine: only 4D outputs are supported (got shape size "
                + std::to_string(out_shape.size()) + ").");
        }

        std::vector<int> sizes_int(out_shape.size());
        for (size_t k = 0; k < out_shape.size(); ++k)
            sizes_int[k] = static_cast<int>(out_shape[k]);

        outputs_.emplace_back(static_cast<int>(sizes_int.size()),
                              sizes_int.data(), CV_32F);
        output_shapes_.push_back(std::move(out_shape));
    }
}


void OnnxInferenceEngine::hwc_to_nchw(const cv::Mat& src)
{
    const int H = static_cast<int>(in_h_);
    const int W = static_cast<int>(in_w_);
    const size_t plane = static_cast<size_t>(H) * W;

    std::vector<cv::Mat> ch_planes;
    ch_planes.reserve(3);
    for (int c = 0; c < 3; ++c) {
        ch_planes.emplace_back(H, W, CV_32FC1,
                               input_tensor_values_.data() + c * plane);
    }
    cv::split(src, ch_planes);
}


void OnnxInferenceEngine::run(const cv::Mat& input_img)
{
    CV_Assert(input_img.type() == CV_32FC3);
    CV_Assert(input_img.cols == in_w_ && input_img.rows == in_h_);

    hwc_to_nchw(input_img);

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(1);
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values_.data(),
        input_tensor_values_.size(),
        input_shape_.data(),
        input_shape_.size()));

    std::vector<Ort::Value> output_tensors;
    output_tensors.reserve(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i) {
        cv::Mat& m = outputs_[i];
        const size_t n_elem = static_cast<size_t>(m.total());
        output_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            m.ptr<float>(),
            n_elem,
            output_shapes_[i].data(),
            output_shapes_[i].size()));
    }

    session_->Run(Ort::RunOptions{nullptr},
                  input_names_.data(),  input_tensors.data(),  input_tensors.size(),
                  output_names_.data(), output_tensors.data(), output_tensors.size());
}

#endif


#ifdef XMODEL_MODE

// ============================================================================
//  XmodelInferenceEngine：DPU NHWC → NCHW int8 轉置
// ============================================================================
//
//  DPU 原生輸出為 NHWC int8。本引擎在 run() 結束時做一次 element-wise 轉置
//  到預配置的 NCHW buffer，方便 PostProcessor 直接消費（PostProcessor 內部
//  使用 NCHW，因為先前實驗顯示 NHWC PostProcessor 在 ARM Cortex-A 上反而
//  較慢 —— 詳見 util.h 中 YOLOPostProcessor 上方的歷史筆記）。
// ============================================================================

#define DPU_OUTPUT_CACHEABLE 0

#include <xir/graph/graph.hpp>
#include <xir/attrs/attrs.hpp>
#include <xir/tensor/tensor.hpp>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>

namespace {

inline float get_input_scale(const xir::Tensor* t) {
    int fp = t->template get_attr<int>("fix_point");
    return std::exp2f(static_cast<float>(fp));
}

inline float get_output_scale(const xir::Tensor* t) {
    int fp = t->template get_attr<int>("fix_point");
    return std::exp2f(-static_cast<float>(fp));
}

} // namespace


XmodelInferenceEngine::XmodelInferenceEngine(const std::string& xmodel_path)
{
    initialize_model_info(xmodel_path);
}

XmodelInferenceEngine::~XmodelInferenceEngine() = default;

void XmodelInferenceEngine::initialize_model_info(const std::string& xmodel_path)
{
    // 1. Deserialize xmodel & 挑出 DPU subgraph
    graph_ = xir::Graph::deserialize(xmodel_path);
    const auto* root = graph_->get_root_subgraph();

    const xir::Subgraph* dpu_subgraph = nullptr;
    for (auto* c : root->children_topological_sort()) {
        if (c->has_attr("device") && c->get_attr<std::string>("device") == "DPU") {
            dpu_subgraph = c;
            break;
        }
    }

    if (!dpu_subgraph) {
        throw std::runtime_error("XmodelInferenceEngine: no DPU subgraph found in " + xmodel_path);
    }

    // 2. 建 runner
    attrs_  = xir::Attrs::create();
    runner_ = vart::RunnerExt::create_runner(dpu_subgraph, attrs_.get());

    // 3. 抓 input / output tensor buffers
    input_tensor_buffers_  = runner_->get_inputs();
    output_tensor_buffers_ = runner_->get_outputs();

    // 4. Input meta & 建立輸入替身
    {
        const auto* in_t  = input_tensor_buffers_[0]->get_tensor();
        const auto  shape = in_t->get_shape();
        in_h_ = shape[1];
        in_w_ = shape[2];
        in_c_ = shape[3];
        input_scale_ = get_input_scale(in_t);

        uint64_t data_addr = 0u;
        size_t   size_bytes = 0u;
        std::tie(data_addr, size_bytes) = input_tensor_buffers_[0]->data({0, 0, 0, 0});

        input_mat_ = cv::Mat(in_h_, in_w_, CV_8SC3, reinterpret_cast<void*>(data_addr)); 
    }

    // ===== Output meta:建 NHWC 替身 + NCHW buffer + (條件性) cacheable 暫存區 =====
    const size_t num_outs = output_tensor_buffers_.size();
    outputs_.clear();              outputs_.reserve(num_outs);
    outputs_nchw_.clear();         outputs_nchw_.reserve(num_outs);
    output_scales_.clear();        output_scales_.reserve(num_outs);
    output_cache_buffers_.clear(); output_cache_buffers_.reserve(num_outs);

    for (size_t i = 0; i < num_outs; ++i) {
        const auto* out_t = output_tensor_buffers_[i]->get_tensor();
        const auto  shape = out_t->get_shape();
        output_scales_.push_back(get_output_scale(out_t));

        std::vector<int> sizes_int(shape.size());
        for (size_t k = 0; k < shape.size(); ++k) {
            sizes_int[k] = static_cast<int>(shape[k]);
        }

        std::vector<int> idx(shape.size(), 0);
        uint64_t data_addr = 0u;
        size_t   size_bytes = 0u;
        std::tie(data_addr, size_bytes) = output_tensor_buffers_[i]->data(idx);

        // NHWC 替身(指向 DPU 實體記憶體)
        outputs_.emplace_back(static_cast<int>(sizes_int.size()),
                            sizes_int.data(), CV_8S,
                            reinterpret_cast<void*>(data_addr));

#if DPU_OUTPUT_CACHEABLE == 0
        // 只有「DPU output 在 DDR」模式才需要 cacheable 暫存區
        const size_t total_bytes = out_t->get_data_size();
        output_cache_buffers_.emplace_back(total_bytes);
#else
        // 「DPU output 已在 cache」模式:配個空 vector 佔位,保持 index 對齊
        output_cache_buffers_.emplace_back();
#endif

        // NCHW int8 buffer(CPU 端配置)
        int C = sizes_int[sizes_int.size() - 1];
        int W = sizes_int[sizes_int.size() - 2];
        int H = sizes_int[sizes_int.size() - 3];
        int sizes_nchw[] = {1, C, H, W};
        outputs_nchw_.emplace_back(4, sizes_nchw, CV_8S);
    }
}


//==============================================================================
// 模式切換:DPU output 的記憶體屬性
//
//   DPU_OUTPUT_CACHEABLE = 0  (預設,適用接 HP port 或 non-coherent HPC)
//     - DPU 直接寫 DDR,CPU cache 不會自動更新
//     - 需要 sync_for_read 來 invalidate CPU cache
//     - 為了轉置時 cache 命中,先 memcpy 到 cacheable 暫存區
//
//   DPU_OUTPUT_CACHEABLE = 1  (接 HPC port 且硬體 coherency 真的生效)
//     - CCI-400 自動維持 cache 一致性
//     - 不需要 sync
//     - 不需要 memcpy,直接從 DPU buffer 讀就好
//
//==============================================================================
void XmodelInferenceEngine::run()
{
#if DPU_OUTPUT_CACHEABLE == 0
    // ───── Mode 0: DPU output 在 DDR,需要手動管 cache ─────

    // 1. flush input cache (CPU 寫的資料 → DDR,讓 DPU 看到)
    for (auto* inp : input_tensor_buffers_) {
        const auto* t = inp->get_tensor();
        inp->sync_for_write(0, t->get_data_size());
    }
#else
    // ───── Mode 1: HPC coherency,硬體自動處理 ─────
    // 不需要 sync,CCI 會 snoop CPU cache
#endif

    // 2. 跑 DPU (兩種模式相同)
    auto v = runner_->execute_async(input_tensor_buffers_, output_tensor_buffers_);
    const int status = runner_->wait(static_cast<int>(v.first), -1);
    (void)status;

#if DPU_OUTPUT_CACHEABLE == 0
    // 3. invalidate output cache (DPU 寫的資料 → DDR,讓 CPU 讀到最新)
    for (auto* out : output_tensor_buffers_) {
        const auto* t = out->get_tensor();
        out->sync_for_read(0, t->get_data_size());
    }

    // 4. 把 DPU output 從 DDR 一次性 memcpy 到 cacheable 暫存區
    //    memcpy 是順序存取 + ARM libc 用 NEON 跑滿頻寬,
    //    複製完後資料留在 L1/L2 cache,後續轉置幾乎全 cache hit。
    for (size_t i = 0; i < output_tensor_buffers_.size(); ++i) {
        const auto* t = output_tensor_buffers_[i]->get_tensor();
        std::memcpy(output_cache_buffers_[i].data(),
                    outputs_[i].ptr<int8_t>(),
                    t->get_data_size());
    }
#endif

        // 5. NHWC → NCHW int8 轉置 (blocked + manual unroll×4, N=1)
    for (size_t i = 0; i < output_tensor_buffers_.size(); ++i) {
        const cv::Mat& nchw = outputs_nchw_[i];
        const int C  = nchw.size[1];
        const int H  = nchw.size[2];
        const int W  = nchw.size[3];
        const int HW = H * W;
        const int WC = W * C;   // == HW * C when H=1, kept for generality

    #if DPU_OUTPUT_CACHEABLE == 0
        const int8_t* __restrict__ src =
            reinterpret_cast<const int8_t*>(output_cache_buffers_[i].data());
    #else
        const int8_t* __restrict__ src = outputs_[i].ptr<int8_t>();
    #endif
        int8_t* __restrict__ dst = outputs_nchw_[i].ptr<int8_t>();

        constexpr int BLOCK = 64;

        #pragma omp parallel for schedule(static) collapse(2) \
            if(static_cast<long>(C) * HW > 100000)
        for (int c0 = 0; c0 < C; c0 += BLOCK) {
            for (int hw0 = 0; hw0 < HW; hw0 += BLOCK) {
                const int c_end  = std::min(c0  + BLOCK, C);
                const int hw_end = std::min(hw0 + BLOCK, HW);

                for (int c = c0; c < c_end; ++c) {
                    // 計算數值的記憶體位置
                    int8_t* __restrict__ dst_row = dst + c * HW;
                    const int8_t* __restrict__ src_c = src + c;  // NHWC: stride=C

                    int hw = hw0;

                    // unroll × 4
                    // 只處理結尾到4的倍數
                    const int hw_end4 = hw0 + ((hw_end - hw0) / 4) * 4;
                    for (; hw < hw_end4; hw += 4) {
                        dst_row[hw + 0] = src_c[(hw + 0) * C];
                        dst_row[hw + 1] = src_c[(hw + 1) * C];
                        dst_row[hw + 2] = src_c[(hw + 2) * C];
                        dst_row[hw + 3] = src_c[(hw + 3) * C];
                    }

                    // 處理4的餘數多出來尾端 scalar
                    for (; hw < hw_end; ++hw)
                        dst_row[hw] = src_c[hw * C];
                }
            }
        }
    }
        
}

#endif