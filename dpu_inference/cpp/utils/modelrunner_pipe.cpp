// modelrunner_pipe.cpp

#ifndef DPU_SYNC_CACHE
#define DPU_SYNC_CACHE 1
#endif
#ifndef DPU_OUTPUT_STAGING
#define DPU_OUTPUT_STAGING 1
#endif

#include "modelrunner_pipe.h"

#include <xir/graph/graph.hpp>
#include <xir/attrs/attrs.hpp>
#include <xir/tensor/tensor.hpp>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

inline float get_input_scale(const xir::Tensor* t) {
    return std::exp2f(static_cast<float>(t->get_attr<int>("fix_point")));
}
inline float get_output_scale(const xir::Tensor* t) {
    return std::exp2f(-static_cast<float>(t->get_attr<int>("fix_point")));
}

// NHWC -> NCHW 分塊轉置(N=1)。cvt 決定每個元素怎麼寫入(複製或反量化)。
template <typename T, typename Cvt>
inline void transpose_blocked(const int8_t* __restrict__ src, T* __restrict__ dst,
                              int C, int HW, Cvt cvt)
{
    constexpr int BLOCK = 64;
    for (int c0 = 0; c0 < C; c0 += BLOCK) {
        const int c_end = std::min(c0 + BLOCK, C);
        for (int hw0 = 0; hw0 < HW; hw0 += BLOCK) {
            const int hw_end  = std::min(hw0 + BLOCK, HW);
            const int hw_end4 = hw0 + ((hw_end - hw0) / 4) * 4;
            for (int cc = c0; cc < c_end; ++cc) {
                T* __restrict__ dst_row = dst + static_cast<size_t>(cc) * HW;
                const int8_t* __restrict__ src_c = src + cc;   // NHWC stride = C
                int hw = hw0;
                for (; hw < hw_end4; hw += 4) {
                    dst_row[hw + 0] = cvt(src_c[(hw + 0) * C]);
                    dst_row[hw + 1] = cvt(src_c[(hw + 1) * C]);
                    dst_row[hw + 2] = cvt(src_c[(hw + 2) * C]);
                    dst_row[hw + 3] = cvt(src_c[(hw + 3) * C]);
                }
                for (; hw < hw_end; ++hw)
                    dst_row[hw] = cvt(src_c[hw * C]);
            }
        }
    }
}

}  // namespace


XmodelPipelineEngine::XmodelPipelineEngine(const std::string& xmodel_path, int n_ctx)
{
    if (n_ctx < 1) n_ctx = 1;

    graph_ = xir::Graph::deserialize(xmodel_path);
    const auto* root = graph_->get_root_subgraph();

    const xir::Subgraph* dpu_sg = nullptr;
    for (auto* c : root->children_topological_sort()) {
        if (c->has_attr("device") && c->get_attr<std::string>("device") == "DPU") {
            dpu_sg = c;
            break;
        }
    }
    if (!dpu_sg)
        throw std::runtime_error("XmodelPipelineEngine: 在 " + xmodel_path
                                 + " 找不到 DPU subgraph");

    attrs_ = xir::Attrs::create();

    ctxs_.resize(static_cast<size_t>(n_ctx));
    for (int i = 0; i < n_ctx; ++i)
        build_ctx(dpu_sg, ctxs_[static_cast<size_t>(i)], i == 0);
}

XmodelPipelineEngine::~XmodelPipelineEngine() = default;


void XmodelPipelineEngine::build_ctx(const xir::Subgraph* sg, Ctx& c, bool first)
{
    c.runner = vart::RunnerExt::create_runner(sg, attrs_.get());
    c.in_tb  = c.runner->get_inputs();
    c.out_tb = c.runner->get_outputs();

    // ---- 輸入 ----
    {
        const auto* t = c.in_tb[0]->get_tensor();
        const auto shape = t->get_shape();
        if (first) {
            in_h_ = shape[1];
            in_w_ = shape[2];
            in_c_ = shape[3];
            input_scale_ = get_input_scale(t);
        }
        uint64_t addr = 0; size_t nbytes = 0;
        std::tie(addr, nbytes) = c.in_tb[0]->data({0, 0, 0, 0});
        c.input_mat = cv::Mat(shape[1], shape[2], CV_8SC3,
                              reinterpret_cast<void*>(addr));
    }

    // ---- 輸出 ----
    const size_t n = c.out_tb.size();
    if (first) { n_out_ = n; output_scales_.reserve(n); }

    c.outputs.reserve(n);
    c.outputs_nchw.reserve(n);
    c.cache_buf.reserve(n);
    c.nchw_ready.assign(n, 0);

    for (size_t i = 0; i < n; ++i) {
        const auto* t = c.out_tb[i]->get_tensor();
        const auto shape = t->get_shape();
        if (first) output_scales_.push_back(get_output_scale(t));

        std::vector<int> sizes(shape.size());
        for (size_t k = 0; k < shape.size(); ++k) sizes[k] = static_cast<int>(shape[k]);

        std::vector<int> idx(shape.size(), 0);
        uint64_t addr = 0; size_t nbytes = 0;
        std::tie(addr, nbytes) = c.out_tb[i]->data(idx);

        c.outputs.emplace_back(static_cast<int>(sizes.size()), sizes.data(),
                               CV_8S, reinterpret_cast<void*>(addr));

#if DPU_OUTPUT_STAGING
        c.cache_buf.emplace_back(t->get_data_size());
#else
        c.cache_buf.emplace_back();
#endif

        const int C = sizes[sizes.size() - 1];
        const int W = sizes[sizes.size() - 2];
        const int H = sizes[sizes.size() - 3];
        int nchw[] = {1, C, H, W};
        c.outputs_nchw.emplace_back(4, nchw, CV_8S);
    }
}


void XmodelPipelineEngine::submit(int ctx)
{
    Ctx& c = ctxs_.at(static_cast<size_t>(ctx));
    std::fill(c.nchw_ready.begin(), c.nchw_ready.end(), 0);

#if DPU_SYNC_CACHE
    for (auto* in : c.in_tb)
        in->sync_for_write(0, in->get_tensor()->get_data_size());
#endif

    c.job = c.runner->execute_async(c.in_tb, c.out_tb);
}


void XmodelPipelineEngine::wait_hw(int ctx)
{
    Ctx& c = ctxs_.at(static_cast<size_t>(ctx));

    const int status = c.runner->wait(static_cast<int>(c.job.first), -1);
    (void)status;

#if DPU_SYNC_CACHE
    for (auto* out : c.out_tb)
        out->sync_for_read(0, out->get_tensor()->get_data_size());
#endif
}


// ── 只做 memcpy(STAGING=0 時什麼都不做)───────────────────────────
void XmodelPipelineEngine::finish(int ctx)
{
    Ctx& c = ctxs_.at(static_cast<size_t>(ctx));

#if DPU_OUTPUT_STAGING
    for (size_t i = 0; i < c.out_tb.size(); ++i)
        std::memcpy(c.cache_buf[i].data(), c.outputs[i].ptr<int8_t>(),
                    c.out_tb[i]->get_tensor()->get_data_size());
#endif

    std::fill(c.nchw_ready.begin(), c.nchw_ready.end(), 0);
}


const int8_t* XmodelPipelineEngine::nhwc_src(const Ctx& c, size_t i) const
{
#if DPU_OUTPUT_STAGING
    return c.cache_buf[i].data();
#else
    return c.outputs[i].ptr<int8_t>();
#endif
}


// ── 轉置 + 反量化,一次走完 ──────────────────────────────────────
void XmodelPipelineEngine::output_float_nchw(int ctx, size_t idx, cv::Mat& dst)
{
    Ctx& c = ctxs_.at(static_cast<size_t>(ctx));
    if (idx >= c.outputs_nchw.size())
        throw std::out_of_range("XmodelPipelineEngine::output_float_nchw: idx 超出範圍");

    const cv::Mat& ref = c.outputs_nchw[idx];
    const int C = ref.size[1], H = ref.size[2], W = ref.size[3];
    int sz[] = {1, C, H, W};
    dst.create(4, sz, CV_32F);   // 形狀相同就不重新配置

    const float s = output_scales_[idx];
    transpose_blocked(nhwc_src(c, idx), dst.ptr<float>(), C, H * W,
                      [s](int8_t v) { return static_cast<float>(v) * s; });
}


// ── int8 NCHW,lazy ──────────────────────────────────────────────
const cv::Mat& XmodelPipelineEngine::output_mat_nchw(int ctx, size_t idx)
{
    Ctx& c = ctxs_.at(static_cast<size_t>(ctx));
    if (idx >= c.outputs_nchw.size())
        throw std::out_of_range("XmodelPipelineEngine::output_mat_nchw: idx 超出範圍");

    if (!c.nchw_ready[idx]) {
        cv::Mat& d = c.outputs_nchw[idx];
        transpose_blocked(nhwc_src(c, idx), d.ptr<int8_t>(),
                          d.size[1], d.size[2] * d.size[3],
                          [](int8_t v) { return v; });
        c.nchw_ready[idx] = 1;
    }
    return c.outputs_nchw[idx];
}