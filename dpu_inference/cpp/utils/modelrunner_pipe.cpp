// modelrunner_pipe.cpp
//
// DPU output 的 cache 屬性:與原本的 modelrunner.cpp 一致,
// 0 = 接 HP / non-coherent HPC,需要手動 sync + memcpy 到 cacheable 暫存區。
#define DPU_OUTPUT_CACHEABLE 0

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

    // 每個 context 各建一個 runner。同一個 subgraph 可以建多個,
    // 各自擁有獨立的輸入/輸出張量緩衝 —— 這正是能夠重疊的關鍵。
    // 硬體只有一顆 DPU 核心時,執行仍會排隊,但 CPU 端的準備工作
    // 可以和硬體執行重疊。
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

#if DPU_OUTPUT_CACHEABLE == 0
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


// ── 交給硬體,立即返回 ────────────────────────────────────────────
void XmodelPipelineEngine::submit(int ctx)
{
    Ctx& c = ctxs_.at(static_cast<size_t>(ctx));

#if DPU_OUTPUT_CACHEABLE == 0
    for (auto* in : c.in_tb)
        in->sync_for_write(0, in->get_tensor()->get_data_size());
#endif

    c.job = c.runner->execute_async(c.in_tb, c.out_tb);
}


// ── DPU 的時間軸:等硬體完成 ──────────────────────────────────────
void XmodelPipelineEngine::wait_hw(int ctx)
{
    Ctx& c = ctxs_.at(static_cast<size_t>(ctx));

    const int status = c.runner->wait(static_cast<int>(c.job.first), -1);
    (void)status;

#if DPU_OUTPUT_CACHEABLE == 0
    for (auto* out : c.out_tb)
        out->sync_for_read(0, out->get_tensor()->get_data_size());
#endif
}


// ── CPU 的時間軸:搬移 + 轉置 ─────────────────────────────────────
void XmodelPipelineEngine::finish(int ctx)
{
    Ctx& c = ctxs_.at(static_cast<size_t>(ctx));

#if DPU_OUTPUT_CACHEABLE == 0
    // 先一次性 memcpy 到 cacheable 暫存區:memcpy 是順序存取,
    // libc 用 NEON 跑滿頻寬,複製完資料留在 cache,後續轉置幾乎全命中。
    for (size_t i = 0; i < c.out_tb.size(); ++i)
        std::memcpy(c.cache_buf[i].data(), c.outputs[i].ptr<int8_t>(),
                    c.out_tb[i]->get_tensor()->get_data_size());
#endif

    // NHWC -> NCHW int8 轉置(blocked + unroll x4,N=1)
    for (size_t i = 0; i < c.out_tb.size(); ++i) {
        const cv::Mat& nchw = c.outputs_nchw[i];
        const int C  = nchw.size[1];
        const int H  = nchw.size[2];
        const int W  = nchw.size[3];
        const int HW = H * W;

#if DPU_OUTPUT_CACHEABLE == 0
        const int8_t* __restrict__ src =
            reinterpret_cast<const int8_t*>(c.cache_buf[i].data());
#else
        const int8_t* __restrict__ src = c.outputs[i].ptr<int8_t>();
#endif
        int8_t* __restrict__ dst = c.outputs_nchw[i].ptr<int8_t>();

        constexpr int BLOCK = 64;

        for (int c0 = 0; c0 < C; c0 += BLOCK) {
            for (int hw0 = 0; hw0 < HW; hw0 += BLOCK) {
                const int c_end  = std::min(c0 + BLOCK, C);
                const int hw_end = std::min(hw0 + BLOCK, HW);

                for (int cc = c0; cc < c_end; ++cc) {
                    int8_t* __restrict__ dst_row = dst + cc * HW;
                    const int8_t* __restrict__ src_c = src + cc;   // NHWC stride = C

                    int hw = hw0;
                    const int hw_end4 = hw0 + ((hw_end - hw0) / 4) * 4;
                    for (; hw < hw_end4; hw += 4) {
                        dst_row[hw + 0] = src_c[(hw + 0) * C];
                        dst_row[hw + 1] = src_c[(hw + 1) * C];
                        dst_row[hw + 2] = src_c[(hw + 2) * C];
                        dst_row[hw + 3] = src_c[(hw + 3) * C];
                    }
                    for (; hw < hw_end; ++hw)
                        dst_row[hw] = src_c[hw * C];
                }
            }
        }
    }
}