#include <yolopproc.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

// ============================================================================
//  YOLOPostProcessor
// ============================================================================

YOLOPostProcessor::YOLOPostProcessor(int batch,
                                      int input_h,
                                      int input_w,
                                      int nc,
                                      int ch,
                                      std::vector<int> strides,
                                      int max_nms,
                                      int max_det)
    : B_(batch),
      input_h_(input_h),
      input_w_(input_w),
      nc_(nc),
      ch_(ch),
      no_(nc + ch * 4),
      strides_(std::move(strides)),
      max_nms_(max_nms),
      max_det_(max_det)
{
    precompute_anchors();
    allocate_buffers();
    bind_views();
    cache_pointers();
    precompute_tables();
}


void YOLOPostProcessor::precompute_anchors()
{
    constexpr float offset = 0.5f;

    hw_per_scale_.clear();
    scale_offsets_.clear();
    std::vector<std::pair<int,int>> hw_list;
    A_ = 0;
    int acc_offset = 0;
    for (int stride : strides_) {
        if (input_h_ % stride != 0 || input_w_ % stride != 0)
            throw std::runtime_error("input size not divisible by stride");
        int H = input_h_ / stride;
        int W = input_w_ / stride;
        hw_list.emplace_back(H, W);
        hw_per_scale_.push_back(H * W);
        scale_offsets_.push_back(acc_offset);
        acc_offset += H * W;
        A_ += H * W;
    }

    anchors_T_ = cv::Mat(2, A_, CV_32F);
    stride_T_  = cv::Mat(1, A_, CV_32F);

    int col = 0;
    for (size_t i = 0; i < strides_.size(); ++i) {
        int H = hw_list[i].first;
        int W = hw_list[i].second;
        float s = static_cast<float>(strides_[i]);
        for (int gy = 0; gy < H; ++gy)
            for (int gx = 0; gx < W; ++gx, ++col) {
                anchors_T_.at<float>(0, col) = static_cast<float>(gx) + offset;
                anchors_T_.at<float>(1, col) = static_cast<float>(gy) + offset;
                stride_T_.at<float>(0, col)  = s;
            }
    }
}


void YOLOPostProcessor::allocate_buffers()
{
    x_cat_   = cv::Mat(std::vector<int>{B_, no_,     A_}, CV_32F);
    dfl_out_ = cv::Mat(std::vector<int>{B_, 4,       A_}, CV_32F);
    output_  = cv::Mat(std::vector<int>{B_, 4 + nc_, A_}, CV_32F);

    candidates_.resize(max_nms_);
    boxes_cv_.resize(max_nms_);
    scores_cv_.resize(max_nms_);
    indices_.reserve(max_nms_);

    detections_.resize(B_);
    for (auto& db : detections_) {
        db.data.resize(max_det_);
        db.count = 0;
    }

    active_indices_.assign(B_, {});
    active_cls_scores_.assign(B_, {});
    active_dist_.assign(B_, {});
    for (int b = 0; b < B_; ++b) {
        active_indices_[b].reserve(A_);
        active_cls_scores_[b].reserve(static_cast<size_t>(A_) * nc_);
        // 先預留 max_nms 個 anchor 的空間，超過時 vector 會自動擴充
        active_dist_[b].reserve(static_cast<size_t>(max_nms_) * 4 * ch_);
    }
}


void YOLOPostProcessor::bind_views()
{
    const int split = 4 * ch_;
    cv::Range box_r[] = { cv::Range::all(), cv::Range(0, split),   cv::Range::all() };
    cv::Range cls_r[] = { cv::Range::all(), cv::Range(split, no_), cv::Range::all() };
    box_raw_view_ = x_cat_(box_r);
    cls_raw_view_ = x_cat_(cls_r);
}


void YOLOPostProcessor::cache_pointers()
{
    ax_ = anchors_T_.ptr<float>(0);
    ay_ = anchors_T_.ptr<float>(1);
    sv_ = stride_T_.ptr<float>(0);

    output_cx_rows_.resize(B_);
    output_cy_rows_.resize(B_);
    output_w_rows_.resize(B_);
    output_h_rows_.resize(B_);
    output_cls_rows_.assign(B_, std::vector<float*>(nc_, nullptr));

    for (int b = 0; b < B_; ++b) {
        output_cx_rows_[b] = output_.ptr<float>(b, 0);
        output_cy_rows_[b] = output_.ptr<float>(b, 1);
        output_w_rows_[b]  = output_.ptr<float>(b, 2);
        output_h_rows_[b]  = output_.ptr<float>(b, 3);
        for (int c = 0; c < nc_; ++c)
            output_cls_rows_[b][c] = output_.ptr<float>(b, 4 + c);
    }

    dfl_rows_.resize(B_);
    for (int b = 0; b < B_; ++b)
        for (int coord = 0; coord < 4; ++coord)
            dfl_rows_[b][coord] = dfl_out_.ptr<float>(b, coord);

    const int split = 4 * ch_;
    box_raw_rows_.assign(B_, std::vector<const float*>(split, nullptr));
    cls_raw_rows_.assign(B_, std::vector<const float*>(nc_,   nullptr));
    for (int b = 0; b < B_; ++b) {
        for (int c = 0; c < split; ++c)
            box_raw_rows_[b][c] = box_raw_view_.ptr<float>(b, c);
        for (int c = 0; c < nc_; ++c)
            cls_raw_rows_[b][c] = cls_raw_view_.ptr<float>(b, c);
    }

    xcat_rows_.assign(B_, std::vector<float*>(no_, nullptr));
    for (int b = 0; b < B_; ++b)
        for (int c = 0; c < no_; ++c)
            xcat_rows_[b][c] = x_cat_.ptr<float>(b, c);
}


void YOLOPostProcessor::precompute_tables()
{
    constexpr float max_wh = 7680.f;
    class_offsets_.resize(nc_);
    for (int c = 0; c < nc_; ++c)
        class_offsets_[c] = static_cast<float>(c) * max_wh;

    dfl_arange_.resize(ch_);
    std::iota(dfl_arange_.begin(), dfl_arange_.end(), 0.f);
}


// ═══════════════ decode pipeline (A) 合併輸出 ═══════════════

void YOLOPostProcessor::classify_and_build_mask(float conf_thresh)
{
    const int A = A_;

    const float logit_thresh = (conf_thresh > 0.f && conf_thresh < 1.f)
        ? std::log(conf_thresh / (1.f - conf_thresh))
        : -FLT_MAX;

    for (int b = 0; b < B_; ++b) {
        active_indices_[b].clear();

        const std::vector<const float*>& cls_raw = cls_raw_rows_[b];

        // Scan Anthor
        for (int a = 0; a < A; ++a) {
            for (int c = 0; c < nc_; ++c) {
                if (cls_raw[c][a] > logit_thresh)
                {
                    // confidence > th : record position, score, cls
                    active_indices_[b].push_back(a);
                    break;
                }
            }
        }

        const int n_active = static_cast<int>(active_indices_[b].size());

        if (nc_ > 1 && n_active > 0) {
            auto& scores = active_cls_scores_[b];
            scores.resize(static_cast<size_t>(n_active) * nc_);

            const int* RESTRICT act = active_indices_[b].data();
            float* RESTRICT dst = scores.data();

            // 掃描全部 Activate Anchor (conf>th部份)
            for (int i = 0; i < n_active; ++i) {
                const int a = act[i];
                float* RESTRICT row = dst + static_cast<size_t>(i) * nc_;
                // 計算每個 cls 的機率 (sigmoid)
                for (int c = 0; c < nc_; ++c)
                    row[c] = 1.f / (1.f + expf(-cls_raw[c][a]));
            }
        }
    }
}


void YOLOPostProcessor::dfl_decode_masked()
{
    const int ch = ch_;

    for (int b = 0; b < B_; ++b) {
        const std::vector<int>& act = active_indices_[b];
        const int N = static_cast<int>(act.size());
        if (N == 0) continue;

        for (int coord = 0; coord < 4; ++coord) {
            float* RESTRICT out_row = dfl_rows_[b][coord];
            const int base = coord * ch;

            const float* RESTRICT ch_rows[32];
            for (int c = 0; c < ch; ++c)
                ch_rows[c] = box_raw_rows_[b][base + c];

            for (int i = 0; i < N; ++i) {
                // softmax
                const int a = act[i];

                float max_l = ch_rows[0][a];
                for (int c = 1; c < ch; ++c) {
                    float v = ch_rows[c][a];
                    if (v > max_l) max_l = v;
                }

                float sum_e = 0.f, wsum = 0.f;
                for (int c = 0; c < ch; ++c) {
                    float e = expf(ch_rows[c][a] - max_l);
                    sum_e += e;
                    wsum  += e * dfl_arange_[c];
                }
                out_row[a] = wsum / sum_e;
            }
        }
    }
}


void YOLOPostProcessor::dist2bbox_masked()
{
    for (int b = 0; b < B_; ++b) {
        const std::vector<int>& act = active_indices_[b];
        const int N = static_cast<int>(act.size());
        if (N == 0) continue;

        float* RESTRICT cx = output_cx_rows_[b];
        float* RESTRICT cy = output_cy_rows_[b];
        float* RESTRICT ww = output_w_rows_[b];
        float* RESTRICT hh = output_h_rows_[b];
        const float* RESTRICT dl = dfl_rows_[b][0];
        const float* RESTRICT dt = dfl_rows_[b][1];
        const float* RESTRICT dr = dfl_rows_[b][2];
        const float* RESTRICT db = dfl_rows_[b][3];
        const float* RESTRICT ax = ax_;
        const float* RESTRICT ay = ay_;
        const float* RESTRICT sv = sv_;

        for (int i = 0; i < N; ++i) {
            const int a = act[i];
            float lt_x = ax[a] - dl[a];
            float lt_y = ay[a] - dt[a];
            float rb_x = ax[a] + dr[a];
            float rb_y = ay[a] + db[a];
            float s    = sv[a];

            cx[a] = (lt_x + rb_x) * 0.5f * s;
            cy[a] = (lt_y + rb_y) * 0.5f * s;
            ww[a] = (rb_x - lt_x) * s;
            hh[a] = (rb_y - lt_y) * s;
        }
    }
}


const cv::Mat& YOLOPostProcessor::decode(const std::vector<cv::Mat>& x,
                                         float conf_thresh)
{
    if (x[0].size[0] != B_)
        throw std::runtime_error("feature map batch mismatch");

    const int B = B_;

    // NCHW concat: 一個 channel 一次 memcpy (H*W floats)
    for (size_t i = 0; i < x.size(); ++i) {
        const cv::Mat& xi = x[i];
        const int hw = xi.size[2] * xi.size[3];
        const int col_offset = scale_offsets_[i];

        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < no_; ++c) {
                const float* src = xi.ptr<float>(b, c);
                float*       dst = xcat_rows_[b][c] + col_offset;
                std::memcpy(dst, src, hw * sizeof(float));
            }
        }
    }

    conf_thresh_cached_ = conf_thresh;
    split_mode_ = false;

    classify_and_build_mask(conf_thresh);
    dfl_decode_masked();
    dist2bbox_masked();

    return output_;
}


// ═══════════════ decode pipeline (B) cls / dist 分開輸出 ═══════════════

void YOLOPostProcessor::validate_split_inputs(const std::vector<cv::Mat>& cls_maps,
                                              const std::vector<cv::Mat>& dist_maps,
                                              int expected_type,
                                              Layout layout) const
{
    const size_t L = strides_.size();
    if (cls_maps.size() != L || dist_maps.size() != L)
        throw std::runtime_error("split decode: cls_maps / dist_maps 數量必須等於 strides 數量 ("
                                 + std::to_string(L) + ")");

    for (size_t s = 0; s < L; ++s) {
        const int H = input_h_ / strides_[s];
        const int W = input_w_ / strides_[s];

        const cv::Mat* mats[2]   = { &cls_maps[s], &dist_maps[s] };
        const int      min_ch[2] = { nc_, 4 * ch_ };
        const char*    names[2]  = { "cls", "dist" };

        for (int k = 0; k < 2; ++k) {
            const cv::Mat& m = *mats[k];
            const std::string tag = std::string(names[k]) + "_maps[" + std::to_string(s) + "]";

            if (m.dims != 4)
                throw std::runtime_error(tag + " 必須是 4 維 Mat");
            if (m.type() != expected_type)
                throw std::runtime_error(tag + " 型別錯誤（float 版需 CV_32F，int8 版需 CV_8S）");
            if (!m.isContinuous())
                throw std::runtime_error(tag + " 必須是連續記憶體");
            if (m.size[0] != B_)
                throw std::runtime_error(tag + " batch 大小不符");

            const int mh = (layout == Layout::NHWC) ? m.size[1] : m.size[2];
            const int mw = (layout == Layout::NHWC) ? m.size[2] : m.size[3];
            const int mc = (layout == Layout::NHWC) ? m.size[3] : m.size[1];
            if (mh != H || mw != W)
                throw std::runtime_error(tag + " H/W 不符，預期 " + std::to_string(H) + "x"
                                         + std::to_string(W) + "（請確認順序為 stride 8, 16, 32）");
            // channel 可以大於需求（DPU channel padding），只使用前 nc / 4*ch 個
            if (mc < min_ch[k])
                throw std::runtime_error(tag + " channel 數不足，至少需要 " + std::to_string(min_ch[k]));
        }
    }
}


// ─────────────────────────────────────────────────────────────
//  判斷 buffer 是否 cacheable
//
//  Linux 不提供 user space 查詢頁面快取屬性的 API，改用讀取速度判斷：
//    cacheable     : 第二次讀取命中 L1，約 1~2 ns / byte
//    non-cacheable : 每次讀取都要到 DDR，約 100 ns / byte
//  兩者差距接近 100 倍，以 10 ns / byte 為界線。
//  只讀前 256 bytes（non-cacheable 時約 25 us），取 3 次中最快的一次排除干擾。
// ─────────────────────────────────────────────────────────────
static volatile int g_probe_sink = 0;

bool YOLOPostProcessor::probe_cacheable(const void* ptr, size_t bytes)
{
    using clk = std::chrono::steady_clock;
    const size_t n = std::min<size_t>(bytes, 256);
    if (n == 0) return true;

    const volatile int8_t* v = static_cast<const volatile int8_t*>(ptr);
    int sink = 0;
    for (size_t i = 0; i < n; ++i) sink += v[i];          // 第一次：載入 cache（若可快取）

    double best_ns = 1e18;
    for (int t = 0; t < 3; ++t) {
        auto t0 = clk::now();
        for (size_t i = 0; i < n; ++i) sink += v[i];      // volatile：每個 byte 都實際讀取
        auto t1 = clk::now();
        best_ns = std::min(best_ns,
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
    }
    g_probe_sink = sink;                                  // 避免編譯器把讀取最佳化掉

    return (best_ns / static_cast<double>(n)) < 10.0;
}


bool YOLOPostProcessor::is_cacheable(const void* ptr, size_t bytes)
{
    switch (cls_mem_hint_) {
        case MemHint::Cacheable:    return true;
        case MemHint::NonCacheable: return false;
        case MemHint::Auto:         break;
    }
    // Auto：同一個 buffer 位址只量一次（VART 的輸出 buffer 位址固定）
    auto it = mem_probe_cache_.find(ptr);
    if (it != mem_probe_cache_.end()) return it->second;
    const bool c = probe_cacheable(ptr, bytes);
    mem_probe_cache_.emplace(ptr, c);
    return c;
}


//
//  掃 cls → 通過閾值的 anchor 才讀 dist
//
//  T = float  : 已反量化的 float 輸入，cls_scales / dist_scales 傳 nullptr
//  T = int8_t : DPU 原始輸出，閾值換算成 int8 直接比較，只有通過的 anchor 才反量化
//
//  產生：
//    active_indices_[b]    : 通過的 anchor 全域 index（與合併輸出路徑相同的編號）
//    active_cls_scores_[b] : N × nc，sigmoid 後的分數
//    active_dist_[b]       : N × (4*ch)，float logits，供 dfl_decode_gathered 使用
//
template <typename T>
void YOLOPostProcessor::gather_split(const std::vector<cv::Mat>& cls_maps,
                                     const std::vector<cv::Mat>& dist_maps,
                                     const float* cls_scales,
                                     const float* dist_scales,
                                     float conf_thresh,
                                     Layout layout)
{
    constexpr bool kInt8 = std::is_same<T, int8_t>::value;
    const int D  = 4 * ch_;
    const int nc = nc_;
    const bool nhwc = (layout == Layout::NHWC);

    const float logit_thresh = (conf_thresh > 0.f && conf_thresh < 1.f)
        ? std::log(conf_thresh / (1.f - conf_thresh))
        : -FLT_MAX;

    for (int b = 0; b < B_; ++b) {
        active_indices_[b].clear();
        active_cls_scores_[b].clear();
        active_dist_[b].clear();
    }
    if (kInt8) last_cls_cacheable_.assign(strides_.size(), false);

    for (size_t s = 0; s < strides_.size(); ++s) {
        const cv::Mat& cm = cls_maps[s];
        const cv::Mat& dm = dist_maps[s];
        const int HW     = hw_per_scale_[s];
        const int a_base = scale_offsets_[s];

        // channel stride（NHWC 時 = 實際 channel 數，可能含 DPU padding）
        const int cC = nhwc ? cm.size[3] : cm.size[1];
        const int dC = nhwc ? dm.size[3] : dm.size[1];

        const float cs = kInt8 ? cls_scales[s]  : 1.f;
        const float ds = kInt8 ? dist_scales[s] : 1.f;

        // cls 閾值：int8 版換算成整數，比較時不需要浮點運算
        //   v * cs > logit  <=>  v > logit / cs  <=>  v > floor(logit / cs)
        int   thr_i = 0;
        float thr_f = logit_thresh;
        if (kInt8) {
            if (logit_thresh == -FLT_MAX) thr_i = -129;       // 全部通過
            else {
                float t = std::floor(logit_thresh / cs);
                thr_i = (t >= 127.f) ? 127 : (t < -129.f ? -129 : static_cast<int>(t));
            }
        }
        auto pass = [&](T v) -> bool {
            if (kInt8) return static_cast<int>(v) > thr_i;
            else       return static_cast<float>(v) > thr_f;
        };

        const size_t cls_elems = static_cast<size_t>(HW) * cC;
        const size_t dist_elems = static_cast<size_t>(HW) * dC;

        // int8：判斷 cls buffer 是否 cacheable（整個 Mat 屬於同一塊記憶體，每個 head 判斷一次）
        bool cls_cacheable = true;
        if constexpr (kInt8) {
            cls_cacheable = is_cacheable(cm.ptr<T>(0), cls_elems * B_);
            last_cls_cacheable_[s] = cls_cacheable;
        }

        for (int b = 0; b < B_; ++b) {
            const T* cls_ptr = cm.ptr<T>(0) + b * cls_elems;
            if constexpr (kInt8) {
                // non-cacheable（DPU 輸出 buffer）逐 byte 讀非常慢，
                // 先把整張 cls 圖用 memcpy（寬載入）搬到一般記憶體再掃描；
                // 已經是 cacheable 就直接掃描，省下這次複製
                if (!cls_cacheable) {
                    if (cls_stage_.size() < cls_elems) cls_stage_.resize(cls_elems);
                    std::memcpy(cls_stage_.data(), cls_ptr, cls_elems);
                    cls_ptr = cls_stage_.data();
                }
            }
            const T* dist_ptr = dm.ptr<T>(0) + b * dist_elems;

            std::vector<int>&   act    = active_indices_[b];
            std::vector<float>& scores = active_cls_scores_[b];
            std::vector<float>& dist   = active_dist_[b];

            for (int p = 0; p < HW; ++p) {
                // ── 1. 掃 cls：任一類別超過閾值就算通過 ──
                bool hit = false;
                if (nhwc) {
                    const T* c = cls_ptr + static_cast<size_t>(p) * cC;
                    for (int k = 0; k < nc; ++k) if (pass(c[k])) { hit = true; break; }
                } else {
                    for (int k = 0; k < nc; ++k)
                        if (pass(cls_ptr[static_cast<size_t>(k) * HW + p])) { hit = true; break; }
                }
                if (!hit) continue;          // 沒通過：完全不讀這個 anchor 的 dist

                // ── 2. 記錄 anchor 與 sigmoid 分數 ──
                act.push_back(a_base + p);
                for (int k = 0; k < nc; ++k) {
                    const T v = nhwc ? cls_ptr[static_cast<size_t>(p) * cC + k]
                                     : cls_ptr[static_cast<size_t>(k) * HW + p];
                    scores.push_back(1.f / (1.f + expf(-static_cast<float>(v) * cs)));
                }

                // ── 3. 讀取這個 anchor 的 dist（4*ch 個值），轉成 float 存進連續 vector ──
                const size_t off = dist.size();
                dist.resize(off + D);
                float* RESTRICT dst = dist.data() + off;
                if (nhwc) {
                    // NHWC：同一個 anchor 的 4*ch 個值是連續的，一次 memcpy（寬載入）
                    T tmp[4 * 32];
                    std::memcpy(tmp, dist_ptr + static_cast<size_t>(p) * dC, D * sizeof(T));
                    for (int j = 0; j < D; ++j) dst[j] = static_cast<float>(tmp[j]) * ds;
                } else {
                    // NCHW：每個 channel 相隔 HW
                    for (int j = 0; j < D; ++j)
                        dst[j] = static_cast<float>(dist_ptr[static_cast<size_t>(j) * HW + p]) * ds;
                }
            }
        }
    }
}


// 對收集好的連續 dist 做 DFL：每個 anchor 4 組 × ch 個 bin，softmax 後求期望值
void YOLOPostProcessor::dfl_decode_gathered()
{
    const int ch = ch_;
    const int D  = 4 * ch;

    for (int b = 0; b < B_; ++b) {
        const std::vector<int>& act = active_indices_[b];
        const int N = static_cast<int>(act.size());
        if (N == 0) continue;

        const float* RESTRICT src = active_dist_[b].data();
        const float* RESTRICT w   = dfl_arange_.data();

        for (int i = 0; i < N; ++i) {
            const int a = act[i];
            const float* RESTRICT anchor_dist = src + static_cast<size_t>(i) * D;

            for (int coord = 0; coord < 4; ++coord) {
                const float* RESTRICT v = anchor_dist + coord * ch;   // 連續的 ch 個 bin

                float max_l = v[0];
                for (int c = 1; c < ch; ++c) if (v[c] > max_l) max_l = v[c];

                float sum_e = 0.f, wsum = 0.f;
                for (int c = 0; c < ch; ++c) {
                    float e = expf(v[c] - max_l);
                    sum_e += e;
                    wsum  += e * w[c];
                }
                // 寫到 dfl_out_ 的 anchor 位置，後面沿用 dist2bbox_masked()
                dfl_rows_[b][coord][a] = wsum / sum_e;
            }
        }
    }
}


const cv::Mat& YOLOPostProcessor::decode(const std::vector<cv::Mat>& cls_maps,
                                         const std::vector<cv::Mat>& dist_maps,
                                         float conf_thresh,
                                         Layout layout)
{
    if (ch_ > 32) throw std::runtime_error("split decode: ch 最大支援 32");
    validate_split_inputs(cls_maps, dist_maps, CV_32F, layout);

    conf_thresh_cached_ = conf_thresh;
    split_mode_ = true;

    gather_split<float>(cls_maps, dist_maps, nullptr, nullptr, conf_thresh, layout);
    dfl_decode_gathered();
    dist2bbox_masked();
    return output_;
}


const cv::Mat& YOLOPostProcessor::decode(const std::vector<cv::Mat>& cls_maps,
                                         const std::vector<cv::Mat>& dist_maps,
                                         const std::vector<float>&   cls_scales,
                                         const std::vector<float>&   dist_scales,
                                         float conf_thresh,
                                         Layout layout)
{
    if (ch_ > 32) throw std::runtime_error("split decode: ch 最大支援 32");
    validate_split_inputs(cls_maps, dist_maps, CV_8S, layout);
    if (cls_scales.size() != strides_.size() || dist_scales.size() != strides_.size())
        throw std::runtime_error("split decode: cls_scales / dist_scales 數量必須等於 strides 數量");
    for (size_t s = 0; s < strides_.size(); ++s)
        if (!(cls_scales[s] > 0.f) || !(dist_scales[s] > 0.f))
            throw std::runtime_error("split decode: scale 必須大於 0");

    conf_thresh_cached_ = conf_thresh;
    split_mode_ = true;

    gather_split<int8_t>(cls_maps, dist_maps, cls_scales.data(), dist_scales.data(),
                         conf_thresh, layout);
    dfl_decode_gathered();
    dist2bbox_masked();
    return output_;
}


// ═══════════════ NMS ═══════════════

void YOLOPostProcessor::nms_single_batch(int b,
                                          float conf_thresh,
                                          float iou_thresh)
{
    cand_count_ = 0;

    if (conf_thresh_cached_ != conf_thresh) {
        throw std::runtime_error(
            "nms conf_thresh must match the conf_thresh used in decode()");
    }

    const float* cx_row = output_cx_rows_[b];
    const float* cy_row = output_cy_rows_[b];
    const float* ww_row = output_w_rows_[b];
    const float* hh_row = output_h_rows_[b];

    const std::vector<int>&   act         = active_indices_[b];
    const std::vector<float>& cls_scores  = active_cls_scores_[b];
    const int N = static_cast<int>(act.size());

    for (int i = 0; i < N; ++i) {
        const int a = act[i];

        float cx = cx_row[a], cy = cy_row[a];
        float w  = ww_row[a], h  = hh_row[a];
        float x1 = cx - w * 0.5f, y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f, y2 = cy + h * 0.5f;

        if (nc_ > 1) {
            const float* RESTRICT row = cls_scores.data()
                                      + static_cast<size_t>(i) * nc_;
            for (int c = 0; c < nc_; ++c) {
                float score = row[c];
                if (score > conf_thresh && cand_count_ < max_nms_) {
                    candidates_[cand_count_++] = {
                        x1, y1, x2, y2, score, static_cast<float>(c)
                    };
                }
            }
        } else {
            // nc_ == 1:直接對那唯一一類的 logit 取 sigmoid
            //   分開輸出路徑沒有填 x_cat_，改用 gather 時算好的分數
            float score = split_mode_
                ? cls_scores[i]
                : 1.f / (1.f + expf(-cls_raw_rows_[b][0][a]));
            if (cand_count_ < max_nms_) {
                candidates_[cand_count_++] = {x1, y1, x2, y2, score, 0.f};
            }
        }

        if (cand_count_ >= max_nms_) break;
    }

    if (cand_count_ == 0) {
        detections_[b].count = 0;
        return;
    }

    std::sort(candidates_.begin(),
              candidates_.begin() + cand_count_,
              [](const auto& a, const auto& b){ return a[4] > b[4]; });

    for (int i = 0; i < cand_count_; ++i) {
        const auto& c = candidates_[i];
        int   cls_id = static_cast<int>(c[5]);
        float off    = class_offsets_[cls_id];
        boxes_cv_[i]  = cv::Rect2d(c[0] + off, c[1] + off,
                                    c[2] - c[0], c[3] - c[1]);
        scores_cv_[i] = c[4];
    }

    boxes_cv_.resize(cand_count_);
    scores_cv_.resize(cand_count_);

    cv::dnn::NMSBoxes(boxes_cv_, scores_cv_,
                      conf_thresh, iou_thresh, indices_);

    boxes_cv_.resize(max_nms_);
    scores_cv_.resize(max_nms_);

    int keep = std::min(static_cast<int>(indices_.size()), max_det_);
    auto& db = detections_[b];
    for (int k = 0; k < keep; ++k) {
        const auto& row = candidates_[indices_[k]];
        db.data[k] = Detection{
            row[0], row[1], row[2], row[3],
            row[4], static_cast<int>(row[5])
        };
    }
    db.count = keep;
}


void YOLOPostProcessor::nms(float conf_thresh, float iou_thresh)
{
    for (int b = 0; b < B_; ++b)
        nms_single_batch(b, conf_thresh, iou_thresh);
}


// ═══════════════ process ═══════════════

const std::vector<DetectionBatch>& YOLOPostProcessor::process(
    const std::vector<cv::Mat>& feature_maps,
    float conf_thresh,
    float iou_thresh)
{
    decode(feature_maps, conf_thresh);
    nms(conf_thresh, iou_thresh);
    return detections_;
}


const std::vector<DetectionBatch>& YOLOPostProcessor::process(
    const std::vector<cv::Mat>& cls_maps,
    const std::vector<cv::Mat>& dist_maps,
    float  conf_thresh,
    float  iou_thresh,
    Layout layout)
{
    decode(cls_maps, dist_maps, conf_thresh, layout);
    nms(conf_thresh, iou_thresh);
    return detections_;
}


const std::vector<DetectionBatch>& YOLOPostProcessor::process(
    const std::vector<cv::Mat>& cls_maps,
    const std::vector<cv::Mat>& dist_maps,
    const std::vector<float>&   cls_scales,
    const std::vector<float>&   dist_scales,
    float  conf_thresh,
    float  iou_thresh,
    Layout layout)
{
    decode(cls_maps, dist_maps, cls_scales, dist_scales, conf_thresh, layout);
    nms(conf_thresh, iou_thresh);
    return detections_;
}