// hls_resize.cpp — resize_kernel_0 的硬體資訊、driver 與 API 實作
//
// 檔案結構(新增 IP 時照抄這個順序):
//   1. 硬體資訊    位址、暫存器 offset(匿名 namespace,外部看不到)
//   2. Kernel      繼承 AxiLiteIp,只負責寫參數暫存器
//   3. device()    IpHolder 單例 + 註冊 clear_buffers
//   4. API         include/hls_resize.h 宣告的函式

#include "hls_resize.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

namespace hls {
namespace resize {

// 內部型別(AxiLiteIp、IpHolder、stage_input…)在 hls::detail
using namespace hls::detail;

namespace {

// ============================================================
//  1. 硬體資訊(resize_dpu.xsa)
//    control : s_axi_control @ 0xB000_0000, 0x10000 bytes (32-bit)
//    m_axi   : gmem0 -> S_AXI_HP2_FPD, gmem1 -> S_AXI_HP3_FPD(非 coherent,必須 sync)
//    irq     : xlconcat In0 -> pl_ps_irq0[0]
// ============================================================
constexpr uint64_t    kCtrlPhys = 0xB0000000ull;
constexpr size_t      kCtrlSpan = 0x10000;
constexpr const char* kUioName  = "resize_kernel_0";

// 參數暫存器(xresize_kernel_hw.h);0x00–0x0C 是共用的 hls::ap_ctrl
namespace reg {
constexpr uint32_t IN_PTR        = 0x10;  // 64-bit: 0x10 / 0x14
constexpr uint32_t OUT_PTR       = 0x1c;  // 64-bit: 0x1c / 0x20
constexpr uint32_t TOTAL_WORDS   = 0x28;
constexpr uint32_t TOTAL_RESULTS = 0x30;
constexpr uint32_t OUT_WORDS     = 0x38;
constexpr uint32_t OUT_W         = 0x40;
constexpr uint32_t SCALE_MODE    = 0x48;  // 1 bit
constexpr uint32_t INV_SCALE     = 0x50;  // 16 bit
}  // namespace reg


// ============================================================
//  2. Kernel —— 暫存器層
//  UIO 查找、中斷、ap_ctrl 都由 AxiLiteIp 處理,這裡只管參數。
//  in_ptr / out_ptr 必須是「實體位址」。
// ============================================================
class Kernel : public AxiLiteIp {
public:
    Kernel(const std::string& name, uint64_t ctrl_phys, bool allow_devmem)
        : AxiLiteIp(name, ctrl_phys, allow_devmem, kCtrlSpan) {}

    void program(const Params& p, uint64_t in_phys, uint64_t out_phys) {
        wr64(reg::IN_PTR,  in_phys);
        wr64(reg::OUT_PTR, out_phys);
        wr(reg::TOTAL_WORDS,   p.total_words);
        wr(reg::TOTAL_RESULTS, p.total_results);
        wr(reg::OUT_WORDS,     p.out_words);
        wr(reg::OUT_W,         p.out_w);
        wr(reg::SCALE_MODE,    p.scale_mode & 1u);
        wr(reg::INV_SCALE,     p.inv_scale & 0xFFFFu);
    }
};


// ============================================================
//  3. 裝置單例
// ============================================================
IpHolder<Kernel>& device() {
    static IpHolder<Kernel> d(kUioName, kCtrlPhys);
    return d;
}

// 讓 hls::clear_buffers() 也能清掉這個模組的快取
const CleanupRegistrar cleanup_registration{[] { device().clear_cache(); }};

}  // namespace


// ============================================================
//  4-1. Params / plan_for
// ============================================================
bool Params::feasible(uint32_t in_w, uint32_t in_h, uint32_t scale) {
    if (scale != 2 && scale != 3)     return false;
    if (in_w % scale || in_h % scale) return false;

    const uint32_t ow = in_w / scale, oh = in_h / scale;
    if (ow == 0 || oh == 0)           return false;
    if (ow > kOutWMax)                return false;
    if (ow % n_out_for(scale))        return false;

    if ((static_cast<uint64_t>(in_w) * in_h * 3u) % 16u) return false;
    if ((static_cast<uint64_t>(ow)   * oh   * 3u) % 16u) return false;
    return true;
}

Params Params::make(uint32_t in_w, uint32_t in_h, uint32_t scale) {
    Params p;

    if (scale != 2 && scale != 3)
        throw std::invalid_argument("scale 只支援 2 或 3(整數倍 box filter)");
    if (in_w % scale || in_h % scale)
        throw std::invalid_argument("輸入尺寸必須能被 scale 整除");

    p.scale = scale;
    p.in_w  = in_w;
    p.in_h  = in_h;
    p.out_w = in_w / scale;
    p.out_h = in_h / scale;

    if (p.out_w > kOutWMax)
        throw std::invalid_argument("out_w 超過 OUT_W_MAX (960)");

    const uint32_t n_out = n_out_for(scale);
    if (p.out_w % n_out)
        throw std::invalid_argument("out_w 必須是 " + std::to_string(n_out) + " 的倍數");

    p.in_bytes  = in_w    * in_h    * 3u;
    p.out_bytes = p.out_w * p.out_h * 3u;

    if (p.in_bytes  % 16u) throw std::invalid_argument("輸入位元組數必須是 16 的倍數");
    if (p.out_bytes % 16u) throw std::invalid_argument("輸出位元組數必須是 16 的倍數");

    p.total_words   = p.in_bytes  / 16u;
    p.out_words     = p.out_bytes / 16u;          // 整張圖,不是每列
    p.total_results = p.out_w * p.out_h / n_out;  // 每筆結果帶 n_out 個 pixel
    p.scale_mode    = (scale == 3) ? 1u : 0u;

    // 四捨五入。3 倍取 65536/9=7281 會讓全白 2295*7281>>16 = 254(少 1),
    // 取 7282 才得到 255 —— 原始碼註解的值就是進位後的。
    const uint32_t s2 = scale * scale;
    p.inv_scale = (65536u + s2 / 2u) / s2;        // 2倍→16384, 3倍→7282

    return p;
}

std::string Params::describe() const {
    char b[512];
    std::snprintf(b, sizeof b,
             "%ux%u -> %ux%u (%ux 縮小)\n"
             "  total_words   = %u\n"
             "  total_results = %u\n"
             "  out_words     = %u\n"
             "  out_w         = %u\n"
             "  scale_mode    = %u (%s)\n"
             "  inv_scale     = %u",
             in_w, in_h, out_w, out_h, scale,
             total_words, total_results, out_words, out_w,
             scale_mode, scale == 3 ? "SCALE_3" : "SCALE_2", inv_scale);
    return b;
}

Plan plan_for(uint32_t in_w, uint32_t in_h, uint32_t need_w, uint32_t need_h) {
    Plan plan;

    if (need_w == 0 || need_h == 0) {
        plan.reason = "目標尺寸為 0";
        return plan;
    }
    if (need_w >= in_w && need_h >= in_h) {
        plan.reason = "目標不小於輸入,不需縮小";
        return plan;
    }
    // 連 2 倍都不到就沒得談(IP 最小倍率是 2)
    if (in_w < need_w * 2 || in_h < need_h * 2) {
        plan.reason = "縮放倍率不足 2 倍";
        return plan;
    }

    const char* why = "尺寸不符 IP 限制";

    for (uint32_t s : {3u, 2u}) {
        if (in_w / s < need_w || in_h / s < need_h) { why = "此倍率會縮過頭";           continue; }
        if (in_w % s || in_h % s)                   { why = "輸入尺寸無法被倍率整除";   continue; }
        const uint32_t ow = in_w / s;
        if (ow > Params::kOutWMax)                  { why = "中間寬度超過 OUT_W_MAX (960)"; continue; }
        if (ow % Params::n_out_for(s))              { why = "中間寬度不是 n_out 的倍數"; continue; }
        if (!Params::feasible(in_w, in_h, s))       { why = "位元組數不是 16 的倍數";   continue; }

        plan.use_ip = true;
        plan.scale  = s;
        plan.params = Params::make(in_w, in_h, s);
        plan.exact  = (plan.params.out_w == need_w && plan.params.out_h == need_h);
        plan.reason = plan.exact ? "IP 一次到位" : "IP 縮整數倍,零頭交給 CPU";
        return plan;
    }

    plan.reason = why;
    return plan;
}


// ============================================================
//  4-2. 裝置設定與查詢
// ============================================================
void configure()                          { device().set_name(device().default_name()); }
void configure(const std::string& name)   { device().set_name(name); }
void use_devmem()                         { device().set_ctrl_phys(device().default_phys(), true); }
void use_devmem(uint64_t ctrl_phys)       { device().set_ctrl_phys(ctrl_phys, true); }
void set_ctrl_phys(uint64_t ctrl_phys)    { device().set_ctrl_phys(ctrl_phys, false); }

std::string device_info() { return device().info(); }
bool        using_irq()   { return device().using_irq(); }
bool        available()   { return device().open() && pool_available(); }

std::string last_error() {
    const std::string e = device().error();
    return e.empty() ? pool_error() : e;
}


// ============================================================
//  4-3. downscale
// ============================================================
bool downscale(const cv::Mat& img, int scale, cv::Mat& out,
               Timing* timing, int timeout_ms) {
    if (img.empty() || img.type() != CV_8UC3) return false;

    Params p;
    try {
        p = Params::make(img.cols, img.rows, scale);
    } catch (const std::exception&) {
        return false;   // 條件不合,由呼叫端決定要不要退回 CPU
    }

    auto& d = device();
    std::lock_guard<std::mutex> lk(d.mutex());

    Kernel* ip = d.try_open();
    if (!ip || !ip->is_idle()) return false;

    auto pool = pool_ptr();               // shared_ptr:確保 pool 活得夠久
    if (!pool) return false;

    DmaMat* dst = d.cache().get(pool, p.out_h, p.out_w, CV_8UC3);
    if (!dst) return false;

    const Stopwatch total;

    DmaInput in;
    if (!stage_input(pool, d.cache(), img, in)) return false;

    ip->program(p, in.phys, dst->phys());

    const Stopwatch run;
    if (!ip->run(timeout_ms)) return false;
    const double run_ms = run.ms();

    const double sync_out_ms = finish_output(*pool, dst->mat().data, p.out_bytes);

    out = dst->mat();                     // 淺拷貝,資料仍在 DMA 記憶體

    if (timing) {
        timing->copy_ms   = in.copy_ms;
        timing->sync_ms   = in.sync_ms + sync_out_ms;
        timing->run_ms    = run_ms;
        timing->total_ms  = total.ms();
        timing->zero_copy = in.zero_copy;
    }
    return true;
}


// ============================================================
//  4-4. letterbox
// ============================================================
void letterbox(const cv::Mat& img, int input_size, Result& res) {
    CV_Assert(!img.empty() && img.type() == CV_8UC3);

    const int orig_h = img.rows, orig_w = img.cols;

    float r = std::min(static_cast<float>(input_size) / orig_h,
                       static_cast<float>(input_size) / orig_w);
    r = std::min(r, 1.0f);

    const int pad_w = static_cast<int>(std::round(orig_w * r));
    const int pad_h = static_cast<int>(std::round(orig_h * r));

    const float dw = (input_size - pad_w) / 2.0f;
    const float dh = (input_size - pad_h) / 2.0f;

    const int top  = static_cast<int>(std::round(dh - 0.1f));
    const int left = static_cast<int>(std::round(dw - 0.1f));

    res.ratio   = {r, r};
    res.pad     = {dw, dh};
    res.content = cv::Rect(left, top, pad_w, pad_h);

    res.img.create(input_size, input_size, img.type());
    res.img.setTo(cv::Scalar(0, 0, 0));
    cv::Mat roi = res.img(res.content);

    res.used_ip   = false;
    res.ip_scale  = 0;
    res.mid_w     = orig_w;
    res.mid_h     = orig_h;
    res.zero_copy = false;
    res.timing    = Timing{};

    const Plan plan = plan_for(orig_w, orig_h, pad_w, pad_h);
    res.reason = plan.reason;

    if (plan.use_ip) {
        cv::Mat mid;
        const Stopwatch sw;
        const bool ok = downscale(img, static_cast<int>(plan.scale), mid, &res.timing);
        res.ip_ms = sw.ms();
        if (ok) {
            res.used_ip   = true;
            res.ip_scale  = static_cast<int>(plan.scale);
            res.mid_w     = mid.cols;
            res.mid_h     = mid.rows;
            res.zero_copy = res.timing.zero_copy;

            const Stopwatch post;
            if (plan.exact)
                mid.copyTo(roi);          // IP 一次到位,免二次縮放
            else
                cv::resize(mid, roi, roi.size(), 0, 0, cv::INTER_AREA);
            res.timing.post_ms   = post.ms();
            res.timing.total_ms += res.timing.post_ms;
            return;
        }
        res.reason = "IP 執行失敗,已退回 CPU";
    }

    // ---- 退路:結果一樣正確,只是比較慢 ----
    cv::resize(img, roi, roi.size(), 0, 0, cv::INTER_AREA);
}

Result letterbox(const cv::Mat& img, int input_size) {
    Result res;
    letterbox(img, input_size, res);
    return res;
}


// ============================================================
//  4-5. verify
// ============================================================
VerifyReport verify(const cv::Mat& img, int scale) {
    VerifyReport rep;

    if (img.empty() || img.type() != CV_8UC3) {
        rep.reason = "輸入必須是非空的 CV_8UC3";
        return rep;
    }

    Params p;
    try {
        p = Params::make(img.cols, img.rows, scale);
    } catch (const std::exception&) {
        rep.reason = "尺寸不符 IP 限制";
        return rep;
    }

    cv::Mat out;
    Timing t{};
    if (!downscale(img, scale, out, &t)) {
        rep.reason = "IP 執行失敗";
        return rep;
    }
    rep.ms  = t.total_ms;
    rep.ran = true;

    // 軟體模型:與 HLS 相同的整數運算,(sum * inv_scale) >> 16
    const int s = scale;
    rep.total = static_cast<uint64_t>(p.out_w) * p.out_h * 3u;

    for (int oy = 0; oy < static_cast<int>(p.out_h); ++oy) {
        const uint8_t* orow = out.ptr<uint8_t>(oy);
        for (int ox = 0; ox < static_cast<int>(p.out_w); ++ox) {
            for (int ch = 0; ch < 3; ++ch) {
                uint32_t sum = 0;
                for (int dy = 0; dy < s; ++dy) {
                    const uint8_t* irow = img.ptr<uint8_t>(oy * s + dy);
                    for (int dx = 0; dx < s; ++dx)
                        sum += irow[(ox * s + dx) * 3 + ch];
                }
                const int expect = static_cast<int>((sum * p.inv_scale) >> 16);
                const int diff   = std::abs(expect - static_cast<int>(orow[ox * 3 + ch]));
                if (diff) {
                    ++rep.mismatches;
                    if (diff > rep.max_diff) rep.max_diff = diff;
                }
            }
        }
    }

    rep.reason = rep.mismatches ? "與軟體模型不符" : "完全一致";
    return rep;
}

}  // namespace resize
}  // namespace hls