// hls_uyvy_resize.cpp — uyvy_resize_0 的硬體資訊、driver 與 API 實作
//
// 檔案結構(與 hls_resize.cpp 相同):
//   1. 硬體資訊    位址、暫存器 offset
//   2. Kernel      繼承 AxiLiteIp,只負責寫參數暫存器
//   3. device()    IpHolder 單例 + 註冊 clear_buffers
//   4. API         include/hls_uyvy_resize.h 宣告的函式

#include "hls_uyvy_resize.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace hls {
namespace uyvy_resize {

// 內部型別(AxiLiteIp、IpHolder、stage_input…)在 hls::detail
using namespace hls::detail;

namespace {

// ============================================================
//  1. 硬體資訊
// ============================================================

// uyvyResizeDPU.xsa(pl.dtsi):
//   uyvy_resize_0: uyvy_resize@a0000000 { reg = <0x0 0xa0000000 0x0 0x10000>;
//                                         interrupts = <0 89 4>; }   // pl_ps_irq0[0]
// 有了位址:UIO 名稱對不上時會改用位址尋找,use_devmem() 不帶參數也能用。
constexpr uint64_t    kCtrlPhys = 0xA0000000ull;
constexpr size_t      kCtrlSpan = 0x10000;

// uio_pdrv_genirq 用的是裝置樹「節點名去掉 @位址」,不是 label:
//   uyvy_resize_0: uyvy_resize@a0010000 { ... }   → UIO 名稱是 "uyvy_resize"
constexpr const char* kUioName  = "uyvy_resize";

// 參數暫存器(xuyvy_resize_hw.h);0x00–0x0C 是共用的 hls::detail::ap_ctrl
namespace reg {
constexpr uint32_t UYVY_PTR   = 0x10;   // 64-bit: 0x10 / 0x14,輸入 UYVY 的實體位址
constexpr uint32_t RGB_PTR    = 0x1c;   // 64-bit: 0x1c / 0x20,輸出 RGB 的實體位址
constexpr uint32_t IMG_W      = 0x28;   // 12 bit
constexpr uint32_t IMG_H      = 0x30;   // 12 bit
constexpr uint32_t SCALE_MODE = 0x38;   // 1 bit:0 = 縮小 2 倍, 1 = 縮小 3 倍
}  // namespace reg

constexpr uint32_t kMask12 = 0xFFFu;


// ============================================================
//  2. Kernel —— 暫存器層
// ============================================================
class Kernel : public AxiLiteIp {
public:
    Kernel(const std::string& name, uint64_t ctrl_phys, bool allow_devmem)
        : AxiLiteIp(name, ctrl_phys, allow_devmem, kCtrlSpan) {}

    void program(const Params& p, uint64_t in_phys, uint64_t out_phys) {
        wr64(reg::UYVY_PTR,  in_phys);
        wr64(reg::RGB_PTR,   out_phys);
        wr(reg::IMG_W,       p.in_w & kMask12);
        wr(reg::IMG_H,       p.in_h & kMask12);
        wr(reg::SCALE_MODE,  p.scale_mode & 1u);
    }

    // 讀回確認。位址對應錯誤、或綁到別顆 IP 時最容易在這裡發現。
    bool readback_ok(const Params& p, uint64_t in_phys, uint64_t out_phys) const {
        const uint64_t in  = rd(reg::UYVY_PTR) | (uint64_t(rd(reg::UYVY_PTR + 4)) << 32);
        const uint64_t out = rd(reg::RGB_PTR)  | (uint64_t(rd(reg::RGB_PTR  + 4)) << 32);
        return in == in_phys && out == out_phys &&
               (rd(reg::IMG_W) & kMask12) == p.in_w &&
               (rd(reg::IMG_H) & kMask12) == p.in_h &&
               (rd(reg::SCALE_MODE) & 1u) == p.scale_mode;
    }
};


// ============================================================
//  3. 裝置單例
// ============================================================
IpHolder<Kernel>& device() {
    static IpHolder<Kernel> d(kUioName, kCtrlPhys);
    return d;
}

const CleanupRegistrar cleanup_registration{[] { device().clear_cache(); }};

// 執行期的錯誤(逾時、讀回不符…),受 device().mutex() 保護
std::string& run_error() {
    static std::string e;
    return e;
}


// ============================================================
//  CPU 位元一致模型
//
//  UYVY → RGB(與 IP 的 cvt_pair 一致):
//    d = U - 128,e = V - 128,權重單位 1/256,算術右移 = 向下取整
//      R = clamp(Y + floor(359*e / 256))
//      G = clamp(Y - floor((88*d + 183*e) / 256))   兩項先相加再截斷一次
//      B = clamp(Y + floor(454*d / 256))
//  box filter:(sum * inv_scale) >> 16,2 倍 16384、3 倍 7282
// ============================================================
inline int floor_div256(int x) {
    return (x >= 0) ? (x >> 8) : -((-x + 255) >> 8);
}

inline uint8_t clamp_u8(int v) {
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// 一列 UYVY → 一列 RGB
void convert_row(const uint8_t* src, uint8_t* dst, int width) {
    for (int x = 0; x < width; x += 2, src += 4, dst += 6) {
        const int u = src[0], y0 = src[1], v = src[2], y1 = src[3];
        const int d = u - 128, e = v - 128;
        const int r_off = floor_div256(359 * e);
        const int g_off = floor_div256(88 * d + 183 * e);
        const int b_off = floor_div256(454 * d);
        dst[0] = clamp_u8(y0 + r_off); dst[1] = clamp_u8(y0 - g_off); dst[2] = clamp_u8(y0 + b_off);
        dst[3] = clamp_u8(y1 + r_off); dst[4] = clamp_u8(y1 - g_off); dst[5] = clamp_u8(y1 + b_off);
    }
}

bool check_input(const cv::Mat& uyvy, int scale, Params& p, const char*& why) {
    if (uyvy.empty() || uyvy.type() != CV_8UC2) {
        why = "輸入必須是非空的 CV_8UC2(UYVY)";
        return false;
    }
    if (scale != 2 && scale != 3) {
        why = "scale 只支援 2 或 3";
        return false;
    }
    if (!Params::feasible(uyvy.cols, uyvy.rows, scale)) {
        why = "尺寸不符 IP 限制(見 Params::make 的錯誤訊息)";
        return false;
    }
    p = Params::make(uyvy.cols, uyvy.rows, scale);
    return true;
}

}  // namespace


// ============================================================
//  4-1. Params
// ============================================================
bool Params::feasible(uint32_t in_w, uint32_t in_h, uint32_t scale) {
    try {
        make(in_w, in_h, scale);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

Params Params::make(uint32_t in_w, uint32_t in_h, uint32_t scale) {
    if (scale != 2 && scale != 3)
        throw std::invalid_argument("scale 只支援 2 或 3");
    if (in_w == 0 || in_h == 0 || in_w > kDimMax || in_h > kDimMax)
        throw std::invalid_argument("img_w / img_h 必須在 1 ~ 4095 之間(12-bit)");
    if (in_w % 16)
        throw std::invalid_argument("img_w 必須是 16 的倍數");
    if (scale == 3 && in_w % 48)
        throw std::invalid_argument("3 倍時 img_w 必須是 48 的倍數");
    if (in_h % scale)
        throw std::invalid_argument(scale == 3 ? "3 倍時 img_h 必須是 3 的倍數"
                                               : "2 倍時 img_h 必須是偶數");

    Params p;
    p.in_w       = in_w;
    p.in_h       = in_h;
    p.scale      = scale;
    p.scale_mode = (scale == 3) ? 1u : 0u;
    p.out_w      = in_w / scale;
    p.out_h      = in_h / scale;

    if (p.out_w > kOutWMax)
        throw std::invalid_argument("輸出寬度超過 960(IP 的 line buffer 上限)");

    p.in_bytes  = in_w * in_h * 2u;
    p.out_bytes = p.out_w * p.out_h * 3u;
    p.out_words = (p.out_bytes + 15u) / 16u;
    return p;
}

std::string Params::describe() const {
    char b[256];
    std::snprintf(b, sizeof b,
                  "UYVY %ux%u -> RGB %ux%u (%u 倍, scale_mode=%u)\n"
                  "  in_bytes  = %u\n"
                  "  out_bytes = %u\n"
                  "  out_words = %u%s",
                  in_w, in_h, out_w, out_h, scale, scale_mode,
                  in_bytes, out_bytes, out_words,
                  (out_bytes % 16) ? "(最後一個 word 補 0)" : "");
    return b;
}


// ============================================================
//  4-2. 裝置設定與查詢
// ============================================================
void configure()                        { device().set_name(device().default_name()); }
void configure(const std::string& name) { device().set_name(name); }
void use_devmem()                       { device().set_ctrl_phys(device().default_phys(), true); }
void use_devmem(uint64_t ctrl_phys)     { device().set_ctrl_phys(ctrl_phys, true); }
void set_ctrl_phys(uint64_t ctrl_phys)  { device().set_ctrl_phys(ctrl_phys, false); }

std::string device_info() { return device().info(); }
bool        using_irq()   { return device().using_irq(); }
bool        available()   { return device().open() && pool_available(); }

std::string last_error() {
    std::string e;
    {
        std::lock_guard<std::mutex> lk(device().mutex());
        e = run_error();
    }
    if (e.empty()) e = device().error();
    if (e.empty()) e = pool_error();
    return e;
}


// ============================================================
//  4-3. downscale —— 只用 IP
// ============================================================
bool downscale(const cv::Mat& uyvy, int scale, cv::Mat& rgb,
               Timing* timing, int timeout_ms) {
    Params p;
    const char* why = "";
    if (!check_input(uyvy, scale, p, why)) return false;

    auto& d = device();
    std::lock_guard<std::mutex> lk(d.mutex());
    run_error().clear();

    Kernel* ip = d.try_open();
    if (!ip) return false;
    if (!ip->is_idle()) { run_error() = "IP 忙碌中(上一次還沒結束?)"; return false; }

    auto pool = pool_ptr();
    if (!pool) return false;

    // 輸出 buffer 的配置量已補到 64 byte 的倍數,
    // 所以 IP 在最後一個 word 補 0 寫出時不會越界。
    DmaMat* dst = d.cache().get(pool, p.out_h, p.out_w, CV_8UC3);
    if (!dst) { run_error() = "DMA pool 配置輸出 buffer 失敗(空間不足?)"; return false; }

    const Stopwatch total;

    DmaInput in;
    if (!stage_input(pool, d.cache(), uyvy, in)) {
        run_error() = "DMA pool 配置輸入 buffer 失敗(空間不足?)";
        return false;
    }

    ip->program(p, in.phys, dst->phys());
    if (!ip->readback_ok(p, in.phys, dst->phys())) {
        run_error() = "暫存器讀回不符 —— 綁到的裝置可能不是 uyvy_resize,請檢查 device_info()";
        return false;
    }

    const Stopwatch run;
    if (!ip->run(timeout_ms)) { run_error() = "等待 IP 完成逾時"; return false; }
    const double run_ms = run.ms();

    const double sync_out_ms = finish_output(*pool, dst->mat().data, p.out_words * 16u);

    rgb = dst->mat();                 // 不複製,資料仍在 DMA 記憶體

    if (timing) {
        timing->copy_ms   = in.copy_ms;
        timing->sync_ms   = in.sync_ms + sync_out_ms;
        timing->run_ms    = run_ms;
        timing->post_ms   = 0;
        timing->total_ms  = total.ms();
        timing->zero_copy = in.zero_copy;
    }
    return true;
}


// ============================================================
//  4-4. reference —— CPU 位元一致模型
// ============================================================
bool reference(const cv::Mat& uyvy, int scale, cv::Mat& rgb) {
    Params p;
    const char* why = "";
    if (!check_input(uyvy, scale, p, why)) return false;

    const int s   = scale;
    const int W   = static_cast<int>(p.in_w);
    const int ow  = static_cast<int>(p.out_w);
    const int oh  = static_cast<int>(p.out_h);
    const uint32_t inv = (s == 3) ? 7282u : 16384u;
    const size_t row_bytes = static_cast<size_t>(W) * 3;

    cv::Mat out(oh, ow, CV_8UC3);
    std::vector<uint8_t> rows(row_bytes * s);    // 一個輸出列需要的 s 列 RGB

    for (int oy = 0; oy < oh; ++oy) {
        for (int dy = 0; dy < s; ++dy)
            convert_row(uyvy.ptr<uint8_t>(oy * s + dy), rows.data() + dy * row_bytes, W);

        uint8_t* q = out.ptr<uint8_t>(oy);
        for (int ox = 0; ox < ow; ++ox) {
            for (int c = 0; c < 3; ++c) {
                uint32_t sum = 0;
                for (int dy = 0; dy < s; ++dy) {
                    const uint8_t* r = rows.data() + dy * row_bytes + (ox * s) * 3 + c;
                    for (int dx = 0; dx < s; ++dx) sum += r[dx * 3];
                }
                q[ox * 3 + c] = static_cast<uint8_t>((sum * inv) >> 16);
            }
        }
    }
    rgb = out;
    return true;
}


// ============================================================
//  4-5. process —— 主要介面
// ============================================================
Result process(const cv::Mat& uyvy, int scale) {
    Result res;
    Params p;
    if (!check_input(uyvy, scale, p, res.reason)) return res;

    cv::Mat view;
    if (downscale(uyvy, scale, view, &res.timing)) {
        const Stopwatch post;
        res.rgb = view.clone();       // 從 DMA buffer 複製出來,下次呼叫不會蓋掉
        res.timing.post_ms   = post.ms();
        res.timing.total_ms += res.timing.post_ms;
        res.used_ip = true;
        res.reason  = "IP";
        return res;
    }

    // ---- 退路:結果位元完全相同,只是比較慢 ----
    const Stopwatch cpu;
    reference(uyvy, scale, res.rgb);
    res.timing = Timing{};
    res.timing.total_ms = cpu.ms();
    res.reason = "IP 不可用,已改用 CPU(結果相同);原因見 last_error()";
    return res;
}


// ============================================================
//  4-5b. plan_for / letterbox
// ============================================================
Plan plan_for(uint32_t in_w, uint32_t in_h, uint32_t need_w, uint32_t need_h) {
    Plan plan;

    if (need_w == 0 || need_h == 0) {
        plan.reason = "目標尺寸為 0";
        return plan;
    }
    // IP 最小倍率是 2,連 2 倍都不到就只能交給 CPU
    if (in_w < need_w * 2 || in_h < need_h * 2) {
        plan.reason = "縮放倍率不足 2 倍,改用 CPU";
        return plan;
    }

    const char* why = "尺寸不符 IP 限制";
    for (uint32_t s : {3u, 2u}) {
        if (in_w / s < need_w || in_h / s < need_h) { why = "此倍率會縮過頭"; continue; }
        if (!Params::feasible(in_w, in_h, s))       { why = "尺寸不符 IP 限制(見 Params::make)"; continue; }

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

void letterbox(const cv::Mat& uyvy, int input_size, LetterboxResult& res,
               bool allow_ip) {
    CV_Assert(!uyvy.empty() && uyvy.type() == CV_8UC2 && input_size > 0);

    const int orig_h = uyvy.rows, orig_w = uyvy.cols;

    // ---- 幾何:與 hls::resize::letterbox 完全相同 ----
    float r = std::min(static_cast<float>(input_size) / orig_h,
                       static_cast<float>(input_size) / orig_w);
    r = std::min(r, 1.0f);                                   // 不放大

    const int pad_w = static_cast<int>(std::round(orig_w * r));
    const int pad_h = static_cast<int>(std::round(orig_h * r));

    const float dw = (input_size - pad_w) / 2.0f;
    const float dh = (input_size - pad_h) / 2.0f;

    const int top  = static_cast<int>(std::round(dh - 0.1f));
    const int left = static_cast<int>(std::round(dw - 0.1f));

    res.ratio   = {r, r};
    res.pad     = {dw, dh};
    res.content = cv::Rect(left, top, pad_w, pad_h);

    res.img.create(input_size, input_size, CV_8UC3);
    // 只塗黑邊,不整張 setTo —— 內容區馬上會被覆蓋
    if (top > 0)
        res.img(cv::Rect(0, 0, input_size, top)).setTo(cv::Scalar::all(0));
    if (top + pad_h < input_size)
        res.img(cv::Rect(0, top + pad_h, input_size, input_size - top - pad_h))
            .setTo(cv::Scalar::all(0));
    if (left > 0)
        res.img(cv::Rect(0, top, left, pad_h)).setTo(cv::Scalar::all(0));
    if (left + pad_w < input_size)
        res.img(cv::Rect(left + pad_w, top, input_size - left - pad_w, pad_h))
            .setTo(cv::Scalar::all(0));
    cv::Mat roi = res.img(res.content);

    res.used_ip   = false;
    res.ip_scale  = 0;
    res.mid_w     = orig_w;
    res.mid_h     = orig_h;
    res.zero_copy = false;
    res.timing    = Timing{};
    res.ip_ms     = 0.0;

    const Plan plan = plan_for(orig_w, orig_h, pad_w, pad_h);
    res.reason = plan.reason;
    if (plan.use_ip && !allow_ip) res.reason = "IP 已由呼叫端停用,使用 CPU";

    if (plan.use_ip && allow_ip) {
        cv::Mat mid;                          // 指向 IP 的 DMA 輸出,下次呼叫會被蓋
        const Stopwatch sw;
        const bool ok = downscale(uyvy, static_cast<int>(plan.scale), mid, &res.timing);
        res.ip_ms = sw.ms();
        if (ok) {
            res.used_ip   = true;
            res.ip_scale  = static_cast<int>(plan.scale);
            res.mid_w     = mid.cols;
            res.mid_h     = mid.rows;
            res.zero_copy = res.timing.zero_copy;

            const Stopwatch post;
            if (plan.exact)
                mid.copyTo(roi);              // IP 一次到位,只剩一次 copy
            else
                cv::resize(mid, roi, roi.size(), 0, 0, cv::INTER_AREA);
            res.timing.post_ms   = post.ms();
            res.timing.total_ms += res.timing.post_ms;
            return;
        }
        res.reason = "IP 執行失敗,已退回 CPU(原因見 last_error())";
    }

    // ---- 退路:OpenCV 轉色 + 縮放 ----
    // 與 IP 不是位元一致(係數與捨入不同),但對推論沒有實質影響。
    const Stopwatch cpu;
    thread_local cv::Mat full_rgb;
    cv::cvtColor(uyvy, full_rgb, cv::COLOR_YUV2RGB_UYVY);
    if (full_rgb.size() == roi.size())
        full_rgb.copyTo(roi);
    else
        cv::resize(full_rgb, roi, roi.size(), 0, 0, cv::INTER_AREA);
    res.timing.post_ms  = cpu.ms();
    res.timing.total_ms = res.timing.post_ms;
}

LetterboxResult letterbox(const cv::Mat& uyvy, int input_size, bool allow_ip) {
    LetterboxResult res;
    letterbox(uyvy, input_size, res, allow_ip);
    return res;
}


// ============================================================
//  4-6. verify
// ============================================================
VerifyReport verify(const cv::Mat& uyvy, int scale) {
    VerifyReport rep;
    Params p;
    const char* why = "";
    if (!check_input(uyvy, scale, p, why)) {
        rep.reason = why;
        return rep;
    }

    cv::Mat hw, ref;
    Timing t{};
    if (!downscale(uyvy, scale, hw, &t)) {
        rep.reason = "IP 執行失敗: " + last_error();
        return rep;
    }
    rep.ran = true;
    rep.ms  = t.total_ms;
    reference(uyvy, scale, ref);

    rep.total = p.out_bytes;
    for (int y = 0; y < static_cast<int>(p.out_h); ++y) {
        const uint8_t* a = hw.ptr<uint8_t>(y);
        const uint8_t* b = ref.ptr<uint8_t>(y);
        for (int i = 0; i < static_cast<int>(p.out_w) * 3; ++i) {
            const int diff = std::abs(int(a[i]) - int(b[i]));
            if (diff) {
                ++rep.mismatches;
                ++rep.ch_err[i % 3];
                rep.max_diff = std::max(rep.max_diff, diff);
            }
        }
    }

    // IP 會把最後一個 word 補滿 16 byte 寫出,補位應該是 0。
    // hw 指向 DMA buffer,緊接在影像資料之後的就是補位區(在配置範圍內)。
    const uint8_t* pad = hw.data + p.out_bytes;
    for (uint32_t i = 0; i < p.out_words * 16u - p.out_bytes; ++i)
        if (pad[i]) ++rep.pad_dirty;

    if (rep.passed()) {
        rep.reason = "完全一致";
    } else if (rep.pad_dirty && !rep.mismatches) {
        rep.reason = "影像一致,但最後一個 word 的補位不是 0";
    } else if (rep.ch_err[0] && rep.ch_err[2] && !rep.ch_err[1] && rep.max_diff > 50) {
        rep.reason = "R 與 B 錯、G 對 —— byte 順序(RGB/BGR)可能反了";
    } else if (rep.mismatches == rep.total && rep.max_diff > 100) {
        rep.reason = "全錯且差值大 —— 檢查 cache 同步、參數或 bitstream 版本";
    } else if (rep.max_diff <= 2) {
        rep.reason = "差值很小 —— 檢查 uyvy2rgb 的截斷方向或權重量化";
    } else {
        rep.reason = "與 CPU 模型不符";
    }
    return rep;
}


cv::Mat make_test_uyvy(int width, int height) {
    cv::Mat m(height, width, CV_8UC2);
    for (int y = 0; y < height; ++y) {
        uint8_t* q = m.ptr<uint8_t>(y);
        for (int x = 0; x < width; x += 2, q += 4) {
            q[0] = static_cast<uint8_t>(x * 3 + y);              // U
            q[1] = static_cast<uint8_t>(x + y * 3);              // Y0
            q[2] = static_cast<uint8_t>(255 - x * 2 + y * 5);    // V
            q[3] = static_cast<uint8_t>(x * 5 + y);              // Y1
        }
    }
    return m;
}

}  // namespace uyvy_resize
}  // namespace hls