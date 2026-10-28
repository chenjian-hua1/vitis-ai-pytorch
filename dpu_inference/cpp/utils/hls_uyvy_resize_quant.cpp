// hls_uyvy_resize_quant.cpp — uyvy_resize_quant 的硬體資訊、driver 與 API 實作
//
// 檔案結構(與 hls_uyvy_resize.cpp 相同):
//   1. 硬體資訊    位址、暫存器 offset
//   2. Kernel      繼承 AxiLiteIp,只負責寫參數暫存器
//   3. device()    IpHolder 單例 + 註冊 clear_buffers
//   4. API         include/hls_uyvy_resize_quant.h 宣告的函式
//
// 本版修改(量化):
//   - quant_byte() 原本是佔位用的 int8(v - 128),與 IP / norm_and_fix_letterbox 的公式完全不同。
//     改為 quant_byte(v, ch):直接查 quantize_lut.h 的 kLut(與 IP 的 ROM 是同一份),
//     即 norm_and_fix_letterbox 在 fix_point = kFixPoint 時的結果(每通道不同的 mean/std)。
//   - 受影響的地方一併修正:verify 的參考值、letterbox 的 quant 黑邊(每通道不同)、
//     零頭縮放後的 CPU 重新量化、CPU 退路。
//   - 新增 ip_fix_point() / quant_pad(),讓呼叫端比對 xmodel 的 fix_point、取得黑邊值。
//   - verify 的 quant 不符時,自動判斷 IP 實際相當於哪個 fix_point、或是否 R/B 對調。
//
// 本版修改(letterbox 效能):
//   - quant 可直接寫進呼叫端提供的 Mat(例如 DPU 輸入緩衝),省掉 res.quant → DPU 輸入的一次複製;
//     也可以完全不產生 quant(呼叫端改用 norm_and_fix_letterbox 時)。
//   - 黑邊不再每幀重填:同一塊 buffer、同一個內容區、同一個填值只填一次
//     (持有 buffer 的參照,位址不會被重新配置給別人,不會誤判)。

#include "hls_uyvy_resize_quant.h"
#include "../../../hls_ip/src/impl/quantize_lut.h"        // 與 HLS 共用:kLut / kFixPoint(ARM 端編譯時請拿掉其中的 #include <ap_int.h>)

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>      // access()

namespace hls {
namespace uyvy_resize_quant {

using namespace hls::detail;

namespace {

// ============================================================
//  1. 硬體資訊
// ============================================================

// 來自板子的實測:/proc/device-tree/amba_pl/uyvyResizeQuant@a0000000
//   位址 0xA0000000 已確認;節點名是駝峰式 "uyvyResizeQuant"(不是 uyvy_resize_quant)
constexpr uint64_t    kCtrlPhys = 0xA0000000ull;
constexpr size_t      kCtrlSpan = 0x10000;

// uio_pdrv_genirq 的名稱 = 裝置樹節點名去掉 @位址
constexpr const char* kUioName  = "uyvyResizeQuant";

// UIO 找不到時改用 /dev/mem(需 root、輪詢、無中斷)。
// 只有 /proc/device-tree 找得到這顆 IP 的節點時才啟用,避免對不存在的 PL 位址存取造成卡死。
constexpr bool kDevmemFallback = true;

// 參數暫存器(xuyvyresizequant_hw.h);0x00–0x0C 是共用的 hls::detail::ap_ctrl
namespace reg {
constexpr uint32_t UYVY_PTR   = 0x10;   // 64-bit: 0x10 / 0x14,輸入 UYVY 的實體位址
constexpr uint32_t RESIZE_PTR = 0x1c;   // 64-bit: 0x1c / 0x20,輸出 RGB 的實體位址
constexpr uint32_t QUANT_PTR  = 0x28;   // 64-bit: 0x28 / 0x2c,輸出 quant 的實體位址
constexpr uint32_t IMG_W      = 0x34;   // 12 bit
constexpr uint32_t IMG_H      = 0x3c;   // 12 bit
constexpr uint32_t SCALE_MODE = 0x44;   // 1 bit:0 = 縮小 2 倍, 1 = 縮小 3 倍
}  // namespace reg

constexpr uint32_t kMask12 = 0xFFFu;


// ============================================================
//  2. Kernel —— 暫存器層
// ============================================================
class Kernel : public AxiLiteIp {
public:
    Kernel(const std::string& name, uint64_t ctrl_phys, bool allow_devmem)
        : AxiLiteIp(name, ctrl_phys, allow_devmem, kCtrlSpan) {}

    void program(const Params& p, uint64_t in_phys, uint64_t out_phys, uint64_t quant_phys) {
        wr64(reg::UYVY_PTR,   in_phys);
        wr64(reg::RESIZE_PTR, out_phys);
        wr64(reg::QUANT_PTR,  quant_phys);
        wr(reg::IMG_W,        p.in_w & kMask12);
        wr(reg::IMG_H,        p.in_h & kMask12);
        wr(reg::SCALE_MODE,   p.scale_mode & 1u);
    }

    // 讀回確認:位址對應錯誤、或綁到別顆 IP 時最容易在這裡發現。
    bool readback_ok(const Params& p, uint64_t in_phys, uint64_t out_phys,
                     uint64_t quant_phys) const {
        auto rd64 = [this](uint32_t off) {
            return uint64_t(rd(off)) | (uint64_t(rd(off + 4)) << 32);
        };
        return rd64(reg::UYVY_PTR)   == in_phys &&
               rd64(reg::RESIZE_PTR) == out_phys &&
               rd64(reg::QUANT_PTR)  == quant_phys &&
               (rd(reg::IMG_W) & kMask12) == p.in_w &&
               (rd(reg::IMG_H) & kMask12) == p.in_h &&
               (rd(reg::SCALE_MODE) & 1u) == p.scale_mode;
    }
};


// ============================================================
//  3. 裝置單例
// ============================================================
bool overlay_has_node() {
    char node[96];
    std::snprintf(node, sizeof node, "%s@%llx", kUioName,
                  static_cast<unsigned long long>(kCtrlPhys));
    for (const char* bus : {"/proc/device-tree/axi/", "/proc/device-tree/amba/",
                            "/proc/device-tree/amba_pl/", "/proc/device-tree/"}) {
        if (::access((std::string(bus) + node).c_str(), F_OK) == 0) return true;
    }
    return false;
}

IpHolder<Kernel>& device() {
    static IpHolder<Kernel> d(kUioName, kCtrlPhys);
    static const bool init = [] {
        if (kDevmemFallback && kCtrlPhys != 0 && overlay_has_node())
            d.set_ctrl_phys(kCtrlPhys, true);
        return true;
    }();
    (void)init;
    return d;
}

const CleanupRegistrar cleanup_registration{[] { device().clear_cache(); }};

std::string& run_error() {
    static std::string e;
    return e;
}


// ============================================================
//  黑邊快取
//
//  記錄每塊 buffer 的黑邊是以什麼內容區、什麼值填的;全部相同就跳過。
//  held 持有 buffer 的一份參照(引用計數 +1),確保它不會被釋放,
//  因此「data 相同」一定是同一塊記憶體,不會發生舊 buffer 釋放後新 buffer 配到同一位址的誤判。
//  (外部記憶體包成的 Mat 沒有引用計數,持有它不影響任何事。)
//  thread_local:每個執行緒各自一份,不需要鎖。
// ============================================================
class BorderCache {
public:
    void fill(cv::Mat& m, const cv::Rect& c, const cv::Scalar& v) {
        for (Rec& r : recs_) {
            if (r.held.data != m.data) continue;
            if (r.held.rows == m.rows && r.held.cols == m.cols && r.held.type() == m.type() &&
                r.content == c && r.value == v)
                return;                                        // 已經是正確的黑邊
            paint(m, c, v);                                    // 同一塊 buffer,參數變了 → 重填
            r = Rec{m, c, v};
            return;
        }
        paint(m, c, v);                                        // 新的 buffer → 佔一個位置
        recs_[next_] = Rec{m, c, v};
        next_ = (next_ + 1) % kN;
    }
    void clear() { for (Rec& r : recs_) r = Rec{}; next_ = 0; }

private:
    static void paint(cv::Mat& m, const cv::Rect& c, const cv::Scalar& v) {
        const int x0 = c.x, x1 = c.x + c.width, y0 = c.y, y1 = c.y + c.height;
        if (y0 > 0)      m.rowRange(0, y0).setTo(v);
        if (y1 < m.rows) m.rowRange(y1, m.rows).setTo(v);
        if (y1 > y0) {
            if (x0 > 0)      m(cv::Rect(0,  y0, x0,          y1 - y0)).setTo(v);
            if (x1 < m.cols) m(cv::Rect(x1, y0, m.cols - x1, y1 - y0)).setTo(v);
        }
    }
    struct Rec {
        cv::Mat    held;
        cv::Rect   content{-1, -1, -1, -1};
        cv::Scalar value;
    };
    static constexpr int kN = 4;          // RGB 畫布 + 幾塊 DPU 輸入緩衝,4 個足夠
    Rec recs_[kN];
    int next_ = 0;
};

BorderCache& border_cache() {
    thread_local BorderCache c;
    return c;
}


// ============================================================
//  CPU 位元一致模型
//
//  UYVY → RGB(與 IP 的 cvt_pair 一致):
//    d = U - 128,e = V - 128,權重單位 1/256,算術右移 = 向下取整
//      R = clamp(Y + floor(359*e / 256))
//      G = clamp(Y - floor((88*d + 183*e) / 256))
//      B = clamp(Y + floor(454*d / 256))
//  box filter:(sum * inv_scale) >> 16,2 倍 16384、3 倍 7282
//  quant:quant_byte(v, ch) = kLut(與 IP 的 ROM 同一份)
//        = norm_and_fix_letterbox(fix_point = kFixPoint)
// ============================================================
inline int floor_div256(int x) {
    return (x >= 0) ? (x >> 8) : -((-x + 255) >> 8);
}

inline uint8_t clamp_u8(int v) {
    return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

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

// 把一塊 CV_8UC3 的 RGB 逐 byte 量化到同尺寸的 CV_8SC3(byte 順序 R,G,B → 通道 = 位置 % 3)
void quantize_mat(const cv::Mat& rgb, cv::Mat& q) {
    CV_Assert(rgb.type() == CV_8UC3 && q.type() == CV_8SC3 && rgb.size() == q.size());
    const int n = rgb.cols * 3;
    for (int y = 0; y < rgb.rows; ++y) {
        const uint8_t* s = rgb.ptr<uint8_t>(y);
        int8_t*        d = q.ptr<int8_t>(y);
        for (int i = 0; i < n; i += 3) {
            d[i + 0] = static_cast<int8_t>(kLut.r[s[i + 0]]);
            d[i + 1] = static_cast<int8_t>(kLut.g[s[i + 1]]);
            d[i + 2] = static_cast<int8_t>(kLut.b[s[i + 2]]);
        }
    }
}

// ---- 診斷用:norm_and_fix_letterbox 的公式(任意 fix_point),只在 verify 失敗時使用 ----
int8_t normfix_formula(int v, int ch, int fix_point) {
    static const double mean[3] = {0.485, 0.456, 0.406};
    static const double stdv[3] = {0.229, 0.224, 0.225};
    const double fp = std::ldexp(1.0, fix_point);
    const double y  = v * (1. / (stdv[ch] * 255.) * fp) + (-mean[ch] / stdv[ch] * fp);
    return static_cast<int8_t>(y < -128. ? -128. : (y > 127. ? 127. : y));
}

// IP 的 quant 是否等於某個 fix_point 的 norm_and_fix(swap_rb:R/B 的表對調)。
// rgb 必須是正確的 RGB(verify 中 RGB 已確認一致時才呼叫)。回傳吻合的 fix_point,找不到回傳 INT_MIN。
int match_fix_point(const cv::Mat& rgb, const cv::Mat& q, int out_w, int out_h, bool swap_rb) {
    for (int fp = 0; fp <= 10; ++fp) {
        bool ok = true;
        for (int y = 0; ok && y < out_h; ++y) {
            const uint8_t* s = rgb.ptr<uint8_t>(y);
            const int8_t*  d = q.ptr<int8_t>(y);
            for (int i = 0; i < out_w * 3; ++i) {
                const int c  = i % 3;
                const int lc = swap_rb ? 2 - c : c;          // IP 用來查這個 byte 的表
                if (d[i] != normfix_formula(s[i], lc, fp)) { ok = false; break; }
            }
        }
        if (ok) return fp;
    }
    return INT_MIN;
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
//  量化(與 IP 的 ROM 完全相同)
// ============================================================

// ch:0 = R、1 = G、2 = B
int8_t quant_byte(uint8_t v, int ch) {
    const unsigned char* t = (ch == 0) ? kLut.r : (ch == 1) ? kLut.g : kLut.b;
    return static_cast<int8_t>(t[v]);
}

// IP 的量化 LUT 編譯時的 fix_point;必須等於 xmodel 輸入的 fix_point
int ip_fix_point() {
    return kFixPoint;
}

// 黑色 (0,0,0) 量化後的值(letterbox 的 quant 黑邊),每個通道不同
cv::Scalar quant_pad() {
    return cv::Scalar(quant_byte(0, 0), quant_byte(0, 1), quant_byte(0, 2));
}


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

    p.in_bytes    = in_w * in_h * 2u;
    p.out_bytes   = p.out_w * p.out_h * 3u;
    p.out_words   = (p.out_bytes + 15u) / 16u;
    p.quant_bytes = p.out_bytes;                    // quant 與 RGB 同尺寸、同 word 對齊
    p.quant_words = (p.quant_bytes + 15u) / 16u;
    return p;
}

std::string Params::describe() const {
    char b[384];
    std::snprintf(b, sizeof b,
                  "UYVY %ux%u -> RGB+quant %ux%u (%u 倍, scale_mode=%u, quant fix_point=%d)\n"
                  "  in_bytes    = %u\n"
                  "  out_bytes   = %u (words = %u%s)\n"
                  "  quant_bytes = %u (words = %u)",
                  in_w, in_h, out_w, out_h, scale, scale_mode, kFixPoint,
                  in_bytes,
                  out_bytes, out_words, (out_bytes % 16) ? ",最後一個 word 補 0" : "",
                  quant_bytes, quant_words);
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
//  4-3. downscale —— 只用 IP(同時產生 RGB 與 quant)
//  in_phys == 0:輸入經 stage_input(已在 pool 內就 zero-copy,否則複製進去)
//  in_phys != 0:輸入由呼叫端保證在實體連續、已 clean 的記憶體,直接用
// ============================================================
namespace {
bool downscale_impl(const cv::Mat& uyvy, int scale, cv::Mat& rgb, cv::Mat& quant,
                    Timing* timing, int timeout_ms, uint64_t in_phys) {
    Params p;
    const char* why = "";
    if (!check_input(uyvy, scale, p, why)) {
        std::lock_guard<std::mutex> lk(device().mutex());
        run_error() = why;
        return false;
    }
    if (in_phys && !uyvy.isContinuous()) {
        std::lock_guard<std::mutex> lk(device().mutex());
        run_error() = "downscale_phys:輸入必須是連續的 Mat";
        return false;
    }

    auto& d = device();
    std::lock_guard<std::mutex> lk(d.mutex());
    run_error().clear();

    Kernel* ip = d.try_open();
    if (!ip) return false;
    if (!ip->is_idle()) { run_error() = "IP 忙碌中(上一次還沒結束?)"; return false; }

    auto pool = pool_ptr();
    if (!pool) return false;

    // 輸出 buffer 的配置量已補到 64 byte 的倍數,IP 補 0 寫出最後一個 word 時不會越界。
    // 注意:cache 以 (rows, cols, type) 為鍵,RGB 用 CV_8UC3、quant 用 CV_8SC3,
    // 兩者型別不同才會拿到不同的 buffer(下面還會檢查實體位址以防萬一)。
    DmaMat* dst = d.cache().get(pool, p.out_h, p.out_w, CV_8UC3);
    if (!dst) { run_error() = "DMA pool 配置 RGB 輸出 buffer 失敗(空間不足?)"; return false; }
    DmaMat* qdst = d.cache().get(pool, p.out_h, p.out_w, CV_8SC3);
    if (!qdst) { run_error() = "DMA pool 配置 quant 輸出 buffer 失敗(空間不足?)"; return false; }
    if (dst->phys() == qdst->phys()) {
        run_error() = "RGB 與 quant 輸出拿到同一塊 buffer(cache 以尺寸為鍵?),請改用獨立配置";
        return false;
    }

    const Stopwatch total;

    DmaInput in;
    if (in_phys) {
        in.phys      = in_phys;
        in.data      = uyvy.data;
        in.bytes     = p.in_bytes;
        in.zero_copy = true;
    } else if (!stage_input(pool, d.cache(), uyvy, in)) {
        run_error() = "DMA pool 配置輸入 buffer 失敗(空間不足?)";
        return false;
    }

    ip->program(p, in.phys, dst->phys(), qdst->phys());
    if (!ip->readback_ok(p, in.phys, dst->phys(), qdst->phys())) {
        run_error() = "暫存器讀回不符 —— 綁到的裝置可能不是 uyvy_resize_quant,請檢查 device_info()";
        return false;
    }

    const Stopwatch run;
    if (!ip->run(timeout_ms)) { run_error() = "等待 IP 完成逾時"; return false; }
    const double run_ms = run.ms();

    const double sync_out_ms  = finish_output(*pool, dst->mat().data,  p.out_words   * 16u);
    const double sync_qout_ms = finish_output(*pool, qdst->mat().data, p.quant_words * 16u);

    rgb   = dst->mat();               // 不複製,資料仍在 DMA 記憶體
    quant = qdst->mat();

    if (timing) {
        timing->copy_ms   = in.copy_ms;
        timing->sync_ms   = in.sync_ms + sync_out_ms + sync_qout_ms;
        timing->run_ms    = run_ms;
        timing->post_ms   = 0;
        timing->total_ms  = total.ms();
        timing->zero_copy = in.zero_copy;
    }
    return true;
}
}  // namespace

bool downscale(const cv::Mat& uyvy, int scale, cv::Mat& rgb, cv::Mat& quant,
               Timing* timing, int timeout_ms) {
    return downscale_impl(uyvy, scale, rgb, quant, timing, timeout_ms, 0);
}

bool downscale_phys(const cv::Mat& uyvy, uint64_t in_phys, int scale,
                    cv::Mat& rgb, cv::Mat& quant, Timing* timing, int timeout_ms) {
    if (!in_phys) return false;
    return downscale_impl(uyvy, scale, rgb, quant, timing, timeout_ms, in_phys);
}


// ============================================================
//  4-4. reference —— CPU 位元一致模型(RGB 與 quant 都與 IP 位元一致)
// ============================================================
bool reference(const cv::Mat& uyvy, int scale, cv::Mat& rgb, cv::Mat* quant) {
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
    std::vector<uint8_t> rows(row_bytes * s);

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

    if (quant) {
        cv::Mat qm(oh, ow, CV_8SC3);
        quantize_mat(out, qm);
        *quant = qm;
    }
    return true;
}


// ============================================================
//  4-5. process —— 主要介面
// ============================================================
Result process(const cv::Mat& uyvy, int scale) {
    Result res;
    Params p;
    if (!check_input(uyvy, scale, p, res.reason)) return res;

    cv::Mat view, qview;
    if (downscale(uyvy, scale, view, qview, &res.timing)) {
        const Stopwatch post;
        res.rgb   = view.clone();     // 從 DMA buffer 複製出來,下次呼叫不會蓋掉
        res.quant = qview.clone();
        res.timing.post_ms   = post.ms();
        res.timing.total_ms += res.timing.post_ms;
        res.used_ip = true;
        res.reason  = "IP";
        return res;
    }

    // ---- 退路:RGB 與 quant 都與 IP 位元一致(quant 用同一份 kLut),只是比較慢 ----
    const Stopwatch cpu;
    reference(uyvy, scale, res.rgb, &res.quant);
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
               bool allow_ip, uint64_t in_phys, cv::Mat* quant_dst, bool want_quant) {
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

    // ---- 輸出 buffer ----
    //   RGB  :res.img(尺寸 / 型別相同時 create 不會重新配置)
    //   quant:want_quant = false → 不產生(res.quant 清空)
    //          quant_dst != nullptr → 直接寫進呼叫端的 Mat(例如 DPU 輸入),res.quant 與它共用記憶體
    //          否則 → res.quant
    res.img.create(input_size, input_size, CV_8UC3);
    if (!want_quant) {
        res.quant.release();
    } else if (quant_dst) {
        CV_Assert(quant_dst->type() == CV_8SC3 &&
                  quant_dst->rows == input_size && quant_dst->cols == input_size);
        res.quant = *quant_dst;
    } else {
        res.quant.create(input_size, input_size, CV_8SC3);
    }

    // ---- 黑邊:只在 buffer / 內容區 / 填值改變時才填(通常只有第一幀)----
    //   RGB 黑邊 = 0;quant 黑邊 = quant_pad()(每個通道不同,與 norm_and_fix_letterbox 相同)
    BorderCache& bc = border_cache();
    bc.fill(res.img, res.content, cv::Scalar::all(0));
    if (want_quant) bc.fill(res.quant, res.content, quant_pad());

    cv::Mat roi  = res.img(res.content);
    cv::Mat qroi = want_quant ? res.quant(res.content) : cv::Mat();

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
        cv::Mat mid, qmid;                    // 指向 IP 的 DMA 輸出,下次呼叫會被蓋
        const Stopwatch sw;
        const bool ok = in_phys
            ? downscale_phys(uyvy, in_phys, static_cast<int>(plan.scale), mid, qmid, &res.timing)
            : downscale(uyvy, static_cast<int>(plan.scale), mid, qmid, &res.timing);
        res.ip_ms = sw.ms();
        if (ok) {
            res.used_ip   = true;
            res.ip_scale  = static_cast<int>(plan.scale);
            res.mid_w     = mid.cols;
            res.mid_h     = mid.rows;
            res.zero_copy = res.timing.zero_copy;

            const Stopwatch post;
            if (plan.exact) {
                mid.copyTo(roi);                          // IP 一次到位:RGB 一次 copy
                if (want_quant) qmid.copyTo(qroi);        // quant 一次 copy(quant_dst 時直接進 DPU 輸入)
            } else {
                cv::resize(mid, roi, roi.size(), 0, 0, cv::INTER_AREA);
                if (want_quant) quantize_mat(roi, qroi);  // 縮過之後 IP 的 quant 已不適用,由 CPU 用同一份 kLut 重新量化
            }
            res.timing.post_ms   = post.ms();
            res.timing.total_ms += res.timing.post_ms;
            return;
        }
        res.reason = "IP 執行失敗,已退回 CPU(原因見 last_error())";
    }

    // ---- 退路:OpenCV 轉色 + 縮放 + CPU 量化(kLut)----
    // RGB 與 IP 不是位元一致(係數與捨入不同),但對推論沒有實質影響;
    // quant 對「這張 RGB」而言與 IP / norm_and_fix 的公式相同。
    const Stopwatch cpu;
    thread_local cv::Mat full_rgb;
    cv::cvtColor(uyvy, full_rgb, cv::COLOR_YUV2RGB_UYVY);
    if (full_rgb.size() == roi.size())
        full_rgb.copyTo(roi);
    else
        cv::resize(full_rgb, roi, roi.size(), 0, 0, cv::INTER_AREA);
    if (want_quant) quantize_mat(roi, qroi);
    res.timing.post_ms  = cpu.ms();
    res.timing.total_ms = res.timing.post_ms;
}

void letterbox_invalidate_borders() {
    border_cache().clear();
}

LetterboxResult letterbox(const cv::Mat& uyvy, int input_size, bool allow_ip,
                          uint64_t in_phys) {
    LetterboxResult res;
    letterbox(uyvy, input_size, res, allow_ip, in_phys, nullptr, true);
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

    cv::Mat hw, hwq, ref, refq;
    Timing t{};
    if (!downscale(uyvy, scale, hw, hwq, &t)) {
        rep.reason = "IP 執行失敗: " + last_error();
        return rep;
    }
    rep.ran = true;
    rep.ms  = t.total_ms;
    reference(uyvy, scale, ref, &refq);

    // ---- RGB ----
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

    // ---- quant(參考值 = kLut,即 norm_and_fix_letterbox 在 fix_point = kFixPoint)----
    rep.q_total = p.quant_bytes;
    for (int y = 0; y < static_cast<int>(p.out_h); ++y) {
        const int8_t* a = hwq.ptr<int8_t>(y);
        const int8_t* b = refq.ptr<int8_t>(y);
        for (int i = 0; i < static_cast<int>(p.out_w) * 3; ++i) {
            const int diff = std::abs(int(a[i]) - int(b[i]));
            if (diff) {
                ++rep.q_mismatches;
                ++rep.q_ch_err[i % 3];
                rep.q_max_diff = std::max(rep.q_max_diff, diff);
            }
        }
    }

    // 最後一個 word 的補位應為 0(DMA buffer 緊接在影像資料之後)
    {
        const uint8_t* pad = hw.data + p.out_bytes;
        for (uint32_t i = 0; i < p.out_words * 16u - p.out_bytes; ++i)
            if (pad[i]) ++rep.pad_dirty;
        const uint8_t* qpad = reinterpret_cast<const uint8_t*>(hwq.data) + p.quant_bytes;
        for (uint32_t i = 0; i < p.quant_words * 16u - p.quant_bytes; ++i)
            if (qpad[i]) ++rep.q_pad_dirty;
    }

    // ---- 診斷訊息 ----
    char msg[256];
    if (rep.passed()) {
        std::snprintf(msg, sizeof msg, "RGB 與 quant 皆完全一致(quant fix_point=%d)", kFixPoint);
        rep.reason = msg;
    } else if (rep.rgb_ok() && !rep.quant_ok()) {
        if (!rep.q_mismatches) {
            rep.reason = "RGB 對,但 quant 最後一個 word 的補位不是 0";
        } else {
            // RGB 已確認正確 → 用它推算 IP 實際用的量化表
            const int ow = static_cast<int>(p.out_w), oh = static_cast<int>(p.out_h);
            const int fp      = match_fix_point(hw, hwq, ow, oh, false);
            const int fp_swap = match_fix_point(hw, hwq, ow, oh, true);
            if (fp != INT_MIN)
                std::snprintf(msg, sizeof msg,
                              "RGB 對、quant 不符 —— IP 的 LUT 是 fix_point=%d,軟體的 quantize_lut.h 是 %d:"
                              "bitstream 與軟體版本不一致", fp, kFixPoint);
            else if (fp_swap != INT_MIN)
                std::snprintf(msg, sizeof msg,
                              "RGB 對、quant 不符 —— IP 的量化 R/B 查表對調(fix_point=%d),"
                              "檢查 quantize_core 的位元位置", fp_swap);
            else
                std::snprintf(msg, sizeof msg,
                              "RGB 對、quant 不符 —— 不等於任何 fix_point 的 norm_and_fix,"
                              "檢查 quantize_core 與 bitstream 版本");
            rep.reason = msg;
        }
    } else if (!rep.rgb_ok() && rep.quant_ok()) {
        rep.reason = "RGB 不符但 quant 一致(異常)—— 檢查 resize_axi_bus 的位址與 cache 同步";
    } else if (rep.pad_dirty && !rep.mismatches) {
        rep.reason = "影像一致,但 RGB 最後一個 word 的補位不是 0";
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

}  // namespace uyvy_resize_quant
}  // namespace hls