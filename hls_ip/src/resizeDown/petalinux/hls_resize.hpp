// hls_resize.hpp — resize_kernel_0 的 host 端介面
//
//   namespace hls::resize
//     Params        暫存器參數的計算與檢查(這些數值的唯一來源)
//     Result        letterbox 的輸出
//     configure()   指定 UIO 名稱
//     available()   IP 能不能用
//     letterbox()   主要介面:等比縮放 + 置中補黑邊
//     downscale()   低階介面:純 2x / 3x 整數倍縮小
//
// 命名帶 resize 字樣,之後加別的 IP 時各自一個 namespace:
//   hls::resize::letterbox(...)
//   hls::nms::apply(...)
//   hls::conv::run(...)
// 共用的 DMA pool 在 hls_common.hpp,所有 IP 用同一塊。
//
// 函式沒有取名 resize(),是為了避開查找衝突 —— 若呼叫端寫了
// using namespace hls,一個叫 resize 的 namespace 會讓 resize(...)
// 這種呼叫找到 namespace 名而編譯失敗。

#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "hls_common.hpp"
#include "resize_kernel.hpp"

namespace hls {
namespace resize {

// ============================================================
//  Params —— resize_kernel 的暫存器參數
//  全部依 resize_areaDown.cpp 推導,不要自己另外算。
// ============================================================
struct Params {
    uint32_t total_words   = 0;   // 0x28  輸入的 128-bit word 總數
    uint32_t total_results = 0;   // 0x30  compute_side 產出的結果筆數
    uint32_t out_words     = 0;   // 0x38  輸出的 128-bit word 總數(整張圖)
    uint32_t out_w         = 0;   // 0x40  輸出寬度(pixel)
    uint32_t scale_mode    = 0;   // 0x48  0 = SCALE_2, 1 = SCALE_3
    uint32_t inv_scale     = 0;   // 0x50  65536 / scale²,四捨五入

    uint32_t in_w = 0, in_h = 0, out_h = 0;
    uint32_t in_bytes = 0, out_bytes = 0;
    uint32_t scale = 0;

    static constexpr uint32_t kOutWMax = 960;   // HLS 的 OUT_W_MAX

    // 一次運算產出幾個輸出欄:3 倍 2 欄、2 倍 4 欄
    static constexpr uint32_t n_out_for(uint32_t s) { return (s == 3) ? 2u : 4u; }

    // 快速篩選,不丟例外
    static bool feasible(uint32_t in_w, uint32_t in_h, uint32_t scale) {
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

    // 條件不合會丟 std::invalid_argument,訊息說明是哪一項
    static Params make(uint32_t in_w, uint32_t in_h, uint32_t scale) {
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
            throw std::invalid_argument(
                "out_w 必須是 " + std::to_string(n_out) + " 的倍數");

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

    std::string describe() const {
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
};


// ============================================================
//  Plan —— 由「輸入尺寸 + 目標尺寸」直接推出該怎麼做
//
//  這是 letterbox 唯一需要問的問題:我要從 in 縮到 need,
//  IP 能幫上什麼忙?挑倍率與算參數合在同一處,不會不同步。
// ============================================================
struct Plan {
    bool     use_ip = false;   // IP 派得上用場嗎
    uint32_t scale  = 0;       // 用幾倍(0 = 沒用)
    Params   params{};         // use_ip 時才有意義
    bool     exact  = false;   // IP 輸出剛好等於目標,連二次縮放都免了

    // use_ip = false 時說明原因,方便查為什麼沒吃到硬體加速
    const char* reason = "";
};

// need_w / need_h 是最終想要的尺寸。
// 倍率大的優先,讓 IP 多做一點;挑不到就回傳 use_ip = false 並附上原因。
inline Plan plan_for(uint32_t in_w, uint32_t in_h,
                     uint32_t need_w, uint32_t need_h) {
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
        if (in_w / s < need_w || in_h / s < need_h) {
            why = "此倍率會縮過頭";
            continue;
        }
        if (in_w % s || in_h % s) {
            why = "輸入尺寸無法被倍率整除";
            continue;
        }
        const uint32_t ow = in_w / s;
        if (ow > Params::kOutWMax) {
            why = "中間寬度超過 OUT_W_MAX (960)";
            continue;
        }
        if (ow % Params::n_out_for(s)) {
            why = "中間寬度不是 n_out 的倍數";
            continue;
        }
        if (!Params::feasible(in_w, in_h, s)) {
            why = "位元組數不是 16 的倍數";
            continue;
        }

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


// 各階段耗時,用來判斷瓶頸在哪
struct Timing {
    double copy_ms = 0;   // 把影像搬進 DMA 記憶體(zero-copy 時為 0)
    double sync_ms = 0;   // cache 維護(進 + 出)
    double run_ms  = 0;   // IP 實際運算 + 等待
    double post_ms = 0;   // 零頭的 cv::resize
    double total_ms = 0;
};

// letterbox 的結果
struct Result {
    cv::Mat     img;                // input_size x input_size,已填黑邊
    cv::Point2f ratio{1.f, 1.f};    // 實際縮放比例
    cv::Point2f pad{0.f, 0.f};      // 單邊留白
    cv::Rect    content;            // 有效影像在 img 中的位置

    // ---- 除錯用,不影響幾何 ----
    bool used_ip  = false;          // 這次有沒有真的用到 IP
    int  ip_scale = 0;              // 用了幾倍(0 = 沒用)
    int  mid_w = 0, mid_h = 0;      // IP 輸出的中間尺寸
    double ip_ms = 0.0;             // IP 那一段的實際耗時(含資料搬移)
    Timing timing{};                // 明細
    bool zero_copy = false;         // 輸入是否已在 DMA 記憶體,免去複製
    const char* reason = "";        // 為什麼用了/沒用 IP
};


// resize_kernel_0 的 s_axi_control 實體位址(見 resize_dpu.xsa)
inline constexpr uint64_t kDefaultCtrlPhys = 0xB0000000ull;

namespace detail {

class Device {
public:
    static Device& get() {
        static Device d;
        return d;
    }

    void set_name(const std::string& n) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (n == name_) return;
        name_ = n;
        ip_.reset();
        tried_ = false;
        cache_.clear();
    }

    // 控制暫存器的實體位址。名字對不上時用它找 UIO;
    // allow_devmem 為 true 時,連 UIO 都找不到才改用 /dev/mem。
    void set_ctrl_phys(uint64_t phys, bool allow_devmem) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (phys == ctrl_phys_ && allow_devmem == allow_devmem_) return;
        ctrl_phys_    = phys;
        allow_devmem_ = allow_devmem;
        ip_.reset();
        tried_ = false;
    }

    ResizeKernel* try_open() {
        if (tried_) return ip_.get();
        tried_ = true;
        try {
            ip_ = std::make_unique<ResizeKernel>(name_, ctrl_phys_, allow_devmem_);
        } catch (const std::exception& e) {
            error_ = e.what();
            ip_.reset();
        }
        return ip_.get();
    }

    hls::detail::BufferCache& cache() { return cache_; }
    ResizeKernel* peek() { return ip_.get(); }
    std::mutex& mutex() { return mtx_; }
    const std::string& error() const { return error_; }

    static void clear_cache() { get().cache_.clear(); }

private:
    Device() = default;
    std::string name_ = "resize_kernel_0";
    uint64_t    ctrl_phys_ = kDefaultCtrlPhys;
    bool        allow_devmem_ = false;
    std::string error_;
    bool tried_ = false;
    std::unique_ptr<ResizeKernel> ip_;
    hls::detail::BufferCache cache_;
    std::mutex mtx_;
};

// 讓 hls::clear_buffers() 也能清掉這個模組的快取
inline const hls::detail::CleanupRegistrar reg_cleanup{&Device::clear_cache};

}  // namespace detail


// ---- 設定與查詢 ----

// 指定 UIO 裝置名稱。可省略,預設 resize_kernel_0。
inline void configure(const std::string& uio_name = "resize_kernel_0") {
    detail::Device::get().set_name(uio_name);
}

// 啟用 /dev/mem 後路:UIO 建不起來時,直接映射控制暫存器的實體位址。
// resize_kernel_0 的控制介面在 0xB000_0000(見 resize_dpu.xsa)。
//
// 這是裝置樹還沒設好時的暫時手段。代價:需要 root、沒有中斷(改輪詢)、
// 沒有任何存取保護。裝置樹修好後請把這行拿掉。
inline void use_devmem(uint64_t ctrl_phys = kDefaultCtrlPhys) {
    detail::Device::get().set_ctrl_phys(ctrl_phys, true);
}

// 指定控制暫存器位址(僅用於依位址尋找 UIO,不啟用 /dev/mem)
inline void set_ctrl_phys(uint64_t ctrl_phys) {
    detail::Device::get().set_ctrl_phys(ctrl_phys, false);
}

// 實際綁到的裝置說明,用於確認找對了沒
inline std::string device_info() {
    detail::Device& d = detail::Device::get();
    std::lock_guard<std::mutex> lk(d.mutex());
    ResizeKernel* ip = d.try_open();
    if (!ip) return "未開啟";
    if (!ip->has_irq()) return "/dev/mem(輪詢,無中斷)";

    std::string s = "/dev/uio" + std::to_string(ip->uio_index())
                  + " name=\"" + ip->actual_name() + "\"";
    if (ip->matched_by_addr())
        s += "  [名稱不符,靠位址找到 —— 建議改用這個名字呼叫 configure()]";
    return s;
}

// 目前這顆 IP 是走 UIO(有中斷)還是 /dev/mem(輪詢)
inline bool using_irq() {
    detail::Device& d = detail::Device::get();
    std::lock_guard<std::mutex> lk(d.mutex());
    ResizeKernel* ip = d.try_open();
    return ip && ip->has_irq();
}

// IP 與 DMA pool 是否都可用
inline bool available() {
    detail::Device& d = detail::Device::get();
    std::lock_guard<std::mutex> lk(d.mutex());
    return d.try_open() != nullptr && hls::pool_available();
}

inline std::string last_error() {
    const std::string e = detail::Device::get().error();
    return e.empty() ? hls::pool_error() : e;
}


// ============================================================
//  downscale —— 低階介面,純整數倍縮小
//
//  out 指向 DMA 記憶體裡的快取 buffer(淺拷貝),下次呼叫會被覆寫;
//  要保留請自行 clone()。
//  scale 只能是 2 或 3;條件不合或 IP 不可用時回傳 false,不自動退回 CPU。
// ============================================================
inline bool downscale(const cv::Mat& img, int scale, cv::Mat& out,
                      Timing* timing = nullptr, int timeout_ms = 2000) {
    if (img.empty() || img.type() != CV_8UC3) return false;

    Params p;
    try {
        p = Params::make(img.cols, img.rows, scale);
    } catch (const std::exception&) {
        return false;   // 條件不合,由呼叫端決定要不要退回 CPU
    }

    detail::Device& d = detail::Device::get();
    std::lock_guard<std::mutex> lk(d.mutex());

    ResizeKernel* ip = d.try_open();
    if (!ip) return false;

    auto pool = hls::pool_ptr();          // shared_ptr:確保 pool 活得夠久
    if (!pool) return false;

    DmaMat* dst = d.cache().get(pool, p.out_h, p.out_w, CV_8UC3);
    if (!dst) return false;

    const auto t_begin = std::chrono::steady_clock::now();
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms  = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // ---- 輸入:已經在 pool 裡就直接用,省掉一次 6 MB 複製 ----
    const uint8_t* src_data = img.data;
    const bool packed = img.isContinuous() &&
                        img.step == static_cast<size_t>(img.cols) * 3;
    const bool zero_copy = packed && pool->contains(src_data);

    uint64_t src_phys = 0;
    size_t   src_len  = p.in_bytes;

    auto t0 = now();
    if (zero_copy) {
        src_phys = pool->phys_of(src_data);
    } else {
        DmaMat* src = d.cache().get(pool, p.in_h, p.in_w, CV_8UC3);
        if (!src) return false;
        img.copyTo(src->mat());           // 唯一一次複製:heap -> DMA 記憶體
        src_phys = src->phys();
        src_data = static_cast<const uint8_t*>(src->mat().data);
    }
    const double copy_ms = ms(t0, now());

    if (!ip->is_idle()) return false;

    // ---- 只 flush 輸入這一段,不動整個 pool ----
    t0 = now();
    pool->sync_range_for_device(src_data, src_len);
    const double sync_in_ms = ms(t0, now());

    ip->set_in_ptr (src_phys);
    ip->set_out_ptr(dst->phys());
    ip->set_total_words  (p.total_words);
    ip->set_total_results(p.total_results);
    ip->set_out_words    (p.out_words);
    ip->set_out_w        (p.out_w);
    ip->set_scale_mode   (p.scale_mode);
    ip->set_inv_scale    (p.inv_scale);

    t0 = now();
    ip->irq_enable();
    ip->start();
    const bool ok = ip->wait_done_irq(timeout_ms);
    const double run_ms = ms(t0, now());
    if (!ok) return false;

    // ---- 只 invalidate 輸出這一段 ----
    t0 = now();
    pool->sync_range_for_cpu(dst->mat().data, p.out_bytes);
    const double sync_out_ms = ms(t0, now());

    out = dst->mat();                // 淺拷貝,資料仍在 DMA 記憶體

    if (timing) {
        timing->copy_ms  = copy_ms;
        timing->sync_ms  = sync_in_ms + sync_out_ms;
        timing->run_ms   = run_ms;
        timing->total_ms = ms(t_begin, now());
    }
    return true;
}


// ============================================================
//  input_buffer —— 取得一塊位於 DMA 記憶體的輸入緩衝
//
//  把影像直接解碼 / 擷取到這裡,downscale 與 letterbox 就會走 zero-copy,
//  省掉整整一次全幀複製(1080p 約 3 ms)。
//
//  slot 讓同一組尺寸能拿到多塊獨立的緩衝 —— 多執行緒 pipeline 必須用
//  不同的 slot,否則擷取端會覆寫 IP 正在讀的資料。
//  相同的 (width, height, slot) 每次回傳同一塊。
//  IP 不可用時回傳空的 Mat。
// ============================================================
inline cv::Mat input_buffer(int width, int height, int slot = 0) {
    detail::Device& d = detail::Device::get();
    std::lock_guard<std::mutex> lk(d.mutex());

    auto pool = hls::pool_ptr();
    if (!pool) return cv::Mat();

    DmaMat* m = d.cache().get(pool, height, width, CV_8UC3, slot);
    return m ? m->mat() : cv::Mat();
}


// ============================================================
//  letterbox —— 主要介面
//
//  等比縮放到最長邊 = input_size,置中並補黑邊。
//  幾何行為與純 CPU 版完全一致;IP 只吃掉整數倍那一段,
//  零頭交給 cv::INTER_AREA。IP 不可用時整段退回 CPU,結果仍正確。
// ============================================================
inline void letterbox(const cv::Mat& img, int input_size, Result& res) {
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

    res.used_ip  = false;
    res.ip_scale = 0;
    res.mid_w    = orig_w;
    res.mid_h    = orig_h;

    // ---- 由輸入與目標尺寸直接推出計畫 ----
    const Plan plan = plan_for(orig_w, orig_h, pad_w, pad_h);
    res.reason = plan.reason;

    if (plan.use_ip) {
        cv::Mat mid;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = downscale(img, static_cast<int>(plan.scale), mid, &res.timing);
        res.ip_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        res.zero_copy = (res.timing.copy_ms < 0.01);
        if (ok) {
            res.used_ip  = true;
            res.ip_scale = static_cast<int>(plan.scale);
            res.mid_w    = mid.cols;
            res.mid_h    = mid.rows;

            const auto t1 = std::chrono::steady_clock::now();
            if (plan.exact)
                mid.copyTo(roi);          // IP 一次到位,免二次縮放
            else
                cv::resize(mid, roi, roi.size(), 0, 0, cv::INTER_AREA);
            res.timing.post_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - t1).count();
            res.timing.total_ms += res.timing.post_ms;
            return;
        }
        res.reason = "IP 執行失敗,已退回 CPU";
    }

    // ---- 退路:結果一樣正確,只是比較慢 ----
    cv::resize(img, roi, roi.size(), 0, 0, cv::INTER_AREA);
}

inline Result letterbox(const cv::Mat& img, int input_size) {
    Result res;
    letterbox(img, input_size, res);
    return res;
}


// ============================================================
//  verify —— 拿 IP 的輸出與軟體 box filter 逐 byte 比對
//
//  上層不需要自己配 buffer、寫暫存器,呼叫這個就能驗證硬體正確性。
//  回傳 ran = false 表示 IP 跑不起來或條件不合(不代表結果錯)。
// ============================================================
struct VerifyReport {
    bool     ran        = false;
    uint64_t mismatches = 0;
    int      max_diff   = 0;
    uint64_t total      = 0;
    double   ms         = 0.0;
    const char* reason  = "";

    bool passed() const { return ran && mismatches == 0; }
};

inline VerifyReport verify(const cv::Mat& img, int scale) {
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
    const int s = static_cast<int>(scale);
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
                const int got    = orow[ox * 3 + ch];
                const int d      = std::abs(expect - got);
                if (d) {
                    ++rep.mismatches;
                    if (d > rep.max_diff) rep.max_diff = d;
                }
            }
        }
    }

    rep.reason = rep.mismatches ? "與軟體模型不符" : "完全一致";
    return rep;
}


// 產生一張適合驗證的測試圖。
// 三個通道方向各異 —— HLS 原始碼特別提醒單色圖與純水平漸層
// 在輸出欄錯位時值剛好相同,完全測不出該類 bug。
inline cv::Mat make_test_pattern(int width, int height) {
    cv::Mat m(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = m.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            row[x * 3 + 0] = static_cast<uint8_t>(x * 255 / std::max(1, width - 1));
            row[x * 3 + 1] = static_cast<uint8_t>(y * 255 / std::max(1, height - 1));
            row[x * 3 + 2] = static_cast<uint8_t>((x + y) & 0xFF);
        }
    }
    return m;
}

}  // namespace resize
}  // namespace hls