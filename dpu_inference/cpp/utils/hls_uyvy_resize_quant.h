// hls_uyvy_resize_quant.h — uyvy_resize_quant 的對外介面
//
// IP 功能:UYVY 4:2:2 → RGB888,同時做整數倍 box filter 縮小(2 倍 / 3 倍),
//         並「同時」輸出兩份結果:
//           resize_axi_bus  縮小後的 RGB888(uint8)
//           quant_axi_bus   縮小後 RGB 的量化結果(int8,與 RGB 同尺寸)
//
// 與舊版 hls_uyvy_resize.h 的差異:
//   * IP 多了一個輸出指標 quant_axi_bus(0x28),暫存器位址整體後移
//   * downscale / process / letterbox / verify 都同時處理 rgb 與 quant 兩份輸出
//   * namespace 改為 hls::uyvy_resize_quant
//
// 最簡單的用法:
//
//     #include "hls_uyvy_resize_quant.h"
//
//     cv::Mat uyvy = ...;                                        // CV_8UC2
//     auto r = hls::uyvy_resize_quant::process(uyvy, 2);         // 縮小 2 倍
//     if (r.rgb.empty()) std::cerr << r.reason << "\n";
//     // r.rgb   : CV_8UC3, R G B
//     // r.quant : CV_8SC3, R G B(量化後)
//
// 量化模型(與 IP 位元一致):
//   IP 的量化是寫死在 ROM 裡的三張查表(quantize_lut.h 的 kLut),沒有參數暫存器。
//   CPU 端直接使用同一份 quantize_lut.h,因此 quant_byte() 與 IP 完全相同:
//
//     q = trunc( clamp( v * scale[ch] + bias[ch], -128, 127 ) )      (往 0 截斷)
//     scale[ch] = 2^fix_point / (std[ch] * 255)
//     bias[ch]  = -mean[ch] / std[ch] * 2^fix_point
//     mean = {0.485, 0.456, 0.406}、std = {0.229, 0.224, 0.225}(R, G, B)
//
//   即 norm_and_fix_letterbox(fix_point = ip_fix_point())的結果,每個通道不同。
//
//   注意:
//   * fix_point 在合成時決定(quantize_lut.h 的 kFixPoint),可用 ip_fix_point() 查詢。
//     它必須等於 xmodel 輸入的 fix_point,IP 的 quant 才能直接當 DPU 輸入。
//   * 編譯這個函式庫用的 quantize_lut.h 必須與產生 bitstream 時是同一份,
//     否則 verify 的 quant 比對會失敗(verify 會指出 IP 實際的 fix_point)。
//
// 尺寸限制(同舊版):
//   img_w  16 的倍數;3 倍時為 48 的倍數;最大 4095(12-bit)
//   img_h  2 倍時為偶數;3 倍時為 3 的倍數;最大 4095
//   輸出寬度(img_w / scale)不超過 960

#pragma once

#include "hls_common.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

namespace hls {
namespace uyvy_resize_quant {

using hls::Timing;
using hls::input_buffer;


// ============================================================
//  Params —— 暫存器參數與尺寸限制
// ============================================================
struct Params {
    uint32_t in_w = 0, in_h = 0;
    uint32_t scale = 0;                 // 2 或 3
    uint32_t scale_mode = 0;            // 0 = 縮小 2 倍, 1 = 縮小 3 倍
    uint32_t out_w = 0, out_h = 0;
    uint32_t in_bytes    = 0;           // in_w * in_h * 2
    uint32_t out_bytes   = 0;           // out_w * out_h * 3(resize 輸出)
    uint32_t out_words   = 0;           // resize 輸出的 128-bit word 數(向上取整,尾端補 0)
    uint32_t quant_bytes = 0;           // quant 輸出位元組數(與 out_bytes 相同:每個 byte 對應一個 RGB byte)
    uint32_t quant_words = 0;           // quant 輸出的 128-bit word 數

    static constexpr uint32_t kDimMax  = 4095;
    static constexpr uint32_t kOutWMax = 960;

    static bool feasible(uint32_t in_w, uint32_t in_h, uint32_t scale);
    static Params make(uint32_t in_w, uint32_t in_h, uint32_t scale);   // 不合會丟 std::invalid_argument
    std::string describe() const;
};


// ============================================================
//  裝置設定與查詢
// ============================================================
void configure();
void configure(const std::string& uio_name);
void use_devmem();
void use_devmem(uint64_t ctrl_phys);
void set_ctrl_phys(uint64_t ctrl_phys);

std::string device_info();
bool        using_irq();
bool        available();
std::string last_error();


// ============================================================
//  量化(CPU 模型,與 IP 的 ROM 位元一致)
// ============================================================

// 單一 byte 的量化結果。ch:0 = R、1 = G、2 = B(每個通道的 mean / std 不同)
int8_t quant_byte(uint8_t v, int ch);

// IP 量化 LUT 合成時的 fix_point(quantize_lut.h 的 kFixPoint)。
// 必須等於 xmodel 輸入的 fix_point,IP 的 quant 才能直接當 DPU 輸入。
int ip_fix_point();

// 黑色 (0,0,0) 量化後的值 = (quant_byte(0,0), quant_byte(0,1), quant_byte(0,2))。
// letterbox 的 quant 黑邊就是填這個值;例:fix_point=5 → (-67,-65,-57)、6 → (-128,-128,-115)
cv::Scalar quant_pad();


// ============================================================
//  process —— 主要介面
// ============================================================
struct Result {
    cv::Mat     rgb;                 // CV_8UC3,R G B
    cv::Mat     quant;               // CV_8SC3,R G B(量化後)
    bool        used_ip = false;
    Timing      timing{};
    const char* reason = "";
};

// IP 不可用時退回 CPU 的 reference(),RGB 與 quant 都與 IP 位元一致
Result process(const cv::Mat& uyvy, int scale);


// ============================================================
//  低階介面
// ============================================================

// 只用 IP,不退回 CPU。rgb / quant 指向 DMA 記憶體裡的快取 buffer(不複製),
// 下次呼叫會被覆寫;要保留請自行 clone()。
bool downscale(const cv::Mat& uyvy, int scale, cv::Mat& rgb, cv::Mat& quant,
               Timing* timing = nullptr, int timeout_ms = 2000);

// 輸入已在「實體連續、cache 已 clean」的 DMA 記憶體中(條件同舊版)。
bool downscale_phys(const cv::Mat& uyvy, uint64_t in_phys, int scale,
                    cv::Mat& rgb, cv::Mat& quant,
                    Timing* timing = nullptr, int timeout_ms = 2000);

// CPU 版,RGB 與 quant 都與 IP 位元一致(也是 process 的退路)。quant 可為 nullptr。
bool reference(const cv::Mat& uyvy, int scale, cv::Mat& rgb, cv::Mat* quant = nullptr);


// ============================================================
//  letterbox —— UYVY 直接變成模型輸入(input_size x input_size)
//
//  img   : RGB uint8,黑邊為 0
//  quant : RGB int8(CV_8SC3),黑邊為 quant_pad()(每個通道不同)
//
//  quant 的輸出位置:
//    want_quant = false        → 不產生 quant(res.quant 為空),省下複製與量化
//    quant_dst  != nullptr     → 直接寫進 *quant_dst(例如 DPU 輸入緩衝,CV_8SC3、input_size 正方形),
//                                res.quant 與它共用同一塊記憶體 —— 省掉一次「res.quant → DPU 輸入」的複製
//    quant_dst  == nullptr     → 寫進 res.quant(與舊版相同)
//
//  黑邊只在「buffer / 內容區 / 填值」改變時才填(通常只有第一幀),之後每幀只寫內容區。
//  因此呼叫端不可改寫 img / quant 的黑邊;若改寫了(例如測試時整張 setTo),
//  請呼叫 letterbox_invalidate_borders(),下次 letterbox 會重填。
//  黑邊紀錄是 thread_local,只影響呼叫的那條執行緒。
//
//  IP 剛好整除時 img / quant 皆直接取自 IP;需要 CPU 再縮零頭,或走 CPU 退路時,
//  quant 由 CPU 依最終 RGB 內容、以同一份 LUT 重新量化。
//
//  quant 的 fix_point 固定為 ip_fix_point()。若 xmodel 的 fix_point 不同,
//  請改用 norm_and_fix_letterbox(img, xmodel 的 fix_point)產生 DPU 輸入。
// ============================================================
struct Plan {
    bool        use_ip = false;
    uint32_t    scale  = 0;
    bool        exact  = false;
    Params      params{};
    const char* reason = "";
};

Plan plan_for(uint32_t in_w, uint32_t in_h, uint32_t need_w, uint32_t need_h);

struct LetterboxResult {
    cv::Mat     img;                  // input_size x input_size, CV_8UC3, RGB
    cv::Mat     quant;                // input_size x input_size, CV_8SC3, RGB(量化);
                                      // quant_dst 時與它共用記憶體,want_quant=false 時為空
    cv::Rect    content;
    cv::Point2f ratio{1.f, 1.f};
    cv::Point2f pad{0.f, 0.f};

    bool        used_ip   = false;
    int         ip_scale  = 0;
    int         mid_w     = 0, mid_h = 0;
    bool        zero_copy = false;
    Timing      timing{};
    double      ip_ms     = 0.0;
    const char* reason    = "";
};

void letterbox(const cv::Mat& uyvy, int input_size, LetterboxResult& res,
               bool allow_ip = true, uint64_t in_phys = 0,
               cv::Mat* quant_dst = nullptr, bool want_quant = true);
LetterboxResult letterbox(const cv::Mat& uyvy, int input_size, bool allow_ip = true,
                          uint64_t in_phys = 0);

// 清除黑邊紀錄(呼叫端改寫了黑邊時使用);下次 letterbox 會重填黑邊
void letterbox_invalidate_borders();


// ============================================================
//  verify —— 上板先跑,確認 RGB 與 quant 兩份輸出都與 CPU 模型逐 byte 一致
//
//  quant 的參考值是 quant_byte()(與 IP 同一份 LUT)。不一致時 reason 會指出
//  IP 實際相當於哪個 fix_point,或是否 R/B 查表對調。
// ============================================================
struct VerifyReport {
    bool     ran        = false;

    // resize(RGB)輸出
    uint64_t mismatches = 0;
    uint64_t total      = 0;
    uint64_t ch_err[3]  = {0, 0, 0};
    int      max_diff   = 0;
    int      pad_dirty  = 0;

    // quant 輸出
    uint64_t q_mismatches = 0;
    uint64_t q_total      = 0;
    uint64_t q_ch_err[3]  = {0, 0, 0};
    int      q_max_diff   = 0;
    int      q_pad_dirty  = 0;

    double   ms         = 0.0;
    std::string reason;

    bool rgb_ok()   const { return ran && mismatches == 0 && pad_dirty == 0; }
    bool quant_ok() const { return ran && q_mismatches == 0 && q_pad_dirty == 0; }
    bool passed()   const { return rgb_ok() && quant_ok(); }
};

VerifyReport verify(const cv::Mat& uyvy, int scale);

cv::Mat make_test_uyvy(int width, int height);

}  // namespace uyvy_resize_quant
}  // namespace hls