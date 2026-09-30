// hls_resize.h — resize_kernel_0 的對外介面
//
// 硬體位址、暫存器、driver 都在 src/hls_resize.cpp,這裡只有 API。
//
//   namespace hls::resize
//     Params / plan_for()   IP 的尺寸限制與參數(可用來事先判斷吃不吃得到硬體加速)
//     configure() ...       裝置設定與查詢
//     downscale()           低階:純 2x / 3x 整數倍縮小
//     letterbox()           主要:等比縮放 + 置中補黑邊
//     verify()              與軟體模型逐 byte 比對
//
// 函式沒有取名 resize(),是為了避開查找衝突 —— 若呼叫端寫了
// using namespace hls,一個叫 resize 的 namespace 會讓 resize(...)
// 這種呼叫找到 namespace 名而編譯失敗。

#pragma once

#include "hls_common.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

namespace hls {
namespace resize {

// 共用工具,沿用舊名稱(hls::resize::Timing 等)
using hls::Timing;
using hls::input_buffer;
using hls::make_test_pattern;


// ============================================================
//  Params —— 暫存器參數與尺寸限制
//  全部依 resize_areaDown.cpp 推導,不要自己另外算。
// ============================================================
struct Params {
    uint32_t total_words   = 0;   // 輸入的 128-bit word 總數
    uint32_t total_results = 0;   // compute_side 產出的結果筆數
    uint32_t out_words     = 0;   // 輸出的 128-bit word 總數(整張圖)
    uint32_t out_w         = 0;   // 輸出寬度(pixel)
    uint32_t scale_mode    = 0;   // 0 = SCALE_2, 1 = SCALE_3
    uint32_t inv_scale     = 0;   // 65536 / scale²,四捨五入

    uint32_t in_w = 0, in_h = 0, out_h = 0;
    uint32_t in_bytes = 0, out_bytes = 0;
    uint32_t scale = 0;

    static constexpr uint32_t kOutWMax = 960;   // HLS 的 OUT_W_MAX

    // 一次運算產出幾個輸出欄:3 倍 2 欄、2 倍 4 欄
    static constexpr uint32_t n_out_for(uint32_t s) { return (s == 3) ? 2u : 4u; }

    // 快速篩選,不丟例外
    static bool feasible(uint32_t in_w, uint32_t in_h, uint32_t scale);

    // 條件不合會丟 std::invalid_argument,訊息說明是哪一項
    static Params make(uint32_t in_w, uint32_t in_h, uint32_t scale);

    std::string describe() const;
};

// 由「輸入尺寸 + 目標尺寸」推出 IP 能幫上什麼忙
struct Plan {
    bool     use_ip = false;   // IP 派得上用場嗎
    uint32_t scale  = 0;       // 用幾倍(0 = 沒用)
    Params   params{};         // use_ip 時才有意義
    bool     exact  = false;   // IP 輸出剛好等於目標,連二次縮放都免了
    const char* reason = "";   // use_ip = false 時說明原因
};

// 倍率大的優先,讓 IP 多做一點;挑不到就回傳 use_ip = false 並附上原因。
Plan plan_for(uint32_t in_w, uint32_t in_h, uint32_t need_w, uint32_t need_h);


// ============================================================
//  裝置設定與查詢
// ============================================================

// 指定 UIO 裝置名稱。不呼叫時用預設名稱 resize_kernel_0。
void configure();
void configure(const std::string& uio_name);

// 啟用 /dev/mem 後路:UIO 建不起來時,直接映射控制暫存器。
// 不帶參數時用硬體預設位址。這是裝置樹還沒設好時的暫時手段:
// 需要 root、沒有中斷(改輪詢)、沒有任何存取保護。
void use_devmem();
void use_devmem(uint64_t ctrl_phys);

// 指定控制暫存器位址(僅用於依位址尋找 UIO,不啟用 /dev/mem)
void set_ctrl_phys(uint64_t ctrl_phys);

std::string device_info();   // 實際綁到的裝置,用於確認找對了沒
bool        using_irq();     // UIO(有中斷)或 /dev/mem(輪詢)
bool        available();     // IP 與 DMA pool 是否都可用
std::string last_error();


// ============================================================
//  downscale —— 低階介面,純整數倍縮小
//
//  out 指向 DMA 記憶體裡的快取 buffer(淺拷貝),下次呼叫會被覆寫;
//  要保留請自行 clone()。
//  scale 只能是 2 或 3;條件不合或 IP 不可用時回傳 false,不自動退回 CPU。
// ============================================================
bool downscale(const cv::Mat& img, int scale, cv::Mat& out,
               Timing* timing = nullptr, int timeout_ms = 2000);


// ============================================================
//  letterbox —— 主要介面
//
//  等比縮放到最長邊 = input_size,置中並補黑邊。
//  幾何行為與純 CPU 版完全一致;IP 只吃掉整數倍那一段,
//  零頭交給 cv::INTER_AREA。IP 不可用時整段退回 CPU,結果仍正確。
// ============================================================
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

void   letterbox(const cv::Mat& img, int input_size, Result& res);
Result letterbox(const cv::Mat& img, int input_size);


// ============================================================
//  verify —— 拿 IP 的輸出與軟體 box filter 逐 byte 比對
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

VerifyReport verify(const cv::Mat& img, int scale);

}  // namespace resize
}  // namespace hls