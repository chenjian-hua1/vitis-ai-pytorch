// hls_uyvy_resize.h — uyvy_resize_0 的對外介面
//
// IP 功能:UYVY 4:2:2 → RGB888,同時做整數倍 box filter 縮小(2 倍 / 3 倍)
//
// 最簡單的用法(main 只要這樣):
//
//     #include "hls_uyvy_resize.h"
//
//     cv::Mat uyvy = ...;                                  // CV_8UC2,寬 x 高
//     auto r = hls::uyvy_resize::process(uyvy, 2);         // 縮小 2 倍
//     if (r.rgb.empty()) std::cerr << r.reason << "\n";    // 尺寸不符時才會失敗
//
//   IP 不可用時會自動改用 CPU 的位元一致模型,結果完全相同,只是比較慢;
//   r.used_ip 告訴你這次有沒有真的用到硬體。
//
// 資料格式:
//   輸入  CV_8UC2,每 4 byte = U, Y0, V, Y1(兩個 pixel 共用一組色差)。
//         這跟 OpenCV COLOR_YUV2BGR_UYVY 吃的格式相同。
//   輸出  CV_8UC3,byte 順序是 R, G, B(不是 OpenCV 慣用的 BGR)。
//         要 imshow / imwrite 時請先 cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR)。
//
// 尺寸限制(IP 的硬體限制):
//   img_w  16 的倍數;3 倍時為 48 的倍數;最大 4095(12-bit)
//   img_h  2 倍時為偶數;3 倍時為 3 的倍數;最大 4095
//   輸出寬度(img_w / scale)不超過 960
//   例:1920x1080 → 2 倍 960x540、3 倍 640x360 都可以。
//
// zero-copy:
//   輸入如果已經在 DMA 記憶體裡,就不必再複製一次(1080p 約 4 MB)。
//   用 hls::input_buffer(w, h, slot, CV_8UC2) 取得緩衝,把影像直接擷取到裡面。

#pragma once

#include "hls_common.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

namespace hls {
namespace uyvy_resize {

using hls::Timing;
using hls::input_buffer;


// ============================================================
//  Params —— 暫存器參數與尺寸限制
// ============================================================
struct Params {
    uint32_t in_w = 0, in_h = 0;        // img_w / img_h
    uint32_t scale = 0;                 // 2 或 3
    uint32_t scale_mode = 0;            // 0 = 縮小 2 倍, 1 = 縮小 3 倍
    uint32_t out_w = 0, out_h = 0;
    uint32_t in_bytes  = 0;             // in_w * in_h * 2
    uint32_t out_bytes = 0;             // out_w * out_h * 3
    uint32_t out_words = 0;             // IP 實際寫出的 128-bit word 數(向上取整,尾端補 0)

    static constexpr uint32_t kDimMax  = 4095;   // img_w / img_h 為 12-bit
    static constexpr uint32_t kOutWMax = 960;    // IP 內 line buffer 上限

    // 快速篩選,不丟例外
    static bool feasible(uint32_t in_w, uint32_t in_h, uint32_t scale);

    // 條件不合會丟 std::invalid_argument,訊息說明是哪一項
    static Params make(uint32_t in_w, uint32_t in_h, uint32_t scale);

    std::string describe() const;
};


// ============================================================
//  裝置設定與查詢
//  都不呼叫也能用;只有 UIO 名稱或位址跟預設不同時才需要。
// ============================================================

// 指定 UIO 裝置名稱。預設是 "uyvy_resize"(裝置樹節點名去掉 @位址)。
// 不確定時在板子上執行 cat /sys/class/uio/uio*/name 查看。
void configure();
void configure(const std::string& uio_name);

// 啟用 /dev/mem 後路(裝置樹還沒設好時的暫時手段:需要 root、改輪詢、無存取保護)
void use_devmem();                      // 用 hls_uyvy_resize.cpp 裡設定的位址
void use_devmem(uint64_t ctrl_phys);

// 指定控制暫存器位址(僅用於依位址尋找 UIO,不啟用 /dev/mem)
void set_ctrl_phys(uint64_t ctrl_phys);

std::string device_info();   // 實際綁到的裝置
bool        using_irq();     // UIO(有中斷)或 /dev/mem(輪詢)
bool        available();     // IP 與 DMA pool 是否都可用
std::string last_error();


// ============================================================
//  process —— 主要介面
//
//  回傳的 rgb 是獨立的 Mat,可以放心保存。
//  尺寸或格式不符時 rgb 為空,reason 說明原因。
// ============================================================
struct Result {
    cv::Mat     rgb;                 // CV_8UC3,R G B 順序
    bool        used_ip = false;     // 這次有沒有用到硬體
    Timing      timing{};            // 各階段耗時(只有用到 IP 時才有意義)
    const char* reason = "";
};

Result process(const cv::Mat& uyvy, int scale);


// ============================================================
//  低階介面
// ============================================================

// 只用 IP,不退回 CPU。
// rgb 指向 DMA 記憶體裡的快取 buffer(不複製),下次呼叫會被覆寫;
// 要保留請自行 clone()。適合輸出馬上要交給下一顆 IP 的情況。
bool downscale(const cv::Mat& uyvy, int scale, cv::Mat& rgb,
               Timing* timing = nullptr, int timeout_ms = 2000);

// CPU 版,與 IP 位元完全一致(也是 process 的退路)
bool reference(const cv::Mat& uyvy, int scale, cv::Mat& rgb);


// ============================================================
//  letterbox —— UYVY 直接變成模型輸入(input_size x input_size, RGB)
//
//  流程:IP 做 UYVY→RGB + 整數倍縮小 → CPU 把零頭縮到目標大小
//        (剛好整除時只是一次 copy)→ 貼進補黑邊的正方形。
//  邏輯與 hls::resize::letterbox 相同(等比、置中、補 0、不放大),
//  所以座標換算(ratio / pad / content)可以直接沿用。
//
//  用法:
//     hls::uyvy_resize::LetterboxResult lb;          // 放在迴圈外重用
//     hls::uyvy_resize::letterbox(uyvy, 640, lb);
//     // lb.img:640x640 CV_8UC3,R G B 順序
//
//  1080p → 640:3 倍剛好 640x360,IP 一次到位
//  720p  → 640:2 倍剛好 640x360,IP 一次到位
//  縮放不到 2 倍(例如 640x480 → 640)或 IP 不可用時,
//  改用 OpenCV cvtColor + resize(結果與 IP 非位元一致,但差異極小)。
// ============================================================
struct Plan {
    bool        use_ip = false;
    uint32_t    scale  = 0;
    bool        exact  = false;       // IP 輸出剛好等於目標,不用再縮
    Params      params{};
    const char* reason = "";
};

// 要把 in_w x in_h 縮到 need_w x need_h,IP 該用幾倍。
// 優先選最大的倍率(IP 做愈多、CPU 零頭愈少),但不縮過頭。
Plan plan_for(uint32_t in_w, uint32_t in_h, uint32_t need_w, uint32_t need_h);

struct LetterboxResult {
    cv::Mat     img;                  // input_size x input_size, CV_8UC3, RGB
    cv::Rect    content;              // 影像在 img 裡的位置(其餘是黑邊)
    cv::Point2f ratio{1.f, 1.f};      // 原圖 → content 的縮放
    cv::Point2f pad{0.f, 0.f};        // 左 / 上的補邊(浮點,與 YOLO 慣例一致)

    bool        used_ip   = false;
    int         ip_scale  = 0;
    int         mid_w     = 0, mid_h = 0;   // IP 輸出尺寸
    bool        zero_copy = false;
    Timing      timing{};             // run/sync/copy 為 IP 段,post_ms 為 CPU 零頭縮放
    double      ip_ms     = 0.0;
    const char* reason    = "";
};

// res.img 尺寸相同時會重用記憶體,放在迴圈外可避免每幀配置。
// uyvy 必須是 CV_8UC2、寬為偶數。
// allow_ip = false 時直接走 CPU(IP 出錯後用來避免每幀重試、每幀等逾時)。
// IP 失敗的詳細原因請看 last_error()。
void letterbox(const cv::Mat& uyvy, int input_size, LetterboxResult& res,
               bool allow_ip = true);
LetterboxResult letterbox(const cv::Mat& uyvy, int input_size, bool allow_ip = true);


// ============================================================
//  verify —— 上板時先跑這個,確認硬體結果與 CPU 模型逐 byte 一致
//  ran = false 表示 IP 跑不起來或尺寸不符(不代表結果錯)。
// ============================================================
struct VerifyReport {
    bool     ran        = false;
    uint64_t mismatches = 0;
    uint64_t total      = 0;
    uint64_t ch_err[3]  = {0, 0, 0};   // R, G, B 各自的錯誤數
    int      max_diff   = 0;
    int      pad_dirty  = 0;           // 最後一個 word 的補位 byte 不是 0 的數量
    double   ms         = 0.0;
    std::string reason;

    bool passed() const { return ran && mismatches == 0 && pad_dirty == 0; }
};

VerifyReport verify(const cv::Mat& uyvy, int scale);

// 產生一張適合驗證的 UYVY 測試圖(Y、U、V 在水平與垂直方向都有變化)
cv::Mat make_test_uyvy(int width, int height);

}  // namespace uyvy_resize
}  // namespace hls