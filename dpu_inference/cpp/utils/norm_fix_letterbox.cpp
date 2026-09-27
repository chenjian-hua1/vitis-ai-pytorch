// norm_fix_letterbox.cpp
// norm_fix_letterbox.h 的實作。所有內部細節都放在匿名命名空間，對外只有兩個函式。
//
// 計算路徑在編譯期決定：
//   - AArch64（ARMv8，含 KV260 的 Cortex-A53）→ NEON 查表（TBL/TBX）
//   - 其他平台（x86、32-bit ARM 等）          → 一般計算（原本 norm_and_fix 的 float 公式）
//   - 編譯時加 -DLBN_FORCE_GENERIC           → 在 ARM 上也強制走一般計算（方便比對或除錯）
//
// 注意：NEON 查表用到的 vqtbl4q / vqtbx4q 只存在於 AArch64；
//       32-bit ARM（ARMv7）即使有 NEON 也沒有這兩個指令，因此會走一般計算。

#include "norm_fix_letterbox.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
 
// ---------------------------------------------------------------------------
// 編譯期路徑選擇
// ---------------------------------------------------------------------------
#if !defined(LBN_FORCE_GENERIC) && \
    ((defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64))
#define LBN_USE_NEON_LUT 1
#include <arm_neon.h>
#else
#define LBN_USE_NEON_LUT 0
#endif
 
#if defined(__GNUC__)
#define LBN_RESTRICT __restrict__
#else
#define LBN_RESTRICT
#endif
 
namespace {
 
// ---- 正規化參數：與原本 .cpp 的定義逐字相同 ----
constexpr float kMeanR = 0.485f, kMeanG = 0.456f, kMeanB = 0.406f;
constexpr float kStdR  = 0.229f, kStdG  = 0.224f, kStdB  = 0.225f;
 
constexpr float kU8ScaleR = 1.f/(kStdR*255.f);
constexpr float kU8ScaleG = 1.f/(kStdG*255.f);
constexpr float kU8ScaleB = 1.f/(kStdB*255.f);
constexpr float kU8BiasR  = -kMeanR/kStdR;
constexpr float kU8BiasG  = -kMeanG/kStdG;
constexpr float kU8BiasB  = -kMeanB/kStdB;
 
#if LBN_USE_NEON_LUT
// ===========================================================================
//  AArch64：NEON 查表
// ===========================================================================
 
// ===== BEGIN LUT BUILD =====
// 建立查表：與原本 norm_and_fix 相同的公式與運算順序
//   scale = kU8Scale * 2^fix_point、bias = kU8Bias * 2^fix_point
//   y     = clamp(v * scale + bias, -128, 127) → 往 0 截斷成 int8
//
// v * scale + bias 刻意用 std::fma（乘加只捨入一次）：
//   GCC 在 aarch64 上預設會把原本 norm_and_fix 的 a*b+c 編譯成 fmla（融合乘加），
//   只有 -ffp-contract=off 或 -O0 才不會。以這組參數來說，G 通道 v=102 時
//   理論值剛好是整數（-0.25 × 2^fix_point），融合與否會差 1。
//   明確呼叫 std::fma 讓查表結果不受編譯選項影響，固定與 KV260 上 -O2/-O3 的 float 版一致。
//   若你的 float 版是用 -ffp-contract=off 編的，把 std::fma(...) 改成 v * s + b 即可。
void build_lut(int fix_point, int8_t lut[768])
{
    const float fp = std::exp2f(static_cast<float>(fix_point));
    const float scale[3] = {kU8ScaleR * fp, kU8ScaleG * fp, kU8ScaleB * fp};
    const float bias[3]  = {kU8BiasR  * fp, kU8BiasG  * fp, kU8BiasB  * fp};
    for (int c = 0; c < 3; ++c)
        for (int v = 0; v < 256; ++v) {
            const float y = std::fma(static_cast<float>(v), scale[c], bias[c]);
            lut[256*c + v] = static_cast<int8_t>(std::clamp(y, -128.f, 127.f));
        }
}
// ===== END LUT BUILD =====
 
// ===== BEGIN KERNEL =====
// lut：三張表連續排放，[0..255]=R, [256..511]=G, [512..767]=B
// src/dst：交錯的 RGB，共 total 個 pixel
 
// 256 項的表拆成 4 段，每段 64 bytes 剛好是一次 vqtbl4q / vqtbx4q 的容量。
// TBL：索引超出 0..63 → 輸出 0；TBX：索引超出範圍 → 保留原值。
// 每查完一段就把索引減 64，四段串起來即涵蓋 0..255。
inline uint8x16_t lookup256(const uint8_t* LBN_RESTRICT t, uint8x16_t idx)
{
    const uint8x16_t c64 = vdupq_n_u8(64);
    uint8x16_t r = vqtbl4q_u8(vld1q_u8_x4(t), idx);
    idx = vsubq_u8(idx, c64); r = vqtbx4q_u8(r, vld1q_u8_x4(t +  64), idx);
    idx = vsubq_u8(idx, c64); r = vqtbx4q_u8(r, vld1q_u8_x4(t + 128), idx);
    idx = vsubq_u8(idx, c64); r = vqtbx4q_u8(r, vld1q_u8_x4(t + 192), idx);
    return r;
}
 
void lut_kernel(const uint8_t* LBN_RESTRICT src,
                int8_t*        LBN_RESTRICT dst,
                std::ptrdiff_t total,
                const int8_t*  LBN_RESTRICT lut)
{
    // 3 張表共需 48 個 q 暫存器，超過 NEON 的 32 個，所以每次迴圈從 L1 重新載入。
    // 表只有 768 bytes，一定常駐 L1。
    const uint8_t* tR = reinterpret_cast<const uint8_t*>(lut);
    const uint8_t* tG = tR + 256;
    const uint8_t* tB = tR + 512;
    uint8_t* d = reinterpret_cast<uint8_t*>(dst);
 
    std::ptrdiff_t i = 0;
    for (; i + 16 <= total; i += 16) {
        uint8x16x3_t p = vld3q_u8(src + 3*i);   // 反交錯成 R/G/B 三個平面
        p.val[0] = lookup256(tR, p.val[0]);
        p.val[1] = lookup256(tG, p.val[1]);
        p.val[2] = lookup256(tB, p.val[2]);
        vst3q_u8(d + 3*i, p);                   // 交錯寫回
    }
    for (; i < total; ++i) {                    // 尾端不足 16 pixel
        dst[3*i + 0] = lut[      src[3*i + 0]];
        dst[3*i + 1] = lut[256 + src[3*i + 1]];
        dst[3*i + 2] = lut[512 + src[3*i + 2]];
    }
}
// ===== END KERNEL =====
 
#else
// ===========================================================================
//  其他平台：一般計算（原本 norm_and_fix 的公式，逐字相同）
// ===========================================================================
 
// ===== BEGIN GENERIC KERNEL =====
// scale/bias 已乘上 2^fix_point
void float_kernel(const uint8_t* LBN_RESTRICT src,
                  int8_t*        LBN_RESTRICT dst,
                  std::ptrdiff_t total,
                  const float scale[3], const float bias[3])
{
    const float scaleR = scale[0], scaleG = scale[1], scaleB = scale[2];
    const float biasR  = bias[0],  biasG  = bias[1],  biasB  = bias[2];
 
    #pragma omp simd
    for (std::ptrdiff_t i = 0; i < total; ++i) {
        dst[3*i + 0] = static_cast<int8_t>(std::clamp(src[3*i + 0] * scaleR + biasR, -128.f, 127.f));
        dst[3*i + 1] = static_cast<int8_t>(std::clamp(src[3*i + 1] * scaleG + biasG, -128.f, 127.f));
        dst[3*i + 2] = static_cast<int8_t>(std::clamp(src[3*i + 2] * scaleB + biasB, -128.f, 127.f));
    }
}
// ===== END GENERIC KERNEL =====
 
#endif
 
// ---------------------------------------------------------------------------
//  快取
// ---------------------------------------------------------------------------
 
// 記錄某個 out buffer 的黑邊是以什麼參數填的。
// held 持有該 buffer 的一份參照（引用計數 +1），確保它不會被釋放；
// 因此只要 out.data 與 held.data 相同，就一定是同一塊記憶體，
// 不會發生「舊 buffer 釋放後，新 buffer 剛好配到同一個位址」而誤判的情況。
struct PadRecord {
    cv::Mat held;
    int y0 = -1, y1 = -1, fix_point = INT_MIN;
};
 
constexpr int kMaxBuffers = 4;
 
struct Cache {
    int fix_point = INT_MIN;             // 目前參數對應的 fix_point；INT_MIN → 尚未準備
#if LBN_USE_NEON_LUT
    int8_t lut[768];                     // NEON 路徑：查表
#else
    float scale[3], bias[3];             // 一般路徑：已乘上 2^fix_point 的係數
#endif
    int8_t    pad[3];                    // 黑色 (0,0,0) 正規化後的值
    PadRecord pads[kMaxBuffers];
    int       next = 0;                  // 滿了之後輪流覆蓋
};
 
Cache& cache()
{
    thread_local Cache c;
    return c;
}
 
// fix_point 改變時，準備該路徑需要的參數與黑邊值
void prepare(Cache& c, int fix_point)
{
#if LBN_USE_NEON_LUT
    build_lut(fix_point, c.lut);
    c.pad[0] = c.lut[0];
    c.pad[1] = c.lut[256];
    c.pad[2] = c.lut[512];
#else
    const float fp = std::exp2f(static_cast<float>(fix_point));
    c.scale[0] = kU8ScaleR * fp;  c.bias[0] = kU8BiasR * fp;
    c.scale[1] = kU8ScaleG * fp;  c.bias[1] = kU8BiasG * fp;
    c.scale[2] = kU8ScaleB * fp;  c.bias[2] = kU8BiasB * fp;
    // src = 0 時 0 * scale + bias = bias（無論是否融合乘加都一樣）
    for (int k = 0; k < 3; ++k)
        c.pad[k] = static_cast<int8_t>(std::clamp(c.bias[k], -128.f, 127.f));
#endif
    c.fix_point = fix_point;
}
 
} // namespace
 
void norm_letterbox_reset()
{
    cache() = Cache{};
}
 
void norm_and_fix_letterbox(const cv::Mat& x, int fix_point, int y0, int y1, cv::Mat& out)
{
    CV_Assert(x.type() == CV_8UC3 && x.isContinuous());
    CV_Assert(0 <= y0 && y0 <= y1 && y1 <= x.rows);
 
    Cache& c = cache();
 
    // 1) 參數：只跟 fix_point 有關，改變時才重新準備
    if (c.fix_point != fix_point)
        prepare(c, fix_point);
 
    // 2) 輸出 buffer：尺寸或型別不符才重新配置
    if (out.rows != x.rows || out.cols != x.cols || out.type() != CV_8SC3 || !out.isContinuous())
        out.create(x.rows, x.cols, CV_8SC3);
 
    // 3) 黑邊：找這個 out buffer 的紀錄，參數都相同才跳過
    PadRecord* rec = nullptr;
    for (PadRecord& r : c.pads)
        if (!r.held.empty() && r.held.data == out.data) { rec = &r; break; }
 
    const bool need_fill = !rec
        || rec->y0 != y0 || rec->y1 != y1 || rec->fix_point != fix_point
        || rec->held.rows != out.rows || rec->held.cols != out.cols;
 
    if (need_fill) {
        const cv::Scalar v(c.pad[0], c.pad[1], c.pad[2]);
        if (y0 > 0)         out.rowRange(0, y0).setTo(v);
        if (y1 < out.rows)  out.rowRange(y1, out.rows).setTo(v);
 
        if (!rec) {                                              // 新的 buffer → 佔一個位置
            rec = &c.pads[c.next];
            c.next = (c.next + 1) % kMaxBuffers;
        }
        rec->held      = out;                                    // 持有參照，防止位址被重用
        rec->y0        = y0;
        rec->y1        = y1;
        rec->fix_point = fix_point;
    }
 
    // 4) 只計算影像內容區域（整列切出的 rowRange 仍是連續記憶體）
    const std::ptrdiff_t total = std::ptrdiff_t(y1 - y0) * x.cols;
    if (total > 0) {
#if LBN_USE_NEON_LUT
        lut_kernel(x.ptr<uint8_t>(y0), out.ptr<int8_t>(y0), total, c.lut);
#else
        float_kernel(x.ptr<uint8_t>(y0), out.ptr<int8_t>(y0), total, c.scale, c.bias);
#endif
    }
}
 
