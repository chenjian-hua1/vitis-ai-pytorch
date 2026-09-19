// =====================================================================
//  uyvy2rgb.h  --  UYVY 4:2:2 -> RGB24 轉換器
// =====================================================================
#ifndef UYVY2RGB_H
#define UYVY2RGB_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>


#define COSIM_MODE


// ---------------------------------------------------------------------
//  Compile Time Function
// ---------------------------------------------------------------------
// 計算該數字需要使用多少位元
constexpr int LOG2_CEIL(int x) {
    // 定義域：x >= 1
    // ceil(log2(1)) = 0, ceil(log2(2)) = 1, ceil(log2(3)) = 2, ...
    int r = 0;
    int p = 1;
    // 直到 2^r 超過x
    while (p <= x) {
        p*=2;
        ++r;
    }
    return r;
}

// 2^? 形式其中一個bit=1 其他0
constexpr bool IS_POW2(int x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

// 計算 for index (0..maxium-1) 需要的位元寬度
// 規則：若 maxium 是 2 的冪次方 -> LOG2_CEIL(maxium + 1)
// 否則 -> LOG2_CEIL(maxium)
constexpr int FOR_IDX_BITS(int maxium) {
    return IS_POW2(maxium) ? LOG2_CEIL(maxium + 1) : LOG2_CEIL(maxium);
}


// ---------------------------------------------------------------------
//  Data bus width
// ---------------------------------------------------------------------
#define DATA_BUS_W 128

// ---------------------------------------------------------------------
//  最大圖片size
// ---------------------------------------------------------------------
#define MAX_IMG_W 1920
#define MAX_IMG_H 1080

// ---------------------------------------------------------------------
//  buffer 大小 —— 必須跟 uyvy2rgb.cpp 的 depth= 完全一致
//  512 beats = 128x32 的輸入，768 = 對應輸出 (x1.5)
// ---------------------------------------------------------------------
constexpr int MAX_IN = MAX_IMG_W*MAX_IMG_H*16/DATA_BUS_W +1, 
    MAX_OUT = MAX_IMG_W*MAX_IMG_H*24/DATA_BUS_W +1;

// ---------------------------------------------------------------------
//  型別
// ---------------------------------------------------------------------

// 係數: 一位整數 + 八位小數。AP_RND 讓權重四捨五入:
//   1.772 -> 454   0.344136 -> 88   1.402 -> 359   0.714136 -> 183
typedef ap_ufixed<9, 1, AP_RND, AP_SAT> uvy_w_t;

// DSP48E2 乘法輸入 (A port, 打包兩個權重)。必須有號。
typedef ap_int<27>  dsp_w_t;
// DSP48E2 乘法輸入 (B port, 帶符號的 D 或 E)。必須有號。
typedef ap_int<18>  dsp_x_t;
// DSP48E2 輸出 (P)。必須有號。
typedef ap_int<48>  dsp_acc_t;

// ---------------------------------------------------------------------
//  Sub Modules
// ---------------------------------------------------------------------

// out = in - 128，靠翻轉 MSB 完成，不需要減法器
void subtract_128(ap_uint<8> &in, ap_int<8> &out);

// acc = w * x + b，綁定到一顆三級 pipeline 的 DSP48E2
void mac(dsp_w_t &w, dsp_x_t &x, dsp_acc_t &b, dsp_acc_t &acc);

// 帶 +256 偏移的 clamp。輸入 v 代表真值 v-256，
// 有效輸出對應 v[9:8] == 01，所以只是 2-bit 解碼。
void clamp_ofs(ap_uint<10> &v, ap_uint<8> &out);

// 單一 pixel 轉換 (unit test 用；頂層走的是 cvt_pair)
//   d = U-128, e = V-128, pixel = {R[23:16], G[15:8], B[7:0]}
void cvt_core(ap_uint<8> &y, ap_int<8> &d, ap_int<8> &e, ap_uint<24> &pixel);

// ---------------------------------------------------------------------
//  Top Module
// ---------------------------------------------------------------------

// uyvy_axi_bus : UYVY 來源 (m_axi read)
// rgb_axi_bus  : RGB24 目的 (m_axi write)
// img_w        : 影像寬，必須是 16 的倍數
// img_h        : 影像高
// Max 4096x4096
void uyvy2rgb(ap_uint<128> *uyvy_axi_bus, ap_uint<128> *rgb_axi_bus,
              ap_uint<12> img_w, ap_uint<12> img_h);

// ---------------------------------------------------------------------
//  已停用
// ---------------------------------------------------------------------
// clamp_0_255 被 clamp_ofs 取代。後者不需要獨立的 sign 輸入，
// 也不需要 in[9]|in[8] 的溢位比較。若舊碼還在引用可暫時保留宣告:
// void clamp_0_255(ap_uint<10> &in, bool &sign, ap_uint<8> &out);

#endif // UYVY2RGB_H