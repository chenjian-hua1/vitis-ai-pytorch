// =====================================================================
//  uyvy2rgb.h  --  UYVY 4:2:2 -> RGB24 轉換器
// =====================================================================
#ifndef UYVY2RGB_H
#define UYVY2RGB_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// ---------------------------------------------------------------------
//  型別
// ---------------------------------------------------------------------

// 係數: 一位整數 + 八位小數。AP_RND 四捨五入後:
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

// out = in - 128，翻轉 MSB 即可，不需要減法器
void subtract_128(ap_uint<8> &in, ap_int<8> &out);

// acc = w * x + b，綁定到一顆三級 pipeline 的 DSP48E2
void mac(dsp_w_t &w, dsp_x_t &x, dsp_acc_t &b, dsp_acc_t &acc);

// 有號 clamp，三個通道共用。輸入直接代表真值，不含偏移。
//   v[9:8] = 00 -> 0~255 -> v[7:0]
//   v[9:8] = 01 -> >=256 -> 255
//   v[9:8] = 1x -> 負    -> 0
void clamp_s(ap_int<10> &v, ap_uint<8> &out);

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

#endif // UYVY2RGB_H
