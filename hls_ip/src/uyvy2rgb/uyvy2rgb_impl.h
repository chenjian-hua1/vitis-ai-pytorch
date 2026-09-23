/******************************************************************************
 * uyvy2rgb_impl.h
 *
 * UYVY 4:2:2 -> RGB 的運算實作
 *
 *   只放「算」的部分：減 128、DSP 乘加、clamp、一組 UYVY 轉兩個 pixel。
 *   資料怎麼選進來、pixel 怎麼排成記憶體 byte 順序寫出，都在各 top 檔：
 *     uyvy2rgb_top.cpp / uyvy_resize_top.cpp
 *****************************************************************************/

#ifndef UYVY2RGB_IMPL_H
#define UYVY2RGB_IMPL_H

#include "ap_int.h"
#include "ap_fixed.h"

/* ---- DSP48E2 埠型別 ----
 *   A port 27-bit signed：打包兩個權重 {w_hi, guard, w_lo}
 *   B port 18-bit signed：色差 D / E（符號延伸到 18 bit）
 *   P      48-bit：乘加結果，取 [16:0] 與 [33:18] 兩個欄位 */
typedef ap_int<27>              dsp_w_t;
typedef ap_int<18>              dsp_x_t;
typedef ap_int<48>              dsp_acc_t;

/* ---- 權重型別 ----
 *   Q1.8 無號，取最近值：1.772 -> 454/256，1.402 -> 359/256，
 *   0.344136 -> 88/256，0.714136 -> 183/256
 *   w_lo 取 [8:0]，w_hi 取 [7:0] */
typedef ap_ufixed<9, 1, AP_RND> uvy_w_t;

void subtract_128(ap_uint<8> &in, ap_int<8> &out);
void mac(dsp_w_t &w, dsp_x_t &x, dsp_acc_t &b, dsp_acc_t &acc);
void clamp_s(ap_int<10> &v, ap_uint<8> &out);

/* 單一 pixel，unit test 用 */
void cvt_core(ap_uint<8> &y, ap_int<8> &d, ap_int<8> &e, ap_uint<24> &pixel);

/* 一組 (U, Y0, V, Y1) -> 兩個 {R,G,B} pixel，2 顆 DSP */
void cvt_pair(ap_uint<8> y0, ap_uint<8> y1,
              ap_uint<8> u,  ap_uint<8> v,
              ap_uint<24> &px0, ap_uint<24> &px1);

#endif /* UYVY2RGB_IMPL_H */
