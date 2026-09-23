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
#include "uyvy2rgb.h"          /* dsp_w_t / dsp_x_t / dsp_acc_t / uvy_w_t */

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
