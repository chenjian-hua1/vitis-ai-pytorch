/******************************************************************************
 * resize_impl.h
 *
 * resize（整數倍 box-filter 縮小）的運算實作
 *
 *   只放「算」的部分：fabric 加法器 + DSP（前加器 + 乘法器）。
 *   資料怎麼選進來、結果怎麼選出去，都在各 top 檔：
 *     resize_top.cpp / uyvy_resize_top.cpp
 *****************************************************************************/

#ifndef RESIZE_IMPL_H
#define RESIZE_IMPL_H

#include "ap_int.h"

/* ---- 縮小模式 ---- */
#define SCALE_2    0                /* 2 倍縮小：2x2 box */
#define SCALE_3    1                /* 3 倍縮小：3x3 box */

/* ---- 位元寬 ---- */
#define ACCW       12               /* 單通道單欄累加寬度（= DSP lane 寬度） */
#define LBW        (ACCW * 2 * 3)   /* 72 bit：line buffer 一格 = 2 欄 x RGB */
#define OPW        (ACCW * 2)       /* 24 bit：DSP 前加器運算元 */
#define SRW        16               /* scale_rate 埠寬度 */
#define PROD_W     (OPW + SRW)      /* 40 bit：乘積 */

#define FRAC_S3    16               /* 3 倍：scale_rate 用 16 bit 表示 */
#define FRAC_S2    2                /* 2 倍：scale_rate 用 2 bit 表示 */

/* 單顆 DSP：P = (A + D) * B */
ap_uint<PROD_W> dsp_addmul(ap_uint<OPW> a, ap_uint<OPW> d, ap_uint<SRW> b);

/* 兩組共用一顆 DSP：pack=1 打包兩組，pack=0 只算 lo 組 */
void dsp_shared(bool             pack,
                ap_uint<ACCW>    a_hi,
                ap_uint<ACCW>    d_hi,
                ap_uint<ACCW>    a_lo,
                ap_uint<ACCW>    d_lo,
                ap_uint<SRW>     b,
                ap_uint<PROD_W> &p_hi,
                ap_uint<PROD_W> &p_lo);

/* 一顆 DSP 份的 PE：fabric 加法 + 分組 + dsp_shared */
void resize_pe(ap_uint<8>       u0,
               ap_uint<8>       u1,
               ap_uint<8>       u2,
               ap_uint<ACCW>    v,
               ap_uint<OPW>     lbp,
               bool             s3,
               ap_uint<SRW>     mul_b,
               ap_uint<PROD_W> &p_hi,
               ap_uint<PROD_W> &p_lo);

#endif /* RESIZE_IMPL_H */
