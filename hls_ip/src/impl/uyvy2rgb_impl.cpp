/******************************************************************************
 * uyvy2rgb_impl.cpp
 *
 * UYVY -> RGB 運算實作，宣告見 uyvy2rgb_impl.h
 *****************************************************************************/

#include "uyvy2rgb_impl.h"

/* =====================================================================
 *  out = in - 128
 *  out[7]   = !in[7]        // >=128 為正
 *  out[6:0] =  in[6:0]      // 數值
 * ===================================================================== */
void subtract_128(ap_uint<8> &in, ap_int<8> &out) {
    out.range(6, 0) = in.range(6, 0);
    out[7]          = !in[7];
}

/* =====================================================================
 *  acc = w * x + b
 *  Use one DSP48E2 three pipelined for multiply and accumulate operation
 * ===================================================================== */
void mac(dsp_w_t &w, dsp_x_t &x, dsp_acc_t &b, dsp_acc_t &acc) {
    acc = w * x + b;
#pragma HLS BIND_OP variable=acc op=mul impl=dsp latency=3
}

/* =====================================================================
 *  有號 clamp，三個通道共用。輸入 v 是 ap_int<10>，直接代表真值。
 *
 *      v[9:8] = 00  ->     0 ~  255   ->  v[7:0]
 *      v[9:8] = 01  ->   256 ~  511   ->  255
 *      v[9:8] = 10  ->  -512 ~ -257   ->  0
 *      v[9:8] = 11  ->  -256 ~   -1   ->  0
 * ===================================================================== */
void clamp_s(ap_int<10> &v, ap_uint<8> &out) {
    out = (v[9]) ? ap_uint<8>(0)
                 : ((v[8]) ? ap_uint<8>(255) : ap_uint<8>(v.range(7, 0)));
}

/* =====================================================================
 *  cvt_core: 單一 pixel，給 unit test 用。
 *
 *  Y 放在 C port 的低位，低位欄位是 w_lo*D + Y*256，
 *  正負不再單純由 D 決定 -> 必須用 +acc[17] 做借位修正。
 *
 *  打包配置 (A port, 27-bit signed):
 *      bit 26    : 0             權重為正
 *      bit 25:18 : w_hi  Q0.8    高位欄位
 *      bit 17: 9 : 0             guard
 *      bit  8: 0 : w_lo  Q1.8    低位欄位
 * ===================================================================== */
void cvt_core(ap_uint<8> &y, ap_int<8> &d, ap_int<8> &e, ap_uint<24> &pixel) {
    uvy_w_t d_wlo = 1.772, d_whi = 0.344136, e_wlo = 1.402, e_whi = 0.714136;

    // ---- D dsp ----
    dsp_x_t d_x;
    for (int i = 7; i < 18; i++) {
#pragma HLS UNROLL
        d_x[i] = d[7];
    }
    d_x.range(6, 0) = d.range(6, 0);

    dsp_w_t d_w;
    d_w[26]           = 0;
    d_w.range(25, 18) = d_whi.range(7, 0);
    d_w.range(17,  9) = 0;
    d_w.range( 8,  0) = d_wlo.range(8, 0);

    dsp_acc_t d_b     = 0;
    d_b.range(15, 8)  = y;

    dsp_acc_t d_acc;
    mac(d_w, d_x, d_b, d_acc);

    ap_int<18> d_lo = d_acc.range(17,  0);
    ap_int<16> d_hi = d_acc.range(33, 18);
    d_hi += d_acc[17];                       // 借位修正

    // ---- E dsp ----
    dsp_x_t e_x;
    for (int i = 7; i < 18; i++) {
#pragma HLS UNROLL
        e_x[i] = e[7];
    }
    e_x.range(6, 0) = e.range(6, 0);

    dsp_w_t e_w;
    e_w[26]           = 0;
    e_w.range(25, 18) = e_whi.range(7, 0);
    e_w.range(17,  9) = 0;
    e_w.range( 8,  0) = e_wlo.range(8, 0);

    dsp_acc_t e_b     = 0;
    e_b.range(15, 8)  = y;

    dsp_acc_t e_acc;
    mac(e_w, e_x, e_b, e_acc);

    ap_int<18> e_lo = e_acc.range(17,  0);
    ap_int<16> e_hi = e_acc.range(33, 18);
    e_hi += e_acc[17];                       // 借位修正

    ap_int<17> hi_sum = ap_int<17>(d_hi) + ap_int<17>(e_hi);
    ap_int<9>  hi_int = hi_sum >> 8;

    ap_int<10> b_v = ap_int<10>(d_lo >> 8);
    ap_int<10> r_v = ap_int<10>(e_lo >> 8);
    ap_int<10> g_v = ap_int<10>(ap_int<11>(y) - ap_int<11>(hi_int));

    ap_uint<8> r, g, b;
    clamp_s(b_v, b);
    clamp_s(r_v, r);
    clamp_s(g_v, g);

    pixel.range(23, 16) = r;
    pixel.range(15,  8) = g;
    pixel.range( 7,  0) = b;
}

/* =====================================================================
 *  cvt_pair: 一組 UYVY (U, Y0, V, Y1) -> 兩個 RGB pixel，只用 2 顆 DSP。
 *
 *  Y 不放 C port，兩個 pixel 各自在 fabric 加自己的 Y
 *  (Y*256 對整數位的貢獻剛好是 Y，沒有小數被截掉 -> 零精度損失)。
 *
 *  於是 C port 的低位是乾淨的，低位欄位就只有 w_lo*D:
 *    - |w_lo*D| <= 58112 < 2^16  ->  直接取 acc[16:0] 當有號 17-bit
 *    - 低位為負 <=> D 為負       ->  借位補償變成 C port bit18 = d[7]，一條線
 *    - 高位 acc[33:18] 直接就是 w_hi*D，不需要任何修正
 * ===================================================================== */
void cvt_pair(ap_uint<8> y0, ap_uint<8> y1,
              ap_uint<8> u,  ap_uint<8> v,
              ap_uint<24> &px0, ap_uint<24> &px1) {
#pragma HLS INLINE
    uvy_w_t d_wlo = 1.772, d_whi = 0.344136, e_wlo = 1.402, e_whi = 0.714136;

    ap_int<8> d, e;
    subtract_128(u, d);
    subtract_128(v, e);

    // ---- D dsp ----
    dsp_x_t d_x;
    for (int i = 7; i < 18; i++) {
#pragma HLS UNROLL
        d_x[i] = d[7];
    }
    d_x.range(6, 0) = d.range(6, 0);

    dsp_w_t d_w;
    d_w[26]           = 0;
    d_w.range(25, 18) = d_whi.range(7, 0);
    d_w.range(17,  9) = 0;
    d_w.range( 8,  0) = d_wlo.range(8, 0);

    dsp_acc_t d_b = 0;
    d_b[18] = d[7];                          // 高位 2 補時補 1，一條線

    dsp_acc_t d_acc;
    mac(d_w, d_x, d_b, d_acc);

    ap_int<17> d_lo = d_acc.range(16,  0);   // w_lo*D   -58112 ~ 57658
    ap_int<16> d_hi = d_acc.range(33, 18);   // w_hi*D   -11264 ~ 11176

    // ---- E dsp ----
    dsp_x_t e_x;
    for (int i = 7; i < 18; i++) {
#pragma HLS UNROLL
        e_x[i] = e[7];
    }
    e_x.range(6, 0) = e.range(6, 0);

    dsp_w_t e_w;
    e_w[26]           = 0;
    e_w.range(25, 18) = e_whi.range(7, 0);
    e_w.range(17,  9) = 0;
    e_w.range( 8,  0) = e_wlo.range(8, 0);

    dsp_acc_t e_b = 0;
    e_b[18] = e[7];

    dsp_acc_t e_acc;
    mac(e_w, e_x, e_b, e_acc);

    ap_int<17> e_lo = e_acc.range(16,  0);   // w_lo*E   -45952 ~ 45593
    ap_int<16> e_hi = e_acc.range(33, 18);   // w_hi*E   -23424 ~ 23241

    // ---- G 的色度項兩個 pixel 共用，只截斷一次 ----
    ap_int<17> hi_sum = ap_int<17>(d_hi) + ap_int<17>(e_hi);
    ap_int<9>  hi_int = hi_sum >> 8;         // -136 ~ 134

    ap_int<9> d_int = ap_int<9>(d_lo >> 8);  // -227 ~ 225
    ap_int<9> e_int = ap_int<9>(e_lo >> 8);  // -180 ~ 178

    // ---- 兩個 pixel 各自加自己的 Y ----
    ap_uint<8>  ya[2];
    ap_uint<24> po[2];
    ya[0] = y0;  ya[1] = y1;

PIX: for (int k = 0; k < 2; k++) {
#pragma HLS UNROLL
        ap_int<10> b_v = ap_int<10>(ap_int<11>(d_int) + ap_int<11>(ya[k]));
        ap_int<10> r_v = ap_int<10>(ap_int<11>(e_int) + ap_int<11>(ya[k]));
        ap_int<10> g_v = ap_int<10>(ap_int<11>(ya[k]) - ap_int<11>(hi_int));

        ap_uint<8> r, g, b;
        clamp_s(b_v, b);
        clamp_s(r_v, r);
        clamp_s(g_v, g);

        po[k].range(23, 16) = r;
        po[k].range(15,  8) = g;
        po[k].range( 7,  0) = b;
    }
    px0 = po[0];
    px1 = po[1];
}
