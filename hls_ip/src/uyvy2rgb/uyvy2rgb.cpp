// =====================================================================
//  uyvy2rgb.cpp  --  UYVY 4:2:2 -> RGB24，AXI4 master
//
//  架構:
//    DDR --m_axi(128b)--> read_convert --stream(192b)--> pack_write --m_axi(128b)--> DDR
//                          4 x cvt_pair                   2:3 打包
//
//  一拍 128-bit 輸入 = 4 組 UYVY = 8 個 pixel = 192-bit RGB。
//  8 個 pixel 只用 8 顆 DSP: 同組 UYVY 的兩個 pixel 共用 U、V，
//  色度乘法算一次，各自的 Y 在 fabric 加 (零精度損失)。
//
//  限制: img_w 必須是 16 的倍數。
// =====================================================================

#include "uyvy2rgb.h"
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// =====================================================================
//  Sub Module
//  out = in - 128;
//  out[sign]     = !in[MSB];       // >=128
//  out[sign-1:0] =  in[MSB-1:0];   // value
// =====================================================================
void subtract_128(ap_uint<8> &in, ap_int<8> &out) {
    // 正: in[6:0]   負(2補數表示法): 128 - (-(in[6:0]-128)) = in[6:0]
    out.range(6, 0) = in.range(6, 0);
    // 正負號取決於 in>=128 (看第7位元)
    out[7] = !in[7];
}

// =====================================================================
//  Sub Module
//  acc = w * x + b;
//  Use one DSP48E2 three pipelined for multiply and accumulate operation
// =====================================================================
void mac(dsp_w_t &w, dsp_x_t &x, dsp_acc_t &b, dsp_acc_t &acc) {
    acc = w * x + b;
// Use one three pipelined DSP48E2
#pragma HLS BIND_OP variable=acc op=mul impl=dsp latency=3
}

// =====================================================================
//  Sub Module
//  帶 +256 偏移的 clamp，三個通道共用。
//
//  輸入 v 是 10-bit 無號，代表的真值 = v - 256。
//  有效輸出 [0,255] 恰好對應 v 屬於 [256, 512)，也就是 v[9:8] == 01，
//  所以整個 clamp 只是一個 2-bit 解碼，不需要符號判斷也不需要比較器。
//
//      v[9:8] = 00  ->  真值 < 0      ->  0
//      v[9:8] = 01  ->  真值 0 ~ 255  ->  v[7:0]
//      v[9:8] = 1x  ->  真值 >= 256   ->  255
// =====================================================================
void clamp_ofs(ap_uint<10> &v, ap_uint<8> &out) {
    out = (v[9]) ? ap_uint<8>(255)
                 : ((v[8]) ? ap_uint<8>(v.range(7, 0)) : ap_uint<8>(0));
}

// =====================================================================
//  Sub Module
//  單一 pixel 版本，給 unit test 用。資料路徑跟 cvt_pair 相同，
//  差別只在 Y 放在 DSP 的 C port 而不是 fabric。
//
//  d : input (U-128)  signed
//  e : input (V-128)  signed
//
//  R = clamp(Y + 1.402   *E)
//  G = clamp(Y - 0.344136*D - 0.714136*E)
//  B = clamp(Y + 1.772   *D)
//
//  打包配置 (A port, 27-bit signed):
//      bit 26    : 0            權重為正
//      bit 25:18 : w2  Q0.8     高位欄位
//      bit 17: 9 : 0            guard
//      bit  8: 0 : w1  Q1.8     低位欄位
//
//  C port = { bit16=1 , y , 8'd0 }。那個 2^16 讓低位欄位恆非負，
//  所以不會向高位借位，高位讀出來直接就是 w2*x，不必 +P[17]。
// =====================================================================
void cvt_core(ap_uint<8> &y, ap_int<8> &d, ap_int<8> &e, ap_uint<24> &pixel) {
    uvy_w_t d_w1 = 1.772, d_w2 = 0.344136, e_w1 = 1.402, e_w2 = 0.714136;

    // ---- D dsp ----
    dsp_x_t d_x;
    for (int i = 7; i < 18; i++) {           // 起點是 7: bit 7 本身就是符號位
#pragma HLS UNROLL
        d_x[i] = d[7];
    }
    d_x.range(6, 0) = d.range(6, 0);

    dsp_w_t d_w;
    d_w[26]           = 0;
    d_w.range(25, 18) = d_w2.range(7, 0);
    d_w.range(17,  9) = 0;
    d_w.range( 8,  0) = d_w1.range(8, 0);

    dsp_acc_t d_b   = 0;
    d_b[16]         = 1;                     // K = 2^16
    d_b.range(15,8) = y;

    dsp_acc_t d_acc;
    mac(d_w, d_x, d_b, d_acc);

    ap_uint<10> b_v  = d_acc.range(17,  8);
    ap_int<16>  d_hi = d_acc.range(33, 18);

    // ---- E dsp ----
    dsp_x_t e_x;
    for (int i = 7; i < 18; i++) {
#pragma HLS UNROLL
        e_x[i] = e[7];
    }
    e_x.range(6, 0) = e.range(6, 0);

    dsp_w_t e_w;
    e_w[26]           = 0;
    e_w.range(25, 18) = e_w2.range(7, 0);
    e_w.range(17,  9) = 0;
    e_w.range( 8,  0) = e_w1.range(8, 0);

    dsp_acc_t e_b   = 0;
    e_b[16]         = 1;
    e_b.range(15,8) = y;

    dsp_acc_t e_acc;
    mac(e_w, e_x, e_b, e_acc);

    ap_uint<10> r_v  = e_acc.range(17,  8);
    ap_int<16>  e_hi = e_acc.range(33, 18);

    // ---- G: 兩個高位欄位先在 Q8.8 相加，只截斷一次 ----
    //      若各自先右移再相減，floor 誤差會同向累積成 2 LSB
    ap_int<17> hi_sum = ap_int<17>(d_hi) + ap_int<17>(e_hi);
    ap_int<9>  hi_int = hi_sum >> 8;

    ap_uint<9> y_ofs;                        // {1, y} = 256 + Y，拼接不是加法
    y_ofs[8]          = 1;
    y_ofs.range(7, 0) = y;

    ap_uint<10> g_v = ap_uint<10>(ap_int<11>(y_ofs) - ap_int<11>(hi_int));

    ap_uint<8> r, g, b;
    clamp_ofs(b_v, b);
    clamp_ofs(r_v, r);
    clamp_ofs(g_v, g);

    pixel.range(23, 16) = r;
    pixel.range(15,  8) = g;
    pixel.range( 7,  0) = b;
}

// =====================================================================
//  Sub Module
//  cvt_pair: 一組 UYVY (U, Y0, V, Y1) -> 兩個 RGB pixel，只用 2 顆 DSP。
//
//  C port 只放 2^16 偏移，不放 Y。兩個 pixel 各自的 Y 在 fabric 加 --
//  因為 Y*256 對整數位的貢獻剛好是 Y，沒有小數部分會被截掉，
//  所以跟把 Y 放進 C port 的版本精度完全相同 (實測都是 1 LSB)。
//
//      D-DSP: 低位 = 1.772*D + 256 (29~480)   高位 = 0.344136*D
//      E-DSP: 低位 = 1.402*E + 256 (77~433)   高位 = 0.714136*E
// =====================================================================
static void cvt_pair(ap_uint<8> y0, ap_uint<8> y1,
                     ap_uint<8> u,  ap_uint<8> v,
                     ap_uint<24> &px0, ap_uint<24> &px1) {
    uvy_w_t d_w1 = 1.772, d_w2 = 0.344136, e_w1 = 1.402, e_w2 = 0.714136;

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
    d_w.range(25, 18) = d_w2.range(7, 0);
    d_w.range(17,  9) = 0;
    d_w.range( 8,  0) = d_w1.range(8, 0);

    dsp_acc_t d_b = 0;
    d_b[16] = 1;                             // K = 2^16，C port 不放 Y

    dsp_acc_t d_acc;
    mac(d_w, d_x, d_b, d_acc);

    ap_uint<10> d_lo = d_acc.range(17,  8);
    ap_int<16>  d_hi = d_acc.range(33, 18);

    // ---- E dsp ----
    dsp_x_t e_x;
    for (int i = 7; i < 18; i++) {
#pragma HLS UNROLL
        e_x[i] = e[7];
    }
    e_x.range(6, 0) = e.range(6, 0);

    dsp_w_t e_w;
    e_w[26]           = 0;
    e_w.range(25, 18) = e_w2.range(7, 0);
    e_w.range(17,  9) = 0;
    e_w.range( 8,  0) = e_w1.range(8, 0);

    dsp_acc_t e_b = 0;
    e_b[16] = 1;

    dsp_acc_t e_acc;
    mac(e_w, e_x, e_b, e_acc);

    ap_uint<10> e_lo = e_acc.range(17,  8);
    ap_int<16>  e_hi = e_acc.range(33, 18);

    // ---- G 的色度項兩個 pixel 共用 ----
    ap_int<17> hi_sum = ap_int<17>(d_hi) + ap_int<17>(e_hi);
    ap_int<9>  hi_int = hi_sum >> 8;         // -135 ~ 132

    // ---- 兩個 pixel 各自加自己的 Y ----
    ap_uint<8>  ya[2];
    ap_uint<24> po[2];
    ya[0] = y0;  ya[1] = y1;

PIX: for (int k = 0; k < 2; k++) {
#pragma HLS UNROLL
        ap_uint<9> y_ofs;
        y_ofs[8]          = 1;
        y_ofs.range(7, 0) = ya[k];

        ap_uint<10> bv = ap_uint<10>(d_lo + ya[k]);                        // 29 ~ 735
        ap_uint<10> rv = ap_uint<10>(e_lo + ya[k]);                        // 77 ~ 688
        ap_uint<10> gv = ap_uint<10>(ap_int<11>(y_ofs) - ap_int<11>(hi_int)); // 124 ~ 645

        ap_uint<8> r, g, b;
        clamp_ofs(bv, b);
        clamp_ofs(rv, r);
        clamp_ofs(gv, g);

        po[k].range(23, 16) = r;
        po[k].range(15,  8) = g;
        po[k].range( 7,  0) = b;
    }
    px0 = po[0];
    px1 = po[1];
}

// ---------------------------------------------------------------------
//  pixel {R,G,B} -> 記憶體 byte 順序 R,G,B (byte0 = R)
//  想要 BGR 的話直接回傳 p 不要換位
// ---------------------------------------------------------------------
static inline ap_uint<24> to_mem(ap_uint<24> p) {
    ap_uint<24> m;
    m.range( 7,  0) = p.range(23, 16);   // R
    m.range(15,  8) = p.range(15,  8);   // G
    m.range(23, 16) = p.range( 7,  0);   // B
    return m;
}

// =====================================================================
//  Stage 1: 讀 DDR + 轉換
//    每拍讀 128-bit = 4 組 UYVY，4 個 cvt_pair 並行 -> 8 pixel = 192-bit
// =====================================================================
static void read_convert(const ap_uint<128> *in,
                         hls::stream<ap_uint<192> > &fifo,
                         ap_uint<32> in_beats) {
RD: for (ap_uint<32> i = 0; i < in_beats; i++) {
#pragma HLS PIPELINE II=1
        ap_uint<128> raw = in[i];
        ap_uint<192> out;

GRP:    for (int gidx = 0; gidx < 4; gidx++) {
#pragma HLS UNROLL
            int s = gidx * 32;
            ap_uint<8> u  = raw.range(s +  7, s +  0);
            ap_uint<8> y0 = raw.range(s + 15, s +  8);
            ap_uint<8> v  = raw.range(s + 23, s + 16);
            ap_uint<8> y1 = raw.range(s + 31, s + 24);

            ap_uint<24> p0, p1;
            cvt_pair(y0, y1, u, v, p0, p1);

            out.range(gidx * 48 + 23, gidx * 48 +  0) = to_mem(p0);
            out.range(gidx * 48 + 47, gidx * 48 + 24) = to_mem(p1);
        }
        fifo.write(out);
    }
}

// =====================================================================
//  Stage 2: 192 -> 128 (leftover) + 寫 DDR
//    每 cycle 最多一次 stream read、剛好一次 m_axi write -> II=1
//    位址 out[i] 純遞增，是唯一的寫入點，burst 自然成立
// =====================================================================
static void repack_write(hls::stream<ap_uint<192> > &si,
                         ap_uint<128> *out, ap_uint<32> out_beats) {
    ap_uint<128> res = 0;
    ap_uint<2>   st  = 0;          // 0: 無殘留   1: 殘留 64   2: 殘留 128

WR: for (ap_uint<32> i = 0; i < out_beats; i++) {
#pragma HLS PIPELINE II=1
        ap_uint<128> w;

        // 3 states FSM 看每個clk選擇啥資料
        if (st == 2) {             // 殘留已滿 128，這拍不讀
            w  = res;
            st = 0;
        } else {
            ap_uint<192> d = si.read();
            if (st == 0) {
                w   = d.range(127, 0);
                res = d.range(191, 128);          // 留 64
                st  = 1;
            } else {
                w.range( 63,  0) = res.range(63, 0);
                w.range(127, 64) = d.range(63, 0);
                res = d.range(191, 64);           // 留 128
                st  = 2;
            }
        }
        out[i] = w;
    }
}

// =====================================================================
//  Top Module
//    uyvy_axi_bus : UYVY 來源 (read)
//    rgb_axi_bus  : RGB24 目的 (write)
//    img_w, img_h : 影像寬高，img_w 必須是 16 的倍數
//    Max 4096x4096
// =====================================================================
void uyvy2rgb(ap_uint<128> *uyvy_axi_bus, ap_uint<128> *rgb_axi_bus,
              ap_uint<12> img_w, ap_uint<12> img_h) {

    // ---- AXI4 master: 兩個獨立 bundle，讀寫才能同時進行 ----
#pragma HLS INTERFACE m_axi port=uyvy_axi_bus offset=slave bundle=gmem0 \
                     depth=MAX_IN max_read_burst_length=16 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=rgb_axi_bus  offset=slave bundle=gmem1 \
                     depth=MAX_OUT max_write_burst_length=16 num_write_outstanding=8

    // ---- AXI4-Lite: 位址與純量參數 ----
#pragma HLS INTERFACE s_axilite port=uyvy_axi_bus bundle=control
#pragma HLS INTERFACE s_axilite port=rgb_axi_bus  bundle=control
#pragma HLS INTERFACE s_axilite port=img_w        bundle=control
#pragma HLS INTERFACE s_axilite port=img_h        bundle=control
#pragma HLS INTERFACE s_axilite port=return       bundle=control

#pragma HLS DATAFLOW

    ap_uint<32> in_beats  = (ap_uint<32>(img_w) >> 3) * ap_uint<32>(img_h);
    ap_uint<32> out_beats = in_beats + (in_beats >> 1);      // x1.5

    hls::stream<ap_uint<192> > pix_fifo;
#pragma HLS STREAM        variable=pix_fifo depth=16
#pragma HLS BIND_STORAGE  variable=pix_fifo type=fifo impl=srl

    read_convert (uyvy_axi_bus, pix_fifo, in_beats);
    repack_write (pix_fifo, rgb_axi_bus, out_beats);
}