// =====================================================================
//  uyvy2rgb.cpp  --  UYVY 4:2:2 -> RGB24，AXI4 master
//
//  架構:
//    DDR --m_axi(128b)--> read_convert --stream(192b)--> repack_write --m_axi(128b)--> DDR
//                          4 x cvt_pair                   leftover 192->128
//
//  一拍 128-bit 輸入 = 4 組 UYVY = 8 pixel = 192-bit RGB，只用 8 顆 DSP。
//  限制: img_w 必須是 16 的倍數。
// =====================================================================

#include "uyvy2rgb.h"
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// =====================================================================
//  Sub Module
//  out = in - 128
//  out[7]   = !in[7]        // >=128 為正
//  out[6:0] =  in[6:0]      // 數值
// =====================================================================
void subtract_128(ap_uint<8> &in, ap_int<8> &out) {
    out.range(6, 0) = in.range(6, 0);
    out[7]          = !in[7];
}

// =====================================================================
//  Sub Module
//  acc = w * x + b
//  Use one DSP48E2 three pipelined for multiply and accumulate operation
// =====================================================================
void mac(dsp_w_t &w, dsp_x_t &x, dsp_acc_t &b, dsp_acc_t &acc) {
    acc = w * x + b;
#pragma HLS BIND_OP variable=acc op=mul impl=dsp latency=3
}

// =====================================================================
//  Sub Module
//  有號 clamp，三個通道共用。輸入 v 是 ap_int<10>，直接代表真值。
//
//      v[9:8] = 00  ->     0 ~  255   ->  v[7:0]
//      v[9:8] = 01  ->   256 ~  511   ->  255
//      v[9:8] = 10  ->  -512 ~ -257   ->  0
//      v[9:8] = 11  ->  -256 ~   -1   ->  0
//
//  10 與 11 都是負數，所以硬體上塌成兩層單 bit 的 mux。
// =====================================================================
void clamp_s(ap_int<10> &v, ap_uint<8> &out) {
    out = (v[9]) ? ap_uint<8>(0)
                 : ((v[8]) ? ap_uint<8>(255) : ap_uint<8>(v.range(7, 0)));
}

// =====================================================================
//  Sub Module
//  cvt_core: 單一 pixel，給 unit test 用。
//
//  這裡 Y 放在 C port 的低位，所以低位欄位是 w_lo*D + Y*256，
//  正負不再單純由 D 決定 -> 必須用 +acc[17] 做借位修正。
//  (cvt_pair 把 Y 移到 fabric，就可以改用一條線補償，見下面)
//
//  打包配置 (A port, 27-bit signed):
//      bit 26    : 0             權重為正
//      bit 25:18 : w_hi  Q0.8    高位欄位
//      bit 17: 9 : 0             guard
//      bit  8: 0 : w_lo  Q1.8    低位欄位
// =====================================================================
void cvt_core(ap_uint<8> &y, ap_int<8> &d, ap_int<8> &e, ap_uint<24> &pixel) {
    uvy_w_t d_wlo = 1.772, d_whi = 0.344136, e_wlo = 1.402, e_whi = 0.714136;

    // ---- D dsp ----
    dsp_x_t d_x;
    for (int i = 7; i < 18; i++) {           // bit 7 本身就是符號位，一起填
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
    d_b.range(15, 8)  = y;                   // Y 對齊到 Q_.8

    dsp_acc_t d_acc;
    mac(d_w, d_x, d_b, d_acc);

    ap_int<18> d_lo = d_acc.range(17,  0);   // w_lo*D + Y*256   -58112 ~ 122938
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

    // ---- 兩個高位欄位先在 Q8.8 相加，只截斷一次 ----
    ap_int<17> hi_sum = ap_int<17>(d_hi) + ap_int<17>(e_hi);
    ap_int<9>  hi_int = hi_sum >> 8;         // -136 ~ 134

    ap_int<10> b_v = ap_int<10>(d_lo >> 8);                            // -227 ~ 480
    ap_int<10> r_v = ap_int<10>(e_lo >> 8);                            // -180 ~ 433
    ap_int<10> g_v = ap_int<10>(ap_int<11>(y) - ap_int<11>(hi_int));   // -134 ~ 391

    ap_uint<8> r, g, b;
    clamp_s(b_v, b);
    clamp_s(r_v, r);
    clamp_s(g_v, g);

    pixel.range(23, 16) = r;
    pixel.range(15,  8) = g;
    pixel.range( 7,  0) = b;
}

// =====================================================================
//  Sub Module
//  cvt_pair: 一組 UYVY (U, Y0, V, Y1) -> 兩個 RGB pixel，只用 2 顆 DSP。
//
//  Y 不放 C port，兩個 pixel 各自在 fabric 加自己的 Y
//  (Y*256 對整數位的貢獻剛好是 Y，沒有小數被截掉 -> 零精度損失)。
//
//  於是 C port 的低位是乾淨的，低位欄位就只有 w_lo*D:
//    - |w_lo*D| <= 58112 < 2^16  ->  直接取 acc[16:0] 當有號 17-bit
//    - 低位為負 <=> D 為負       ->  借位補償變成 C port bit18 = d[7]，一條線
//    - 高位 acc[33:18] 直接就是 w_hi*D，不需要任何修正
// =====================================================================
static void cvt_pair(ap_uint<8> y0, ap_uint<8> y1,
                     ap_uint<8> u,  ap_uint<8> v,
                     ap_uint<24> &px0, ap_uint<24> &px1) {
#pragma HLS INLINE off
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
    d_b[18] = d[7];                          // <-- 高位2補時補1，一條線，零 LUT

    dsp_acc_t d_acc;
    mac(d_w, d_x, d_b, d_acc);

    ap_int<17> d_lo = d_acc.range(16,  0);   // w_lo*D   -58112 ~ 57658
    ap_int<16> d_hi = d_acc.range(33, 18);   // w_hi*D   -11264 ~ 11176，不必修正

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
    e_b[18] = e[7];                          // <-- 同上

    dsp_acc_t e_acc;
    mac(e_w, e_x, e_b, e_acc);

    ap_int<17> e_lo = e_acc.range(16,  0);   // w_lo*E   -45952 ~ 45593
    ap_int<16> e_hi = e_acc.range(33, 18);   // w_hi*E   -23424 ~ 23241

    // ---- G 的色度項兩個 pixel 共用，只截斷一次 ----
    //      若各自先右移再相加，floor 誤差會同向累積成 2 LSB
    ap_int<17> hi_sum = ap_int<17>(d_hi) + ap_int<17>(e_hi);   // -34688 ~ 34417
    ap_int<9>  hi_int = hi_sum >> 8;                           // -136 ~ 134

    ap_int<9> d_int = ap_int<9>(d_lo >> 8);  // 1.7734*D  -227 ~ 225
    ap_int<9> e_int = ap_int<9>(e_lo >> 8);  // 1.4023*E  -180 ~ 178

    // ---- 兩個 pixel 各自加自己的 Y ----
    ap_uint<8>  ya[2];
    ap_uint<24> po[2];
    ya[0] = y0;  ya[1] = y1;

PIX: for (int k = 0; k < 2; k++) {
#pragma HLS UNROLL
        ap_int<10> b_v = ap_int<10>(ap_int<11>(d_int) + ap_int<11>(ya[k]));  // -227 ~ 480
        ap_int<10> r_v = ap_int<10>(ap_int<11>(e_int) + ap_int<11>(ya[k]));  // -180 ~ 433
        ap_int<10> g_v = ap_int<10>(ap_int<11>(ya[k]) - ap_int<11>(hi_int)); // -134 ~ 391

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

// ---------------------------------------------------------------------
//  pixel {R,G,B} -> 記憶體 byte 順序 R,G,B (byte0 = R)
//  想要 BGR 的話直接回傳 p 不要換位
// ---------------------------------------------------------------------
static inline ap_uint<24> to_mem(ap_uint<24> p) {
#pragma HLS INLINE
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
//
//    殘留量只會是 0 / 64 / 128 三種，用一個 3 路多工器選線:
//      res=0   : 讀一筆，輸出 d[127:0]，留 d[191:128]  (64)
//      res=64  : 讀一筆，輸出 {d[63:0], res}，留 d[191:64] (128)
//      res=128 : 不讀，直接輸出 res，清空
//
//    每 cycle 最多一次 stream read、剛好一次 m_axi write -> II=1
//    out[i] 是唯一的寫入點且位址純遞增，burst 才推得出來
// =====================================================================
static void repack_write(hls::stream<ap_uint<192> > &si,
                         ap_uint<128> *out, ap_uint<32> out_beats) {
    ap_uint<128> res = 0;
    ap_uint<2>   st  = 0;

WR: for (ap_uint<32> i = 0; i < out_beats; i++) {
#pragma HLS PIPELINE II=1
        // st[1]=1 只出現在 st==2，所以「要不要讀」是單一 bit，不是比較
        ap_uint<192> d = 0;
        if (!st[1]) d = si.read();

        ap_uint<128> w;

        // switch 的 select 直接就是 st 兩個 bit，合成出來是純 mux
        switch (st) {
        case 0:                                  // 無殘留: 輸出低 128，留 64
            w   = d.range(127,   0);
            res = d.range(191, 128);
            st  = 1;
            break;

        case 1:                                  // 殘留 64: 拼成 128，留 128
            w.range( 63,  0) = res.range(63, 0);
            w.range(127, 64) = d.range( 63, 0);
            res = d.range(191, 64);
            st  = 2;
            break;

        default:                                 // 殘留 128: 直接輸出，清空
            w   = res;
            st  = 0;
            break;
        }
        out[i] = w;
    }
}

// =====================================================================
//  Top Module
//    img_w 必須是 16 的倍數。Max 4096x4096。
// =====================================================================
void uyvy2rgb(ap_uint<128> *uyvy_axi_bus, ap_uint<128> *rgb_axi_bus,
              ap_uint<12> img_w, ap_uint<12> img_h) {

#pragma HLS INTERFACE m_axi port=uyvy_axi_bus offset=slave bundle=gmem0 \
                     depth=512 max_read_burst_length=128 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=rgb_axi_bus  offset=slave bundle=gmem1 \
                     depth=768 max_write_burst_length=128 num_write_outstanding=4

#pragma HLS INTERFACE s_axilite port=uyvy_axi_bus bundle=control
#pragma HLS INTERFACE s_axilite port=rgb_axi_bus  bundle=control
#pragma HLS INTERFACE s_axilite port=img_w        bundle=control
#pragma HLS INTERFACE s_axilite port=img_h        bundle=control
#pragma HLS INTERFACE s_axilite port=return       bundle=control

#pragma HLS DATAFLOW

    // 每列 img_w 個 pixel，一拍 8 個 pixel
    ap_uint<32> in_beats  = (ap_uint<32>(img_w) >> 3) * ap_uint<32>(img_h);
    ap_uint<32> out_beats = in_beats + (in_beats >> 1);      // x1.5

    hls::stream<ap_uint<192> > pix_fifo;
#pragma HLS STREAM        variable=pix_fifo depth=32
#pragma HLS BIND_STORAGE  variable=pix_fifo type=fifo impl=srl

    read_convert (uyvy_axi_bus, pix_fifo, in_beats);
    repack_write (pix_fifo, rgb_axi_bus, out_beats);
}
