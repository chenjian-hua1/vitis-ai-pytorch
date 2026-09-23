// =====================================================================
//  uyvy2rgb_top.cpp  --  UYVY 4:2:2 -> RGB24，AXI4 master
//
//  架構:
//    DDR --m_axi(128b)--> read_convert --stream(192b)--> repack_write --m_axi(128b)--> DDR
//                          4 x cvt_pair                   192 -> 128 對齊
//
//  一拍 128-bit 輸入 = 4 組 UYVY = 8 pixel = 192-bit RGB，只用 8 顆 DSP。
//  限制: img_w 必須是 16 的倍數。Max 4096x4096。
//
//  運算用 uyvy2rgb_impl.cpp 的 cvt_pair，本檔負責資料選擇：
//
//    計算資料選擇
//      uyvy_in_select    128-bit -> 4 組 (U, Y0, V, Y1)
//    寫出資料選擇
//      uyvy_out_select   8 個 {R,G,B} -> 記憶體 byte 順序 -> 192-bit
//      repack_write      192 -> 128 輸出對齊狀態機
//
//  注意：concat 一律指定給完整寬度的變數，
//  不可直接寫進 .range()，否則會經過 64-bit 轉換被截斷。
// =====================================================================

#include "uyvy2rgb_top.h"
#include "uyvy2rgb_impl.h"      // cvt_pair
#include <hls_stream.h>
#include <ap_int.h>

#define UYVY_GRP   4        // 一拍 4 組 UYVY
#define UYVY_PIX   8        // 一拍 8 個 pixel


/* ################################################################
 *
 *  第一段：UYVY 讀取 + 轉 RGB
 *
 * ################################################################ */

/* ================================================================
 *  計算資料選擇：raw[32g+31 : 32g] = {Y1, V, Y0, U}，g = 0..3
 *  固定切片，無狀態
 * ================================================================ */
static void uyvy_in_select(const ap_uint<128> &raw,
                              ap_uint<8> u [UYVY_GRP],
                              ap_uint<8> y0[UYVY_GRP],
                              ap_uint<8> v [UYVY_GRP],
                              ap_uint<8> y1[UYVY_GRP])
{
#pragma HLS INLINE
    for (int g = 0; g < UYVY_GRP; g++) {
#pragma HLS UNROLL
        const int s = g * 32;
        u [g] = raw.range(s +  7, s +  0);
        y0[g] = raw.range(s + 15, s +  8);
        v [g] = raw.range(s + 23, s + 16);
        y1[g] = raw.range(s + 31, s + 24);
    }
}

/* ================================================================
 *  寫出資料選擇：8 個 pixel -> 192-bit
 *
 *    cvt_pair 輸出 {R,G,B} = px[23:16], px[15:8], px[7:0]
 *    記憶體 byte 順序 R,G,B（byte0 = R），px[k] 放在 [24k+23 : 24k]
 *    想要 BGR 的話 m = px[k] 不要換位即可
 * ================================================================ */
static ap_uint<192> uyvy_out_select(const ap_uint<24> px[UYVY_PIX])
{
#pragma HLS INLINE
    ap_uint<192> out = 0;
    for (int k = 0; k < UYVY_PIX; k++) {
#pragma HLS UNROLL
        ap_uint<24> m;
        m.range( 7,  0) = px[k].range(23, 16);   /* R */
        m.range(15,  8) = px[k].range(15,  8);   /* G */
        m.range(23, 16) = px[k].range( 7,  0);   /* B */
        out.range(k * 24 + 23, k * 24) = m;
    }
    return out;
}

static void read_convert(const ap_uint<128>          *in,
                         hls::stream<ap_uint<192> >  &rgb_out,
                         ap_uint<12>                  img_w,
                         ap_uint<12>                  img_h)
{
    const ap_uint<32> in_beats = (ap_uint<32>(img_w) >> 3) * ap_uint<32>(img_h);

RD: for (ap_uint<32> i = 0; i < in_beats; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN
        ap_uint<128> raw = in[i];

        ap_uint<8> u[UYVY_GRP], y0[UYVY_GRP], v[UYVY_GRP], y1[UYVY_GRP];
#pragma HLS ARRAY_PARTITION variable=u  complete
#pragma HLS ARRAY_PARTITION variable=y0 complete
#pragma HLS ARRAY_PARTITION variable=v  complete
#pragma HLS ARRAY_PARTITION variable=y1 complete
        uyvy_in_select(raw, u, y0, v, y1);           /* 計算資料選擇 */

        ap_uint<24> px[UYVY_PIX];
#pragma HLS ARRAY_PARTITION variable=px complete
        for (int g = 0; g < UYVY_GRP; g++) {           /* 運算：4 x cvt_pair */
#pragma HLS UNROLL
            cvt_pair(y0[g], y1[g], u[g], v[g], px[2 * g], px[2 * g + 1]);
        }

        rgb_out.write(uyvy_out_select(px));            /* 寫出資料選擇 */
    }
}


/* ################################################################
 *
 *  第二段：寫出資料選擇 —— 192 -> 128 輸出對齊狀態機 + AXI 寫出
 *
 *    每筆輸入 192 bit，每拍輸出 128 bit，暫存長度 L 只有三種：
 *
 *    狀態  暫存 L  讀入  輸出 w                    暫存 res'          下一狀態
 *    S0      0     是    d[127:0]                  d[191:128] (64)    S1
 *    S1     64     是    {d[63:0], res[63:0]}      d[191:64]  (128)   S2
 *    S2    128     否    res                       —                  S0
 *
 *    週期 3 拍：讀 2 筆（384 bit）、寫 3 筆（384 bit）
 *    要不要讀 = !st[1]（只有 S2 的 st[1]=1），單一 bit
 *    out[i] 是唯一寫入點、位址純遞增 -> burst 推得出來，II=1
 *
 * ################################################################ */

static void repack_write(hls::stream<ap_uint<192> > &si,
                         ap_uint<128> *out, ap_uint<32> out_beats) {
    ap_uint<128> res = 0;
    ap_uint<2>   st  = 0;

WR: for (ap_uint<32> i = 0; i < out_beats; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_OUT

        ap_uint<192> d = 0;
        if (!st[1])
            d = si.read();

        ap_uint<128> w    = 0;
        ap_uint<128> nres = res;
        ap_uint<2>   nst  = 0;

        switch (st) {
        case 0:  w = d.range(127, 0);
                 nres = d.range(191, 128);                        nst = 1; break;
        case 1:  w = (d.range(63, 0), res.range(63, 0));
                 nres = d.range(191, 64);                         nst = 2; break;
        case 2:  w = res;                                         nst = 0; break;
        default:                                                  nst = 0; break;
        }

        out[i] = w;                          // 唯一寫入點

        res = nres;
        st  = nst;
    }
}


/* ################################################################
 *
 *  Top
 *    img_w 必須是 16 的倍數。Max 4096x4096。
 *
 * ################################################################ */

void uyvy2rgb(ap_uint<128> *uyvy_axi_bus, ap_uint<128> *rgb_axi_bus,
              ap_uint<12> img_w, ap_uint<12> img_h) {

#pragma HLS INTERFACE m_axi port=uyvy_axi_bus offset=slave bundle=gmem0 \
                     depth=MAX_IN max_read_burst_length=128 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=rgb_axi_bus  offset=slave bundle=gmem1 \
                     depth=MAX_OUT max_write_burst_length=128 num_write_outstanding=4

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

    read_convert (uyvy_axi_bus, pix_fifo, img_w, img_h);
    repack_write (pix_fifo, rgb_axi_bus, out_beats);
}
