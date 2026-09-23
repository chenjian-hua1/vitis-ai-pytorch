/******************************************************************************
 * uyvy_resize_top.cpp
 *
 * AXI -> uyvy2rgb -> resize -> AXI，四段 DATAFLOW
 *
 *   DDR --m_axi 128b--> read_convert  4 x cvt_pair，一拍 8 pixel
 *                            |  hls::stream<ap_uint<192>>  rgb_ch    (8 pixel RGB)
 *                            v
 *                       compute_side  對齊 -> 6 x resize_pe -> line buffer
 *                            |  hls::stream<ap_uint<96>>   result_ch
 *                            v
 *                       pack_side     48/96 -> 128 對齊
 *                            |  hls::stream<ap_uint<128>>  word_ch
 *                            v
 *   DDR <--m_axi 128b-- axi_write_side
 *
 * 檔案分工
 *   uyvy2rgb_impl.cpp   UYVY -> RGB 運算（subtract_128 / mac / clamp_s / cvt_pair）
 *   resize_impl.cpp     resize 運算（dsp_addmul / dsp_shared / resize_pe）
 *   uyvy2rgb_top.cpp    uyvy2rgb 單獨 top
 *   resize_top.cpp      resize 單獨 top
 *   本檔（top）         把兩者串起來，並負責所有的資料選擇：
 *
 *   計算資料選擇（選哪些位元送進運算）
 *     uyvy_in_select      128-bit -> 4 組 (U, Y0, V, Y1)
 *     resize_in_select    192-bit 對齊狀態機 + 6 個 PE 的輸入
 *
 *   寫出資料選擇（運算結果怎麼排、往哪寫）
 *     uyvy_out_select     8 個 {R,G,B} -> 記憶體 byte 順序 -> 192-bit
 *     resize_out_select   lane -> 輸出欄、依小數位數取 8 bit、line buffer 寫回
 *     pack_side           48/96 -> 128 輸出對齊狀態機
 *
 * ============================================================
 *  串接後的差異
 * ============================================================
 *
 * 1. resize 輸入改成 192-bit（8 pixel），不再是 128-bit
 *      2 倍：每次運算正好 8 pixel = 1 拍，不需要對齊狀態機
 *      3 倍：每次運算 6 pixel，8 與 6 的公倍數 24 pixel = 3 拍 = 4 次運算
 *            狀態機從 9 狀態縮成 4 狀態（見 resize_in_select）
 *
 * 2. compute_side 的迴圈改成「每拍一次運算」，需要時才讀 FIFO
 *      2 倍：每拍讀一筆，8 pixel/拍，與 read_convert 同速
 *      3 倍：4 拍讀 3 筆，6 pixel/拍，read_convert 被 FIFO 反壓
 *    1920x1080 @250MHz：
 *      2 倍  259200 拍  ~1.04 ms
 *      3 倍  345600 拍  ~1.38 ms（瓶頸在 resize 一拍只能算 6 pixel）
 *
 * 3. 中間不再經過 DDR：原本 uyvy2rgb 的 192 -> 128 repack
 *    與 resize 的 128-bit 讀取都不需要了
 *
 * 4. 所有計數由 img_w / img_h / scale_mode 在各段內自行算出，
 *    host 只需給三個參數；scale_rate 改為內部常數
 *
 * 5. byte 順序：uyvy_out_select 產生 byte0 = R；resize 對三個 byte 通道
 *    一視同仁，輸出維持相同順序
 *
 * 注意：concat 一律指定給完整寬度的變數，
 * 不可直接寫進 .range()，否則會經過 64-bit 轉換被截斷。
 *****************************************************************************/

#include "uyvy_resize.h"
#include "resize_areaDown.h"
#include "uyvy2rgb_impl.h"      /* cvt_pair */
#include "resize_impl.h"        /* resize_pe、位元寬常數 */
#include "ap_int.h"
#include "hls_stream.h"

/* ---------------------------------------------------------------- 常數 */

#define UYVY_GRP   4        /* 一拍 4 組 UYVY */
#define UYVY_PIX   8        /* 一拍 8 個 pixel */

#define OUT_W_MAX  960      /* 輸出寬度上限 */
#define QUAD_W_MAX 240      /* OUT_W_MAX / 4，每 bank 的深度 */

#define SR_S3      7282     /* 3 倍 scale_rate：65536/9，16 bit */
#define SR_S2      1        /* 2 倍 scale_rate：4/4，2 bit */

/* ---- DSP 顆數 ----
 * 組 = 一個輸出欄的一個通道
 *   3 倍：2 欄 x RGB = 6 組，B 16 bit 無法打包 -> 每顆 1 組
 *   2 倍：4 欄 x RGB = 12 組，B 2 bit 可打包  -> 每顆 2 組 */
#define GRP_S3        6
#define GRP_S2        12
#define LANE_S3       1
#define LANE_S2       2
#define DSP_NEED(g,l) (((g) + (l) - 1) / (l))
#define DSP_MAX(a,b)  ((a) > (b) ? (a) : (b))
#define N_DSP         DSP_MAX(DSP_NEED(GRP_S3, LANE_S3), DSP_NEED(GRP_S2, LANE_S2))   /* = 6 */

#define RGB_FIFO_DEPTH  32
#define RES_FIFO_DEPTH  64
#define WORD_FIFO_DEPTH 64

#define MAX_IN_BEATS    (1920 * 1080 / 8)          /* co-sim 深度，2 倍最大輸入 */
#define MAX_OUT_WORDS   (960 * 540 * 3 / 16)       /* co-sim 深度，2 倍最大輸出 */


/* ================================================================
 *  尺寸計算（各段各自呼叫，不在 DATAFLOW 區域內做運算）
 *  除以常數 3 會被合成成乘法 + 位移
 * ================================================================ */
static void out_size(ap_uint<12> img_w, ap_uint<12> img_h, bool s3,
                     ap_uint<12> &out_w, ap_uint<12> &out_h)
{
#pragma HLS INLINE
    out_w = s3 ? (ap_uint<12>)(img_w / 3) : (ap_uint<12>)(img_w >> 1);
    out_h = s3 ? (ap_uint<12>)(img_h / 3) : (ap_uint<12>)(img_h >> 1);
}


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
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_BEATS
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
 *  第二段：resize
 *
 * ################################################################ */

/* ================================================================
 *  計算資料選擇
 *
 *  (1) 192-bit 輸入對齊狀態機（d = 本拍讀入，hold = 暫存）
 *
 *   3 倍（每次運算 6 pixel = 144 bit）
 *   狀態  暫存 L  讀入  運算資料 win                   新暫存 hold'        下一狀態
 *   S0      0     是    d[143:0]                       d[191:144]  (48)    S1
 *   S1     48     是    {d[ 95:0], hold[47:0]}         d[191: 96]  (96)    S2
 *   S2     96     是    {d[ 47:0], hold[95:0]}         d[191: 48] (144)    S3
 *   S3    144     否    hold[143:0]                    —                   S0
 *   週期 4 次運算、讀 3 筆
 *
 *   2 倍（每次運算 8 pixel = 192 bit = 正好一拍）
 *   只有 S0：每拍讀一筆，win = d
 *
 *   要不要讀只看狀態（3 倍 S3 不讀），在呼叫前由迴圈決定
 *
 *  (2) PE 輸入選擇（slot s x 通道 c）
 *      3 倍：u0,u1,u2 = p[3s..3s+2]，v = line_buf 該欄（bsel 選 bank，
 *            slot0 取高 lane、slot1 取低 lane）
 *      2 倍：u0,u1,u2 = p[4s..4s+2]，v = p[4s+3]，
 *            lbp = {line_buf[i], line_buf[i+1]}（slot0 -> bankA，slot1 -> bankB）
 * ================================================================ */
static void resize_in_select(const ap_uint<192> &d,
                             const ap_uint<144> &hold,
                             ap_uint<2>          st,
                             bool                s3,
                             const ap_uint<LBW> &qA,
                             const ap_uint<LBW> &qB,
                             bool                bsel,
                             ap_uint<144>       &nhold,
                             ap_uint<2>         &nst,
                             ap_uint<8>          u0[2][3],
                             ap_uint<8>          u1[2][3],
                             ap_uint<8>          u2[2][3],
                             ap_uint<ACCW>       v[2][3],
                             ap_uint<OPW>        lbp[2][3])
{
#pragma HLS INLINE

    /* ---- (1) 輸入對齊狀態機 ---- */
    ap_uint<192> win = 0;
    nhold = hold;
    nst   = 0;
    if (s3) {
        switch (st) {
        case 0: win = d.range(143, 0);
                nhold = d.range(191, 144);                         nst = 1; break;
        case 1: win = (d.range(95, 0), hold.range(47, 0));
                nhold = d.range(191, 96);                          nst = 2; break;
        case 2: win = (d.range(47, 0), hold.range(95, 0));
                nhold = d.range(191, 48);                          nst = 3; break;
        case 3: win = hold;                                        nst = 0; break;
        default:                                                   nst = 0; break;
        }
    } else {
        win = d;                                                   nst = 0;
    }

    /* ---- (2) PE 輸入選擇 ---- */
    ap_uint<LBW> q3 = bsel ? qB : qA;          /* 3 倍本次使用的 bank */

    for (int s = 0; s < 2; s++) {
#pragma HLS UNROLL
        for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
            const int b3 = 3*s*24 + c*8;       /* 3 倍：slot s 第一個 pixel 的通道 c */
            const int b2 = 4*s*24 + c*8;       /* 2 倍 */
            ap_uint<LBW> q2 = (s == 0) ? qA : qB;

            if (s3) {
                u0[s][c]  = win.range(b3      + 7, b3);
                u1[s][c]  = win.range(b3 + 24 + 7, b3 + 24);
                u2[s][c]  = win.range(b3 + 48 + 7, b3 + 48);
                if (s == 0) v[s][c] = q3.range(c*24 + 23, c*24 + 12);
                else        v[s][c] = q3.range(c*24 + 11, c*24);
                lbp[s][c] = 0;
            } else {
                u0[s][c]  = win.range(b2      + 7, b2);
                u1[s][c]  = win.range(b2 + 24 + 7, b2 + 24);
                u2[s][c]  = win.range(b2 + 48 + 7, b2 + 48);
                v[s][c]   = (ap_uint<8>)win.range(b2 + 72 + 7, b2 + 72);
                lbp[s][c] = q2.range(c*24 + 23, c*24);
            }
        }
    }
}

/* ================================================================
 *  寫出資料選擇
 *
 *  (1) lane -> 輸出欄：ph/pl[s][c] 是第 s 顆（x 通道 c）DSP 拆出來的兩個 lane
 *        3 倍：只有 lo 有效，slot s = 輸出欄 s
 *        2 倍：hi = 欄 2s，lo = 欄 2s+1
 *
 *  (2) 輸出 pixel：依 scale_rate 小數位數取 8 bit
 *        3 倍 gp[23:16]，2 倍 gp[9:2]
 *
 *  (3) line buffer 寫回：last_row 寫 0；否則 B = 1，乘積即累加值，取 [11:0]
 *        bankA 每通道 = {欄0, 欄1}
 *        bankB 每通道 = 3 倍：同 bankA（由 bsel 決定寫哪個 bank）
 *                       2 倍：{欄2, 欄3}
 * ================================================================ */
static void resize_out_select(const ap_uint<PROD_W> ph[2][3],
                              const ap_uint<PROD_W> pl[2][3],
                              bool                  s3,
                              bool                  bsel,
                              bool                  last_row,
                              ap_uint<96>          &res,
                              ap_uint<LBW>         &wA,
                              ap_uint<LBW>         &wB,
                              bool                 &weA,
                              bool                 &weB)
{
#pragma HLS INLINE

    /* ---- (1) lane -> 輸出欄 ---- */
    ap_uint<PROD_W> gp[4][3];
#pragma HLS ARRAY_PARTITION variable=gp complete dim=0
    for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
        if (s3) {
            gp[0][c] = pl[0][c];
            gp[1][c] = pl[1][c];
            gp[2][c] = 0;
            gp[3][c] = 0;
        } else {
            gp[0][c] = ph[0][c];
            gp[1][c] = pl[0][c];
            gp[2][c] = ph[1][c];
            gp[3][c] = pl[1][c];
        }
    }

    /* ---- (2) 輸出 pixel ---- */
    res = 0;
    for (int col = 0; col < 4; col++) {
#pragma HLS UNROLL
        for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
            ap_uint<8> px;
            if (s3) px = gp[col][c].range(FRAC_S3 + 7, FRAC_S3);
            else    px = gp[col][c].range(FRAC_S2 + 7, FRAC_S2);
            if (col < 2 || !s3)                         /* 3 倍只有 2 欄 */
                res.range(col*24 + c*8 + 7, col*24 + c*8) = px;
        }
    }

    /* ---- (3) line buffer 寫回 ---- */
    wA = 0;
    wB = 0;
    if (!last_row) {
        for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
            ap_uint<ACCW> g0 = gp[0][c].range(ACCW - 1, 0);
            ap_uint<ACCW> g1 = gp[1][c].range(ACCW - 1, 0);
            ap_uint<ACCW> g2 = gp[2][c].range(ACCW - 1, 0);
            ap_uint<ACCW> g3 = gp[3][c].range(ACCW - 1, 0);
            ap_uint<OPW>  sA  = (g0, g1);
            ap_uint<OPW>  s23 = (g2, g3);
            ap_uint<OPW>  sB  = s3 ? sA : s23;
            wA.range(c*24 + 23, c*24) = sA;
            wB.range(c*24 + 23, c*24) = sB;
        }
    }
    weA = !s3 || !bsel;
    weB = !s3 ||  bsel;
}

/* ================================================================
 *  compute_side：每拍一次運算
 *    resize_in_select -> resize_pe x N_DSP -> resize_out_select
 * ================================================================ */
static void compute_side(hls::stream<ap_uint<192> > &rgb_in,
                         hls::stream<ap_uint<96> >  &result_out,
                         ap_uint<12>                 img_w,
                         ap_uint<12>                 img_h,
                         ap_uint<1>                  scale_mode)
{
    ap_uint<LBW> lbA[QUAD_W_MAX];   /* 欄 4k, 4k+1 */
    ap_uint<LBW> lbB[QUAD_W_MAX];   /* 欄 4k+2, 4k+3 */
#pragma HLS BIND_STORAGE variable=lbA type=RAM_S2P impl=BRAM
#pragma HLS BIND_STORAGE variable=lbB type=RAM_S2P impl=BRAM

    const bool s3 = (scale_mode == SCALE_3);

    ap_uint<12> out_w, out_h;
    out_size(img_w, img_h, s3, out_w, out_h);

    /* 運算次數 = 輸入 pixel 數 / 每次運算 pixel 數（6 或 8） */
    const ap_uint<32> n_pix     = ap_uint<32>(img_w) * ap_uint<32>(img_h);
    const ap_uint<32> total_ops = s3 ? (ap_uint<32>)(n_pix / 6) : (ap_uint<32>)(n_pix >> 3);

    const ap_uint<LOG2_CEIL(3)> v_taps = s3 ? 3 : 2;
    const ap_uint<3>            n_out  = s3 ? 2 : 4;
    const ap_uint<SRW>          sr     = s3 ? (ap_uint<SRW>)SR_S3 : (ap_uint<SRW>)SR_S2;
    const ap_uint<LOG2_CEIL((OUT_W_MAX+3)>>2)> quad_w = (out_w + 3) >> 2;

    /* ---- 輸入對齊狀態 ---- */
    ap_uint<144> hold = 0;
    ap_uint<2>   st   = 0;

    /* ---- 位置追蹤 ---- */
    ap_uint<LOG2_CEIL(OUT_W_MAX)> ox           = 0;
    ap_uint<LOG2_CEIL(3)>         row_in_block = 0;

    init_loop: for (int i = 0; i < quad_w; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=QUAD_W_MAX
        lbA[i] = 0; lbB[i] = 0;
    }

    op_loop: for (ap_uint<32> i = 0; i < total_ops; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=345600
#pragma HLS DEPENDENCE variable=lbA inter false
#pragma HLS DEPENDENCE variable=lbB inter false

        /* ---- 讀入：3 倍 S3 不讀，其餘每拍讀一筆 ---- */
        ap_uint<192> d = 0;
        if (!(s3 && st == 3))
            d = rgb_in.read();

        ap_uint<10> base_idx = ox >> 2;
        bool        bsel     = ox[1];     /* 3 倍：0 -> bankA，1 -> bankB */
        bool        last_row = (row_in_block == v_taps - 1);

        ap_uint<LBW> qA = lbA[base_idx];
        ap_uint<LBW> qB = lbB[base_idx];

        /* ---- 計算資料選擇 ---- */
        ap_uint<144>  nhold;
        ap_uint<2>    nst;
        ap_uint<8>    u0[2][3], u1[2][3], u2[2][3];
        ap_uint<ACCW> v[2][3];
        ap_uint<OPW>  lbp[2][3];
#pragma HLS ARRAY_PARTITION variable=u0  complete dim=0
#pragma HLS ARRAY_PARTITION variable=u1  complete dim=0
#pragma HLS ARRAY_PARTITION variable=u2  complete dim=0
#pragma HLS ARRAY_PARTITION variable=v   complete dim=0
#pragma HLS ARRAY_PARTITION variable=lbp complete dim=0
        resize_in_select(d, hold, st, s3, qA, qB, bsel, nhold, nst, u0, u1, u2, v, lbp);
        hold = nhold;
        st   = nst;

        /* ---- 運算：N_DSP 顆，每顆服務 1 組（3 倍）或 2 組（2 倍） ---- */
        ap_uint<SRW>    mul_b = last_row ? sr : (ap_uint<SRW>)1;
        ap_uint<PROD_W> ph[2][3], pl[2][3];
#pragma HLS ARRAY_PARTITION variable=ph complete dim=0
#pragma HLS ARRAY_PARTITION variable=pl complete dim=0
        dsp_loop: for (int k = 0; k < N_DSP; k++) {
#pragma HLS UNROLL
            const int s = k / 3;
            const int c = k % 3;
            resize_pe(u0[s][c], u1[s][c], u2[s][c], v[s][c], lbp[s][c],
                      s3, mul_b, ph[s][c], pl[s][c]);
        }

        /* ---- 寫出資料選擇 ---- */
        ap_uint<96>  res;
        ap_uint<LBW> wA, wB;
        bool         weA, weB;
        resize_out_select(ph, pl, s3, bsel, last_row, res, wA, wB, weA, weB);

        if (last_row)
            result_out.write(res);
        if (weA) lbA[base_idx] = wA;
        if (weB) lbB[base_idx] = wB;

        ox += n_out;
        if (ox >= out_w) {
            ox = 0;
            row_in_block++;
            if (row_in_block == v_taps)
                row_in_block = 0;
        }
    }
}


/* ################################################################
 *
 *  第三段：寫出資料選擇 —— 48/96 -> 128 輸出對齊狀態機
 *
 *  3 倍（每筆 result 48 bit，r = res[47:0]）
 *   狀態  暫存 L  動作                                       寫出  下一 L
 *   S0      0    hold[ 47:  0] = r                              —     48
 *   S1     48    hold[ 95: 48] = r                              —     96
 *   S2     96    out {r[31:0], hold[95:0]}；hold[15:0]=r[47:32] 是    16
 *   S3     16    hold[ 63: 16] = r                              —     64
 *   S4     64    hold[111: 64] = r                              —    112
 *   S5    112    out {r[15:0], hold[111:0]}；hold[31:0]=r[47:16] 是   32
 *   S6     32    hold[ 79: 32] = r                              —     80
 *   S7     80    out {r[47:0], hold[79:0]}                      是     0
 *
 *  2 倍（每筆 result 96 bit，r = res[95:0]）
 *   S0      0    hold[95:0] = r                                 —     96
 *   S1     96    out {r[31:0], hold[95:0]}；hold[63:0]=r[95:32] 是    64
 *   S2     64    out {r[63:0], hold[63:0]}；hold[31:0]=r[95:64] 是    32
 *   S3     32    out {r[95:0], hold[31:0]}                      是     0
 *
 * ################################################################ */

static void pack_side(hls::stream<ap_uint<96> >  &result_in,
                      hls::stream<ap_uint<128> > &word_out,
                      ap_uint<12>                 img_w,
                      ap_uint<12>                 img_h,
                      ap_uint<1>                  scale_mode)
{
    const bool s3 = (scale_mode == SCALE_3);

    ap_uint<12> out_w, out_h;
    out_size(img_w, img_h, s3, out_w, out_h);
    const ap_uint<32> out_pix       = ap_uint<32>(out_w) * ap_uint<32>(out_h);
    const ap_uint<32> total_results = s3 ? (ap_uint<32>)(out_pix >> 1)
                                         : (ap_uint<32>)(out_pix >> 2);

    ap_uint<112> hold = 0;
    ap_uint<3>   st   = 0;

    pack_loop: for (ap_uint<32> r = 0; r < total_results; r++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=129600

        ap_uint<96>  res = result_in.read();
        ap_uint<48>  r3  = res.range(47, 0);

        ap_uint<112> nh   = hold;
        ap_uint<128> word = 0;
        bool         emit = false;
        ap_uint<3>   nst  = 0;

        if (s3) {
            switch (st) {
            case 0: nh.range( 47,  0) = r3;                           nst = 1; break;
            case 1: nh.range( 95, 48) = r3;                           nst = 2; break;
            case 2: word = (r3.range(31, 0), hold.range( 95, 0)); emit = true;
                    nh.range( 15,  0) = r3.range(47, 32);             nst = 3; break;
            case 3: nh.range( 63, 16) = r3;                           nst = 4; break;
            case 4: nh.range(111, 64) = r3;                           nst = 5; break;
            case 5: word = (r3.range(15, 0), hold.range(111, 0)); emit = true;
                    nh.range( 31,  0) = r3.range(47, 16);             nst = 6; break;
            case 6: nh.range( 79, 32) = r3;                           nst = 7; break;
            case 7: word = (r3,              hold.range( 79, 0)); emit = true;
                                                                      nst = 0; break;
            default:                                                  nst = 0; break;
            }
        } else {
            switch (st) {
            case 0: nh.range( 95,  0) = res;                          nst = 1; break;
            case 1: word = (res.range(31, 0), hold.range( 95, 0)); emit = true;
                    nh.range( 63,  0) = res.range(95, 32);            nst = 2; break;
            case 2: word = (res.range(63, 0), hold.range( 63, 0)); emit = true;
                    nh.range( 31,  0) = res.range(95, 64);            nst = 3; break;
            case 3: word = (res,              hold.range( 31, 0)); emit = true;
                                                                      nst = 0; break;
            default:                                                  nst = 0; break;
            }
        }

        if (emit)
            word_out.write(word);

        hold = nh;
        st   = nst;
    }

    /* 收尾：輸出 pixel 數不是 16 的倍數時，補 0 成最後一個 word */
    if (st != 0) {
        ap_uint<7> L = 0;
        if (s3) {
            switch (st) {
            case 1: L =  48; break;  case 2: L =  96; break;
            case 3: L =  16; break;  case 4: L =  64; break;
            case 5: L = 112; break;  case 6: L =  32; break;
            case 7: L =  80; break;  default: break;
            }
        } else {
            switch (st) {
            case 1: L = 96; break;   case 2: L = 64; break;
            case 3: L = 32; break;   default: break;
            }
        }
        ap_uint<112> mask = (((ap_uint<113>)1) << L) - 1;
        ap_uint<128> tail = hold & mask;
        word_out.write(tail);
    }
}


/* ################################################################
 *
 *  第四段：AXI 寫出（位址即迴圈變數、無條件包裹 -> burst）
 *
 * ################################################################ */

static void axi_write_side(hls::stream<ap_uint<128> > &word_in,
                           ap_uint<128>               *out,
                           ap_uint<12>                 img_w,
                           ap_uint<12>                 img_h,
                           ap_uint<1>                  scale_mode)
{
    const bool s3 = (scale_mode == SCALE_3);

    ap_uint<12> out_w, out_h;
    out_size(img_w, img_h, s3, out_w, out_h);
    const ap_uint<32> out_pix   = ap_uint<32>(out_w) * ap_uint<32>(out_h);
    const ap_uint<32> out_words = (out_pix * 3 + 15) >> 4;   /* ceil(bytes / 16) */

    write_loop: for (ap_uint<32> i = 0; i < out_words; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_OUT_WORDS
        out[i] = word_in.read();
    }
}


/* ################################################################
 *
 *  Top
 *
 * ################################################################ */

void uyvy_resize(ap_uint<128> *uyvy_axi_bus,
                 ap_uint<128> *rgb_axi_bus,
                 ap_uint<12>   img_w,
                 ap_uint<12>   img_h,
                 ap_uint<1>    scale_mode)
{
#pragma HLS INTERFACE m_axi port=uyvy_axi_bus offset=slave bundle=gmem0 \
                     depth=MAX_IN_BEATS  max_read_burst_length=128  num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=rgb_axi_bus  offset=slave bundle=gmem1 \
                     depth=MAX_OUT_WORDS max_write_burst_length=128 num_write_outstanding=4

#pragma HLS INTERFACE s_axilite port=uyvy_axi_bus bundle=control
#pragma HLS INTERFACE s_axilite port=rgb_axi_bus  bundle=control
#pragma HLS INTERFACE s_axilite port=img_w        bundle=control
#pragma HLS INTERFACE s_axilite port=img_h        bundle=control
#pragma HLS INTERFACE s_axilite port=scale_mode   bundle=control
#pragma HLS INTERFACE s_axilite port=return       bundle=control

#pragma HLS DATAFLOW

    hls::stream<ap_uint<192> > rgb_ch;
    hls::stream<ap_uint<96>  > result_ch;
    hls::stream<ap_uint<128> > word_ch;
#pragma HLS STREAM       variable=rgb_ch    depth=RGB_FIFO_DEPTH
#pragma HLS STREAM       variable=result_ch depth=RES_FIFO_DEPTH
#pragma HLS STREAM       variable=word_ch   depth=WORD_FIFO_DEPTH
#pragma HLS BIND_STORAGE variable=rgb_ch    type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=result_ch type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=word_ch   type=fifo impl=srl

    read_convert  (uyvy_axi_bus, rgb_ch, img_w, img_h);
    compute_side  (rgb_ch, result_ch, img_w, img_h, scale_mode);
    pack_side     (result_ch, word_ch, img_w, img_h, scale_mode);
    axi_write_side(word_ch, rgb_axi_bus, img_w, img_h, scale_mode);
}
