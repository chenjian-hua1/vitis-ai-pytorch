/******************************************************************************
 * uyvyResizeQuant_top.cpp
 *
 * AXI -> uyvy2rgb -> resize -> quantize -> AXI，兩路輸出同時寫入 DDR
 *
 *   DDR --m_axi gmem0 (128b)--> compute_side
 *        uyvy2rgb -> 對齊 -> 12 x resize_pe -> out_select
 *                                   |                 |
 *                                   |             quantize
 *                                   v                 v
 *                              pack_side(R)      pack_side(Q)
 *                                   | rs_ch           | q_ch   （各一個 FIFO）
 *                                   v                 v
 *   DDR <--m_axi gmem1-- axi_write_side   axi_write_side --m_axi gmem2--> DDR
 *        (resize 結果)                         (quantize 結果)
 *
 *   - 三組獨立 m_axi：gmem0 讀入、gmem1 寫 resize 結果、gmem2 寫 quantize 結果
 *   - 兩路輸出的資料量、格式完全相同（byte0 = R，每 16 byte 一個 word，尾端補 0），
 *     out_words 共用同一個值
 *   - 兩條路各有一組 pack_side 狀態（hold / pst），互不干擾
 *   - 兩個 axi_write_side 是獨立的 DATAFLOW process，各自 burst，
 *     任一邊 DDR 忙碌只會讓自己的 FIFO 滿，compute_side 會一起被反壓（兩路速率相同）
 *
 * ---------------------------------------------------------------------------
 * 以下為原本的說明
 * ---------------------------------------------------------------------------
 * Quantize 的位置
 *   resize_out_select 輸出的 res（96 bit = 4 個 RGB pixel）是最終輸出像素，
 *   在進 pack_side 前通過 quantize_kernel<96>（= 2 個 quantize_core，每個處理 2 pixel）。
 *   line buffer 的回寫值（wA / wB）是累加中間值，不量化。
 *   只在 last_row（會輸出 pixel 的那一拍）才需要量化結果。
 *
 *   quantize_kernel 全展開、純資料路徑，beat_loop 仍維持 II=1，
 *   只增加管線深度（latency），不影響吞吐。
 *
 *   byte 順序：res 每個 pixel 為 byte0 = R, byte1 = G, byte2 = B，
 *   quantize_core 收到的 48 bit = {pixel1, pixel0}，pixel0 在低位。
 *   若 quantize_core 預期不同的通道順序，改 quant_select() 即可。
 *
 * 其餘架構（FIFO 取捨、3 倍一次算 4 block、參數計算只在 top 做一次）
 * 與 uyvy_resize_top.cpp 相同，這裡不再重複說明。
 *
 * 注意：concat 一律指定給完整寬度的變數，不可直接寫進 .range()。
 *****************************************************************************/

// #include "uyvy_resize_top.h"
#include "uyvyResizeQuant_top.h"
#include "../../impl/base_func.h"
#include "../../impl/uyvy2rgb_impl.h"
#include "../../impl/resize_impl.h"        /* resize_pe、位元寬常數 */
#include "../../impl/quantize_impl.h"      /* quantize_core */
#include "ap_int.h"
#include "hls_stream.h"

/* ---------------------------------------------------------------- 常數 */

#define UYVY_GRP   4
#define UYVY_PIX   8

#define OUT_W_MAX  960
#define QUAD_W_MAX 240

#define SR_S3      7282
#define SR_S2      1

#define N_COL         4
#define GRP_S3        (N_COL * 3)
#define GRP_S2        (N_COL * 3)
#define LANE_S3       1
#define LANE_S2       2
#define DSP_NEED(g,l) (((g) + (l) - 1) / (l))
#define DSP_MAX(a,b)  ((a) > (b) ? (a) : (b))
#define N_DSP         DSP_MAX(DSP_NEED(GRP_S3, LANE_S3), DSP_NEED(GRP_S2, LANE_S2))   /* = 12 */
#define N_SLOT        (N_DSP / 3)                                                    /* = 4  */

#define RES_BITS        (N_COL * 24)               /* 96：每次運算 4 pixel */

#define WORD_FIFO_DEPTH 64

#define MAX_IN_BEATS    (1920 * 1080 / 8)
#define MAX_OUT_WORDS   (960 * 540 * 3 / 16)


/* ================================================================
 *  參數計算：只在 top 算一次（fabric 乘法器，不佔 DSP）
 * ================================================================ */
static ap_uint<24> mul12(ap_uint<12> a, ap_uint<12> b)
{
    ap_uint<24> p = a * b;
#pragma HLS BIND_OP variable=p op=mul impl=fabric latency=2
    return p;
}
 
static void calc_params(ap_uint<12>  img_w,
                        ap_uint<12>  img_h,
                        bool         s3,
                        ap_uint<32> &in_beats,
                        ap_uint<32> &out_words,
                        ap_uint<12> &out_w)
{
#pragma HLS ALLOCATION function instances=mul12 limit=1
    const ap_uint<12> K_DIV3 = 1366;                     /* x/3 = (x*1366)>>12 */
 
    ap_uint<24> w1366 = mul12(img_w, K_DIV3);
    ap_uint<24> h1366 = mul12(img_h, K_DIV3);
    out_w = s3 ? (ap_uint<12>)(w1366 >> 12) : (ap_uint<12>)(img_w >> 1);
    ap_uint<12> out_h = s3 ? (ap_uint<12>)(h1366 >> 12) : (ap_uint<12>)(img_h >> 1);
 
    ap_uint<24> out_pix = mul12(out_w, out_h);
    ap_uint<26> out_bytes = ((ap_uint<26>)out_pix << 1) + out_pix;     /* x3 */
    out_words     = (out_bytes + 15) >> 4;                             /* ceil(bytes/16) */
 
    ap_uint<24> beats = mul12((ap_uint<12>)(img_w >> 3), img_h);
    in_beats = beats;
}
 
 
/* ################################################################
 *  UYVY -> RGB（INLINE）
 * ################################################################ */
 
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
 
static ap_uint<192> uyvy_convert(const ap_uint<128> &raw)
{
#pragma HLS INLINE
    ap_uint<8> u[UYVY_GRP], y0[UYVY_GRP], v[UYVY_GRP], y1[UYVY_GRP];
#pragma HLS ARRAY_PARTITION variable=u  complete
#pragma HLS ARRAY_PARTITION variable=y0 complete
#pragma HLS ARRAY_PARTITION variable=v  complete
#pragma HLS ARRAY_PARTITION variable=y1 complete
    uyvy_in_select(raw, u, y0, v, y1);
 
    ap_uint<24> px[UYVY_PIX];
#pragma HLS ARRAY_PARTITION variable=px complete
    for (int g = 0; g < UYVY_GRP; g++) {
#pragma HLS UNROLL
        cvt_pair(y0[g], y1[g], u[g], v[g], px[2 * g], px[2 * g + 1]);
    }
 
    return uyvy_out_select(px);
}
 
 
/* ################################################################
 *  resize（INLINE 選擇邏輯）
 * ################################################################ */
 
/* ================================================================
 *  計算資料選擇
 *
 *  (1) 192-bit 輸入對齊狀態機（beat = 本拍，prev = 上一拍）
 *      每次運算完剩下的位元一定是當前 beat 的高位，只需存上一拍
 *
 *   3 倍（每次運算 12 pixel = 288 bit）
 *   狀態  暫存 L  do_op  運算資料 win[287:0]              下一狀態
 *   S0      0      0     —（本拍整筆存入 prev）           S1
 *   S1    192      1     {beat[ 95:0], prev[191:  0]}    S2   剩 beat[191:96]
 *   S2     96      1     {beat[191:0], prev[191: 96]}    S0   剩 0
 *   週期 3 拍、運算 2 次
 *
 *   2 倍（每次運算 8 pixel = 192 bit = 正好一拍）
 *   每拍 do_op = 1，win = beat，不需要狀態
 *
 *  (2) PE 輸入選擇（slot s = 輸出欄 0..3，通道 c）
 *      3 倍：u0,u1,u2 = p[3s..3s+2]，v = line_buf 欄 s
 *            （欄 0/1 在 bankA 高/低 lane、欄 2/3 在 bankB 高/低 lane）
 *      2 倍：slot 0..1 使用；u0,u1,u2 = p[4s..4s+2]，v = p[4s+3]，
 *            lbp = {line_buf[2s], line_buf[2s+1]}（slot0 -> bankA，slot1 -> bankB）
 *            slot 2..3 輸入給 0（閒置）
 * ================================================================ */
static void resize_in_select(const ap_uint<192> &beat,
                             const ap_uint<192> &prev,
                             ap_uint<2>          st,
                             bool                s3,
                             const ap_uint<LBW> &qA,
                             const ap_uint<LBW> &qB,
                             ap_uint<2>         &nst,
                             bool               &do_op,
                             ap_uint<8>          u0[N_SLOT][3],
                             ap_uint<8>          u1[N_SLOT][3],
                             ap_uint<8>          u2[N_SLOT][3],
                             ap_uint<ACCW>       v[N_SLOT][3],
                             ap_uint<OPW>        lbp[N_SLOT][3])
{
#pragma HLS INLINE
 
    /* ---- (1) 輸入對齊狀態機 ---- */
    ap_uint<288> win = 0;
    nst   = 0;
    do_op = false;
    if (s3) {
        switch (st) {
        case 0:                                                      nst = 1; break; /* 暖機 */
        case 1: win = (beat.range( 95, 0), prev.range(191,  0)); do_op = true; nst = 2; break;
        case 2: win = (beat.range(191, 0), prev.range(191, 96)); do_op = true; nst = 0; break;
        default:                                                     nst = 0; break;
        }
    } else {
        win   = beat;
        do_op = true;
        nst   = 0;
    }
 
    /* ---- (2) PE 輸入選擇 ---- */
    for (int s = 0; s < N_SLOT; s++) {
#pragma HLS UNROLL
        for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
            const int b3 = 3*s*24 + c*8;
            const int b2 = 4*s*24 + c*8;
 
            ap_uint<LBW>  qs   = (s < 2) ? qA : qB;
            ap_uint<ACCW> lane;
            if ((s & 1) == 0) lane = qs.range(c*24 + 23, c*24 + 12);
            else              lane = qs.range(c*24 + 11, c*24);
 
            if (s3) {
                u0[s][c]  = win.range(b3      + 7, b3);
                u1[s][c]  = win.range(b3 + 24 + 7, b3 + 24);
                u2[s][c]  = win.range(b3 + 48 + 7, b3 + 48);
                v[s][c]   = lane;
                lbp[s][c] = 0;
            } else if (s < 2) {
                ap_uint<LBW> q2 = (s == 0) ? qA : qB;
                u0[s][c]  = win.range(b2      + 7, b2);
                u1[s][c]  = win.range(b2 + 24 + 7, b2 + 24);
                u2[s][c]  = win.range(b2 + 48 + 7, b2 + 48);
                v[s][c]   = (ap_uint<8>)win.range(b2 + 72 + 7, b2 + 72);
                lbp[s][c] = q2.range(c*24 + 23, c*24);
            } else {
                u0[s][c] = 0; u1[s][c] = 0; u2[s][c] = 0; v[s][c] = 0; lbp[s][c] = 0;
            }
        }
    }
}
 
static void resize_out_select(const ap_uint<PROD_W> ph[N_SLOT][3],
                              const ap_uint<PROD_W> pl[N_SLOT][3],
                              bool                  s3,
                              bool                  last_row,
                              ap_uint<RES_BITS>    &res,
                              ap_uint<LBW>         &wA,
                              ap_uint<LBW>         &wB)
{
#pragma HLS INLINE
 
    ap_uint<PROD_W> gp[N_COL][3];
#pragma HLS ARRAY_PARTITION variable=gp complete dim=0
    for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
        if (s3) {
            gp[0][c] = pl[0][c];
            gp[1][c] = pl[1][c];
            gp[2][c] = pl[2][c];
            gp[3][c] = pl[3][c];
        } else {
            gp[0][c] = ph[0][c];
            gp[1][c] = pl[0][c];
            gp[2][c] = ph[1][c];
            gp[3][c] = pl[1][c];
        }
    }
 
    res = 0;
    for (int col = 0; col < N_COL; col++) {
#pragma HLS UNROLL
        for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
            ap_uint<8> px;
            if (s3) px = gp[col][c].range(FRAC_S3 + 7, FRAC_S3);
            else    px = gp[col][c].range(FRAC_S2 + 7, FRAC_S2);
            res.range(col*24 + c*8 + 7, col*24 + c*8) = px;
        }
    }
 
    wA = 0;
    wB = 0;
    if (!last_row) {
        for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
            ap_uint<ACCW> g0 = gp[0][c].range(ACCW - 1, 0);
            ap_uint<ACCW> g1 = gp[1][c].range(ACCW - 1, 0);
            ap_uint<ACCW> g2 = gp[2][c].range(ACCW - 1, 0);
            ap_uint<ACCW> g3 = gp[3][c].range(ACCW - 1, 0);
            ap_uint<OPW>  sA = (g0, g1);
            ap_uint<OPW>  sB = (g2, g3);
            wA.range(c*24 + 23, c*24) = sA;
            wB.range(c*24 + 23, c*24) = sB;
        }
    }
}
 
 
/* ################################################################
 *  Quantize（INLINE）
 * ################################################################ */
 
/* ================================================================
 *  quantize_kernel<LENGTH>：LENGTH/24 個 pixel，每個 quantize_core 處理 2 pixel
 *  （與 quantize_top.cpp 相同；static 避免與獨立 top 的符號衝突）
 * ================================================================ */
template <int LENGTH>
static void quantize_kernel(const ap_uint<LENGTH> &input, ap_uint<LENGTH> &output)
{
#pragma HLS INLINE
    static_assert(LENGTH % 24 == 0, "LENGTH must be a multiple of 24");
 
    for (int i = 0; i < LENGTH / 48; i++) {
#pragma HLS UNROLL
        const int base = 48 * i;
        ap_uint<48> quantize_output;
        quantize_core(input.range(base + 47, base), quantize_output);
        output.range(base + 47, base) = quantize_output;
    }
 
    if constexpr (LENGTH % 48 == 24) {
        constexpr int base = (LENGTH / 48) * 48;
        ap_uint<48> last_pixels = 0;
        last_pixels.range(23, 0) = input.range(base + 23, base);
 
        ap_uint<48> quantize_output;
        quantize_core(last_pixels, quantize_output);
        output.range(base + 23, base) = quantize_output.range(23, 0);
    }
}
 
/* ================================================================
 *  quantize_stage：res（4 pixel, 96 bit）-> 量化後 96 bit
 *  96 = 2 x 48，剛好 2 個 quantize_core，無餘數 pixel
 *  兩種縮放模式共用同一組 core（量化發生在 resize 之後）
 * ================================================================ */
static ap_uint<RES_BITS> quantize_stage(const ap_uint<RES_BITS> &res)
{
#pragma HLS INLINE
    ap_uint<RES_BITS> q = 0;
    quantize_kernel<RES_BITS>(res, q);
    return q;
}
 
 
/* ################################################################
 *  寫出資料選擇 —— 96 -> 128 輸出對齊狀態機（INLINE）
 * ################################################################ */
 
/* ================================================================
 *  時序優化：st 改為 4-bit one-hot 環形移位（st[0]=S0 ... st[3]=S3）
 *    - 下一狀態 = 旋轉一位，無解碼、無比較
 *    - w 用 AND-OR 組出，hold 用 shift 選擇，皆為 1~2 層 LUT
 *    - S3 的 hold 為 don't care，故 hold 更新不需判斷 S3
 * ================================================================ */
static void pack_side(const ap_uint<96>          &res,
                      ap_uint<96>                &hold,
                      ap_uint<4>                 &st,
                      hls::stream<ap_uint<128> > &word_out)
{
#pragma HLS INLINE
    const bool s0 = st[0];
    const bool s1 = st[1];
    const bool s2 = st[2];
    const bool s3 = st[3];
 
    ap_uint<128> w1 = (res.range(31, 0), hold.range(95, 0));   /* 96 + 32 */
    ap_uint<128> w2 = (res.range(63, 0), hold.range(63, 0));   /* 64 + 64 */
    ap_uint<128> w3 = (res,              hold.range(31, 0));   /* 32 + 96 */
    ap_uint<128> w  = (s1 ? w1 : (ap_uint<128>)0)
                    | (s2 ? w2 : (ap_uint<128>)0)
                    | (s3 ? w3 : (ap_uint<128>)0);
 
    ap_uint<96> r32 = res >> 32;
    ap_uint<96> r64 = res >> 64;
    hold = s0 ? res : (s1 ? r32 : r64);
 
    ap_uint<4> nst;                                  /* 旋轉一位 */
    nst[0] = st[3]; nst[1] = st[0]; nst[2] = st[1]; nst[3] = st[2];
    st = nst;
 
    if (!s0)                                         /* 唯一寫出點 */
        word_out.write(w);
}
 
/* 收尾：輸出 pixel 數不是 16 的倍數時，補 0 成最後一個 word */
static void pack_tail(const ap_uint<96>          &hold,
                      ap_uint<4>                  st,
                      hls::stream<ap_uint<128> > &word_out)
{
#pragma HLS INLINE
    if (!st[0]) {
        ap_uint<7> L = st[1] ? 96 : (st[2] ? 64 : 32);
        ap_uint<96>  mask = (((ap_uint<97>)1) << L) - 1;
        ap_uint<128> tail = hold & mask;
        word_out.write(tail);
    }
}
 
 
/* ================================================================
 *  compute_side（DATAFLOW process）
 *    uyvy_convert -> resize_in_select -> resize_pe x N_DSP
 *                 -> resize_out_select -> quantize_stage -> pack_side -> word_ch
 *
 *  控制路徑時序優化（迴圈攜帶的狀態都改成「暫存器直出、下一值只靠暫存器 + 常數」）
 *
 *  原本：ox += 4 -> (ox >= out_w) 比較 -> 清零 / row++ -> (row == v_taps) 比較
 *        且 base_idx = ox>>2、last_row = (row == v_taps-1)、mul_b = last_row ? sr : 1
 *        都是從這條鏈後面長出來，直接進 BRAM 位址與 DSP 輸入。
 *
 *  現在：
 *   (a) 欄位計數改用 quad 計數 qx（每次 +1），直接當 line buffer 位址（不用 >>2）
 *   (b) 換列偵測用暫存器 col_last（= qx 為最後一欄）
 *       下一拍的 col_last 用「qx == quad_last-1」預先算好，
 *       比較對象是暫存器與迴圈不變量，與 qx+1 的加法並行，不在加法之後
 *   (c) 列相位用 one-hot row_oh；last_row、mul_b 各自是暫存器，
 *       只在換列時更新，DSP 的 B 輸入不再經過 mux
 *   (d) 輸入對齊狀態機維持原本的 2-bit st + switch（在 resize_in_select 內）
 * ================================================================ */
typedef ap_uint<8> qcol_t;            /* QUAD_W_MAX = 240 < 256 */
 
static void compute_side(const ap_uint<128>         *in,
                         hls::stream<ap_uint<128> > &rs_word_out,    /* resize 結果 */
                         hls::stream<ap_uint<128> > &q_word_out,     /* quantize 結果 */
                         ap_uint<32>                 in_beats,
                         ap_uint<12>                 out_w,
                         ap_uint<1>                  scale_mode)
{
    ap_uint<LBW> lbA[QUAD_W_MAX];   /* 欄 4k, 4k+1 */
    ap_uint<LBW> lbB[QUAD_W_MAX];   /* 欄 4k+2, 4k+3 */
#pragma HLS BIND_STORAGE variable=lbA type=RAM_S2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=lbB type=RAM_S2P impl=BRAM latency=2
 
    const bool s3 = (scale_mode == SCALE_3);
    const ap_uint<SRW> sr = s3 ? (ap_uint<SRW>)SR_S3 : (ap_uint<SRW>)SR_S2;
 
    /* ---- 迴圈不變量（啟動時算一次）---- */
    const qcol_t quad_w       = (out_w + 3) >> 2;
    const bool   single_col   = (quad_w <= 1);
    const qcol_t quad_last_m1 = (quad_w >= 2) ? (qcol_t)(quad_w - 2) : (qcol_t)0;
 
    /* ---- 輸入對齊狀態 ---- */
    ap_uint<192> prev = 0;
    ap_uint<2>   st   = 0;
 
    /* ---- 位置追蹤（全部是暫存器）---- */
    qcol_t       qx       = 0;                 /* 目前 quad 欄，同時是 line buffer 位址 */
    bool         col_last = single_col;        /* qx == quad_w - 1 */
    ap_uint<3>   row_oh   = 1;                 /* 列相位 one-hot */
    bool         last_row = false;             /* 本列是否輸出（= s3 ? row_oh[2] : row_oh[1]） */
    ap_uint<SRW> mul_b    = 1;                 /* last_row ? sr : 1 */
 
    /* ---- 輸出對齊狀態（pack_side，one-hot）---- */
    ap_uint<96> hold_rs = 0;               /* resize 路 */
    ap_uint<4>  pst_rs  = 1;
    ap_uint<96> hold_q  = 0;               /* quantize 路 */
    ap_uint<4>  pst_q   = 1;
 
    init_loop: for (int i = 0; i < quad_w; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=QUAD_W_MAX
        lbA[i] = 0; lbB[i] = 0;
    }
 
    beat_loop: for (ap_uint<32> i = 0; i < in_beats; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_BEATS
#pragma HLS DEPENDENCE variable=lbA inter false
#pragma HLS DEPENDENCE variable=lbB inter false
 
        /* ---- 每拍無條件讀一筆 AXI 並轉成 8 個 RGB pixel ---- */
        ap_uint<192> beat = uyvy_convert(in[i]);
 
        /* line buffer 位址 = qx 暫存器直出 */
        ap_uint<LBW> qA = lbA[qx];
        ap_uint<LBW> qB = lbB[qx];
 
        ap_uint<8>    u0[N_SLOT][3], u1[N_SLOT][3], u2[N_SLOT][3];
        ap_uint<ACCW> v[N_SLOT][3];
        ap_uint<OPW>  lbp[N_SLOT][3];
#pragma HLS ARRAY_PARTITION variable=u0  complete dim=0
#pragma HLS ARRAY_PARTITION variable=u1  complete dim=0
#pragma HLS ARRAY_PARTITION variable=u2  complete dim=0
#pragma HLS ARRAY_PARTITION variable=v   complete dim=0
#pragma HLS ARRAY_PARTITION variable=lbp complete dim=0
        ap_uint<2> nst;
        bool       do_op;
        resize_in_select(beat, prev, st, s3, qA, qB, nst, do_op, u0, u1, u2, v, lbp);
        st   = nst;
        prev = beat;
 
        if (do_op) {
            /* ---- 運算：mul_b 為暫存器，直接進 DSP ---- */
            ap_uint<PROD_W> ph[N_SLOT][3], pl[N_SLOT][3];
#pragma HLS ARRAY_PARTITION variable=ph complete dim=0
#pragma HLS ARRAY_PARTITION variable=pl complete dim=0
            dsp_loop: for (int k = 0; k < N_DSP; k++) {
#pragma HLS UNROLL
                const int s = k / 3;
                const int c = k % 3;
                resize_pe(u0[s][c], u1[s][c], u2[s][c], v[s][c], lbp[s][c],
                          s3, mul_b, ph[s][c], pl[s][c]);
            }
 
            ap_uint<RES_BITS> res;
            ap_uint<LBW>      wA, wB;
            resize_out_select(ph, pl, s3, last_row, res, wA, wB);
 
            if (last_row) {
                ap_uint<RES_BITS> qres = quantize_stage(res);
                pack_side(res,  hold_rs, pst_rs, rs_word_out);   /* resize 結果（量化前）*/
                pack_side(qres, hold_q,  pst_q,  q_word_out);    /* quantize 結果 */
            }
            lbA[qx] = wA;
            lbB[qx] = wB;
 
            /* ---- 位置更新：全部只看暫存器與迴圈不變量 ---- */
            if (col_last) {
                qx       = 0;
                col_last = single_col;
 
                /* 換列：one-hot 前進；last_row 的下一值直接由目前相位決定
                 * 3 倍：下一列是最後一列 <=> 目前在 row 1；2 倍：<=> 目前在 row 0 */
                const bool nlast = s3 ? (bool)row_oh[1] : (bool)row_oh[0];
                ap_uint<3> nr;
                nr[0] = last_row;
                nr[1] = row_oh[0];
                nr[2] = row_oh[1] && !last_row;
                row_oh   = nr;
                last_row = nlast;
                mul_b    = nlast ? sr : (ap_uint<SRW>)1;
            } else {
                col_last = (qx == quad_last_m1);   /* 與 qx + 1 並行 */
                qx       = qx + 1;
            }
        }
    }
 
    pack_tail(hold_rs, pst_rs, rs_word_out);
    pack_tail(hold_q,  pst_q,  q_word_out);
}
 
 
/* ################################################################
 *  第二段（DATAFLOW process）：AXI 寫出
 * ################################################################ */
 
static void axi_write_side(hls::stream<ap_uint<128> > &word_in,
                           ap_uint<128>               *out,
                           ap_uint<32>                 out_words)
{
    write_loop: for (ap_uint<32> i = 0; i < out_words; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_OUT_WORDS
        out[i] = word_in.read();
    }
}
 
 
/* ################################################################
 *  DATAFLOW 區域：compute_side -> (rs_ch, q_ch) -> 兩個 axi_write_side
 * ################################################################ */
 
static void uyvyResizeQuant_dataflow(ap_uint<128> *uyvy_axi_bus,
                                     ap_uint<128> *resize_axi_bus,
                                     ap_uint<128> *quant_axi_bus,
                                     ap_uint<32>   in_beats,
                                     ap_uint<32>   out_words,
                                     ap_uint<12>   out_w,
                                     ap_uint<1>    scale_mode)
{
#pragma HLS DATAFLOW
 
    hls::stream<ap_uint<128> > rs_ch;
#pragma HLS STREAM       variable=rs_ch   depth=WORD_FIFO_DEPTH
#pragma HLS BIND_STORAGE variable=rs_ch   type=fifo impl=srl
 
    hls::stream<ap_uint<128> > q_ch;
#pragma HLS STREAM       variable=q_ch    depth=WORD_FIFO_DEPTH
#pragma HLS BIND_STORAGE variable=q_ch    type=fifo impl=srl
 
    compute_side  (uyvy_axi_bus, rs_ch, q_ch, in_beats, out_w, scale_mode);
    axi_write_side(rs_ch, resize_axi_bus, out_words);
    axi_write_side(q_ch,  quant_axi_bus,  out_words);
}
 
 
/* ################################################################
 *  Top
 *    gmem0：輸入 UYVY
 *    gmem1：resize 結果（RGB，量化前）
 *    gmem2：quantize 結果（RGB）
 * ################################################################ */
 
void uyvyResizeQuant(ap_uint<128> *uyvy_axi_bus,
                     ap_uint<128> *resize_axi_bus,
                     ap_uint<128> *quant_axi_bus,
                     ap_uint<12>   img_w,
                     ap_uint<12>   img_h,
                     ap_uint<1>    scale_mode)
{
#pragma HLS INTERFACE m_axi port=uyvy_axi_bus   offset=slave bundle=uyvy_in \
                     depth=MAX_IN_BEATS  max_read_burst_length=128  num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=resize_axi_bus offset=slave bundle=resize_out \
                     depth=MAX_OUT_WORDS max_write_burst_length=32 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=quant_axi_bus  offset=slave bundle=quant_out \
                     depth=MAX_OUT_WORDS max_write_burst_length=32 num_write_outstanding=2
 
#pragma HLS INTERFACE s_axilite port=uyvy_axi_bus   bundle=control
#pragma HLS INTERFACE s_axilite port=resize_axi_bus bundle=control
#pragma HLS INTERFACE s_axilite port=quant_axi_bus  bundle=control
#pragma HLS INTERFACE s_axilite port=img_w          bundle=control
#pragma HLS INTERFACE s_axilite port=img_h          bundle=control
#pragma HLS INTERFACE s_axilite port=scale_mode     bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control
 
    ap_uint<32> in_beats, out_words;
    ap_uint<12> out_w;
    calc_params(img_w, img_h, scale_mode == SCALE_3, in_beats, out_words, out_w);
 
    uyvyResizeQuant_dataflow(uyvy_axi_bus, resize_axi_bus, quant_axi_bus,
                             in_beats, out_words, out_w, scale_mode);
}
 