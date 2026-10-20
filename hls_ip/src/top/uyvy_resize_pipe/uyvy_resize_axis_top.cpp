/******************************************************************************
 * uyvy_resize_axis_top.cpp   （AXI-Stream 版，時間多工）
 *
 * AXIS -> uyvy2rgb -> resize -> AXIS，單一 compute 函式（不使用 DATAFLOW）
 *
 *   in_stream (AXIS 128b) --> uyvy2rgb -> 對齊 -> N_PE x resize_pe -> pack
 *                                                        |
 *   out_stream (AXIS 128b) <-----------------------------+
 *
 * ============================================================
 *  設計重點
 * ============================================================
 *   1. 介面：in_stream / out_stream 為 128-bit AXIS，
 *      img_w / img_h / scale_mode 由 s_axilite 設定。
 *
 *   2. 時間多工：LOOP_II 決定「一筆輸入用幾個 clk 完成運算」，
 *      beat_loop 的 II = LOOP_II，
 *        resize_pe  實體數 = N_DSP / LOOP_II
 *        cvt_pair   實體數 = ceil(4 / LOOP_II)
 *      共用由 ALLOCATION function 強制，對象是不展開的 wrapper
 *      （cvt_pair_shared / resize_pe_shared）。
 *      注意：IP 吞吐量 = 時脈 / LOOP_II 筆，要維持資料速率需讓 IP 時脈
 *      為輸入 AXIS 時脈的 LOOP_II 倍（用 AXIS Clock Converter 隔開）。
 *
 *   3. 輸入計數器：x（9 bit）/ y（12 bit）拆段，結束條件只剩暫存器旗標。
 *
 *   4. 輸出位置追蹤：qx / col_last / row_cnt / last_row / mul_b_r 皆為暫存器旗標。
 *
 *   5. TLAST：輸出最後一個 word 帶 TLAST（wcnt 與 out_words 比較）；
 *      輸入不依賴 TLAST，以 img_w/8 x img_h 計數，上游必須剛好送滿。
 *
 *   6. TKEEP：全 1（最後一個 word 尾端補 0 也算有效）。
 *
 * 前提：img_w 為 16 的倍數（3 倍時為 48 的倍數），n_col = img_w/8 >= 2，
 *       img_h >= 2，out_w 為 4 的倍數，quad_w >= 2。
 *
 * 注意：concat 一律指定給完整寬度的變數，
 * 不可直接寫進 .range()，否則會經過 64-bit 轉換被截斷。
 *****************************************************************************/

#include "uyvy_resize_axis_top.h"
#include "../../impl/base_func.h"
#include "../../impl/uyvy2rgb_impl.h"
#include "../../impl/resize_impl.h"        /* resize_pe、位元寬常數 */
#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

/* ---------------------------------------------------------------- 型別
 * ap_axiu<128,0,0,0>：data 128 bit，無 user/id/dest，含 keep/strb/last */
typedef ap_axiu<128, 0, 0, 0> axis_t;
typedef hls::stream<axis_t>   axis_stream_t;

/* ---------------------------------------------------------------- 常數 */

#define UYVY_GRP   4        /* 一拍 4 組 UYVY */
#define UYVY_PIX   8        /* 一拍 8 個 pixel */

#define OUT_W_MAX  960      /* 輸出寬度上限 */
#define QUAD_W_MAX 240      /* OUT_W_MAX / 4，每 bank 的深度 */

#define SR_S3      7282     /* 3 倍 scale_rate：65536/9，16 bit */
#define SR_S2      1        /* 2 倍 scale_rate：4/4，2 bit */

#define N_COL         4
#define GRP_S3        (N_COL * 3)
#define GRP_S2        (N_COL * 3)
#define LANE_S3       1
#define LANE_S2       2
#define DSP_NEED(g,l) (((g) + (l) - 1) / (l))
#define DSP_MAX(a,b)  ((a) > (b) ? (a) : (b))
#define N_DSP         DSP_MAX(DSP_NEED(GRP_S3, LANE_S3), DSP_NEED(GRP_S2, LANE_S2))   /* = 12 */
#define N_SLOT        (N_DSP / 3)                                                    /* = 4  */

#define MAX_IN_BEATS    (1920 * 1080 / 8)          /* tripcount 上限 */


/* ================================================================
 *  計算資源控制
 *
 *  LOOP_II：一筆輸入用幾個 clk 完成運算（= beat_loop 的 II）
 *      1  : 12 個 PE，4 個 cvt_pair
 *      2  :  6 個 PE，2 個 cvt_pair
 *      3  :  4 個 PE，2 個 cvt_pair
 *      4  :  3 個 PE，1 個 cvt_pair
 *      6  :  2 個 PE，1 個 cvt_pair
 *     12  :  1 個 PE，1 個 cvt_pair
 *
 *  可在編譯選項以 -DLOOP_II=3 覆寫。LOOP_II 必須整除 N_DSP。
 *  這是排程目標，實際 II 與 DSP 數請看綜合報告。
 * ================================================================ */
#ifndef LOOP_II
#define LOOP_II 2
#endif

#define RESIZE_LOOP_II        LOOP_II
#define RESIZE_UNROLL_FACTOR  (N_DSP / RESIZE_LOOP_II)                         /* resize_pe 實體數 */
#define CVT_UNROLL_FACTOR     ((UYVY_GRP + LOOP_II - 1) / LOOP_II)             /* cvt_pair 實體數 */

#if (LOOP_II < 1)
#error "LOOP_II must be >= 1"
#endif
#if (N_DSP % LOOP_II) != 0
#error "N_DSP must be divisible by LOOP_II"
#endif


/* ================================================================
 *  參數計算：只在 top 算一次
 * ================================================================ */
static ap_uint<24> mul12(ap_uint<12> a, ap_uint<12> b)
{
    ap_uint<24> p = a * b;
// #pragma HLS BIND_OP variable=p op=mul impl=fabric latency=3
#pragma HLS BIND_OP variable=p op=mul impl=dsp latency=3
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
    ap_uint<26> out_bytes = ((ap_uint<26>)out_pix << 1) + out_pix;     /* x3：移位加法 */
    out_words     = (out_bytes + 15) >> 4;                             /* ceil(bytes/16) */

    ap_uint<24> beats = mul12((ap_uint<12>)(img_w >> 3), img_h);
    in_beats = beats;                                                  /* compute 不使用，僅保留介面 */
}


/* ################################################################
 *  共用單元（wrapper）
 *
 *  cvt_pair / resize_pe 若被 HLS 展開進呼叫端，ALLOCATION function 會因
 *  找不到呼叫而被忽略（HLS 214-209）。所以用 INLINE off 的 wrapper 當共用對象，
 *  wrapper 內部的原函式照常展開。PIPELINE II=1：共用實體每 clk 可收新輸入。
 * ################################################################ */

static void cvt_pair_shared(ap_uint<8>   y0,
                            ap_uint<8>   y1,
                            ap_uint<8>   u,
                            ap_uint<8>   v,
                            ap_uint<24> &px0,
                            ap_uint<24> &px1)
{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    cvt_pair(y0, y1, u, v, px0, px1);
}

static void resize_pe_shared(ap_uint<8>        a0,
                             ap_uint<8>        a1,
                             ap_uint<8>        a2,
                             ap_uint<ACCW>     vv,
                             ap_uint<OPW>      lb,
                             bool              s3,
                             ap_uint<SRW>      mul_b,
                             ap_uint<PROD_W>  &ph,
                             ap_uint<PROD_W>  &pl)
{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    resize_pe(a0, a1, a2, vv, lb, s3, mul_b, ph, pl);
}


/* ################################################################
 *  UYVY -> RGB（INLINE）
 *  純接線的迴圈一律完全展開；共用只發生在 cvt_pair_shared
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
// #pragma HLS INLINE
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
        // cvt_pair_shared(y0[g], y1[g], u[g], v[g], px[2 * g], px[2 * g + 1]);
        cvt_pair(y0[g], y1[g], u[g], v[g], px[2 * g], px[2 * g + 1]);
    }

    return uyvy_out_select(px);
}


/* ################################################################
 *  resize 選擇邏輯（INLINE）
 * ################################################################ */

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

    /* ---- (A) 選擇訊號：st / s3 各自解碼，彼此平行 ---- */
    const bool sel_a = s3  && (st == 1);        /* 3 倍 S1 */
    const bool sel_b = s3  && (st == 2);        /* 3 倍 S2 */
    const bool sel_x = !s3;                     /* 2 倍 */

    /* ---- (B) 狀態轉移與 do_op：與資料選擇無關 ---- */
    do_op = sel_a || sel_b || sel_x;
    if (!s3)           nst = 0;
    else if (st == 0)  nst = 1;
    else if (st == 1)  nst = 2;
    else               nst = 0;                 /* st == 2 或非法值 */

    /* ---- (C) win：三個來源 AND-OR，沒選中（暖機）為 0 ---- */
    const ap_uint<288> win_a = (beat.range( 95, 0), prev.range(191,  0));
    const ap_uint<288> win_b = (beat.range(191, 0), prev.range(191, 96));
    const ap_uint<288> win_x = (ap_uint<288>)beat;

    ap_uint<288> win = 0;
    if (sel_a) win |= win_a;
    if (sel_b) win |= win_b;
    if (sel_x) win |= win_x;

    /* ---- (D) PE 輸入選擇 ---- */
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
                              ap_uint<96>          &res,
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
 *  輸出 AXIS word 組裝
 * ################################################################ */

static axis_t make_axis_word(const ap_uint<128> &data, bool last)
{
// #pragma HLS INLINE
    axis_t w;
    w.data = data;
    w.keep = -1;            /* 16 byte 全有效；若要精確 keep，可在 last 時依剩餘 byte 數設定 */
    w.strb = -1;
    w.last = last ? 1 : 0;
    return w;
}


/* ################################################################
 *  96 -> 128 輸出對齊狀態機（INLINE）
 *    wcnt 記錄已寫出的 word 數，用來產生 TLAST
 * ################################################################ */

static void pack_side(const ap_uint<96>  &res,
                      ap_uint<96>        &hold,
                      ap_uint<2>         &st,
                      ap_uint<32>        &wcnt,
                      ap_uint<32>         out_words,
                      axis_stream_t      &out_stream)
{
// #pragma HLS INLINE
    ap_uint<128> w    = 0;
    bool         emit = false;

    switch (st) {
    case 0:                                  /* 0 -> 96 */
        hold.range(95, 0) = res;
        st = 1;
        break;
    case 1:                                  /* 96 + 32，寫出，留 64 */
        w.range( 95,  0) = hold.range(95, 0);
        w.range(127, 96) = res.range(31, 0);
        hold.range(63, 0) = res.range(95, 32);
        emit = true;
        st = 2;
        break;
    case 2:                                  /* 64 + 64，寫出，留 32 */
        w.range( 63,  0) = hold.range(63, 0);
        w.range(127, 64) = res.range(63, 0);
        hold.range(31, 0) = res.range(95, 64);
        emit = true;
        st = 3;
        break;
    case 3:                                  /* 32 + 96，寫出，留 0 */
        w.range( 31,  0) = hold.range(31, 0);
        w.range(127, 32) = res;
        emit = true;
        st = 0;
        break;
    default:
        st = 0;
        break;
    }

    if (emit) {                              /* 唯一寫出點 */
        bool is_last = (wcnt == out_words - 1);
        out_stream.write(make_axis_word(w, is_last));
        wcnt++;
    }
}

/* 收尾：輸出 byte 數不是 16 的倍數時，補 0 成最後一個 word（一定是 TLAST） */
static void pack_tail(const ap_uint<96> &hold,
                      ap_uint<2>         st,
                      axis_stream_t     &out_stream)
{
// #pragma HLS INLINE
    if (st != 0) {
        ap_uint<7> L = 0;
        switch (st) {
        case 1: L = 96; break;
        case 2: L = 64; break;
        case 3: L = 32; break;
        default: break;
        }
        ap_uint<96>  mask = (((ap_uint<97>)1) << L) - 1;
        ap_uint<128> tail = hold & mask;
        out_stream.write(make_axis_word(tail, true));
    }
}


/* ================================================================
 *  compute：單層 pipeline 迴圈 + 拆分輸入計數器
 *
 *   - 結束條件：done = x_last && y_last（暫存器旗標）
 *   - 輸出位置：qx / col_last / row_cnt / last_row / mul_b_r
 *   - 時間多工：II = LOOP_II，cvt_pair_shared / resize_pe_shared 由 ALLOCATION 限制實體數
 * ================================================================ */
static void compute(axis_stream_t &in_stream,
                    axis_stream_t &out_stream,
                    ap_uint<12>    img_w,
                    ap_uint<12>    img_h,
                    ap_uint<32>    out_words,
                    ap_uint<12>    out_w,
                    ap_uint<1>     scale_mode)
{

    ap_uint<LBW> lbA[QUAD_W_MAX];
    ap_uint<LBW> lbB[QUAD_W_MAX];
#pragma HLS BIND_STORAGE variable=lbA type=RAM_S2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=lbB type=RAM_S2P impl=BRAM latency=2

    const bool s3 = (scale_mode == SCALE_3);

    const ap_uint<LOG2_CEIL(3)> v_taps = s3 ? 3 : 2;
    const ap_uint<SRW>          sr     = s3 ? (ap_uint<SRW>)SR_S3 : (ap_uint<SRW>)SR_S2;
    const ap_uint<LOG2_CEIL((OUT_W_MAX+3)>>2)> quad_w = (out_w + 3) >> 2;

    /* ---- 輸入計數器（拆成 x / y 兩段）---- */
    const ap_uint<9>  n_col    = (ap_uint<9>)(img_w >> 3);
    const ap_uint<9>  col_m2   = n_col - 2;        /* x == n_col-2 -> 下一拍是列尾 */
    const ap_uint<12> h_m2     = img_h - 2;        /* y == img_h-2 -> 換列後是最後一列 */

    ap_uint<9>  x      = 0;
    ap_uint<12> y      = 0;
    bool        x_last = false;                    /* x == n_col-1（n_col >= 2） */
    bool        y_last = false;                    /* y == img_h-1（img_h >= 2） */
    bool        done   = false;

    /* ---- 輸入對齊狀態 ---- */
    ap_uint<192> prev = 0;
    ap_uint<2>   st   = 0;

    /* ---- 輸出位置追蹤 ---- */
    const ap_uint<8> quad_w_m2 = quad_w - 2;
    const ap_uint<2> row_m2    = v_taps - 2;

    ap_uint<8>   qx       = 0;
    bool         col_last = false;
    ap_uint<2>   row_cnt  = 0;
    bool         last_row = false;
    ap_uint<SRW> mul_b_r  = 1;

    /* ---- 輸出對齊狀態 ---- */
    ap_uint<96> hold = 0;
    ap_uint<2>  pst  = 0;
    ap_uint<32> wcnt = 0;

    init_loop: for (int i = 0; i < quad_w; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=QUAD_W_MAX
        lbA[i] = 0; lbB[i] = 0;
    }

    beat_loop: while (!done) {
#pragma HLS PIPELINE II=RESIZE_LOOP_II
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_BEATS
#pragma HLS DEPENDENCE variable=lbA inter false
#pragma HLS DEPENDENCE variable=lbB inter false

        /* 這一拍是否為最後一拍：只用暫存器旗標 */
        done = x_last && y_last;

        /* ---- 輸入計數器更新（每拍無條件）---- */
        if (x_last) {                              /* 換列 */
            x      = 0;
            x_last = false;
            y_last = (y == h_m2);
            y++;
        } else {
            x_last = (x == col_m2);
            x++;
        }

        /* ---- 每拍無條件從 AXIS 讀一筆並轉成 8 個 RGB pixel ---- */
        axis_t       in_word = in_stream.read();
        ap_uint<192> beat    = uyvy_convert(in_word.data);

        const ap_uint<8> base_idx = qx;

        ap_uint<LBW> qA = lbA[base_idx];
        ap_uint<LBW> qB = lbB[base_idx];

        ap_uint<2>    nst;
        bool          do_op;
        ap_uint<8>    u0[N_SLOT][3], u1[N_SLOT][3], u2[N_SLOT][3];
        ap_uint<ACCW> v[N_SLOT][3];
        ap_uint<OPW>  lbp[N_SLOT][3];
#pragma HLS ARRAY_PARTITION variable=u0  complete dim=0
#pragma HLS ARRAY_PARTITION variable=u1  complete dim=0
#pragma HLS ARRAY_PARTITION variable=u2  complete dim=0
#pragma HLS ARRAY_PARTITION variable=v   complete dim=0
#pragma HLS ARRAY_PARTITION variable=lbp complete dim=0
        resize_in_select(beat, prev, st, s3, qA, qB, nst, do_op, u0, u1, u2, v, lbp);
        st   = nst;
        prev = beat;

        if (do_op) {
            const ap_uint<SRW> mul_b = mul_b_r;
            ap_uint<PROD_W> ph[N_SLOT][3], pl[N_SLOT][3];
#pragma HLS ARRAY_PARTITION variable=ph complete dim=0
#pragma HLS ARRAY_PARTITION variable=pl complete dim=0

            /* RESIZE_LOOP_II 個 phase，每個 phase RESIZE_UNROLL_FACTOR 顆 PE。
             * 全部展開，共用由 ALLOCATION 與 beat_loop 的 II 決定
             * （外層已 PIPELINE，內層不需要再加 PIPELINE）
             * 第 k 組 -> slot s = k/3，通道 c = k%3 */
            dsp_loop: for (int iter = 0; iter < RESIZE_LOOP_II; iter++) {
// #pragma HLS UNROLL
            // #pragma HLS PIPELINE II=1
                dsp_parallel: for (int lane = 0; lane < RESIZE_UNROLL_FACTOR; lane++) {
#pragma HLS UNROLL
                    const int k = iter * RESIZE_UNROLL_FACTOR + lane;
                    const int s = k / 3;
                    const int c = k % 3;
                    // resize_pe(u0[s][c], u1[s][c], u2[s][c],
                    //         v[s][c], lbp[s][c],
                    //         s3, mul_b, ph[s][c], pl[s][c]);
                    resize_pe_shared(u0[s][c], u1[s][c], u2[s][c],
                                     v[s][c], lbp[s][c],
                                     s3, mul_b, ph[s][c], pl[s][c]);
                }
            }

            ap_uint<96>  res;
            ap_uint<LBW> wA, wB;
            resize_out_select(ph, pl, s3, last_row, res, wA, wB);

            if (last_row)
                pack_side(res, hold, pst, wcnt, out_words, out_stream);
            lbA[base_idx] = wA;
            lbB[base_idx] = wB;

            /* ---- 輸出位置更新 ---- */
            if (col_last) {
                qx       = 0;
                col_last = false;
                if (last_row) {
                    row_cnt  = 0;
                    last_row = false;
                    mul_b_r  = 1;
                } else {
                    const bool nl = (row_cnt == row_m2);
                    row_cnt++;
                    last_row = nl;
                    mul_b_r  = nl ? sr : (ap_uint<SRW>)1;
                }
            } else {
                col_last = (qx == quad_w_m2);
                qx++;
            }
        }
    }

    pack_tail(hold, pst, out_stream);
}


/* ################################################################
 *  Top：AXIS 介面 + 參數計算（一次）+ compute
 * ################################################################ */

void uyvy_resize(axis_stream_t &in_stream,
                 axis_stream_t &out_stream,
                 ap_uint<12>    img_w,
                 ap_uint<12>    img_h,
                 ap_uint<1>     scale_mode)
{
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream

#pragma HLS INTERFACE s_axilite port=img_w      bundle=control
#pragma HLS INTERFACE s_axilite port=img_h      bundle=control
#pragma HLS INTERFACE s_axilite port=scale_mode bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    ap_uint<32> in_beats, out_words;               /* in_beats 不給 compute 使用，HLS 會移除其邏輯 */
    ap_uint<12> out_w;
    calc_params(img_w, img_h, scale_mode == SCALE_3, in_beats, out_words, out_w);

    compute(in_stream, out_stream, img_w, img_h, out_words, out_w, scale_mode);
}