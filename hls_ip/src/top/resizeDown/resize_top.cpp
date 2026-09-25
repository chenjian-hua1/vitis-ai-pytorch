/******************************************************************************
 * resize_top.cpp
 *
 * AXI -> resize -> AXI，三段 DATAFLOW（輸入 RGB888 packed，128-bit）
 *
 *   DDR --m_axi 128b--> compute_side   對齊 -> 6 x resize_pe -> line buffer
 *                            |  hls::stream<ap_uint<96>>   result_ch
 *                            v
 *                       pack_side      48/96 -> 128 對齊
 *                            |  hls::stream<ap_uint<128>>  word_ch
 *                            v
 *   DDR <--m_axi 128b-- axi_write_side
 *
 * 運算用 resize_impl.cpp 的 resize_pe，本檔負責資料選擇：
 *
 *   計算資料選擇
 *     resize_in_select    128-bit 對齊狀態機 + 6 個 PE 的輸入
 *   寫出資料選擇
 *     resize_out_select   lane -> 輸出欄、依小數位數取 8 bit、line buffer 寫回
 *     pack_side           48/96 -> 128 輸出對齊狀態機
 *
 * 與 uyvy_resize_top.cpp 的差別只在輸入寬度：這裡每拍 128 bit，
 * 一次運算需要 144 / 192 bit，所以對齊狀態機較長，
 * 迴圈以「輸入拍」為單位，湊滿才運算（do_op）。
 *
 * 注意：concat 一律指定給完整寬度的變數，
 * 不可直接寫進 .range()，否則會經過 64-bit 轉換被截斷。
 *****************************************************************************/

#include "resize_top.h"
#include "../../impl/resize_impl.h" /* resize_pe、位元寬常數 */
#include "ap_int.h"
#include "hls_stream.h"

/* ---------------------------------------------------------------- 常數 */

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

#define RES_FIFO_DEPTH  32
#define WORD_FIFO_DEPTH 32

#define MAX_IN_BEATS    (1920 * 1080 * 3 / 16)     /* co-sim 深度 */
#define MAX_OUT_WORDS   (960 * 540 * 3 / 16)       /* co-sim 深度，2 倍較大 */


/* ================================================================
 *  參數計算：只在 top 算一次，乘法綁 fabric，不佔 DSP
 *
 *  上一版在每個 DATAFLOW process 內各自算 img_w/3、img_h/3、out_w*out_h，
 *  「除以常數 3」會被 HLS 轉成乘法，每段各耗 2~3 顆 DSP，
 *  report 顯示 compute_side 8、pack_side 3、axi_write_side 3，共 14 顆。
 *
 *  現在集中在 top（DATAFLOW 區域外）算一次，再把計數傳進各段；
 *  這些乘法只在啟動時算一次，用 BIND_OP impl=fabric 放到 LUT。
 *
 *  只用一組純組合乘法器：
 *    所有乘法都呼叫 mul12（INLINE off），並用
 *      ALLOCATION function instances=mul12 limit=1
 *    限制整個 calc_params 只能有一個 mul12 實體。HLS 會把每次呼叫
 *    排在不同的狀態，前面加一組輸入多工器，結果存進暫存器，
 *    等於一個乘法器分時使用 4～5 拍。
 *    mul12 內 BIND_OP 不指定 latency -> 純組合乘法器，不佔 DSP。
 *    1366 以一般運算元傳入 mul12，不會被常數化成另一組移位加法器。
 *
 *    時序：每拍路徑 = 暫存器 -> 輸入多工器 -> 組合乘法 -> 暫存器，
 *    若 HLS 仍把後面的 x3 加法排進同一拍而違反 setup，
 *    可加大 set_clock_uncertainty，讓排程把加法推到下一拍。
 *
 *  除以 3：x / 3 = (x * 1366) >> 12
 *    x = 3k 時 3k * 1366 = 4098k，(4098k) >> 12 = k + (2k >> 12) = k（k <= 2047）
 *    3 倍模式已要求 img_w、img_h 為 3 的倍數，12-bit 範圍內精確
 * ================================================================ */
/* 共用的 12 x 12 純組合乘法器（calc_params 內限定只有一個實體） */
static ap_uint<24> mul12(ap_uint<12> a, ap_uint<12> b)
{
// #pragma HLS INLINE off
    ap_uint<24> p = a * b;
#pragma HLS BIND_OP variable=p op=mul impl=fabric latency=2
    return p;
}

static void calc_params(ap_uint<12>  img_w,
                        ap_uint<12>  img_h,
                        bool         s3,
                        ap_uint<32> &total_words,
                        ap_uint<32> &total_results,
                        ap_uint<32> &out_words,
                        ap_uint<12> &out_w)
{
// #pragma HLS INLINE offㄋ
#pragma HLS ALLOCATION function instances=mul12 limit=1
    const ap_uint<12> K_DIV3 = 1366;                     /* x/3 = (x*1366)>>12 */

    /* ---- 第 1、2 次：除以 3 ---- */
    ap_uint<24> w1366 = mul12(img_w, K_DIV3);
    ap_uint<24> h1366 = mul12(img_h, K_DIV3);
    out_w = s3 ? (ap_uint<12>)(w1366 >> 12) : (ap_uint<12>)(img_w >> 1);
    ap_uint<12> out_h = s3 ? (ap_uint<12>)(h1366 >> 12) : (ap_uint<12>)(img_h >> 1);

    /* ---- 第 3 次：輸出 pixel 數 ---- */
    ap_uint<24> out_pix = mul12(out_w, out_h);
    ap_uint<26> out_bytes = ((ap_uint<26>)out_pix << 1) + out_pix;     /* x3：移位加法 */
    total_results = s3 ? (ap_uint<32>)(out_pix >> 1) : (ap_uint<32>)(out_pix >> 2);
    out_words     = (out_bytes + 15) >> 4;                             /* ceil(bytes/16) */

    /* ---- 第 4 次：輸入 pixel 數 ---- */
    ap_uint<24> n_pix = mul12(img_w, img_h);
    ap_uint<26> in_bytes = ((ap_uint<26>)n_pix << 1) + n_pix;          /* x3：移位加法 */
    total_words = in_bytes >> 4;
}


/* ################################################################
 *
 *  第一段：resize
 *
 * ################################################################ */

/* ================================================================
 *  計算資料選擇
 *
 *  (1) 128-bit 輸入對齊狀態機
 *      每次運算完剩下的位元一定是當前 beat 的高位，
 *      所以只存上一拍 prev，暫存 = prev 的高 L 位，切片全為常數
 *
 *   3 倍（每次運算 144 bit）
 *   狀態  暫存 L  do_op  運算資料 win[143:0]              下一狀態
 *   S0      0      0     —（本拍整筆存入 prev）           S1
 *   S1    128      1     {beat[ 15:0], prev[127:  0]}    S2
 *   S2    112      1     {beat[ 31:0], prev[127: 16]}    S3
 *   S3     96      1     {beat[ 47:0], prev[127: 32]}    S4
 *   S4     80      1     {beat[ 63:0], prev[127: 48]}    S5
 *   S5     64      1     {beat[ 79:0], prev[127: 64]}    S6
 *   S6     48      1     {beat[ 95:0], prev[127: 80]}    S7
 *   S7     32      1     {beat[111:0], prev[127: 96]}    S8
 *   S8     16      1     {beat[127:0], prev[127:112]}    S0
 *   週期 9 拍、運算 8 次
 *
 *   2 倍（每次運算 192 bit）
 *   S0      0      0     —                               S1
 *   S1    128      1     {beat[ 63:0], prev[127:  0]}    S2
 *   S2     64      1     {beat[127:0], prev[127: 64]}    S0
 *   週期 3 拍、運算 2 次
 *
 *  (2) PE 輸入選擇（slot s x 通道 c）
 *      3 倍：u0,u1,u2 = p[3s..3s+2]，v = line_buf 該欄（bsel 選 bank，
 *            slot0 取高 lane、slot1 取低 lane）
 *      2 倍：u0,u1,u2 = p[4s..4s+2]，v = p[4s+3]，
 *            lbp = {line_buf[i], line_buf[i+1]}（slot0 -> bankA，slot1 -> bankB）
 * ================================================================ */
static void resize_in_select(const ap_uint<128> &beat,
                             const ap_uint<128> &prev,
                             ap_uint<4>          st,
                             bool                s3,
                             const ap_uint<LBW> &qA,
                             const ap_uint<LBW> &qB,
                             bool                bsel,
                             ap_uint<4>         &nst,
                             ap_uint<8>          u0[2][3],
                             ap_uint<8>          u1[2][3],
                             ap_uint<8>          u2[2][3],
                             ap_uint<ACCW>       v[2][3],
                             ap_uint<OPW>        lbp[2][3])
{
#pragma HLS INLINE

    /* ---- (1) 輸入對齊狀態機 ---- */
    ap_uint<192> win = 0;
    nst = 0;
    if (s3) {
        switch (st) {
        case 0:                                                     nst = 1; break; /* 暖機 */
        case 1: win = (beat.range( 15, 0), prev.range(127,   0)); nst = 2; break;
        case 2: win = (beat.range( 31, 0), prev.range(127,  16)); nst = 3; break;
        case 3: win = (beat.range( 47, 0), prev.range(127,  32)); nst = 4; break;
        case 4: win = (beat.range( 63, 0), prev.range(127,  48)); nst = 5; break;
        case 5: win = (beat.range( 79, 0), prev.range(127,  64)); nst = 6; break;
        case 6: win = (beat.range( 95, 0), prev.range(127,  80)); nst = 7; break;
        case 7: win = (beat.range(111, 0), prev.range(127,  96)); nst = 8; break;
        case 8: win = (beat.range(127, 0), prev.range(127, 112)); nst = 0; break;
        default:                                                    nst = 0; break;
        }
    } else {
        switch (st) {
        case 0:                                                     nst = 1; break; /* 暖機 */
        case 1: win = (beat.range( 63, 0), prev.range(127,  0));  nst = 2; break;
        case 2: win = (beat.range(127, 0), prev.range(127, 64));  nst = 0; break;
        default:                                                    nst = 0; break;
        }
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
 *  compute_side：每拍讀一筆 AXI，湊滿才運算
 *    resize_in_select -> resize_pe x N_DSP -> resize_out_select
 * ================================================================ */
static void compute_side(ap_uint<128>              *in_ptr,
                         hls::stream<ap_uint<96> > &result_out,
                         ap_uint<32>                total_words,
                         ap_uint<12>                out_w,
                         ap_uint<1>                 scale_mode)
{
    ap_uint<LBW> lbA[QUAD_W_MAX];   /* 欄 4k, 4k+1 */
    ap_uint<LBW> lbB[QUAD_W_MAX];   /* 欄 4k+2, 4k+3 */
#pragma HLS BIND_STORAGE variable=lbA type=RAM_S2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=lbB type=RAM_S2P impl=BRAM latency=2
// #pragma HLS BIND_STORAGE variable=lbA type=RAM_S2P impl=LUTRAM
// #pragma HLS BIND_STORAGE variable=lbB type=RAM_S2P impl=LUTRAM

    const bool s3 = (scale_mode == SCALE_3);

    const ap_uint<LOG2_CEIL(3)> v_taps = s3 ? 3 : 2;
    const ap_uint<3>            n_out  = s3 ? 2 : 4;
    const ap_uint<SRW>          sr     = s3 ? (ap_uint<SRW>)SR_S3 : (ap_uint<SRW>)SR_S2;
    const ap_uint<LOG2_CEIL((OUT_W_MAX+3)>>2)> quad_w = (out_w + 3) >> 2;

    /* ---- 輸入對齊狀態 ---- */
    ap_uint<128> prev = 0;
    ap_uint<4>   st   = 0;

    /* ---- 位置追蹤 ---- */
    ap_uint<LOG2_CEIL(OUT_W_MAX)> ox           = 0;
    ap_uint<LOG2_CEIL(3)>         row_in_block = 0;

    init_loop: for (int i = 0; i < quad_w; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=QUAD_W_MAX
        lbA[i] = 0; lbB[i] = 0;
    }

    main_loop: for (ap_uint<32> i = 0; i < total_words; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_BEATS
#pragma HLS DEPENDENCE variable=lbA inter false
#pragma HLS DEPENDENCE variable=lbB inter false

        /* ---- 無條件連續讀取，burst inference 條件最佳 ---- */
        ap_uint<128> beat = in_ptr[i];

        ap_uint<10> base_idx = ox >> 2;
        bool        bsel     = ox[1];     /* 3 倍：0 -> bankA，1 -> bankB */
        bool        do_op    = (st != 0);
        bool        last_row = (row_in_block == v_taps - 1);

        /* line buffer 每拍都讀（讀取無副作用），寫入才受 do_op 控制 */
        ap_uint<LBW> qA = lbA[base_idx];
        ap_uint<LBW> qB = lbB[base_idx];

        /* ---- 計算資料選擇 ---- */
        ap_uint<4>    nst;
        ap_uint<8>    u0[2][3], u1[2][3], u2[2][3];
        ap_uint<ACCW> v[2][3];
        ap_uint<OPW>  lbp[2][3];
#pragma HLS ARRAY_PARTITION variable=u0  complete dim=0
#pragma HLS ARRAY_PARTITION variable=u1  complete dim=0
#pragma HLS ARRAY_PARTITION variable=u2  complete dim=0
#pragma HLS ARRAY_PARTITION variable=v   complete dim=0
#pragma HLS ARRAY_PARTITION variable=lbp complete dim=0
        resize_in_select(beat, prev, st, s3, qA, qB, bsel, nst, u0, u1, u2, v, lbp);
        st   = nst;
        prev = beat;

        if (do_op) {
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
}


/* ################################################################
 *
 *  第二段：寫出資料選擇 —— 48/96 -> 128 輸出對齊狀態機
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
 *  寫法與 uyvy2rgb 的 repack_write 相同：
 *    先判斷「湊滿要寫出」的狀態，其餘狀態只把 r 存進 hold 的固定位置；
 *    w 用 .range() 逐段指定，hold / st 直接更新；word_out 只有一個寫出點
 *
 * ################################################################ */

static void pack_side(hls::stream<ap_uint<96> >  &result_in,
                      hls::stream<ap_uint<128> > &word_out,
                      ap_uint<32>                 total_results,
                      ap_uint<1>                  scale_mode)
{
    const bool s3 = (scale_mode == SCALE_3);

    ap_uint<112> hold = 0;
    ap_uint<3>   st   = 0;

    pack_loop: for (ap_uint<32> r = 0; r < total_results; r++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=129600

        ap_uint<96>  res = result_in.read();
        ap_uint<128> w;
        bool         emit;

        if (s3) {
            /* ---- 3 倍：每筆 48 bit ---- */
            ap_uint<48> r3 = res.range(47, 0);

            if (st == 2 || st == 5 || st == 7) {     /* 湊滿 128，寫出 */
                emit = true;
                if (st == 2) {                       /* 暫存 96 + 32 */
                    w.range( 95,  0) = hold.range(95, 0);
                    w.range(127, 96) = r3.range(31, 0);
                    hold.range(15, 0) = r3.range(47, 32);    /* 留 16 */
                    st = 3;
                } else if (st == 5) {                /* 暫存 112 + 16 */
                    w.range(111,   0) = hold.range(111, 0);
                    w.range(127, 112) = r3.range(15, 0);
                    hold.range(31, 0) = r3.range(47, 16);    /* 留 32 */
                    st = 6;
                } else {                             /* 暫存 80 + 48 */
                    w.range( 79,  0) = hold.range(79, 0);
                    w.range(127, 80) = r3;                   /* 留 0 */
                    st = 0;
                }
            } else {                                 /* 湊不滿，存進暫存 */
                emit = false;
                if      (st == 0) hold.range( 47,  0) = r3;  /* 0  -> 48  */
                else if (st == 1) hold.range( 95, 48) = r3;  /* 48 -> 96  */
                else if (st == 3) hold.range( 63, 16) = r3;  /* 16 -> 64  */
                else if (st == 4) hold.range(111, 64) = r3;  /* 64 -> 112 */
                else              hold.range( 79, 32) = r3;  /* 32 -> 80  */
                st = st + 1;
            }
        } else {
            /* ---- 2 倍：每筆 96 bit ---- */
            if (st == 0) {                           /* 暫存 0，湊不滿 */
                emit = false;
                hold.range(95, 0) = res;             /* 留 96 */
                st = 1;
            } else {                                 /* 湊滿 128，寫出 */
                emit = true;
                if (st == 1) {                       /* 暫存 96 + 32 */
                    w.range( 95,  0) = hold.range(95, 0);
                    w.range(127, 96) = res.range(31, 0);
                    hold.range(63, 0) = res.range(95, 32);   /* 留 64 */
                    st = 2;
                } else if (st == 2) {                /* 暫存 64 + 64 */
                    w.range( 63,  0) = hold.range(63, 0);
                    w.range(127, 64) = res.range(63, 0);
                    hold.range(31, 0) = res.range(95, 64);   /* 留 32 */
                    st = 3;
                } else {                             /* 暫存 32 + 96 */
                    w.range( 31,  0) = hold.range(31, 0);
                    w.range(127, 32) = res;                  /* 留 0 */
                    st = 0;
                }
            }
        }

        if (emit)                                    /* 唯一寫出點 */
            word_out.write(w);
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
 *  第三段：AXI 寫出（位址即迴圈變數、無條件包裹 -> burst）
 *
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
 *
 *  DATAFLOW 區域：只放 process 呼叫，計數由 top 算好傳入
 *
 * ################################################################ */

static void resize_dataflow(ap_uint<128> *in_ptr,
                            ap_uint<128> *out_ptr,
                            ap_uint<32>   total_words,
                            ap_uint<32>   total_results,
                            ap_uint<32>   out_words,
                            ap_uint<12>   out_w,
                            ap_uint<1>    scale_mode)
{
#pragma HLS DATAFLOW

    hls::stream<ap_uint<96>  > result_ch;
    hls::stream<ap_uint<128> > word_ch;
#pragma HLS STREAM       variable=result_ch depth=RES_FIFO_DEPTH
#pragma HLS STREAM       variable=word_ch   depth=WORD_FIFO_DEPTH
#pragma HLS BIND_STORAGE variable=result_ch type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=word_ch   type=fifo impl=srl

    compute_side  (in_ptr, result_ch, total_words, out_w, scale_mode);
    pack_side     (result_ch, word_ch, total_results, scale_mode);
    axi_write_side(word_ch, out_ptr, out_words);
}


/* ################################################################
 *
 *  Top：介面 + 參數計算（一次）+ DATAFLOW
 *
 * ################################################################ */

void resize_kernel(ap_uint<128> *in_ptr,
                   ap_uint<128> *out_ptr,
                   ap_uint<12>   img_w,
                   ap_uint<12>   img_h,
                   ap_uint<1>    scale_mode)
{
#pragma HLS INTERFACE m_axi port=in_ptr  offset=slave bundle=gmem0 \
                     depth=MAX_IN_BEATS  max_read_burst_length=64  num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=out_ptr offset=slave bundle=gmem1 \
                     depth=MAX_OUT_WORDS max_write_burst_length=64 num_write_outstanding=8

#pragma HLS INTERFACE s_axilite port=in_ptr     bundle=control
#pragma HLS INTERFACE s_axilite port=out_ptr    bundle=control
#pragma HLS INTERFACE s_axilite port=img_w      bundle=control
#pragma HLS INTERFACE s_axilite port=img_h      bundle=control
#pragma HLS INTERFACE s_axilite port=scale_mode bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    ap_uint<32> total_words, total_results, out_words;
    ap_uint<12> out_w;
    calc_params(img_w, img_h, scale_mode == SCALE_3,
                total_words, total_results, out_words, out_w);

    resize_dataflow(in_ptr, out_ptr, total_words, total_results, out_words,
                    out_w, scale_mode);
}