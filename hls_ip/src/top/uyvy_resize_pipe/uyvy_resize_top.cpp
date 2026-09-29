/******************************************************************************
 * uyvy_resize_top.cpp
 *
 * AXI -> uyvy2rgb -> resize -> AXI，兩段 DATAFLOW
 *
 *   DDR --m_axi 128b--> compute_side   uyvy2rgb -> 對齊 -> 12 x resize_pe -> pack
 *                            |  hls::stream<ap_uint<128>>  word_ch（唯一的 FIFO）
 *                            v
 *   DDR <--m_axi 128b-- axi_write_side
 *
 * ============================================================
 *  FIFO 的取捨：只在「速率不匹配」或「AXI 前生產不規則」時才需要
 * ============================================================
 *
 *   原本三個 FIFO：
 *     rgb_ch     read_convert -> compute_side
 *                  生產：每拍固定 1 筆；消化：每拍固定讀 1 筆
 *                  -> 速率相同且都無條件，不需要緩衝，合併成同一個迴圈
 *     result_ch  compute_side -> pack_side
 *                  生產：只在 last_row，每拍最多 1 筆；消化：每拍可收 1 筆
 *                  -> 生產 <= 消化，後面也不是 AXI，不需要緩衝，pack 併入迴圈
 *     word_ch    pack -> axi_write_side
 *                  生產：湊滿 128 bit 才出，不規則；消化：AXI 寫出
 *                  -> 保留。若在同一迴圈內有條件地寫 out_ptr，
 *                     gmem1 的 burst 推斷會失敗（每個 word 變獨立交易）；
 *                     拆開後 axi_write_side 位址即迴圈變數、無條件，burst 必成
 *
 *   in_ptr 的讀取在合併後的迴圈裡仍是每拍無條件的 in_ptr[i]，讀端 burst 不受影響。
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
 *     resize_in_select    192-bit 對齊狀態機 + 12 個 PE 的輸入
 *
 *   寫出資料選擇（運算結果怎麼排、往哪寫）
 *     uyvy_out_select     8 個 {R,G,B} -> 記憶體 byte 順序 -> 192-bit
 *     resize_out_select   lane -> 輸出欄、依小數位數取 8 bit、line buffer 寫回
 *     pack_side           96 -> 128 輸出對齊狀態機（INLINE，每筆 result 呼叫一次）
 *
 *   除了 compute_side / axi_write_side 兩個 DATAFLOW process，其餘全部 INLINE
 *
 * ============================================================
 *  3 倍模式一次算 4 個 block（12 pixel）
 * ============================================================
 *
 * 上一版 3 倍一次只算 2 個 block（6 pixel），但 read_convert 每拍送 8 pixel，
 * resize 跟不上，整條管線被拖到 6 pixel/拍（1080p 需 345600 拍）。
 *
 * 為什麼是 4 個 block 而不是 3 個：
 *   3 個 block = 9 pixel/次，速率夠（9 >= 8），但 1920 / 3 = 640 個輸出欄，
 *   640 不是 3 的倍數，每列最後一次運算會橫跨兩列輸入（不同 row_in_block、
 *   不同 line buffer 欄），必須另外處理列尾。
 *   4 個 block = 12 pixel/次，640 / 4 = 160 整除；而且與 2 倍模式一樣
 *   每次產出 4 個輸出欄，line buffer 與輸出格式兩種模式完全相同。
 *
 * 兩種模式每次運算都產出 4 個輸出欄：
 *   3 倍：12 pixel（288 bit），4 欄 x RGB = 12 組，每顆 DSP 1 組 -> 12 顆
 *   2 倍： 8 pixel（192 bit），4 欄 x RGB = 12 組，每顆 DSP 2 組 ->  6 顆
 *   N_DSP = max(12, 6) = 12（2 倍時後 6 顆閒置）
 *
 * 速率（每拍讀一筆 AXI、轉出 192 bit，迴圈以「輸入拍」為單位）：
 *   2 倍：每拍 1 次運算
 *   3 倍：每 3 拍 2 次運算（對齊狀態機 3 個狀態）
 *   兩種模式都跟上 AXI 輸入的 8 pixel/拍
 *   1920x1080 @250MHz：兩種模式都是 259200 拍，約 1.04 ms
 *
 * 其他：
 *   - 中間不經過 DDR；所有計數由 top 算一次（共用一個 fabric 乘法器）
 *   - byte 順序：uyvy_out_select 產生 byte0 = R，resize 輸出維持相同順序
 *   - 尺寸限制與上一版相同（見 uyvy_resize.h）：
 *       3 倍 img_w 為 48 的倍數 -> out_w 為 16 的倍數，自然是 4 的倍數
 *
 * 注意：concat 一律指定給完整寬度的變數，
 * 不可直接寫進 .range()，否則會經過 64-bit 轉換被截斷。
 *****************************************************************************/

#include "uyvy_resize_top.h"
#include "../../impl/base_func.h"
#include "../../impl/uyvy2rgb_impl.h"
#include "../../impl/resize_impl.h"        /* resize_pe、位元寬常數 */
#include "ap_int.h"
#include "hls_stream.h"

/* ---------------------------------------------------------------- 常數 */

#define UYVY_GRP   4        /* 一拍 4 組 UYVY */
#define UYVY_PIX   8        /* 一拍 8 個 pixel */

#define OUT_W_MAX  960      /* 輸出寬度上限 */
#define QUAD_W_MAX 240      /* OUT_W_MAX / 4，每 bank 的深度 */

#define SR_S3      7282     /* 3 倍 scale_rate：65536/9，16 bit */
#define SR_S2      1        /* 2 倍 scale_rate：4/4，2 bit */

/* ---- 每次運算的輸出欄數與 DSP 顆數 ----
 * 組 = 一個輸出欄的一個通道
 *   3 倍：4 欄 x RGB = 12 組，B 16 bit 無法打包 -> 每顆 1 組
 *   2 倍：4 欄 x RGB = 12 組，B 2 bit 可打包  -> 每顆 2 組 */
#define N_COL         4
#define GRP_S3        (N_COL * 3)
#define GRP_S2        (N_COL * 3)
#define LANE_S3       1
#define LANE_S2       2
#define DSP_NEED(g,l) (((g) + (l) - 1) / (l))
#define DSP_MAX(a,b)  ((a) > (b) ? (a) : (b))
#define N_DSP         DSP_MAX(DSP_NEED(GRP_S3, LANE_S3), DSP_NEED(GRP_S2, LANE_S2))   /* = 12 */
#define N_SLOT        (N_DSP / 3)                                                    /* = 4  */

#define WORD_FIFO_DEPTH 64

#define MAX_IN_BEATS    (1920 * 1080 / 8)          /* co-sim 深度 */
#define MAX_OUT_WORDS   (960 * 540 * 3 / 16)       /* co-sim 深度，2 倍最大輸出 */


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
 *    等於一個乘法器分時使用 4 拍。
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
                        ap_uint<32> &in_beats,
                        ap_uint<32> &out_words,
                        ap_uint<12> &out_w)
{
// #pragma HLS INLINE off
#pragma HLS ALLOCATION function instances=mul12 limit=1
    const ap_uint<12> K_DIV3 = 1366;                     /* x/3 = (x*1366)>>12 */

    /* ---- 第 1、2 次：除以 3 ---- */
    ap_uint<24> w1366 = mul12(img_w, K_DIV3);
    ap_uint<24> h1366 = mul12(img_h, K_DIV3);
    out_w = s3 ? (ap_uint<12>)(w1366 >> 12) : (ap_uint<12>)(img_w >> 1);
    ap_uint<12> out_h = s3 ? (ap_uint<12>)(h1366 >> 12) : (ap_uint<12>)(img_h >> 1);

    /* ---- 第 3 次：輸出 pixel 數 -> 輸出 word 數 ---- */
    ap_uint<24> out_pix = mul12(out_w, out_h);
    ap_uint<26> out_bytes = ((ap_uint<26>)out_pix << 1) + out_pix;     /* x3：移位加法 */
    out_words     = (out_bytes + 15) >> 4;                             /* ceil(bytes/16) */

    /* ---- 第 4 次：輸入拍數，一拍 8 pixel（resize 迴圈也以輸入拍為單位）---- */
    ap_uint<24> beats = mul12((ap_uint<12>)(img_w >> 3), img_h);
    in_beats = beats;
}


/* ################################################################
 *
 *  UYVY -> RGB（INLINE）
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

/* ================================================================
 *  一拍 UYVY -> 8 個 RGB pixel（INLINE 進 compute_side 的迴圈）
 *    uyvy_in_select -> 4 x cvt_pair -> uyvy_out_select
 * ================================================================ */
static ap_uint<192> uyvy_convert(const ap_uint<128> &raw)
{
#pragma HLS INLINE
    ap_uint<8> u[UYVY_GRP], y0[UYVY_GRP], v[UYVY_GRP], y1[UYVY_GRP];
#pragma HLS ARRAY_PARTITION variable=u  complete
#pragma HLS ARRAY_PARTITION variable=y0 complete
#pragma HLS ARRAY_PARTITION variable=v  complete
#pragma HLS ARRAY_PARTITION variable=y1 complete
    uyvy_in_select(raw, u, y0, v, y1);                  /* 計算資料選擇 */

    ap_uint<24> px[UYVY_PIX];
#pragma HLS ARRAY_PARTITION variable=px complete
    for (int g = 0; g < UYVY_GRP; g++) {                /* 運算：4 x cvt_pair */
#pragma HLS UNROLL
        cvt_pair(y0[g], y1[g], u[g], v[g], px[2 * g], px[2 * g + 1]);
    }

    return uyvy_out_select(px);                         /* 寫出資料選擇 */
}


/* ################################################################
 *
 *  resize（INLINE 選擇邏輯）
 *
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
            const int b3 = 3*s*24 + c*8;       /* 3 倍：slot s 第一個 pixel 的通道 c */
            const int b2 = 4*s*24 + c*8;       /* 2 倍（只用 s < 2） */

            /* 欄 s 所在的 bank 與 lane */
            ap_uint<LBW>  qs   = (s < 2) ? qA : qB;
            ap_uint<ACCW> lane;
            if ((s & 1) == 0) lane = qs.range(c*24 + 23, c*24 + 12);   /* 偶數欄：高 lane */
            else              lane = qs.range(c*24 + 11, c*24);        /* 奇數欄：低 lane */

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

/* ================================================================
 *  寫出資料選擇
 *
 *  (1) lane -> 輸出欄
 *        3 倍：slot s 只有 lo 有效，欄 s = pl[s]
 *        2 倍：slot 0..1，欄 2s = ph[s]，欄 2s+1 = pl[s]
 *
 *  (2) 輸出 pixel：依 scale_rate 小數位數取 8 bit，兩種模式都是 4 個 pixel
 *        3 倍 gp[23:16]，2 倍 gp[9:2]
 *
 *  (3) line buffer 寫回：last_row 寫 0；否則 B = 1，乘積即累加值，取 [11:0]
 *        bankA 每通道 = {欄0, 欄1}，bankB 每通道 = {欄2, 欄3}
 *        兩種模式每次都寫兩個 bank
 * ================================================================ */
static void resize_out_select(const ap_uint<PROD_W> ph[N_SLOT][3],
                              const ap_uint<PROD_W> pl[N_SLOT][3],
                              bool                  s3,
                              bool                  last_row,
                              ap_uint<96>          &res,
                              ap_uint<LBW>         &wA,
                              ap_uint<LBW>         &wB)
{
#pragma HLS INLINE

    /* ---- (1) lane -> 輸出欄 ---- */
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

    /* ---- (2) 輸出 pixel ---- */
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
            ap_uint<OPW>  sA = (g0, g1);
            ap_uint<OPW>  sB = (g2, g3);
            wA.range(c*24 + 23, c*24) = sA;
            wB.range(c*24 + 23, c*24) = sB;
        }
    }
}

/* ################################################################
 *
 *  寫出資料選擇 —— 96 -> 128 輸出對齊狀態機（INLINE）
 *
 *  兩種模式每筆 result 都是 4 個 pixel = 96 bit，狀態機相同
 *   狀態  暫存 L  動作                                        寫出  下一 L
 *   S0      0    hold[95:0] = r                                  —     96
 *   S1     96    out {r[31:0], hold[95:0]}；hold[63:0]=r[95:32]  是    64
 *   S2     64    out {r[63:0], hold[63:0]}；hold[31:0]=r[95:64]  是    32
 *   S3     32    out {r[95:0], hold[31:0]}                       是     0
 *
 *  INLINE 進 compute_side 的迴圈，只在產生 result 的那一拍呼叫；
 *  hold / st 由 compute_side 保存，湊滿的 word 寫進 word_ch。
 *
 *  寫法：switch(st) 每個 case 對應狀態表的一列
 *    w 用 .range() 逐段組出，hold / st 直接更新；word_out 只有一個寫出點
 *
 * ################################################################ */

static void pack_side(const ap_uint<96>          &res,
                      ap_uint<96>                &hold,
                      ap_uint<2>                 &st,
                      hls::stream<ap_uint<128> > &word_out)
{
#pragma HLS INLINE
    ap_uint<128> w    = 0;
    bool         emit = false;                   /* 只有寫出的狀態設為 true */

    switch (st) {
    case 0:                                  /* 0 -> 96 */
        hold.range(95, 0) = res;
        st = 1;
        break;
    case 1:                                  /* 暫存 96 + 32，寫出，留 64 */
        w.range( 95,  0) = hold.range(95, 0);
        w.range(127, 96) = res.range(31, 0);
        hold.range(63, 0) = res.range(95, 32);
        emit = true;
        st = 2;
        break;
    case 2:                                  /* 暫存 64 + 64，寫出，留 32 */
        w.range( 63,  0) = hold.range(63, 0);
        w.range(127, 64) = res.range(63, 0);
        hold.range(31, 0) = res.range(95, 64);
        emit = true;
        st = 3;
        break;
    case 3:                                  /* 暫存 32 + 96，寫出，留 0 */
        w.range( 31,  0) = hold.range(31, 0);
        w.range(127, 32) = res;
        emit = true;
        st = 0;
        break;
    default:                                 /* st 為 2 bit，0~3 皆已涵蓋 */
        st = 0;
        break;
    }

    if (emit)                                    /* 唯一寫出點（寫進 FIFO） */
        word_out.write(w);
}

/* 收尾：輸出 pixel 數不是 16 的倍數時，補 0 成最後一個 word */
static void pack_tail(const ap_uint<96>          &hold,
                      ap_uint<2>                  st,
                      hls::stream<ap_uint<128> > &word_out)
{
#pragma HLS INLINE
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
        word_out.write(tail);
    }
}


/* ================================================================
 *  compute_side（DATAFLOW process）：每拍讀一筆 AXI，湊滿才運算
 *    uyvy_convert -> resize_in_select -> resize_pe x N_DSP
 *                 -> resize_out_select -> pack_side -> word_ch
 * ================================================================ */
static void compute_side(const ap_uint<128>         *in,
                         hls::stream<ap_uint<128> > &word_out,
                         ap_uint<32>                 in_beats,
                         ap_uint<12>                 out_w,
                         ap_uint<1>                  scale_mode)
{
    ap_uint<LBW> lbA[QUAD_W_MAX];   /* 欄 4k, 4k+1 */
    ap_uint<LBW> lbB[QUAD_W_MAX];   /* 欄 4k+2, 4k+3 */
#pragma HLS BIND_STORAGE variable=lbA type=RAM_S2P impl=BRAM
#pragma HLS BIND_STORAGE variable=lbB type=RAM_S2P impl=BRAM

    const bool s3 = (scale_mode == SCALE_3);

    const ap_uint<LOG2_CEIL(3)> v_taps = s3 ? 3 : 2;
    const ap_uint<SRW>          sr     = s3 ? (ap_uint<SRW>)SR_S3 : (ap_uint<SRW>)SR_S2;
    const ap_uint<LOG2_CEIL((OUT_W_MAX+3)>>2)> quad_w = (out_w + 3) >> 2;

    /* ---- 輸入對齊狀態 ---- */
    ap_uint<192> prev = 0;
    ap_uint<2>   st   = 0;

    /* ---- 位置追蹤：兩種模式每次都前進 4 欄 ---- */
    ap_uint<LOG2_CEIL(OUT_W_MAX)> ox           = 0;
    ap_uint<LOG2_CEIL(3)>         row_in_block = 0;

    /* ---- 輸出對齊狀態（pack_side）---- */
    ap_uint<96> hold = 0;
    ap_uint<2>  pst  = 0;

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

        /* ---- 每拍無條件讀一筆 AXI（讀端 burst 不受影響）並轉成 8 個 RGB pixel ---- */
        ap_uint<192> beat = uyvy_convert(in[i]);

        ap_uint<10> base_idx = ox >> 2;
        bool        last_row = (row_in_block == v_taps - 1);

        ap_uint<LBW> qA = lbA[base_idx];
        ap_uint<LBW> qB = lbB[base_idx];

        /* ---- 計算資料選擇 ---- */
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
            /* ---- 運算：N_DSP 顆，3 倍每顆 1 組、2 倍每顆 2 組（後 6 顆閒置）---- */
            ap_uint<SRW>    mul_b = last_row ? sr : (ap_uint<SRW>)1;
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

            /* ---- 寫出資料選擇 ---- */
            ap_uint<96>  res;
            ap_uint<LBW> wA, wB;
            resize_out_select(ph, pl, s3, last_row, res, wA, wB);

            if (last_row)
                pack_side(res, hold, pst, word_out);   /* 湊滿才寫進 FIFO */
            lbA[base_idx] = wA;
            lbB[base_idx] = wB;

            ox += N_COL;
            if (ox >= out_w) {
                ox = 0;
                row_in_block++;
                if (row_in_block == v_taps)
                    row_in_block = 0;
            }
        }
    }

    pack_tail(hold, pst, word_out);
}


/* ################################################################
 *
 *  第二段（DATAFLOW process）：AXI 寫出（位址即迴圈變數、無條件包裹 -> burst）
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
 *  DATAFLOW 區域：compute_side -> word_ch -> axi_write_side
 *
 * ################################################################ */

static void uyvy_resize_dataflow(ap_uint<128> *uyvy_axi_bus,
                                 ap_uint<128> *rgb_axi_bus,
                                 ap_uint<32>   in_beats,
                                 ap_uint<32>   out_words,
                                 ap_uint<12>   out_w,
                                 ap_uint<1>    scale_mode)
{
#pragma HLS DATAFLOW

    hls::stream<ap_uint<128> > word_ch;
#pragma HLS STREAM       variable=word_ch   depth=WORD_FIFO_DEPTH
#pragma HLS BIND_STORAGE variable=word_ch   type=fifo impl=srl

    compute_side  (uyvy_axi_bus, word_ch, in_beats, out_w, scale_mode);
    axi_write_side(word_ch, rgb_axi_bus, out_words);
}


/* ################################################################
 *
 *  Top：介面 + 參數計算（一次）+ DATAFLOW
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

    ap_uint<32> in_beats, out_words;
    ap_uint<12> out_w;
    calc_params(img_w, img_h, scale_mode == SCALE_3, in_beats, out_words, out_w);

    uyvy_resize_dataflow(uyvy_axi_bus, rgb_axi_bus, in_beats, out_words,
                         out_w, scale_mode);
}