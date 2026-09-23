/******************************************************************************
 * resize_kernel_fast.cpp
 *
 * 整數倍 Box-filter 縮小 (2x / 3x)，RGB888 packed，AXI 128-bit 介面
 *
 * 三段 DATAFLOW：運算 -> 位元累積 -> AXI 寫出
 *
 *   compute_side    每拍讀一筆 AXI，RGB 三通道並行
 *        |            3 倍：一次運算 2 個 3-pixel block -> 2 個輸出欄
 *        |            2 倍：一次運算 4 個 2-pixel block -> 4 個輸出欄
 *        |  hls::stream<ap_uint<96>>   result_ch
 *        v
 *   pack_side       把 48/96-bit 結果拼成 128-bit word
 *        |  hls::stream<ap_uint<128>>  word_ch
 *        v
 *   axi_write_side  out_ptr[i] = word_in.read()
 *
 * ============================================================
 *  DSP 資料路徑：P = (A + D) * B，兩種模式共用 6 顆 DSP
 * ============================================================
 *
 * 累加與正規化都在 DSP 裡完成（前加器做累加、乘法器做正規化），
 * 非最後一列時 B = 1，乘法只是讓累加值原樣通過。
 * 6 顆 DSP = 2 個 slot x RGB 三通道，兩種模式使用同一組硬體。
 *
 *   3 倍（每 slot 一個輸出欄）：
 *     A = p0 + p1                          (fabric，<= 510)
 *     D = p2 + line_buf[i]                 (fabric，<= 1785)
 *     B = last_row ? scale_rate : 1        scale_rate 16 bit = 65536/9 = 7282
 *     P = ((p0+p1) + (p2+line_buf[i])) * B
 *     輸出 8 bit = P[23:16]                （小數位 16）
 *
 *   2 倍（每 slot 兩個輸出欄，SIMD 打包）：
 *     A = {p0+p1, 00, p2+p3}               每 lane 12 bit（10 bit 值 + 2 bit 0）
 *     D = {line_buf[i], 00, line_buf[i+1]}
 *     B = last_row ? scale_rate : 1        scale_rate 2 bit = 4/4 = 1
 *     P = (A + D) * B
 *     輸出 8 bit：高 lane = P[21:14]，低 lane = P[9:2]（小數位 2）
 *
 *   2 倍 lane 不互相污染的條件：
 *     單 lane 累加最大 4 x 255 = 1020 < 2^10   -> 前加器不進位到鄰 lane
 *     乘上 2-bit B（最大 3）後 <= 3060 < 2^12  -> 乘積不溢出 lane 的 12 bit
 *     中間那 2 bit 的 0 就是留給乘法結果長大用的。
 *
 *   位元寬與 DSP48E2 對照：
 *     前加器運算元 24 bit（unsigned，轉 signed 需 25 bit）<= 27
 *     B 16 bit（轉 signed 17 bit）                            <= 18
 *
 * ============================================================
 *  line buffer：2 個 bank，每格 72 bit = 2 欄 x RGB x 12 bit
 * ============================================================
 *
 *   bankA 存欄 {4k, 4k+1}，bankB 存欄 {4k+2, 4k+3}，idx = ox >> 2
 *   每格內通道 c 佔 [c*24+23 : c*24]，其中
 *     [c*24+23 : c*24+12] = 偶數欄（高 lane）
 *     [c*24+11 : c*24   ] = 奇數欄（低 lane）
 *   這個排列正好就是 2 倍模式 DSP 的 D 運算元，切片即可直接餵入。
 *
 *   3 倍：ox 每次 +2，bank = ox[1]，只讀寫其中一個 bank 的一格
 *   2 倍：ox 每次 +4，兩個 bank 各讀一格、各寫一格
 *   => 一拍最多讀 4 欄、寫 4 欄，但每個 bank 只需 1 讀 1 寫。
 *
 *   因此用 RAM_S2P（simple dual port）即可，不需要 T2P：
 *     BRAM36 的 SDP 模式可配 512 x 72，一個 bank 剛好一顆 BRAM36；
 *     TDP 模式單埠最寬只有 36 bit，72 bit 會變成兩倍 BRAM。
 *   功能上改成 RAM_T2P 也正確，只是多耗 BRAM。
 *
 * ============================================================
 *  輸入對齊狀態機（取代舊版 leftover + 可變位移）
 * ============================================================
 *
 * 關鍵觀察：每次運算完，剩下的位元一定是「當前 beat 的高位部分」。
 * 所以不必存 leftover 與 leftover_len，只要存上一拍的 beat（prev），
 * 暫存資料一律是 prev 的高 L 位元：prev[127 : 128-L]。
 * 每個狀態的 L 是常數，切片位置全部變成固定接線，
 * 舊版 256-bit 可變位移器（barrel shifter）完全消失，只剩一個多工器。
 *
 * 3 倍（每次運算 144 bit）
 *   狀態  暫存 L  do_op  運算資料 win[143:0]                   下一狀態 L'
 *   S0      0      0     —（本拍整筆存入 prev）                S1  128
 *   S1    128      1     {beat[ 15:0], prev[127:  0]}         S2  112
 *   S2    112      1     {beat[ 31:0], prev[127: 16]}         S3   96
 *   S3     96      1     {beat[ 47:0], prev[127: 32]}         S4   80
 *   S4     80      1     {beat[ 63:0], prev[127: 48]}         S5   64
 *   S5     64      1     {beat[ 79:0], prev[127: 64]}         S6   48
 *   S6     48      1     {beat[ 95:0], prev[127: 80]}         S7   32
 *   S7     32      1     {beat[111:0], prev[127: 96]}         S8   16
 *   S8     16      1     {beat[127:0], prev[127:112]}         S0    0
 *   一般式 Sk (k=1..8)：{beat[16k-1:0], prev[127:16k-16]}
 *   週期 9 拍、運算 8 次 -> 平均 0.889
 *
 * 2 倍（每次運算 192 bit）
 *   狀態  暫存 L  do_op  運算資料 win[191:0]                   下一狀態 L'
 *   S0      0      0     —（本拍整筆存入 prev）                S1  128
 *   S1    128      1     {beat[ 63:0], prev[127:  0]}         S2   64
 *   S2     64      1     {beat[127:0], prev[127: 64]}         S0    0
 *   週期 3 拍、運算 2 次 -> 平均 0.667
 *
 * 每一拍都無條件 prev = beat（下一狀態需要的剩餘位元就在裡面）。
 * 前提：main_loop 每拍都讀一筆 in_ptr（II=1、無條件讀取），本設計成立。
 *
 *****************************************************************************/

#include "resize_areaDown.h"
#include "ap_int.h"
#include "hls_stream.h"

/* ---------------------------------------------------------------- 常數 */

#define SCALE_2    0        /* 2 倍縮小：2x2 box */
#define SCALE_3    1        /* 3 倍縮小：3x3 box */

#define OUT_W_MAX  960      /* 輸出寬度上限 */
#define QUAD_W_MAX 240      /* OUT_W_MAX / 4，每 bank 的深度 */

#define ACCW       12               /* 單通道單欄累加寬度（= DSP lane 寬度） */
#define LBW        (ACCW * 2 * 3)   /* 72 bit：一格 = 2 欄 x RGB */
#define OPW        (ACCW * 2)       /* 24 bit：DSP 前加器運算元 */
#define SRW        16               /* scale_rate 埠寬度 */
#define PROD_W     (OPW + SRW)      /* 40 bit：乘積 */

#define FRAC_S3    16       /* 3 倍：scale_rate 用 16 bit 表示 */
#define FRAC_S2    2        /* 2 倍：scale_rate 用 2 bit 表示 */

/* ---- DSP 顆數 ----
 * 組 = 一個輸出欄的一個通道
 *   3 倍：2 欄 x RGB = 6 組，B 16 bit 無法打包 -> 每顆 1 組
 *   2 倍：4 欄 x RGB = 12 組，B 2 bit 可打包  -> 每顆 2 組
 * 實體 DSP 數取兩模式需求的最大值 */
#define GRP_S3        6
#define GRP_S2        12
#define LANE_S3       1
#define LANE_S2       2
#define DSP_NEED(g,l) (((g) + (l) - 1) / (l))
#define DSP_MAX(a,b)  ((a) > (b) ? (a) : (b))
#define N_DSP         DSP_MAX(DSP_NEED(GRP_S3, LANE_S3), DSP_NEED(GRP_S2, LANE_S2))   /* = 6 */


#define RES_FIFO_DEPTH  64
#define WORD_FIFO_DEPTH 64

#define IN_DEPTH   388800    /* 1920*1080*3/16 */
#define OUT_DEPTH   97200    /* 960*540*3/16，2 倍模式較大者 */


/* ################################################################
 *
 *  compute_side 用到的 function：
 *
 *    select_logic   選擇邏輯（純多工 / 接線，不含算術）
 *                   輸入對齊狀態機 + 從 win / line buffer 選出 PE 輸入
 *    dsp_addmul     單顆 DSP：P = (A + D) * B
 *    dsp_shared     兩組共用一顆 DSP：打包 -> dsp_addmul -> 拆 lane
 *    output_logic   其餘計算：輸出 pixel 取位 + 寫回資料組合
 *
 *  fabric 加法與「呼叫幾顆 dsp_shared」寫在 compute_side 主迴圈裡。
 *
 *  注意：concat 一律指定給完整寬度的變數，
 *  不可直接寫進 .range()，否則會經過 64-bit 轉換被截斷。
 *
 * ################################################################ */

/* ================================================================
 *  選擇邏輯
 *
 *  (1) 輸入對齊狀態機（推導表見檔頭），兩種模式寫同一個 win
 *  (2) PE 輸入選擇（slot s x 通道 c），讓 PE 內固定做
 *        h0 = u0 + u1，h1 = u2 + v
 *      3 倍：u0,u1,u2 = p[3s..3s+2]，v = line_buf 該欄（bsel 選 bank，
 *            slot0 取高 lane、slot1 取低 lane）
 *      2 倍：u0,u1,u2 = p[4s..4s+2]，v = p[4s+3]，
 *            lbp = {line_buf[i], line_buf[i+1]}（slot0 -> bankA，slot1 -> bankB）
 * ================================================================ */
static void select_logic(const ap_uint<128> &beat,
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

    sel_s: for (int s = 0; s < 2; s++) {
#pragma HLS UNROLL
        sel_c: for (int c = 0; c < 3; c++) {
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
 *  DSP 計算：P = (A + D) * B
 *
 *  對應 DSP48E2：前加器 (A + D) + 乘法器 (* B)
 *  INLINE 後每個呼叫點各自展開成一顆 DSP
 *
 *  位元寬：A、D 24 bit unsigned（轉 signed 25 bit <= 27）
 *          B    16 bit unsigned（轉 signed 17 bit <= 18）
 *  lane 不溢位由呼叫端的值域保證（見 dsp_shared 註解）
 * ================================================================ */
static ap_uint<PROD_W> dsp_addmul(ap_uint<OPW> a, ap_uint<OPW> d, ap_uint<SRW> b)
{
#pragma HLS INLINE
    ap_uint<OPW>    pre = a + d;       /* 前加器 */
    ap_uint<PROD_W> m   = pre * b;     /* 乘法器 */
#pragma HLS BIND_OP variable=m op=mul impl=dsp
    return m;
}


/* ================================================================
 *  兩組共用一顆 DSP
 *
 *  pack = 1（2 倍）：一顆 DSP 服務兩組
 *    A = {a_hi, a_lo}，D = {d_hi, d_lo}      每 lane 12 bit
 *    P = (A + D) * B
 *    p_hi = P[23:12]，p_lo = P[11:0]
 *    成立條件：每 lane (a + d) <= 1020 < 2^10，B <= 3
 *              -> 前加不進位、乘積 <= 3060 < 2^12 不溢出 lane
 *
 *  pack = 0（3 倍）：一顆 DSP 只服務一組（B 為 16 bit，乘積 28 bit，
 *    塞不進 12-bit lane，無法打包）
 *    A = a_lo，D = d_lo，p_lo = P（完整乘積），p_hi = 0
 *
 *  拆出來之後，每一組都拿到「自己的乘積」，後續取位不必再管 lane：
 *    輸出 8 bit = p[FRAC+7 : FRAC]，寫回 = p[11:0]
 * ================================================================ */
static void dsp_shared(bool             pack,
                       ap_uint<ACCW>    a_hi,
                       ap_uint<ACCW>    d_hi,
                       ap_uint<ACCW>    a_lo,
                       ap_uint<ACCW>    d_lo,
                       ap_uint<SRW>     b,
                       ap_uint<PROD_W> &p_hi,
                       ap_uint<PROD_W> &p_lo)
{
#pragma HLS INLINE
    ap_uint<OPW> A2 = (a_hi, a_lo);            /* 打包：{hi, lo} */
    ap_uint<OPW> D2 = (d_hi, d_lo);
    ap_uint<OPW> A  = pack ? A2 : (ap_uint<OPW>)a_lo;
    ap_uint<OPW> D  = pack ? D2 : (ap_uint<OPW>)d_lo;

    ap_uint<PROD_W> P = dsp_addmul(A, D, b);   /* 唯一的 DSP 呼叫點 */

    if (pack) {
        p_hi = P.range(2*ACCW - 1, ACCW);
        p_lo = P.range(ACCW - 1, 0);
    } else {
        p_hi = 0;
        p_lo = P;
    }
}

/* ================================================================
 *  其餘計算：輸出 pixel 取位 + 寫回資料組合
 *
 *  gp[col][c] 為每組（輸出欄 x 通道）自己的乘積
 *    3 倍：col 0..1；2 倍：col 0..3
 *
 *  輸出 8 bit：3 倍 gp[23:16]、2 倍 gp[9:2]
 *  寫回：last_row 寫 0；否則 B = 1，乘積即累加值，取 [11:0]
 *    bankA 每通道 = {col0, col1}
 *    bankB 每通道 = 3 倍：與 bankA 相同（由 bsel 決定寫哪個 bank）
 *                   2 倍：{col2, col3}
 * ================================================================ */
static void output_logic(const ap_uint<PROD_W> gp[4][3],
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

    /* ---- 輸出 pixel ---- */
    res = 0;
    out_col: for (int col = 0; col < 4; col++) {
#pragma HLS UNROLL
        out_c: for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
            ap_uint<8> px;
            if (s3) px = gp[col][c].range(FRAC_S3 + 7, FRAC_S3);
            else    px = gp[col][c].range(FRAC_S2 + 7, FRAC_S2);
            if (col < 2 || !s3)                         /* 3 倍只有 2 欄 */
                res.range(col*24 + c*8 + 7, col*24 + c*8) = px;
        }
    }

    /* ---- 寫回資料 ---- */
    wA = 0;
    wB = 0;
    if (!last_row) {
        wb_c: for (int c = 0; c < 3; c++) {
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
 *  第一段：讀取 + 運算
 *
 *    select_logic -> fabric 加法 -> dsp_shared x N_DSP -> output_logic
 *
 *  DSP 顆數在這裡決定（見 N_DSP 定義）：
 *    3 倍 6 組、每顆 1 組 -> 需 6 顆
 *    2 倍 12 組、每顆 2 組 -> 需 6 顆
 *  呼叫次數是編譯期常數 N_DSP，兩種模式共用同一批 DSP，
 *  模式只決定每顆是否打包。
 * ================================================================ */

static void compute_side(ap_uint<128>              *in_ptr,
                         hls::stream<ap_uint<96> > &result_out,
                         int                        total_words,
                         int                        out_w,
                         ap_uint<1>                 scale_mode,
                         ap_uint<SRW>               inv_scale)
{
    ap_uint<LBW> lbA[QUAD_W_MAX];   /* 欄 4k, 4k+1 */
    ap_uint<LBW> lbB[QUAD_W_MAX];   /* 欄 4k+2, 4k+3 */
#pragma HLS BIND_STORAGE variable=lbA type=RAM_S2P impl=BRAM
#pragma HLS BIND_STORAGE variable=lbB type=RAM_S2P impl=BRAM

    /* ---- 輸入對齊狀態 ---- */
    ap_uint<128> prev = 0;
    ap_uint<4>   st   = 0;

    /* ---- 位置追蹤 ---- */
    ap_uint<LOG2_CEIL(OUT_W_MAX)> ox           = 0;
    ap_uint<LOG2_CEIL(3)>         row_in_block = 0;

    const bool                  s3     = (scale_mode == SCALE_3);
    const bool                  pack   = !s3;          /* 2 倍：兩組共用一顆 DSP */
    const ap_uint<LOG2_CEIL(3)> v_taps = s3 ? 3 : 2;
    const ap_uint<3>            n_out  = s3 ? 2 : 4;   /* 需 3 bit 才存得下 4 */
    const ap_uint<LOG2_CEIL((OUT_W_MAX+3)>>2)> quad_w = (out_w + 3) >> 2;

    /* scale_rate：3 倍用完整 16 bit，2 倍只取低 2 bit */
    const ap_uint<SRW> sr = s3 ? inv_scale
                               : (ap_uint<SRW>)inv_scale.range(FRAC_S2 - 1, 0);

    init_loop: for (int i = 0; i < quad_w; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=QUAD_W_MAX
        lbA[i] = 0; lbB[i] = 0;
    }

    main_loop: for (ap_uint<FOR_IDX_BITS(400000)> i = 0; i < total_words; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=400000
#pragma HLS DEPENDENCE variable=lbA inter false
#pragma HLS DEPENDENCE variable=lbB inter false

        ap_uint<128> beat = in_ptr[i];

        ap_uint<10> base_idx = ox >> 2;
        bool        bsel     = ox[1];     /* 3 倍：0 -> bankA，1 -> bankB */
        bool        do_op    = (st != 0);
        bool        last_row = (row_in_block == v_taps - 1);

        /* line buffer 每拍都讀（讀取無副作用），寫入才受 do_op 控制 */
        ap_uint<LBW> qA = lbA[base_idx];
        ap_uint<LBW> qB = lbB[base_idx];

        /* ---- 選擇邏輯 ---- */
        ap_uint<4>    nst;
        ap_uint<8>    u0[2][3], u1[2][3], u2[2][3];
        ap_uint<ACCW> v[2][3];
        ap_uint<OPW>  lbp[2][3];
#pragma HLS ARRAY_PARTITION variable=u0  complete dim=0
#pragma HLS ARRAY_PARTITION variable=u1  complete dim=0
#pragma HLS ARRAY_PARTITION variable=u2  complete dim=0
#pragma HLS ARRAY_PARTITION variable=v   complete dim=0
#pragma HLS ARRAY_PARTITION variable=lbp complete dim=0
        select_logic(beat, prev, st, s3, qA, qB, bsel, nst, u0, u1, u2, v, lbp);
        st   = nst;
        prev = beat;

        if (do_op) {
            ap_uint<SRW> mul_b = last_row ? sr : (ap_uint<SRW>)1;

            /* ---- N_DSP 顆 DSP，每顆服務 1 組（3 倍）或 2 組（2 倍） ---- */
            ap_uint<PROD_W> gp[4][3];            /* 每組（輸出欄 x 通道）的乘積 */
#pragma HLS ARRAY_PARTITION variable=gp complete dim=0
            gp_init: for (int col = 0; col < 4; col++) {
#pragma HLS UNROLL
                for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
                    gp[col][c] = 0;
                }
            }

            dsp_loop: for (int k = 0; k < N_DSP; k++) {
#pragma HLS UNROLL
                const int s = k / 3;             /* slot */
                const int c = k % 3;             /* 通道 */

                /* fabric 加法（兩種模式共用） */
                ap_uint<ACCW> h0 = u0[s][c] + u1[s][c];
                ap_uint<ACCW> h1 = u2[s][c] + v[s][c];
                ap_uint<ACCW> lb_hi = lbp[s][c].range(OPW  - 1, ACCW);
                ap_uint<ACCW> lb_lo = lbp[s][c].range(ACCW - 1, 0);

                /* 組的分配
                 *   3 倍：只用 lo，a = p0+p1 (h0)，d = p2+line_buf (h1)
                 *   2 倍：hi = 欄 2s   ，a = h0，d = line_buf[i]
                 *         lo = 欄 2s+1，a = h1，d = line_buf[i+1] */
                ap_uint<ACCW> a_lo = s3 ? h0 : h1;
                ap_uint<ACCW> d_lo = s3 ? h1 : lb_lo;

                ap_uint<PROD_W> p_hi, p_lo;
                dsp_shared(pack, h0, lb_hi, a_lo, d_lo, mul_b, p_hi, p_lo);

                /* 把 lane 結果放回各自的組 */
                if (s3) {
                    gp[s][c] = p_lo;
                } else {
                    gp[2*s    ][c] = p_hi;
                    gp[2*s + 1][c] = p_lo;
                }
            }

            /* ---- 其餘計算 ---- */
            ap_uint<96>  res;
            ap_uint<LBW> wA, wB;
            bool         weA, weB;
            output_logic(gp, s3, bsel, last_row, res, wA, wB, weA, weB);

            if (last_row)
                result_out.write(res);
            if (weA) lbA[base_idx] = wA;      /* 每個 bank 單一寫入點 */
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


/* ================================================================
 *  第二段：位元累積（pack）—— 輸出對齊狀態機
 *
 *  與輸入端相同的觀察：每次湊滿 128 bit 寫出後，剩下的位元一定是
 *  「當前 result 的高位部分」；湊不滿時則把 result 放到 hold 的
 *  固定位置。每個狀態的暫存長度 L 是常數，所有切片都是固定接線，
 *  舊版 224-bit 可變位移器消失。
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
 *   週期 8 筆 result -> 3 個 word（384 bit）
 *
 *  2 倍（每筆 result 96 bit，r = res[95:0]）
 *   狀態  暫存 L  動作                                       寫出  下一 L
 *   S0      0    hold[95:0] = r                                 —     96
 *   S1     96    out {r[31:0], hold[95:0]}；hold[63:0]=r[95:32] 是    64
 *   S2     64    out {r[63:0], hold[63:0]}；hold[31:0]=r[95:64] 是    32
 *   S3     32    out {r[95:0], hold[31:0]}                      是     0
 *   週期 4 筆 result -> 3 個 word（384 bit）
 *
 *  hold 最寬需要 112 bit（3 倍 S5）。
 *  每個狀態只讀 hold 中「前面狀態剛寫過」的位元，其餘位元的舊值不影響結果。
 * ================================================================ */


static void pack_side(hls::stream<ap_uint<96> >  &result_in,
                      hls::stream<ap_uint<128> > &word_out,
                      int                         total_results,
                      ap_uint<1>                  scale_mode)
{
    ap_uint<112> hold = 0;
    ap_uint<3>   st   = 0;
    const bool   s3   = (scale_mode == SCALE_3);

    pack_loop: for (ap_uint<FOR_IDX_BITS(300000)> r = 0; r < total_results; r++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=300000

        ap_uint<96>  res = result_in.read();
        ap_uint<48>  r3  = res.range(47, 0);   /* 3 倍只有低 48 bit 有效 */

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

        if (emit)                      /* 單一寫出點 */
            word_out.write(word);

        hold = nh;
        st   = nst;
    }

    /* 收尾：結束時不在 S0 代表還有 L bit 未寫出，補 0 成一個 word。
     * 3 倍模式不會觸發；2 倍在 W*H 不是 64 的倍數時觸發。
     * host 端 out_words 需取 ceil(out_bits / 128)。 */
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


/* ================================================================
 *  第三段：AXI 寫出 —— 未修改
 * ================================================================ */

static void axi_write_side(hls::stream<ap_uint<128> > &word_in,
                           ap_uint<128>               *out_ptr,
                           int                         out_words)
{
    write_loop: for (ap_uint<FOR_IDX_BITS(100000)> i = 0; i < out_words; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=100000
        out_ptr[i] = word_in.read();
    }
}


/* ================================================================
 *  Top-level
 * ================================================================ */

void resize_kernel(ap_uint<128> *in_ptr,
                   ap_uint<128> *out_ptr,
                   int           total_words,
                   int           total_results,
                   int           out_words,
                   int           out_w,
                   ap_uint<1>    scale_mode,
                   ap_uint<16>   inv_scale)
{
#pragma HLS INTERFACE m_axi port=in_ptr  bundle=gmem0 offset=slave depth=IN_DEPTH \
    max_read_burst_length=64  num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=out_ptr bundle=gmem1 offset=slave depth=OUT_DEPTH \
    max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=total_words
#pragma HLS INTERFACE s_axilite port=total_results
#pragma HLS INTERFACE s_axilite port=out_words
#pragma HLS INTERFACE s_axilite port=out_w
#pragma HLS INTERFACE s_axilite port=scale_mode
#pragma HLS INTERFACE s_axilite port=inv_scale
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW

    hls::stream<ap_uint<96> >  result_ch;
    hls::stream<ap_uint<128> > word_ch;
#pragma HLS STREAM variable=result_ch depth=RES_FIFO_DEPTH
#pragma HLS STREAM variable=word_ch   depth=WORD_FIFO_DEPTH
#pragma HLS BIND_STORAGE variable=result_ch type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=word_ch   type=fifo impl=srl

    compute_side  (in_ptr, result_ch, total_words, out_w, scale_mode, inv_scale);
    pack_side     (result_ch, word_ch, total_results, scale_mode);
    axi_write_side(word_ch, out_ptr, out_words);
}


/******************************************************************************
 * Host 端參數
 *
 *   3 倍  1920x1080 -> 640x360
 *     total_words   = 388800
 *     total_results = 115200
 *     out_words     = 43200
 *     out_w         = 640    （必須是 2 的倍數）
 *     scale_mode    = SCALE_3
 *     inv_scale     = 65536 / 9 = 7282        （16 bit scale_rate）
 *
 *   2 倍  1920x1080 -> 960x540
 *     total_words   = 388800
 *     total_results = 129600
 *     out_words     = 97200
 *     out_w         = 960    （必須是 4 的倍數）
 *     scale_mode    = SCALE_2
 *     inv_scale     = 1                        （2 bit scale_rate = 4/4）
 *                     ※ 舊版的 16384 低 2 bit 為 0，會讓輸出全黑
 *
 *
 * 合成後確認
 *
 *   1. main_loop / pack_loop / write_loop 的 Interval 皆為 1
 *   2. DSP = 6（2 slot x RGB，兩種模式共用）
 *      bind_op report 應為 add-mul（前加器吸收進 DSP）；
 *      若前加法出現在 LUT，DSP 數仍是 6 但 fabric 會多 6 個 24-bit 加法器
 *   3. BRAM：2 個 bank x 240 x 72 bit，S2P 下每 bank 一顆 BRAM36
 *   4. lbA/lbB 設了 DEPENDENCE inter false：同一格的前後兩次存取間隔
 *      約 out_w/n_out 次運算，必須大於 read -> DSP -> write 的管線深度，
 *      因此 out_w 不可太小（實用上 >= 64 即安全）
 *****************************************************************************/
