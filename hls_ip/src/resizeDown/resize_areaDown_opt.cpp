/******************************************************************************
 * resize_kernel_fast.cpp
 *
 * 整數倍 Box-filter 縮小 (2x / 3x)，RGB888 packed，AXI 128-bit 介面
 *
 * 三段 DATAFLOW：運算 -> 位元累積 -> AXI 寫出
 *
 *   compute_side    每拍讀一筆 AXI，RGB 三通道並行累加
 *        |            3 倍：一次運算 2 個 3-pixel block -> 2 個輸出欄
 *        |            2 倍：一次運算 4 個 2-pixel block -> 4 個輸出欄
 *        |  hls::stream<ap_uint<96>>   result_ch
 *        v
 *   pack_side       把 48/96-bit 結果拼成 128-bit word
 *        |            仍有條件判斷，但只碰 FIFO 不碰 AXI
 *        |  hls::stream<ap_uint<128>>  word_ch
 *        v
 *   axi_write_side  out_ptr[i] = word_in.read()
 *                    位址即迴圈變數、無條件包裹，burst inference 必成
 *
 * ============================================================
 *  撰寫規則
 * ============================================================
 *
 *  1. 位元打包 / 拆解一律寫成 range 對 range，不使用 shift + OR。
 *  2. 加減乘一律明確指定運算元與結果寬度：
 *       ap_uint<N+1> = ap_uint<N>(a) + ap_uint<N>(b)
 *     旁邊註明數值上限作為依據。
 *  3. 有規律的切片寫成 UNROLL 迴圈，索引可用乘法，
 *     展開後為常數，只剩接線。
 *
 * ============================================================
 *  為什麼寫出要拆成兩段
 * ============================================================
 *
 * 舊版把 out_ptr[word_idx++] 寫在 last_row 與 acc_len>=128 兩層
 * 條件裡，HLS 判定為條件式存取，gmem1 的 burst 推斷失敗：
 *   每個 word 變成獨立 AXI 交易，往返約 12 拍
 *
 * 3 倍模式輸出 43200 word：
 *   43200 x 12 = 518400 拍，與讀取端的 388800 拍相當，
 *   反壓回主迴圈後總時間翻倍（實測 3307us vs 理論 1555us @250MHz）
 *
 * 拆開後，條件判斷留在 pack_side（只碰 FIFO），
 * axi_write_side 只做純粹的連續寫出。
 *
 * ============================================================
 *  運算路徑與 DSP
 * ============================================================
 *
 *  每組 g、每通道 k：
 *    A = bank 累加值 + 第一個 pixel     fabric 加法器
 *    D = 第二個 pixel + 第三個 pixel    fabric 加法器
 *
 *  g0 / g1（3 倍、2 倍共用）：
 *    M = (A + D) * B                    DSP48E2：pre-adder + 乘法器
 *    B = last_row ? inv_scale : 65536
 *      last_row   ：M[23:16] = 正規化後的 8-bit 輸出
 *      非 last_row：M[27:16] = A + D，寫回 BRAM
 *    pre-adder 的 AD 在 DSP 內部拉不出來，若在 fabric 另算一份
 *    A + D，HLS 就不會把加法吸收進 DSP，故非最後一列也乘 65536。
 *
 *  g2 / g3（只有 2 倍會用到）：
 *    2 倍權重 16384 = 2^14，(S * 2^14) >> 16 = S >> 2
 *    正規化只是取 S[9:2]，不需要 DSP。
 *
 *  DSP 總數 = 2 組 x 3 通道 = 6
 *
 *  分組：
 *    3 倍：g0 = (bank + p0) + (p1 + p2)
 *          g1 = (bank + p3) + (p4 + p5)
 *    2 倍：g  = (bank + p[2g]) + (p[2g+1] + 0)
 *
 * ============================================================
 *  leftover 狀態序列（設計驗證指紋）
 * ============================================================
 *
 *   3 倍 (144 bit/次)：{128,112,96,80,64,48,32,16,0} 週期 9 批
 *                      do_op = {0,1,1,1,1,1,1,1,1}  平均 0.889
 *   2 倍 (192 bit/次)：{128,64,0}                    週期 3 批
 *                      do_op = {0,1,1}              平均 0.667
 *
 *   兩者 leftover 上限皆 128 bit -> window = 128+128 = 256 bit
 *   第一批必然 do_op = 0（128 < 144），這是正常的暖機行為
 *
 *   程式內以 lsel = leftover_len / 16 表示：
 *     3 倍 {8,7,6,5,4,3,2,1,0}，2 倍 {8,4,0}
 *
 *  leftover 恆為「上一筆 beat 的最高 leftover_len bit」：
 *    op_bits >= 144 > 128 >= leftover_len，所以一次運算會把舊 leftover
 *    全部吃掉，剩下的只來自本次 beat 的高位：
 *      do_op  ：新 leftover = beat[127 : op_bits - len]（beat 的最高 new_len bit）
 *      !do_op ：只發生在 len = 0，新 leftover = beat（最高 128 bit）
 *    因此只需存 prev_beat 與 lsel，迴圈回授路徑上沒有 256-bit 位移器。
 *
 * ============================================================
 *  line buffer 分 bank
 * ============================================================
 *
 * 一次運算同時寫入 n_out 個相鄰輸出欄（3倍 2 個、2倍 4 個）。
 * 若用單一陣列，HLS 無法證明索引不衝突，會報 200-885 埠不足。
 * 故拆成 4 塊獨立陣列，索引一律 idx = ox >> 2。
 * 每格存 36-bit：[35:24]=B  [23:12]=G  [11:0]=R
 *   3 倍：ox 每次 +2，交替使用 (bank0,bank1) 與 (bank2,bank3)
 *   2 倍：ox 每次 +4，固定使用 bank0~bank3
 * 兩種模式每塊陣列每拍最多 1 讀 1 寫，T2P 雙埠足夠。
 *
 *****************************************************************************/

#include "resize_areaDown.h"
#include "ap_int.h"
#include "hls_stream.h"

/* ---------------------------------------------------------------- 常數 */

#define SCALE_2    0        /* 2 倍縮小：2x2 box */
#define SCALE_3    1        /* 3 倍縮小：3x3 box */

#define OUT_W_MAX  960      /* 輸出寬度上限（ox 用 10 bit） */
#define QUAD_W_MAX 240      /* OUT_W_MAX / 4，每 bank 的深度 */

/* 單通道累加器 12 bit（3 倍最壞 9 x 255 = 2295 < 4096）
 * RGB 打包後 36 bit，剛好是單顆 BRAM18 最寬配置（512 x 36）。
 * 下方 range 皆以 ACCW=12、PACKW=36 計算，更改時需一併修改。 */
#define ACCW       12
#define PACKW      36

#define OP_BITS_S3 144      /* 6 pixel x 24 bit */
#define OP_BITS_S2 192      /* 8 pixel x 24 bit */

/* result FIFO：compute -> pack
 * 生產 0.889 筆/拍、消費 1.0 筆/拍，只需吸收瞬時波動 */
#define RES_FIFO_DEPTH  64

/* word FIFO：pack -> axi_write
 * 生產 0.333 word/拍、消費 1.0 word/拍，餘裕更大 */
#define WORD_FIFO_DEPTH 64

/* co-simulation 用的模擬記憶體大小，必須是編譯期常數。
 * testbench 的緩衝區配置必須 >= 這裡的值，否則 co-sim
 * 存取模擬記憶體時會越界（症狀為 SIGSEGV）。
 * depth 只影響模擬，不影響合成出來的硬體。 */
#define IN_DEPTH   388800    /* 1920*1080*3/16 */
#define OUT_DEPTH   97200    /* 960*540*3/16，2 倍模式較大者 */

/* total_results 上限。注意它不是 OUT_DEPTH：
 *   3 倍 640*360/2 = 115200、2 倍 960*540/4 = 129600，都大於 97200 */
#define RES_MAX    129600

/* 迴圈索引型別 */
typedef ap_uint<FOR_IDX_BITS(QUAD_W_MAX)> quad_idx_t;
typedef ap_uint<FOR_IDX_BITS(IN_DEPTH)>   main_idx_t;
typedef ap_uint<FOR_IDX_BITS(RES_MAX)>    pack_idx_t;
typedef ap_uint<FOR_IDX_BITS(OUT_DEPTH)>  wr_idx_t;


/* ================================================================
 *  第一段：讀取 + 加法樹 + 正規化
 *
 *  result 96-bit 打包格式，pixel g 的通道 k 位於
 *    [g*24 + k*8 + 7 : g*24 + k*8]，k: 0=R 1=G 2=B
 *  3 倍：低 48 bit 有效（2 個輸出 pixel）
 *  2 倍：全 96 bit 有效（4 個輸出 pixel）
 * ================================================================ */

static void compute_side(ap_uint<128>                  *in_ptr,
                         hls::stream<ap_uint<96> >     &result_out,
                         ap_uint<LOG2_CEIL(IN_DEPTH)>  total_words,
                         ap_uint<LOG2_CEIL(OUT_W_MAX)> out_w,
                         ap_uint<1>                    scale_mode,
                         ap_uint<16>                   inv_scale)
{
    ap_uint<PACKW> lb0[QUAD_W_MAX];
    ap_uint<PACKW> lb1[QUAD_W_MAX];
    ap_uint<PACKW> lb2[QUAD_W_MAX];
    ap_uint<PACKW> lb3[QUAD_W_MAX];
#pragma HLS BIND_STORAGE variable=lb0 type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=lb1 type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=lb2 type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=lb3 type=RAM_T2P impl=BRAM

    /* ---- 輸入側狀態 ----
     * leftover 永遠是「上一筆 beat 的最高 leftover_len 個 bit」
     * （證明見檔頭），所以直接存整筆 beat，回授路徑上沒有任何邏輯。
     * leftover_len 只會是 16 的倍數，以 lsel = leftover_len / 16 表示。 */
    ap_uint<128> prev_beat = 0;
    ap_uint<4>   lsel      = 0;      /* 0..8 */

    /* ---- 位置追蹤 ---- */
    ap_uint<10> ox           = 0;    /* 目前輸出欄，0..958 */
    ap_uint<2>  row_in_block = 0;    /* 目前在 block 的第幾列，0..2 */

    const bool        s3      = (scale_mode == SCALE_3);
    const ap_uint<2>  v_taps  = s3 ? 3 : 2;
    /* 一次運算產出幾個輸出欄。需 3 bit 才存得下 4：
     * 若用 ap_uint<LOG2_CEIL(4)> = ap_uint<2>，4 會變成 0，ox 永不前進 */
    const ap_uint<3>  n_out   = s3 ? 2 : 4;
    /* 一次運算消耗的位元數 / 16：3 倍 144/16 = 9，2 倍 192/16 = 12 */
    const ap_uint<4>  op_sel  = s3 ? (ap_uint<4>)(OP_BITS_S3 / 16)
                                   : (ap_uint<4>)(OP_BITS_S2 / 16);

    /* v_taps - 1：2b - 2b，結果 1 或 2 */
    const ap_uint<2>  v_last  = ap_uint<2>(v_taps) - ap_uint<2>(1);

    /* quad_w = (out_w + 3) >> 2
     * out_w <= 960，+3 <= 963 (11b)，>>2 <= 240 (8b) */
    const ap_uint<11> out_w_p3 = ap_uint<10>(out_w) + ap_uint<10>(3);
    const ap_uint<LOG2_CEIL(QUAD_W_MAX)> quad_w = out_w_p3.range(10, 2);

    init_loop: for (quad_idx_t i = 0; i < quad_w; i = quad_idx_t(i) + quad_idx_t(1)) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=QUAD_W_MAX
        lb0[i] = 0; lb1[i] = 0; lb2[i] = 0; lb3[i] = 0;
    }

    /* ================================================================
     *  主迴圈：每拍讀一筆 AXI
     * ================================================================ */

    main_loop: for (main_idx_t i = 0; i < total_words; i = main_idx_t(i) + main_idx_t(1)) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=IN_DEPTH
#pragma HLS DEPENDENCE variable=lb0 inter false
#pragma HLS DEPENDENCE variable=lb1 inter false
#pragma HLS DEPENDENCE variable=lb2 inter false
#pragma HLS DEPENDENCE variable=lb3 inter false

        /* ---- 無條件連續讀取，burst inference 條件最佳 ---- */
        ap_uint<128> beat = in_ptr[i];
        ap_uint<128> prev_beat_q = prev_beat;   /* 本次使用的上一筆 beat */

        /* ---- 狀態更新（迴圈回授，只有 4-bit 運算）----
         * total / 16 = lsel + 8：4b + 4b -> 5b，<= 16 */
        ap_uint<5> tsel  = ap_uint<4>(lsel) + ap_uint<4>(8);
        bool       do_op = (tsel >= op_sel);

        /* 湊不滿一次運算就全部留到下次；只會發生在 lsel = 0（第一批必然如此）
         * tsel - op_sel：只在 do_op 時使用，不會下溢，<= 7 */
        ap_uint<5> tsub = ap_uint<5>(tsel) - ap_uint<5>(op_sel);

        ap_uint<4> lsel_cur = lsel;
        lsel      = do_op ? ap_uint<4>(tsub.range(3, 0))    /* <= 7 */
                          : ap_uint<4>(tsel.range(3, 0));   /* 必為 8 */
        prev_beat = beat;

        /* ---- 組 window（前饋路徑，可被 pipeline 切開）----
         * cat = {beat, prev_beat}
         * w   = cat 右移 (8 - lsel) * 16，即 {beat, prev_beat 最高 lsel*16 bit}
         * lsel 只有 9 種值，寫成 9 選 1 mux，取代通用 barrel shifter */
        ap_uint<256> cat;
        cat.range(255, 128) = beat;
        cat.range(127,   0) = prev_beat_q;

        ap_uint<256> w = 0;
        switch (lsel_cur) {
            case 0:  w.range(127, 0) = cat.range(255, 128); break;
            case 1:  w.range(143, 0) = cat.range(255, 112); break;
            case 2:  w.range(159, 0) = cat.range(255,  96); break;
            case 3:  w.range(175, 0) = cat.range(255,  80); break;
            case 4:  w.range(191, 0) = cat.range(255,  64); break;
            case 5:  w.range(207, 0) = cat.range(255,  48); break;
            case 6:  w.range(223, 0) = cat.range(255,  32); break;
            case 7:  w.range(239, 0) = cat.range(255,  16); break;
            default: w = cat;                               break;
        }

        if (do_op) {

            /* ---- 取出 8 個 pixel：px[j][k]（3 倍只用前 6 個）----
             * 位置 [j*24 + k*8 + 7 : j*24 + k*8] */
            ap_uint<8> px[8][3];
#pragma HLS ARRAY_PARTITION variable=px complete dim=0

            px_loop: for (ap_uint<4> j = 0; j < 8; j = ap_uint<4>(j) + ap_uint<4>(1)) {
#pragma HLS UNROLL
                ap_uint<9> base = ap_uint<4>(j) * ap_uint<5>(24);            /* <= 168 */
                px_ch_loop: for (ap_uint<3> k = 0; k < 3; k = ap_uint<3>(k) + ap_uint<3>(1)) {
#pragma HLS UNROLL
                    ap_uint<6> off = ap_uint<2>(k) * ap_uint<4>(8);          /* <= 16  */
                    ap_uint<9> lo  = ap_uint<8>(base) + ap_uint<8>(off);     /* <= 184 */
                    ap_uint<9> hi  = ap_uint<8>(lo)   + ap_uint<8>(7);       /* <= 191 */
                    px[j][k] = w.range(hi, lo);
                }
            }

            /* ---- 讀出四個 bank ---- */
            ap_uint<8> base_idx = ox.range(9, 2);    /* ox >> 2，<= 239 */
            ap_uint<2> bsel     = ox.range(1, 0);    /* ox & 3，3 倍時交替 0 / 2 */
            bool       alt      = s3 && (bsel != 0); /* 3 倍且輪到 bank2/3 */

            /* 每塊只讀一次 36-bit，切片是純接線，不消耗記憶體埠 */
            ap_uint<PACKW> q[4];
#pragma HLS ARRAY_PARTITION variable=q complete
            q[0] = lb0[base_idx];
            q[1] = lb1[base_idx];
            q[2] = lb2[base_idx];
            q[3] = lb3[base_idx];

            /* cur[g][k]：位置 [k*12 + 11 : k*12] */
            ap_uint<ACCW> cur[4][3];
#pragma HLS ARRAY_PARTITION variable=cur complete dim=0

            cur_loop: for (ap_uint<3> g = 0; g < 4; g = ap_uint<3>(g) + ap_uint<3>(1)) {
#pragma HLS UNROLL
                cur_ch_loop: for (ap_uint<3> k = 0; k < 3; k = ap_uint<3>(k) + ap_uint<3>(1)) {
#pragma HLS UNROLL
                    ap_uint<6> lo = ap_uint<2>(k) * ap_uint<4>(12);          /* <= 24 */
                    ap_uint<6> hi = ap_uint<5>(lo) + ap_uint<5>(11);         /* <= 35 */
                    cur[g][k] = q[g].range(hi, lo);
                }
            }

            bool last_row = (row_in_block == v_last);

            /* DSP 的 B 埠：17 bit 無號，DSP48E2 的 18-bit 有號 B 埠放得下 */
            ap_uint<17> mul_b = last_row ? ap_uint<17>(inv_scale) : ap_uint<17>(65536);

            /* 刻意不在分支裡各自寫 BRAM，避免同一陣列出現
             * 兩個寫入點而被判定需要兩個寫埠（HLS 200-885） */
            ap_uint<96>    res = 0;
            ap_uint<PACKW> pk[4];
#pragma HLS ARRAY_PARTITION variable=pk complete

            grp_loop: for (ap_uint<3> g = 0; g < 4; g = ap_uint<3>(g) + ap_uint<3>(1)) {
#pragma HLS UNROLL
                ap_uint<8> rbase = ap_uint<3>(g) * ap_uint<5>(24);           /* <= 72 */
                pk[g] = 0;

                grp_ch_loop: for (ap_uint<3> k = 0; k < 3; k = ap_uint<3>(k) + ap_uint<3>(1)) {
#pragma HLS UNROLL
                    /* ---- 本組的 bank 舊值 ---- */
                    ap_uint<ACCW> prev;
                    if      (g == 0) prev = alt ? cur[2][k] : cur[0][k];
                    else if (g == 1) prev = alt ? cur[3][k] : cur[1][k];
                    else             prev = cur[g][k];

                    /* ---- 本組的三個 pixel ---- */
                    ap_uint<8> x, y, z;
                    if (g == 0) {
                        x = px[0][k];
                        y = px[1][k];
                        z = s3 ? px[2][k] : ap_uint<8>(0);
                    } else if (g == 1) {
                        x = s3 ? px[3][k] : px[2][k];
                        y = s3 ? px[4][k] : px[3][k];
                        z = s3 ? px[5][k] : ap_uint<8>(0);
                    } else if (g == 2) {
                        x = px[4][k];
                        y = px[5][k];
                        z = 0;
                    } else {
                        x = px[6][k];
                        y = px[7][k];
                        z = 0;
                    }

                    /* A：11b + 11b -> 12b
                     *    舊值最大 1530（3 倍 2 列 x 765），+255 = 1785 < 2048 */
                    ap_uint<12> pre_a = ap_uint<11>(prev) + ap_uint<11>(x);

                    /* D：8b + 8b -> 9b，最大 510（g2/g3 的 z 恆 0，加法會被折疊） */
                    ap_uint<9>  pre_d = ap_uint<8>(y) + ap_uint<8>(z);

                    ap_uint<8>  o_val;    /* 正規化後輸出 */
                    ap_uint<12> s_val;    /* 寫回 BRAM 的累加值 */

                    if (g < 2) {
                        /* ---- g0/g1：走 DSP ----
                         * (A + D) * B，寫在同一個運算式才會吸收 pre-adder
                         *   A + D：11b + 11b -> 12b，最大 2295
                         *   * B  ：12b x 17b -> 29b，最大 2295 x 65536
                         * 各自獨立佔用一顆 DSP，勿加 ALLOCATION limit */
                        ap_uint<29> m = ap_uint<12>(ap_uint<11>(pre_a) + ap_uint<11>(pre_d))
                                      * ap_uint<17>(mul_b);
#pragma HLS BIND_OP variable=m op=mul impl=dsp
                        o_val = m.range(23, 16);
                        s_val = m.range(27, 16);
                    } else {
                        /* ---- g2/g3：只有 2 倍會用到，不需要 DSP ----
                         *   A + D：11b + 11b -> 12b，2 倍最大 4 x 255 = 1020
                         *   /4 即取 [9:2]
                         * 3 倍時這兩組的結果不會被使用 */
                        ap_uint<12> s = ap_uint<11>(pre_a) + ap_uint<11>(pre_d);
                        o_val = s.range(9, 2);
                        s_val = s;
                    }

                    /* ---- 輸出：[g*24 + k*8 + 7 : g*24 + k*8] ----
                     * 3 倍時 g2/g3 為無效值，pack_side 只取低 48 bit */
                    ap_uint<6> ooff = ap_uint<2>(k) * ap_uint<4>(8);          /* <= 16 */
                    ap_uint<8> olo  = ap_uint<7>(rbase) + ap_uint<7>(ooff);   /* <= 88 */
                    ap_uint<8> ohi  = ap_uint<7>(olo)   + ap_uint<7>(7);      /* <= 95 */
                    res.range(ohi, olo) = o_val;

                    /* ---- 寫回值：[k*12 + 11 : k*12]，last_row 時歸零 ---- */
                    ap_uint<6> wlo = ap_uint<2>(k) * ap_uint<4>(12);          /* <= 24 */
                    ap_uint<6> whi = ap_uint<5>(wlo) + ap_uint<5>(11);        /* <= 35 */
                    pk[g].range(whi, wlo) = last_row ? ap_uint<12>(0) : s_val;
                }
            }

            /* 送進 FIFO，位元累積與 AXI 寫出交給後續兩段處理 */
            if (last_row)
                result_out.write(res);

            /* ---- 寫回：每塊陣列最多一次寫入 ---- */
            if (s3) {
                if (bsel) {
                    lb2[base_idx] = pk[0];
                    lb3[base_idx] = pk[1];
                } else {
                    lb0[base_idx] = pk[0];
                    lb1[base_idx] = pk[1];
                }
            } else {
                lb0[base_idx] = pk[0];
                lb1[base_idx] = pk[1];
                lb2[base_idx] = pk[2];
                lb3[base_idx] = pk[3];
            }

            /* ox + n_out：10b + 10b -> 11b，最大 958 + 2 = 960 */
            ap_uint<11> ox_next = ap_uint<10>(ox) + ap_uint<10>(n_out);
            if (ox_next >= out_w) {
                ox = 0;
                /* row_in_block + 1：最大 2 + 1 = 3，2 bit 足夠 */
                row_in_block = ap_uint<2>(row_in_block) + ap_uint<2>(1);
                if (row_in_block == v_taps)
                    row_in_block = 0;
            } else {
                ox = ox_next.range(9, 0);
            }
        }
    }
}


/* ================================================================
 *  第二段：位元累積（pack）
 *
 *  把 48/96-bit 的運算結果拼成 128-bit word，湊滿才丟進 FIFO。
 *  這段仍有條件判斷，但只碰 FIFO 不碰 AXI，
 *  burst inference 不受影響。
 *
 *  acc_len 在每次迭代開始時恆 < 128（7 bit）
 *  殘餘序列：3 倍 {48,96,16,64,112,32,80,0}、2 倍 {96,64,32,0}
 * ================================================================ */

static void pack_side(hls::stream<ap_uint<96> >    &result_in,
                      hls::stream<ap_uint<128> >   &word_out,
                      ap_uint<LOG2_CEIL(RES_MAX)>  total_results,
                      ap_uint<1>                   scale_mode)
{
    ap_uint<224> acc     = 0;   /* 最壞 127 + 96 = 223 bit */
    ap_uint<7>   acc_len = 0;   /* 0..127 */

    const ap_uint<7> res_bits = (scale_mode == SCALE_3)
                              ? (ap_uint<7>)48 : (ap_uint<7>)96;
    /* res_bits - 1：47 或 95 */
    const ap_uint<7> res_hi   = ap_uint<7>(res_bits) - ap_uint<7>(1);

    pack_loop: for (pack_idx_t r = 0; r < total_results; r = pack_idx_t(r) + pack_idx_t(1)) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=RES_MAX

        ap_uint<96> res = result_in.read();

        /* acc_len + res_bits：7b + 7b -> 8b，最大 127 + 96 = 223 */
        ap_uint<8> sum_len = ap_uint<7>(acc_len) + ap_uint<7>(res_bits);
        /* sum_len - 1：最大 222 */
        ap_uint<8> sum_hi  = ap_uint<8>(sum_len) - ap_uint<8>(1);

        acc.range(sum_hi, acc_len) = res.range(res_hi, 0);

        if (sum_len >= 128) {
            word_out.write(acc.range(127, 0));

            /* sum_len - 128：最大 95 */
            ap_uint<7> rem_len = ap_uint<8>(sum_len) - ap_uint<8>(128);

            /* rem_len 為 0 時 range(-1,0) 是未定義行為，必須 guard */
            if (rem_len > 0) {
                ap_uint<96> rem    = acc.range(sum_hi, 128);
                ap_uint<7>  rem_hi = ap_uint<7>(rem_len) - ap_uint<7>(1);
                acc = 0;
                acc.range(rem_hi, 0) = rem;
            } else {
                acc = 0;
            }
            acc_len = rem_len;
        } else {
            acc_len = sum_len.range(6, 0);   /* 此分支 sum_len < 128 */
        }
    }

    /* 收尾：殘餘位元補成最後一個 word
     * 640x360 RGB 總輸出 5529600 bit / 128 = 43200 整除，
     * 此分支實務上不會觸發，保留作為 assertion */
    if (acc_len > 0)
        word_out.write(acc.range(127, 0));
}


/* ================================================================
 *  第三段：AXI 寫出
 *
 *  迴圈只做「讀 FIFO、寫 DDR」，位址是純粹的迴圈變數 i，
 *  沒有任何條件包裹——這是 burst inference 最理想的形式。
 * ================================================================ */

static void axi_write_side(hls::stream<ap_uint<128> >    &word_in,
                           ap_uint<128>                  *out_ptr,
                           ap_uint<LOG2_CEIL(OUT_DEPTH)> out_words)
{
    write_loop: for (wr_idx_t i = 0; i < out_words; i = wr_idx_t(i) + wr_idx_t(1)) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=OUT_DEPTH
        out_ptr[i] = word_in.read();
    }
}


/* ================================================================
 *  Top-level
 * ================================================================ */

void resize_kernel(ap_uint<128> *in_ptr,
                   ap_uint<128> *out_ptr,
                   ap_uint<32>  total_words,
                   ap_uint<32>  total_results,
                   ap_uint<32>  out_words,
                   ap_uint<32>  out_w,
                   ap_uint<1>   scale_mode,
                   ap_uint<16>  inv_scale)
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

    /* 兩個 FIFO 都很淺，用 SRL 實作不佔 BRAM */
#pragma HLS BIND_STORAGE variable=result_ch type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=word_ch   type=fifo impl=srl

    /* 32-bit 暫存器截成各段實際需要的寬度，slice 寬度需與參數型別一致 */
    ap_uint<LOG2_CEIL(IN_DEPTH)>  tw = total_words  .range(LOG2_CEIL(IN_DEPTH)  - 1, 0);
    ap_uint<LOG2_CEIL(OUT_W_MAX)> ow = out_w        .range(LOG2_CEIL(OUT_W_MAX) - 1, 0);
    ap_uint<LOG2_CEIL(RES_MAX)>   tr = total_results.range(LOG2_CEIL(RES_MAX)   - 1, 0);
    ap_uint<LOG2_CEIL(OUT_DEPTH)> wn = out_words    .range(LOG2_CEIL(OUT_DEPTH) - 1, 0);

    compute_side  (in_ptr, result_ch, tw, ow, scale_mode, inv_scale);
    pack_side     (result_ch, word_ch, tr, scale_mode);
    axi_write_side(word_ch, out_ptr, wn);
}


/******************************************************************************
 * Host 端參數
 *
 *   3 倍  1920x1080 -> 640x360
 *     total_words   = 1920 * 1080 * 3 / 16 = 388800
 *     total_results = 640 * 360 / 2        = 115200  （一次運算產出 2 欄）
 *     out_words     = 640 * 360 * 3 / 16   = 43200   （輸出 128-bit word 數）
 *     out_w         = 640    （必須是 2 的倍數）
 *     scale_mode    = SCALE_3
 *     inv_scale     = 65536 / 9 = 7282
 *
 *   2 倍  1920x1080 -> 960x540
 *     total_words   = 388800
 *     total_results = 960 * 540 / 4        = 129600  （一次運算產出 4 欄）
 *     out_words     = 960 * 540 * 3 / 16   = 97200
 *     out_w         = 960    （必須是 4 的倍數）
 *     scale_mode    = SCALE_2
 *     inv_scale     = 65536 / 4 = 16384   （g0/g1 的 DSP 仍會用到）
 *
 *
 * 效能預期
 *
 *   總拍數 = total_words = 388800
 *   KV260 @ 250 MHz -> 1.56 ms -> 640 FPS
 *
 *
 * 合成後確認
 *
 *   1. 三個迴圈的 Interval 皆為 1
 *      main_loop / pack_loop / write_loop
 *
 *   2. DSP = 6（g0/g1 x 三通道）
 *      Vivado synth 後查 DSP cell：AMULTSEL = AD 代表 pre-adder 已吸收；
 *      若為 A，代表 A+D 仍在 fabric。
 *      切勿加 ALLOCATION instances=mul limit=N，那會強制共用、破壞 II=1。
 *
 *   2b. BRAM：4 塊 line buffer，每塊 out_w/4 x 36 bit
 *      若 report 顯示 12 顆，代表 RGB 打包沒生效
 *
 *   3. console 出現 in_ptr 與 out_ptr 的 burst inferred 訊息
 *
 *   4. co-sim 的 main_loop Iteration Max II 應接近 1
 *      （synthesis 的 II=1 只是排程結果，co-sim 才反映 AXI 實際延遲）
 *
 *
 * C simulation 驗證
 *
 *   印出 lsel * 16（即 leftover_len）序列比對：
 *     3 倍應走 {128,112,96,80,64,48,32,16,0} 週期 9
 *     2 倍應走 {128,64,0}                    週期 3
 *   第一批 do_op 必為 0（128 < 144），這是正常暖機
 *
 *   測試圖務必包含隨機與垂直漸層兩種 pattern——單色圖與水平漸層
 *   在輸出欄錯位時剛好值相同，完全無法偵測該類 bug。
 *****************************************************************************/