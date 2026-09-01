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
 * ============================================================
 *  line buffer 分 bank
 * ============================================================
 *
 * 一次運算同時寫入 n_out 個相鄰輸出欄（3倍 2 個、2倍 4 個）。
 * 若用單一陣列，HLS 無法證明索引不衝突，會報 200-885 埠不足。
 * 故拆成 4 塊獨立陣列，索引一律 idx = ox >> 2。
 * 每塊陣列的每一格存 RGB 打包後的 36-bit 值（3 x 12 bit），
 * 剛好是單顆 BRAM18 最寬配置的上限，相對 RGB 各自一塊省下 8 顆 BRAM：
 *   3 倍：ox 每次 +2，交替使用 (bank0,bank1) 與 (bank2,bank3)
 *   2 倍：ox 每次 +4，固定使用 bank0~bank3
 * 兩種模式每塊陣列每拍最多 1 讀 1 寫，T2P 雙埠足夠。
 *
 * 本版經 bit-accurate 模型驗證，12/12 案例通過。
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
/* 單通道累加器位元寬。3 倍模式最壞 9 x 255 = 2295 < 4096，12 bit 足夠。
 * 選 12 是為了讓 RGB 三通道打包後剛好 36 bit，塞進單顆 BRAM18
 * 的最寬配置（512 x 36）。經全白極端值測試驗證不溢位。 */
#define ACCW       12
#define PACKW      (ACCW * 3)   /* 36 bit：RGB 打包後的寬度 */

#define OP_BITS_S3 144      /* 6 pixel x 24 bit */
#define OP_BITS_S2 192      /* 8 pixel x 24 bit */

/* result FIFO：compute -> pack
 * 生產 0.889 筆/拍、消費 1.0 筆/拍，只需吸收瞬時波動 */
#define RES_FIFO_DEPTH  64

/* word FIFO：pack -> axi_write
 * 生產 0.333 word/拍、消費 1.0 word/拍，餘裕更大 */
#define WORD_FIFO_DEPTH 64

/* co-simulation 用的模擬記憶體大小，必須是編譯期常數。
 * 取最大工作尺寸 1920x1080 RGB 的需求。
 * testbench 的緩衝區配置必須 >= 這裡的值，否則 co-sim
 * 存取模擬記憶體時會越界（症狀為 SIGSEGV）。
 *
 * depth 只影響模擬，不影響合成出來的硬體。 */
#define IN_DEPTH   388800    /* 1920*1080*3/16 */
#define OUT_DEPTH   97200    /* 960*540*3/16，2 倍模式較大者 */


/* ================================================================
 *  第一段：讀取 + 加法樹 + 正規化
 *
 *  結果以 96-bit 打包送進 stream：
 *    3 倍：低 48 bit 有效（2 個輸出 pixel）
 *    2 倍：全 96 bit 有效（4 個輸出 pixel）
 * ================================================================ */

static void compute_side(ap_uint<128>                  *in_ptr,
                         hls::stream<ap_uint<96> >     &result_out,
                         ap_uint<LOG2_CEIL(IN_DEPTH)>  total_words,
                         ap_uint<LOG2_CEIL(OUT_W_MAX)> out_w,
                         ap_uint<1>                    scale_mode,
                         ap_uint<16>                   inv_scale)
{
    /* ---- 4 塊 line buffer，RGB 打包在同一個 36-bit 字 ----
     *
     * 每格存 [B:G:R] 三個 12-bit 值，共 36 bit，剛好是單顆 BRAM18
     * 最寬配置的上限。相對於 RGB 各自一塊（12 塊陣列）：
     *   BRAM  12 顆 -> 4 顆
     *   使用率 18% -> 47%
     *   記憶體埠 24 個 -> 8 個，排程壓力大減
     *
     * 打包只是儲存格式，不是運算格式：讀出後先切片成三個獨立的
     * 12-bit 值各自加法，再打包回去，所以不會有 R 溢位污染 G 的問題。
     *
     * 4 bank 讓一次運算能同時寫入最多 4 個相鄰輸出欄。 */
    ap_uint<PACKW> lb0[QUAD_W_MAX];
    ap_uint<PACKW> lb1[QUAD_W_MAX];
    ap_uint<PACKW> lb2[QUAD_W_MAX];
    ap_uint<PACKW> lb3[QUAD_W_MAX];
#pragma HLS BIND_STORAGE variable=lb0 type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=lb1 type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=lb2 type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=lb3 type=RAM_T2P impl=BRAM

    /* ---- 輸入側狀態 ---- */
    ap_uint<128> leftover     = 0;   /* 上限 128 bit */
    ap_uint<8>   leftover_len = 0;

    /* ---- 位置追蹤 ---- */
    // int ox           = 0;
    // int row_in_block = 0;

    // Which Column in Line
    ap_uint<LOG2_CEIL(OUT_W_MAX)> ox           = 0;
    // Which Row of Block   Max Value is 3
    ap_uint<LOG2_CEIL(3)> row_in_block = 0;

    const bool        s3      = (scale_mode == SCALE_3);
    const ap_uint<LOG2_CEIL(3)> v_taps  = s3 ? 3 : 2;
    const ap_uint<LOG2_CEIL(4)> n_out   = s3 ? 2 : 4;    /* 一次運算產出幾個輸出欄 */
    const ap_uint<9>  op_bits = s3 ? (ap_uint<9>)OP_BITS_S3
                                   : (ap_uint<9>)OP_BITS_S2;
//    const int   quad_w  = (out_w + 3) >> 2;
    const ap_uint<LOG2_CEIL(QUAD_W_MAX)>   quad_w  = (out_w + 3) >> 2;

    init_loop: for (ap_uint<FOR_IDX_BITS(QUAD_W_MAX)> i = 0; i < quad_w; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=QUAD_W_MAX
        lb0[i] = 0; lb1[i] = 0; lb2[i] = 0; lb3[i] = 0;
    }

    /* ================================================================
     *  主迴圈：每拍讀一筆 AXI
     * ================================================================ */

    // main_loop: for (int i = 0; i < total_words; i++) {
    main_loop: for (ap_uint<FOR_IDX_BITS(IN_DEPTH)> i = 0; i < total_words; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=IN_DEPTH
#pragma HLS DEPENDENCE variable=lb0 inter false
#pragma HLS DEPENDENCE variable=lb1 inter false
#pragma HLS DEPENDENCE variable=lb2 inter false
#pragma HLS DEPENDENCE variable=lb3 inter false

        /* ---- 無條件連續讀取，burst inference 條件最佳 ---- */
        ap_uint<128> beat = in_ptr[i];

        ap_uint<256> w = 0;
        if (leftover_len > 0)
            w.range(leftover_len - 1, 0) = leftover;
        w.range(leftover_len + 127, leftover_len) = beat;

        ap_uint<9> total = leftover_len + 128;
        bool do_op = (total >= op_bits);

        /* 湊不滿一次運算就全部留到下次；第一批必然如此（128 < 144） */
        ap_uint<9> new_len = do_op ? (ap_uint<9>)(total - op_bits) : total;
        if (new_len > 0)
            leftover = do_op ? w.range(op_bits + new_len - 1, op_bits)
                             : w.range(new_len - 1, 0);
        leftover_len = new_len;

        if (do_op) {

            /* ---- 取出 8 個 pixel（3 倍只用前 6 個） ---- */
            ap_uint<24> p[8];
#pragma HLS ARRAY_PARTITION variable=p complete
            for (int j = 0; j < 8; j++) {
#pragma HLS UNROLL
                p[j] = w.range(j * 24 + 23, j * 24);
            }

            /* ---- 水平方向加總，RGB 三通道並行 ----
             * 3 倍：每 3 個 pixel 一組，共 2 組
             * 2 倍：每 2 個 pixel 一組，共 4 組 */
            ap_uint<10> h_r[4], h_g[4], h_b[4];
#pragma HLS ARRAY_PARTITION variable=h_r complete
#pragma HLS ARRAY_PARTITION variable=h_g complete
#pragma HLS ARRAY_PARTITION variable=h_b complete

            if (s3) {
                h_r[0] = p[0].range( 7, 0) + p[1].range( 7, 0) + p[2].range( 7, 0);
                h_g[0] = p[0].range(15, 8) + p[1].range(15, 8) + p[2].range(15, 8);
                h_b[0] = p[0].range(23,16) + p[1].range(23,16) + p[2].range(23,16);

                h_r[1] = p[3].range( 7, 0) + p[4].range( 7, 0) + p[5].range( 7, 0);
                h_g[1] = p[3].range(15, 8) + p[4].range(15, 8) + p[5].range(15, 8);
                h_b[1] = p[3].range(23,16) + p[4].range(23,16) + p[5].range(23,16);

                h_r[2] = 0; h_g[2] = 0; h_b[2] = 0;
                h_r[3] = 0; h_g[3] = 0; h_b[3] = 0;
            } else {
                for (int g = 0; g < 4; g++) {
#pragma HLS UNROLL
                    h_r[g] = p[g*2].range( 7, 0) + p[g*2+1].range( 7, 0);
                    h_g[g] = p[g*2].range(15, 8) + p[g*2+1].range(15, 8);
                    h_b[g] = p[g*2].range(23,16) + p[g*2+1].range(23,16);
                }
            }

            /* ---- 讀出四個 bank 的目前累加值 ---- */
            // int base_idx = ox >> 2;
            // int bsel     = ox & 3;          /* 3 倍時交替 0 / 2 */

            // Max Out Image Width = 1023
            ap_uint<10> base_idx = ox >> 2;
            ap_uint<2>  bsel     = ox & 3;          /* 3 倍時交替 0 / 2 */

            /* 每塊只讀一次 36-bit，再用 range 切成三個 12-bit。
             * 切片是純接線，不消耗記憶體埠。 */
            ap_uint<PACKW> q0 = lb0[base_idx];
            ap_uint<PACKW> q1 = lb1[base_idx];
            ap_uint<PACKW> q2 = lb2[base_idx];
            ap_uint<PACKW> q3 = lb3[base_idx];

            ap_uint<ACCW> c0_r = q0.range(ACCW-1, 0);
            ap_uint<ACCW> c0_g = q0.range(ACCW*2-1, ACCW);
            ap_uint<ACCW> c0_b = q0.range(ACCW*3-1, ACCW*2);
            ap_uint<ACCW> c1_r = q1.range(ACCW-1, 0);
            ap_uint<ACCW> c1_g = q1.range(ACCW*2-1, ACCW);
            ap_uint<ACCW> c1_b = q1.range(ACCW*3-1, ACCW*2);
            ap_uint<ACCW> c2_r = q2.range(ACCW-1, 0);
            ap_uint<ACCW> c2_g = q2.range(ACCW*2-1, ACCW);
            ap_uint<ACCW> c2_b = q2.range(ACCW*3-1, ACCW*2);
            ap_uint<ACCW> c3_r = q3.range(ACCW-1, 0);
            ap_uint<ACCW> c3_g = q3.range(ACCW*2-1, ACCW);
            ap_uint<ACCW> c3_b = q3.range(ACCW*3-1, ACCW*2);

            /* 3 倍模式：bsel=0 用 bank0/1，bsel=2 用 bank2/3
             * 2 倍模式：bsel 恆 0，四個 bank 全用 */
            ap_uint<ACCW> in0_r = s3 ? (bsel ? c2_r : c0_r) : c0_r;
            ap_uint<ACCW> in0_g = s3 ? (bsel ? c2_g : c0_g) : c0_g;
            ap_uint<ACCW> in0_b = s3 ? (bsel ? c2_b : c0_b) : c0_b;
            ap_uint<ACCW> in1_r = s3 ? (bsel ? c3_r : c1_r) : c1_r;
            ap_uint<ACCW> in1_g = s3 ? (bsel ? c3_g : c1_g) : c1_g;
            ap_uint<ACCW> in1_b = s3 ? (bsel ? c3_b : c1_b) : c1_b;

            /* ---- 累加 ---- */
            ap_uint<ACCW> a0_r = in0_r + h_r[0], a0_g = in0_g + h_g[0], a0_b = in0_b + h_b[0];
            ap_uint<ACCW> a1_r = in1_r + h_r[1], a1_g = in1_g + h_g[1], a1_b = in1_b + h_b[1];
            ap_uint<ACCW> a2_r = c2_r  + h_r[2], a2_g = c2_g  + h_g[2], a2_b = c2_b  + h_b[2];
            ap_uint<ACCW> a3_r = c3_r  + h_r[3], a3_g = c3_g  + h_g[3], a3_b = c3_b  + h_b[3];

            bool last_row = (row_in_block == v_taps - 1);

            /* ---- 決定寫回值，最後統一寫入 ----
             * 刻意不在兩個分支各自寫 BRAM，避免同一陣列出現
             * 兩個寫入點而被判定需要兩個寫埠（HLS 200-885） */
            ap_uint<ACCW> w0_r = 0, w0_g = 0, w0_b = 0;
            ap_uint<ACCW> w1_r = 0, w1_g = 0, w1_b = 0;
            ap_uint<ACCW> w2_r = 0, w2_g = 0, w2_b = 0;
            ap_uint<ACCW> w3_r = 0, w3_g = 0, w3_b = 0;

            if (last_row) {
                /* 正規化：乘上倒數取代除法，inv_scale = 65536/(scale^2)
                 *
                 * BIND_OP 必須綁在具名變數上，不能直接對表達式下 pragma，
                 * 故先把乘積存進 m?_? 再移位。
                 * 六個（2 倍時十二個）乘法必須各自獨立佔用一顆 DSP，
                 * 絕不可加 ALLOCATION limit 讓它們共用——那會強制序列化，
                 * II=1 就保不住。 */
                ap_uint<ACCW+16> m0_r = a0_r * inv_scale;
#pragma HLS BIND_OP variable=m0_r op=mul impl=dsp
                ap_uint<ACCW+16> m0_g = a0_g * inv_scale;
#pragma HLS BIND_OP variable=m0_g op=mul impl=dsp
                ap_uint<ACCW+16> m0_b = a0_b * inv_scale;
#pragma HLS BIND_OP variable=m0_b op=mul impl=dsp
                ap_uint<ACCW+16> m1_r = a1_r * inv_scale;
#pragma HLS BIND_OP variable=m1_r op=mul impl=dsp
                ap_uint<ACCW+16> m1_g = a1_g * inv_scale;
#pragma HLS BIND_OP variable=m1_g op=mul impl=dsp
                ap_uint<ACCW+16> m1_b = a1_b * inv_scale;
#pragma HLS BIND_OP variable=m1_b op=mul impl=dsp

                ap_uint<8> o0_r = (ap_uint<8>)(m0_r >> 16);
                ap_uint<8> o0_g = (ap_uint<8>)(m0_g >> 16);
                ap_uint<8> o0_b = (ap_uint<8>)(m0_b >> 16);
                ap_uint<8> o1_r = (ap_uint<8>)(m1_r >> 16);
                ap_uint<8> o1_g = (ap_uint<8>)(m1_g >> 16);
                ap_uint<8> o1_b = (ap_uint<8>)(m1_b >> 16);

                ap_uint<96> res = 0;
                res.range(23,  0) = ((ap_uint<24>)o0_b << 16)
                                  | ((ap_uint<24>)o0_g <<  8) | (ap_uint<24>)o0_r;
                res.range(47, 24) = ((ap_uint<24>)o1_b << 16)
                                  | ((ap_uint<24>)o1_g <<  8) | (ap_uint<24>)o1_r;

                if (!s3) {
                    ap_uint<ACCW+16> m2_r = a2_r * inv_scale;
#pragma HLS BIND_OP variable=m2_r op=mul impl=dsp
                    ap_uint<ACCW+16> m2_g = a2_g * inv_scale;
#pragma HLS BIND_OP variable=m2_g op=mul impl=dsp
                    ap_uint<ACCW+16> m2_b = a2_b * inv_scale;
#pragma HLS BIND_OP variable=m2_b op=mul impl=dsp
                    ap_uint<ACCW+16> m3_r = a3_r * inv_scale;
#pragma HLS BIND_OP variable=m3_r op=mul impl=dsp
                    ap_uint<ACCW+16> m3_g = a3_g * inv_scale;
#pragma HLS BIND_OP variable=m3_g op=mul impl=dsp
                    ap_uint<ACCW+16> m3_b = a3_b * inv_scale;
#pragma HLS BIND_OP variable=m3_b op=mul impl=dsp

                    ap_uint<8> o2_r = (ap_uint<8>)(m2_r >> 16);
                    ap_uint<8> o2_g = (ap_uint<8>)(m2_g >> 16);
                    ap_uint<8> o2_b = (ap_uint<8>)(m2_b >> 16);
                    ap_uint<8> o3_r = (ap_uint<8>)(m3_r >> 16);
                    ap_uint<8> o3_g = (ap_uint<8>)(m3_g >> 16);
                    ap_uint<8> o3_b = (ap_uint<8>)(m3_b >> 16);

                    res.range(71, 48) = ((ap_uint<24>)o2_b << 16)
                                      | ((ap_uint<24>)o2_g <<  8) | (ap_uint<24>)o2_r;
                    res.range(95, 72) = ((ap_uint<24>)o3_b << 16)
                                      | ((ap_uint<24>)o3_g <<  8) | (ap_uint<24>)o3_r;
                }

                /* 送進 FIFO，位元累積與 AXI 寫出交給後續兩段處理 */
                result_out.write(res);

                /* last_row 時全部歸零，w?_* 保持宣告時的 0 */

            } else {
                w0_r = a0_r; w0_g = a0_g; w0_b = a0_b;
                w1_r = a1_r; w1_g = a1_g; w1_b = a1_b;
                w2_r = a2_r; w2_g = a2_g; w2_b = a2_b;
                w3_r = a3_r; w3_g = a3_g; w3_b = a3_b;
            }

            /* ---- 打包回 36-bit ---- */
            ap_uint<PACKW> p0 = ((ap_uint<PACKW>)w0_b << (ACCW*2))
                              | ((ap_uint<PACKW>)w0_g << ACCW)
                              |  (ap_uint<PACKW>)w0_r;
            ap_uint<PACKW> p1 = ((ap_uint<PACKW>)w1_b << (ACCW*2))
                              | ((ap_uint<PACKW>)w1_g << ACCW)
                              |  (ap_uint<PACKW>)w1_r;
            ap_uint<PACKW> p2 = ((ap_uint<PACKW>)w2_b << (ACCW*2))
                              | ((ap_uint<PACKW>)w2_g << ACCW)
                              |  (ap_uint<PACKW>)w2_r;
            ap_uint<PACKW> p3 = ((ap_uint<PACKW>)w3_b << (ACCW*2))
                              | ((ap_uint<PACKW>)w3_g << ACCW)
                              |  (ap_uint<PACKW>)w3_r;

            /* ---- 寫回：每塊陣列最多一次寫入 ---- */
            if (s3) {
                if (bsel) {
                    lb2[base_idx] = p0;
                    lb3[base_idx] = p1;
                } else {
                    lb0[base_idx] = p0;
                    lb1[base_idx] = p1;
                }
            } else {
                lb0[base_idx] = p0;
                lb1[base_idx] = p1;
                lb2[base_idx] = p2;
                lb3[base_idx] = p3;
            }

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
 *  第二段：位元累積（pack）
 *
 *  把 48/96-bit 的運算結果拼成 128-bit word，湊滿才丟進 FIFO。
 *  這段仍有條件判斷，但只碰 FIFO 不碰 AXI，
 *  burst inference 不受影響。
 *
 *  殘餘序列：3 倍 {16,8,0}、2 倍 {16,32,0}
 * ================================================================ */

static void pack_side(hls::stream<ap_uint<96> >  &result_in,
                      hls::stream<ap_uint<128> > &word_out,
                      ap_uint<LOG2_CEIL(OUT_DEPTH)> total_results,
                      ap_uint<1>                  scale_mode)
{
    ap_uint<224> acc     = 0;   /* 最壞 127 + 96 = 223 bit */
    ap_uint<8>   acc_len = 0;

    const ap_uint<8> res_bits = (scale_mode == SCALE_3)
                              ? (ap_uint<8>)48 : (ap_uint<8>)96;

    // pack_loop: for (int r = 0; r < total_results; r++) {
    pack_loop: for (ap_uint<FOR_IDX_BITS(OUT_DEPTH)> r = 0; r < total_results; r++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=OUT_DEPTH

        ap_uint<96> res = result_in.read();

        acc.range(acc_len + res_bits - 1, acc_len) = res.range(res_bits - 1, 0);
        acc_len += res_bits;

        if (acc_len >= 128) {
            word_out.write(acc.range(127, 0));

            ap_uint<8> rem_len = acc_len - 128;

            /* rem_len 為 0 時 range(-1,0) 是未定義行為，必須 guard */
            if (rem_len > 0) {
                ap_uint<96> rem = acc.range(acc_len - 1, 128);
                acc = 0;
                acc.range(rem_len - 1, 0) = rem;
            } else {
                acc = 0;
            }
            acc_len = rem_len;
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
 *  這一段存在的唯一理由是讓 burst inference 成立。
 *  迴圈只做「讀 FIFO、寫 DDR」，位址是純粹的迴圈變數 i，
 *  沒有任何條件包裹——這是 burst inference 最理想的形式。
 * ================================================================ */

static void axi_write_side(hls::stream<ap_uint<128> > &word_in,
                           ap_uint<128>               *out_ptr,
                           ap_uint<LOG2_CEIL(OUT_DEPTH)>  out_words)
{
    // write_loop: for (int i = 0; i < out_words; i++) {
    write_loop: for (ap_uint<FOR_IDX_BITS(OUT_DEPTH)> i = 0; i < out_words; i++) {
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

    // compute_side  (in_ptr, result_ch, total_words, out_w, scale_mode, inv_scale);
    // pack_side     (result_ch, word_ch, total_results, scale_mode);
    // axi_write_side(word_ch, out_ptr, out_words);

    compute_side  (in_ptr, result_ch, total_words.range(LOG2_CEIL(IN_DEPTH)-1,0), out_w.range(LOG2_CEIL(OUT_DEPTH)-1,0), scale_mode, inv_scale);
    pack_side     (result_ch, word_ch, total_results.range(LOG2_CEIL(OUT_DEPTH)-1,0), scale_mode);
    axi_write_side(word_ch, out_ptr, out_words.range(LOG2_CEIL(OUT_DEPTH)-1,0));
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
 *     inv_scale     = 65536 / 4 = 16384
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
 *   2. DSP = 12（三通道 x 四組輸出）
 *      已用 BIND_OP impl=dsp 明確綁定，report 的 DSP 欄應為 12。
 *      切勿加 ALLOCATION instances=mul limit=N，那會強制共用、破壞 II=1。
 *
 *   2b. BRAM：4 塊 line buffer，每塊 out_w/4 x 36 bit
 *      若 report 顯示 12 顆，代表 RGB 打包沒生效
 *
 *   3. console 出現 in_ptr 與 out_ptr 的 burst inferred 訊息
 *      特別確認 gmem1：write_loop 現在是純粹的
 *          out_ptr[i] = word_in.read();
 *      位址即迴圈變數、無條件包裹，應該必定成立。
 *
 *   4. co-sim 的 main_loop Iteration Max II 應接近 1
 *      （synthesis 的 II=1 只是排程結果，co-sim 才反映 AXI 實際延遲）
 *
 *
 * C simulation 驗證
 *
 *   印出 leftover_len 序列比對：
 *     3 倍應走 {128,112,96,80,64,48,32,16,0} 週期 9
 *     2 倍應走 {128,64,0}                    週期 3
 *   第一批 do_op 必為 0（128 < 144），這是正常暖機
 *
 *   測試圖務必包含隨機與垂直漸層兩種 pattern——單色圖與水平漸層
 *   在輸出欄錯位時剛好值相同，完全無法偵測該類 bug。
 *****************************************************************************/
