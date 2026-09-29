/******************************************************************************
 * uyvy_resize_tb.cpp
 *
 * uyvy_resize（AXI -> uyvy2rgb -> resize -> AXI）的 C 驗證 testbench
 *
 * 介面：uyvy_resize.h
 *   void uyvy_resize(ap_uint<128> *uyvy_axi_bus, ap_uint<128> *rgb_axi_bus,
 *                    ap_uint<12> img_w, ap_uint<12> img_h,
 *                    ap_uint<1>  scale_mode);
 *
 * 驗證方式：
 *   黃金模型分兩段，都用純整數 C 寫成，不依賴 HLS 型別：
 *     (1) golden_uyvy2rgb  與 cvt_pair 位元一致的 UYVY -> RGB
 *     (2) golden_resize    與 resize 位元一致的 box filter
 *   IP 只輸出縮小後的結果，所以比對的是 (2) 的輸出。
 *
 * 驗證項目：
 *   1. 黃金 UYVY 模型的自我檢查（已知輸入 -> 已知輸出，含 clamp 兩端）
 *   2. 輸出像素與黃金模型逐一比對
 *   3. 最後一個 word 的補位 byte 為 0（輸出不是 16 byte 整數倍時）
 *   4. 輸出區尾端沒有被越界寫入
 *   5. 參數合法性（12-bit 範圍、倍數限制、輸出寬度上限）
 *
 * 編譯（純 C 模擬）：
 *   g++ -std=c++11 -I$XILINX_HLS/include \
 *       uyvy_resize_tb.cpp uyvy_resize_top.cpp resize_impl.cpp uyvy2rgb_impl.cpp -o tb
 *   ./tb
 *   加 -DSKIP_BIG_CASES 可跳過 1920x1080 等大尺寸案例
 *
 * 或在 Vitis HLS 中加入為 testbench 檔案後執行 C Simulation
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "ap_int.h"
#include "uyvy_resize_top.h"        /* uyvy_resize、SCALE_2 / SCALE_3 */


/* 必須與 uyvy_resize_top.cpp 的 m_axi depth 一致 */
#define IN_DEPTH    259200      /* = MAX_IN_BEATS  = 1920*1080/8    */
#define OUT_DEPTH    97200      /* = MAX_OUT_WORDS = 960*540*3/16   */
#define OUT_W_MAX      960      /* line buffer 上限 */


/* ================================================================
 *  黃金模型 (1)：UYVY -> RGB，與 cvt_pair 位元一致
 *
 *  d = U - 128，e = V - 128（有號 8 bit）
 *  權重為 Q1.8 / Q0.8 取最近值（uvy_w_t = ap_ufixed<9,1,AP_RND>）：
 *    1.772    -> 454 / 256     B 的色差項
 *    1.402    -> 359 / 256     R 的色差項
 *    0.344136 ->  88 / 256     G 的 D 項
 *    0.714136 -> 183 / 256     G 的 E 項
 *
 *  截斷方式與硬體相同（算術右移 = 向下取整）：
 *    B = clamp(Y + floor(454*d / 256))
 *    R = clamp(Y + floor(359*e / 256))
 *    G = clamp(Y - floor((88*d + 183*e) / 256))   兩項先相加再截斷一次
 *
 *  記憶體 byte 順序 R, G, B（byte0 = R）
 * ================================================================ */

static inline int floor_div256(int x)
{
    /* 向下取整的 x / 256，對負數也正確（等同算術右移 8） */
    return (x >= 0) ? (x >> 8) : -((-x + 255) >> 8);
}

static inline unsigned char clamp_u8(int v)
{
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

static void golden_pixel(int y, int u, int v, unsigned char *rgb)
{
    int d = u - 128;
    int e = v - 128;
    int b = y + floor_div256(454 * d);
    int r = y + floor_div256(359 * e);
    int g = y - floor_div256(88 * d + 183 * e);
    rgb[0] = clamp_u8(r);
    rgb[1] = clamp_u8(g);
    rgb[2] = clamp_u8(b);
}

/* UYVY 記憶體排列：每 4 byte = {U, Y0, V, Y1}，兩個 pixel 共用 U、V */
static void golden_uyvy2rgb(const unsigned char *uyvy, int w, int h,
                            unsigned char *rgb)
{
    int n_pairs = w * h / 2;
    for (int k = 0; k < n_pairs; k++) {
        const unsigned char *q = uyvy + k * 4;
        int u = q[0], y0 = q[1], v = q[2], y1 = q[3];
        golden_pixel(y0, u, v, rgb + (2 * k)     * 3);
        golden_pixel(y1, u, v, rgb + (2 * k + 1) * 3);
    }
}


/* ================================================================
 *  黃金模型 (2)：box filter，與 resize 位元一致
 *
 *  (sum * inv_scale) >> 16
 *    3 倍 inv = 7282   <-> IP 16-bit scale_rate 7282，取 [23:16]
 *    2 倍 inv = 16384  <-> IP 2-bit  scale_rate 1，   取 [9:2]（= sum >> 2）
 * ================================================================ */

static void golden_resize(const unsigned char *src, int src_w, int src_h,
                          unsigned char *dst, int scale)
{
    unsigned int inv_scale = (scale == 3) ? 7282 : 16384;
    int dst_w = src_w / scale;
    int dst_h = src_h / scale;

    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            unsigned int sum[3] = {0, 0, 0};
            for (int dy = 0; dy < scale; dy++)
                for (int dx = 0; dx < scale; dx++) {
                    const unsigned char *p =
                        src + ((y * scale + dy) * src_w + (x * scale + dx)) * 3;
                    for (int c = 0; c < 3; c++) sum[c] += p[c];
                }
            unsigned char *q = dst + (y * dst_w + x) * 3;
            for (int c = 0; c < 3; c++)
                q[c] = (unsigned char)((sum[c] * inv_scale) >> 16);
        }
    }
}


/* ================================================================
 *  第一部分：黃金 UYVY 模型自我檢查
 *
 *  用幾組可以手算的輸入確認公式、截斷方向與 clamp 兩端，
 *  避免黃金模型本身寫錯卻和 IP「一起錯」。
 * ================================================================ */

static bool self_check_golden()
{
    struct { int y, u, v; int r, g, b; const char *name; } t[] = {
        /* 灰階：d = e = 0，三通道都等於 Y */
        {128, 128, 128, 128, 128, 128, "灰階 Y=128"},
        {  0, 128, 128,   0,   0,   0, "黑"},
        {255, 128, 128, 255, 255, 255, "白"},
        /* d = +127：B = 200 + floor(454*127/256) = 200 + 225 -> 425 -> clamp 255
         *           G = 200 - floor(88*127/256)  = 200 - 43  = 157 */
        {200, 255, 128, 200, 157, 255, "U 最大，B 上溢 clamp"},
        /* d = -128：B = 50 + floor(-58112/256) = 50 - 227 -> clamp 0
         *           G = 50 - floor(-11264/256) = 50 + 44 = 94 */
        { 50,   0, 128,  50,  94,   0, "U 最小，B 下溢 clamp"},
        /* e = -1：R = 100 + floor(-359/256) = 100 - 2 = 98（向下取整，不是 -1）
         *         G = 100 - floor(-183/256) = 100 + 1 = 101 */
        {100, 128, 127,  98, 101, 100, "負數向下取整"},
        /* e = +127：R = 10 + floor(45593/256) = 10 + 178 = 188
         *           G = 10 - floor(23241/256) = 10 - 90 -> clamp 0 */
        { 10, 128, 255, 188,   0,  10, "G 下溢 clamp"},
    };

    printf("\n[黃金 UYVY 模型自我檢查]\n");
    bool ok = true;
    for (unsigned i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
        unsigned char rgb[3];
        golden_pixel(t[i].y, t[i].u, t[i].v, rgb);
        bool pass = (rgb[0] == t[i].r && rgb[1] == t[i].g && rgb[2] == t[i].b);
        printf("  %-22s Y=%3d U=%3d V=%3d -> RGB=(%3d,%3d,%3d) %s",
               t[i].name, t[i].y, t[i].u, t[i].v, rgb[0], rgb[1], rgb[2],
               pass ? "OK" : "<-- 錯誤");
        if (!pass) printf("，預期 (%d,%d,%d)", t[i].r, t[i].g, t[i].b);
        printf("\n");
        ok &= pass;
    }
    printf("  結果：%s\n", ok ? "通過" : "失敗");
    return ok;
}


/* ================================================================
 *  測試圖產生（UYVY）
 *
 *  mode 0: 隨機                       一般性檢查
 *  mode 1: 飽和色（U/V 推到 0 或 255） 觸發 clamp 兩端
 *  mode 2: 水平漸層（Y 與色差都隨 x 變）偵測 ox 錯位
 *  mode 3: 垂直漸層（Y 與色差都隨 y 變）偵測 row_in_block 錯位
 *  mode 4: 彩條（8 條，每條不同 U/V）   偵測色差共用與通道順序
 * ================================================================ */

static void gen_uyvy(unsigned char *uyvy, int w, int h, int mode)
{
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x += 2) {
            unsigned char *q = uyvy + (y * w + x) * 2;   /* 每 2 pixel 4 byte */
            int u, y0, v, y1;
            switch (mode) {
            case 0:
                u = rand() & 0xFF; y0 = rand() & 0xFF;
                v = rand() & 0xFF; y1 = rand() & 0xFF;
                break;
            case 1:
                u  = ((x / 2 + y) & 1) ? 255 : 0;
                v  = ((x / 2 + y) & 2) ? 255 : 0;
                y0 = ((x + y) & 1) ? 235 : 16;
                y1 = 255 - y0;
                break;
            case 2:
                u = (x * 3) & 0xFF;      y0 = x & 0xFF;
                v = (255 - x * 2) & 0xFF; y1 = (x + 1) & 0xFF;
                break;
            case 3:
                u = (y * 3) & 0xFF;      y0 = y & 0xFF;
                v = (255 - y * 2) & 0xFF; y1 = (y * 5) & 0xFF;
                break;
            default: {
                static const unsigned char bar_u[8] = { 90,  54, 166, 128, 128,  90, 202, 240};
                static const unsigned char bar_v[8] = {240,  34, 146, 128,  16, 110, 222, 110};
                int bar = (x * 8) / w;
                u = bar_u[bar]; v = bar_v[bar];
                y0 = y1 = 60 + bar * 20;
                break;
            }
            }
            q[0] = (unsigned char)u;
            q[1] = (unsigned char)y0;
            q[2] = (unsigned char)v;
            q[3] = (unsigned char)y1;
        }
    }
}


/* ================================================================
 *  byte 陣列 <-> 128-bit word
 * ================================================================ */

static void pack_to_words(const unsigned char *src, int nbytes,
                          std::vector<ap_uint<128> > &words)
{
    words.assign(IN_DEPTH, 0);
    for (int i = 0; i < nbytes; i++)
        words[i / 16].range((i % 16) * 8 + 7, (i % 16) * 8) = src[i];
}

static unsigned char word_byte(const std::vector<ap_uint<128> > &words, int i)
{
    return (unsigned char)words[i / 16].range((i % 16) * 8 + 7, (i % 16) * 8);
}


/* ================================================================
 *  單一測試案例
 * ================================================================ */

static bool run_case(int w, int h, int scale, int img_mode, const char *name)
{
    int dst_w = w / scale;
    int dst_h = h / scale;
    int uyvy_bytes = w * h * 2;
    int dst_bytes  = dst_w * dst_h * 3;
    int in_beats   = uyvy_bytes / 16;
    int out_words  = (dst_bytes + 15) / 16;

    printf("\n========================================\n");
    printf("測試案例：%s\n", name);
    printf("  UYVY %dx%d -> RGB %dx%d, scale=%d\n", w, h, dst_w, dst_h, scale);

    /* --- 參數合法性（見 uyvy_resize.h）--- */
    bool ok = true;
    if (w < 1 || w > 4095 || h < 1 || h > 4095) {
        printf("  *** img_w/img_h 超出 ap_uint<12> 範圍 ***\n"); ok = false;
    }
    if (w % 16) { printf("  *** img_w 必須是 16 的倍數 ***\n"); ok = false; }
    if (scale == 3) {
        if (w % 48) { printf("  *** 3 倍時 img_w 必須是 48 的倍數 ***\n"); ok = false; }
        if (h % 3)  { printf("  *** 3 倍時 img_h 必須是 3 的倍數 ***\n");  ok = false; }
    } else {
        if (h % 2)  { printf("  *** 2 倍時 img_h 必須是偶數 ***\n");       ok = false; }
    }
    if (dst_w > OUT_W_MAX) {
        printf("  *** 輸出寬度 %d 超過 %d ***\n", dst_w, OUT_W_MAX); ok = false;
    }
    if (in_beats > IN_DEPTH || out_words > OUT_DEPTH) {
        printf("  *** 超出 depth 設定 (in %d/%d, out %d/%d) ***\n",
               in_beats, IN_DEPTH, out_words, OUT_DEPTH); ok = false;
    }
    if (!ok) {
        printf("  結果：失敗（參數不合法，未呼叫 IP）\n");
        return false;
    }
    printf("  in_beats=%d out_words=%d%s\n", in_beats, out_words,
           (dst_bytes % 16) ? "（最後一個 word 由 pack 收尾補 0）" : "");

    /* --- 產生測試資料與黃金結果 --- */
    std::vector<unsigned char> uyvy(uyvy_bytes);
    std::vector<unsigned char> rgb(w * h * 3);
    std::vector<unsigned char> ref(dst_bytes);

    gen_uyvy(&uyvy[0], w, h, img_mode);
    golden_uyvy2rgb(&uyvy[0], w, h, &rgb[0]);
    golden_resize(&rgb[0], w, h, &ref[0], scale);

    /* --- 執行 IP --- */
    std::vector<ap_uint<128> > in_words;
    pack_to_words(&uyvy[0], uyvy_bytes, in_words);
    std::vector<ap_uint<128> > out_words_buf(OUT_DEPTH, 0);

    ap_uint<1> mode = (scale == 3) ? SCALE_3 : SCALE_2;
    uyvy_resize(&in_words[0], &out_words_buf[0],
                (ap_uint<12>)w, (ap_uint<12>)h, mode);

    /* --- 比對 --- */
    int err_cnt = 0, first_err = -1, max_diff = 0;
    int ch_err[3] = {0, 0, 0};
    for (int i = 0; i < dst_bytes; i++) {
        int d = (int)word_byte(out_words_buf, i) - (int)ref[i];
        if (d < 0) d = -d;
        if (d) {
            err_cnt++;
            ch_err[i % 3]++;
            if (first_err < 0) first_err = i;
            if (d > max_diff) max_diff = d;
        }
    }
    printf("  比對結果：%d / %d byte 不符\n", err_cnt, dst_bytes);

    /* --- 補位區與越界檢查 --- */
    int pad_dirty = 0;
    for (int i = dst_bytes; i < out_words * 16; i++)
        if (word_byte(out_words_buf, i) != 0) pad_dirty++;
    if (pad_dirty)
        printf("  *** 最後一個 word 的補位區有 %d byte 非 0 ***\n", pad_dirty);

    int tail_dirty = 0;
    for (int i = out_words; i < OUT_DEPTH; i++)
        if (out_words_buf[i] != 0) tail_dirty++;
    if (tail_dirty)
        printf("  *** 輸出區尾端有 %d 個 word 被寫入（越界寫）***\n", tail_dirty);

    if (err_cnt) {
        int px = first_err / 3;
        printf("  各通道錯誤數：R=%d G=%d B=%d，最大差值 %d\n",
               ch_err[0], ch_err[1], ch_err[2], max_diff);
        printf("  首個錯誤：byte %d (pixel %d, 座標 %d,%d, 通道 %d)\n",
               first_err, px, px % dst_w, px / dst_w, first_err % 3);
        printf("  附近資料 (ref | got)：\n");
        for (int i = px * 3; i < px * 3 + 12 && i < dst_bytes; i += 3) {
            unsigned char g0 = word_byte(out_words_buf, i);
            unsigned char g1 = word_byte(out_words_buf, i + 1);
            unsigned char g2 = word_byte(out_words_buf, i + 2);
            printf("    px %4d: (%3d,%3d,%3d) | (%3d,%3d,%3d)%s\n", i / 3,
                   ref[i], ref[i+1], ref[i+2], g0, g1, g2,
                   (ref[i]==g0 && ref[i+1]==g1 && ref[i+2]==g2) ? "" : " <--");
        }

        /* 診斷提示 */
        if (ch_err[0] && ch_err[2] && !ch_err[1] && max_diff > 50)
            printf("  提示：R 與 B 錯、G 對 -> 可能是 byte 順序（RGB/BGR）反了\n");
        else if (ch_err[0] == ch_err[1] && ch_err[1] == ch_err[2])
            printf("  提示：三通道錯誤數相同 -> 檢查 ox / row_in_block 或對齊狀態機\n");
        if (max_diff <= 2)
            printf("  提示：差值很小 -> 檢查 uyvy2rgb 的截斷方向或權重量化\n");
        else if (max_diff > 100)
            printf("  提示：差值很大 -> 可能是資料流失或位元對齊錯誤\n");
    }

    bool pass = (err_cnt == 0) && (pad_dirty == 0) && (tail_dirty == 0);
    printf("  結果：%s\n", pass ? "通過" : "失敗");
    return pass;
}


/* ================================================================
 *  main
 * ================================================================ */

int main()
{
    srand(12345);
    int total = 0, passed = 0;

    printf("################################################\n");
    printf("#  uyvy_resize C 驗證\n");
    printf("#  介面：uyvy_resize.h（img_w / img_h / scale_mode）\n");
#ifdef SKIP_BIG_CASES
    printf("#  模式：僅小尺寸（SKIP_BIG_CASES）\n");
#else
    printf("#  模式：完整（含 1920x1080）\n");
#endif
    printf("################################################\n");

    /* ---- 第一部分：黃金模型自我檢查 ---- */
    total++; if (self_check_golden()) passed++;

    /* ---- 第二部分：功能驗證 ---- */
    printf("\n================================================\n");
    printf("第二部分：功能驗證\n");
    printf("================================================\n");

    static const char *pat[5] = {"隨機", "飽和色", "水平漸層", "垂直漸層", "彩條"};
    char name[64];

    /* 最小尺寸：3 倍 48x3 -> 16x1，2 倍 16x2 -> 8x1（觸發 pack 收尾） */
    for (int m = 0; m < 5; m++) {
        snprintf(name, sizeof(name), "3倍 48x3 %s", pat[m]);
        total++; if (run_case(48, 3, 3, m, name)) passed++;
        snprintf(name, sizeof(name), "2倍 16x2 %s（收尾）", pat[m]);
        total++; if (run_case(16, 2, 2, m, name)) passed++;
    }

    /* 小尺寸，多列：檢查 row_in_block 與 line buffer 清零 */
    for (int m = 0; m < 5; m++) {
        snprintf(name, sizeof(name), "3倍 96x36 %s", pat[m]);
        total++; if (run_case(96, 36, 3, m, name)) passed++;
        snprintf(name, sizeof(name), "2倍 64x8 %s", pat[m]);
        total++; if (run_case(64, 8, 2, m, name)) passed++;
    }

    /* 2 倍收尾（輸出 pixel 數不是 16 的倍數）：48x6 -> 24x3 = 72 px */
    total++; if (run_case(48, 6, 2, 3, "2倍 48x6 垂直漸層（收尾）")) passed++;

    /* 中等尺寸 */
    total++; if (run_case(480, 270, 3, 0, "3倍 480x270 隨機")) passed++;
    total++; if (run_case(480, 270, 2, 4, "2倍 480x270 彩條")) passed++;

#ifndef SKIP_BIG_CASES
    /* 實際工作解析度與寬度上限 */
    total++; if (run_case(1920, 1080, 3, 0, "3倍 1920x1080 隨機"))     passed++;
    // total++; if (run_case(1920, 1080, 3, 3, "3倍 1920x1080 垂直漸層")) passed++;
    total++; if (run_case(1920, 1080, 2, 0, "2倍 1920x1080 隨機"))     passed++;
    // total++; if (run_case(1920, 1080, 2, 3, "2倍 1920x1080 垂直漸層")) passed++;
    // total++; if (run_case(2880,   90, 3, 1, "3倍 2880x90 飽和色（輸出寬 960 上限）")) passed++;
#endif

    /* ---- 總結 ---- */
    printf("\n################################################\n");
    printf("#  總計：%d / %d 通過\n", passed, total);
    printf("################################################\n");

    if (passed != total) {
        printf("\n除錯建議：\n");
        printf("  0. 自我檢查就失敗 -> 先修黃金模型，不要看 IP 結果\n");
        printf("  1. 只有最小尺寸錯 -> 檢查對齊狀態機的暖機與 pack 收尾\n");
        printf("  2. 只有 3 倍錯   -> 檢查 12 pixel 對齊（S1/S2 的切片）\n");
        printf("  3. 只有 2 倍錯   -> 檢查 DSP 打包 lane 與 [9:2] 取位\n");
        printf("  4. 飽和色才錯     -> 檢查 clamp_s 或色差符號延伸\n");
        printf("  5. 補位或尾端髒   -> 檢查 out_words 與 pack 收尾遮罩\n");
    }

    return (passed == total) ? 0 : 1;
}