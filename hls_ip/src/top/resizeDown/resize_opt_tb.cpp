/******************************************************************************
 * resize_tb.cpp
 *
 * resize_kernel 的 C 驗證 testbench
 *
 * 介面版本：total_words / total_results / out_words / out_w 為 ap_uint<32>
 *           （舊版為 int，兩者 mangled name 不同，混用會在連結階段報錯）
 *
 * 驗證項目：
 *   1. leftover 狀態序列是否符合理論推導（設計指紋）
 *   2. 每批運算次數是否落在合法範圍（固定拍數週期成立的前提）
 *   3. 輸出像素與黃金參考模型逐一比對
 *   4. 輸出總量是否正確（無資料流失、無多餘輸出）
 *   5. RGB 三通道是否各自獨立（用刻意設計的測試圖偵測通道污染）
 *   6. 傳入 kernel 的參數是否在 32-bit 無號範圍內（新介面新增的檢查）
 *
 * 編譯（純 C 模擬，不需要 Vitis）：
 *   g++ -std=c++11 -I$XILINX_HLS/include resize_tb.cpp resize_kernel_single.cpp -o tb
 *   ./tb
 *
 * 或在 Vitis HLS 中加入為 testbench 檔案後執行 C Simulation
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "ap_int.h"

#define SCALE_2 0
#define SCALE_3 1

/* ----------------------------------------------------------------
 *  待測 kernel 的宣告
 *
 *  這份宣告必須與 kernel 的定義逐字相符。若型別不一致，
 *  C++ 會產生不同的 mangled name，連結時出現
 *  "undefined reference to resize_kernel(...)"。
 *  換句話說，這裡的宣告本身就是一道編譯期的介面檢查。
 * ---------------------------------------------------------------- */
void resize_kernel(ap_uint<128> *in_ptr,
                   ap_uint<128> *out_ptr,
                   ap_uint<32>   total_words,
                   ap_uint<32>   total_results,
                   ap_uint<32>   out_words,
                   ap_uint<32>   out_w,
                   ap_uint<1>    scale_mode,
                   ap_uint<16>   inv_scale);


/* ================================================================
 *  黃金參考模型
 *
 *  純軟體的 box filter，與 OpenCV 的 INTER_AREA 在整數倍縮小時
 *  行為一致：每個輸出像素 = 對應 scale x scale 區塊的平均值。
 *
 *  注意：這裡刻意使用與 kernel 相同的定點運算
 *  (sum * inv_scale) >> 16
 *  而非浮點除法，否則會因捨入方式不同而產生 +-1 的差異。
 * ================================================================ */

static void golden_resize(const unsigned char *src, int src_w, int src_h,
                          unsigned char *dst, int scale,
                          unsigned int inv_scale)
{
    int dst_w = src_w / scale;
    int dst_h = src_h / scale;

    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            unsigned int sum_r = 0, sum_g = 0, sum_b = 0;

            for (int dy = 0; dy < scale; dy++) {
                for (int dx = 0; dx < scale; dx++) {
                    int sy = y * scale + dy;
                    int sx = x * scale + dx;
                    const unsigned char *p = src + (sy * src_w + sx) * 3;
                    sum_r += p[0];
                    sum_g += p[1];
                    sum_b += p[2];
                }
            }

            unsigned char *q = dst + (y * dst_w + x) * 3;
            q[0] = (unsigned char)((sum_r * inv_scale) >> 16);
            q[1] = (unsigned char)((sum_g * inv_scale) >> 16);
            q[2] = (unsigned char)((sum_b * inv_scale) >> 16);
        }
    }
}


/* ================================================================
 *  leftover 狀態序列的獨立推導
 *
 *  用純軟體重跑一次狀態機，確認理論序列正確。
 *  這不依賴 kernel，是對設計推導本身的檢查。
 * ================================================================ */

static bool verify_leftover_sequence(int op_bits, const int *expected,
                                      int period, const char *label)
{
    printf("\n[leftover 序列驗證] %s (op_bits=%d)\n", label, op_bits);

    int leftover_len = 0;
    bool ok = true;
    int max_leftover = 0;

    for (int i = 0; i < period; i++) {
        int total = leftover_len + 128;
        int ops   = (total >= op_bits) ? 1 : 0;   /* 本版每批最多 1 次運算 */
        int consumed = ops * op_bits;
        int new_len  = total - consumed;

        printf("  批 %d: L=%3d  total=%3d  ops=%d  L'=%3d",
               i, leftover_len, total, ops, new_len);

        if (new_len != expected[i]) {
            printf("   <-- 錯誤，預期 %d", expected[i]);
            ok = false;
        }
        printf("\n");

        if (ops < 0 || ops > 1) {
            printf("  *** ops = %d 超出 [0,1] ***\n", ops);
            ok = false;
        }

        if (new_len > max_leftover) max_leftover = new_len;
        leftover_len = new_len;
    }

    printf("  週期結束時 L = %d (應為 0)\n", leftover_len);
    if (leftover_len != 0) ok = false;

    printf("  leftover 最大值 = %d bit (window 需 >= %d bit)\n",
           max_leftover, max_leftover + 128);
    if (max_leftover + 128 > 256) {
        printf("  *** window 的 256 bit 不夠用 ***\n");
        ok = false;
    }

    printf("  結果：%s\n", ok ? "通過" : "失敗");
    return ok;
}


/* ================================================================
 *  測試圖產生
 *
 *  mode 0: 隨機（一般性檢查）
 *  mode 1: RGB 通道刻意錯開（偵測通道污染）
 *          R 全部給大值、G 中值、B 小值，
 *          若有 carry 污染會立刻顯現
 *  mode 2: 水平漸層（偵測 ox 錯位）
 *  mode 3: 垂直漸層（偵測 row_in_block 錯位）
 * ================================================================ */

static void gen_test_image(unsigned char *img, int w, int h, int mode)
{
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char *p = img + (y * w + x) * 3;
            switch (mode) {
            case 0:
                p[0] = rand() & 0xFF;
                p[1] = rand() & 0xFF;
                p[2] = rand() & 0xFF;
                break;
            case 1:
                p[0] = 250;                    /* R 接近飽和，易觸發進位 */
                p[1] = 128;
                p[2] = 5;
                break;
            case 2:
                p[0] = (unsigned char)(x & 0xFF);
                p[1] = (unsigned char)((x * 2) & 0xFF);
                p[2] = (unsigned char)((x * 3) & 0xFF);
                break;
            case 3:
                p[0] = (unsigned char)(y & 0xFF);
                p[1] = (unsigned char)((y * 2) & 0xFF);
                p[2] = (unsigned char)((y * 3) & 0xFF);
                break;
            }
        }
    }
}


/* ================================================================
 *  byte 陣列 <-> 128-bit word 的打包/解包
 * ================================================================ */

/* 必須與 kernel 的 #pragma HLS INTERFACE depth 一致，改動時兩邊要同步。
 * 緩衝區一律配到完整解析度的大小，跑小圖時多配的部分閒置即可。
 *
 * 注意：co-sim 會依 depth 模擬整段記憶體，跑滿 388800 word 需時甚久。
 *       功能迭代建議只跑 C simulation，co-sim 留到最後驗證。 */
#define IN_DEPTH   388800      /* 1920*1080*3/16 */
#define OUT_DEPTH   97200      /* 960*540*3/16，2 倍模式較大者 */

static void pack_to_words(const unsigned char *src, int nbytes,
                          std::vector<ap_uint<128> > &words)
{
    int nwords = (nbytes + 15) / 16;
    if (nwords < IN_DEPTH) nwords = IN_DEPTH;
    words.assign(nwords, 0);

    for (int i = 0; i < nbytes; i++) {
        int w = i / 16;
        int b = i % 16;
        words[w].range(b * 8 + 7, b * 8) = src[i];
    }
}

static void unpack_from_words(const std::vector<ap_uint<128> > &words,
                              unsigned char *dst, int nbytes)
{
    for (int i = 0; i < nbytes; i++) {
        int w = i / 16;
        int b = i % 16;
        dst[i] = (unsigned char)words[w].range(b * 8 + 7, b * 8);
    }
}


/* ================================================================
 *  參數合法性檢查
 *
 *  新介面的四個參數是 ap_uint<32>（無號）。tb 內部仍用 int 計算，
 *  若不小心算出負值或超過 2^32-1，隱式轉換會靜默回捲成巨大的正數，
 *  在 kernel 裡變成幾乎跑不完的迴圈。這裡先擋下來。
 * ================================================================ */

static bool check_u32(long long v, const char *name)
{
    if (v < 0 || v > 4294967295LL) {
        printf("  *** 參數 %s = %lld 超出 ap_uint<32> 範圍 ***\n", name, v);
        return false;
    }
    if (v == 0) {
        printf("  *** 參數 %s = 0，kernel 不會產生任何輸出 ***\n", name);
        return false;
    }
    return true;
}


/* ================================================================
 *  單一測試案例
 * ================================================================ */

static bool run_case(int src_w, int src_h, int scale, int img_mode,
                     const char *case_name)
{
    int dst_w = src_w / scale;
    int dst_h = src_h / scale;

    unsigned int inv_scale = (scale == 3) ? 7282 : 16384;
    ap_uint<1>   mode      = (scale == 3) ? SCALE_3 : SCALE_2;

    int src_bytes = src_w * src_h * 3;
    int dst_bytes = dst_w * dst_h * 3;

    printf("\n========================================\n");
    printf("測試案例：%s\n", case_name);
    printf("  %dx%d -> %dx%d, scale=%d, inv=%u\n",
           src_w, src_h, dst_w, dst_h, scale, inv_scale);

    /* --- 前置檢查：尺寸是否符合設計假設 --- */
    if (src_w % scale != 0 || src_h % scale != 0) {
        printf("  *** 跳過：尺寸不能被 scale 整除 ***\n");
        return false;
    }
    if (src_bytes % 16 != 0) {
        printf("  *** 跳過：輸入位元組數 %d 不是 16 的倍數 ***\n", src_bytes);
        return false;
    }

    int total_words = src_bytes / 16;
    int n_out_words = (dst_bytes + 15) / 16;
    printf("  total_words = %d, 預期輸出 %d word\n", total_words, n_out_words);

    if (total_words > IN_DEPTH || n_out_words > OUT_DEPTH) {
        printf("  *** 跳過：超出 depth 設定 (in %d/%d, out %d/%d)\n",
               total_words, IN_DEPTH, n_out_words, OUT_DEPTH);
        printf("      請調大 kernel 與 tb 的 IN_DEPTH/OUT_DEPTH ***\n");
        return false;
    }
    /* 本版一次產出多欄：3 倍 2 欄、2 倍 4 欄，out_w 必須整除 */
    int n_out = (scale == 3) ? 2 : 4;
    if (dst_w % n_out != 0) {
        printf("  *** 跳過：out_w=%d 必須是 %d 的倍數 ***\n", dst_w, n_out);
        return false;
    }

    /* total_results = 運算次數：3 倍一次產出 2 欄、2 倍一次產出 4 欄 */
    int total_results = (dst_w * dst_h) / n_out;

    /* --- 介面參數檢查（ap_uint<32> 為無號，先擋掉負值/溢位） --- */
    bool arg_ok = true;
    arg_ok &= check_u32(total_words,   "total_words");
    arg_ok &= check_u32(total_results, "total_results");
    arg_ok &= check_u32(n_out_words,   "out_words");
    arg_ok &= check_u32(dst_w,         "out_w");
    if (!arg_ok) {
        printf("  結果：失敗（參數不合法，未呼叫 kernel）\n");
        return false;
    }

    /* --- 產生測試資料 --- */
    std::vector<unsigned char> src(src_bytes);
    std::vector<unsigned char> ref(dst_bytes);
    std::vector<unsigned char> got(dst_bytes, 0);

    gen_test_image(&src[0], src_w, src_h, img_mode);
    golden_resize(&src[0], src_w, src_h, &ref[0], scale, inv_scale);

    /* --- 打包成 128-bit word --- */
    std::vector<ap_uint<128> > in_words;
    pack_to_words(&src[0], src_bytes, in_words);

    /* 配置到 OUT_DEPTH 大小以配合 pragma 的 depth 設定 */
    std::vector<ap_uint<128> > out_words(OUT_DEPTH, 0);

    /* --- 執行 kernel --- */
    /* 四個純量參數都明確轉成 ap_uint<32>，避免隱式轉換遮蔽問題 */
    ap_uint<32> arg_total_words   = (ap_uint<32>)total_words;
    ap_uint<32> arg_total_results = (ap_uint<32>)total_results;
    ap_uint<32> arg_out_words     = (ap_uint<32>)n_out_words;
    ap_uint<32> arg_out_w         = (ap_uint<32>)dst_w;

    printf("  kernel 參數：total_words=%u total_results=%u "
           "out_words=%u out_w=%u mode=%u inv=%u\n",
           arg_total_words.to_uint(), arg_total_results.to_uint(),
           arg_out_words.to_uint(),   arg_out_w.to_uint(),
           (unsigned)mode.to_uint(),  inv_scale);

    resize_kernel(&in_words[0], &out_words[0],
                  arg_total_words,
                  arg_total_results,
                  arg_out_words,
                  arg_out_w,
                  mode,
                  (ap_uint<16>)inv_scale);

    unpack_from_words(out_words, &got[0], dst_bytes);

    /* --- 比對 --- */
    int  err_cnt   = 0;
    int  first_err = -1;
    int  max_diff  = 0;
    int  ch_err[3] = {0, 0, 0};

    for (int i = 0; i < dst_bytes; i++) {
        int d = (int)got[i] - (int)ref[i];
        if (d < 0) d = -d;
        if (d != 0) {
            err_cnt++;
            ch_err[i % 3]++;
            if (first_err < 0) first_err = i;
            if (d > max_diff) max_diff = d;
        }
    }

    printf("  比對結果：%d / %d byte 不符\n", err_cnt, dst_bytes);

    /* --- 尾端檢查：輸出區超過 dst_bytes 的部分應維持為 0 --- */
    int tail_dirty = 0;
    for (int i = n_out_words; i < OUT_DEPTH; i++) {
        if (out_words[i] != 0) { tail_dirty++; }
    }
    if (tail_dirty > 0) {
        printf("  *** 警告：輸出區尾端有 %d 個 word 被寫入（越界寫）***\n",
               tail_dirty);
    }

    if (err_cnt > 0) {
        printf("  各通道錯誤數：R=%d G=%d B=%d\n",
               ch_err[0], ch_err[1], ch_err[2]);
        printf("  最大差值：%d\n", max_diff);

        int px = first_err / 3;
        printf("  首個錯誤：byte %d (pixel %d, 座標 %d,%d, 通道 %d)\n",
               first_err, px, px % dst_w, px / dst_w, first_err % 3);

        /* 印出首個錯誤附近的資料 */
        printf("  附近資料 (ref | got)：\n");
        int start = (first_err / 3) * 3;
        for (int i = start; i < start + 12 && i < dst_bytes; i += 3) {
            printf("    px %4d: (%3d,%3d,%3d) | (%3d,%3d,%3d) %s\n",
                   i / 3,
                   ref[i], ref[i+1], ref[i+2],
                   got[i], got[i+1], got[i+2],
                   (ref[i]==got[i] && ref[i+1]==got[i+1] && ref[i+2]==got[i+2])
                       ? "" : "<--");
        }

        /* 診斷提示 */
        if (ch_err[0] > 0 && ch_err[1] == 0 && ch_err[2] == 0)
            printf("  提示：只有 R 錯 -> 可能是通道切片位置錯誤\n");
        else if (ch_err[0] == ch_err[1] && ch_err[1] == ch_err[2])
            printf("  提示：三通道錯誤數相同 -> 可能是 ox/row_in_block 錯位\n");
        if (max_diff > 100)
            printf("  提示：差值很大 -> 可能是資料流失或位元對齊錯誤\n");
        else if (max_diff <= 2)
            printf("  提示：差值很小 -> 可能只是定點捨入差異\n");
    }

    bool pass = (err_cnt == 0) && (tail_dirty == 0);
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
    printf("#  resize_kernel C 驗證\n");
    printf("#  介面：ap_uint<32> 純量參數版\n");
#ifdef SKIP_BIG_CASES
    printf("#  模式：僅小尺寸（定義 SKIP_BIG_CASES 已跳過大案例）\n");
#else
    printf("#  模式：完整（含 1920x1080，C sim 約數十秒）\n");
#endif
    printf("################################################\n");

    /* ---- 第一部分：leftover 狀態序列 ---- */

    printf("\n================================================\n");
    printf("第一部分：leftover 狀態序列（設計推導檢查）\n");
    printf("================================================\n");

    /* 本版一次運算吃 144 bit (3倍) 或 192 bit (2倍) */
    int exp_s3[9] = {128, 112, 96, 80, 64, 48, 32, 16, 0};
    int exp_s2[3] = {128, 64, 0};

    total++;
    if (verify_leftover_sequence(144, exp_s3, 9, "3 倍模式")) passed++;

    total++;
    if (verify_leftover_sequence(192, exp_s2, 3, "2 倍模式")) passed++;

    /* ---- 第二部分：功能驗證 ---- */

    printf("\n================================================\n");
    printf("第二部分：功能驗證\n");
    printf("================================================\n");

    /* 小尺寸，容易人工檢查。
     * 48x9: 48 是 LCM(128,72)/24 = 48 pixel 的對齊週期，
     *       48*9*3 = 1296 byte，1296/16 = 81 word */
    total++; if (run_case(48,  9,  3, 0, "3倍 48x9 隨機"))       passed++;
    total++; if (run_case(48,  9,  3, 1, "3倍 48x9 通道污染測試")) passed++;
    total++; if (run_case(48,  9,  3, 2, "3倍 48x9 水平漸層"))   passed++;
    total++; if (run_case(48,  9,  3, 3, "3倍 48x9 垂直漸層"))   passed++;

    /* 16 pixel 是 2 倍模式的對齊週期 */
    /* 2 倍：out_w 需為 4 的倍數，96/2=48 符合 */
    total++; if (run_case(96,  8,  2, 0, "2倍 96x8 隨機"))       passed++;
    total++; if (run_case(96,  8,  2, 1, "2倍 96x8 通道污染測試")) passed++;
    total++; if (run_case(96,  8,  2, 2, "2倍 96x8 水平漸層"))   passed++;
    total++; if (run_case(96,  8,  2, 3, "2倍 96x8 垂直漸層"))   passed++;

    /* 中等尺寸 */
    total++; if (run_case(192, 27, 3, 0, "3倍 192x27 隨機"))     passed++;
    total++; if (run_case(192, 24, 2, 0, "2倍 192x24 隨機"))     passed++;

    /* 大尺寸案例
     * C simulation 約數十秒；co-sim 這幾個會跑很久，
     * 需要時可用 SKIP_BIG_CASES 開關關掉。 */
#ifndef SKIP_BIG_CASES
    total++; if (run_case(480,  270, 3, 0, "3倍 480x270 隨機"))     passed++;
    total++; if (run_case(480,  270, 2, 0, "2倍 480x270 隨機"))     passed++;

    /* 實際工作解析度 */
    total++; if (run_case(1920, 1080, 3, 0, "3倍 1920x1080 隨機")) passed++;
    total++; if (run_case(1920, 1080, 3, 3, "3倍 1920x1080 垂直漸層")) passed++;
    total++; if (run_case(1920, 1080, 2, 0, "2倍 1920x1080 隨機")) passed++;
    total++; if (run_case(1920, 1080, 2, 3, "2倍 1920x1080 垂直漸層")) passed++;
#endif

    /* ---- 總結 ---- */

    printf("\n################################################\n");
    printf("#  總計：%d / %d 通過\n", passed, total);
    printf("################################################\n");

    if (passed != total) {
        printf("\n除錯建議（依優先順序）：\n");
        printf("  0. 若連結時報 undefined reference -> kernel 定義與此宣告的\n");
        printf("     型別不一致，四個純量參數必須都是 ap_uint<32>\n");
        printf("  1. 若輸出全為 0 -> 檢查 kernel 內是否誤把 ap_uint<32> 拿去\n");
        printf("     和有號數比較，或迴圈上界被截斷\n");
        printf("  2. 若 leftover 序列就錯 -> 檢查 op_bits 與 do_op 的計算\n");
        printf("  3. 若只有某一通道錯     -> 檢查 px0~px3 的 .range() 切片位置\n");
        printf("  4. 若三通道錯誤數相同   -> 檢查 ox / row_in_block 的推進條件\n");
        printf("  5. 若差值很小(<=2)      -> 可能只是定點捨入，可接受\n");
        printf("  6. 若小尺寸過大尺寸錯   -> 檢查 acc/acc_len 的殘餘處理\n");
        printf("  7. 若尾端被寫髒         -> out_words 上界或位址遞增條件有誤\n");
    }

    return (passed == total) ? 0 : 1;
}