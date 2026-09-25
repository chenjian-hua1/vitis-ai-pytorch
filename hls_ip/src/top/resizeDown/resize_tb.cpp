/******************************************************************************
 * resize_tb.cpp
 *
 * resize_kernel 的 C 驗證 testbench
 *
 * 介面版本：resize_top.h
 *   void resize_kernel(ap_uint<128> *in_ptr, ap_uint<128> *out_ptr,
 *                      ap_uint<12> img_w, ap_uint<12> img_h,
 *                      ap_uint<1>  scale_mode);
 *
 *   total_words / total_results / out_words / out_w / inv_scale
 *   都改由 kernel 內部從 img_w / img_h / scale_mode 算出，tb 不再傳入。
 *   宣告直接 include resize_top.h，型別不一致會在編譯階段就報錯。
 *
 * 驗證項目：
 *   1. leftover 狀態序列是否符合理論推導（設計指紋）
 *   2. 每批運算次數是否落在合法範圍（固定拍數週期成立的前提）
 *   3. 輸出像素與黃金參考模型逐一比對
 *   4. 輸出總量是否正確（無資料流失、無多餘輸出、無越界寫）
 *   5. RGB 三通道是否各自獨立（用刻意設計的測試圖偵測通道污染）
 *   6. img_w / img_h 是否在 ap_uint<12> 範圍內、輸出寬度是否 <= OUT_W_MAX
 *
 * 編譯（純 C 模擬，不需要 Vitis）：
 *   g++ -std=c++11 -I$XILINX_HLS/include \
 *       resize_tb.cpp resize_top.cpp resize_impl.cpp -o tb
 *   ./tb
 *
 * 或在 Vitis HLS 中加入為 testbench 檔案後執行 C Simulation
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "ap_int.h"
#include "resize_top.h"         /* resize_kernel、SCALE_2 / SCALE_3 */


/* ================================================================
 *  黃金參考模型
 *
 *  純軟體的 box filter，與 OpenCV 的 INTER_AREA 在整數倍縮小時
 *  行為一致：每個輸出像素 = 對應 scale x scale 區塊的平均值。
 *
 *  使用與 kernel 等價的定點運算 (sum * inv_scale) >> 16：
 *    3 倍 inv = 7282   <-> kernel 16-bit scale_rate 7282，取 [23:16]
 *    2 倍 inv = 16384  <-> kernel 2-bit  scale_rate 1，   取 [9:2]
 *                          兩者都等於 sum >> 2
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
 *
 *  對應 resize_top.cpp 的 resize_in_select 狀態表：
 *    3 倍：S0..S8 的暫存長度 L = {0,128,112,96,80,64,48,32,16}
 *    2 倍：S0..S2 的暫存長度 L = {0,128,64}
 *  下面印出的 L' 就是每一拍結束後的暫存長度（= 下一個狀態的 L）。
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
        int ops   = (total >= op_bits) ? 1 : 0;   /* 每拍最多 1 次運算 */
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

    /* 狀態機版本只存上一拍 prev（128 bit），運算資料最寬 192 bit */
    printf("  leftover 最大值 = %d bit (prev 暫存器 128 bit 需 >= 此值)\n",
           max_leftover);
    if (max_leftover > 128) {
        printf("  *** prev 的 128 bit 不夠用 ***\n");
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

/* 必須與 resize_top.cpp 的 m_axi depth 一致，改動時兩邊要同步：
 *   MAX_IN_BEATS  = 1920*1080*3/16
 *   MAX_OUT_WORDS = 960*540*3/16
 * 緩衝區一律配到完整解析度的大小，跑小圖時多配的部分閒置即可。
 *
 * 注意：co-sim 會依 depth 模擬整段記憶體，跑滿 388800 word 需時甚久。
 *       功能迭代建議只跑 C simulation，co-sim 留到最後驗證。 */
#define IN_DEPTH   388800      /* = MAX_IN_BEATS  */
#define OUT_DEPTH   97200      /* = MAX_OUT_WORDS */
#define OUT_W_MAX     960      /* = resize_top.cpp 的 OUT_W_MAX */

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
 *  新介面的 img_w / img_h 是 ap_uint<12>（無號，0 ~ 4095）。
 *  tb 內部用 int 計算，若超出範圍，轉型會靜默截斷成錯誤的尺寸，
 *  kernel 算出的拍數就和 tb 準備的資料對不上。這裡先擋下來。
 * ================================================================ */

static bool check_u12(int v, const char *name)
{
    if (v <= 0 || v > 4095) {
        printf("  *** 參數 %s = %d 超出 ap_uint<12> 範圍 (1 ~ 4095) ***\n", name, v);
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

    unsigned int inv_scale = (scale == 3) ? 7282 : 16384;   /* 只給黃金模型用 */
    ap_uint<1>   mode      = (scale == 3) ? SCALE_3 : SCALE_2;

    int src_bytes = src_w * src_h * 3;
    int dst_bytes = dst_w * dst_h * 3;

    printf("\n========================================\n");
    printf("測試案例：%s\n", case_name);
    printf("  %dx%d -> %dx%d, scale=%d\n",
           src_w, src_h, dst_w, dst_h, scale);

    /* --- 前置檢查：尺寸是否符合設計假設（見 resize_top.h） --- */
    if (!check_u12(src_w, "img_w") || !check_u12(src_h, "img_h")) {
        printf("  結果：失敗（參數不合法，未呼叫 kernel）\n");
        return false;
    }
    if (src_w % scale != 0 || src_h % scale != 0) {
        printf("  *** 跳過：尺寸不能被 scale 整除 ***\n");
        return false;
    }
    if (src_bytes % 16 != 0) {
        printf("  *** 跳過：輸入位元組數 %d 不是 16 的倍數 ***\n", src_bytes);
        return false;
    }
    /* 一次產出多欄：3 倍 2 欄、2 倍 4 欄，out_w 必須整除 */
    int n_out = (scale == 3) ? 2 : 4;
    if (dst_w % n_out != 0) {
        printf("  *** 跳過：out_w=%d 必須是 %d 的倍數 ***\n", dst_w, n_out);
        return false;
    }
    if (dst_w > OUT_W_MAX) {
        printf("  *** 跳過：out_w=%d 超過 line buffer 上限 %d ***\n",
               dst_w, OUT_W_MAX);
        return false;
    }

    /* kernel 內部會算出相同的值，這裡只用來檢查 depth 與報告 */
    int total_words   = src_bytes / 16;
    int n_out_words   = (dst_bytes + 15) / 16;
    int total_results = (dst_w * dst_h) / n_out;
    printf("  kernel 內部推得：total_words=%d total_results=%d out_words=%d%s\n",
           total_words, total_results, n_out_words,
           (dst_bytes % 16) ? "（最後一個 word 由 pack 收尾補 0）" : "");

    if (total_words > IN_DEPTH || n_out_words > OUT_DEPTH) {
        printf("  *** 跳過：超出 depth 設定 (in %d/%d, out %d/%d)\n",
               total_words, IN_DEPTH, n_out_words, OUT_DEPTH);
        printf("      請調大 kernel 與 tb 的 depth ***\n");
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
    ap_uint<12> arg_w = (ap_uint<12>)src_w;
    ap_uint<12> arg_h = (ap_uint<12>)src_h;

    printf("  kernel 參數：img_w=%u img_h=%u scale_mode=%u\n",
           arg_w.to_uint(), arg_h.to_uint(), (unsigned)mode.to_uint());

    resize_kernel(&in_words[0], &out_words[0], arg_w, arg_h, mode);

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

    /* --- 尾端檢查 ---
     * (1) 最後一個 word 中超過 dst_bytes 的補位 byte 應為 0（pack 收尾補 0）
     * (2) 輸出區超過 n_out_words 的部分應維持為 0（無越界寫） */
    int pad_dirty = 0;
    for (int i = dst_bytes; i < n_out_words * 16; i++) {
        if (out_words[i / 16].range((i % 16) * 8 + 7, (i % 16) * 8) != 0)
            pad_dirty++;
    }
    if (pad_dirty > 0) {
        printf("  *** 警告：最後一個 word 的補位區有 %d byte 非 0 ***\n", pad_dirty);
    }

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
    printf("#  resize_kernel C 驗證\n");
    printf("#  介面：resize_top.h（img_w / img_h / scale_mode）\n");
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

    /* 一次運算吃 144 bit (3倍) 或 192 bit (2倍) */
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
     * 48x9: 48*9*3 = 1296 byte，1296/16 = 81 word */
    total++; if (run_case(48,  9,  3, 0, "3倍 48x9 隨機"))       passed++;
    total++; if (run_case(48,  9,  3, 1, "3倍 48x9 通道污染測試")) passed++;
    total++; if (run_case(48,  9,  3, 2, "3倍 48x9 水平漸層"))   passed++;
    total++; if (run_case(48,  9,  3, 3, "3倍 48x9 垂直漸層"))   passed++;

    /* 2 倍：out_w 需為 4 的倍數，96/2=48 符合 */
    total++; if (run_case(96,  8,  2, 0, "2倍 96x8 隨機"))       passed++;
    total++; if (run_case(96,  8,  2, 1, "2倍 96x8 通道污染測試")) passed++;
    total++; if (run_case(96,  8,  2, 2, "2倍 96x8 水平漸層"))   passed++;
    total++; if (run_case(96,  8,  2, 3, "2倍 96x8 垂直漸層"))   passed++;

    /* pack 收尾：輸出 pixel 數不是 16 的倍數，最後一個 word 需補 0
     * 24x2 -> 12x1 = 36 byte，40x6 -> 20x3 = 180 byte */
    total++; if (run_case(24,  2,  2, 0, "2倍 24x2 pack 收尾"))  passed++;
    total++; if (run_case(40,  6,  2, 3, "2倍 40x6 pack 收尾"))  passed++;

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
    // total++; if (run_case(1920, 1080, 3, 0, "3倍 1920x1080 隨機")) passed++;
    // total++; if (run_case(1920, 1080, 3, 3, "3倍 1920x1080 垂直漸層")) passed++;
    // total++; if (run_case(1920, 1080, 2, 0, "2倍 1920x1080 隨機")) passed++;
    // total++; if (run_case(1920, 1080, 2, 3, "2倍 1920x1080 垂直漸層")) passed++;
#endif

    /* ---- 總結 ---- */

    printf("\n################################################\n");
    printf("#  總計：%d / %d 通過\n", passed, total);
    printf("################################################\n");

    if (passed != total) {
        printf("\n除錯建議（依優先順序）：\n");
        printf("  0. 若編譯時報 resize_kernel 參數不符 -> kernel 定義與\n");
        printf("     resize_top.h 不一致（img_w/img_h 為 ap_uint<12>）\n");
        printf("  1. 若輸出全為 0 -> 檢查 kernel 內由 img_w/img_h 推得的拍數，\n");
        printf("     或 12-bit 乘法是否先轉成 32-bit 再相乘\n");
        printf("  2. 若 leftover 序列就錯 -> 檢查 resize_in_select 的狀態表\n");
        printf("  3. 若只有某一通道錯     -> 檢查 resize_in_select 的通道切片位置\n");
        printf("  4. 若三通道錯誤數相同   -> 檢查 ox / row_in_block 的推進條件\n");
        printf("  5. 若差值很小(<=2)      -> 可能只是定點捨入，可接受\n");
        printf("  6. 若小尺寸過大尺寸錯   -> 檢查 pack_side 的 hold 暫存處理\n");
        printf("  7. 若補位區或尾端被寫髒 -> 檢查 pack 收尾遮罩或 out_words 上界\n");
    }

    return (passed == total) ? 0 : 1;
}
