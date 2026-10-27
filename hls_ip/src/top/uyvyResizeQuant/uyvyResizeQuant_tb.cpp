/******************************************************************************
 * uyvyResizeQuant_tb.cpp
 *
 * uyvyResizeQuant 的 C simulation / co-sim testbench
 *
 * Golden model（純軟體，逐 pixel 計算，與硬體的 tile 結構無關）：
 *   1. UYVY -> RGB      直接呼叫 uyvy2rgb_impl 的 cvt_pair（與硬體共用同一顆運算）
 *   2. Resize           N x N box filter（N = 2 或 3）
 *                         3 倍：(sum9 * 7282) >> 16      （scale_rate = 65536/9，16 bit 小數）
 *                         2 倍： sum4 >> 2               （scale_rate = 1，2 bit 小數）
 *                         取低 8 bit
 *   3. Quantize         每 2 個 pixel（48 bit）呼叫一次 quantize_core，pixel0 在低位
 *   4. Pack             byte 順序 R,G,B（byte0 = R），每 16 byte 一個 128-bit word，尾端補 0
 *
 * 若出現「全面性」的不一致，優先檢查這份 golden 的假設：
 *   - resize 公式與 SR_S3 / FRAC（見 golden_resize）
 *   - 通道 byte 順序（見 golden_uyvy2rgb / golden_quantize）
 *
 * 檢查項目
 *   - 每個輸出 word 逐 byte 比對（含尾端補 0 的 byte）
 *   - 輸出 out_words 之後直到 depth 結尾都必須仍是毒值（偵測多寫）
 *   - 輸出 buffer 預先填毒值，漏寫的 word 會被抓到
 *
 * 【重要】co-sim 的 buffer 大小
 *   co-sim wrapper 會依 top 的 m_axi depth= 把整個陣列複製進 / 出 RTL 模擬，
 *   不管實際影像多大。所以 TB 的 in / out buffer 一律配置到 depth 大小，
 *   否則 wrapper 會讀寫 buffer 之外的記憶體而 segfault（ENTER_WRAPC）。
 *   TB_MAX_IN_BEATS / TB_MAX_OUT_WORDS 必須 >= top 的 MAX_IN_BEATS / MAX_OUT_WORDS。
 *
 * 尺寸限制（與 DUT 相同）
 *   3 倍：img_w 為 48 的倍數、img_h 為 3 的倍數
 *   2 倍：img_w 為 8 的倍數（out_w 為 4 的倍數）、img_h 為 2 的倍數
 *
 * 全尺寸測項：在 TB Cflags 加 -DTB_FULL_HD 即可打開
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>

#include "ap_int.h"
#include "../../impl/resize_impl.h"        /* SCALE_2 / SCALE_3 */
#include "uyvyResizeQuant_top.h"
#include "../../impl/uyvy2rgb_impl.h"        /* cvt_pair */
#include "../../impl/quantize_impl.h"        /* quantize_core */

using std::vector;

#define TB_SR_S3     7282

/* 必須 >= top 的 depth（MAX_IN_BEATS / MAX_OUT_WORDS） */
#ifndef TB_MAX_IN_BEATS
#define TB_MAX_IN_BEATS   (1920 * 1080 / 8)          /* 259200 */
#endif
#ifndef TB_MAX_OUT_WORDS
#define TB_MAX_OUT_WORDS  (960 * 540 * 3 / 16)       /* 97200  */
#endif

static const ap_uint<1> MODE_3 = (ap_uint<1>)((int)SCALE_3);
static const ap_uint<1> MODE_2 = (ap_uint<1>)((int)SCALE_2);

/* ---------------------------------------------------------------- 輸入產生 */

enum Pattern { PAT_RANDOM = 0, PAT_ZERO, PAT_FULL, PAT_GRADIENT };

static const char *pat_name(int p)
{
    switch (p) {
    case PAT_RANDOM:   return "random";
    case PAT_ZERO:     return "all-0x00";
    case PAT_FULL:     return "all-0xFF";
    default:           return "gradient";
    }
}

/* UYVY byte 流：每 4 byte = {U, Y0, V, Y1}，row-major，共 w*h*2 byte */
static vector<uint8_t> gen_uyvy(int w, int h, int pat, unsigned seed)
{
    vector<uint8_t> b((size_t)w * h * 2);
    srand(seed);
    for (size_t i = 0; i < b.size(); i++) {
        switch (pat) {
        case PAT_RANDOM:   b[i] = (uint8_t)(rand() & 0xFF);                 break;
        case PAT_ZERO:     b[i] = 0x00;                                     break;
        case PAT_FULL:     b[i] = 0xFF;                                     break;
        default:           b[i] = (uint8_t)((i * 7 + i / (w * 2) * 13) & 0xFF); break;
        }
    }
    return b;
}

/* byte 流 -> 128-bit word（little-endian：byte k 放在 [8k+7:8k]），尾端補 0 */
static vector<ap_uint<128> > bytes_to_words(const vector<uint8_t> &b)
{
    size_t n = (b.size() + 15) / 16;
    vector<ap_uint<128> > w(n, 0);
    for (size_t i = 0; i < n; i++) {
        ap_uint<128> x = 0;
        for (int k = 0; k < 16; k++) {
            size_t idx = i * 16 + k;
            uint8_t v = (idx < b.size()) ? b[idx] : 0;
            x.range(8 * k + 7, 8 * k) = v;
        }
        w[i] = x;
    }
    return w;
}

/* ---------------------------------------------------------------- Golden */

/* UYVY -> RGB，輸出 byte 順序 R,G,B */
static vector<uint8_t> golden_uyvy2rgb(const vector<uint8_t> &uyvy, int w, int h)
{
    vector<uint8_t> rgb((size_t)w * h * 3);
    for (size_t j = 0; j < (size_t)w * h / 2; j++) {
        ap_uint<8> u  = uyvy[4 * j + 0];
        ap_uint<8> y0 = uyvy[4 * j + 1];
        ap_uint<8> v  = uyvy[4 * j + 2];
        ap_uint<8> y1 = uyvy[4 * j + 3];
        ap_uint<24> p0, p1;
        cvt_pair(y0, y1, u, v, p0, p1);              /* {R,G,B} = p[23:16], p[15:8], p[7:0] */
        ap_uint<24> px[2] = { p0, p1 };
        for (int k = 0; k < 2; k++) {
            size_t o = (2 * j + k) * 3;
            rgb[o + 0] = (uint8_t)px[k].range(23, 16).to_uint();   /* R */
            rgb[o + 1] = (uint8_t)px[k].range(15,  8).to_uint();   /* G */
            rgb[o + 2] = (uint8_t)px[k].range( 7,  0).to_uint();   /* B */
        }
    }
    return rgb;
}

static vector<uint8_t> golden_resize(const vector<uint8_t> &rgb, int w, int h, int n,
                                     int &ow, int &oh)
{
    ow = w / n;
    oh = h / n;
    vector<uint8_t> out((size_t)ow * oh * 3);
    for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++)
            for (int c = 0; c < 3; c++) {
                unsigned sum = 0;
                for (int dy = 0; dy < n; dy++)
                    for (int dx = 0; dx < n; dx++)
                        sum += rgb[((size_t)(oy * n + dy) * w + (ox * n + dx)) * 3 + c];
                unsigned val = (n == 3) ? ((sum * TB_SR_S3) >> 16) : (sum >> 2);
                out[((size_t)oy * ow + ox) * 3 + c] = (uint8_t)(val & 0xFF);
            }
    return out;
}

/* 每 2 pixel 呼叫一次 quantize_core（奇數個 pixel 時最後一個補 0，與 quantize_kernel 一致） */
static vector<uint8_t> golden_quantize(const vector<uint8_t> &img)
{
    vector<uint8_t> out(img.size());
    size_t npix = img.size() / 3;
    for (size_t p = 0; p < npix; p += 2) {
        ap_uint<48> in48 = 0, out48;
        int cnt = (p + 1 < npix) ? 2 : 1;
        for (int k = 0; k < cnt; k++)
            for (int c = 0; c < 3; c++)
                in48.range(24 * k + 8 * c + 7, 24 * k + 8 * c) = img[(p + k) * 3 + c];
        quantize_core(in48, out48);
        for (int k = 0; k < cnt; k++)
            for (int c = 0; c < 3; c++)
                out[(p + k) * 3 + c] = (uint8_t)out48.range(24 * k + 8 * c + 7, 24 * k + 8 * c).to_uint();
    }
    return out;
}

/* 比對一個輸出 buffer；回傳錯誤數（逐 byte 錯誤 + depth 結尾前被改寫的 word 數） */
static int check_buffer(const char *tag,
                        const vector<ap_uint<128> > &out,
                        const vector<ap_uint<128> > &gold,
                        size_t nbytes, int ow,
                        const ap_uint<128> &poison)
{
    const size_t out_words = gold.size();
    int errors = 0;
    const int MAX_PRINT = 8;
    for (size_t i = 0; i < out_words; i++) {
        for (int k = 0; k < 16; k++) {
            unsigned d = out[i].range(8 * k + 7, 8 * k).to_uint();
            unsigned g = gold[i].range(8 * k + 7, 8 * k).to_uint();
            if (d != g) {
                if (errors < MAX_PRINT) {
                    size_t bi = i * 16 + k;
                    if (bi < nbytes) {
                        size_t pix = bi / 3;
                        printf("    [%s] MISMATCH word %zu byte %2d (out pixel x=%zu y=%zu ch=%c): dut=0x%02X gold=0x%02X\n",
                               tag, i, k, pix % ow, pix / ow, "RGB"[bi % 3], d, g);
                    } else {
                        printf("    [%s] MISMATCH word %zu byte %2d (tail padding): dut=0x%02X gold=0x%02X\n",
                               tag, i, k, d, g);
                    }
                }
                errors++;
            }
        }
    }
    /* out_words 之後直到 depth 結尾必須仍是毒值 */
    int over = 0;
    for (size_t g = out_words; g < out.size(); g++)
        if (out[g] != poison) over++;
    if (over)
        printf("    [%s] OVERWRITE: out_words=%zu 之後有 %d 個 word 被改寫\n", tag, out_words, over);
    return errors + over;
}

/* ---------------------------------------------------------------- 單一測項 */

struct TestCase {
    int         w, h;
    int         n;          /* 2 或 3 */
    int         pat;
    unsigned    seed;
};

static bool run_test(const TestCase &t, int idx)
{
    const int w = t.w, h = t.h, n = t.n;

    /* ---- 輸入：實際資料放前面，其餘補 0，總長 = depth ---- */
    vector<uint8_t> uyvy = gen_uyvy(w, h, t.pat, t.seed);
    vector<ap_uint<128> > in = bytes_to_words(uyvy);
    if (in.size() > (size_t)TB_MAX_IN_BEATS) {
        printf("[FAIL] #%d %dx%d (%dx): 輸入 %zu words 超過 depth %d，請放大 MAX_IN_BEATS\n",
               idx, w, h, n, in.size(), (int)TB_MAX_IN_BEATS);
        return false;
    }
    in.resize(TB_MAX_IN_BEATS, 0);

    /* ---- Golden ---- */
    vector<uint8_t> rgb = golden_uyvy2rgb(uyvy, w, h);
    int ow, oh;
    vector<uint8_t> rs  = golden_resize(rgb, w, h, n, ow, oh);
    vector<uint8_t> q   = golden_quantize(rs);
    vector<ap_uint<128> > gold = bytes_to_words(q);
    const size_t out_words = gold.size();
    if (out_words > (size_t)TB_MAX_OUT_WORDS) {
        printf("[FAIL] #%d %dx%d (%dx): 輸出 %zu words 超過 depth %d，請放大 MAX_OUT_WORDS\n",
               idx, w, h, n, out_words, (int)TB_MAX_OUT_WORDS);
        return false;
    }

    /* ---- DUT：兩個輸出 buffer 都是 depth 大小，全部預填毒值 ---- */
    ap_uint<128> poison = 0;
    for (int k = 0; k < 16; k++) poison.range(8 * k + 7, 8 * k) = 0xA5;
    vector<ap_uint<128> > out_rs(TB_MAX_OUT_WORDS, poison);   /* resize 結果（gmem1）*/
    vector<ap_uint<128> > out_q (TB_MAX_OUT_WORDS, poison);   /* quantize 結果（gmem2）*/

    uyvyResizeQuant(in.data(), out_rs.data(), out_q.data(), w, h, (n == 3) ? MODE_3 : MODE_2);

    /* ---- 比對：resize 對 resize golden，quantize 對 quantize golden ---- */
    vector<ap_uint<128> > gold_rs = bytes_to_words(rs);
    int err_rs = check_buffer("resize", out_rs, gold_rs, rs.size(), ow, poison);
    int err_q  = check_buffer("quant ", out_q,  gold,    q.size(),  ow, poison);

    bool pass = (err_rs == 0) && (err_q == 0);
    printf("[%s] #%d %dx%d -> %dx%d (%dx) %-9s words=%zu  err_resize=%d err_quant=%d\n",
           pass ? "PASS" : "FAIL", idx, w, h, ow, oh, n, pat_name(t.pat), out_words, err_rs, err_q);
    return pass;
}

/* ---------------------------------------------------------------- main */

int main()
{
    vector<TestCase> tests;

    /* ---- 2 倍 ---- */
    tests.push_back({   8,  2, 2, PAT_RANDOM,   1 });   /* out 4x1：單一 quad 欄 + 尾端補 0 */
    tests.push_back({   8,  4, 2, PAT_RANDOM,   2 });   /* out 4x2：quad_w = 1 */
    tests.push_back({  16,  4, 2, PAT_RANDOM,   3 });
    tests.push_back({  64, 32, 2, PAT_RANDOM,   4 });
    tests.push_back({  64, 32, 2, PAT_ZERO,     0 });
    tests.push_back({  64, 32, 2, PAT_FULL,     0 });
    tests.push_back({  64, 32, 2, PAT_GRADIENT, 0 });
    tests.push_back({ 128, 48, 2, PAT_RANDOM,   5 });
    tests.push_back({ 200, 10, 2, PAT_RANDOM,   6 });   /* out_w = 100，quad_w = 25 */

    /* ---- 3 倍 ---- */
    tests.push_back({  48,  3, 3, PAT_RANDOM,   7 });   /* out 16x1 */
    tests.push_back({  48,  6, 3, PAT_RANDOM,   8 });
    tests.push_back({  96,  9, 3, PAT_RANDOM,   9 });
    tests.push_back({  96,  9, 3, PAT_ZERO,     0 });
    tests.push_back({  96,  9, 3, PAT_FULL,     0 });   /* 累加極值：255*9*7282>>16 */
    tests.push_back({  96,  9, 3, PAT_GRADIENT, 0 });
    tests.push_back({ 144, 12, 3, PAT_RANDOM,  10 });
    tests.push_back({ 192, 30, 3, PAT_RANDOM,  11 });

// #ifdef TB_FULL_HD
    /* ---- 1920x1080（2 倍剛好用滿 depth：quad_w = 240、97200 words）---- */
    tests.push_back({ 1920, 1080, 2, PAT_RANDOM, 100 });
    tests.push_back({ 1920, 1080, 3, PAT_RANDOM, 101 });
//     tests.push_back({ 1920, 1080, 2, PAT_FULL,   0   });
//     tests.push_back({ 1920, 1080, 3, PAT_FULL,   0   });
// #endif

    int fails = 0;
    for (size_t i = 0; i < tests.size(); i++)
        if (!run_test(tests[i], (int)i)) fails++;

    printf("\n==== %zu tests, %d failed ====\n", tests.size(), fails);
    return fails ? 1 : 0;
}