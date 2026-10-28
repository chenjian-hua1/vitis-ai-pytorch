/******************************************************************************
 * uyvyResizeQuant_tb.cpp
 *
 * uyvyResizeQuant 的 C simulation / co-sim testbench
 *
 * Golden model（純軟體，逐 pixel 計算，與硬體的 tile 結構無關）：
 *   1. UYVY -> RGB      直接呼叫 uyvy2rgb_impl 的 cvt_pair（與硬體共用同一顆運算）
 *   2. Resize           N x N box filter（N = 2 或 3）
 *                       3 倍：(sum9 * 7282) >> 16      （scale_rate = 65536/9，16 bit 小數）
 *                       2 倍： sum4 >> 2               （scale_rate = 1，2 bit 小數）
 *                       取低 8 bit
 *   3. Quantize         與 KV260 上 norm_and_fix_letterbox（一般計算路徑 double_kernel）逐字相同：
 *                         fp    = 2^fix_point
 *                         scale = 1/(std*255) * fp,  bias = -mean/std * fp   （每個通道不同）
 *                         y     = clamp(v * scale + bias, -128, 127) → 往 0 截斷成 int8
 *                       fix_point 用 TB_FIX_POINT（= xmodel 輸入的 fix_point，預設 6）
 *                       【不使用】DUT 的 quantize_lut.h 計算 golden，避免「拿自己驗證自己」
 *   4. Pack             byte 順序 R,G,B（byte0 = R），每 16 byte 一個 128-bit word，尾端補 0
 *
 * 檢查項目
 *   0. LUT 自我檢查（不跑 DUT）：
 *        - DUT 的 kFixPoint 是否等於 TB_FIX_POINT
 *        - DUT 的 kLut（768 項）是否與 norm_and_fix golden 逐項相同
 *          不同時會掃 fix_point 0..10 找出 kLut 實際對應的值，並判斷是否 R/B 對調
 *        - golden 的融合乘加（aarch64 -O2 的 fmla）與一般乘加結果是否相同
 *   1. 每個輸出 word 逐 byte 比對（含尾端補 0 的 byte），quant 另外統計 R/G/B 錯誤數
 *   2. 輸出 out_words 之後直到 depth 結尾都必須仍是毒值（偵測多寫）
 *   3. 輸出 buffer 預先填毒值，漏寫的 word 會被抓到
 *   4. 結尾印出 quant golden 輸入值的覆蓋率（每通道 0..255 實際測到幾個值）
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
 * 編譯選項（TB Cflags）
 *   -DTB_FIX_POINT=N   xmodel 輸入的 fix_point（預設 6）
 *   -DTB_NO_FULL_HD    關掉 1920x1080 測項（co-sim 時可縮短時間）
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>

#include "ap_int.h"
#include "../../impl/resize_impl.h"        /* SCALE_2 / SCALE_3 */
#include "uyvyResizeQuant_top.h"
#include "../../impl/uyvy2rgb_impl.h"      /* cvt_pair */
#include "../../impl/quantize_lut.h"       /* 只用於 LUT 自我檢查（kLut / kFixPoint），golden 不用它 */

using std::vector;

#define TB_SR_S3     7282

/* 必須 >= top 的 depth（MAX_IN_BEATS / MAX_OUT_WORDS） */
#ifndef TB_MAX_IN_BEATS
#define TB_MAX_IN_BEATS   (1920 * 1080 / 8)          /* 259200 */
#endif
#ifndef TB_MAX_OUT_WORDS
#define TB_MAX_OUT_WORDS  (960 * 540 * 3 / 16)       /* 97200  */
#endif

/* xmodel（DPU）輸入的 fix_point = norm_and_fix_letterbox 的 fix_point 參數 */
#ifndef TB_FIX_POINT
#define TB_FIX_POINT 5
#endif

static const ap_uint<1> MODE_3 = (ap_uint<1>)((int)SCALE_3);
static const ap_uint<1> MODE_2 = (ap_uint<1>)((int)SCALE_2);

/* 相容 C++11 / C++14 的 clamp 實作 */
template <typename T>
static inline T clamp_val(T v, T lo, T hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

/* ================================================================
 *  norm_and_fix_letterbox 的參考實作
 *  常數與運算順序逐字複製自 norm_fix_letterbox.cpp 的一般計算路徑
 *  （prepare() + double_kernel()），只是改成逐 byte 計算。
 * ================================================================ */
namespace normfix_ref {

constexpr double kMeanR = 0.485, kMeanG = 0.456, kMeanB = 0.406;
constexpr double kStdR  = 0.229, kStdG  = 0.224, kStdB  = 0.225;

constexpr double kU8ScaleR = 1./(kStdR*255.);
constexpr double kU8ScaleG = 1./(kStdG*255.);
constexpr double kU8ScaleB = 1./(kStdB*255.);
constexpr double kU8BiasR  = -kMeanR/kStdR;
constexpr double kU8BiasG  = -kMeanG/kStdG;
constexpr double kU8BiasB  = -kMeanB/kStdB;

struct Params {
    int    fix_point;
    double scale[3], bias[3];      /* 已乘上 2^fix_point */
    int8_t pad[3];                 /* 黑色 (0,0,0) 正規化後的值（letterbox 黑邊） */
};

/* 對應 prepare()。原本用 std::exp2f(fix_point)；整數次方時 ldexp(1, fp) 結果完全相同 */
static Params prepare(int fix_point)
{
    Params p;
    p.fix_point = fix_point;
    const double fp = std::ldexp(1.0, fix_point);
    p.scale[0] = kU8ScaleR * fp;  p.bias[0] = kU8BiasR * fp;
    p.scale[1] = kU8ScaleG * fp;  p.bias[1] = kU8BiasG * fp;
    p.scale[2] = kU8ScaleB * fp;  p.bias[2] = kU8BiasB * fp;
    for (int k = 0; k < 3; ++k)
        p.pad[k] = static_cast<int8_t>(clamp_val(p.bias[k], -128.0, 127.0));
    return p;
}

/* 對應 double_kernel 的一個 byte：clamp(src * scale + bias) → int8（往 0 截斷）
 * fused = false：乘、加各捨入一次（volatile 防止編譯器自動合成 FMA）
 * fused = true ：融合乘加（KV260 上 GCC -O2 會把 a*b+c 編成 fmla）            */
static inline int8_t norm_fix_one(unsigned v, int c, const Params &p, bool fused = false)
{
    double y;
    if (fused) {
        y = std::fma(static_cast<double>(v), p.scale[c], p.bias[c]);
    } else {
        volatile double prod = v * p.scale[c];
        y = prod + p.bias[c];
    }
    return static_cast<int8_t>(clamp_val(y, -128.0, 127.0));
}

} // namespace normfix_ref

static const normfix_ref::Params g_nf = normfix_ref::prepare(TB_FIX_POINT);

/* quant golden 實際測到的輸入值（每通道 0..255），用來報告覆蓋率 */
static bool g_hit[3][256];

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

/* norm_and_fix_letterbox 套用在 resize 後的 RGB（相當於 letterbox 的內容區），byte 順序 R,G,B */
static vector<uint8_t> golden_norm_and_fix(const vector<uint8_t> &rgb, const normfix_ref::Params &p)
{
    vector<uint8_t> out(rgb.size());
    for (size_t i = 0; i < rgb.size(); i++) {
        const int c = (int)(i % 3);
        g_hit[c][rgb[i]] = true;
        out[i] = static_cast<uint8_t>(normfix_ref::norm_fix_one(rgb[i], c, p));
    }
    return out;
}

/* ---------------------------------------------------------------- 0. LUT 自我檢查 */

/* kLut（DUT 的查表）第 c 通道、輸入 v 的值 */
static inline int dut_lut(int c, int v)
{
    const unsigned char *t = (c == 0) ? kLut.r : (c == 1) ? kLut.g : kLut.b;
    return static_cast<int8_t>(t[v]);
}

static bool check_lut_tables()
{
    using namespace normfix_ref;
    bool ok = true;

    printf("==== 0. LUT 自我檢查（golden = norm_and_fix_letterbox，fix_point=%d）====\n", TB_FIX_POINT);
    printf("    golden 黑邊 (0,0,0) = (%d, %d, %d)\n", g_nf.pad[0], g_nf.pad[1], g_nf.pad[2]);

    /* (a) DUT 編譯時的 fix_point */
    if (kFixPoint != TB_FIX_POINT) {
        printf("    [FAIL] quantize_lut.h 的 kFixPoint=%d，但 xmodel 輸入是 TB_FIX_POINT=%d\n",
               kFixPoint, TB_FIX_POINT);
        ok = false;
    } else {
        printf("    [PASS] kFixPoint = TB_FIX_POINT = %d\n", TB_FIX_POINT);
    }

    /* (b) golden 本身：融合乘加與一般乘加是否相同（KV260 -O2 會融合） */
    int fma_diff = 0;
    for (int c = 0; c < 3; c++)
        for (int v = 0; v < 256; v++)
            if (norm_fix_one(v, c, g_nf, false) != norm_fix_one(v, c, g_nf, true)) {
                if (fma_diff < 4)
                    printf("    [NOTE] ch=%c v=%d 一般乘加=%d 融合乘加=%d\n", "RGB"[c], v,
                           norm_fix_one(v, c, g_nf, false), norm_fix_one(v, c, g_nf, true));
                fma_diff++;
            }
    if (fma_diff)
        printf("    [WARN] 融合乘加與否有 %d 項不同：KV260 上 norm_and_fix 的結果取決於編譯選項\n", fma_diff);
    else
        printf("    [PASS] 融合乘加與一般乘加 768 項完全相同（不受 -ffp-contract 影響）\n");

    /* (c) DUT 的 kLut vs golden */
    int err[3] = {0, 0, 0}, shown = 0;
    for (int c = 0; c < 3; c++)
        for (int v = 0; v < 256; v++) {
            const int g = norm_fix_one(v, c, g_nf);
            const int d = dut_lut(c, v);
            if (d != g) {
                if (shown++ < 8)
                    printf("    [kLut] ch=%c v=%3d  kLut=%4d  golden=%4d\n", "RGB"[c], v, d, g);
                err[c]++;
            }
        }
    if (err[0] + err[1] + err[2] == 0) {
        printf("    [PASS] kLut 768 項與 norm_and_fix golden 完全相同\n");
    } else {
        ok = false;
        printf("    [FAIL] kLut 與 golden 不符  R/G/B = %d/%d/%d\n", err[0], err[1], err[2]);

        /* 診斷：kLut 實際對應哪個 fix_point？是否 R/B 對調？ */
        for (int fp = 0; fp <= 10; fp++) {
            const Params q = prepare(fp);
            int n = 0, n_swap = 0;
            for (int c = 0; c < 3; c++)
                for (int v = 0; v < 256; v++) {
                    if (dut_lut(c, v)     != norm_fix_one(v, c, q))     n++;
                    if (dut_lut(2 - c, v) != norm_fix_one(v, c, q))     n_swap++;
                }
            if (n == 0)
                printf("    → kLut 等於 fix_point=%d 的 norm_and_fix：請把 kFixPoint 改成 %d 重新合成\n",
                       fp, TB_FIX_POINT);
            if (n_swap == 0)
                printf("    → kLut 等於 fix_point=%d、但 R/B 表對調\n", fp);
        }
    }
    printf("\n");
    return ok;
}

/* ---------------------------------------------------------------- 比對 */

/* 比對一個輸出 buffer；回傳錯誤數（逐 byte 錯誤 + depth 結尾前被改寫的 word 數）
 * ch_err 非 NULL 時，統計資料區內 R/G/B 各自的錯誤數                              */
static int check_buffer(const char *tag,
                        const vector<ap_uint<128> > &out,
                        const vector<ap_uint<128> > &gold,
                        size_t nbytes, int ow,
                        const ap_uint<128> &poison,
                        int ch_err[3] = NULL,
                        bool is_quant = false)
{
    const size_t out_words = gold.size();
    int errors = 0;
    const int MAX_PRINT = 8;
    if (ch_err) ch_err[0] = ch_err[1] = ch_err[2] = 0;
    for (size_t i = 0; i < out_words; i++) {
        for (int k = 0; k < 16; k++) {
            unsigned d = out[i].range(8 * k + 7, 8 * k).to_uint();
            unsigned g = gold[i].range(8 * k + 7, 8 * k).to_uint();
            if (d != g) {
                size_t bi = i * 16 + k;
                if (bi < nbytes && ch_err) ch_err[bi % 3]++;
                if (errors < MAX_PRINT) {
                    if (bi < nbytes) {
                        size_t pix = bi / 3;
                        if (is_quant)
                            printf("    [%s] MISMATCH word %zu byte %2d (out pixel x=%zu y=%zu ch=%c): dut=%4d gold=%4d\n",
                                   tag, i, k, pix % ow, pix / ow, "RGB"[bi % 3],
                                   (int)(int8_t)d, (int)(int8_t)g);
                        else
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
    vector<uint8_t> q   = golden_norm_and_fix(rs, g_nf);
    vector<ap_uint<128> > gold_rs = bytes_to_words(rs);
    vector<ap_uint<128> > gold_q  = bytes_to_words(q);
    const size_t out_words = gold_q.size();
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

    /* ---- 比對：resize 對 resize golden，quant 對 norm_and_fix golden ---- */
    int ce[3];
    int err_rs = check_buffer("resize", out_rs, gold_rs, rs.size(), ow, poison);
    int err_q  = check_buffer("quant ", out_q,  gold_q,  q.size(),  ow, poison, ce, true);

    bool pass = (err_rs == 0) && (err_q == 0);
    printf("[%s] #%d %dx%d -> %dx%d (%dx) %-9s words=%zu  err_resize=%d err_quant=%d",
           pass ? "PASS" : "FAIL", idx, w, h, ow, oh, n, pat_name(t.pat), out_words, err_rs, err_q);
    if (err_q)
        printf("  (quant R/G/B 錯 %d/%d/%d)", ce[0], ce[1], ce[2]);
    printf("\n");

    /* quant 錯誤型態提示（resize 正確時才有意義） */
    if (err_rs == 0 && err_q) {
        if (ce[1] == 0 && ce[0] && ce[2])
            printf("    → 只有 R/B 錯：quantize_core 的 R/B 查表對調\n");
        else if (ce[0] + ce[1] + ce[2] == 0)
            printf("    → 資料區正確，錯在尾端補 0 或多寫\n");
    }
    return pass;
}

/* ---------------------------------------------------------------- main */

int main()
{
    int fails = 0;

    /* ---- 0. LUT 自我檢查 ---- */
    const bool lut_ok = check_lut_tables();
    if (!lut_ok) fails++;

    /* ---- 1. 資料路徑測試 ---- */
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

#ifndef TB_NO_FULL_HD
    /* ---- 1920x1080（2 倍剛好用滿 depth：quad_w = 240、97200 words）---- */
    tests.push_back({ 1920, 1080, 2, PAT_RANDOM, 100 });
    tests.push_back({ 1920, 1080, 3, PAT_RANDOM, 101 });
#endif

    printf("==== 1. 資料路徑（golden = uyvy2rgb → box resize → norm_and_fix）====\n");
    for (size_t i = 0; i < tests.size(); i++)
        if (!run_test(tests[i], (int)i)) fails++;

    /* ---- 覆蓋率：quant golden 實際測到的輸入值 ---- */
    int cov[3] = {0, 0, 0};
    for (int c = 0; c < 3; c++)
        for (int v = 0; v < 256; v++)
            cov[c] += g_hit[c][v];
    printf("\n[INFO] quant 輸入值覆蓋率 R/G/B = %d/%d/%d（共 256）%s\n", cov[0], cov[1], cov[2],
           (cov[0] < 256 || cov[1] < 256 || cov[2] < 256)
               ? "；未覆蓋的值由「0. LUT 自我檢查」逐項驗證" : "");

    printf("\n==== LUT 自我檢查 %s，%zu 個資料路徑測項，總共 %d 項失敗 ====\n",
           lut_ok ? "PASS" : "FAIL", tests.size(), fails);
    return fails ? 1 : 0;
}