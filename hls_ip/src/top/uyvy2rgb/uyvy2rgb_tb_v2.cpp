// =====================================================================
//  tb_uyvy2rgb.cpp  --  csim / cosim 共用的 testbench
//
//  用 COSIM_MODE 分流 (自訂巨集，由 tcl 傳入):
//      add_files -tb tb_uyvy2rgb.cpp -cflags "-DCOSIM_MODE"   (cosim)
//      add_files -tb tb_uyvy2rgb.cpp                          (csim)
//
//      csim  : 跑完整測試 (窮舉、隨機、多種圖樣)
//      cosim : 只跑必要的部分，縮短 RTL 模擬時間
//
//  ***  重要  ***
//  cosim 的 wrapper 會照 pragma 裡的 depth 搬資料，所以 buffer 的元素數
//  必須「剛好等於」depth，而且每次呼叫都一樣大。本檔固定配置 MAX_IN /
//  MAX_OUT，小圖只用前面一段。depth 對不上會在 ENTER_WRAPC 階段 SIGSEGV。
//
//      #pragma HLS INTERFACE m_axi port=uyvy_axi_bus ... depth=MAX_IN
//      #pragma HLS INTERFACE m_axi port=rgb_axi_bus  ... depth=MAX_OUT
// =====================================================================

//#include "uyvy2rgb.h"
#include "uyvy2rgb_top.h"
#include <ap_int.h>
#include <ap_fixed.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------
//  測試影像上限 —— 若 uyvy2rgb.h 已定義就沿用
//  MAX_IN  = (MAX_IMG_W/8) * MAX_IMG_H  必須等於 gmem0 的 depth
//  MAX_OUT = MAX_IN * 3/2               必須等於 gmem1 的 depth
// ---------------------------------------------------------------------
#ifndef MAX_IMG_W
#define MAX_IMG_W 128
#endif
#ifndef MAX_IMG_H
#define MAX_IMG_H 32
#endif
#ifndef MAX_IN
#define MAX_IN   ((size_t)(MAX_IMG_W / 8) * MAX_IMG_H)       // 512
#endif
#ifndef MAX_OUT
#define MAX_OUT  (MAX_IN * 3 / 2)                            // 768
#endif

// 容許誤差 (LSB)。8-bit 權重量化下實測最大 1，留 2 當緩衝。
static const int TOL = 2;

// cvt_core 的 pixel 位元排列
static inline void unpack_pixel(const ap_uint<24> &p, int &R, int &G, int &B) {
    R = (int)p.range(23, 16);
    G = (int)p.range(15,  8);
    B = (int)p.range( 7,  0);
}

// 頂層輸出在記憶體裡的 byte 順序，對應 uyvy2rgb.cpp 的 to_mem()
static const int MEM_R = 0, MEM_G = 1, MEM_B = 2;

static inline ap_int<8> make_de(int uv) { return (ap_int<8>)(uv - 128); }

// ---------------------------------------------------------------------
//  黃金參考模型 (BT.601 full-range, 浮點)
// ---------------------------------------------------------------------
static inline int clamp8(double v) {
    int i = (int)std::lround(v);
    return i < 0 ? 0 : (i > 255 ? 255 : i);
}
static void golden(int Y, int U, int V, int &R, int &G, int &B) {
    double D = (double)U - 128.0, E = (double)V - 128.0;
    R = clamp8(Y + 1.402    * E);
    G = clamp8(Y - 0.344136 * D - 0.714136 * E);
    B = clamp8(Y + 1.772    * D);
}

// ---------------------------------------------------------------------
//  128-bit 記憶體的 byte 存取
// ---------------------------------------------------------------------
static inline void put_byte(std::vector<ap_uint<128> > &m, size_t i, int v) {
    m[i >> 4].range(((i & 15) << 3) + 7, (i & 15) << 3) = (ap_uint<8>)v;
}
static inline int get_byte(const std::vector<ap_uint<128> > &m, size_t i) {
    return (int)m[i >> 4].range(((i & 15) << 3) + 7, (i & 15) << 3);
}

// =====================================================================
//  [1] subtract_128
// =====================================================================
static int test_subtract_128() {
    std::printf("[1] subtract_128 全範圍\n");
    int fail = 0;
    for (int i = 0; i < 256; i++) {
        ap_uint<8> in = (ap_uint<8>)i;
        ap_int<8>  out = 0;
        subtract_128(in, out);
        if ((int)out != i - 128) {
            if (fail < 8)
                std::printf("    in=%3d got=%4d expect=%4d  FAIL\n",
                            i, (int)out, i - 128);
            fail++;
        }
    }
    std::printf(fail ? "    %d / 256 錯誤\n\n" : "    256 / 256 通過\n\n", fail);
    return fail;
}

// =====================================================================
//  [2] clamp_s  (有號輸入，直接代表真值，不含偏移)
//      ap_int<10> 全範圍 -512 ~ 511，期望 clamp(v, 0, 255)
// =====================================================================
static int test_clamp_s() {
    std::printf("[2] clamp_s 全範圍 (-512~511)\n");
    int fail = 0;
    for (int i = -512; i < 512; i++) {
        ap_int<10> v = (ap_int<10>)i;
        ap_uint<8> o = 0;
        clamp_s(v, o);
        int expect = i < 0 ? 0 : (i > 255 ? 255 : i);
        if ((int)o != expect) {
            if (fail < 8)
                std::printf("    v=%4d (bits[9:8]=%d%d) got=%3d expect=%3d  FAIL\n",
                            i, (int)v[9], (int)v[8], (int)o, expect);
            fail++;
        }
    }
    std::printf(fail ? "    %d / 1024 錯誤\n\n" : "    1024 / 1024 通過\n\n", fail);
    return fail;
}

// =====================================================================
//  [3] cvt_core  (csim only —— 窮舉太慢，cosim 不跑)
// =====================================================================
struct Stats {
    int n = 0, fail = 0;
    int maxR = 0, maxG = 0, maxB = 0;
    int wY = 0, wU = 0, wV = 0, wErr = 0;
    bool untouched = false;
};

static void run_one(int Y, int U, int V, Stats &st, bool verbose) {
    ap_uint<8> y = (ap_uint<8>)Y;
    ap_int<8>  d = make_de(U);
    ap_int<8>  e = make_de(V);
    ap_uint<24> pixel = 0xA5A5A5;              // sentinel

    cvt_core(y, d, e, pixel);
    if (pixel == (ap_uint<24>)0xA5A5A5) st.untouched = true;

    int gR, gG, gB, dR, dG, dB;
    golden(Y, U, V, gR, gG, gB);
    unpack_pixel(pixel, dR, dG, dB);

    int eR = std::abs(dR - gR), eG = std::abs(dG - gG), eB = std::abs(dB - gB);
    int worst = std::max(eR, std::max(eG, eB));

    st.n++;
    st.maxR = std::max(st.maxR, eR);
    st.maxG = std::max(st.maxG, eG);
    st.maxB = std::max(st.maxB, eB);
    if (worst > st.wErr) { st.wErr = worst; st.wY = Y; st.wU = U; st.wV = V; }
    if (worst > TOL) st.fail++;

    if (verbose || worst > TOL)
        std::printf("    Y=%3d U=%3d V=%3d (D=%4d E=%4d) | ref(%3d,%3d,%3d) "
                    "dut(%3d,%3d,%3d) err(%d,%d,%d) %s\n",
                    Y, U, V, (int)d, (int)e, gR, gG, gB, dR, dG, dB,
                    eR, eG, eB, worst <= TOL ? "ok" : "FAIL");
}

static int test_cvt_core() {
    std::printf("[3] cvt_core\n");
    struct { int y, u, v; const char *name; } vec[] = {
        {   0, 128, 128, "black"       },
        { 255, 128, 128, "white"       },
        { 128, 128, 128, "mid gray"    },
        {  81,  90, 240, "red"         },
        { 145,  54,  34, "green"       },
        {  41, 240, 110, "blue"        },
        { 210,  16, 146, "yellow"      },
        { 170, 166,  16, "cyan"        },
        { 106, 202, 222, "magenta"     },
        {   0,   0,   0, "min all"     },
        { 255, 255, 255, "max all"     },
        { 255,   0, 255, "clamp hi/lo" },
        {   0, 255,   0, "clamp lo/hi" },
        { 128, 127, 129, "D=-1 E=+1"   },
        { 128, 129, 127, "D=+1 E=-1"   },
        { 128,   0,   0, "D=E=-128"    },
        { 128, 255, 255, "D=E=+127"    },
        {   0,   0, 255, "B/R 負向最深" },   // 打 clamp_s 的 bits[9:8]=11
        { 255, 255,   0, "B 正向最高"   },   // 打 clamp_s 的 bits[9:8]=01
    };
    Stats sd;
    std::printf("  定向向量\n");
    for (auto &t : vec) {
        std::printf("  -- %s\n", t.name);
        run_one(t.y, t.u, t.v, sd, true);
    }
    std::printf("    %d / %d 通過\n", sd.n - sd.fail, sd.n);

    std::printf("  隨機 20000 組\n");
    Stats sr;
    std::srand(0xC0FFEE);
    for (int i = 0; i < 20000; i++)
        run_one(std::rand() & 0xFF, std::rand() & 0xFF, std::rand() & 0xFF, sr, false);
    std::printf("    %d / %d 通過   maxErr R=%d G=%d B=%d\n",
                sr.n - sr.fail, sr.n, sr.maxR, sr.maxG, sr.maxB);

    std::printf("  步進掃描 (step=8)\n");
    Stats ss;
    for (int Y = 0; Y < 256; Y += 8)
      for (int U = 0; U < 256; U += 8)
        for (int V = 0; V < 256; V += 8)
            run_one(Y, U, V, ss, false);
    std::printf("    %d / %d 通過   maxErr R=%d G=%d B=%d\n",
                ss.n - ss.fail, ss.n, ss.maxR, ss.maxG, ss.maxB);
    if (ss.wErr)
        std::printf("    最差向量 Y=%d U=%d V=%d (err=%d)\n", ss.wY, ss.wU, ss.wV, ss.wErr);
    if (sd.untouched || sr.untouched || ss.untouched)
        std::printf("    !! pixel 未被寫入 (仍是 sentinel)\n");
    std::printf("\n");
    return sd.fail + sr.fail + ss.fail;
}

// =====================================================================
//  [4] subtract_128 -> cvt_core 串接  (csim only)
// =====================================================================
static int test_chained() {
    std::printf("[4] subtract_128 -> cvt_core 串接\n");
    int fail = 0;
    for (int Y = 0; Y < 256; Y += 16)
      for (int U = 0; U < 256; U += 16)
        for (int V = 0; V < 256; V += 16) {
            ap_uint<8> uin = (ap_uint<8>)U, vin = (ap_uint<8>)V;
            ap_int<8>  d = 0, e = 0;
            subtract_128(uin, d);
            subtract_128(vin, e);

            ap_uint<8>  y = (ap_uint<8>)Y;
            ap_uint<24> pixel = 0;
            cvt_core(y, d, e, pixel);

            int gR, gG, gB, dR, dG, dB;
            golden(Y, U, V, gR, gG, gB);
            unpack_pixel(pixel, dR, dG, dB);
            if (std::max(std::abs(dR-gR), std::max(std::abs(dG-gG), std::abs(dB-gB))) > TOL) {
                if (fail < 8)
                    std::printf("    Y=%3d U=%3d V=%3d ref(%3d,%3d,%3d) dut(%3d,%3d,%3d) FAIL\n",
                                Y, U, V, gR, gG, gB, dR, dG, dB);
                fail++;
            }
        }
    std::printf(fail ? "    %d 組錯誤\n\n" : "    全部通過\n\n", fail);
    return fail;
}

// =====================================================================
//  [5] uyvy2rgb 整張影像  (csim / cosim 都跑)
//      頂層走的是 cvt_pair (C port 一條線補償借位)，跟 cvt_core 不同路徑，
//      所以這一項才是真正驗證合成對象的測試
// =====================================================================
enum Pattern { PAT_RAMP, PAT_BARS, PAT_RANDOM, PAT_EXTREME, PAT_MIXED };

static void gen_frame(int w, int h, Pattern pat,
                      std::vector<int> &Y, std::vector<int> &U, std::vector<int> &V) {
    Y.assign((size_t)w * h, 0);
    U.assign((size_t)w * h / 2, 0);
    V.assign((size_t)w * h / 2, 0);
    static const int bar[8][3] = {
        {235,128,128},{210, 16,146},{170,166, 16},{145, 54, 34},
        { 81, 90,240},{ 41,240,110},{ 16,128,128},{128,128,128}};
    for (int r = 0; r < h; r++)
      for (int c = 0; c < w; c++) {
        size_t pi = (size_t)r * w + c, gi = pi >> 1;
        Pattern p = pat;
        if (pat == PAT_MIXED)                  // 一張圖混三種，給 cosim 用
            p = (r < 2) ? PAT_EXTREME : ((r < 4) ? PAT_RANDOM : PAT_RAMP);
        switch (p) {
        case PAT_RAMP:
            Y[pi] = (c * 255) / (w - 1);
            U[gi] = (r * 255) / (h - 1);
            V[gi] = 255 - (r * 255) / (h - 1);
            break;
        case PAT_BARS: {
            int b = (c * 8) / w;
            Y[pi] = bar[b][0]; U[gi] = bar[b][1]; V[gi] = bar[b][2];
            break; }
        case PAT_EXTREME:
            Y[pi] = ((r + c) & 1) ? 255 : 0;
            U[gi] = (c & 2) ? 255 : 0;
            V[gi] = (r & 1) ? 0 : 255;
            break;
        default:
            Y[pi] = std::rand() & 0xFF;
            U[gi] = std::rand() & 0xFF;
            V[gi] = std::rand() & 0xFF;
        }
      }
}

static int test_frame(int w, int h, Pattern pat, const char *name) {
    size_t in_beats  = (size_t)(w / 8) * h;
    size_t out_beats = in_beats * 3 / 2;

    if (w % 16) {
        std::printf("  %-10s %4dx%-4d  skip (img_w 必須是 16 的倍數)\n", name, w, h);
        return 0;
    }
    if (in_beats > MAX_IN || out_beats > MAX_OUT) {
        std::printf("  %-10s %4dx%-4d  skip (超出 MAX_IN/MAX_OUT)\n", name, w, h);
        return 0;
    }

    std::vector<int> Y, U, V;
    gen_frame(w, h, pat, Y, U, V);

    // 固定配置 MAX，小圖只用前面一段 —— cosim 的 depth 必須對得上
    std::vector<ap_uint<128> > in_mem (MAX_IN,  0);
    std::vector<ap_uint<128> > out_mem(MAX_OUT, 0xDEAD);

    for (size_t g = 0; g < (size_t)w * h / 2; g++) {
        put_byte(in_mem, g * 4 + 0, U[g]);
        put_byte(in_mem, g * 4 + 1, Y[g * 2 + 0]);
        put_byte(in_mem, g * 4 + 2, V[g]);
        put_byte(in_mem, g * 4 + 3, Y[g * 2 + 1]);
    }

    uyvy2rgb(in_mem.data(), out_mem.data(), (ap_uint<12>)w, (ap_uint<12>)h);

    int fail = 0, maxErr = 0;
    for (size_t pi = 0; pi < (size_t)w * h; pi++) {
        size_t gi = pi >> 1;
        int gR, gG, gB;
        golden(Y[pi], U[gi], V[gi], gR, gG, gB);

        int dR = get_byte(out_mem, pi * 3 + MEM_R);
        int dG = get_byte(out_mem, pi * 3 + MEM_G);
        int dB = get_byte(out_mem, pi * 3 + MEM_B);

        int w2 = std::max(std::abs(dR-gR), std::max(std::abs(dG-gG), std::abs(dB-gB)));
        maxErr = std::max(maxErr, w2);
        if (w2 > TOL) {
            if (fail < 8)
                std::printf("    pixel %5zu (r=%zu c=%zu) YUV(%3d,%3d,%3d) "
                            "ref(%3d,%3d,%3d) dut(%3d,%3d,%3d) FAIL\n",
                            pi, pi / w, pi % w, Y[pi], U[gi], V[gi],
                            gR, gG, gB, dR, dG, dB);
            fail++;
        }
    }

    // 檢查輸出區尾端沒被多寫 (leftover 狀態機若多跑一拍會踩到這裡)
    int overrun = 0;
    for (size_t k = out_beats; k < MAX_OUT; k++)
        if (out_mem[k] != (ap_uint<128>)0xDEAD) overrun++;

    std::printf("  %-10s %4dx%-4d  %6zu pixel  maxErr=%d  %s\n",
                name, w, h, (size_t)w * h, maxErr,
                (fail || overrun) ? "FAIL" : "ok");
    if (fail)    std::printf("             %d 個 pixel 超出容許誤差\n", fail);
    if (overrun) std::printf("             %d 個 beat 寫超過 out_beats\n", overrun);
    return fail + overrun;
}

static int test_top() {
    std::printf("[5] uyvy2rgb 整張影像\n");
    std::srand(0x5EED);
    int fail = 0;

#ifdef COSIM_MODE
    // cosim: 一張混合圖樣就夠，RTL 模擬時間才不會爆
    fail += test_frame(MAX_IMG_W, MAX_IMG_H, PAT_MIXED, "mixed");
#else
    fail += test_frame( 16,  4, PAT_EXTREME, "extreme");
    fail += test_frame( 32,  8, PAT_BARS,    "colorbar");
    fail += test_frame( 64, 16, PAT_RAMP,    "ramp");
    fail += test_frame(MAX_IMG_W, MAX_IMG_H, PAT_RANDOM, "random");
    fail += test_frame(MAX_IMG_W, MAX_IMG_H, PAT_MIXED,  "mixed");
#endif

    std::printf("\n");
    return fail;
}

// =====================================================================

int main() {
#ifdef COSIM_MODE
    std::printf("(cosim mode)\n");
#else
    std::printf("(csim mode)\n");
#endif
    std::printf("=== uyvy2rgb testbench ===\n");
    std::printf("    TOL = %d LSB,  MAX_IMG = %dx%d,  MAX_IN = %zu,  MAX_OUT = %zu\n\n",
                TOL, MAX_IMG_W, MAX_IMG_H, (size_t)MAX_IN, (size_t)MAX_OUT);

    int f = 0;
    f += test_subtract_128();
    f += test_clamp_s();

#ifndef COSIM_MODE
    // 這兩項是純 C 的單元測試，跟 RTL 無關，cosim 跑只是浪費時間
    f += test_cvt_core();
    f += test_chained();
#endif

    f += test_top();

    std::printf("=== 總結 ===\n");
    if (f == 0) { std::printf("  PASS\n"); return 0; }
    std::printf("  FAIL (%d 個錯誤)\n", f);
    return 1;
}
