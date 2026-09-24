// =====================================================================
//  uyvy2rgb_test.c  --  KV260 裸機 (standalone) 測試程式
//
//  1. 正確性: 多種尺寸 x 多種圖樣，逐 pixel 跟浮點黃金模型比對 (TOL LSB)
//             並檢查輸出緩衝尾端有沒有被多寫 (guard 區)
//  2. 速度  : 每個尺寸跑 N_RUN 次取平均，換算 ms / fps / 頻寬 / 效率，
//             並跟 A53 軟體轉換比較加速倍數
//
//  需要:
//    - Vitis 用含 uyvy2rgb_0 的 XSA 建 platform (standalone, psu_cortexa53_0)
//    - HLS 產生的驅動 xuyvy2rgb.h (匯出 IP 時自動附帶，BSP 會帶進來)
//    - PL_CLK_HZ 改成你 clk_wiz 給 ap_clk 的實際頻率
// =====================================================================

#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xil_types.h"
#include "xil_cache.h"
#include "xuyvy2rgb.h"

// 計時: SDT flow (Vitis 2023.2+) 把 xtime_l.h 換成 xiltimer 函式庫。
// 需要在 platform 的 standalone domain 裡啟用 xiltimer (見檔尾說明)。
#ifdef SDT
  #include "xiltimer.h"
  #include "sleep.h"
#else
  #include "xtime_l.h"
  #include "sleep.h"
#endif

// COUNTS_PER_SECOND 通常由上面的標頭提供；萬一沒有，用 CPU timestamp 頻率補上
#ifndef COUNTS_PER_SECOND
  #if   defined(XPAR_CPU_CORTEXA53_0_TIMESTAMP_CLK_FREQ)
    #define COUNTS_PER_SECOND  (XPAR_CPU_CORTEXA53_0_TIMESTAMP_CLK_FREQ)
  #elif defined(XPAR_CPU_TIMESTAMP_CLK_FREQ)
    #define COUNTS_PER_SECOND  (XPAR_CPU_TIMESTAMP_CLK_FREQ)
  #else
    #error "找不到 COUNTS_PER_SECOND，請在 xparameters.h 搜 TIMESTAMP_CLK_FREQ 後手動定義"
  #endif
#endif

// ---------------------------------------------------------------------
//  可調參數
// ---------------------------------------------------------------------
#ifndef PL_CLK_HZ
#define PL_CLK_HZ     300000000ULL    // ap_clk 頻率 (clk_wiz 輸出)，只用來算理論時間與效率
#endif
#define MAX_W         1920
#define MAX_H         1080
#define TOL           2               // 容許誤差 (LSB)，C sim 實測最大 1
#define N_RUN         20              // 速度量測重複次數
#define GUARD_BYTES   4096            // 輸出尾端的越界偵測區
#define SENTINEL      0xA5

// 驅動初始化參數，要跟 xuyvy2rgb.h 的 SDT 分支一致:
//   SDT flow (Vitis 2023.2 之後，含 2024.2): Initialize(ptr, UINTPTR BaseAddress)
//   舊版 flow                              : Initialize(ptr, u16 DeviceId)
#ifdef SDT
  #if   defined(XPAR_UYVY2RGB_0_BASEADDR)
    #define IP_INIT_ARG   XPAR_UYVY2RGB_0_BASEADDR
  #elif defined(XPAR_XUYVY2RGB_0_BASEADDR)
    #define IP_INIT_ARG   XPAR_XUYVY2RGB_0_BASEADDR
  #else
    #error "xparameters.h 找不到 uyvy2rgb 的 BASEADDR 巨集，請搜 UYVY2RGB 後手動填入"
  #endif
#else
  #if   defined(XPAR_UYVY2RGB_0_DEVICE_ID)
    #define IP_INIT_ARG   XPAR_UYVY2RGB_0_DEVICE_ID
  #elif defined(XPAR_XUYVY2RGB_0_DEVICE_ID)
    #define IP_INIT_ARG   XPAR_XUYVY2RGB_0_DEVICE_ID
  #else
    #error "xparameters.h 找不到 uyvy2rgb 的 DEVICE_ID 巨集，請搜 UYVY2RGB 後手動填入"
  #endif
#endif

// ---------------------------------------------------------------------
//  緩衝區: 4KB 對齊，讓 AXI burst 不會在 4KB 邊界被拆開
// ---------------------------------------------------------------------
static u8 uyvy_buf[MAX_W * MAX_H * 2]               __attribute__((aligned(4096)));
static u8 rgb_buf [MAX_W * MAX_H * 3 + GUARD_BYTES] __attribute__((aligned(4096)));
static u8 sw_buf  [MAX_W * MAX_H * 3]               __attribute__((aligned(4096)));

static XUyvy2rgb ip;

// ---------------------------------------------------------------------
//  計時: A53 generic timer
// ---------------------------------------------------------------------
static inline u64 ticks_to_us(u64 t) {
    return (t * 1000000ULL) / COUNTS_PER_SECOND;
}

// ---------------------------------------------------------------------
//  黃金參考 (BT.601 full-range)，不依賴 libm
// ---------------------------------------------------------------------
static inline u8 clamp_round(double v) {
    int i = (int)(v >= 0.0 ? v + 0.5 : v - 0.5);
    return (u8)(i < 0 ? 0 : (i > 255 ? 255 : i));
}

static inline void golden(int Y, int U, int V, u8 *r, u8 *g, u8 *b) {
    double D = (double)U - 128.0, E = (double)V - 128.0;
    *r = clamp_round(Y + 1.402    * E);
    *g = clamp_round(Y - 0.344136 * D - 0.714136 * E);
    *b = clamp_round(Y + 1.772    * D);
}

// 軟體版轉換 (整數定點)，只拿來當速度比較基準
static void sw_convert(const u8 *in, u8 *out, int w, int h) {
    int n = w * h / 2;
    for (int g = 0; g < n; g++) {
        int U = in[g*4+0] - 128, Y0 = in[g*4+1], V = in[g*4+2] - 128, Y1 = in[g*4+3];
        int rv = (359 * V) >> 8;
        int gv = (88 * U + 183 * V) >> 8;
        int bv = (454 * U) >> 8;
        int ys[2] = { Y0, Y1 };
        for (int k = 0; k < 2; k++) {
            int R = ys[k] + rv, G = ys[k] - gv, B = ys[k] + bv;
            u8 *p = &out[(g*2 + k) * 3];
            p[0] = (u8)(R < 0 ? 0 : (R > 255 ? 255 : R));
            p[1] = (u8)(G < 0 ? 0 : (G > 255 ? 255 : G));
            p[2] = (u8)(B < 0 ? 0 : (B > 255 ? 255 : B));
        }
    }
}

// ---------------------------------------------------------------------
//  測試圖樣
// ---------------------------------------------------------------------
enum { PAT_BARS, PAT_RAMP, PAT_EXTREME, PAT_RANDOM, PAT_COUNT };
static const char *pat_name[PAT_COUNT] = { "colorbar", "ramp", "extreme", "random" };

static u32 lcg = 0x12345678;
static inline u8 rnd8(void) { lcg = lcg * 1664525u + 1013904223u; return (u8)(lcg >> 24); }

static void gen_frame(int pat, int w, int h) {
    static const u8 bar[8][3] = {                    // Y, U, V
        {235,128,128},{210, 16,146},{170,166, 16},{145, 54, 34},
        { 81, 90,240},{ 41,240,110},{ 16,128,128},{128,128,128}};
    lcg = 0x12345678;
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c += 2) {
            u8 *q = &uyvy_buf[((size_t)r * w + c) * 2];   // U Y0 V Y1
            switch (pat) {
            case PAT_BARS: {
                int b = (c * 8) / w;
                q[0] = bar[b][1]; q[1] = bar[b][0]; q[2] = bar[b][2]; q[3] = bar[b][0];
                break; }
            case PAT_RAMP:
                q[0] = (u8)((r * 255) / (h - 1));
                q[1] = (u8)((c * 255) / (w - 1));
                q[2] = (u8)(255 - (r * 255) / (h - 1));
                q[3] = (u8)(((c + 1) * 255) / (w - 1));
                break;
            case PAT_EXTREME:
                q[0] = (c & 2) ? 255 : 0;
                q[1] = ((r + c) & 1) ? 255 : 0;
                q[2] = (r & 1) ? 0 : 255;
                q[3] = ((r + c + 1) & 1) ? 255 : 0;
                break;
            default:
                q[0] = rnd8(); q[1] = rnd8(); q[2] = rnd8(); q[3] = rnd8();
            }
        }
    }
}

// ---------------------------------------------------------------------
//  跑一次 IP，回傳 Start 到 Done 的 timer ticks
//  呼叫前 cache 已由外層處理好
// ---------------------------------------------------------------------
static u64 run_ip(int w, int h) {
    XUyvy2rgb_Set_uyvy_axi_bus(&ip, (u64)(UINTPTR)uyvy_buf);
    XUyvy2rgb_Set_rgb_axi_bus (&ip, (u64)(UINTPTR)rgb_buf);
    XUyvy2rgb_Set_img_w(&ip, (u32)w);
    XUyvy2rgb_Set_img_h(&ip, (u32)h);

    XTime t0, t1;
    XTime_GetTime(&t0);
    XUyvy2rgb_Start(&ip);
    while (!XUyvy2rgb_IsDone(&ip))
        ;
    XTime_GetTime(&t1);
    return (u64)(t1 - t0);
}

// ---------------------------------------------------------------------
//  正確性檢查
// ---------------------------------------------------------------------
static int verify(int w, int h, int *max_err) {
    size_t npix = (size_t)w * h, out_bytes = npix * 3;
    int fail = 0, maxe = 0;

    for (size_t pi = 0; pi < npix; pi++) {
        size_t g = pi >> 1;
        int U = uyvy_buf[g*4+0], V = uyvy_buf[g*4+2];
        int Y = uyvy_buf[g*4 + 1 + (pi & 1) * 2];
        u8 r, gg, b;
        golden(Y, U, V, &r, &gg, &b);

        const u8 *p = &rgb_buf[pi * 3];              // 記憶體順序 R, G, B
        int er = p[0] > r  ? p[0] - r  : r  - p[0];
        int eg = p[1] > gg ? p[1] - gg : gg - p[1];
        int eb = p[2] > b  ? p[2] - b  : b  - p[2];
        int e  = er > eg ? er : eg;  e = e > eb ? e : eb;
        if (e > maxe) maxe = e;
        if (e > TOL) {
            if (fail < 5)
                printf("      pixel %lu (r=%lu c=%lu) YUV(%d,%d,%d) "
                       "ref(%d,%d,%d) ip(%d,%d,%d)\n",
                       (unsigned long)pi, (unsigned long)(pi / w), (unsigned long)(pi % w),
                       Y, U, V, r, gg, b, p[0], p[1], p[2]);
            fail++;
        }
    }

    // 尾端 guard 區必須保持 sentinel，否則代表 IP 寫超過 out_beats
    int overrun = 0;
    for (size_t k = 0; k < GUARD_BYTES; k++)
        if (rgb_buf[out_bytes + k] != SENTINEL) overrun++;
    if (overrun)
        printf("      !! guard 區有 %d bytes 被改寫 (寫超過範圍)\n", overrun);

    *max_err = maxe;
    return fail + overrun;
}

// ---------------------------------------------------------------------
//  單一尺寸 x 單一圖樣: 正確性 (+ 選擇性量速度)
// ---------------------------------------------------------------------
static int test_case(int w, int h, int pat, int measure) {
    size_t in_bytes  = (size_t)w * h * 2;
    size_t out_bytes = (size_t)w * h * 3;

    gen_frame(pat, w, h);
    memset(rgb_buf, SENTINEL, out_bytes + GUARD_BYTES);

    // HP port 不 coherent: 輸入 flush 到 DDR；輸出區也先 flush，
    // 避免之後 cache 把舊的 dirty line 寫回蓋掉 IP 的結果
    Xil_DCacheFlushRange((UINTPTR)uyvy_buf, in_bytes);
    Xil_DCacheFlushRange((UINTPTR)rgb_buf,  out_bytes + GUARD_BYTES);

    u64 t = run_ip(w, h);

    Xil_DCacheInvalidateRange((UINTPTR)rgb_buf, out_bytes + GUARD_BYTES);

    int maxe = 0;
    int fail = verify(w, h, &maxe);
    printf("  %4dx%-4d %-9s maxErr=%d  %-4s  (%lu us)\n",
           w, h, pat_name[pat], maxe, fail ? "FAIL" : "ok",
           (unsigned long)ticks_to_us(t));
    if (fail) printf("      %d 個錯誤\n", fail);

    if (measure && !fail) {
        // ---- 速度: 重複 N_RUN 次，資料不必重算 ----
        u64 sum = 0, tmin = ~0ULL, tmax = 0;
        for (int i = 0; i < N_RUN; i++) {
            u64 ti = run_ip(w, h);
            sum += ti;
            if (ti < tmin) tmin = ti;
            if (ti > tmax) tmax = ti;
        }
        u64 avg_us  = ticks_to_us(sum / N_RUN);
        u64 min_us  = ticks_to_us(tmin);
        u64 max_us  = ticks_to_us(tmax);

        // 理論下限: 寫端每 cycle 一個 128-bit beat
        // (用浮點，小圖的理論值不到 1 us，整數除法會變 0)
        u64    out_beats = out_bytes / 16;
        double ideal_us  = (double)out_beats * 1e6 / (double)PL_CLK_HZ;

        // 軟體基準
        XTime s0, s1;
        XTime_GetTime(&s0);
        sw_convert(uyvy_buf, sw_buf, w, h);
        XTime_GetTime(&s1);
        u64 sw_us = ticks_to_us((u64)(s1 - s0));

        double avg_s  = (double)(sum / N_RUN) / (double)COUNTS_PER_SECOND;
        double avg_uf = avg_s * 1e6;
        double fps    = avg_s > 0 ? 1.0 / avg_s : 0.0;
        double gbps   = avg_s > 0 ? (double)(in_bytes + out_bytes) / avg_s / 1e9 : 0.0;
        double eff    = avg_uf > 0 ? 100.0 * ideal_us / avg_uf : 0.0;
        double spd    = avg_uf > 0 ? (double)sw_us / avg_uf : 0.0;

        printf("      速度 (%d 次): avg %lu us  min %lu  max %lu\n",
               N_RUN, (unsigned long)avg_us, (unsigned long)min_us, (unsigned long)max_us);
        printf("      %.1f fps   DDR 讀+寫 %.2f GB/s   理論 %.1f us -> 效率 %.1f%%\n",
               fps, gbps, ideal_us, eff);
        printf("      A53 軟體 %lu us -> 加速 %.2fx\n", (unsigned long)sw_us, spd);
    }
    return fail;
}

// =====================================================================

int main(void) {
    printf("\n=== uyvy2rgb 硬體測試 ===\n");

    // ---- 計時器暖機 + 自我檢查 ----
    // xiltimer 的計時器在第一次 sleep 呼叫時才初始化，
    // 沒暖機的話 XTime_GetTime 可能一直回傳 0
    usleep(1000);
    {
        XTime a, b;
        XTime_GetTime(&a);
        usleep(10000);                       // 10 ms
        XTime_GetTime(&b);
        u64 us = ticks_to_us((u64)(b - a));
        printf("    計時器檢查: usleep(10000) 量到 %lu us", (unsigned long)us);
        if (us < 5000 || us > 20000) {
            printf("  <-- 異常，時間數據不可信\n");
            printf("    檢查 platform 是否啟用 xiltimer、COUNTS_PER_SECOND 是否正確\n");
        } else {
            printf("  ok\n");
        }
    }

    printf("    PL clk %lu MHz, timer %lu Hz, TOL %d LSB\n\n",
           (unsigned long)(PL_CLK_HZ / 1000000), (unsigned long)COUNTS_PER_SECOND, TOL);

    if (XUyvy2rgb_Initialize(&ip, IP_INIT_ARG) != XST_SUCCESS) {
        printf("IP 初始化失敗，檢查 xparameters.h 裡的巨集名稱\n");
        return -1;
    }
    if (!XUyvy2rgb_IsIdle(&ip))
        printf("警告: IP 不在 idle 狀態 (可能 PL 沒有 clock / reset)\n");

    static const int sizes[][2] = {
        {   64,   16 },
        {  640,  480 },
        { 1280,  720 },
        { 1920, 1080 },
    };
    int nsz = sizeof(sizes) / sizeof(sizes[0]);
    int total_fail = 0;

    for (int s = 0; s < nsz; s++) {
        int w = sizes[s][0], h = sizes[s][1];
        if (w % 16) { printf("  %dx%d skip (寬度需為 16 的倍數)\n", w, h); continue; }
        printf("[%dx%d]\n", w, h);
        for (int p = 0; p < PAT_COUNT; p++)
            total_fail += test_case(w, h, p, p == PAT_RANDOM);   // random 那組順便量速度
        printf("\n");
    }

    printf("=== 總結: %s", total_fail ? "FAIL" : "PASS");
    if (total_fail) printf(" (%d 個錯誤)", total_fail);
    printf(" ===\n");
    return total_fail ? 1 : 0;
}