/******************************************************************************
 * main_uyvy_resize.c
 *
 * XUyvy_resize HLS IP 的 Vitis 裸機測試程式
 *
 * IP 功能：UYVY 4:2:2 -> RGB888，再做整數倍 Box-filter 縮小（2x / 3x）
 *
 * 介面（對應 xuyvy_resize.h）：
 *   uyvy_axi_bus   m_axi 輸入，UYVY（每 4 byte = U, Y0, V, Y1）
 *   rgb_axi_bus    m_axi 輸出，RGB888 packed，byte 順序 R, G, B
 *   img_w, img_h   輸入影像寬高（ap_uint<12>，1 ~ 4095）
 *   scale_mode     0 = 2 倍、1 = 3 倍
 *
 * 尺寸限制（見 uyvy_resize.h）：
 *   img_w：16 的倍數；3 倍時為 48 的倍數
 *   img_h：3 倍時為 3 的倍數，2 倍時為偶數
 *   輸出寬度 <= 960
 *
 * 注意事項：
 *   1. m_axi 走實體位址，緩衝區必須 32-byte 對齊
 *   2. 若啟用 D-Cache，送出前 flush 輸入、收回前 invalidate 輸出
 *   3. 輸出 byte 數不是 16 的倍數時，IP 會把最後一個 word 補 0 寫出，
 *      所以輸出緩衝區要配到 out_words * 16 byte
 *
 * 在 Vitis 的 application 裡只能有一個 main()，
 * 若專案中已有其他 main.c，請移除或改用本檔。
 *****************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "platform.h"
#include "xil_printf.h"
#include "xil_cache.h"
// 計時: SDT flow (Vitis 2023.2+) 把 xtime_l.h 換成 xiltimer 函式庫。
// 需要在 platform 的 standalone domain 裡啟用 xiltimer。
#ifdef SDT
  #include "xiltimer.h"
  #include "sleep.h"
#else
  #include "xtime_l.h"
  #include "sleep.h"
#endif
#include "xparameters.h"
#include "xuyvy_resize.h"

/* ---------------------------------------------------------------- 測試參數 */

/* 輸入尺寸固定，兩種縮小倍率在執行時各跑一次。
 * 先用小尺寸（例如 96x36）驗證功能，通過後再換成 1920x1080。 */
#define SRC_W       1920
#define SRC_H       1080

#define SCALE_MODE_2  0
#define SCALE_MODE_3  1

#define OUT_W_MAX   960      /* IP 內 line buffer 上限 */
#define IMG_DIM_MAX 4095     /* img_w / img_h 為 ap_uint<12> */

#define IP_CLK_MHZ  250      /* 只用於印出理論時間，請依實際 IP 時脈修改 */

#define SRC_BYTES   (SRC_W * SRC_H * 2)                    /* UYVY：每 pixel 2 byte */

/* 輸出緩衝區以 2 倍（輸出較大）配置，並向上取整到 16 byte */
#define DST_BYTES_MAX   ((SRC_W / 2) * (SRC_H / 2) * 3)
#define DST_ALLOC       (((DST_BYTES_MAX + 15) / 16) * 16)

/* 輸出緩衝區尾端的哨兵區，用來偵測 IP 越界寫 */
#define GUARD_BYTES 64
#define GUARD_VAL   0xA5

/* 32-byte 對齊，配合 AXI burst 與 cache line */
static uint8_t src_buf[SRC_BYTES + 32]               __attribute__((aligned(32)));
static uint8_t dst_buf[DST_ALLOC + GUARD_BYTES + 32] __attribute__((aligned(32)));
static uint8_t ref_buf[DST_BYTES_MAX]                __attribute__((aligned(32)));

XUyvy_resize UyvyResizeInst;


/* ================================================================
 *  單次測試的尺寸資訊（依 scale 推算，僅供 golden / 比對 / 顯示用）
 * ================================================================ */

typedef struct {
    int scale;          /* 2 或 3 */
    int scale_mode;     /* 傳給 IP 的值 */
    int dst_w, dst_h;
    int dst_bytes;
    int out_words;      /* IP 實際寫出的 word 數（向上取整） */
    int inv_scale;      /* golden 用：65536 / scale^2 */
} case_cfg_t;

static void make_cfg(int scale, case_cfg_t *cfg)
{
    cfg->scale      = scale;
    cfg->scale_mode = (scale == 3) ? SCALE_MODE_3 : SCALE_MODE_2;
    cfg->dst_w      = SRC_W / scale;
    cfg->dst_h      = SRC_H / scale;
    cfg->dst_bytes  = cfg->dst_w * cfg->dst_h * 3;
    cfg->out_words  = (cfg->dst_bytes + 15) / 16;
    cfg->inv_scale  = (scale == 3) ? 7282 : 16384;
}


/* ================================================================
 *  尺寸合法性檢查（對應 uyvy_resize.h 的限制）
 * ================================================================ */

static int check_size(const case_cfg_t *cfg)
{
    int ok = 1;

    if (SRC_W < 1 || SRC_W > IMG_DIM_MAX || SRC_H < 1 || SRC_H > IMG_DIM_MAX) {
        xil_printf("  ERROR: img_w/img_h 超出 12-bit 範圍 (1 ~ %d)\r\n", IMG_DIM_MAX);
        ok = 0;
    }
    if (SRC_W % 16) {
        xil_printf("  ERROR: img_w 需為 16 的倍數\r\n");
        ok = 0;
    }
    if (cfg->scale == 3) {
        if (SRC_W % 48) { xil_printf("  ERROR: 3 倍時 img_w 需為 48 的倍數\r\n"); ok = 0; }
        if (SRC_H % 3)  { xil_printf("  ERROR: 3 倍時 img_h 需為 3 的倍數\r\n");  ok = 0; }
    } else {
        if (SRC_H % 2)  { xil_printf("  ERROR: 2 倍時 img_h 需為偶數\r\n");       ok = 0; }
    }
    if (cfg->dst_w > OUT_W_MAX) {
        xil_printf("  ERROR: 輸出寬度 %d 超過 %d\r\n", cfg->dst_w, OUT_W_MAX);
        ok = 0;
    }
    return ok;
}


/* ================================================================
 *  黃金模型 (1)：UYVY -> RGB，與 IP 的 cvt_pair 位元一致
 *
 *  d = U - 128，e = V - 128
 *  權重為 Q1.8 / Q0.8 取最近值：454、359、88、183（單位 1/256）
 *  截斷方式與硬體相同（算術右移 = 向下取整）：
 *    R = clamp(Y + floor(359*e / 256))
 *    G = clamp(Y - floor((88*d + 183*e) / 256))   兩項先相加再截斷一次
 *    B = clamp(Y + floor(454*d / 256))
 * ================================================================ */

static int floor_div256(int x)
{
    /* 向下取整的 x / 256，對負數也正確（等同算術右移 8） */
    return (x >= 0) ? (x >> 8) : -((-x + 255) >> 8);
}

static int clamp_u8(int v)
{
    return (v < 0) ? 0 : ((v > 255) ? 255 : v);
}

/* 取出輸入影像 (x, y) 的 RGB；同一組的兩個 pixel 共用 U、V */
static void src_rgb(const uint8_t *uyvy, int x, int y, int rgb[3])
{
    const uint8_t *q = uyvy + (y * SRC_W + (x & ~1)) * 2;   /* 該組的起點 */
    int u  = q[0];
    int yy = (x & 1) ? q[3] : q[1];
    int v  = q[2];
    int d  = u - 128;
    int e  = v - 128;

    rgb[0] = clamp_u8(yy + floor_div256(359 * e));
    rgb[1] = clamp_u8(yy - floor_div256(88 * d + 183 * e));
    rgb[2] = clamp_u8(yy + floor_div256(454 * d));
}


/* ================================================================
 *  黃金模型 (2)：box filter，與 IP 的 resize 位元一致
 *
 *  (sum * inv_scale) >> 16
 *    3 倍 inv = 7282   <-> IP 16-bit scale_rate 7282，取 [23:16]
 *    2 倍 inv = 16384  <-> IP 2-bit  scale_rate 1，   取 [9:2]（= sum >> 2）
 *
 *  直接由 UYVY 逐 pixel 轉換後累加，不需要額外 W*H*3 的 RGB 中間緩衝區
 * ================================================================ */

static void golden_uyvy_resize(const uint8_t *uyvy, uint8_t *dst, const case_cfg_t *cfg)
{
    int x, y, dx, dy, c;
    int s = cfg->scale;

    for (y = 0; y < cfg->dst_h; y++) {
        for (x = 0; x < cfg->dst_w; x++) {
            uint32_t sum[3] = {0, 0, 0};

            for (dy = 0; dy < s; dy++) {
                for (dx = 0; dx < s; dx++) {
                    int rgb[3];
                    src_rgb(uyvy, x * s + dx, y * s + dy, rgb);
                    for (c = 0; c < 3; c++)
                        sum[c] += (uint32_t)rgb[c];
                }
            }

            uint8_t *q = dst + (y * cfg->dst_w + x) * 3;
            for (c = 0; c < 3; c++)
                q[c] = (uint8_t)((sum[c] * (uint32_t)cfg->inv_scale) >> 16);
        }
    }
}


/* ================================================================
 *  黃金模型自我檢查：幾組可以手算的輸入（含 clamp 兩端與負數取整）
 *  避免黃金模型本身寫錯卻和 IP「一起錯」
 * ================================================================ */

static int self_check_golden(void)
{
    static const struct { int y, u, v, r, g, b; } t[] = {
        {128, 128, 128, 128, 128, 128},     /* 灰階 */
        {  0, 128, 128,   0,   0,   0},     /* 黑 */
        {255, 128, 128, 255, 255, 255},     /* 白 */
        {200, 255, 128, 200, 157, 255},     /* U 最大，B 上溢 clamp */
        { 50,   0, 128,  50,  94,   0},     /* U 最小，B 下溢 clamp */
        {100, 128, 127,  98, 101, 100},     /* 負數向下取整 */
        { 10, 128, 255, 188,   0,  10},     /* G 下溢 clamp */
    };
    int i, ok = 1;

    for (i = 0; i < (int)(sizeof(t) / sizeof(t[0])); i++) {
        uint8_t q[4];
        int rgb[3];
        q[0] = (uint8_t)t[i].u; q[1] = (uint8_t)t[i].y;
        q[2] = (uint8_t)t[i].v; q[3] = (uint8_t)t[i].y;

        /* 借用 src_rgb 的計算：暫時把這一組放在 src_buf 開頭 */
        memcpy(src_buf, q, 4);
        src_rgb(src_buf, 0, 0, rgb);

        if (rgb[0] != t[i].r || rgb[1] != t[i].g || rgb[2] != t[i].b) {
            xil_printf("  黃金模型錯誤: Y=%d U=%d V=%d -> (%d,%d,%d)，預期 (%d,%d,%d)\r\n",
                       t[i].y, t[i].u, t[i].v, rgb[0], rgb[1], rgb[2],
                       t[i].r, t[i].g, t[i].b);
            ok = 0;
        }
    }
    xil_printf("  黃金模型自我檢查：%s\r\n", ok ? "通過" : "失敗");
    return ok;
}


/* ================================================================
 *  測試圖產生（UYVY）
 *
 *  mode 0: 偽隨機                        一般性檢查
 *  mode 1: 飽和色（U/V 推到 0 或 255）   觸發 clamp 兩端
 *  mode 2: 水平漸層                      偵測輸出欄錯位
 *  mode 3: 垂直漸層                      偵測 row_in_block 錯位（最關鍵）
 *  mode 4: 彩條（8 條不同 U/V）          偵測色差共用與通道順序
 * ================================================================ */

static void gen_uyvy(uint8_t *uyvy, int mode)
{
    static const uint8_t bar_u[8] = { 90,  54, 166, 128, 128,  90, 202, 240};
    static const uint8_t bar_v[8] = {240,  34, 146, 128,  16, 110, 222, 110};
    uint32_t seed = 12345;
    int x, y;

    for (y = 0; y < SRC_H; y++) {
        for (x = 0; x < SRC_W; x += 2) {
            uint8_t *q = uyvy + (y * SRC_W + x) * 2;
            int u, y0, v, y1;

            switch (mode) {
            case 0:
                seed = seed * 1103515245u + 12345u; u  = (seed >> 16) & 0xFF;
                seed = seed * 1103515245u + 12345u; y0 = (seed >> 16) & 0xFF;
                seed = seed * 1103515245u + 12345u; v  = (seed >> 16) & 0xFF;
                seed = seed * 1103515245u + 12345u; y1 = (seed >> 16) & 0xFF;
                break;
            case 1:
                u  = ((x / 2 + y) & 1) ? 255 : 0;
                v  = ((x / 2 + y) & 2) ? 255 : 0;
                y0 = ((x + y) & 1) ? 235 : 16;
                y1 = 255 - y0;
                break;
            case 2:
                u = (x * 3) & 0xFF;        y0 = x & 0xFF;
                v = (255 - x * 2) & 0xFF;  y1 = (x + 1) & 0xFF;
                break;
            case 3:
                u = (y * 3) & 0xFF;        y0 = y & 0xFF;
                v = (255 - y * 2) & 0xFF;  y1 = (y * 5) & 0xFF;
                break;
            default: {
                int bar = (x * 8) / SRC_W;
                u = bar_u[bar]; v = bar_v[bar];
                y0 = y1 = 60 + bar * 20;
                break;
            }
            }
            q[0] = (uint8_t)u;
            q[1] = (uint8_t)y0;
            q[2] = (uint8_t)v;
            q[3] = (uint8_t)y1;
        }
    }
}


/* ================================================================
 *  執行一次 IP 並計時
 * ================================================================ */

static int run_ip(const case_cfg_t *cfg, uint64_t *counts_out)
{
    XTime t_start, t_end;
    uint32_t timeout;

    if (!XUyvy_resize_IsReady(&UyvyResizeInst)) {
        xil_printf("  ERROR: IP not ready\r\n");
        return XST_FAILURE;
    }

    /* ---- 設定參數 ---- */
    XUyvy_resize_Set_uyvy_axi_bus(&UyvyResizeInst, (u64)(uintptr_t)src_buf);
    XUyvy_resize_Set_rgb_axi_bus (&UyvyResizeInst, (u64)(uintptr_t)dst_buf);
    XUyvy_resize_Set_img_w       (&UyvyResizeInst, SRC_W);
    XUyvy_resize_Set_img_h       (&UyvyResizeInst, SRC_H);
    XUyvy_resize_Set_scale_mode  (&UyvyResizeInst, cfg->scale_mode);

    /* 讀回確認暫存器寫入正確（位址對應錯誤時最容易在這裡發現） */
    if (XUyvy_resize_Get_uyvy_axi_bus(&UyvyResizeInst) != (u64)(uintptr_t)src_buf ||
        XUyvy_resize_Get_rgb_axi_bus (&UyvyResizeInst) != (u64)(uintptr_t)dst_buf ||
        XUyvy_resize_Get_img_w       (&UyvyResizeInst) != SRC_W ||
        XUyvy_resize_Get_img_h       (&UyvyResizeInst) != SRC_H ||
        XUyvy_resize_Get_scale_mode  (&UyvyResizeInst) != (u32)cfg->scale_mode) {
        xil_printf("  ERROR: 參數讀回不符 (img_w=%d img_h=%d mode=%d)\r\n",
                   (int)XUyvy_resize_Get_img_w(&UyvyResizeInst),
                   (int)XUyvy_resize_Get_img_h(&UyvyResizeInst),
                   (int)XUyvy_resize_Get_scale_mode(&UyvyResizeInst));
        return XST_FAILURE;
    }

    /* ---- 啟動並等待 ---- */
    XTime_GetTime(&t_start);
    XUyvy_resize_Start(&UyvyResizeInst);

    timeout = 0xFFFFFFFFu;
    while (!XUyvy_resize_IsDone(&UyvyResizeInst)) {
        if (--timeout == 0) {
            xil_printf("  ERROR: timeout waiting for IP\r\n");
            return XST_FAILURE;
        }
    }
    XTime_GetTime(&t_end);

    XUyvy_resize_InterruptClear(&UyvyResizeInst, 1);

    *counts_out = (uint64_t)(t_end - t_start);
    return XST_SUCCESS;
}


/* ================================================================
 *  單一測試案例
 * ================================================================ */

static int run_case(const case_cfg_t *cfg, int mode, const char *name)
{
    int i, err_cnt = 0, first_err = -1, max_diff = 0;
    int ch_err[3] = {0, 0, 0};
    int pad_dirty = 0, guard_dirty = 0;
    int out_alloc = cfg->out_words * 16;         /* IP 實際寫出的範圍 */
    uint64_t counts = 0;
    uint32_t us;

    xil_printf("\r\n--- %d 倍 %s ---\r\n", cfg->scale, name);

    gen_uyvy(src_buf, mode);
    golden_uyvy_resize(src_buf, ref_buf, cfg);

    /* 輸出區清 0，其後放哨兵值偵測越界寫 */
    memset(dst_buf, 0, out_alloc);
    memset(dst_buf + out_alloc, GUARD_VAL, GUARD_BYTES);

    /* IP 透過 m_axi 直接讀寫 DDR，繞過 CPU cache。
     * 送出前把輸入與輸出區（含哨兵）flush 到 DDR。 */
    Xil_DCacheFlushRange((UINTPTR)src_buf, SRC_BYTES);
    Xil_DCacheFlushRange((UINTPTR)dst_buf, out_alloc + GUARD_BYTES);

    if (run_ip(cfg, &counts) != XST_SUCCESS)
        return XST_FAILURE;

    /* 收回結果前 invalidate，確保讀到 IP 寫的新值而非 cache 舊值 */
    Xil_DCacheInvalidateRange((UINTPTR)dst_buf, out_alloc + GUARD_BYTES);

    /* ---- 比對 ---- */
    for (i = 0; i < cfg->dst_bytes; i++) {
        int d = (int)dst_buf[i] - (int)ref_buf[i];
        if (d < 0) d = -d;
        if (d != 0) {
            err_cnt++;
            ch_err[i % 3]++;
            if (first_err < 0) first_err = i;
            if (d > max_diff) max_diff = d;
        }
    }

    /* 最後一個 word 的補位 byte 應為 0（IP 收尾補 0） */
    for (i = cfg->dst_bytes; i < out_alloc; i++)
        if (dst_buf[i] != 0) pad_dirty++;

    /* 哨兵區不應被寫到 */
    for (i = 0; i < GUARD_BYTES; i++)
        if (dst_buf[out_alloc + i] != GUARD_VAL) guard_dirty++;

    us = (uint32_t)(counts * 1000000ULL / COUNTS_PER_SECOND);
    xil_printf("  耗時 %lu counts (%lu us)\r\n", (unsigned long)counts, (unsigned long)us);
    xil_printf("  比對 %d / %d byte 不符\r\n", err_cnt, cfg->dst_bytes);
    if (pad_dirty)
        xil_printf("  ERROR: 最後一個 word 的補位區有 %d byte 非 0\r\n", pad_dirty);
    if (guard_dirty)
        xil_printf("  ERROR: 哨兵區有 %d byte 被改寫（IP 越界寫）\r\n", guard_dirty);

    if (err_cnt) {
        int px = first_err / 3;
        int start = px * 3;
        xil_printf("  各通道錯誤 R=%d G=%d B=%d, 最大差值 %d\r\n",
                   ch_err[0], ch_err[1], ch_err[2], max_diff);
        xil_printf("  首錯 byte %d (px %d, 座標 %d,%d, ch %d)\r\n",
                   first_err, px, px % cfg->dst_w, px / cfg->dst_w, first_err % 3);

        xil_printf("  附近資料 (ref | got):\r\n");
        for (i = start; i < start + 12 && i < cfg->dst_bytes; i += 3) {
            xil_printf("    px %3d: (%3d,%3d,%3d) | (%3d,%3d,%3d)%s\r\n",
                       i / 3,
                       ref_buf[i], ref_buf[i+1], ref_buf[i+2],
                       dst_buf[i], dst_buf[i+1], dst_buf[i+2],
                       (ref_buf[i] == dst_buf[i] &&
                        ref_buf[i+1] == dst_buf[i+1] &&
                        ref_buf[i+2] == dst_buf[i+2]) ? "" : "  <--");
        }

        /* 診斷提示 */
        if (ch_err[0] && ch_err[2] && !ch_err[1] && max_diff > 50)
            xil_printf("  提示: R 與 B 錯、G 對 -> 可能是 byte 順序 (RGB/BGR) 反了\r\n");
        else if (ch_err[0] == ch_err[1] && ch_err[1] == ch_err[2])
            xil_printf("  提示: 三通道錯誤數相同 -> 檢查 ox / row_in_block 或對齊狀態機\r\n");
        if (err_cnt == cfg->dst_bytes && max_diff > 100)
            xil_printf("  提示: 全錯且差值大 -> 檢查參數傳遞或 cache 操作\r\n");
        else if (max_diff <= 2)
            xil_printf("  提示: 差值很小 -> 檢查 uyvy2rgb 截斷方向或權重量化\r\n");
    }

    if (err_cnt || pad_dirty || guard_dirty)
        return XST_FAILURE;

    xil_printf("  PASS\r\n");
    return XST_SUCCESS;
}


/* ================================================================
 *  main
 * ================================================================ */

int main()
{
    int Status;
    int total = 0, passed = 0;
    int si;
    const int scales[2] = {3, 2};
    static const char *names[5] = {"偽隨機", "飽和色", "水平漸層", "垂直漸層", "彩條"};

    init_platform();

    xil_printf("\r\n");
    xil_printf("================================================\r\n");
    xil_printf("  XUyvy_resize IP 測試\r\n");
    xil_printf("================================================\r\n");
    xil_printf("  輸入 UYVY %dx%d，測試 3 倍與 2 倍\r\n", SRC_W, SRC_H);
    xil_printf("  src_buf @ 0x%08X, dst_buf @ 0x%08X\r\n",
               (unsigned)(uintptr_t)src_buf, (unsigned)(uintptr_t)dst_buf);
    xil_printf("  理論拍數 %d（一拍 8 pixel），%d MHz 約 %d us\r\n",
               SRC_W * SRC_H / 8, IP_CLK_MHZ, SRC_W * SRC_H / 8 / IP_CLK_MHZ);

    total++;
    if (self_check_golden()) passed++;

    /* ---- 初始化 IP ----
     * 傳統 flow 用 DEVICE_ID，SDT flow（Vitis 2023.2+）改用 BASEADDR。
     * 巨集名稱依 block design 的 instance 名稱而定，請對照 xparameters.h。 */
#ifdef SDT
    XUyvy_resize_Config *ConfigPtr =
        XUyvy_resize_LookupConfig(XPAR_UYVY_RESIZE_0_BASEADDR);
#else
    XUyvy_resize_Config *ConfigPtr =
        XUyvy_resize_LookupConfig(XPAR_UYVY_RESIZE_0_DEVICE_ID);
#endif
    if (!ConfigPtr) {
        xil_printf("ERROR: LookupConfig failed\r\n");
        return XST_FAILURE;
    }

    Status = XUyvy_resize_CfgInitialize(&UyvyResizeInst, ConfigPtr);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: CfgInitialize failed\r\n");
        return XST_FAILURE;
    }
    xil_printf("  IP 初始化完成\r\n");

    /* ---- 執行測試 ---- */
    for (si = 0; si < 2; si++) {
        case_cfg_t cfg;
        int m;

        make_cfg(scales[si], &cfg);

        xil_printf("\r\n================================================\r\n");
        xil_printf("  %d 倍：UYVY %dx%d -> RGB %dx%d  (scale_mode=%d)\r\n",
                   cfg.scale, SRC_W, SRC_H, cfg.dst_w, cfg.dst_h, cfg.scale_mode);
        xil_printf("  IP 內部推得：in_beats=%d out_words=%d%s\r\n",
                   SRC_BYTES / 16, cfg.out_words,
                   (cfg.dst_bytes % 16) ? "（最後一個 word 補 0）" : "");

        if (!check_size(&cfg)) {
            xil_printf("  跳過 %d 倍（尺寸不符限制）\r\n", cfg.scale);
            total += 5;
            continue;
        }

        for (m = 0; m < 5; m++) {
            total++;
            if (run_case(&cfg, m, names[m]) == XST_SUCCESS) passed++;
        }
    }

    xil_printf("\r\n================================================\r\n");
    xil_printf("  總計 %d / %d 通過\r\n", passed, total);
    xil_printf("================================================\r\n");

    if (passed != total) {
        xil_printf("\r\n除錯順序:\r\n");
        xil_printf("  0. 黃金模型自我檢查失敗 -> 先修測試程式，不要看 IP 結果\r\n");
        xil_printf("  1. 參數讀回不符   -> 檢查 xparameters.h 的 base address\r\n");
        xil_printf("  2. 全錯且輸出為 0 -> 檢查 cache flush/invalidate\r\n");
        xil_printf("  3. 全錯且值奇怪   -> 確認 bitstream 與 driver 是同一版 IP\r\n");
        xil_printf("  4. 部分錯         -> 檢查 ox / row_in_block 推進\r\n");
        xil_printf("  5. 補位或哨兵被寫 -> 檢查 IP 的 out_words 與收尾邏輯\r\n");
        xil_printf("  6. 耗時約理論兩倍 -> 檢查 gmem1 寫出端 burst 是否成立\r\n");
    }

    cleanup_platform();
    return (passed == total) ? XST_SUCCESS : XST_FAILURE;
}