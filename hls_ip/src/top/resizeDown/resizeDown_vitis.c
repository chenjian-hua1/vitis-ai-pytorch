/******************************************************************************
 * main.c
 *
 * XResize_kernel HLS IP 的 Vitis 裸機測試程式
 *
 * IP 功能：整數倍 Box-filter 縮小（2x / 3x），RGB888 packed
 *
 * 介面（對應 xresize_kernel.h，resize_top.h 版本）：
 *   in_ptr, out_ptr   m_axi（實體位址，需 cache flush/invalidate）
 *   img_w, img_h      輸入影像寬高（ap_uint<12>，1 ~ 4095）
 *   scale_mode        0 = 2 倍、1 = 3 倍
 *
 *   舊版的 total_words / total_results / out_words / out_w / inv_scale
 *   已改由 IP 內部從 img_w / img_h / scale_mode 算出，driver 不再提供
 *   對應的 Set 函式。
 *
 * 尺寸限制（見 resize_top.h）：
 *   img_w * img_h 為 16 的倍數（輸入 byte 數為 128-bit 整數倍）
 *   3 倍：img_w 為 6 的倍數、img_h 為 3 的倍數
 *   2 倍：img_w 為 8 的倍數、img_h 為偶數
 *   輸出寬度 <= 960
 *
 * 注意事項：
 *   1. m_axi 走實體位址，緩衝區必須 32-byte 對齊
 *   2. 若啟用 D-Cache，送出前 flush 輸入、收回前 invalidate 輸出
 *   3. 輸出 byte 數不是 16 的倍數時，IP 會把最後一個 word 補 0 寫出，
 *      所以輸出緩衝區要配到 OUT_WORDS * 16 byte
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
#include "xresize_kernel.h"

/* ---------------------------------------------------------------- 測試參數 */

/* 輸入尺寸固定，兩種縮小倍率在執行時各跑一次。
 * 先用小尺寸（例如 96x36）驗證功能，通過後再換成 1920x1080。 */
#define SRC_W       1920
#define SRC_H       1080

#define SCALE_MODE_2  0
#define SCALE_MODE_3  1

#define OUT_W_MAX   960      /* IP 內 line buffer 上限 */
#define IMG_DIM_MAX 4095     /* img_w / img_h 為 ap_uint<12> */

#define SRC_BYTES   (SRC_W * SRC_H * 3)

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

XResize_kernel ResizeInst;


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
 *  尺寸合法性檢查（對應 resize_top.h 的限制）
 * ================================================================ */

static int check_size(const case_cfg_t *cfg)
{
    int ok = 1;

    if (SRC_W < 1 || SRC_W > IMG_DIM_MAX || SRC_H < 1 || SRC_H > IMG_DIM_MAX) {
        xil_printf("  ERROR: img_w/img_h 超出 12-bit 範圍 (1 ~ %d)\r\n", IMG_DIM_MAX);
        ok = 0;
    }
    if ((SRC_W * SRC_H) % 16) {
        xil_printf("  ERROR: img_w * img_h = %d 不是 16 的倍數\r\n", SRC_W * SRC_H);
        ok = 0;
    }
    if (cfg->scale == 3) {
        if (SRC_W % 6) { xil_printf("  ERROR: 3 倍時 img_w 需為 6 的倍數\r\n"); ok = 0; }
        if (SRC_H % 3) { xil_printf("  ERROR: 3 倍時 img_h 需為 3 的倍數\r\n"); ok = 0; }
    } else {
        if (SRC_W % 8) { xil_printf("  ERROR: 2 倍時 img_w 需為 8 的倍數\r\n"); ok = 0; }
        if (SRC_H % 2) { xil_printf("  ERROR: 2 倍時 img_h 需為偶數\r\n");     ok = 0; }
    }
    if (cfg->dst_w > OUT_W_MAX) {
        xil_printf("  ERROR: 輸出寬度 %d 超過 %d\r\n", cfg->dst_w, OUT_W_MAX);
        ok = 0;
    }
    return ok;
}


/* ================================================================
 *  黃金參考模型
 *
 *  使用與 IP 等價的定點運算 (sum * inv_scale) >> 16，
 *  而非浮點除法，否則會因捨入方式不同產生 ±1 差異。
 *    3 倍 inv = 7282   <-> IP 16-bit scale_rate 7282，取 [23:16]
 *    2 倍 inv = 16384  <-> IP 2-bit  scale_rate 1，   取 [9:2]（= sum >> 2）
 * ================================================================ */

static void golden_resize(const uint8_t *src, uint8_t *dst, const case_cfg_t *cfg)
{
    int x, y, dx, dy, c;
    int s = cfg->scale;

    for (y = 0; y < cfg->dst_h; y++) {
        for (x = 0; x < cfg->dst_w; x++) {
            uint32_t sum[3] = {0, 0, 0};

            for (dy = 0; dy < s; dy++) {
                for (dx = 0; dx < s; dx++) {
                    const uint8_t *p =
                        src + ((y * s + dy) * SRC_W + (x * s + dx)) * 3;
                    for (c = 0; c < 3; c++)
                        sum[c] += p[c];
                }
            }

            uint8_t *q = dst + (y * cfg->dst_w + x) * 3;
            for (c = 0; c < 3; c++)
                q[c] = (uint8_t)((sum[c] * (uint32_t)cfg->inv_scale) >> 16);
        }
    }
}


/* ================================================================
 *  測試圖產生
 *
 *  mode 0: 偽隨機（一般性檢查）
 *  mode 1: RGB 通道刻意錯開（偵測通道污染）
 *  mode 2: 水平漸層
 *  mode 3: 垂直漸層（偵測輸出欄錯位，最關鍵的一種）
 * ================================================================ */

static void gen_image(uint8_t *img, int mode)
{
    int x, y;
    uint32_t seed = 12345;

    for (y = 0; y < SRC_H; y++) {
        for (x = 0; x < SRC_W; x++) {
            uint8_t *p = img + (y * SRC_W + x) * 3;
            switch (mode) {
            case 0:
                seed = seed * 1103515245u + 12345u;
                p[0] = (seed >> 16) & 0xFF;
                seed = seed * 1103515245u + 12345u;
                p[1] = (seed >> 16) & 0xFF;
                seed = seed * 1103515245u + 12345u;
                p[2] = (seed >> 16) & 0xFF;
                break;
            case 1:
                p[0] = 250; p[1] = 128; p[2] = 5;
                break;
            case 2:
                p[0] = (uint8_t)(x & 0xFF);
                p[1] = (uint8_t)((x * 2) & 0xFF);
                p[2] = (uint8_t)((x * 3) & 0xFF);
                break;
            default:
                p[0] = (uint8_t)(y & 0xFF);
                p[1] = (uint8_t)((y * 2) & 0xFF);
                p[2] = (uint8_t)((y * 3) & 0xFF);
                break;
            }
        }
    }
}


/* ================================================================
 *  執行一次 IP 並計時
 * ================================================================ */

static int run_ip(const case_cfg_t *cfg, uint64_t *cycles_out)
{
    XTime t_start, t_end;
    uint32_t timeout;

    if (!XResize_kernel_IsReady(&ResizeInst)) {
        xil_printf("  ERROR: IP not ready\r\n");
        return XST_FAILURE;
    }

    /* ---- 設定參數：只剩 5 個 ---- */
    XResize_kernel_Set_in_ptr    (&ResizeInst, (u64)(uintptr_t)src_buf);
    XResize_kernel_Set_out_ptr   (&ResizeInst, (u64)(uintptr_t)dst_buf);
    XResize_kernel_Set_img_w     (&ResizeInst, SRC_W);
    XResize_kernel_Set_img_h     (&ResizeInst, SRC_H);
    XResize_kernel_Set_scale_mode(&ResizeInst, cfg->scale_mode);

    /* 讀回確認暫存器寫入正確（位址對應錯誤時最容易在這裡發現） */
    if (XResize_kernel_Get_img_w(&ResizeInst)      != SRC_W ||
        XResize_kernel_Get_img_h(&ResizeInst)      != SRC_H ||
        XResize_kernel_Get_scale_mode(&ResizeInst) != (u32)cfg->scale_mode) {
        xil_printf("  ERROR: 參數讀回不符 (img_w=%d img_h=%d mode=%d)\r\n",
                   (int)XResize_kernel_Get_img_w(&ResizeInst),
                   (int)XResize_kernel_Get_img_h(&ResizeInst),
                   (int)XResize_kernel_Get_scale_mode(&ResizeInst));
        return XST_FAILURE;
    }

    /* ---- 啟動並等待 ---- */
    XTime_GetTime(&t_start);
    XResize_kernel_Start(&ResizeInst);

    timeout = 0xFFFFFFFFu;
    while (!XResize_kernel_IsDone(&ResizeInst)) {
        if (--timeout == 0) {
            xil_printf("  ERROR: timeout waiting for IP\r\n");
            return XST_FAILURE;
        }
    }
    XTime_GetTime(&t_end);

    XResize_kernel_InterruptClear(&ResizeInst, 1);

    *cycles_out = (uint64_t)(t_end - t_start);
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
    uint64_t cycles = 0;

    xil_printf("\r\n--- %d 倍 %s ---\r\n", cfg->scale, name);

    gen_image(src_buf, mode);
    golden_resize(src_buf, ref_buf, cfg);

    /* 輸出區清 0，其後放哨兵值偵測越界寫 */
    memset(dst_buf, 0, out_alloc);
    memset(dst_buf + out_alloc, GUARD_VAL, GUARD_BYTES);

    /* IP 透過 m_axi 直接讀寫 DDR，繞過 CPU cache。
     * 送出前把輸入與輸出區（含哨兵）flush 到 DDR。 */
    Xil_DCacheFlushRange((UINTPTR)src_buf, SRC_BYTES);
    Xil_DCacheFlushRange((UINTPTR)dst_buf, out_alloc + GUARD_BYTES);

    if (run_ip(cfg, &cycles) != XST_SUCCESS)
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

    xil_printf("  耗時 %lu counts (%lu us)\r\n",
               (unsigned long)cycles,
               (unsigned long)(cycles * 1000000ULL / COUNTS_PER_SECOND));
    xil_printf("  比對 %d / %d byte 不符\r\n", err_cnt, cfg->dst_bytes);
    if (pad_dirty)
        xil_printf("  ERROR: 最後一個 word 的補位區有 %d byte 非 0\r\n", pad_dirty);
    if (guard_dirty)
        xil_printf("  ERROR: 哨兵區有 %d byte 被改寫（IP 越界寫）\r\n", guard_dirty);

    if (err_cnt) {
        int px = first_err / 3;
        xil_printf("  各通道錯誤 R=%d G=%d B=%d, 最大差值 %d\r\n",
                   ch_err[0], ch_err[1], ch_err[2], max_diff);
        xil_printf("  首錯 byte %d (px %d, 座標 %d,%d, ch %d)\r\n",
                   first_err, px, px % cfg->dst_w, px / cfg->dst_w, first_err % 3);

        xil_printf("  附近資料 (ref | got):\r\n");
        int start = px * 3;
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
        if (ch_err[0] && !ch_err[1] && !ch_err[2])
            xil_printf("  提示: 只有 R 錯 -> 檢查通道切片位置\r\n");
        else if (ch_err[0] == ch_err[1] && ch_err[1] == ch_err[2])
            xil_printf("  提示: 三通道錯誤數相同 -> 檢查 ox / row_in_block\r\n");
        if (err_cnt == cfg->dst_bytes && max_diff > 100)
            xil_printf("  提示: 全錯且差值大 -> 檢查參數傳遞或 cache 操作\r\n");
        else if (max_diff <= 2)
            xil_printf("  提示: 差值很小 -> 可能只是定點捨入\r\n");
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
    static const char *names[4] = {"偽隨機", "通道污染", "水平漸層", "垂直漸層"};

    init_platform();

    xil_printf("\r\n");
    xil_printf("================================================\r\n");
    xil_printf("  XResize_kernel IP 測試\r\n");
    xil_printf("================================================\r\n");
    xil_printf("  輸入 %dx%d，測試 3 倍與 2 倍\r\n", SRC_W, SRC_H);
    xil_printf("  src_buf @ 0x%08X, dst_buf @ 0x%08X\r\n",
               (unsigned)(uintptr_t)src_buf, (unsigned)(uintptr_t)dst_buf);

    /* ---- 初始化 IP ----
     * 傳統 flow 用 DEVICE_ID，SDT flow（Vitis 2023.2+）改用 BASEADDR。
     * 巨集名稱依 block design 的 instance 名稱而定，請對照 xparameters.h。 */
#ifdef SDT
    XResize_kernel_Config *ConfigPtr =
        XResize_kernel_LookupConfig(XPAR_RESIZE_KERNEL_0_BASEADDR);
#else
    XResize_kernel_Config *ConfigPtr =
        XResize_kernel_LookupConfig(XPAR_RESIZE_KERNEL_0_DEVICE_ID);
#endif
    if (!ConfigPtr) {
        xil_printf("ERROR: LookupConfig failed\r\n");
        return XST_FAILURE;
    }

    Status = XResize_kernel_CfgInitialize(&ResizeInst, ConfigPtr);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: CfgInitialize failed\r\n");
        return XST_FAILURE;
    }
    xil_printf("  IP 初始化完成\r\n");

    /* ---- 執行測試 ----
     * 四種 pattern 缺一不可：
     *   單色與水平漸層在輸出欄錯位時剛好值相同，抓不到該類 bug，
     *   隨機與垂直漸層才能暴露空間錯位問題。 */
    for (si = 0; si < 2; si++) {
        case_cfg_t cfg;
        int m;

        make_cfg(scales[si], &cfg);

        xil_printf("\r\n================================================\r\n");
        xil_printf("  %d 倍：%dx%d -> %dx%d  (scale_mode=%d)\r\n",
                   cfg.scale, SRC_W, SRC_H, cfg.dst_w, cfg.dst_h, cfg.scale_mode);
        xil_printf("  IP 內部推得：total_words=%d out_words=%d%s\r\n",
                   SRC_BYTES / 16, cfg.out_words,
                   (cfg.dst_bytes % 16) ? "（最後一個 word 補 0）" : "");

        if (!check_size(&cfg)) {
            xil_printf("  跳過 %d 倍（尺寸不符限制）\r\n", cfg.scale);
            total += 4;
            continue;
        }

        for (m = 0; m < 4; m++) {
            total++;
            if (run_case(&cfg, m, names[m]) == XST_SUCCESS) passed++;
        }
    }

    xil_printf("\r\n================================================\r\n");
    xil_printf("  總計 %d / %d 通過\r\n", passed, total);
    xil_printf("================================================\r\n");

    if (passed != total) {
        xil_printf("\r\n除錯順序:\r\n");
        xil_printf("  1. 參數讀回不符   -> 檢查 xparameters.h 的 base address\r\n");
        xil_printf("  2. 全錯且輸出為 0 -> 檢查 cache flush/invalidate\r\n");
        xil_printf("  3. 全錯且值奇怪   -> 確認 bitstream 與 driver 是同一版 IP\r\n");
        xil_printf("  4. 部分錯         -> 檢查 ox / row_in_block 推進\r\n");
        xil_printf("  5. 補位或哨兵被寫 -> 檢查 IP 的 out_words 與收尾邏輯\r\n");
        xil_printf("  6. 差值 <= 2      -> 定點捨入，可接受\r\n");
    }

    cleanup_platform();
    return (passed == total) ? XST_SUCCESS : XST_FAILURE;
}