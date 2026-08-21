/******************************************************************************
 * main.c
 *
 * XResize_kernel HLS IP 的 Vitis 裸機測試程式
 *
 * IP 功能：整數倍 Box-filter 縮小（2x / 3x），RGB888 packed
 *
 * 介面：
 *   in_ptr, out_ptr   m_axi（實體位址，需 cache flush/invalidate）
 *   total_words       輸入的 128-bit word 數 = W * H * 3 / 16
 *   total_results     運算次數 = (out_w * out_h) / n_out
 *   out_words         輸出的 128-bit word 數 = out_w * out_h * 3 / 16
 *   out_w             輸出寬度（3 倍需 2 的倍數、2 倍需 4 的倍數）
 *   scale_mode        0 = 2 倍、1 = 3 倍
 *   inv_scale         65536 / (scale^2)：3 倍 = 7282、2 倍 = 16384
 *
 * 注意事項：
 *   1. m_axi 走實體位址，緩衝區必須 32-byte 對齊
 *   2. 若啟用 D-Cache，送出前 flush 輸入、收回前 invalidate 輸出
 *   3. total_results 是「運算次數」不是像素數
 *   4. out_words 是輸出的 word 數，與 total_results 不同：
 *        3 倍 640x360：total_results = 115200、out_words = 43200
 *        2 倍 960x540：total_results = 129600、out_words = 97200
 *****************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "platform.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xtime_l.h"
#include "xparameters.h"
#include "xresize_kernel.h"

/* ---------------------------------------------------------------- 測試參數 */

/* 先用小尺寸驗證功能，通過後再換成 1920x1080。
 * 小圖的好處是可以把整張結果印出來人工檢查。 */
#define SRC_W       1920
#define SRC_H       1080
#define SCALE       3        /* 3 或 2 */

#define SCALE_MODE_2  0
#define SCALE_MODE_3  1

#define DST_W       (SRC_W / SCALE)
#define DST_H       (SRC_H / SCALE)
#define N_OUT       ((SCALE == 3) ? 2 : 4)   /* 一次運算產出幾個輸出欄 */

#define SRC_BYTES   (SRC_W * SRC_H * 3)
#define DST_BYTES   (DST_W * DST_H * 3)

#define TOTAL_WORDS   (SRC_BYTES / 16)
#define TOTAL_RESULTS ((DST_W * DST_H) / N_OUT)   /* 運算次數 */
/* 輸出 128-bit word 數。用向上取整：pack_side 收尾時若 acc 還有
 * 殘餘位元會多寫一個 word，少算會漏掉最後一筆。
 * 常見尺寸（640x360、960x540）皆整除，此分支不會觸發。 */
#define OUT_WORDS     ((DST_BYTES + 15) / 16)
#define INV_SCALE     ((SCALE == 3) ? 7282 : 16384)
#define SCALE_MODE    ((SCALE == 3) ? SCALE_MODE_3 : SCALE_MODE_2)

/* 32-byte 對齊，配合 AXI burst 與 cache line */
static uint8_t src_buf[SRC_BYTES + 32] __attribute__((aligned(32)));
static uint8_t dst_buf[DST_BYTES + 32] __attribute__((aligned(32)));
static uint8_t ref_buf[DST_BYTES]      __attribute__((aligned(32)));

XResize_kernel ResizeInst;


/* ================================================================
 *  黃金參考模型
 *
 *  使用與 IP 相同的定點運算 (sum * inv_scale) >> 16，
 *  而非浮點除法，否則會因捨入方式不同產生 ±1 差異。
 * ================================================================ */

static void golden_resize(const uint8_t *src, uint8_t *dst)
{
    int x, y, dx, dy, c;

    for (y = 0; y < DST_H; y++) {
        for (x = 0; x < DST_W; x++) {
            uint32_t sum[3] = {0, 0, 0};

            for (dy = 0; dy < SCALE; dy++) {
                for (dx = 0; dx < SCALE; dx++) {
                    const uint8_t *p =
                        src + ((y * SCALE + dy) * SRC_W + (x * SCALE + dx)) * 3;
                    for (c = 0; c < 3; c++)
                        sum[c] += p[c];
                }
            }

            uint8_t *q = dst + (y * DST_W + x) * 3;
            for (c = 0; c < 3; c++)
                q[c] = (uint8_t)((sum[c] * INV_SCALE) >> 16);
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

static int run_ip(uint64_t *cycles_out)
{
    XTime t_start, t_end;
    uint32_t timeout;

    if (!XResize_kernel_IsReady(&ResizeInst)) {
        xil_printf("  ERROR: IP not ready\r\n");
        return XST_FAILURE;
    }

    /* ---- 設定參數 ---- */
    XResize_kernel_Set_in_ptr       (&ResizeInst, (u64)(uintptr_t)src_buf);
    XResize_kernel_Set_out_ptr      (&ResizeInst, (u64)(uintptr_t)dst_buf);
    XResize_kernel_Set_total_words  (&ResizeInst, TOTAL_WORDS);
    XResize_kernel_Set_total_results(&ResizeInst, TOTAL_RESULTS);
    XResize_kernel_Set_out_words    (&ResizeInst, OUT_WORDS);
    XResize_kernel_Set_out_w        (&ResizeInst, DST_W);
    XResize_kernel_Set_scale_mode   (&ResizeInst, SCALE_MODE);
    XResize_kernel_Set_inv_scale    (&ResizeInst, INV_SCALE);

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

static int run_case(int mode, const char *name)
{
    int i, err_cnt = 0, first_err = -1, max_diff = 0;
    int ch_err[3] = {0, 0, 0};
    uint64_t cycles = 0;

    xil_printf("\r\n--- %s ---\r\n", name);

    gen_image(src_buf, mode);
    golden_resize(src_buf, ref_buf);
    memset(dst_buf, 0, DST_BYTES);

    /* IP 透過 m_axi 直接讀寫 DDR，繞過 CPU cache。
     * 送出前把輸入 flush 到 DDR，避免 IP 讀到舊資料。 */
    Xil_DCacheFlushRange((UINTPTR)src_buf, SRC_BYTES);
    Xil_DCacheFlushRange((UINTPTR)dst_buf, DST_BYTES);

    if (run_ip(&cycles) != XST_SUCCESS)
        return XST_FAILURE;

    /* 收回結果前 invalidate，確保讀到 IP 寫的新值而非 cache 舊值 */
    Xil_DCacheInvalidateRange((UINTPTR)dst_buf, DST_BYTES);

    /* ---- 比對 ---- */
    for (i = 0; i < DST_BYTES; i++) {
        int d = (int)dst_buf[i] - (int)ref_buf[i];
        if (d < 0) d = -d;
        if (d != 0) {
            err_cnt++;
            ch_err[i % 3]++;
            if (first_err < 0) first_err = i;
            if (d > max_diff) max_diff = d;
        }
    }

    xil_printf("  耗時 %lu counts (%lu us)\r\n",
               (unsigned long)cycles,
               (unsigned long)(cycles * 1000000ULL / COUNTS_PER_SECOND));
    xil_printf("  比對 %d / %d byte 不符\r\n", err_cnt, DST_BYTES);

    if (err_cnt) {
        int px = first_err / 3;
        xil_printf("  各通道錯誤 R=%d G=%d B=%d, 最大差值 %d\r\n",
                   ch_err[0], ch_err[1], ch_err[2], max_diff);
        xil_printf("  首錯 byte %d (px %d, 座標 %d,%d, ch %d)\r\n",
                   first_err, px, px % DST_W, px / DST_W, first_err % 3);

        xil_printf("  附近資料 (ref | got):\r\n");
        int start = px * 3;
        for (i = start; i < start + 12 && i < DST_BYTES; i += 3) {
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
        if (err_cnt == DST_BYTES && max_diff > 100)
            xil_printf("  提示: 全錯且差值大 -> 檢查參數傳遞或 cache 操作\r\n");
        else if (max_diff <= 2)
            xil_printf("  提示: 差值很小 -> 可能只是定點捨入\r\n");

        return XST_FAILURE;
    }

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

    init_platform();

    xil_printf("\r\n");
    xil_printf("================================================\r\n");
    xil_printf("  XResize_kernel IP 測試\r\n");
    xil_printf("================================================\r\n");
    xil_printf("  輸入 %dx%d -> 輸出 %dx%d  (scale=%d)\r\n",
               SRC_W, SRC_H, DST_W, DST_H, SCALE);
    xil_printf("  total_words   = %d\r\n", TOTAL_WORDS);
    xil_printf("  total_results = %d\r\n", TOTAL_RESULTS);
    xil_printf("  out_words     = %d\r\n", OUT_WORDS);
    xil_printf("  out_w         = %d\r\n", DST_W);
    xil_printf("  scale_mode    = %d\r\n", SCALE_MODE);
    xil_printf("  inv_scale     = %d\r\n", INV_SCALE);
    xil_printf("  src_buf @ 0x%08X, dst_buf @ 0x%08X\r\n",
               (unsigned)(uintptr_t)src_buf, (unsigned)(uintptr_t)dst_buf);

    /* ---- 參數合法性檢查 ---- */
    if (SRC_W % SCALE || SRC_H % SCALE) {
        xil_printf("  ERROR: 尺寸不能被 scale 整除\r\n");
        return XST_FAILURE;
    }
    if (SRC_BYTES % 16) {
        xil_printf("  ERROR: 輸入位元組數 %d 不是 16 的倍數\r\n", SRC_BYTES);
        return XST_FAILURE;
    }
    if (DST_W % N_OUT) {
        xil_printf("  ERROR: out_w=%d 必須是 %d 的倍數\r\n", DST_W, N_OUT);
        return XST_FAILURE;
    }
    if (DST_BYTES % 16) {
        xil_printf("  WARN: 輸出 %d byte 非 16 的倍數，"
                   "最後一個 word 含殘餘位元\r\n", DST_BYTES);
    }

    /* ---- 初始化 IP ---- */
    XResize_kernel_Config *ConfigPtr =
        XResize_kernel_LookupConfig(XPAR_RESIZE_KERNEL_0_DEVICE_ID);
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
    total++; if (run_case(0, "偽隨機")     == XST_SUCCESS) passed++;
    total++; if (run_case(1, "通道污染")   == XST_SUCCESS) passed++;
    total++; if (run_case(2, "水平漸層")   == XST_SUCCESS) passed++;
    total++; if (run_case(3, "垂直漸層")   == XST_SUCCESS) passed++;

    xil_printf("\r\n================================================\r\n");
    xil_printf("  總計 %d / %d 通過\r\n", passed, total);
    xil_printf("================================================\r\n");

    if (passed != total) {
        xil_printf("\r\n除錯順序:\r\n");
        xil_printf("  1. 全錯且輸出為 0 -> 檢查 cache flush/invalidate\r\n");
        xil_printf("  2. 全錯且值奇怪   -> 檢查參數是否對齊 IP 簽名\r\n");
        xil_printf("  3. 部分錯         -> 檢查 ox / row_in_block 推進\r\n");
        xil_printf("  4. 差值 <= 2      -> 定點捨入，可接受\r\n");
    }

    cleanup_platform();
    return (passed == total) ? XST_SUCCESS : XST_FAILURE;
}
