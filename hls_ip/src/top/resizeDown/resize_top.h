/******************************************************************************
 * resize_top.h
 *
 * AXI -> resize -> AXI 單獨 top 的宣告（輸入 RGB888 packed，128-bit）
 *****************************************************************************/

#ifndef RESIZE_TOP_H
#define RESIZE_TOP_H

#include "ap_int.h"
#include "../../impl/resize_impl.h" /* SCALE_2 / SCALE_3 */

// 計算該數字需要使用多少位元
constexpr int LOG2_CEIL(int x) {
    // 定義域：x >= 1
    // ceil(log2(1)) = 0, ceil(log2(2)) = 1, ceil(log2(3)) = 2, ...
    int r = 0;
    int p = 1;
    // 直到 2^r 超過x
    while (p <= x) {
        p*=2;
        ++r;
    }
    return r;
}

// 2^? 形式其中一個bit=1 其他0
constexpr bool IS_POW2(int x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

// 計算 for index (0..maxium-1) 需要的位元寬度
// 規則：若 maxium 是 2 的冪次方 -> LOG2_CEIL(maxium + 1)
// 否則 -> LOG2_CEIL(maxium)
constexpr int FOR_IDX_BITS(int maxium) {
    return IS_POW2(maxium) ? LOG2_CEIL(maxium + 1) : LOG2_CEIL(maxium);
}

/*
 * 限制
 *   img_w * img_h 為 16 的倍數（輸入 RGB 總 byte 數為 128-bit 整數倍）
 *   3 倍：img_w 為 6 的倍數、img_h 為 3 的倍數
 *   2 倍：img_w 為 8 的倍數、img_h 為偶數
 *   輸出寬度 <= OUT_W_MAX (960)
 */
void resize_kernel(ap_uint<128> *in_ptr,
                   ap_uint<128> *out_ptr,
                   ap_uint<12>   img_w,
                   ap_uint<12>   img_h,
                   ap_uint<1>    scale_mode);

#endif /* RESIZE_TOP_H */
