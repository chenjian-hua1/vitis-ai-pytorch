/******************************************************************************
 * resize_top.h
 *
 * AXI -> resize -> AXI 單獨 top 的宣告（輸入 RGB888 packed，128-bit）
 *****************************************************************************/

#ifndef RESIZE_TOP_H
#define RESIZE_TOP_H

#include "ap_int.h"
#include "resize_impl.h"     /* SCALE_2 / SCALE_3 */

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
