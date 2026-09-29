/******************************************************************************
 * uyvy_resize.h
 *
 * AXI -> uyvy2rgb -> resize -> AXI 串接 top 的宣告
 *****************************************************************************/

#ifndef UYVY_RESIZE_H
#define UYVY_RESIZE_H

#include "ap_int.h"
#include "../../impl/resize_impl.h"     /* SCALE_2 / SCALE_3 */

/*
 * 限制
 *   img_w：16 的倍數（UYVY 一拍 8 pixel，repack 需成對）
 *          3 倍時還需是 48 的倍數（每次運算 6 pixel、輸出欄需成對）
 *   img_h：3 倍時為 3 的倍數，2 倍時為偶數
 *   輸出寬度 <= OUT_W_MAX (960)
 *     -> 2 倍 img_w <= 1920，3 倍 img_w <= 2880
 */
void uyvy_resize(ap_uint<128> *uyvy_axi_bus,
                 ap_uint<128> *rgb_axi_bus,
                 ap_uint<12>   img_w,
                 ap_uint<12>   img_h,
                 ap_uint<1>    scale_mode);

#endif /* UYVY_RESIZE_H */