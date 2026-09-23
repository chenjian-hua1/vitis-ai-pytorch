/******************************************************************************
 * uyvy2rgb_top.h
 *
 * AXI -> uyvy2rgb -> AXI 單獨 top 的宣告
 *   輸入 UYVY 4:2:2，128-bit（一拍 8 pixel）
 *   輸出 RGB888 packed，128-bit，記憶體 byte 順序 R, G, B（byte0 = R）
 *****************************************************************************/

#ifndef UYVY2RGB_TOP_H
#define UYVY2RGB_TOP_H

#include "ap_int.h"
#include "uyvy2rgb_impl.h"   /* dsp 型別 */

/* co-simulation 用的 m_axi depth（只影響模擬，不影響合成出來的硬體）
 * 取最大尺寸 4096 x 4096 */
#define MAX_IN     (4096 * 4096 / 8)          /* 輸入拍數：一拍 8 pixel */
#define MAX_OUT    (MAX_IN + MAX_IN / 2)      /* 輸出拍數：輸入 x 1.5 */

/*
 * 限制
 *   img_w：16 的倍數（一拍 8 pixel，192 -> 128 repack 需成對）
 *   img_w, img_h <= 4095（12-bit 埠）
 *
 * 輸入拍數  = img_w / 8 * img_h
 * 輸出拍數  = 輸入拍數 * 1.5
 */
void uyvy2rgb(ap_uint<128> *uyvy_axi_bus,
              ap_uint<128> *rgb_axi_bus,
              ap_uint<12>   img_w,
              ap_uint<12>   img_h);

#endif /* UYVY2RGB_TOP_H */
