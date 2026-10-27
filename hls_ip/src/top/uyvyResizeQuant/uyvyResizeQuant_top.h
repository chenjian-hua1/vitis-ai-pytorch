#include "ap_int.h"

/** @brief UYVY to RGB resize and quantize kernel
 * 
 * @param uvy_axi_bus UYVY input data stream
 * @param rgb_axi_bus RGB output data stream
 * @param img_w image width
 * @param img_h image height
 * @param scale_mode 0 is 3 downscale, 1 is 2 downscale
 * 
 */
void uyvyResizeQuant(
    ap_uint<128> *uyvy_axi_bus,
    ap_uint<128> *rgb_axi_bus,
    ap_uint<128> *quant_axi_bus,
    ap_uint<12>   img_w,
    ap_uint<12>   img_h,
    ap_uint<1>    scale_mode
);