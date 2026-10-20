#pragma once
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "../../impl/resize_impl.h"

void uyvy_resize(hls::stream<ap_axiu<128,0,0,0> > &in_stream,
                hls::stream<ap_axiu<128,0,0,0> > &out_stream,
                ap_uint<12> img_w, ap_uint<12> img_h, ap_uint<1> scale_mode);