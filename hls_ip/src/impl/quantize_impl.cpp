#include "quantize_impl.h"
#include "quantize_lut.h"

void quantize_core(const ap_uint<48> &input, ap_uint<48> &output) {
#pragma HLS BIND_STORAGE variable=kLut.r type=rom_2p impl=lutram
#pragma HLS BIND_STORAGE variable=kLut.g type=rom_2p impl=lutram
#pragma HLS BIND_STORAGE variable=kLut.b type=rom_2p impl=lutram

    // 48 bits = 2 pixel 
    // Pixel 24bit: [R:23..16][G:15..8][B:7..0]
    for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
        const int base = 24 * i;
        output.range(base + 23, base + 16) = kLut.r[input.range(base + 23, base + 16)];
        output.range(base + 15, base + 8)  = kLut.g[input.range(base + 15, base + 8)];
        output.range(base + 7,  base)      = kLut.b[input.range(base + 7,  base)];
    }
}