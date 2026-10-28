#include "quantize_impl.h"
#include "quantize_lut.h"

void quantize_core(const ap_uint<48> &input, ap_uint<48> &output) {
#pragma HLS BIND_STORAGE variable=kLut.r type=rom_np impl=lutram
#pragma HLS BIND_STORAGE variable=kLut.g type=rom_np impl=lutram
#pragma HLS BIND_STORAGE variable=kLut.b type=rom_np impl=lutram

    // 48 bits = 2 pixel，記憶體 byte 順序（byte0 = R）
    // Pixel 24bit: [B:23..16][G:15..8][R:7..0]
    for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
        const int base = 24 * i;
        output.range(base + 7,  base)      = kLut.r[input.range(base + 7,  base)];       // R
        output.range(base + 15, base + 8)  = kLut.g[input.range(base + 15, base + 8)];   // G
        output.range(base + 23, base + 16) = kLut.b[input.range(base + 23, base + 16)];  // B
    }
}