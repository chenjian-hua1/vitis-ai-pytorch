#include "quantize_top.h"
#include "../../impl/quantize_impl.h"

/**
 * @brief Kernel function for quantization
 * 
 * quantize LENGTH/24 pixels use quantize_core function
 * 
 * @param input input data of 24-bits/pixel
 * @param output quantized data of 24-bits/pixel
 */
template <int LENGTH>
void quantize_kernel(const ap_uint<LENGTH> &input, ap_uint<LENGTH> &output) {
    // Check LENGTH is a multiple of 24, if not print error message
    static_assert(LENGTH % 24 == 0,
                  "LENGTH must be a multiple of 24");

    // Compile time (Generate LEGTH/48  quantize cores)
    // Process two pixels per iteration. Since LENGTH is a multiple of 24,
    // there can be at most one remaining pixel.
    for (int i = 0; i < LENGTH / 48; i++) {
    #pragma HLS UNROLL
        const int base = 48 * i;
        ap_uint<48> quantize_output;
        quantize_core(input.range(base + 47, base), quantize_output);
        output.range(base + 47, base) = quantize_output;
    }

    // Compile time (Generate one quantize core for leftover pixel)
    // process leftover 1 pixel (24 bits)
    if constexpr (LENGTH % 48 == 24) {
        constexpr int base = (LENGTH / 48) * 48;
        ap_uint<48> last_pixels = 0;
        last_pixels.range(23, 0) = input.range(base + 23, base);

        ap_uint<48> quantize_output;
        quantize_core(last_pixels, quantize_output);
        output.range(base + 23, base) = quantize_output.range(23, 0);
    }
}

void quantize_top(const ap_uint<NUM_PIXEL * 24> &input,
                  ap_uint<NUM_PIXEL * 24> &output) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
    
    quantize_kernel<NUM_PIXEL * 24>(input, output);
}
