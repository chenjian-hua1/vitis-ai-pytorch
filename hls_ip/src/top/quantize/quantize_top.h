#pragma once

#include <ap_int.h>

// Number of RGB pixels processed by one top-level invocation.
constexpr int NUM_PIXEL = 7;

/**
 * @brief HLS top: one invocation processes NUM_PIXEL RGB pixels in parallel.
 * 
 * Generate parrallel ceil(NUM_PIXEL/2) quantize cores
 *
 * @param input 24-bit RGB pixels
 * @param output quantized 24-bit RGB pixels 
 */
void quantize_top(const ap_uint<NUM_PIXEL * 24> &input,
                  ap_uint<NUM_PIXEL * 24> &output);
