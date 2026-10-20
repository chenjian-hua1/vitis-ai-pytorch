#include <ap_int.h>

/**
 * @brief two pixel quantization core (lut method)
 * 
 * Two Pixel[c] share one lut[c] (2 Ports ROM)
 *
 * Quantize Step : input -> normalize -> float2fix -> output
 * 
 * 
 * 
 * uint8 normalize (ImageNet) :
 * 
 * norm[R] = input[R] * 0.01712475345 + -2.117903948
 * 
 * norm[G] = input[G] * 0.01750700176 + -2.035714149
 * 
 * norm[B] = input[B] * 0.01742919348 + -1.804444432
 * 
 * float2fix :
 * 
 * output = norm * 2^fix_point (norm << fix_point)
 * 
 * @param input the uint8 RGB pixel
 * 
 * @param output the quantized uint8 RGB pixel
 * 
 * @param fix_point the fix point of the quantized output
 *
 */
void quantize_core(const ap_uint<48> &input, ap_uint<48> &output);
