#include <opencv2/opencv.hpp>
#include <data_struct.h>

// ============================================================================
//  Fix / Float Conversion
// ============================================================================
void fix2float(const cv::Mat& data, int fix_point, cv::Mat& out);
void float2fix(const cv::Mat& data, int fix_point, cv::Mat& out);

// ============================================================================
//  Pre-Processing
// ============================================================================

void norm(const cv::Mat& x, cv::Mat& out);
void resize(const cv::Mat& img, int input_size, ResizeResult& res);
/**
 * @brief 將 uint8 BGR 影像做 ImageNet 正規化並量化為 int8 fix-point。
 *
 * 對每個 channel c 執行：
 *   dst[c] = clamp( src[c] * (kU8Scale[c] * 2^fix_point)
 *                           + kU8Bias[c]  * 2^fix_point,
 *                   -128, 127 )
 *
 * 等價於先做 norm（減 mean、除 std）再做 float2fix，
 * 但合併成單一 pass 避免中間 float buffer。
 *
 * @param x          輸入影像，必須為 CV_8UC3 (BGR)
 * @param fix_point  DPU input tensor 的 fix-point exponent
 *                   (從 xir::Tensor::get_attr<int>("fix_point") 取得)
 * @param out        輸出影像，CV_8SC3，與 x 同尺寸
 */
void norm_and_fix(const cv::Mat& x, int fix_point, cv::Mat& out);