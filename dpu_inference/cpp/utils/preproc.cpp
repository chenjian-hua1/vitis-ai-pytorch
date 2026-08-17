#include <preproc.h>
#include <numeric>
#include <stdexcept>

namespace {

constexpr float kMeanR = 0.485f, kMeanG = 0.456f, kMeanB = 0.406f;
constexpr float kStdR  = 0.229f, kStdG  = 0.224f, kStdB  = 0.225f;

constexpr float kU8ScaleR = 1.f/(kStdR*255.f);
constexpr float kU8ScaleG = 1.f/(kStdG*255.f);
constexpr float kU8ScaleB = 1.f/(kStdB*255.f);
constexpr float kU8BiasR  = -kMeanR/kStdR;
constexpr float kU8BiasG  = -kMeanG/kStdG;
constexpr float kU8BiasB  = -kMeanB/kStdB;

constexpr float kF32ScaleR = 1.f/kStdR;
constexpr float kF32ScaleG = 1.f/kStdG;
constexpr float kF32ScaleB = 1.f/kStdB;

} // anonymous namespace

// ============================================================================
//  Fix / Float Conversion
// ============================================================================

void fix2float(const cv::Mat& data, int fix_point, cv::Mat& out)
{
    float scale = std::exp2f(-static_cast<float>(fix_point));
    data.convertTo(out, CV_32FC3, scale);
}

void float2fix(const cv::Mat& data, int fix_point, cv::Mat& out)
{
    float scale = std::exp2f(static_cast<float>(fix_point));
    data.convertTo(out, CV_8SC3, scale);
}

// ============================================================================
//  Pre-Processing
// ============================================================================

void norm(const cv::Mat& x, cv::Mat& out)
{
    CV_Assert(x.type() == CV_8UC3);

    out.create(x.rows, x.cols, CV_32FC3);

    const uchar* RESTRICT src = x.ptr<uchar>();
    float*       RESTRICT dst = out.ptr<float>();

    const int total = x.rows * x.cols;

    #pragma omp simd
    for (int i = 0; i < total; i++) {
        dst[3*i + 0] = src[3*i + 0] * kU8ScaleR + kU8BiasR;
        dst[3*i + 1] = src[3*i + 1] * kU8ScaleG + kU8BiasG;
        dst[3*i + 2] = src[3*i + 2] * kU8ScaleB + kU8BiasB;
    }
}

void norm_and_fix(const cv::Mat& x, int fix_point, cv::Mat& out)
{
    CV_Assert(x.type() == CV_8UC3);

    out.create(x.rows, x.cols, CV_8SC3);

    const float fp_scale = std::exp2f(static_cast<float>(fix_point));

    const float scaleR = kU8ScaleR * fp_scale;
    const float scaleG = kU8ScaleG * fp_scale;
    const float scaleB = kU8ScaleB * fp_scale;
    const float biasR  = kU8BiasR  * fp_scale;
    const float biasG  = kU8BiasG  * fp_scale;
    const float biasB  = kU8BiasB  * fp_scale;

    const uchar*  RESTRICT src = x.ptr<uchar>();
    int8_t*       RESTRICT dst = out.ptr<int8_t>();

    const int total = x.rows * x.cols;

    #pragma omp simd
    for (int i = 0; i < total; ++i) {
        dst[3*i + 0] = static_cast<int8_t>(std::clamp(src[3*i + 0] * scaleR + biasR, -128.f, 127.f));
        dst[3*i + 1] = static_cast<int8_t>(std::clamp(src[3*i + 1] * scaleG + biasG, -128.f, 127.f));
        dst[3*i + 2] = static_cast<int8_t>(std::clamp(src[3*i + 2] * scaleB + biasB, -128.f, 127.f));
    }
}

void resize(const cv::Mat& img, int input_size, ResizeResult& res)
{
    const int orig_h = img.rows, orig_w = img.cols;

    float r = std::min(static_cast<float>(input_size) / orig_h,
                    static_cast<float>(input_size) / orig_w);
    r = std::min(r, 1.0f);

    int pad_w = static_cast<int>(std::round(orig_w * r));
    int pad_h = static_cast<int>(std::round(orig_h * r));

    int dw = (input_size - pad_w) / 2.0f;
    int dh = (input_size - pad_h) / 2.0f;

    int top    = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left   = static_cast<int>(std::round(dw - 0.1f));
    int right  = static_cast<int>(std::round(dw + 0.1f));

    res.ratio = {r,  r};
    res.pad   = {dw, dh};
    res.content = cv::Rect(left, top, pad_w, pad_h);

    res.img.create(input_size, input_size, img.type());
    res.img.setTo(cv::Scalar(0,0,0));

    cv::Mat roi = res.img(cv::Rect(left, top, pad_w, pad_h));
    // cv::resize(img, roi, roi.size(), 0, 0, cv::INTER_LINEAR);
    cv::resize(img, roi, roi.size(), 0, 0, cv::INTER_AREA);
}
