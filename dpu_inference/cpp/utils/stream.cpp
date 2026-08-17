#include "stream.h"
#include <iostream>

RtpJpegStreamer::RtpJpegStreamer(int width, int height, double fps,
                                 const std::string& ip, int port, int quality)
    : width_(width), height_(height)
{
    std::string pipeline =
        "appsrc ! "
        "video/x-raw,format=BGR ! "
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        "jpegenc quality=" + std::to_string(quality) + " ! "
        "rtpjpegpay mtu=1400 ! "
        "udpsink host=" + ip + " port=" + std::to_string(port) + " sync=false";

    // FourCC = 0，編碼交給 pipeline 裡的 jpegenc
    writer_.open(pipeline, cv::CAP_GSTREAMER, 0, fps,
                 cv::Size(width, height), true);
}

RtpJpegStreamer::~RtpJpegStreamer()
{
    close();
}

bool RtpJpegStreamer::isOpened() const
{
    return writer_.isOpened();
}

bool RtpJpegStreamer::send(const cv::Mat& frame)
{
    if (frame.empty() || !writer_.isOpened())
        return false;

    cv::Mat out = frame;
    // 尺寸不符時自動縮放，避免 pipeline 協商失敗
    if (frame.cols != width_ || frame.rows != height_) {
        cv::resize(frame, out, cv::Size(width_, height_), 0, 0, cv::INTER_NEAREST);
    }
    writer_.write(out);
    return true;
}

void RtpJpegStreamer::close()
{
    if (writer_.isOpened())
        writer_.release();
}