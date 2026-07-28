#ifndef STREAM_H
#define STREAM_H

#include <opencv2/opencv.hpp>
#include <string>

struct stream_params
{
    std::string ip;
    int port, width, height;
    double fps;
    int quality = 85;
};

class RtpJpegStreamer {
public:
    // 建構時就建立 pipeline(只做一次)
    RtpJpegStreamer(int width, int height, double fps,
                    const std::string& ip, int port, int quality = 85);

    // 解構 -> 關閉 writer
    ~RtpJpegStreamer();

    bool isOpened() const;

    // 每格呼叫這個，只吃 Mat
    bool send(const cv::Mat& frame);

    void close();

private:
    cv::VideoWriter writer_;
    int width_, height_;
};

#endif // STREAM_H