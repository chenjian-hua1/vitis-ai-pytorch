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
    // 介面與原版相同,新增的參數都有預設值,呼叫端不用改。
    //
    // fps 的意義:它只是寫進 caps 的「宣告值」,不控制實際速率 ——
    // 影像多快送出完全取決於你多快呼叫 send()。
    // 但宣告值仍然重要:接收端若 sync=true 會照它排程播放,
    // 宣告與實際不符就會累積延遲或卡頓。
    RtpJpegStreamer(int width, int height, double fps,
                    const std::string& ip, int port, int quality = 85);

    ~RtpJpegStreamer();

    bool isOpened() const;
    bool send(const cv::Mat& frame);
    void close();

    // 實際送出的速率(每 60 幀更新一次),用來和宣告的 fps 對照
    double actualFps() const { return actual_fps_; }
    long long sentFrames() const { return sent_; }

private:
    cv::VideoWriter writer_;
    int     width_, height_;
    bool    warned_size_ = false;
    cv::Mat scaled_;

    long long sent_ = 0;
    double    t_mark_ = 0;
    long long n_mark_ = 0;
    double    actual_fps_ = 0;
};

#endif // STREAM_H