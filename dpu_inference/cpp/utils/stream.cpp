#include "stream.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>

// ============================================================================
//  排查用的環境變數
//
//  改了 pipeline 之後收不到影像時,用這些逐項退回,就能二分出是哪一個
//  改動造成的,不用重編:
//
//    STREAM_PIPELINE="..."   完全自訂 pipeline,忽略下面所有選項
//    STREAM_LEGACY=1         用改動前的原始 pipeline(最快的對照組)
//    STREAM_NO_LIVE=1        拿掉 is-live / do-timestamp / format=time
//    STREAM_NO_QUEUE=1       拿掉 queue
//    STREAM_NO_I420=1        拿掉 videoconvert 之後的 I420 capsfilter(會壞)
//    STREAM_SINK_ASYNC=1     udpsink 不加 async=false
//
//  *** 不要拿掉 I420 那個 capsfilter ***
//
//  沒有它的話 videoconvert 會讓 BGR 直接進 jpegenc,而 jpegenc 收到
//  packed RGB/BGR 時會編成 4:4:4 的 JPEG。但 RFC 2435(RTP/JPEG)只
//  定義了 4:2:0(type 1)與 4:2:2(type 0),rtpjpegpay 遇到 4:4:4 會
//  報 "Unsupported sampling factors" 然後一個封包都不送 —— 發送端看
//  起來完全正常,接收端什麼都收不到。
//
//  這也是為什麼幾乎所有 RTP JPEG 的範例都會明確寫這一段 caps。
// ============================================================================

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && std::string(v) != "0";
}

std::string build_pipeline(const std::string& ip, int port, int quality)
{
    if (const char* custom = std::getenv("STREAM_PIPELINE"))
        if (*custom) return custom;

    if (env_on("STREAM_LEGACY")) {
        // 改動前的原版,一字不改
        return "appsrc ! video/x-raw,format=BGR ! videoconvert ! "
               "video/x-raw,format=I420 ! jpegenc quality=" +
               std::to_string(quality) +
               " ! rtpjpegpay mtu=1400 ! udpsink host=" + ip +
               " port=" + std::to_string(port) + " sync=false";
    }

    std::ostringstream p;
    // 時間戳交給 OpenCV 的 writer 管就好。appsrc 的 do-timestamp 會用
    // 時鐘的 running time 覆蓋掉 writer 寫入的 PTS,兩者打架沒有好處,
    // 而且 udpsink 已經 sync=false,本來就不靠時間戳排程。
    p << "appsrc block=false";
    if (env_on("STREAM_LIVE"))
        p << " is-live=true do-timestamp=true format=time";
    p << " ! video/x-raw,format=BGR ";

    if (!env_on("STREAM_NO_QUEUE"))
        p << "! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 "
             "leaky=downstream ";

    p << "! videoconvert ";
    // 必要,不是最佳化:見上方說明
    if (!env_on("STREAM_NO_I420"))
        p << "! video/x-raw,format=I420 ";

    p << "! jpegenc quality=" << quality << " "
      << "! rtpjpegpay mtu=1400 "
      << "! udpsink host=" << ip << " port=" << port << " sync=false";
    if (!env_on("STREAM_SINK_ASYNC"))
        p << " async=false";

    return p.str();
}

}  // namespace


RtpJpegStreamer::RtpJpegStreamer(int width, int height, double fps,
                                 const std::string& ip, int port, int quality)
    : width_(width), height_(height)
{
    const std::string pipeline = build_pipeline(ip, port, quality);
    std::cout << "[Stream] " << pipeline << std::endl;

    writer_.open(pipeline, cv::CAP_GSTREAMER, 0, fps,
                 cv::Size(width, height), true);

    if (!writer_.isOpened()) {
        std::cerr << "[Stream] pipeline 開啟失敗。用 GST_DEBUG=2 重跑可以看到"
                     " GStreamer 的錯誤(OpenCV 預設會吞掉),"
                     "或先用 STREAM_LEGACY=1 對照。" << std::endl;
    } else {
        std::cout << "[Stream] 已開啟。宣告 " << fps
                  << " fps,實際速率取決於 send() 的呼叫頻率。" << std::endl;
    }

    t_mark_ = now_ms();
}

RtpJpegStreamer::~RtpJpegStreamer() { close(); }

bool RtpJpegStreamer::isOpened() const { return writer_.isOpened(); }

bool RtpJpegStreamer::send(const cv::Mat& frame)
{
    if (frame.empty() || !writer_.isOpened()) return false;

    if (frame.cols != width_ || frame.rows != height_) {
        if (!warned_size_) {
            warned_size_ = true;
            std::cerr << "[Stream] 畫面 " << frame.cols << "x" << frame.rows
                      << " 與串流設定 " << width_ << "x" << height_ << " 不符";
            if (frame.cols < width_ || frame.rows < height_)
                std::cerr << " —— 而且是放大,編碼成本會大幅上升,"
                             "建議把串流尺寸設成 "
                          << frame.cols << "x" << frame.rows;
            std::cerr << std::endl;
        }
        cv::resize(frame, scaled_, cv::Size(width_, height_), 0, 0,
                   cv::INTER_NEAREST);
        writer_.write(scaled_);
    } else {
        writer_.write(frame);
    }

    if (++sent_ - n_mark_ >= 60) {
        const double t = now_ms();
        actual_fps_ = (sent_ - n_mark_) * 1000.0 / (t - t_mark_);
        t_mark_ = t;
        n_mark_ = sent_;
        if (sent_ <= 60)
            std::cout << "[Stream] 已送出 " << sent_ << " 幀,實際 "
                      << actual_fps_ << " fps" << std::endl;
    }
    return true;
}

void RtpJpegStreamer::close()
{
    if (writer_.isOpened()) writer_.release();
}