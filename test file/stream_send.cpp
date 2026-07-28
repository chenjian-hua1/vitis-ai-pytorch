#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    int width = 640;
    int height = 480;
    double fps = 30.0;

    std::string gst_pipeline =
        "appsrc ! "
        "video/x-raw,format=BGR ! "
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        "jpegenc quality=85 ! "
        "rtpjpegpay mtu=1400 ! "
        "udpsink host=192.168.1.100 port=5000 sync=false";

    cv::VideoWriter writer;
    writer.open(gst_pipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);

    if (!writer.isOpened()) {
        std::cerr << "錯誤：無法開啟 GStreamer 寫入管線。請確認 OpenCV 編譯時有啟用 GStreamer。" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "錯誤：無法開啟攝影機。" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    std::cout << "開始傳送 JPEG RTP 串流..." << std::endl;
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "影格為空，停止傳送。" << std::endl;
            break;
        }
        writer.write(frame);
    }

    cap.release();
    writer.release();
    return 0;
}