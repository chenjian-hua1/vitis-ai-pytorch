// opencv_capture.cpp
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#define WIDTH   1280
#define HEIGHT  720
#define NBUF    4
#define NFRAME  300

struct buffer { void *start; size_t length; };

static int xioctl(int fd, int req, void *arg)
{
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

int main(int argc, char *argv[])
{
    std::string mode = "V4L2";
    if (mode=="opencv") {
        cv::VideoCapture cap("/dev/video0", cv::CAP_V4L2);
        if (!cap.isOpened()) {
            std::cerr << "open failed\n";
            return 1;
        }

        // 強制 MJPG 解碼路徑，與 V4L2 版本條件一致
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);

        const int N = 300;
        cv::Mat frame;

        // 暖機，丟棄前幾張
        for (int i = 0; i < 10; i++) cap >> frame;

        auto t0 = std::chrono::high_resolution_clock::now();
        int got = 0;
        for (int i = 0; i < N; i++) {
            if (cap.read(frame) && !frame.empty()) got++;
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::printf("[OpenCV] %d frames, %.3f s, %.2f fps, %.2f ms/frame\n",
                    got, sec, got / sec, sec * 1000.0 / got);
    }
    else {
        const char *dev = (argc > 1) ? argv[1] : "/dev/video0";
        int fd = open(dev, O_RDWR);
        if (fd < 0) { perror("open"); return 1; }

        /* 設定格式：MJPG 1920x1080 */
        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width       = WIDTH;
        fmt.fmt.pix.height      = HEIGHT;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        fmt.fmt.pix.field       = V4L2_FIELD_NONE;
        if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) { perror("S_FMT"); return 1; }

        /* 請求緩衝區 */
        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(req));
        req.count  = NBUF;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) { perror("REQBUFS"); return 1; }

        /* mmap */
        struct buffer buffers[NBUF];
        for (unsigned i = 0; i < req.count; i++) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = i;
            if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) { perror("QUERYBUF"); return 1; }
            buffers[i].length = buf.length;
            buffers[i].start  = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, buf.m.offset);
            if (buffers[i].start == MAP_FAILED) { perror("mmap"); return 1; }
        }

        /* 入列並開始串流 */
        for (unsigned i = 0; i < req.count; i++) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = i;
            xioctl(fd, VIDIOC_QBUF, &buf);
        }
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) { perror("STREAMON"); return 1; }

        /* 暖機 10 張 */
        for (int i = 0; i < 10; i++) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            xioctl(fd, VIDIOC_DQBUF, &buf);
            xioctl(fd, VIDIOC_QBUF, &buf);
        }

        /* 計時擷取 */
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int got = 0;
        for (int i = 0; i < NFRAME; i++) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            if (xioctl(fd, VIDIOC_DQBUF, &buf) < 0) { perror("DQBUF"); break; }
            /* buffers[buf.index].start 此時即為一張 MJPG bytes，長度 buf.bytesused */
            got++;
            xioctl(fd, VIDIOC_QBUF, &buf);   /* 重新入列 */
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("[V4L2]   %d frames, %.3f s, %.2f fps, %.2f ms/frame\n",
            got, sec, got / sec, sec * 1000.0 / got);

        /* 清理 */
        xioctl(fd, VIDIOC_STREAMOFF, &type);
        for (unsigned i = 0; i < req.count; i++)
            munmap(buffers[i].start, buffers[i].length);
        close(fd);
    }

    // 注意：OpenCV 預設會把 MJPG 解碼成 BGR cv::Mat
    return 0;
}