#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

int main(int argc, char *argv[])
{
    const char *dev = (argc > 1) ? argv[1] : "/dev/video0";
    int fd = open(dev, O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    /* 查詢裝置基本資訊 */
    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
        printf("Device : %s\n", dev);
        printf("Driver : %s\n", cap.driver);
        printf("Card   : %s\n", cap.card);
        printf("Bus    : %s\n", cap.bus_info);
        printf("--------------------------------------\n");
    }

    /* 列舉所有支援的像素格式 (編碼格式) */
    struct v4l2_fmtdesc fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    for (fmt.index = 0; ioctl(fd, VIDIOC_ENUM_FMT, &fmt) == 0; fmt.index++) {
        char fourcc[5] = {
            (char)(fmt.pixelformat & 0xFF),
            (char)((fmt.pixelformat >> 8) & 0xFF),
            (char)((fmt.pixelformat >> 16) & 0xFF),
            (char)((fmt.pixelformat >> 24) & 0xFF),
            0
        };
        printf("[%d] Format: %s  (%s)%s\n",
               fmt.index, fourcc, fmt.description,
               (fmt.flags & V4L2_FMT_FLAG_COMPRESSED) ? " [compressed]" : "");

        /* 列舉該格式下支援的解析度 */
        struct v4l2_frmsizeenum fsize;
        memset(&fsize, 0, sizeof(fsize));
        fsize.pixel_format = fmt.pixelformat;

        for (fsize.index = 0; ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fsize) == 0; fsize.index++) {
            if (fsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                printf("    %4u x %-4u",
                       fsize.discrete.width, fsize.discrete.height);

                /* 列舉該解析度支援的影格率 (fps) */
                struct v4l2_frmivalenum fival;
                memset(&fival, 0, sizeof(fival));
                fival.pixel_format = fmt.pixelformat;
                fival.width  = fsize.discrete.width;
                fival.height = fsize.discrete.height;

                printf("  @ ");
                for (fival.index = 0;
                     ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fival) == 0;
                     fival.index++) {
                    if (fival.type == V4L2_FRMIVAL_TYPE_DISCRETE &&
                        fival.discrete.numerator > 0) {
                        double fps = (double)fival.discrete.denominator /
                                     fival.discrete.numerator;
                        printf("%.0f ", fps);
                    }
                }
                printf("fps\n");
            } else {
                /* 連續或階梯式解析度 */
                printf("    %u x %u  ~  %u x %u  (step %u x %u)\n",
                       fsize.stepwise.min_width,  fsize.stepwise.min_height,
                       fsize.stepwise.max_width,  fsize.stepwise.max_height,
                       fsize.stepwise.step_width, fsize.stepwise.step_height);
                break;
            }
        }
        printf("\n");
    }

    close(fd);
    return 0;
}