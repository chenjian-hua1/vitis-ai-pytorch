#include <opencv2/opencv.hpp>
#include <data_struct.h>
#include <tracker.h>

cv::Mat scale_boxes(const cv::Mat&              boxes,
                    std::pair<float,float>       ratio,
                    std::pair<float,float>       pad,
                    cv::Size                     orig_shape);

cv::Mat draw_boxes(const cv::Mat&                    img,
                   const std::vector<Detection>&     detections,
                   const std::vector<std::string>&   class_names = {});

void draw_detection(const cv::Mat& in, cv::Mat& out, const DetectionBatch& detections, const ResizeResult& resize_inf, double fps=0);

// letterbox 座標 → 原圖座標，並轉成 tracker 的輸入格式
std::vector<bytetrack::Box> scale_detections(const DetectionBatch& detections,
                                            const ResizeResult&   resize_inf,
                                            cv::Size              orig_shape);

cv::Mat draw_tracks(const cv::Mat&                     img,
                    const std::vector<bytetrack::Track>& tracks,
                    const std::vector<std::string>&    class_names = {});

void draw_tracking(const cv::Mat&                       in,
                   cv::Mat&                             out,
                   const std::vector<bytetrack::Track>& tracks,
                   double                               fps);