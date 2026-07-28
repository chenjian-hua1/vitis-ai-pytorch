#include <opencv2/opencv.hpp>
#include <util.h>

cv::Mat scale_boxes(const cv::Mat&              boxes,
                    std::pair<float,float>       ratio,
                    std::pair<float,float>       pad,
                    cv::Size                     orig_shape);

cv::Mat draw_boxes(const cv::Mat&                    img,
                   const std::vector<Detection>&     detections,
                   const std::vector<std::string>&   class_names = {});

void draw_detection(const cv::Mat& in, cv::Mat& out, const DetectionBatch& detections, const ResizeResult& resize_inf, double fps=0);