#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("/Users/chenjianhua/vitis-ai-pytorch/dpu_inference/2308.jpg");

    if (img.empty()) {
        std::cout << "Image not found!" << std::endl;
        return -1;
    }

    cv::imshow("Image", img);
    cv::waitKey(0);
    return 0;
}