#include "util.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

double time_now() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string img_path = (argc > 1) ? argv[1] : "/home/jianhua/Desktop/vitis-ai-pytorch/dpu_inference/2308.jpg";
    cv::Mat img = cv::imread(img_path);

    if (img.empty()) {
        std::cerr << "無法開啟圖片: " << img_path << "\n";
        return -1;
    }

    const int ITER = 100;
    const int WARMUP = 10;

    std::cout << "===== YOLO 前後處理效能測試 (平均 " << ITER << " 次) =====\n";

    // ===== 模擬模型輸出 =====
    int nc = 80;
    int ch = 16;
    int no = nc + ch * 4;

    std::vector<cv::Mat> fmaps = {
        cv::Mat(std::vector<int>{1, no, 80, 80}, CV_32F, cv::Scalar(0.01f)),
        cv::Mat(std::vector<int>{1, no, 40, 40}, CV_32F, cv::Scalar(0.01f)),
        cv::Mat(std::vector<int>{1, no, 20, 20}, CV_32F, cv::Scalar(0.01f))
    };

    YOLOPostProcessor yolo_pp(nc, ch, {8, 16, 32});

    // ===== Warmup =====
    ResizeResult resize_result;
    cv::Mat norm_img, fix_img, float_img;
    cv::Mat yolo_raw_out;
    std::vector<std::vector<Detection>> nms_result;

    for (int i = 0; i < WARMUP; ++i) {
        resize(img, 640, resize_result);
        norm(resize_result.img, norm_img);
        float2fix(norm_img, 4, fix_img);
        fix2float(fix_img, 4, float_img);

        yolo_raw_out = yolo_pp(fmaps, 0.25f);
        nms_result = non_max_suppression(yolo_raw_out, 0.25f, 0.45f);
    }

    // =========================
    // 🔥 前處理
    // =========================
    double t_resize = 0, t_norm = 0, t_f2f = 0, t_f2f_back = 0;

    for (int i = 0; i < ITER; ++i) {
        double t0 = time_now();
        // resize_result = resize_zero_copy(img, 640);
        resize(img, 640, resize_result);
        t_resize += time_now() - t0;

        t0 = time_now();
        norm(resize_result.img, norm_img);
        t_norm += time_now() - t0;

        t0 = time_now();
        float2fix(norm_img, 4, fix_img);
        t_f2f += time_now() - t0;

        t0 = time_now();
        fix2float(fix_img, 4, float_img);
        t_f2f_back += time_now() - t0;
    }

    // =========================
    // 🔥 後處理
    // =========================
    double t_post = 0, t_nms = 0;

    // for (int i = 0; i < ITER; ++i) {
    //     double t0 = time_now();
    //     yolo_raw_out = yolo_pp(fmaps, 0.25f);
    //     t_post += time_now() - t0;

    //     // 塞一個高分物件（避免 NMS 空跑）
    //     yolo_raw_out.at<float>(std::vector<int>{0, 4, 0}.data()) = 0.95f;

    //     t0 = time_now();
    //     nms_result = non_max_suppression(yolo_raw_out, 0.25f, 0.45f);
    //     t_nms += time_now() - t0;
    // }

    // =========================
    // 🔥 輸出結果
    // =========================
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\n===== 前處理 =====\n";
    std::cout << "Resize avg      : " << t_resize / ITER << " ms\n";
    std::cout << "Norm avg        : " << t_norm / ITER << " ms\n";
    std::cout << "Float2Fix avg   : " << t_f2f / ITER << " ms\n";
    std::cout << "Fix2Float avg   : " << t_f2f_back / ITER << " ms\n";

    double preprocess = (t_resize + t_norm + t_f2f + t_f2f_back) / ITER;

    std::cout << "Total preprocess: " << preprocess << " ms\n";

    std::cout << "\n===== 後處理 =====\n";
    std::cout << "YOLO decode avg : " << t_post / ITER << " ms\n";
    std::cout << "NMS avg         : " << t_nms / ITER << " ms\n";

    double postprocess = (t_post + t_nms) / ITER;

    std::cout << "Total postprocess: " << postprocess << " ms\n";

    std::cout << "\n===== 總計 =====\n";
    std::cout << "Total pipeline  : " << preprocess + postprocess << " ms\n";

    return 0;
}