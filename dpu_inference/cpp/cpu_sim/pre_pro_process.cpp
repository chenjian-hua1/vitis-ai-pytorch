#include <preproc.h>
#include <yolopproc.h>
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
    std::string img_path = (argc > 1)
        ? argv[1]
        : "/home/jianhua/Desktop/vitis-ai-pytorch/dpu_inference/2308.jpg";
    cv::Mat img = cv::imread(img_path);

    if (img.empty()) {
        std::cerr << "無法開啟圖片: " << img_path << "\n";
        return -1;
    }

    const int ITER   = 100;
    const int WARMUP = 10;

    // conf / iou 集中在這裡，decode 跟 nms 必須用相同 conf
    const float CONF = 0.6f;
    const float IOU  = 0.45f;

    std::cout << "===== YOLO 前後處理效能測試 (平均 " << ITER << " 次) =====\n";

    // ===== 模擬模型輸出 =====
    int nc = 4;                  // ⚠ nc 要跟 fmaps channel 數對齊
    int ch = 16;
    int no = nc + ch * 4;        // 68

    std::vector<cv::Mat> fmaps = {
        cv::Mat(std::vector<int>{1, no, 80, 80}, CV_32F, cv::Scalar(0.01f)),
        cv::Mat(std::vector<int>{1, no, 40, 40}, CV_32F, cv::Scalar(0.01f)),
        cv::Mat(std::vector<int>{1, no, 20, 20}, CV_32F, cv::Scalar(0.01f))
    };

    //  Test N high scores box decode & nms
    //  cls logit 起點 channel = 4*ch = 64，class 0 在 index 64
    //  conf = sigmoid(5.0) ≈ 0.993
    const int high_score_boxs = 50;
    for (int i = 0; i < high_score_boxs; i++)
        fmaps[0].at<float>(std::vector<int>{0, 4 * ch, 0, i}.data()) = 5.0f;

    YOLOPostProcessor yolo_pp(1, 640, 640, nc);

    // ===== 輸入 / 中間 buffer =====
    ResizeResult resize_result;
    cv::Mat norm_img, fix_img, float_img;

    // ===== Warmup =====
    const std::vector<DetectionBatch>* nms_result = nullptr;

    for (int i = 0; i < WARMUP; ++i) {
        resize(img, 640, resize_result);
        norm(resize_result.img, norm_img);
        float2fix(norm_img, 4, fix_img);
        fix2float(fix_img, 4, float_img);

        nms_result = &yolo_pp.process(fmaps, CONF, IOU);
    }

    std::cout << "warmup 後偵測到 " << (*nms_result)[0].count
              << " 個框\n\n";

    // =========================
    // 🔥 前處理
    // =========================
    double t_resize = 0, t_norm = 0, t_f2f = 0, t_f2f_back = 0;

    for (int i = 0; i < ITER; ++i) {
        double t0 = time_now();
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
    // 🔥 後處理（decode / nms 分開計時）
    //    注意：新版 decode 必須帶 conf_thresh，且 nms 必須用相同值。
    //    分開計時仍然反映真實時間——decode 做了早期過濾後，nms 只掃
    //    active anchor，兩階段加總就是 process() 的時間。
    // =========================
    double t_post = 0, t_nms = 0;

    for (int i = 0; i < ITER; ++i) {
        double t0 = time_now();
        yolo_pp.decode(fmaps, CONF);      // ← 修改：補上 conf_thresh
        t_post += time_now() - t0;

        t0 = time_now();
        yolo_pp.nms(CONF, IOU);           // ← 要跟 decode 用同一個 conf
        t_nms += time_now() - t0;
    }

    // =========================
    // 🔥 輸出結果
    // =========================
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "===== 前處理 =====\n";
    std::cout << "Resize avg      : " << t_resize / ITER    << " ms\n";
    std::cout << "Norm avg        : " << t_norm / ITER      << " ms\n";
    std::cout << "Float2Fix avg   : " << t_f2f / ITER       << " ms\n";
    std::cout << "Fix2Float avg   : " << t_f2f_back / ITER  << " ms\n";

    double preprocess = (t_resize + t_norm + t_f2f + t_f2f_back) / ITER;
    std::cout << "Total preprocess: " << preprocess << " ms\n";

    std::cout << "\n===== 後處理 =====\n";
    std::cout << "YOLO decode avg : " << t_post / ITER << " ms\n";
    std::cout << "NMS avg         : " << t_nms / ITER  << " ms\n";

    double postprocess = (t_post + t_nms) / ITER;
    std::cout << "Total postprocess: " << postprocess << " ms\n";

    std::cout << "\n===== 總計 =====\n";
    std::cout << "Total pipeline  : " << preprocess + postprocess << " ms\n";

    return 0;
}