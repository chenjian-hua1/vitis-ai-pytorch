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
    std::string img_path   = (argc > 1) ? argv[1] : "/home/jianhua/Desktop/vitis-ai-pytorch/dpu_inference/2308.jpg";
    std::string model_path = (argc > 2) ? argv[2] : "model.onnx";

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "無法開啟圖片: " << img_path << "\n";
        return -1;
    }

    const int ITER   = 100;
    const int WARMUP = 10;

    std::cout << "===== YOLO 前後處理效能測試 (平均 " << ITER << " 次) =====\n";

    // ===== ONNX Runtime 初始化 =====
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_ort");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // [修改] 獲取輸入與三個輸出節點的名稱
    auto input_name  = session.GetInputNameAllocated(0, allocator);
    auto out_name0   = session.GetOutputNameAllocated(0, allocator);
    auto out_name1   = session.GetOutputNameAllocated(1, allocator);
    auto out_name2   = session.GetOutputNameAllocated(2, allocator);
    
    const char* in_names[]  = { input_name.get() };
    const char* out_names[] = { out_name0.get(), out_name1.get(), out_name2.get() }; // [修改] 加入三個輸出
    const int num_outputs   = 3; // [修改] 定義輸出數量

    // ===== 後處理初始化 =====
    int nc = 80;
    int ch = 16;
    int no = nc + ch * 4;

    // 模型輸出 fmaps 會在推理後從 ort_outputs 填入
    std::vector<cv::Mat> fmaps = {
        cv::Mat(std::vector<int>{1, no, 80, 80}, CV_32F),
        cv::Mat(std::vector<int>{1, no, 40, 40}, CV_32F),
        cv::Mat(std::vector<int>{1, no, 20, 20}, CV_32F)
    };

    YOLOPostProcessor yolo_pp(nc, ch, {8, 16, 32});

    // ===== 預先分配 =====
    ResizeResult resize_result;
    cv::Mat norm_img, fix_img, float_img;
    cv::Mat yolo_raw_out;
    std::vector<std::vector<Detection>> nms_result;

    std::vector<float> input_data(1 * 3 * 640 * 640);
    std::vector<int64_t> tensor_shape = {1, 3, 640, 640};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ===== 前處理 lambda =====
    auto do_preprocess = [&]() {
        resize(img, 640, resize_result);
        norm(resize_result.img, norm_img);
        float2fix(norm_img, 4, fix_img);
        fix2float(fix_img, 4, float_img);
        // HWC → NCHW
        cv::Mat chw;
        cv::dnn::blobFromImage(float_img, chw);
        std::memcpy(input_data.data(), chw.data, input_data.size() * sizeof(float));
    };

    // ===== 推理 lambda =====
    auto do_infer = [&]() {
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(),
            tensor_shape.data(), tensor_shape.size()
        );
        return session.Run(
            Ort::RunOptions{nullptr},
            in_names, &input_tensor, 1,
            out_names, num_outputs // [修改] 使用 3 個輸出
        );
    };

    // ===== 後處理 lambda =====
    auto do_postprocess = [&](std::vector<Ort::Value>& ort_outputs) {
        for (size_t i = 0; i < ort_outputs.size(); ++i) { 
            auto& out   = ort_outputs[i];
            float* data = out.GetTensorMutableData<float>();
            auto shape  = out.GetTensorTypeAndShapeInfo().GetShape();
            
            int batch = (int)shape[0];
            int chans = (int)shape[1];
            int h     = (int)shape[2];
            int w     = (int)shape[3];

            // 創建當前輸出的 Mat 視角 (Zero-copy)
            cv::Mat out_mat(std::vector<int>{batch, chans, h, w}, CV_32F, data);

            // [關鍵修改] 根據特徵圖解析度，精準綁定到對應的 fmaps 索引
            if (h == 80 && w == 80) {
                fmaps[0] = out_mat; // 對應 stride 8
            } else if (h == 40 && w == 40) {
                fmaps[1] = out_mat; // 對應 stride 16
            } else if (h == 20 && w == 20) {
                fmaps[2] = out_mat; // 對應 stride 32
            } else {
                std::cerr << "警告: 收到非預期的特徵圖大小 " << h << "x" << w << "\n";
            }
        }
        
        yolo_raw_out = yolo_pp(fmaps, 0.25f);
        nms_result   = non_max_suppression(yolo_raw_out, 0.25f, 0.45f);
    };

    // ===== Warmup =====
    for (int i = 0; i < WARMUP; ++i) {
        do_preprocess();
        auto ort_outputs = do_infer();
        do_postprocess(ort_outputs);
    }

    // =========================
    // 🔥 計時迴圈
    // =========================
    double t_resize = 0, t_norm = 0, t_f2f = 0, t_f2f_back = 0;
    double t_infer  = 0;
    double t_post   = 0, t_nms = 0;

    for (int i = 0; i < ITER; ++i) {

        // --- 前處理 ---
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

        // HWC → NCHW（不單獨計時，歸入前處理准備）
        cv::Mat chw;
        cv::dnn::blobFromImage(float_img, chw);
        std::memcpy(input_data.data(), chw.data, input_data.size() * sizeof(float));

        // --- 推理 ---
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(),
            tensor_shape.data(), tensor_shape.size()
        );
        t0 = time_now();
        auto ort_outputs = session.Run(
            Ort::RunOptions{nullptr},
            in_names, &input_tensor, 1,
            out_names, num_outputs // [修改] 使用 3 個輸出
        );
        t_infer += time_now() - t0;

        // --- 後處理：動態綁定 Shape ---
        for (size_t j = 0; j < ort_outputs.size(); ++j) {
            float* data = ort_outputs[j].GetTensorMutableData<float>();
            auto shape  = ort_outputs[j].GetTensorTypeAndShapeInfo().GetShape();
            
            int batch = (int)shape[0];
            int chans = (int)shape[1];
            int h     = (int)shape[2];
            int w     = (int)shape[3];

            cv::Mat out_mat(std::vector<int>{batch, chans, h, w}, CV_32F, data);

            // [修改] 根據特徵圖解析度，精準綁定到對應的 fmaps 索引
            if (h == 80 && w == 80) {
                fmaps[0] = out_mat; // 對應 stride 8
            } else if (h == 40 && w == 40) {
                fmaps[1] = out_mat; // 對應 stride 16
            } else if (h == 20 && w == 20) {
                fmaps[2] = out_mat; // 對應 stride 32
            } else {
                std::cerr << "警告: 收到非預期的特徵圖大小 " << h << "x" << w << "\n";
            }
        }

        t0 = time_now();
        yolo_raw_out = yolo_pp(fmaps, 0.25f);
        t_post += time_now() - t0;

        // --- 後處理：NMS ---
        t0 = time_now();
        nms_result = non_max_suppression(yolo_raw_out, 0.25f, 0.45f);
        t_nms += time_now() - t0;
    }

    // =========================
    // 🔥 輸出結果
    // =========================
    std::cout << std::fixed << std::setprecision(3);

    double preprocess  = (t_resize + t_norm + t_f2f + t_f2f_back) / ITER;
    double postprocess = (t_post + t_nms) / ITER;

    std::cout << "\n===== 前處理 =====\n";
    std::cout << "Resize avg      : " << t_resize    / ITER << " ms\n";
    std::cout << "Norm avg        : " << t_norm      / ITER << " ms\n";
    std::cout << "Float2Fix avg   : " << t_f2f       / ITER << " ms\n";
    std::cout << "Fix2Float avg   : " << t_f2f_back  / ITER << " ms\n";
    std::cout << "Total preprocess: " << preprocess          << " ms\n";

    std::cout << "\n===== 模型推理 =====\n";
    std::cout << "Inference avg   : " << t_infer / ITER << " ms\n";

    std::cout << "\n===== 後處理 =====\n";
    std::cout << "YOLO decode avg : " << t_post / ITER << " ms\n";
    std::cout << "NMS avg         : " << t_nms  / ITER << " ms\n";
    std::cout << "Total postprocess: " << postprocess  << " ms\n";

    std::cout << "\n===== 總計 =====\n";
    std::cout << "Total pipeline  : " << preprocess + t_infer / ITER + postprocess << " ms\n";

    // 最後一次結果預覽
    std::cout << "\n===== 最後一次偵測結果 =====\n";
    for (size_t cls = 0; cls < nms_result.size(); ++cls) {
        for (auto& det : nms_result[cls]) {
            std::cout << "Class " << cls
                      << "  conf=" << det.score
                      << "  box=["  << det.x1 << ", " << det.y1
                      << ", "       << det.x2 << ", " << det.y2 << "]\n";
        }
    }

    return 0;
}