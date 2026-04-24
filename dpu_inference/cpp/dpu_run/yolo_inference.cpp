#include "util.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

using Clock = std::chrono::high_resolution_clock;

double time_now() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    // ─────────────────────────────────────────────────────────────
    //  參數：argv[1] = 圖片路徑，argv[2] = Xmodel 模型路徑
    // ─────────────────────────────────────────────────────────────
    std::string img_path = (argc > 1)
        ? argv[1]
        : "/home/jianhua/Desktop/vitis-ai-pytorch/dpu_inference/2308.jpg";
    std::string model_path = (argc > 2)
        ? argv[2]
        : "model/YOLO_int.xmodel";

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "無法開啟圖片: " << img_path << "\n";
        return -1;
    }

    const int ITER   = 1000;
    const int WARMUP = 10;
    const float CONF = 0.1f;
    const float IOU  = 0.45f;

    std::cout << "===== YOLO DPU (Xmodel) 推理效能測試 (平均 " << ITER << " 次) =====\n";

    // 1. 初始化 Xmodel 引擎
    XmodelInferenceEngine engine(model_path);

    const int in_w = static_cast<int>(engine.in_w());
    const int in_h = static_cast<int>(engine.in_h());
    std::cout << "模型輸入: " << in_w << "x" << in_h
              << "  輸出數量: " << engine.num_outputs() << "\n";

    // 2. 根據模型輸出決定後處理參數 (讀取預建的 NCHW 替身 shape)
    const int ch = 16;
    const int no = engine.output_mat_nchw(0).size[1];
    const int nc = no - 4 * ch;

    if (nc <= 0) {
        std::cerr << "模型輸出 channel 數 (" << no
                  << ") 與 YOLO DFL 頭假設不符 (ch=" << ch << ")\n";
        return -1;
    }
    std::cout << "推導出的 nc = " << nc << "\n";

    YOLOPostProcessor yolo_pp(1, in_h, in_w, nc, ch);

    // 3. 準備 Buffer 與 DPU Zero-copy 綁定
    ResizeResult resize_result;
    cv::Mat norm_img;      
    std::vector<cv::Mat> float_outputs(engine.num_outputs()); 
    const std::vector<DetectionBatch>* nms_result = nullptr;

    cv::Mat dpu_input;
    engine.bind_input_mat(dpu_input); // 綁定輸入替身
    
    // 計算輸入與輸出的 Fix Point
    int in_fix_point = static_cast<int>(std::round(std::log2(engine.input_scale())));
    std::vector<int> out_fix_points(engine.num_outputs());
    for (size_t i = 0; i < engine.num_outputs(); ++i) {
        out_fix_points[i] = static_cast<int>(std::round(-std::log2(engine.output_scale(i))));
    }

    // 4. Warmup
    for (int i = 0; i < WARMUP; ++i) {
        resize(img, in_w, resize_result);
        norm(resize_result.img, norm_img);
        
        float2fix(norm_img, in_fix_point, dpu_input); // Zero-copy 寫入 DPU 實體記憶體
        engine.run();                                 // 執行推理，內部排版為 NCHW int8
        
        // 反量化：從 NCHW int8 替身轉回 NCHW float
        for (size_t out_idx = 0; out_idx < engine.num_outputs(); ++out_idx) {
            fix2float(engine.output_mat_nchw(out_idx), out_fix_points[out_idx], float_outputs[out_idx]);
        }
        
        nms_result = &yolo_pp.process(float_outputs, CONF, IOU);
    }
    std::cout << "warmup 後偵測到 " << (*nms_result)[0].count << " 個框\n\n";

    // 5. 前處理計時
    double t_resize = 0, t_norm = 0, t_f2f = 0;
    for (int i = 0; i < ITER; ++i) {
        double t0 = time_now();
        resize(img, in_w, resize_result);
        t_resize += time_now() - t0;

        t0 = time_now();
        norm(resize_result.img, norm_img);
        t_norm += time_now() - t0;

        t0 = time_now();
        float2fix(norm_img, in_fix_point, dpu_input);
        t_f2f += time_now() - t0;
    }

    // 6. 推理計時
    double t_infer = 0;
    for (int i = 0; i < ITER; ++i) {
        double t0 = time_now();
        engine.run();
        t_infer += time_now() - t0;
    }

    // 7. 後處理計時 (反量化 -> Decode -> NMS)
    double t_f2f_back = 0, t_post = 0, t_nms = 0;

    for (int i = 0; i < ITER; ++i) {
        double t0 = time_now();
        for (size_t out_idx = 0; out_idx < engine.num_outputs(); ++out_idx) {
            fix2float(engine.output_mat_nchw(out_idx), out_fix_points[out_idx], float_outputs[out_idx]);
        }
        t_f2f_back += time_now() - t0;
    }

    for (int i = 0; i < ITER; ++i) {
        double t0 = time_now();
        yolo_pp.decode(float_outputs, CONF);
        t_post += time_now() - t0;

        t0 = time_now();
        yolo_pp.nms(CONF, IOU);
        t_nms += time_now() - t0;
    }

    // 8. 輸出結果
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "===== 前處理 =====\n";
    std::cout << "Resize avg      : " << t_resize / ITER << " ms\n";
    std::cout << "Norm avg        : " << t_norm / ITER   << " ms\n";
    std::cout << "Float2Fix avg   : " << t_f2f / ITER    << " ms (Zero-copy)\n";
    
    double preprocess = (t_resize + t_norm + t_f2f) / ITER;
    std::cout << "Total preprocess: " << preprocess << " ms\n";

    std::cout << "\n===== 推理 =====\n";
    double infer = t_infer / ITER;
    std::cout << "DPU inference   : " << infer << " ms\n";

    std::cout << "\n===== 後處理 =====\n";
    std::cout << "Fix2Float avg   : " << t_f2f_back / ITER << " ms (Dequantize)\n";
    std::cout << "YOLO decode avg : " << t_post / ITER     << " ms\n";
    std::cout << "NMS avg         : " << t_nms / ITER      << " ms\n";

    double postprocess = (t_f2f_back + t_post + t_nms) / ITER;
    std::cout << "Total postprocess: " << postprocess << " ms\n";

    std::cout << "\n===== 總計 =====\n";
    std::cout << "Total pipeline  : " << preprocess + infer + postprocess << " ms\n";

    // 9. 繪圖
    const DetectionBatch& last = (*nms_result)[0];
    std::cout << "\n===== 繪圖 =====\n";
    std::cout << "偵測到 " << last.count << " 個框\n";

    if (last.count > 0) {
        cv::Mat boxes_padded(last.count, 4, CV_32F);
        for (int i = 0; i < last.count; ++i) {
            const Detection& d = last.data[i];
            float* r = boxes_padded.ptr<float>(i);
            r[0] = d.x1;  r[1] = d.y1;  r[2] = d.x2;  r[3] = d.y2;
        }

        cv::Mat boxes_orig = scale_boxes(
            boxes_padded, resize_result.ratio, resize_result.pad,
            cv::Size(img.cols, img.rows));

        std::vector<Detection> dets_drawable(last.count);
        for (int i = 0; i < last.count; ++i) {
            const float* r = boxes_orig.ptr<float>(i);
            dets_drawable[i] = Detection{
                r[0], r[1], r[2], r[3],
                last.data[i].score, last.data[i].class_id
            };
        }

        cv::Mat drawn = draw_boxes(img, dets_drawable);
        const std::string out_path = "detection_result.jpg";
        if (cv::imwrite(out_path, drawn)) {
            std::cout << "結果已存至: " << out_path << "\n";
        } else {
            std::cerr << "imwrite 失敗: " << out_path << "\n";
        }
    } else {
        std::cout << "沒有偵測到任何框，跳過繪圖。\n";
    }

    return 0;
}