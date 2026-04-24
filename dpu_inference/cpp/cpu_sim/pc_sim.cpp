#include "util.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>

using Clock = std::chrono::high_resolution_clock;

double time_now() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    // ─────────────────────────────────────────────────────────────
    //  參數：argv[1] = 圖片路徑，argv[2] = ONNX 模型路徑（都可省）
    // ─────────────────────────────────────────────────────────────
    std::string img_path = (argc > 1)
        ? argv[1]
        : "/home/jianhua/Desktop/vitis-ai-pytorch/dpu_inference/2308.jpg";
    std::string model_path = (argc > 2)
        ? argv[2]
        : "/home/jianhua/Desktop/vitis-ai-pytorch/dpu_inference/model/YOLO_int.onnx";

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "無法開啟圖片: " << img_path << "\n";
        return -1;
    }

    const int ITER   = 1000;
    const int WARMUP = 10;

    // conf / iou 集中在這裡，decode 跟 nms 必須用相同 conf
    const float CONF = 0.1f;
    const float IOU  = 0.45f;

    std::cout << "===== YOLO 前處理 + ONNX 推理 + 後處理效能測試 (平均 "
              << ITER << " 次) =====\n";

    // ─────────────────────────────────────────────────────────────
    //  1. 載入 ONNX 模型
    // ─────────────────────────────────────────────────────────────
    //  若不需要客製 SessionOptions，直接用 OnnxInferenceEngine(path) 即可。
    //  這裡示範如何用 callback 調整 intra-op threads 等設定：
    OnnxInferenceEngine engine(
        model_path,
        [](Ort::SessionOptions& opts) {
            // 如果需要多執行緒，可以打開這行：
            // opts.SetIntraOpNumThreads(4);
            // opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            (void)opts;
        });

    const int in_w = static_cast<int>(engine.in_w());
    const int in_h = static_cast<int>(engine.in_h());
    std::cout << "模型輸入: " << in_w << "x" << in_h
              << "  輸出數量: " << engine.num_outputs() << "\n";

    // ─────────────────────────────────────────────────────────────
    //  2. 根據模型輸出決定後處理參數
    //     YOLO v8/v11 頭：每個 scale 輸出 channel = nc + 4*ch
    //     我們從第 0 個輸出的 C 推出 nc（假設 ch=16 是固定 DFL 設計）
    // ─────────────────────────────────────────────────────────────
    const int ch = 16;

    // 先跑一次 dummy 推理確認 engine OK；output_mats() 的 Mat 在 ctor
    // 已分配好（shape 已知），這裡主要是確認推理能跑通。
    {
        cv::Mat dummy = cv::Mat::zeros(in_h, in_w, CV_32FC3);
        engine.run(dummy);
    }

    const std::vector<cv::Mat>& fmaps = engine.output_mats();
    // fmaps[i].size 應該是 {1, no, H_i, W_i}
    const int no = fmaps[0].size[1];           // = nc + 4*ch
    const int nc = no - 4 * ch;

    if (nc <= 0) {
        std::cerr << "模型輸出 channel 數 (" << no
                  << ") 與 YOLO DFL 頭假設不符 (ch=" << ch << ")\n";
        return -1;
    }
    std::cout << "推導出的 nc = " << nc << "\n";

    YOLOPostProcessor yolo_pp(1, in_h, in_w, nc, ch);

    // ─────────────────────────────────────────────────────────────
    //  3. 前處理 / 推理 / 後處理 的中間 buffer
    // ─────────────────────────────────────────────────────────────
    ResizeResult resize_result;
    cv::Mat norm_img;      // CV_32FC3
    cv::Mat fix_img;       // CV_8SC3  （量化模擬；保留 3-channel header）
    cv::Mat float_img;     // CV_32FC3 （反量化後餵給 ONNX）

    const std::vector<DetectionBatch>* nms_result = nullptr;

    // ─────────────────────────────────────────────────────────────
    //  4. Warmup
    // ─────────────────────────────────────────────────────────────
    for (int i = 0; i < WARMUP; ++i) {
        resize(img, in_w, resize_result);
        norm(resize_result.img, norm_img);
        float2fix(norm_img, 4, fix_img);
        fix2float(fix_img, 4, float_img);

        engine.run(float_img);                          // ← ONNX 推理
        nms_result = &yolo_pp.process(engine.output_mats(), CONF, IOU);
    }

    std::cout << "warmup 後偵測到 " << (*nms_result)[0].count
              << " 個框\n\n";

    // ─────────────────────────────────────────────────────────────
    //  5. 前處理計時
    // ─────────────────────────────────────────────────────────────
    double t_resize = 0, t_norm = 0, t_f2f = 0, t_f2f_back = 0;

    for (int i = 0; i < ITER; ++i) {
        double t0 = time_now();
        resize(img, in_w, resize_result);
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

    // ─────────────────────────────────────────────────────────────
    //  6. ONNX 推理計時
    // ─────────────────────────────────────────────────────────────
    double t_infer = 0;
    for (int i = 0; i < ITER; ++i) {
        double t0 = time_now();
        engine.run(float_img);
        t_infer += time_now() - t0;
    }

    // ─────────────────────────────────────────────────────────────
    //  7. 後處理計時（decode / nms 分開）
    //     decode 跟 nms 必須用同一個 CONF，否則 active list 的語意會錯配
    // ─────────────────────────────────────────────────────────────
    double t_post = 0, t_nms = 0;

    for (int i = 0; i < ITER; ++i) {
        double t0 = time_now();
        yolo_pp.decode(engine.output_mats(), CONF);
        t_post += time_now() - t0;

        t0 = time_now();
        yolo_pp.nms(CONF, IOU);
        t_nms += time_now() - t0;
    }

    // ─────────────────────────────────────────────────────────────
    //  8. 輸出結果
    // ─────────────────────────────────────────────────────────────
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "===== 前處理 =====\n";
    std::cout << "Resize avg      : " << t_resize / ITER    << " ms\n";
    std::cout << "Norm avg        : " << t_norm / ITER      << " ms\n";
    std::cout << "Float2Fix avg   : " << t_f2f / ITER       << " ms\n";
    std::cout << "Fix2Float avg   : " << t_f2f_back / ITER  << " ms\n";

    double preprocess = (t_resize + t_norm + t_f2f + t_f2f_back) / ITER;
    std::cout << "Total preprocess: " << preprocess << " ms\n";

    std::cout << "\n===== ONNX 推理 =====\n";
    double infer = t_infer / ITER;
    std::cout << "ONNX inference  : " << infer << " ms\n";

    std::cout << "\n===== 後處理 =====\n";
    std::cout << "YOLO decode avg : " << t_post / ITER << " ms\n";
    std::cout << "NMS avg         : " << t_nms / ITER  << " ms\n";

    double postprocess = (t_post + t_nms) / ITER;
    std::cout << "Total postprocess: " << postprocess << " ms\n";

    std::cout << "\n===== 總計 =====\n";
    std::cout << "Total pipeline  : "
              << preprocess + infer + postprocess << " ms\n";

    // ─────────────────────────────────────────────────────────────
    //  9. 繪圖：把最後一次 NMS 的結果畫到原圖並存檔
    //
    //     流程：
    //       a) detections 座標在「640×640 padded」空間，要先 scale 回原圖
    //       b) scale_boxes 吃 (N,4) CV_32F Mat，所以先把 Detection 的
    //          xyxy 灌進 Mat
    //       c) 把 scale 後的座標寫回 Detection 陣列給 draw_boxes 用
    //       d) draw_boxes 回傳畫好的 copy，imwrite 存檔（不用 imshow 避免
    //          跑在沒 GUI 的環境時掛掉）
    // ─────────────────────────────────────────────────────────────
    const DetectionBatch& last = (*nms_result)[0];
    std::cout << "\n===== 繪圖 =====\n";
    std::cout << "偵測到 " << last.count << " 個框\n";

    if (last.count > 0) {
        // (a) Detection → (N,4) CV_32F
        cv::Mat boxes_padded(last.count, 4, CV_32F);
        for (int i = 0; i < last.count; ++i) {
            const Detection& d = last.data[i];
            float* r = boxes_padded.ptr<float>(i);
            r[0] = d.x1;  r[1] = d.y1;  r[2] = d.x2;  r[3] = d.y2;
        }

        // (b) scale 回原圖座標
        cv::Mat boxes_orig = scale_boxes(
            boxes_padded,
            resize_result.ratio,
            resize_result.pad,
            cv::Size(img.cols, img.rows));

        // (c) 把 scale 後座標寫回 Detection 陣列
        std::vector<Detection> dets_drawable(last.count);
        for (int i = 0; i < last.count; ++i) {
            const float* r = boxes_orig.ptr<float>(i);
            dets_drawable[i] = Detection{
                r[0], r[1], r[2], r[3],
                last.data[i].score, last.data[i].class_id
            };
        }

        // (d) 繪圖 & 存檔
        // class_names 留空 → draw_boxes 會自動用 "Class N" 當 label
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