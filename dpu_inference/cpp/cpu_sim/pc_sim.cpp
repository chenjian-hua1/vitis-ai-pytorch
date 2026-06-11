#include "util.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <csignal>

// 偵測 Ctrl-C
static volatile bool g_running = true;
void signalHandler(int) { g_running = false; }

using Clock = std::chrono::high_resolution_clock;

double time_now() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}

int benchmark(
    std::string src_path, std::string onnx_path, 
    int iter = 1000, int warmup = 10,
    double conf_th = 0.1,
    double iou_th = 0.45
) {
    std::cout << "===== YOLO 前處理 + ONNX 推理 + 後處理效能測試 (平均 " << iter << " 次) =====\n";

    // ─────────────────────────────────────────────────────────────
    //  0. imread 計時（cold + warm）
    // ─────────────────────────────────────────────────────────────
    double t_imread_cold0 = time_now();
    cv::Mat img = cv::imread(src_path);
    double t_imread_cold = time_now() - t_imread_cold0;

    if (img.empty()) {
        std::cerr << "無法開啟圖片: " << src_path << "\n";
        return -1;
    }

    // warmup（讓 OS page cache 暖起來）
    for (int i = 0; i < warmup; ++i) {
        cv::Mat tmp = cv::imread(src_path);
    }

    // 穩態量測
    double t_imread = 0;
    for (int i = 0; i < iter; ++i) {
        double t0 = time_now();
        cv::Mat tmp = cv::imread(src_path);
        t_imread += time_now() - t0;
    }

    // ─────────────────────────────────────────────────────────────
    //  1. 載入 ONNX 模型
    // ─────────────────────────────────────────────────────────────
    OnnxInferenceEngine engine(
        onnx_path,
        [](Ort::SessionOptions& opts) {
            // 如果需要多執行緒,可以打開這行:
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
    // ─────────────────────────────────────────────────────────────
    const int ch = 16;

    {
        cv::Mat dummy = cv::Mat::zeros(in_h, in_w, CV_32FC3);
        engine.run(dummy);
    }

    const std::vector<cv::Mat>& fmaps = engine.output_mats();
    const int no = fmaps[0].size[1];
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
    cv::Mat norm_img;
    cv::Mat fix_img;
    cv::Mat float_img;

    const std::vector<DetectionBatch>* nms_result = nullptr;

    // ─────────────────────────────────────────────────────────────
    //  4. Warmup
    // ─────────────────────────────────────────────────────────────
    for (int i = 0; i < warmup; ++i) {
        resize(img, in_w, resize_result);
        norm(resize_result.img, norm_img);
        float2fix(norm_img, 4, fix_img);
        fix2float(fix_img, 4, float_img);

        engine.run(float_img);
        nms_result = &yolo_pp.process(engine.output_mats(), conf_th, iou_th);
    }

    std::cout << "warmup 後偵測到 " << (*nms_result)[0].count
              << " 個框\n\n";

    // ─────────────────────────────────────────────────────────────
    //  5. 前處理計時
    // ─────────────────────────────────────────────────────────────
    double t_resize = 0, t_norm = 0, t_f2f = 0, t_f2f_back = 0;

    for (int i = 0; i < iter; ++i) {
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
    for (int i = 0; i < iter; ++i) {
        double t0 = time_now();
        engine.run(float_img);
        t_infer += time_now() - t0;
    }

    // ─────────────────────────────────────────────────────────────
    //  7. 後處理計時（decode / nms 分開）
    // ─────────────────────────────────────────────────────────────
    double t_post = 0, t_nms = 0;

    for (int i = 0; i < iter; ++i) {
        double t0 = time_now();
        yolo_pp.decode(engine.output_mats(), conf_th);
        t_post += time_now() - t0;

        t0 = time_now();
        yolo_pp.nms(conf_th, iou_th);
        t_nms += time_now() - t0;
    }

    // ─────────────────────────────────────────────────────────────
    //  8. 輸出結果
    // ─────────────────────────────────────────────────────────────
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "===== 讀檔 =====\n";
    std::cout << "imread (cold)   : " << t_imread_cold      << " ms\n";
    std::cout << "imread (warm)   : " << t_imread / iter    << " ms  ("
              << img.cols << "x" << img.rows << ")\n";

    std::cout << "\n===== 前處理 =====\n";
    std::cout << "Resize avg      : " << t_resize / iter    << " ms\n";
    std::cout << "Norm avg        : " << t_norm / iter      << " ms\n";
    std::cout << "Float2Fix avg   : " << t_f2f / iter       << " ms\n";
    std::cout << "Fix2Float avg   : " << t_f2f_back / iter  << " ms\n";

    double preprocess = (t_resize + t_norm + t_f2f + t_f2f_back) / iter;
    std::cout << "Total preprocess: " << preprocess << " ms\n";

    std::cout << "\n===== ONNX 推理 =====\n";
    double infer = t_infer / iter;
    std::cout << "ONNX inference  : " << infer << " ms\n";

    std::cout << "\n===== 後處理 =====\n";
    std::cout << "YOLO decode avg : " << t_post / iter << " ms\n";
    std::cout << "NMS avg         : " << t_nms / iter  << " ms\n";

    double postprocess = (t_post + t_nms) / iter;
    std::cout << "Total postprocess: " << postprocess << " ms\n";

    std::cout << "\n===== 總計 =====\n";
    std::cout << "Total pipeline  : "
              << preprocess + infer + postprocess << " ms\n";

    // ─────────────────────────────────────────────────────────────
    //  9. 繪圖：把最後一次 NMS 的結果畫到原圖並存檔
    // ─────────────────────────────────────────────────────────────
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
            boxes_padded,
            resize_result.ratio,
            resize_result.pad,
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

    return 0;   // 順便補上原本缺的 return
}

void run_video(std::string onnx_path, std::string video_path, std::string out_file = "",
               double conf_th = 0.1, double iou_th = 0.45) {
    // ======================== 載入 ONNX 模型 =========================================
    OnnxInferenceEngine engine(
        onnx_path,
        [](Ort::SessionOptions& opts) {
            // 如果需要多執行緒,可以打開這行:
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
    // ─────────────────────────────────────────────────────────────
    const int ch = 16;

    {
        cv::Mat dummy = cv::Mat::zeros(in_h, in_w, CV_32FC3);
        engine.run(dummy);
    }

    const std::vector<cv::Mat>& fmaps = engine.output_mats();
    const int no = fmaps[0].size[1];
    const int nc = no - 4 * ch;

    if (nc <= 0) {
        std::cerr << "模型輸出 channel 數 (" << no
                  << ") 與 YOLO DFL 頭假設不符 (ch=" << ch << ")\n";
        return;
    }
    std::cout << "推導出的 nc = " << nc << "\n";

    YOLOPostProcessor yolo_pp(1, in_h, in_w, nc, ch);

    // ─────────────────────────────────────────────────────────────
    //  3. 前處理 / 推理 / 後處理 的中間 buffer
    // ─────────────────────────────────────────────────────────────
    ResizeResult resize_result;
    cv::Mat norm_img;
    cv::Mat fix_img;
    cv::Mat float_img;

    const std::vector<DetectionBatch>* nms_result = nullptr;

    // ======================== 取小寫副檔名（不含點）===================================
    std::string ext = std::filesystem::path(video_path).extension().string();
    if (!ext.empty() && ext.front() == '.') ext.erase(0, 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // ======================== 影片 ====================================================
    if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
             ext == "flv" || ext == "wmv" || ext == "webm" || ext == "m4v")
    {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video: " << video_path << "\n";
            return;
        }

        int    src_w   = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int    src_h   = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double src_fps = cap.get(cv::CAP_PROP_FPS);
        if (src_fps <= 0.0) src_fps = 30.0;

        std::cout << "[Video] " << video_path << "  "
                  << src_w << "x" << src_h
                  << "  fps=" << src_fps
                  << "  frames=" << (int)cap.get(cv::CAP_PROP_FRAME_COUNT) << "\n";

        // ===== 影片寫出器（僅在 out_file 非空時建立）=====
        const bool save_video = !out_file.empty();
        cv::VideoWriter writer;
        if (save_video) {
            writer.open(out_file,
                        cv::VideoWriter::fourcc('m','p','4','v'),
                        src_fps,
                        cv::Size(src_w, src_h));
            if (!writer.isOpened()) {
                std::cerr << "Failed to open VideoWriter: " << out_file
                          << "  (將不寫出影片繼續執行)\n";
            }
        }

        cv::Mat frame;
        cv::Mat rgb_frame;
        int frame_idx = 0;
        double t_start = time_now();
        double t_prev  = t_start;
        int    prev_idx = 0;
        double inst_fps = 0.0;

        while (cap.read(frame)) {
            if (frame.empty()) break;

            cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
            resize(rgb_frame, in_w, resize_result);

            // resize(frame, in_w, resize_result);
            norm(resize_result.img, norm_img);
            // float2fix(norm_img, 4, fix_img);
            engine.run(norm_img);
            nms_result = &yolo_pp.process(engine.output_mats(), conf_th, iou_th);

            ++frame_idx;

            // 每 30 幀更新一次瞬時 FPS
            if (frame_idx % 30 == 0) {
                double t_now = time_now();
                double dt_ms = t_now - t_prev;
                inst_fps = (frame_idx - prev_idx) * 1000.0 / dt_ms;
                t_prev   = t_now;
                prev_idx = frame_idx;
            }

            std::cout << "\rFrame: " << frame_idx
                    << "  FPS: " << std::fixed << std::setprecision(2) << inst_fps
                    << std::flush;

            
            // ===== 繪圖 & 寫檔（僅在需要寫出時執行）=====
            if (save_video && writer.isOpened()) {
                const DetectionBatch& last = (*nms_result)[0];
                cv::Mat drawn;

                if (last.count > 0) {
                    cv::Mat boxes_padded(last.count, 4, CV_32F);
                    for (int i = 0; i < last.count; ++i) {
                        const Detection& d = last.data[i];
                        float* r = boxes_padded.ptr<float>(i);
                        r[0] = d.x1;  r[1] = d.y1;  r[2] = d.x2;  r[3] = d.y2;
                    }

                    cv::Mat boxes_orig = scale_boxes(
                        boxes_padded,
                        resize_result.ratio,
                        resize_result.pad,
                        cv::Size(frame.cols, frame.rows));

                    std::vector<Detection> dets_drawable(last.count);
                    for (int i = 0; i < last.count; ++i) {
                        const float* r = boxes_orig.ptr<float>(i);
                        dets_drawable[i] = Detection{
                            r[0], r[1], r[2], r[3],
                            last.data[i].score, last.data[i].class_id
                        };
                    }

                    drawn = draw_boxes(frame, dets_drawable);
                } else {
                    drawn = frame;
                }

                // 影片畫面疊上 FPS
                {
                    std::ostringstream ss;
                    ss << "FPS: " << std::fixed << std::setprecision(2) << inst_fps;
                    cv::putText(drawn, ss.str(), cv::Point(10, 30),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                                cv::Scalar(0, 255, 0), 2);
                }

                writer.write(drawn);
            }
        }

        if (save_video && writer.isOpened()) {
            writer.release();
        }

        // 結束後印總平均
        double t_end     = time_now();
        double total_ms  = t_end - t_start;
        double avg_fps   = frame_idx * 1000.0 / total_ms;
        std::cout << "\nTotal: " << frame_idx << " frames in "
                << total_ms / 1000.0 << " s  (avg " << avg_fps << " FPS)\n";

        if (save_video && !out_file.empty()) {
            std::cout << "結果影片已存至: " << out_file << "\n";
        }
    }
}



void run_camera(std::string onnx_path, int cameraIdx=0, std::string out_file = "",
               double conf_th = 0.1, double iou_th = 0.45) {
    std::signal(SIGINT, signalHandler);

    // ─────────────────────────────────────────────────────────────
    //  1. 載入 ONNX_MODEL
    // ─────────────────────────────────────────────────────────────
    OnnxInferenceEngine engine(
        onnx_path,
        [](Ort::SessionOptions& opts) {
            // 如果需要多執行緒,可以打開這行:
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
    // ─────────────────────────────────────────────────────────────
    const int ch = 16;

    {
        cv::Mat dummy = cv::Mat::zeros(in_h, in_w, CV_32FC3);
        engine.run(dummy);
    }

    const std::vector<cv::Mat>& fmaps = engine.output_mats();
    const int no = fmaps[0].size[1];
    const int nc = no - 4 * ch;

    if (nc <= 0) {
        std::cerr << "模型輸出 channel 數 (" << no
                  << ") 與 YOLO DFL 頭假設不符 (ch=" << ch << ")\n";
        return;
    }
    std::cout << "推導出的 nc = " << nc << "\n";

    YOLOPostProcessor yolo_pp(1, in_h, in_w, nc, ch);

    // ─────────────────────────────────────────────────────────────
    //  3. 前處理 / 推理 / 後處理 的中間 buffer
    // ─────────────────────────────────────────────────────────────
    ResizeResult resize_result;
    cv::Mat norm_img;
    cv::Mat fix_img;
    cv::Mat float_img;

    const std::vector<DetectionBatch>* nms_result = nullptr;
 
    // ─────────────────────────────────────────────────────────────
    //  4. Camera Setting
    // ─────────────────────────────────────────────────────────────
    Camera::Config cfg;
    cfg.index  = cameraIdx;
 
    Camera cam(cfg);
 
    if (!cam.open()) {
        return ;
    }
 
    std::cout << "按 Ctrl+C 停止擷取" << std::endl;

    // ─────────────────────────────────────────────────────────────
    //  5. 主迴圈 : 讀 frame 進行推理
    // ─────────────────────────────────────────────────────────────
    cv::Mat frame, rgb_frame;
    long long frameCount = 0;
    double t_start = time_now();
    double t_prev  = t_start;
    int    prev_idx = 0;
    double inst_fps = 0.0;

    while (g_running)
    {
        // ★ 呼叫一次即可取得下一幀
        if (!cam.nextFrame(frame)) {
            break;
        }

        // ── 前處理 ──────────────────────────────────────────────────────
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        resize(rgb_frame, in_w, resize_result);
        norm(resize_result.img, norm_img);

        // ── ONNX 推理 ──────────────────────────────────────────────────────
        engine.run(norm_img);

        // ── 後處理 ──────────────────────────────────────────────────────
        nms_result = &yolo_pp.process(engine.output_mats(), conf_th, iou_th);

        // ── 計數 ──────────────────────────────────────────────────────
        ++frameCount;

        // 每 30 幀更新一次瞬時 FPS
        if (frameCount % 30 == 0) {
            double t_now = time_now();
            double dt_ms = t_now - t_prev;
            inst_fps = (frameCount - prev_idx) * 1000.0 / dt_ms;
            t_prev   = t_now;
            prev_idx = frameCount;
        }
 
        // std::cout << "Frame #" << frameCount
        //           << "  size=" << frame.cols << "x" << frame.rows
        //           << "  channels=" << frame.channels()
        //           << std::endl;


        // ── 繪圖 ──────────────────────────────────────────────────────
        bool draw_flag = true;
        cv::Mat drawn;
        if (draw_flag) {
            const DetectionBatch& last = (*nms_result)[0];

            if (last.count > 0) {
                cv::Mat boxes_padded(last.count, 4, CV_32F);
                for (int i = 0; i < last.count; ++i) {
                    const Detection& d = last.data[i];
                    float* r = boxes_padded.ptr<float>(i);
                    r[0] = d.x1;  r[1] = d.y1;  r[2] = d.x2;  r[3] = d.y2;
                }

                cv::Mat boxes_orig = scale_boxes(
                    boxes_padded,
                    resize_result.ratio,
                    resize_result.pad,
                    cv::Size(frame.cols, frame.rows));

                std::vector<Detection> dets_drawable(last.count);
                for (int i = 0; i < last.count; ++i) {
                    const float* r = boxes_orig.ptr<float>(i);
                    dets_drawable[i] = Detection{
                        r[0], r[1], r[2], r[3],
                        last.data[i].score, last.data[i].class_id
                    };
                }

                drawn = draw_boxes(frame, dets_drawable);
            } else {
                drawn = frame;
            }

            // 影片畫面疊上 FPS
            {
                std::ostringstream ss;
                ss << "FPS: " << std::fixed << std::setprecision(2) << inst_fps;
                cv::putText(drawn, ss.str(), cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(0, 255, 0), 2);
            }
        }
        
        // ── 顯示影像 ──────────────────────────────────────────────────────
        // cv::imshow("Camera", frame);
        cv::imshow("Camera", drawn);
 
        // waitKey(1)：等待 1ms 讓 GUI 更新，同時偵測按鍵
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27 /* ESC */) {
            std::cout << "使用者按下退出鍵。" << std::endl;
            break;
        }
    }

    // ── 收尾 ──────────────────────────────────────────────────────────────
    cam.close();
    cv::destroyAllWindows();
    std::cout << "共擷取 " << frameCount << " 幀，程式結束。" << std::endl;
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

    const int ITER   = 1000;
    const int WARMUP = 10;

    // conf / iou 集中在這裡，decode 跟 nms 必須用相同 conf
    const float CONF = 0.5f;
    const float IOU  = 0.45f;

    // // return benchmark(img_path, model_path, 1000, 10, CONF, IOU);
    // // run_camera(model_path);
    // run_video(model_path, "/home/jianhua/Desktop/test_videos/out/DJI_0001_combined.mp4", "/home/jianhua/Desktop/test_videos/pred1.mp4", CONF);
    // // run_video(model_path, "/home/jianhua/Desktop/test_videos/out/DJI_0001_combined.mp4", "", CONF);


    run_camera(model_path, 0, "", CONF, IOU);
    return 0;
}