#include "util.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>

using Clock = std::chrono::high_resolution_clock;

double time_now() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}


void benchmark(
    std::string xmodel_path, cv::Mat &img, 
    int warmup = 10, int iter = 1000, 
    double conf_th = 0.2, double iou_th = 0.45) 
{
    std::cout << "===== YOLO DPU (Xmodel) 推理效能測試 (平均 " << iter << " 次) =====\n";

    // 1. 初始化 Xmodel 引擎
    XmodelInferenceEngine engine(xmodel_path);

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
        return;
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

    // 4. warmup
    for (int i = 0; i < warmup; ++i) {
        resize(img, in_w, resize_result);
        norm(resize_result.img, norm_img);
        
        float2fix(norm_img, in_fix_point, dpu_input); // Zero-copy 寫入 DPU 實體記憶體
        engine.run();                                 // 執行推理，內部排版為 NCHW int8
        
        // 反量化：從 NCHW int8 替身轉回 NCHW float
        for (size_t out_idx = 0; out_idx < engine.num_outputs(); ++out_idx) {
            fix2float(engine.output_mat_nchw(out_idx), out_fix_points[out_idx], float_outputs[out_idx]);
        }
        
        nms_result = &yolo_pp.process(float_outputs, conf_th, iou_th);
    }
    std::cout << "warmup 後偵測到 " << (*nms_result)[0].count << " 個框\n\n";

    // 5. 前處理計時
    double t_resize = 0, t_norm = 0, t_f2f = 0;
    for (int i = 0; i < iter; ++i) {
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
    for (int i = 0; i < iter; ++i) {
        double t0 = time_now();
        engine.run();
        t_infer += time_now() - t0;
    }

    // 7. 後處理計時 (反量化 -> Decode -> NMS)
    double t_f2f_back = 0, t_post = 0, t_nms = 0;

    for (int i = 0; i < iter; ++i) {
        double t0 = time_now();
        for (size_t out_idx = 0; out_idx < engine.num_outputs(); ++out_idx) {
            fix2float(engine.output_mat_nchw(out_idx), out_fix_points[out_idx], float_outputs[out_idx]);
        }
        t_f2f_back += time_now() - t0;
    }

    for (int i = 0; i < iter; ++i) {
        double t0 = time_now();
        yolo_pp.decode(float_outputs, conf_th);
        t_post += time_now() - t0;

        t0 = time_now();
        yolo_pp.nms(conf_th, iou_th);
        t_nms += time_now() - t0;
    }

    // 8. 輸出結果
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "===== 前處理 =====\n";
    std::cout << "Resize avg      : " << t_resize / iter << " ms\n";
    std::cout << "Norm avg        : " << t_norm / iter   << " ms\n";
    std::cout << "Float2Fix avg   : " << t_f2f / iter    << " ms (Zero-copy)\n";
    
    double preprocess = (t_resize + t_norm + t_f2f) / iter;
    std::cout << "Total preprocess: " << preprocess << " ms\n";

    std::cout << "\n===== 推理 =====\n";
    double infer = t_infer / iter;
    std::cout << "DPU inference   : " << infer << " ms\n";

    std::cout << "\n===== 後處理 =====\n";
    std::cout << "Fix2Float avg   : " << t_f2f_back / iter << " ms (Dequantize)\n";
    std::cout << "YOLO decode avg : " << t_post / iter     << " ms\n";
    std::cout << "NMS avg         : " << t_nms / iter      << " ms\n";

    double postprocess = (t_f2f_back + t_post + t_nms) / iter;
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
}

void run_video(std::string xmodel_path, std::string video_path, std::string out_file = "",
               double conf_th = 0.1, double iou_th = 0.45) {
    // 1. 初始化 Xmodel 引擎
    XmodelInferenceEngine engine(xmodel_path);

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
        return;
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

    // ======================== 取小寫副檔名（不含點）===================================
    std::string ext = std::filesystem::path(video_path).extension().string();
    if (!ext.empty() && ext.front() == '.') ext.erase(0, 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // ======================== YUV 原始影片 ============================================
    if (ext == "yuv" || ext == "i420" || ext == "nv12" || ext == "nv21")
    {
        // ===== 從檔名解析規格: name_WxH_FPS_FORMAT.yuv =====
        // 範例: DJI_0001_1920x1080_30_nv12.yuv
        int    src_w = 0, src_h = 0;
        double src_fps = 30.0;
        std::string yuv_fmt = (ext == "yuv") ? "i420" : ext;

        {
            std::string stem = std::filesystem::path(video_path).stem().string();
            std::regex re(R"(_(\d+)x(\d+)_(\d+)_(\w+)$)");
            std::smatch m;
            if (std::regex_search(stem, m, re)) {
                src_w   = std::stoi(m[1]);
                src_h   = std::stoi(m[2]);
                src_fps = std::stod(m[3]);
                yuv_fmt = m[4];
                std::transform(yuv_fmt.begin(), yuv_fmt.end(), yuv_fmt.begin(),
                               [](unsigned char c){ return std::tolower(c); });
            } else {
                std::cerr << "[YUV] 無法從檔名解析規格 (預期格式 name_WxH_FPS_FORMAT.yuv)\n"
                          << "      使用預設值 1920x1080 @ 30 fps i420\n";
                src_w = 1920; src_h = 1080; src_fps = 30.0; yuv_fmt = "i420";
            }
        }

        // ===== 計算每幀大小與轉換代碼 =====
        size_t frame_size = 0;
        int    cv_yuv_code = -1;
        if (yuv_fmt == "i420" || yuv_fmt == "yuv420p") {
            frame_size  = static_cast<size_t>(src_w) * src_h * 3 / 2;
            cv_yuv_code = cv::COLOR_YUV2BGR_I420;
        } else if (yuv_fmt == "nv12") {
            frame_size  = static_cast<size_t>(src_w) * src_h * 3 / 2;
            cv_yuv_code = cv::COLOR_YUV2BGR_NV12;
        } else if (yuv_fmt == "nv21") {
            frame_size  = static_cast<size_t>(src_w) * src_h * 3 / 2;
            cv_yuv_code = cv::COLOR_YUV2BGR_NV21;
        } else if (yuv_fmt == "yuy2" || yuv_fmt == "yuyv") {
            frame_size  = static_cast<size_t>(src_w) * src_h * 2;
            cv_yuv_code = cv::COLOR_YUV2BGR_YUY2;
        } else {
            std::cerr << "[YUV] 不支援的格式: " << yuv_fmt << "\n";
            return;
        }

        // ===== 開檔並算總幀數 =====
        std::ifstream fp(video_path, std::ios::binary | std::ios::ate);
        if (!fp.is_open()) {
            std::cerr << "Failed to open YUV: " << video_path << "\n";
            return;
        }
        size_t file_size    = fp.tellg();
        int    total_frames = static_cast<int>(file_size / frame_size);
        fp.seekg(0, std::ios::beg);

        std::cout << "[YUV] " << video_path << "  "
                  << src_w << "x" << src_h
                  << "  fps=" << src_fps
                  << "  frames=" << total_frames
                  << "  fmt=" << yuv_fmt << "\n";

        // ===== 影片寫出器（僅在 out_file 非空時建立）=====
        const bool save_video = !out_file.empty();
        cv::VideoWriter writer;
        if (save_video) {
            // writer.open(out_file,
            //             cv::VideoWriter::fourcc('m','p','4','v'),
            //             src_fps,
            //             cv::Size(src_w, src_h));

            // 改成
            std::string avi_out = std::filesystem::path(out_file).replace_extension(".avi").string();
            writer.open(avi_out,
                        cv::VideoWriter::fourcc('M','J','P','G'),
                        src_fps,
                        cv::Size(src_w, src_h));
            if (!writer.isOpened()) {
                std::cerr << "Failed to open VideoWriter: " << out_file
                          << "  (將不寫出影片繼續執行)\n";
            }
        }

        // ===== 主迴圈 =====
        std::vector<uint8_t> yuv_buf(frame_size);
        cv::Mat frame;
        int frame_idx = 0;
        double t_start = time_now();
        double t_prev  = t_start;
        int    prev_idx = 0;
        double inst_fps = 0.0;

        while (fp.read(reinterpret_cast<char*>(yuv_buf.data()), frame_size)) {
            // YUV → BGR
            if (yuv_fmt == "yuy2" || yuv_fmt == "yuyv") {
                cv::Mat yuv(src_h, src_w, CV_8UC2, yuv_buf.data());
                cv::cvtColor(yuv, frame, cv_yuv_code);
            } else {
                cv::Mat yuv(src_h * 3 / 2, src_w, CV_8UC1, yuv_buf.data());
                cv::cvtColor(yuv, frame, cv_yuv_code);
            }

            if (frame.empty()) break;

            resize(frame, in_w, resize_result);
            norm(resize_result.img, norm_img);
            float2fix(norm_img, in_fix_point, dpu_input);
            engine.run();

            for (size_t out_idx = 0; out_idx < engine.num_outputs(); ++out_idx) {
                fix2float(engine.output_mat_nchw(out_idx), out_fix_points[out_idx], float_outputs[out_idx]);
            }

            nms_result = &yolo_pp.process(float_outputs, conf_th, iou_th);

            ++frame_idx;

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

                std::ostringstream ss;
                ss << "FPS: " << std::fixed << std::setprecision(2) << inst_fps;
                cv::putText(drawn, ss.str(), cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(0, 255, 0), 2);

                writer.write(drawn);
            }
        }

        if (save_video && writer.isOpened()) {
            writer.release();
        }

        double t_end    = time_now();
        double total_ms = t_end - t_start;
        double avg_fps  = frame_idx * 1000.0 / total_ms;
        std::cout << "\nTotal: " << frame_idx << " frames in "
                  << total_ms / 1000.0 << " s  (avg " << avg_fps << " FPS)\n";

        if (save_video && !out_file.empty()) {
            std::cout << "結果影片已存至: " << out_file << "\n";
        }
    }
    // ======================== 一般影片 (mp4 / avi / mov ...) =========================
    else if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
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
        int frame_idx = 0;
        double t_start = time_now();
        double t_prev  = t_start;
        int    prev_idx = 0;
        double inst_fps = 0.0;

        while (cap.read(frame)) {
            if (frame.empty()) break;

            resize(frame, in_w, resize_result);
            norm(resize_result.img, norm_img);
            float2fix(norm_img, in_fix_point, dpu_input);
            engine.run();

            // all fmaps fix (int8) to float32
            for (size_t out_idx = 0; out_idx < engine.num_outputs(); ++out_idx) {
                fix2float(engine.output_mat_nchw(out_idx), out_fix_points[out_idx], float_outputs[out_idx]);
            }

            nms_result = &yolo_pp.process(float_outputs, conf_th, iou_th);

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
    else {
        std::cerr << "不支援的副檔名: " << ext << "\n";
    }
}



int main(int argc, char** argv) {
    // ─────────────────────────────────────────────────────────────
    //  參數：argv[1] = 圖片路徑，argv[2] = Xmodel 模型路徑
    // ─────────────────────────────────────────────────────────────
    std::string src = (argc > 1)
        ? argv[1]
        : "/home/jianhua/Desktop/vitis-ai-pytorch/dpu_inference/2308.jpg";
    std::string model_path = (argc > 2)
        ? argv[2]
        : "model/YOLO_int.xmodel";

    const int ITER   = 1000;
    const int WARMUP = 10;
    const float CONF = 0.1f;
    const float IOU  = 0.45f;

    // cv::Mat img = cv::imread(src);
    // if (img.empty()) {
    //     std::cerr << "無法開啟檔案: " << src << "\n";
    //     return -1;
    // }
    // benchmark(model_path, img, 10, 1000, CONF, IOU);

    run_video(model_path, src, "pred.avi", CONF, IOU);

    return 0;
}