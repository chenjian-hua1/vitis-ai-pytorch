#include "modelrunner.h"
#include "tracker.h"
#include "stream.h"
#include "drawer.h"
#include "yolopproc.h"
#include "camera.h"
#include "preproc.h"
#include "cli_args.h"

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
#include <cstring>
#include <csignal>

// 偵測 Ctrl-C
static volatile bool g_running = true;
void signalHandler(int) { g_running = false; }
 

using Clock = std::chrono::high_resolution_clock;

double time_now() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}

void run_camera(std::string xmodel_path, Camera::Config cam_conf,
                double conf_th = 0.1, double iou_th = 0.45,
                std::string out_file = "", bool draw=true,
                bool stream=true, stream_params stream_param={}
                ) {
    std::signal(SIGINT, signalHandler);

    // ─────────────────────────────────────────────────────────────
    //  1. 載入 Xmodel & 基本設定
    // ─────────────────────────────────────────────────────────────
    XmodelInferenceEngine engine(xmodel_path);

    const int in_w = static_cast<int>(engine.in_w());
    const int in_h = static_cast<int>(engine.in_h());
    std::cout << "模型輸入: " << in_w << "x" << in_h
              << "  輸出數量: " << engine.num_outputs() << "\n";

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

    ResizeResult resize_result;
    cv::Mat norm_img;
    std::vector<cv::Mat> float_outputs(engine.num_outputs());
    const std::vector<DetectionBatch>* nms_result = nullptr;

    cv::Mat dpu_input;
    engine.bind_input_mat(dpu_input);

    int in_fix_point = static_cast<int>(std::round(std::log2(engine.input_scale())));
    std::vector<int> out_fix_points(engine.num_outputs());
    for (size_t i = 0; i < engine.num_outputs(); ++i) {
        out_fix_points[i] = static_cast<int>(std::round(-std::log2(engine.output_scale(i))));
    }
 
    // ─────────────────────────────────────────────────────────────
    //  2. Camera Setting
    // ─────────────────────────────────────────────────────────────
    Camera cam(cam_conf);
 
    if (!cam.open()) {
        return ;
    }

    FrameGrabber grabber(cam);
    // grabber.start();
 
    std::cout << "按 Ctrl+C 停止擷取" << std::endl;

    // ─────────────────────────────────────────────────────────────
    //  3. Video Streamer Setting
    // ─────────────────────────────────────────────────────────────
    RtpJpegStreamer *streamer;
    if (stream) {
        streamer = new RtpJpegStreamer(stream_param.width, stream_param.height, stream_param.fps, stream_param.ip, stream_param.port, stream_param.quality);
        if (!streamer->isOpened()) {
            std::cerr << "錯誤：無法開啟 GStreamer 發送管線" << std::endl;
            return ;
        }
        std::cout << "串流成功：已開啟 GStreamer 發送管線" << std::endl;
    }

    // ─────────────────────────────────────────────────────────────
    //  4. Tracking Setting
    // ─────────────────────────────────────────────────────────────
    bytetrack::Params p;
    p.max_lost_seconds = 2.;
    p.class_aware  = true;   // 只讓同 class 配對
    bytetrack::BYTETracker tracker(p);

    std::vector<bytetrack::Box> boxes;

    // ─────────────────────────────────────────────────────────────
    //  5. 讀取 Frame 進行推理
    // ─────────────────────────────────────────────────────────────
    cv::Mat frame, drawn;
    cv::Mat rgb_buf;   // 迴圈外宣告,重複使用
    long long frameCount = 0;
    long long lastFrameId = -1, curFrameId = -1;
 
    double t_prev = time_now();
    long long prev_idx = 0;
 
    // 分段耗時統計
    double sum_wait_ms  = 0;  // 等待新幀的時間
    double sum_infer_ms = 0;  // 推理 pipeline 耗時

    double t_inf1 = 0;
    double t_now = 0;
    double dt_ms = 0;
    double inst_fps = 0;

    // while (g_running)
    while (cam.isOpened())
    {
        // ─── 等待新的一幀（非阻塞 polling，背景執行緒已在跑）───────────────────
        double t_wait0 = time_now();
        cam.nextFrame(frame);
        // while (g_running) {
        //     if (grabber.getLatest(frame, curFrameId) && curFrameId != lastFrameId) {
        //         break;
        //     }
        //     std::this_thread::sleep_for(std::chrono::milliseconds(1));
        // }
        // if (!g_running) break;
        // lastFrameId = curFrameId;


        double t_wait1 = time_now();

        const double t_cap = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();

        // ─── 逐 frame 推理 ───────────────────────────────────────────────
        double t_inf0 = time_now();
        resize(frame, in_w, resize_result);
        cv::cvtColor(resize_result.img, rgb_buf, cv::COLOR_BGR2RGB);
        norm_and_fix(rgb_buf, in_fix_point, dpu_input);

        engine.run();

        // All Feature Map 
        for (size_t out_idx = 0; out_idx < engine.num_outputs(); ++out_idx) {
            fix2float(engine.output_mat_nchw(out_idx),
                        out_fix_points[out_idx],
                        float_outputs[out_idx]);
        }


        // ─── 後處理 ───────────────────────────────────────────────
        nms_result = &yolo_pp.process(float_outputs, conf_th, iou_th);

        // ─── Track ──────────────────────────────────────────────────────
        // scale_detections((*nms_result)[0], boxes, resize_result, cv::Size(frame.cols, frame.rows));

        map_detections((*nms_result)[0], boxes, 0.f, 0.f, 1.f, 1.f, cv::Size(in_w, in_w));
        const std::vector<bytetrack::Track>& tracks = tracker.update(boxes, t_cap);
 
        // ─── 時間計數  ──────────────────────────────────────────────────
        t_inf1 = time_now();

        sum_wait_ms  += (t_wait1 - t_wait0);
        sum_infer_ms += (t_inf1 - t_inf0);
        ++frameCount;
 
        if (frameCount % 30 == 0) {
            t_now  = time_now();
            dt_ms  = t_now - t_prev;
            inst_fps = (frameCount - prev_idx) * 1000.0 / dt_ms;
            t_prev   = t_now;
            prev_idx = frameCount;
 
            double avg_wait  = sum_wait_ms  / 30.0;
            double avg_infer = sum_infer_ms / 30.0;
 
            std::cout << "\rFrame: " << frameCount
                      << "  FPS: "        << std::fixed << std::setprecision(2) << inst_fps
                      << "  | Wait: "     << avg_wait  << " ms"
                      << "  | Infer: "    << avg_infer << " ms"
                      << "  | GrabberFps: " << grabber.frameId() // debug: 背景擷取總幀數
                      << std::endl;
 
            sum_wait_ms  = 0;
            sum_infer_ms = 0;
        }

        // ─── 繪製辨識box  ───────────────────────────────────────────────
        if (draw) {
            // draw_detection(frame, drawn, (*nms_result)[0], resize_result, inst_fps);
            draw_tracking(resize_result.img, drawn, tracks, inst_fps);
        }
        else {
            // drawn = frame;
            drawn = resize_result.img;
        }

        // ─── 將影像串流到電腦  ──────────────────────────────────────────────
        if (stream)
            streamer->send(drawn);
    }

    // ── 關閉攝影機 ─────────────────────────────────────────────────────────
    cam.close();
    grabber.stop();
    streamer->close();
    std::cout << "共擷取 " << frameCount << " 幀，程式結束。" << std::endl;
}


int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return -1;
    }

    stream_params stream_param{args.st_ip, args.st_port, args.st_width, args.st_height, args.st_fps, args.st_quality};
    Camera::Config cam_cfg{args.cam_index, args.cam_width, args.cam_height, args.cam_fps, args.cam_fourcc};

    run_camera(args.model_path, cam_cfg, args.conf, args.iou, "", args.draw, args.stream, stream_param);

    return 0;
}
