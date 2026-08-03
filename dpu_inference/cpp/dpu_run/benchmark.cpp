// ============================================================================
// ./benchmark ./model/yolo11n_int.xmodel
// ./benchmark ./model/yolo11n_int.xmodel -n 300 --cam-fps 30 --no-track
// ./benchmark ./model/yolo11n_int.xmodel --stream --ip=192.168.1.50 --port=5000
// ./benchmark --help

//  benchmark_camera.cpp
//  Camera 擷取/解碼 → Resize → BGR2RGB → Norm+Fix → DPU Inference
//                   → Fix2Float → DFL Decode → NMS → Track → Draw → Stream
//  逐階段計時（avg / min / p50 / p95 / max / std / 佔比）
// ============================================================================
//
//  BENCH_DIRECT_CAPTURE = 1 : 直接同步 cam.read()，Capture 欄位 = 真實「擷取+解碼」耗時
//                             （會被 camera FPS 阻塞，但這才是解碼的真實成本）
//  BENCH_DIRECT_CAPTURE = 0 : 沿用 FrameGrabber 背景執行緒，Capture 欄位 = 等待新幀的時間
//                             （反映的是 pipeline 相對於 camera FPS 的餘裕）
//

#ifndef BENCH_DIRECT_CAPTURE
#define BENCH_DIRECT_CAPTURE 0
#endif

#include "cli_args.h"
#include "modelrunner.h"
#include "tracker.h"
#include "stream.h"
#include "drawer.h"
#include "yolopproc.h"
#include "camera.h"
#include "preproc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>


// 偵測 Ctrl-C
static volatile bool g_running = true;
void signalHandler(int) { g_running = false; }
 

using Clock = std::chrono::high_resolution_clock;

double time_now() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}


// ─────────────────────────────────────────────────────────────────────────────
//  分段計時器
// ─────────────────────────────────────────────────────────────────────────────
struct StageTimer {
    std::string          name;
    std::vector<double>  samples;   // 單位: ms

    StageTimer() {}
    explicit StageTimer(const std::string& n) : name(n) {}

    inline void add(double ms) { samples.push_back(ms); }
    bool   empty() const { return samples.empty(); }
    size_t n()     const { return samples.size(); }

    double sum() const {
        double s = 0.0;
        for (size_t i = 0; i < samples.size(); ++i) s += samples[i];
        return s;
    }
    double avg() const { return samples.empty() ? 0.0 : sum() / samples.size(); }

    double mn() const {
        return samples.empty() ? 0.0 : *std::min_element(samples.begin(), samples.end());
    }
    double mx() const {
        return samples.empty() ? 0.0 : *std::max_element(samples.begin(), samples.end());
    }
    // q ∈ [0,1]
    double pct(double q) const {
        if (samples.empty()) return 0.0;
        std::vector<double> t = samples;
        size_t k = static_cast<size_t>(q * (t.size() - 1) + 0.5);
        std::nth_element(t.begin(), t.begin() + k, t.end());
        return t[k];
    }
    double stddev() const {
        if (samples.size() < 2) return 0.0;
        double m = avg(), acc = 0.0;
        for (size_t i = 0; i < samples.size(); ++i) {
            double d = samples[i] - m;
            acc += d * d;
        }
        return std::sqrt(acc / (samples.size() - 1));
    }
};

// 一行表格
static void print_row(const StageTimer& s, double total_for_pct) {
    if (s.empty()) return;
    double share = (total_for_pct > 0.0) ? (s.avg() / total_for_pct * 100.0) : 0.0;
    std::cout << std::left  << std::setw(22) << s.name
              << std::right << std::fixed   << std::setprecision(3)
              << std::setw(9) << s.avg()
              << std::setw(9) << s.mn()
              << std::setw(9) << s.pct(0.50)
              << std::setw(9) << s.pct(0.95)
              << std::setw(9) << s.mx()
              << std::setw(9) << s.stddev()
              << std::setw(8) << std::setprecision(1) << share << "%"
              << "\n";
}

static void print_header(const std::string& title) {
    std::cout << "\n" << title << "\n";
    std::cout << std::left  << std::setw(22) << "Stage"
              << std::right << std::setw(9)  << "avg"
              << std::setw(9) << "min"
              << std::setw(9) << "p50"
              << std::setw(9) << "p95"
              << std::setw(9) << "max"
              << std::setw(9) << "std"
              << std::setw(9) << "share"
              << "\n";
    std::cout << std::string(85, '-') << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  主體
// ─────────────────────────────────────────────────────────────────────────────
void benchmark_camera(std::string xmodel_path,
                      Camera::Config cam_conf,
                      int    warmup    = 30,
                      int    iter      = 300,
                      double conf_th   = 0.2,
                      double iou_th    = 0.45,
                      bool   do_track  = true,
                      bool   draw      = false,
                      bool   stream    = false,
                      stream_params stream_param = {},
                      std::string save_last = "benchmark_last_frame.jpg")
{
    std::signal(SIGINT, signalHandler);

    std::cout << "===== Camera → DPU 全流程分段效能測試 (warmup "
              << warmup << " 幀, 統計 " << iter << " 幀) =====\n";
#if BENCH_DIRECT_CAPTURE
    std::cout << "擷取模式: 同步 cam.read()（Capture = 擷取+解碼真實耗時）\n";
#else
    std::cout << "擷取模式: FrameGrabber 背景執行緒（Capture = 等待新幀耗時）\n";
#endif

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
    //  2. Camera
    // ─────────────────────────────────────────────────────────────
    Camera cam(cam_conf);
    if (!cam.open()) {
        std::cerr << "錯誤：無法開啟攝影機\n";
        return;
    }
#if !BENCH_DIRECT_CAPTURE
    FrameGrabber grabber(cam);
    grabber.start();
    long long lastFrameId = -1, curFrameId = -1;
#endif

    // ─────────────────────────────────────────────────────────────
    //  3. Streamer（可選，只為量測 send 的成本）
    // ─────────────────────────────────────────────────────────────
    RtpJpegStreamer* streamer = nullptr;
    if (stream) {
        streamer = new RtpJpegStreamer(stream_param.width, stream_param.height,
                                       stream_param.fps, stream_param.ip,
                                       stream_param.port, stream_param.quality);
        if (!streamer->isOpened()) {
            std::cerr << "錯誤：無法開啟 GStreamer 發送管線\n";
            delete streamer;
            streamer = nullptr;
            stream = false;
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  4. Tracker
    // ─────────────────────────────────────────────────────────────
    bytetrack::Params p;
    p.max_lost_seconds = 2.0;
    p.class_aware      = true;
    bytetrack::BYTETracker tracker(p);
    std::vector<bytetrack::Box> boxes;

    // ─────────────────────────────────────────────────────────────
    //  5. 計時器
    // ─────────────────────────────────────────────────────────────
    StageTimer s_cap   ("Capture/Decode");
    StageTimer s_resize("Resize+LetterBox");
    StageTimer s_cvt   ("BGR2RGB");
    StageTimer s_quant ("Norm+Fix (quant)");
    StageTimer s_infer ("DPU Inference");
    StageTimer s_deq   ("Fix2Float (dequant)");
    StageTimer s_dfl   ("DFL Decode");
    StageTimer s_nms   ("NMS");
    StageTimer s_track ("Scale+Track");
    StageTimer s_draw  ("Draw");
    StageTimer s_send  ("Stream Send");
    StageTimer s_loop  ("[Loop total]");

    cv::Mat frame, drawn;
    long long det_sum = 0, trk_sum = 0;
    int  done = 0;
    bool aborted = false;

    // ── 取得一幀（依模式切換）──────────────────────────────────────
    // 回傳 false 表示中止
    auto acquire = [&](double& ms) -> bool {
        double t0 = time_now();
#if BENCH_DIRECT_CAPTURE
        // Camera 若沒有 read()，改成你的 API（例如 cam.grab(frame) / cam.capture(frame)）
        bool ok = cam.nextFrame(frame);
        if (!ok || frame.empty()) { ms = 0; return false; }
#else
        while (g_running) {
            if (grabber.getLatest(frame, curFrameId) && curFrameId != lastFrameId) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!g_running) { ms = 0; return false; }
        lastFrameId = curFrameId;
#endif
        ms = time_now() - t0;
        return true;
    };

    // ─────────────────────────────────────────────────────────────
    //  6. Warmup（不計入統計）
    // ─────────────────────────────────────────────────────────────
    std::cout << "\nwarmup ...\n";
    for (int i = 0; i < warmup && g_running; ++i) {
        double dummy = 0;
        if (!acquire(dummy)) { aborted = true; break; }

        resize(frame, in_w, resize_result);
        cv::cvtColor(resize_result.img, resize_result.img, cv::COLOR_BGR2RGB);
        norm_and_fix(resize_result.img, in_fix_point, dpu_input);
        engine.run();
        for (size_t o = 0; o < engine.num_outputs(); ++o)
            fix2float(engine.output_mat_nchw(o), out_fix_points[o], float_outputs[o]);
        nms_result = &yolo_pp.process(float_outputs, conf_th, iou_th);
    }
    if (nms_result)
        std::cout << "warmup 後偵測到 " << (*nms_result)[0].count << " 個框\n";

    // ─────────────────────────────────────────────────────────────
    //  7. 主測試迴圈：單一 pipeline，逐階段計時
    // ─────────────────────────────────────────────────────────────
    std::cout << "開始統計（Ctrl+C 可提早結束）...\n";
    const double t_bench0 = time_now();

    for (int i = 0; i < iter && g_running && !aborted; ++i) {
        double t_loop0 = time_now();
        double t0, ms = 0;

        // ── Capture / Decode ────────────────────────────────────
        if (!acquire(ms)) { aborted = true; break; }
        s_cap.add(ms);

        const double t_cap = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        // ── Resize + LetterBox ──────────────────────────────────
        t0 = time_now();
        resize(frame, in_w, resize_result);
        s_resize.add(time_now() - t0);

        // ── 色彩轉換 ────────────────────────────────────────────
        t0 = time_now();
        cv::cvtColor(resize_result.img, resize_result.img, cv::COLOR_BGR2RGB);
        s_cvt.add(time_now() - t0);

        // ── Normalize + 量化（zero-copy 寫進 DPU input）─────────
        t0 = time_now();
        norm_and_fix(resize_result.img, in_fix_point, dpu_input);
        s_quant.add(time_now() - t0);

        // ── DPU 推理（含 NHWC→NCHW 轉置）────────────────────────
        t0 = time_now();
        engine.run();
        s_infer.add(time_now() - t0);

        // ── 反量化 ──────────────────────────────────────────────
        t0 = time_now();
        for (size_t o = 0; o < engine.num_outputs(); ++o)
            fix2float(engine.output_mat_nchw(o), out_fix_points[o], float_outputs[o]);
        s_deq.add(time_now() - t0);

        // ── DFL Decode ──────────────────────────────────────────
        t0 = time_now();
        yolo_pp.decode(float_outputs, conf_th);
        s_dfl.add(time_now() - t0);

        // ── NMS ─────────────────────────────────────────────────
        t0 = time_now();
        yolo_pp.nms(conf_th, iou_th);
        nms_result = &yolo_pp.detections();   // 或你的 getter
        s_nms.add(time_now() - t0);

        det_sum += (*nms_result)[0].count;

        // ── 座標還原 + Tracking ─────────────────────────────────
        const std::vector<bytetrack::Track>* tracks = nullptr;
        if (do_track) {
            t0 = time_now();
            scale_detections((*nms_result)[0], boxes, resize_result,
                             cv::Size(frame.cols, frame.rows));
            tracks = &tracker.update(boxes, t_cap);
            s_track.add(time_now() - t0);
            trk_sum += static_cast<long long>(tracks->size());
        }

        // ── 繪圖 ────────────────────────────────────────────────
        if (draw) {
            t0 = time_now();
            if (do_track) draw_tracking(frame, drawn, *tracks, 0.0);
            else          draw_detection(frame, drawn, (*nms_result)[0], resize_result, 0.0);
            s_draw.add(time_now() - t0);
        } else {
            drawn = frame;
        }

        // ── 串流 ────────────────────────────────────────────────
        if (stream) {
            t0 = time_now();
            streamer->send(drawn);
            s_send.add(time_now() - t0);
        }

        s_loop.add(time_now() - t_loop0);
        ++done;

        if (done % 30 == 0) {
            std::cout << "\r  進度 " << done << "/" << iter << std::flush;
        }
    }
    const double t_bench_total = time_now() - t_bench0;
    std::cout << "\r  進度 " << done << "/" << iter << "\n";

    // ─────────────────────────────────────────────────────────────
    //  8. 關閉資源
    // ─────────────────────────────────────────────────────────────
#if !BENCH_DIRECT_CAPTURE
    grabber.stop();
#endif
    cam.close();
    if (streamer) { streamer->close(); delete streamer; streamer = nullptr; }

    if (done == 0) {
        std::cout << "沒有取得任何有效幀，無法統計。\n";
        return;
    }

    // ─────────────────────────────────────────────────────────────
    //  9. 報表
    // ─────────────────────────────────────────────────────────────
    const double pre_ms  = s_resize.avg() + s_cvt.avg() + s_quant.avg();
    const double inf_ms  = s_infer.avg();
    const double post_ms = s_deq.avg() + s_dfl.avg() + s_nms.avg();
    const double extra   = s_track.avg() + s_draw.avg() + s_send.avg();
    const double compute = pre_ms + inf_ms + post_ms + extra;   // 不含擷取
    const double loop_ms = s_loop.avg();                        // 含擷取

    print_header("===== 分段耗時（單位 ms，share 以「計算總計」為分母，Capture 以 Loop 為分母）=====");
    // Capture 用 loop 當分母比較有意義
    print_row(s_cap, loop_ms);
    std::cout << std::string(85, '-') << "\n";
    print_row(s_resize, compute);
    print_row(s_cvt,    compute);
    print_row(s_quant,  compute);
    print_row(s_infer,  compute);
    print_row(s_deq,    compute);
    print_row(s_dfl,    compute);
    print_row(s_nms,    compute);
    if (do_track) print_row(s_track, compute);
    if (draw)     print_row(s_draw,  compute);
    if (stream)   print_row(s_send,  compute);
    std::cout << std::string(85, '-') << "\n";
    print_row(s_loop, loop_ms);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n===== 分組彙總 =====\n"
              << "Capture/Decode   : " << s_cap.avg() << " ms\n"
              << "前處理  (resize+cvt+quant) : " << pre_ms  << " ms\n"
              << "推理    (DPU)              : " << inf_ms  << " ms\n"
              << "後處理  (dequant+DFL+NMS)  : " << post_ms << " ms\n";
    if (extra > 0.0)
        std::cout << "其他    (track+draw+stream): " << extra << " ms\n";
    std::cout << "計算總計 (不含擷取)        : " << compute << " ms  →  "
              << std::setprecision(2) << (compute > 0 ? 1000.0 / compute : 0.0)
              << " FPS (理論上限)\n";
    std::cout << std::setprecision(3)
              << "迴圈總計 (含擷取)          : " << loop_ms << " ms  →  "
              << std::setprecision(2) << (loop_ms > 0 ? 1000.0 / loop_ms : 0.0)
              << " FPS\n";
    std::cout << "實測平均 FPS (wall clock)  : "
              << (t_bench_total > 0 ? done * 1000.0 / t_bench_total : 0.0)
              << "  (" << done << " 幀 / " << std::setprecision(1)
              << t_bench_total / 1000.0 << " s)\n";

    std::cout << std::setprecision(2)
              << "平均偵測框數 / 幀          : "
              << static_cast<double>(det_sum) / done << "\n";
    if (do_track)
        std::cout << "平均 track 數 / 幀         : "
                  << static_cast<double>(trk_sum) / done << "\n";
    if (aborted) std::cout << "（測試被中斷，統計基於已完成的 " << done << " 幀）\n";

    // ── 存最後一幀方便確認結果正確 ──────────────────────────────
    if (!save_last.empty() && !drawn.empty()) {
        if (!draw)
            draw_detection(frame, drawn, (*nms_result)[0], resize_result, 0.0);
        if (cv::imwrite(save_last, drawn))
            std::cout << "最後一幀結果已存至: " << save_last << "\n";
        else
            std::cerr << "imwrite 失敗: " << save_last << "\n";
    }
}



// ============================================================================
//  main
//  參數解析見 cli_args.h（--help 可列出所有選項）
// ============================================================================
#include "cli_args.h"
 
int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return -1;
    }
    print_args(args);
 
    // {ip, port, width, height, fps, quality}
    stream_params  stream_param{args.st_ip, args.st_port, args.st_width,
                                args.st_height, args.st_fps, args.st_quality};
    // {index, width, height, fps}
    Camera::Config cam_conf{args.cam_index, args.cam_width,
                            args.cam_height, args.cam_fps};
 
    benchmark_camera(args.model_path, cam_conf,
                     args.warmup, args.iter, args.conf, args.iou,
                     args.track, args.draw, args.stream, stream_param,
                     args.save);
 
    return 0;
}