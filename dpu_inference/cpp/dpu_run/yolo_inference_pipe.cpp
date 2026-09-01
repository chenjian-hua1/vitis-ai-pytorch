// yolo_inference_pipe.cpp — 一個資源一條時間軸的深度流水線
//
// ─────────────────────────────────────────────────────────────
//   T1 Cam          cam.nextFrame()                 CPU + 等相機
//   T2 Resize IP    hls::resize::letterbox()        resize IP
//   T3 Pre  (CPU)   cvtColor + norm_and_fix         CPU
//   T4 DPU          submit + wait_hw                DPU
//   T5 Post (CPU)   finish(轉置) + fix2float + NMS + track   CPU
//   T6 Out          draw + JPEG + RTP               CPU
// ─────────────────────────────────────────────────────────────
//
// 核心觀念:不需要非同步 API,只需要「一個資源一條執行緒」。
// 阻塞式呼叫放在專屬執行緒裡,那條執行緒就是該資源的時間軸;
// 它被硬體卡住的時候,其他執行緒照樣在跑。
//
// 吞吐 = max(各階段耗時),而不是總和。
//
// 兩個前提:
//   1. DPU 要有多組 context(見 modelrunner_pipe.h),否則 T3 會在
//      T4 讀取途中覆寫輸入張量。
//   2. 每個 slot 要有自己的緩衝,理由同上。
//
// 注意:這裡開了 6 條執行緒,但 KV260 只有 4 核。
//       T2 與 T4 大部分時間阻塞在硬體上不佔 CPU,所以可行;
//       真正搶 CPU 的是 T1/T3/T5/T6。

#include "modelrunner_pipe.h"
#include "tracker.h"
#include "stream.h"
#include "drawer.h"
#include "yolopproc.h"
#include "camera.h"
#include "preproc.h"
#include "cli_args.h"
#include "frame_pipeline.h"
#include "hls_resize.hpp"

#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <thread>
#include <vector>

static std::atomic<bool> g_running{true};
static void signalHandler(int) { g_running = false; }

namespace {

struct Slot {
    cv::Mat raw;                       // T1 -> T2(DMA 記憶體)
    const uint8_t* raw_origin = nullptr;
    double  t_cap = 0;

    hls::resize::Result hres;          // T2 -> T3
    ResizeResult        resize_result;
    cv::Mat             rgb_buf;       // T3 用,每 slot 一份

    std::vector<cv::Mat> float_outputs;          // T5 用
    std::vector<bytetrack::Track> tracks;        // T5 -> T6
    double fps_snapshot = 0;
};

// 專案的 ResizeResult 用 std::pair<float,float> 存 ratio / pad,
// 而 hls::resize::Result 用 cv::Point2f。這兩個多載讓轉接層對
// std::pair 與 .x/.y 型別都成立 —— 之後改型別也不用動這裡。
template <class T>
inline void set_xy(std::pair<T, T>& d, float x, float y) {
    d.first  = static_cast<T>(x);
    d.second = static_cast<T>(y);
}
template <class P>
inline auto set_xy(P& d, float x, float y) -> decltype(d.x, d.y, void()) {
    d.x = decltype(d.x)(x);
    d.y = decltype(d.y)(y);
}

// hls::resize::Result -> 專案原有的 ResizeResult
// 幾何行為與原本的 CPU 版 resize() 一致,下游不用改。
// .img 是淺拷貝,沒有額外的資料搬移。
inline void to_project_result(const hls::resize::Result& s, ResizeResult& d) {
    d.img     = s.img;              // cv::Mat 淺拷貝
    d.content = s.content;
    set_xy(d.ratio, s.ratio.x, s.ratio.y);
    set_xy(d.pad,   s.pad.x,   s.pad.y);
}

}  // namespace


void run_camera(std::string xmodel_path, Camera::Config cam_conf,
                double conf_th = 0.1, double iou_th = 0.45,
                std::string out_file = "", bool draw = true,
                bool stream = true, stream_params stream_param = {},
                int n_slots = 4)
{
    std::signal(SIGINT, signalHandler);

    // ---- 模型。context 數 = slot 數,讓每個 slot 各自擁有 DPU 緩衝 ----
    XmodelPipelineEngine engine(xmodel_path, n_slots);

    const int in_w = engine.in_w();
    const int in_h = engine.in_h();
    const int ch = 16;
    const int no = engine.output_mat_nchw(0, 0).size[1];
    const int nc = no - 4 * ch;
    if (nc <= 0) {
        std::cerr << "模型輸出 channel 數 (" << no << ") 與 DFL 假設不符\n";
        return;
    }
    std::cout << "模型輸入 " << in_w << "x" << in_h
              << "  輸出 " << engine.num_outputs()
              << "  nc=" << nc
              << "  DPU context " << engine.n_ctx() << "\n";

    YOLOPostProcessor yolo_pp(1, in_h, in_w, nc, ch);

    const int in_fix_point =
        static_cast<int>(std::round(std::log2(engine.input_scale())));
    std::vector<int> out_fix(engine.num_outputs());
    for (size_t i = 0; i < engine.num_outputs(); ++i)
        out_fix[i] = static_cast<int>(std::round(-std::log2(engine.output_scale(i))));

    // ---- 相機 ----
    Camera cam(cam_conf);
    if (!cam.open()) return;
    const int cam_w = cam.actualWidth();
    const int cam_h = cam.actualHeight();

    // ---- resize IP ----
    const size_t need = static_cast<size_t>(cam_w) * cam_h * 3 * n_slots
                      + static_cast<size_t>(in_w) * in_w * 3 * 2
                      + 8u * 1024 * 1024;
    hls::use_dma_heap("auto", ((need >> 20) + 16) << 20);
    hls::resize::use_devmem();
    bool ip_ready = hls::resize::available();
    std::cout << "[resize IP] " << hls::pool_info() << "\n"
              << "[resize IP] " << (ip_ready ? hls::resize::device_info()
                                             : "不可用:" + hls::resize::last_error())
              << "\n";

    // ---- 串流 ----
    std::unique_ptr<RtpJpegStreamer> streamer;
    if (stream) {
        streamer = std::make_unique<RtpJpegStreamer>(
            stream_param.width, stream_param.height, stream_param.fps,
            stream_param.ip, stream_param.port, stream_param.quality);
        if (!streamer->isOpened()) { std::cerr << "GStreamer 開啟失敗\n"; return; }
    }

    // ---- 追蹤 ----
    bytetrack::Params tp;
    tp.max_lost_seconds = 2.;
    tp.class_aware = true;
    bytetrack::BYTETracker tracker(tp);

    // ---- slot ----
    std::vector<Slot> slots(n_slots);
    for (int i = 0; i < n_slots; ++i) {
        if (ip_ready) {
            slots[i].raw = hls::resize::input_buffer(cam_w, cam_h, i);
            if (slots[i].raw.empty()) ip_ready = false;
        }
        if (slots[i].raw.empty()) slots[i].raw.create(cam_h, cam_w, CV_8UC3);
        slots[i].raw_origin = slots[i].raw.data;
        slots[i].float_outputs.resize(engine.num_outputs());
    }
    std::atomic<bool> zero_copy_ok{ip_ready};

    fpipe::Stage q_free, q1, q2, q3, q4, q5;
    for (int i = 0; i < n_slots; ++i) q_free.push(i);

    fpipe::StageTimer tm1, tm2, tm3, tm4, tm5, tm6;
    // T5 內部細分,17 ms 到底是轉置、反量化還是 NMS 要分開才知道
    fpipe::StageTimer tm5a, tm5b, tm5c;
    // T6 內部細分:繪製 vs 送出
    fpipe::StageTimer tm6a, tm6b;

    std::atomic<long long> n_out{0}, n_drop{0}, n_stale{0};
    std::atomic<double> cur_fps{0.0};
    const double t_start = fpipe::now_ms();

    // ===== T1 Cam =====
    std::thread t1([&] {
        while (g_running) {
            const int s = q_free.pop(0);
            if (s < 0) {                       // 沒有空 slot:丟幀但仍要把影格取走,
                cv::Mat discard;               // 否則 V4L2 佇列會塞住
                cam.nextFrame(discard);
                ++n_drop;
                continue;
            }
            const double t0 = fpipe::now_ms();
            const bool ok = cam.nextFrame(slots[s].raw);
            tm1.add(fpipe::now_ms() - t0);

            if (!ok || slots[s].raw.empty()) { q_free.push(s); continue; }
            if (zero_copy_ok && slots[s].raw.data != slots[s].raw_origin) {
                zero_copy_ok = false;
                std::cerr << "\n[resize IP] 緩衝被重新配置,zero-copy 失效\n";
            }
            slots[s].t_cap = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            q1.push(s);
        }
        q1.stop();
    });

    // ===== T2 Resize IP =====
    std::thread t2([&] {
        while (true) {
            const int s = q1.pop(200);
            if (s < 0) { if (q1.stopped()) break; continue; }
            const double t0 = fpipe::now_ms();
            if (ip_ready) {
                hls::resize::letterbox(slots[s].raw, in_w, slots[s].hres);
                to_project_result(slots[s].hres, slots[s].resize_result);
            } else {
                resize(slots[s].raw, in_w, slots[s].resize_result);
            }
            tm2.add(fpipe::now_ms() - t0);
            q2.push(s);
        }
        q2.stop();
    });

    // ===== T3 前處理(CPU)=====
    std::thread t3([&] {
        while (true) {
            const int s = q2.pop(200);
            if (s < 0) { if (q2.stopped()) break; continue; }
            const double t0 = fpipe::now_ms();
            cv::cvtColor(slots[s].resize_result.img, slots[s].rgb_buf,
                         cv::COLOR_BGR2RGB);
            // 每個 slot 寫自己那組 DPU 輸入緩衝,不會互相覆寫
            cv::Mat dpu_in = engine.input_mat(s);
            norm_and_fix(slots[s].rgb_buf, in_fix_point, dpu_in);
            tm3.add(fpipe::now_ms() - t0);
            q3.push(s);
        }
        q3.stop();
    });

    // ===== T4 DPU =====
    std::thread t4([&] {
        while (true) {
            const int s = q3.pop(200);
            if (s < 0) { if (q3.stopped()) break; continue; }
            const double t0 = fpipe::now_ms();
            engine.submit(s);
            engine.wait_hw(s);
            tm4.add(fpipe::now_ms() - t0);
            q4.push(s);
        }
        q4.stop();
    });

    // ===== T5 後處理(CPU)+ 追蹤 =====
    std::thread t5([&] {
        std::vector<bytetrack::Box> boxes;
        long long frames = 0, prev = 0;
        double t_prev = fpipe::now_ms();
        fpipe::Ema fps_ema(0.3);

        while (true) {
            const int s = q4.pop(200);
            if (s < 0) { if (q4.stopped()) break; continue; }
            const double t0 = fpipe::now_ms();

            double ta = fpipe::now_ms();
            engine.finish(s);                       // memcpy + NHWC->NCHW 轉置
            tm5a.add(fpipe::now_ms() - ta);

            ta = fpipe::now_ms();
            for (size_t i = 0; i < engine.num_outputs(); ++i)
                fix2float(engine.output_mat_nchw(s, i), out_fix[i],
                          slots[s].float_outputs[i]);
            tm5b.add(fpipe::now_ms() - ta);

            ta = fpipe::now_ms();
            const std::vector<DetectionBatch>& nms =
                yolo_pp.process(slots[s].float_outputs, conf_th, iou_th);
            map_detections(nms[0], boxes, 0.f, 0.f, 1.f, 1.f, cv::Size(in_w, in_w));
            // tracker 有狀態且需依序更新,固定在這一條執行緒
            slots[s].tracks = tracker.update(boxes, slots[s].t_cap);
            tm5c.add(fpipe::now_ms() - ta);

            tm5.add(fpipe::now_ms() - t0);

            if (++frames % 30 == 0) {
                const double now = fpipe::now_ms();
                const double inst = (frames - prev) * 1000.0 / (now - t_prev);
                const double smooth = fps_ema.push(inst);
                const double avg = frames * 1000.0 / (now - t_start);
                cur_fps = smooth;
                t_prev = now; prev = frames;

                std::cout << "Frame " << frames
                          << "  FPS " << std::fixed << std::setprecision(1)
                          << smooth << " (瞬時 " << inst << ", 累計 " << avg << ")"
                          << " | cam " << std::setprecision(2) << tm1.avg()
                          << "  rszIP " << tm2.avg()
                          << "  pre " << tm3.avg()
                          << "  dpu " << tm4.avg()
                          << "  post " << tm5.avg()
                          << " (轉置 " << tm5a.avg() << " 反量化 " << tm5b.avg()
                          << " NMS " << tm5c.avg() << ")"
                          << "  out " << tm6.avg()
                          << " (draw " << tm6a.avg() << " send " << tm6b.avg() << ")"
                          << " | 佇列 " << q1.depth() << q2.depth() << q3.depth()
                          << q4.depth() << q5.depth()
                          << "  丟棄 " << n_drop.load()
                          << "  未送出 " << n_stale.load() << std::endl;
            }
            slots[s].fps_snapshot = cur_fps.load();
            q5.push(s);
        }
        q5.stop();
    });

    // ===== T6 繪製 + 串流 =====
    std::thread t6([&] {
        cv::Mat drawn;
        while (true) {
            // 只處理最新的一幀,積壓的直接回收。
            // 串流是監看用途,不該透過 slot 回收反壓整條 pipeline。
            const int s = q5.pop_latest(200, [&](int stale) {
                ++n_stale;
                ++n_out;              // 這幀有完成推理,只是沒送出去
                q_free.push(stale);
            });
            if (s < 0) { if (q5.stopped()) break; continue; }

            const double t0 = fpipe::now_ms();

            double ta = fpipe::now_ms();
            if (draw) draw_tracking(slots[s].resize_result.img, drawn,
                                    slots[s].tracks, slots[s].fps_snapshot);
            else      drawn = slots[s].resize_result.img;
            tm6a.add(fpipe::now_ms() - ta);

            ta = fpipe::now_ms();
            if (stream) streamer->send(drawn);
            tm6b.add(fpipe::now_ms() - ta);

            tm6.add(fpipe::now_ms() - t0);
            ++n_out;
            q_free.push(s);
        }
    });

    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join();

    const double wall = (fpipe::now_ms() - t_start) / 1000.0;
    cam.close();
    if (streamer) streamer->close();

    const double a[6] = {tm1.avg(), tm2.avg(), tm3.avg(),
                         tm4.avg(), tm5.avg(), tm6.avg()};
    const char* nm[6] = {"T1 Cam", "T2 Resize IP", "T3 前處理",
                         "T4 DPU", "T5 後處理+追蹤", "T6 繪製+串流"};
    int worst = 0;
    for (int i = 1; i < 6; ++i) if (a[i] > a[worst]) worst = i;

    std::cout << "\n──────── 統計 ────────\n"
              << "輸出 " << n_out.load() << " 幀,丟棄 " << n_drop.load() << "\n\n";
    for (int i = 0; i < 6; ++i)
        std::cout << "  " << nm[i] << "  " << std::fixed << std::setprecision(2)
                  << a[i] << " ms" << (i == worst ? "   <= 瓶頸" : "") << "\n";
    std::cout << "\n  T5 細分: 轉置 " << tm5a.avg()
              << " / 反量化 " << tm5b.avg() << " / NMS+追蹤 " << tm5c.avg() << " ms\n"
              << "  T6 細分: 繪製 " << tm6a.avg()
              << " / 送出 " << tm6b.avg() << " ms\n";

    // T6 只取最新幀,不會反壓,所以它不列入瓶頸計算
    double worst_core = 0; int wi = 0;
    for (int i = 0; i < 5; ++i) if (a[i] > worst_core) { worst_core = a[i]; wi = i; }

    std::cout << "\n串列的話需要 "
              << (a[0] + a[1] + a[2] + a[3] + a[4] + a[5]) << " ms/幀\n"
              << "流水線後受限於 " << nm[wi] << " 的 " << worst_core
              << " ms/幀,理論上限 " << (1000.0 / worst_core) << " FPS\n"
              << "(T6 " << a[5] << " ms 只取最新幀,不反壓;未送出 "
              << n_stale.load() << " 幀)\n"
              << "實測 " << (n_out.load() / wall) << " FPS\n";
}


int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) { print_usage(argv[0]); return -1; }

    stream_params sp{args.st_ip, args.st_port, args.st_width,
                     args.st_height, args.st_fps, args.st_quality};
    Camera::Config cc{args.cam_index, args.cam_width, args.cam_height,
                      args.cam_fps, args.cam_fourcc};

    run_camera(args.model_path, cc, args.conf, args.iou, "",
               args.draw, args.stream, sp);
    return 0;
}