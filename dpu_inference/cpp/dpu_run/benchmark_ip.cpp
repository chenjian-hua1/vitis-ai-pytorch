// ============================================================================
//  benchmark_camera.cpp — 逐階段效能量測
//
//  ./benchmark ./model/yolo11n_int.xmodel
//  ./benchmark ./model/yolo11n_int.xmodel -n 300 --cam-fps 30 --no-track
//  ./benchmark ./model/yolo11n_int.xmodel --stream --ip=192.168.1.50 --port=5000
//  BENCH_MODE=grabber ./benchmark ...
//
//  流程:
//    Camera → (色彩轉換) → LetterBox(resize IP) → BGR2RGB → Norm+Fix
//           → DPU 硬體 → DPU 輸出整理 → Fix2Float → DFL → NMS
//           → Track → Draw → Stream
//
//  相對於舊版的改動:
//
//  1. DPU 拆成「硬體」與「輸出整理」兩段。
//     舊版 engine.run() 把 submit / wait / memcpy / NHWC→NCHW 轉置全包在
//     一起,量出來的數字上升時分不出是硬體變慢還是 CPU 被搶。硬體時間
//     與核心數無關,CPU 那半才是能優化、也會被競爭影響的部分。
//
//  2. LetterBox 走 resize IP(可用時),並回報是否 zero-copy。
//     IP 不可用會自動退回 CPU 版,結果相同,只是慢。
//
//  3. 色彩轉換獨立成一段。
//     相機給 UYVY 時,轉成 BGR 是一筆固定成本;它到底算在擷取還是處理,
//     會大幅改變兩端的平衡,所以要單獨看得到。
//
//  4. 加上端到端延遲統計。
//     吞吐取「最長的一段」,延遲是「一路相加」,兩者要分開看。
//
//  擷取模式(環境變數 BENCH_MODE):
//    direct  (預設) 同步呼叫,Capture 欄位 = 擷取+解碼的真實耗時
//    grabber        背景執行緒,Capture 欄位 = 等待新幀的時間
// ============================================================================

#include "cli_args.h"
#include "modelrunner_pipe.h"
#include "tracker.h"
#include "stream.h"
#include "drawer.h"
#include "yolopproc.h"
#include "camera.h"
#include "preproc.h"
#include "hls_resize.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

static volatile bool g_running = true;
static void signalHandler(int) { g_running = false; }

using Clock = std::chrono::high_resolution_clock;

static double time_now() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}

static double steady_sec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}


// ─────────────────────────────────────────────────────────────────────────────
//  分段計時器
// ─────────────────────────────────────────────────────────────────────────────
struct StageTimer {
    std::string         name;
    std::vector<double> samples;   // ms
    bool                is_hw = false;   // 硬體時間:不受 CPU 競爭影響

    StageTimer() {}
    StageTimer(const std::string& n, bool hw = false) : name(n), is_hw(hw) {}

    void   add(double ms) { samples.push_back(ms); }
    bool   empty() const  { return samples.empty(); }
    size_t n() const      { return samples.size(); }

    double sum() const {
        double s = 0.0;
        for (double v : samples) s += v;
        return s;
    }
    double avg() const { return samples.empty() ? 0.0 : sum() / samples.size(); }
    double mn()  const { return samples.empty() ? 0.0
                          : *std::min_element(samples.begin(), samples.end()); }
    double mx()  const { return samples.empty() ? 0.0
                          : *std::max_element(samples.begin(), samples.end()); }
    double pct(double q) const {
        if (samples.empty()) return 0.0;
        std::vector<double> t = samples;
        size_t k = static_cast<size_t>(q * (t.size() - 1) + 0.5);
        std::nth_element(t.begin(), t.begin() + k, t.end());
        return t[k];
    }
    double stddev() const {
        if (samples.size() < 2) return 0.0;
        const double m = avg();
        double acc = 0.0;
        for (double v : samples) { const double d = v - m; acc += d * d; }
        return std::sqrt(acc / (samples.size() - 1));
    }
};

static void print_row(const StageTimer& s, double total_for_pct) {
    if (s.empty()) return;
    const double share = (total_for_pct > 0.0) ? (s.avg() / total_for_pct * 100.0) : 0.0;
    std::cout << std::left  << std::setw(22) << s.name
              << std::right << std::fixed   << std::setprecision(3)
              << std::setw(9) << s.avg()
              << std::setw(9) << s.mn()
              << std::setw(9) << s.pct(0.50)
              << std::setw(9) << s.pct(0.95)
              << std::setw(9) << s.mx()
              << std::setw(9) << s.stddev()
              << std::setw(7) << std::setprecision(1) << share << "%"
              << std::setw(5) << (s.is_hw ? "HW" : "CPU")
              << "\n";
}

static void print_header(const std::string& title) {
    std::cout << "\n" << title << "\n"
              << std::left  << std::setw(22) << "Stage"
              << std::right << std::setw(9)  << "avg"
              << std::setw(9) << "min"
              << std::setw(9) << "p50"
              << std::setw(9) << "p95"
              << std::setw(9) << "max"
              << std::setw(9) << "std"
              << std::setw(8) << "share"
              << std::setw(5) << "型別"
              << "\n" << std::string(90, '-') << "\n";
}


// ─────────────────────────────────────────────────────────────────────────────
//  hls::resize::Result → 專案的 ResizeResult
//  兩個多載讓 ratio/pad 是 std::pair 或 .x/.y 型別都成立
// ─────────────────────────────────────────────────────────────────────────────
template <class T>
static void set_xy(std::pair<T, T>& d, float x, float y) {
    d.first = static_cast<T>(x); d.second = static_cast<T>(y);
}
template <class P>
static auto set_xy(P& d, float x, float y) -> decltype(d.x, d.y, void()) {
    d.x = decltype(d.x)(x); d.y = decltype(d.y)(y);
}
static void to_project_result(const hls::resize::Result& s, ResizeResult& d) {
    d.img = s.img;  d.content = s.content;
    set_xy(d.ratio, s.ratio.x, s.ratio.y);
    set_xy(d.pad,   s.pad.x,   s.pad.y);
}


// ─────────────────────────────────────────────────────────────────────────────
//  主體
// ─────────────────────────────────────────────────────────────────────────────
void benchmark_camera(std::string xmodel_path,
                      Camera::Config cam_conf,
                      int    warmup   = 30,
                      int    iter     = 300,
                      double conf_th  = 0.2,
                      double iou_th   = 0.45,
                      bool   do_track = true,
                      bool   draw     = false,
                      bool   stream   = false,
                      stream_params stream_param = {},
                      std::string save_last = "benchmark_last_frame.jpg")
{
    std::signal(SIGINT, signalHandler);

    const char* mode_env = std::getenv("BENCH_MODE");
    const bool use_grabber = mode_env && std::strcmp(mode_env, "grabber") == 0;

    std::cout << "===== Camera → DPU 全流程分段量測 (warmup " << warmup
              << " 幀, 統計 " << iter << " 幀) =====\n"
              << "擷取模式: "
              << (use_grabber ? "FrameGrabber 背景執行緒(Capture = 等待新幀)"
                              : "同步呼叫(Capture = 擷取+解碼真實耗時)")
              << "\n";

    // ---- 1. 模型 ----
    XmodelPipelineEngine engine(xmodel_path, 1);   // 單執行緒量測,一個 context

    const int in_w = engine.in_w();
    const int in_h = engine.in_h();
    const int ch = 16;
    const int no = engine.output_mat_nchw(0, 0).size[1];
    const int nc = no - 4 * ch;
    if (nc <= 0) {
        std::cerr << "模型輸出 channel 數 (" << no
                  << ") 與 YOLO DFL 頭假設不符 (ch=" << ch << ")\n";
        return;
    }
    std::cout << "模型輸入: " << in_w << "x" << in_h
              << "  輸出數量: " << engine.num_outputs()
              << "  nc = " << nc << "\n";

    YOLOPostProcessor yolo_pp(1, in_h, in_w, nc, ch);

    const int in_fix = static_cast<int>(std::round(std::log2(engine.input_scale())));
    std::vector<int> out_fix(engine.num_outputs());
    for (size_t i = 0; i < engine.num_outputs(); ++i)
        out_fix[i] = static_cast<int>(std::round(-std::log2(engine.output_scale(i))));

    ResizeResult resize_result;
    hls::resize::Result hres;
    cv::Mat rgb_buf, drawn;
    std::vector<cv::Mat> float_outputs(engine.num_outputs());
    const std::vector<DetectionBatch>* nms_result = nullptr;
    const cv::Size draw_size(in_w, in_h);

    // ---- 2. 相機 ----
    // raw 模式:色彩轉換不留在 read() 裡,才能單獨量到它的成本
    cam_conf.raw_output = true;
    if (cam_conf.buffer_count <= 0) cam_conf.buffer_count = 4;

    Camera cam(cam_conf);
    if (!cam.open()) { std::cerr << "錯誤:無法開啟攝影機\n"; return; }

    const int cam_w = cam.actualWidth();
    const int cam_h = cam.actualHeight();
    const int cvt_code = cam.conversionCode();     // -1 = 相機已給 BGR
    std::cout << "[Camera] 色彩轉換: "
              << (cvt_code >= 0 ? "由本程式做(可單獨計時)"
                                : "OpenCV 在 read() 內部做(併入 Capture)")
              << "\n";

    // ---- 3. resize IP ----
    const size_t need = static_cast<size_t>(cam_w) * cam_h * 3
                      + static_cast<size_t>(in_w) * in_w * 3 * 2
                      + 8u * 1024 * 1024;
    hls::use_dma_heap("auto", ((need >> 20) + 16) << 20);
    hls::resize::use_devmem();
    bool ip_ready = hls::resize::available();
    std::cout << "[resize IP] "
              << (ip_ready ? hls::resize::device_info()
                           : "不可用:" + hls::resize::last_error()) << "\n";

    // 色彩轉換的目的地。放在 DMA 記憶體,letterbox 才能 zero-copy 讀它。
    cv::Mat bgr_full;
    if (ip_ready) {
        bgr_full = hls::resize::input_buffer(cam_w, cam_h, 0);
        if (bgr_full.empty()) ip_ready = false;
    }
    if (bgr_full.empty()) bgr_full.create(cam_h, cam_w, CV_8UC3);
    const uint8_t* bgr_origin = bgr_full.data;

    // ---- 4. 背景擷取(選用)----
    FrameGrabber grabber(cam);
    if (use_grabber) grabber.start();

    // ---- 5. 串流。尺寸對齊 letterbox 影像,避免每幀多一次縮放 ----
    RtpJpegStreamer* streamer = nullptr;
    if (stream) {
        if (stream_param.width != in_w || stream_param.height != in_h)
            std::cout << "[Stream] 尺寸由 " << stream_param.width << "x"
                      << stream_param.height << " 覆寫為 " << in_w << "x" << in_h
                      << "(直接送 letterbox 影像,不做額外縮放)\n";
        stream_param.width = in_w;
        stream_param.height = in_h;

        streamer = new RtpJpegStreamer(stream_param.width, stream_param.height,
                                       stream_param.fps, stream_param.ip,
                                       stream_param.port, stream_param.quality);
        if (!streamer->isOpened()) {
            std::cerr << "錯誤:無法開啟 GStreamer 發送管線\n";
            delete streamer; streamer = nullptr; stream = false;
        }
    }

    // ---- 6. 追蹤 ----
    bytetrack::Params p;
    p.max_lost_seconds = 2.0;
    p.class_aware = true;
    bytetrack::BYTETracker tracker(p);
    std::vector<bytetrack::Box> boxes;
    const std::vector<bytetrack::Track>* tracks = nullptr;

    // ---- 7. 計時器。is_hw 標記哪些是硬體時間 ----
    StageTimer s_cap  ("Capture");
    StageTimer s_cvt  ("ColorCvt (UYVY→BGR)");
    StageTimer s_rsz  ("LetterBox", true);       // 主要在等 resize IP
    StageTimer s_b2r  ("BGR2RGB");
    StageTimer s_quant("Norm+Fix (quant)");
    StageTimer s_dpu  ("DPU 硬體", true);        // submit + wait_hw
    StageTimer s_fin  ("DPU 輸出整理");          // memcpy + NHWC→NCHW
    StageTimer s_deq  ("Fix2Float (dequant)");
    StageTimer s_dfl  ("DFL Decode");
    StageTimer s_nms  ("NMS");
    StageTimer s_trk  ("Track");
    StageTimer s_draw ("Draw");
    StageTimer s_send ("Stream Send");
    StageTimer s_loop ("[Loop total]");
    StageTimer s_lat  ("[端到端延遲]");

    long long det_sum = 0, trk_sum = 0;
    int  done = 0;
    bool aborted = false;
    bool zero_copy_ok = ip_ready;

    // 取一幀。回傳影像的參照與擷取時間戳。
    cv::Mat sync_frame;
    FrameGrabber::Handle handle;
    auto acquire = [&](const cv::Mat*& out, double& t_cap, double& ms) -> bool {
        const double t0 = time_now();
        if (use_grabber) {
            handle = grabber.acquire(500);
            if (!handle.valid()) { ms = 0; return false; }
            out   = &handle.mat();
            t_cap = handle.timestamp();
        } else {
            // nextFrameLatest:順便清掉 V4L2 佇列裡的積壓,
            // 量到的才是「當下最新一幀」的成本,不含排隊時間
            if (!cam.nextFrameLatest(sync_frame) || sync_frame.empty()) {
                ms = 0; return false;
            }
            out   = &sync_frame;
            t_cap = steady_sec();
        }
        ms = time_now() - t0;
        return true;
    };

    // 一次完整的處理。measure = false 時不計時(暖機用)。
    auto process_one = [&](const cv::Mat& raw, double t_cap, bool measure) {
        double t0;

        // 色彩轉換
        const cv::Mat* src = &raw;
        if (cvt_code >= 0) {
            t0 = time_now();
            cv::cvtColor(raw, bgr_full, cvt_code);
            if (bgr_full.data != bgr_origin) {      // 被重新配置 → DMA 位址失效
                zero_copy_ok = false;
                bgr_full = hls::resize::input_buffer(cam_w, cam_h, 0);
            }
            if (measure) s_cvt.add(time_now() - t0);
            src = &bgr_full;
        }

        // LetterBox
        t0 = time_now();
        if (ip_ready) {
            hls::resize::letterbox(*src, in_w, hres);
            to_project_result(hres, resize_result);
        } else {
            resize(*src, in_w, resize_result);
        }
        if (measure) s_rsz.add(time_now() - t0);

        // BGR2RGB
        t0 = time_now();
        cv::cvtColor(resize_result.img, rgb_buf, cv::COLOR_BGR2RGB);
        if (measure) s_b2r.add(time_now() - t0);

        // Normalize + 量化,直接寫進 DPU 的輸入張量
        t0 = time_now();
        cv::Mat dpu_in = engine.input_mat(0);
        norm_and_fix(rgb_buf, in_fix, dpu_in);
        if (measure) s_quant.add(time_now() - t0);

        // DPU 硬體
        t0 = time_now();
        engine.submit(0);
        engine.wait_hw(0);
        if (measure) s_dpu.add(time_now() - t0);

        // DPU 輸出整理(CPU:memcpy + NHWC→NCHW 轉置)
        t0 = time_now();
        engine.finish(0);
        if (measure) s_fin.add(time_now() - t0);

        // 反量化
        t0 = time_now();
        for (size_t o = 0; o < engine.num_outputs(); ++o)
            fix2float(engine.output_mat_nchw(0, o), out_fix[o], float_outputs[o]);
        if (measure) s_deq.add(time_now() - t0);

        // DFL
        t0 = time_now();
        yolo_pp.decode(float_outputs, conf_th);
        if (measure) s_dfl.add(time_now() - t0);

        // NMS
        t0 = time_now();
        yolo_pp.nms(conf_th, iou_th);
        nms_result = &yolo_pp.detections();
        if (measure) s_nms.add(time_now() - t0);

        if (measure) det_sum += (*nms_result)[0].count;

        // 追蹤。座標已在 letterbox 空間,map 是 identity。
        if (do_track) {
            t0 = time_now();
            map_detections((*nms_result)[0], boxes, 0.f, 0.f, 1.f, 1.f, draw_size);
            tracks = &tracker.update(boxes, t_cap);
            if (measure) { s_trk.add(time_now() - t0);
                           trk_sum += static_cast<long long>(tracks->size()); }
        }

        // 繪製
        if (draw) {
            t0 = time_now();
            if (do_track && tracks) {
                draw_tracking(resize_result.img, drawn, *tracks, 0.0);
            } else {
                ResizeResult identity;
                identity.img = resize_result.img;
                set_xy(identity.ratio, 1.f, 1.f);
                set_xy(identity.pad,   0.f, 0.f);
                draw_detection(resize_result.img, drawn, (*nms_result)[0], identity, 0.0);
            }
            if (measure) s_draw.add(time_now() - t0);
        } else {
            drawn = resize_result.img;      // 淺拷貝
        }

        // 串流
        if (stream) {
            t0 = time_now();
            streamer->send(drawn);
            if (measure) s_send.add(time_now() - t0);
        }
    };

    // ---- 8. 暖機 ----
    std::cout << "\nwarmup ...\n";
    for (int i = 0; i < warmup && g_running; ++i) {
        const cv::Mat* raw = nullptr;
        double t_cap = 0, ms = 0;
        if (!acquire(raw, t_cap, ms)) { aborted = true; break; }
        process_one(*raw, t_cap, false);
        handle = FrameGrabber::Handle();     // 儘早歸還緩衝
    }
    if (nms_result)
        std::cout << "warmup 後偵測到 " << (*nms_result)[0].count << " 個框\n";

    // ---- 9. 主迴圈 ----
    std::cout << "開始統計(Ctrl+C 可提早結束)...\n";
    const double t_bench0 = time_now();

    for (int i = 0; i < iter && g_running && !aborted; ++i) {
        const double t_loop0 = time_now();

        const cv::Mat* raw = nullptr;
        double t_cap = 0, ms = 0;
        if (!acquire(raw, t_cap, ms)) { aborted = true; break; }
        s_cap.add(ms);

        process_one(*raw, t_cap, true);
        handle = FrameGrabber::Handle();

        s_loop.add(time_now() - t_loop0);
        s_lat.add((steady_sec() - t_cap) * 1000.0);
        ++done;

        if (done % 30 == 0)
            std::cout << "\r  進度 " << done << "/" << iter << std::flush;
    }
    const double t_bench_total = time_now() - t_bench0;
    std::cout << "\r  進度 " << done << "/" << iter << "\n";

    // ---- 10. 收尾 ----
    if (use_grabber) grabber.stop();
    cam.close();
    if (streamer) { streamer->close(); delete streamer; streamer = nullptr; }

    if (done == 0) { std::cout << "沒有取得任何有效幀,無法統計。\n"; return; }

    // ---- 11. 報表 ----
    const StageTimer* all[] = {&s_cvt, &s_rsz, &s_b2r, &s_quant, &s_dpu, &s_fin,
                               &s_deq, &s_dfl, &s_nms, &s_trk, &s_draw, &s_send};
    double compute = 0, hw_ms = 0, cpu_ms = 0;
    for (const StageTimer* s : all) {
        compute += s->avg();
        (s->is_hw ? hw_ms : cpu_ms) += s->avg();
    }
    const double loop_ms = s_loop.avg();

    print_header("===== 分段耗時(ms;share 以「計算總計」為分母,Capture 以 Loop 為分母)=====");
    print_row(s_cap, loop_ms);
    std::cout << std::string(90, '-') << "\n";
    for (const StageTimer* s : all) print_row(*s, compute);
    std::cout << std::string(90, '-') << "\n";
    print_row(s_loop, loop_ms);

    std::cout << std::fixed << std::setprecision(3)
              << "\n===== 分組彙總 =====\n"
              << "Capture                     : " << s_cap.avg() << " ms\n"
              << "前處理 (cvt+letterbox+rgb+quant): "
              << (s_cvt.avg() + s_rsz.avg() + s_b2r.avg() + s_quant.avg()) << " ms\n"
              << "DPU 硬體                    : " << s_dpu.avg() << " ms\n"
              << "DPU 輸出整理 (CPU)          : " << s_fin.avg() << " ms\n"
              << "後處理 (dequant+DFL+NMS)    : "
              << (s_deq.avg() + s_dfl.avg() + s_nms.avg()) << " ms\n"
              << "其他 (track+draw+stream)    : "
              << (s_trk.avg() + s_draw.avg() + s_send.avg()) << " ms\n"
              << "計算總計 (不含擷取)         : " << compute << " ms  →  "
              << std::setprecision(2) << (compute > 0 ? 1000.0 / compute : 0.0)
              << " FPS (理論上限)\n";

    std::cout << std::setprecision(3)
              << "\n  其中硬體等待              : " << hw_ms
              << " ms  (DPU + resize IP,與核心數無關,優化 CPU 動不了它)\n"
              << "  其中 CPU 工作             : " << cpu_ms
              << " ms  (可平行化,也是與其他執行緒競爭的部分)\n";

    std::cout << std::setprecision(3)
              << "\n迴圈總計 (含擷取)           : " << loop_ms << " ms  →  "
              << std::setprecision(2) << (loop_ms > 0 ? 1000.0 / loop_ms : 0.0)
              << " FPS\n"
              << std::setprecision(3)
              << "端到端延遲 avg/p95/max      : " << s_lat.avg() << " / "
              << s_lat.pct(0.95) << " / " << s_lat.mx() << " ms\n"
              << std::setprecision(2)
              << "實測平均 FPS (wall clock)   : "
              << (t_bench_total > 0 ? done * 1000.0 / t_bench_total : 0.0)
              << "  (" << done << " 幀 / " << std::setprecision(1)
              << t_bench_total / 1000.0 << " s)\n";

    std::cout << std::setprecision(2)
              << "平均偵測框數 / 幀           : "
              << static_cast<double>(det_sum) / done << "\n";
    if (do_track)
        std::cout << "平均 track 數 / 幀          : "
                  << static_cast<double>(trk_sum) / done << "\n";

    if (ip_ready)
        std::cout << "resize IP                   : 已使用"
                  << (zero_copy_ok ? "(zero-copy)" : "(有額外複製)") << "\n";
    if (use_grabber)
        std::cout << "背景擷取                    : 推掉 V4L2 舊幀 "
                  << grabber.staleSkipped() << ",處理端來不及略過 "
                  << grabber.overwritten() << "\n";
    if (aborted)
        std::cout << "(測試被中斷,統計基於已完成的 " << done << " 幀)\n";

    // ---- 12. 存最後一幀 ----
    if (!save_last.empty() && nms_result && !resize_result.img.empty()) {
        if (!draw) {
            if (do_track && tracks) {
                draw_tracking(resize_result.img, drawn, *tracks, 0.0);
            } else {
                ResizeResult identity;
                identity.img = resize_result.img;
                set_xy(identity.ratio, 1.f, 1.f);
                set_xy(identity.pad,   0.f, 0.f);
                draw_detection(resize_result.img, drawn, (*nms_result)[0], identity, 0.0);
            }
        }
        if (!drawn.empty() && cv::imwrite(save_last, drawn))
            std::cout << "最後一幀結果已存至: " << save_last << "\n";
        else
            std::cerr << "imwrite 失敗: " << save_last << "\n";
    }
}


int main(int argc, char** argv) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) { print_usage(argv[0]); return -1; }
    print_args(args);

    stream_params sp{args.st_ip, args.st_port, args.st_width,
                     args.st_height, args.st_fps, args.st_quality};
    Camera::Config cc{args.cam_index, args.cam_width,
                      args.cam_height, args.cam_fps, args.cam_fourcc};

    benchmark_camera(args.model_path, cc, args.warmup, args.iter,
                     args.conf, args.iou, args.track, args.draw,
                     args.stream, sp, args.save);
    return 0;
}