// yolo_inference_lowlat.cpp — 兩條執行緒的低延遲版本(UYVY 直送 IP)
//
// ─────────────────────────────────────────────────────────────
//  與前一版的差異
//
//  前一版:相機 UYVY → [擷取執行緒 cv::cvtColor 轉 BGR 全幀]
//          → resize IP letterbox → cvtColor BGR2RGB → DPU
//
//  這一版:相機 UYVY(原樣,不做任何 CPU 處理)直接落在 DMA 緩衝
//          → hls::uyvy_resize::letterbox
//              (IP:UYVY→RGB + 2/3 倍縮小;CPU:零頭縮放 / 貼進黑邊)
//          → DPU(已經是 RGB,不用再轉)
//
//  1080p → 640 走 3 倍、720p → 640 走 2 倍,都是 IP 一次到位,
//  CPU 只剩把 640x360 貼進 640x640 的一次 copy。不再經過 resize IP。
//
//  省掉的 CPU 工作:
//    - 全幀 YUV→BGR 轉換(1080p 約數 ms,原本佔著擷取執行緒)
//    - 前處理的 BGR2RGB(IP 輸出本來就是 RGB)
//  多出的 CPU 工作:
//    - 只有要繪製 / 串流時,把 letterbox 後的小圖 RGB2BGR 一次
//
//  UYVY 緩衝由 input_buffer 配置在 DMA 記憶體,IP 讀取是 zero-copy。
//
//  其餘架構不變:FrameGrabber 一條擷取執行緒 + 主執行緒串列處理,
//  in-flight 固定 2 幀。
//
//  環境變數:
//    PIPE_CAM_CORE     擷取執行緒綁核
//    PIPE_V4L2_BUFS    V4L2 buffer 數
//    PIPE_UYVY_PHYS    uyvy_resize 控制暫存器實體位址(十六進位),
//                      設了就走 /dev/mem;不設就用 UIO 名稱 "uyvy_resize"
// ─────────────────────────────────────────────────────────────

#include "modelrunner_pipe.h"
#include "tracker.h"
#include "stream.h"
#include "drawer.h"
#include "yolopproc.h"
#include "camera.h"
#include "preproc.h"
#include "cli_args.h"
#include "frame_pipeline.h"
#include "hls_uyvy_resize.h"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static std::atomic<bool> g_running{true};
static void signalHandler(int) { g_running = false; }

namespace {

inline void set_xy(std::pair<float, float>& d, float x, float y) {
    d.first = x; d.second = y;
}
template <class P>
inline auto set_xy(P& d, float x, float y) -> decltype(d.x, d.y, void()) {
    d.x = decltype(d.x)(x);
    d.y = decltype(d.y)(y);
}

inline void to_project_result(const hls::uyvy_resize::LetterboxResult& s,
                              ResizeResult& d) {
    d.img     = s.img;
    d.content = s.content;
    set_xy(d.ratio, s.ratio.x, s.ratio.y);
    set_xy(d.pad,   s.pad.x,   s.pad.y);
}

// OpenCV 在 CONVERT_RGB=0 時,有些版本回傳 1xN 的 CV_8UC1 而不是 HxW 的
// CV_8UC2。reshape 不動資料指標,所以仍然是 zero-copy。
cv::Mat as_uyvy(const cv::Mat& raw, int w, int h) {
    if (raw.type() == CV_8UC2 && raw.cols == w && raw.rows == h) return raw;
    if (raw.isContinuous() &&
        raw.total() * raw.elemSize() == static_cast<size_t>(w) * h * 2)
        return raw.reshape(2, h);
    return cv::Mat();
}

}  // namespace


void run_camera(std::string xmodel_path, Camera::Config cam_conf,
                double conf_th = 0.1, double iou_th = 0.45,
                std::string out_file = "", bool draw = true,
                bool stream = true, stream_params stream_param = {},
                int cam_core = -1)
{
    (void)out_file;
    std::signal(SIGINT, signalHandler);
    if (const char* e = std::getenv("PIPE_CAM_CORE")) cam_core = std::atoi(e);

    // ---- 模型 ----
    XmodelPipelineEngine engine(xmodel_path, 1);
    const int in_w = engine.in_w();
    const int in_h = engine.in_h();

    const int ch = 16;
    const int no = engine.output_mat_nchw(0, 0).size[1];
    const int nc = no - 4 * ch;
    if (nc <= 0) { std::cerr << "輸出 channel 數與 DFL 假設不符\n"; return; }
    std::cout << "模型 " << in_w << "x" << in_h << "  nc=" << nc << "\n";

    YOLOPostProcessor yolo_pp(1, in_h, in_w, nc, ch);
    const int in_fix = static_cast<int>(std::round(std::log2(engine.input_scale())));
    std::vector<int> out_fix(engine.num_outputs());
    for (size_t i = 0; i < engine.num_outputs(); ++i)
        out_fix[i] = static_cast<int>(std::round(-std::log2(engine.output_scale(i))));

    // ---- 相機:強制 UYVY + raw,影格原封不動交出來 ----
    if (cam_conf.fourcc != "UYVY") {
        std::cout << "[Camera] fourcc " << cam_conf.fourcc
                  << " 改為 UYVY(uyvy_resize IP 只吃 UYVY)\n";
        cam_conf.fourcc = "UYVY";
    }
    cam_conf.raw_output = true;
    if (cam_conf.buffer_count <= 0) cam_conf.buffer_count = 3;
    if (const char* e = std::getenv("PIPE_V4L2_BUFS"))
        cam_conf.buffer_count = std::max(2, std::atoi(e));

    Camera cam(cam_conf);
    if (!cam.open()) return;
    const int cam_w = cam.actualWidth();
    const int cam_h = cam.actualHeight();

    if (cam.actualFourcc() != "UYVY" || !cam.rawMode()) {
        std::cerr << "[Camera] 拿不到 raw UYVY(實際 " << cam.actualFourcc()
                  << (cam.rawMode() ? "" : ",OpenCV 仍在轉換")
                  << "),此程式需要 UYVY 相機\n";
        return;
    }

    // ---- letterbox 規劃:跟每一幀用的是同一個函式,開機時先看一次 ----
    const float r0 = std::min(1.0f, std::min(float(in_w) / cam_h, float(in_w) / cam_w));
    const hls::uyvy_resize::Plan plan = hls::uyvy_resize::plan_for(
        cam_w, cam_h,
        static_cast<uint32_t>(std::round(cam_w * r0)),
        static_cast<uint32_t>(std::round(cam_h * r0)));
    const int sc = plan.use_ip ? static_cast<int>(plan.scale) : 1;

    // ---- DMA pool ----
    const size_t need =
          static_cast<size_t>(cam_w) * cam_h * 2 * 3            // 3 塊 UYVY 擷取緩衝
        + static_cast<size_t>(cam_w) * cam_h * 2                // staging 退路(非 zero-copy 時)
        + static_cast<size_t>(cam_w / sc) * (cam_h / sc) * 3    // IP 輸出
        + 8u * 1024 * 1024;
    hls::use_dma_heap("auto", ((need >> 20) + 16) << 20);

    // ---- uyvy_resize IP ----
    if (const char* e = std::getenv("PIPE_UYVY_PHYS"))
        hls::uyvy_resize::use_devmem(std::strtoull(e, nullptr, 16));

    std::cout << "[letterbox] " << cam_w << "x" << cam_h << " -> " << in_w
              << "x" << in_w << ":" << plan.reason;
    if (plan.use_ip)
        std::cout << "(" << plan.scale << " 倍 -> " << plan.params.out_w
                  << "x" << plan.params.out_h << ")";
    std::cout << "\n";

    // IP 是否啟用:開機檢查沒過、或執行中出錯,就關掉,之後每幀直接走 CPU
    bool ip_enabled = plan.use_ip;
    if (plan.use_ip) {
        if (!hls::uyvy_resize::available()) {
            std::cerr << "[uyvy_resize] 不可用:" << hls::uyvy_resize::last_error()
                      << "\n             裝置:" << hls::uyvy_resize::device_info()
                      << "\n             letterbox 改用 CPU\n";
            ip_enabled = false;
        } else {
            // 開機先驗一次:硬體結果與 CPU 模型逐 byte 比對
            const auto rep = hls::uyvy_resize::verify(
                hls::uyvy_resize::make_test_uyvy(cam_w, cam_h),
                static_cast<int>(plan.scale));
            std::cout << "[uyvy_resize] " << hls::uyvy_resize::device_info()
                      << "  verify: " << rep.reason << "\n";
            if (!rep.passed()) {
                std::cerr << "[uyvy_resize] verify 未通過";
                if (rep.ran)
                    std::cerr << ":不符 " << rep.mismatches << "/" << rep.total
                              << "  R/G/B 錯 " << rep.ch_err[0] << "/" << rep.ch_err[1]
                              << "/" << rep.ch_err[2] << "  最大差 " << rep.max_diff
                              << "  補位非 0 " << rep.pad_dirty;
                std::cerr << "\n             letterbox 改用 CPU\n";
                ip_enabled = false;
            }
        }
    }

    // 三塊擷取緩衝放在 DMA 記憶體 —— 相機影格落在這裡,
    // uyvy_resize IP 直接讀,不必再搬一次。
    std::vector<cv::Mat> cap_bufs(3);
    for (int i = 0; i < 3; ++i) {
        cap_bufs[i] = hls::input_buffer(cam_w, cam_h, i, CV_8UC2);
        if (cap_bufs[i].empty()) cap_bufs[i].create(cam_h, cam_w, CV_8UC2);
    }

    // ---- 串流 ----
    std::unique_ptr<RtpJpegStreamer> streamer;
    if (stream) {
        streamer = std::make_unique<RtpJpegStreamer>(
            stream_param.width, stream_param.height, stream_param.fps,
            stream_param.ip, stream_param.port, stream_param.quality);
        if (!streamer->isOpened()) { std::cerr << "GStreamer 開啟失敗\n"; return; }
    }

    bytetrack::Params tp;
    tp.max_lost_seconds = 2.;
    tp.class_aware = true;
    bytetrack::BYTETracker tracker(tp);

    // tm_rsz 是整個 letterbox,tm_ip 是其中 IP 那段(含 sync)
    fpipe::StageTimer tm_ip, tm_rsz, tm_pre, tm_dpu_hw, tm_dpu_cpu,
                      tm_post, tm_out, tm_lat;
    std::atomic<long long> n_proc{0};
    std::atomic<double> cur_fps{0.0};
    const double t_start = fpipe::now_ms();

    // ===================== 擷取:FrameGrabber,不設 transform =====================
    // 沒有 setTransform → 擷取執行緒只做 cap.read(),影格原樣寫進 DMA 緩衝。
    FrameGrabber grabber(cam);
    if (!grabber.setBuffers(cap_bufs))
        std::cerr << "[FrameGrabber] setBuffers 失敗,改用內部配置的緩衝\n";
    std::cout << "[排程] 擷取執行緒不做轉換,UYVY 直送 "
              << "hls::uyvy_resize::letterbox\n";

    grabber.start();
    if (cam_core >= 0) {
        const bool ok = grabber.pinThread(cam_core);
        std::cout << "[排程] 擷取執行緒綁 core " << cam_core
                  << (ok ? " 成功" : " 失敗") << std::endl;
    }

    // ===================== 主執行緒 =====================
    {
        ResizeResult rr;
        hls::uyvy_resize::LetterboxResult lb;     // 迴圈外重用,lb.img 不會每幀重新配置
        cv::Mat bgr_out, drawn;
        std::vector<cv::Mat> float_outputs(engine.num_outputs());
        std::vector<bytetrack::Box> boxes;
        long long frames = 0, prev = 0;
        double t_prev = fpipe::now_ms();
        fpipe::Ema fps_ema(0.3);
        bool first = true;
        bool warned_copy = false, warned_shape = false;

        while (g_running) {
            FrameGrabber::Handle f = grabber.acquire(200);
            if (!f.valid()) continue;
            const double t_cap_stamp = f.timestamp();

            const cv::Mat uyvy = as_uyvy(f.mat(), cam_w, cam_h);
            if (uyvy.empty()) {
                if (!warned_shape) {
                    warned_shape = true;
                    std::cerr << "[Camera] 影格格式不符預期:" << f.mat().cols << "x"
                              << f.mat().rows << " type=" << f.mat().type() << "\n";
                }
                continue;
            }

            // ---- letterbox:UYVY 進,RGB 正方形出 ----
            double t0 = fpipe::now_ms();
            hls::uyvy_resize::letterbox(uyvy, in_w, lb, ip_enabled);
            to_project_result(lb, rr);
            tm_rsz.add(fpipe::now_ms() - t0);

            // UYVY 已經讀完(lb.img 是另一塊記憶體),立刻歸還緩衝給擷取端
            f = FrameGrabber::Handle();

            if (lb.used_ip) {
                tm_ip.add(lb.ip_ms);
                if (!lb.zero_copy && !warned_copy) {
                    warned_copy = true;
                    std::cerr << "[uyvy_resize] 輸入不在 DMA 記憶體,每幀多一次複製 "
                              << std::fixed << std::setprecision(2) << lb.timing.copy_ms
                              << " ms(OpenCV 可能重新配置了擷取緩衝)\n";
                }
            } else if (ip_enabled) {
                // 這一幀 IP 失敗:印出真正的原因,之後不再重試
                // (逾時的話每幀都會先卡 timeout 才退回 CPU)
                ip_enabled = false;
                std::cerr << "[uyvy_resize] 執行失敗,之後改用 CPU\n"
                          << "             原因:" << hls::uyvy_resize::last_error() << "\n"
                          << "             裝置:" << hls::uyvy_resize::device_info() << "\n"
                          << "             花費:" << std::fixed << std::setprecision(1)
                          << lb.ip_ms << " ms\n";
            }

            // ---- 前處理:已經是 RGB,不用再 cvtColor ----
            t0 = fpipe::now_ms();
            cv::Mat dpu_in = engine.input_mat(0);
            norm_and_fix(rr.img, in_fix, dpu_in);
            tm_pre.add(fpipe::now_ms() - t0);

            // ---- DPU ----
            t0 = fpipe::now_ms();
            engine.submit(0);
            engine.wait_hw(0);
            tm_dpu_hw.add(fpipe::now_ms() - t0);

            t0 = fpipe::now_ms();
            engine.finish(0);
            tm_dpu_cpu.add(fpipe::now_ms() - t0);

            // ---- 後處理 + 追蹤 ----
            t0 = fpipe::now_ms();
            for (size_t i = 0; i < engine.num_outputs(); ++i)
                fix2float(engine.output_mat_nchw(0, i), out_fix[i], float_outputs[i]);
            const std::vector<DetectionBatch>& nms =
                yolo_pp.process(float_outputs, conf_th, iou_th);
            map_detections(nms[0], boxes, 0.f, 0.f, 1.f, 1.f, cv::Size(in_w, in_w));
            const std::vector<bytetrack::Track>& tracks =
                tracker.update(boxes, t_cap_stamp);
            tm_post.add(fpipe::now_ms() - t0);

            // ---- 繪製 + 串流:這裡才轉回 BGR(小圖,很便宜)----
            t0 = fpipe::now_ms();
            if (draw || stream || first) {
                cv::cvtColor(rr.img, bgr_out, cv::COLOR_RGB2BGR);
                if (draw) draw_tracking(bgr_out, drawn, tracks, cur_fps.load());
                else      drawn = bgr_out;
                if (stream) streamer->send(drawn);
            }
            tm_out.add(fpipe::now_ms() - t0);

            tm_lat.add(fpipe::now_ms() - t_cap_stamp * 1000.0);
            ++n_proc;

            if (first) { first = false; cv::imwrite("frame0.png", bgr_out); }

            if (++frames % 30 == 0) {
                const double now = fpipe::now_ms();
                const double inst = (frames - prev) * 1000.0 / (now - t_prev);
                cur_fps = fps_ema.push(inst);
                t_prev = now; prev = frames;

                auto w = [](const fpipe::StageTimer& t) {
                    std::ostringstream o;
                    o << std::fixed << std::setprecision(1)
                      << t.recent_avg() << "/" << t.recent_max();
                    return o.str();
                };
                std::cout << "Frame " << frames
                          << "  FPS " << std::fixed << std::setprecision(1) << cur_fps.load()
                          << " | cam " << std::setprecision(1)
                          << grabber.avgCaptureMs() << "/" << grabber.maxCaptureMs()
                          << "  lbox " << w(tm_rsz)
                          << " (ip " << w(tm_ip) << ")"
                          << "  pre " << w(tm_pre)
                          << "  dpu " << w(tm_dpu_hw)
                          << "  轉置 " << w(tm_dpu_cpu)
                          << "  post " << w(tm_post)
                          << "  out " << w(tm_out)
                          << "  | 延遲 " << w(tm_lat)
                          << "  過期 " << grabber.staleSkipped()
                          << "  略過 " << grabber.overwritten() << std::endl;
            }
        }
    }

    g_running = false;
    grabber.stop();

    const double wall = (fpipe::now_ms() - t_start) / 1000.0;
    cam.close();
    if (streamer) streamer->close();

    const double proc = tm_rsz.avg() + tm_pre.avg() + tm_dpu_hw.avg()
                      + tm_dpu_cpu.avg() + tm_post.avg() + tm_out.avg();
    const double hw = tm_dpu_hw.avg() + tm_ip.avg();

    std::cout << "\n──────── 統計 ────────\n"
              << "處理 " << n_proc.load() << " 幀,"
              << "擷取 " << grabber.frameId() << " 幀,"
              << "推掉 V4L2 舊幀 " << grabber.staleSkipped() << ",略過 "
              << grabber.overwritten() << "\n\n"
              << std::fixed << std::setprecision(2)
              << "  擷取(raw UYVY)     " << grabber.avgCaptureMs() << " ms\n"
              << "  letterbox           " << tm_rsz.avg()
              << " ms  (其中 uyvy_resize IP " << tm_ip.avg() << " ms)\n"
              << "  前處理              " << tm_pre.avg()  << " ms\n"
              << "  DPU 硬體            " << tm_dpu_hw.avg() << " ms\n"
              << "  DPU 輸出整理        " << tm_dpu_cpu.avg() << " ms\n"
              << "  後處理 + 追蹤       " << tm_post.avg() << " ms\n"
              << "  繪製 + 串流         " << tm_out.avg()  << " ms\n"
              << "  ── 處理端合計       " << proc << " ms\n"
              << "     其中硬體等待     " << hw << " ms\n"
              << "     其中 CPU 工作    " << (proc - hw) << " ms\n\n"
              << "端到端延遲 平均 " << tm_lat.avg()
              << "  最近 " << tm_lat.recent_avg()
              << "  最大 " << tm_lat.recent_max() << " ms\n"
              << "吞吐 " << (n_proc.load() / wall) << " FPS"
              << "(上限 " << (proc > 0 ? 1000.0 / proc : 0) << ")\n";
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