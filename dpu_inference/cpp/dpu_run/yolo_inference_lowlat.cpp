// yolo_inference_lowlat.cpp — 兩條執行緒的低延遲版本
//
// ─────────────────────────────────────────────────────────────
//  為什麼放棄深度流水線
//
//  流水線讓吞吐等於「最慢的一段」,但延遲等於「同時在管線裡的幀數
//  × 每幀週期」。六段流水線加上各層緩衝,實測有十幾幀在飛,延遲
//  兩三百毫秒 —— 對即時偵測來說,看到的是好幾拍之前的畫面。
//
//  這個版本只額外開一條執行緒,而且它由 camera.h 的 FrameGrabber 管理:
//
//    [FrameGrabber 的擷取執行緒]  清空 V4L2 佇列取最新,寫進三緩衝之一
//    [主執行緒]                   acquire() 取最新的一幀,一路做到送出
//
//  處理全部留在主執行緒,不再另外開 worker。少一次交接、少一層
//  同步,而且 Ctrl-C 之後主迴圈可以直接收尾,不必等別的執行緒。
//
//  in-flight 固定是 2 幀,延遲 ≈ 一個影格間隔 + 一次完整處理。
//
//  代價:所有階段變回串列,FPS = 1000 / 處理總時間。
//        流水線版本的吞吐較高,但那是拿延遲換來的。
//
//  去積壓分兩層,都在 camera.h 裡:
//    驅動層 —— Camera::nextFrameLatest() 推掉 V4L2 佇列中的舊幀
//    應用層 —— FrameGrabber 的三緩衝,前一幀沒被取走就直接覆蓋
//  兩層加起來,主執行緒拿到的永遠是剛拍到的畫面,與處理速度無關。
//
//  FrameGrabber::acquire() 回傳的是緩衝的「參照」,不做複製 ——
//  1080p 每幀省下 6 MB 的搬移。
//
//  OpenCV 的執行緒設定維持預設,沒有自訂的執行緒池 —— 平行化交給
//  OpenCV 與 OS 自己決定。
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
#include "hls_resize.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
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

inline void to_project_result(const hls::resize::Result& s, ResizeResult& d) {
    d.img     = s.img;
    d.content = s.content;
    set_xy(d.ratio, s.ratio.x, s.ratio.y);
    set_xy(d.pad,   s.pad.x,   s.pad.y);
}

}  // namespace


void run_camera(std::string xmodel_path, Camera::Config cam_conf,
                double conf_th = 0.1, double iou_th = 0.45,
                std::string out_file = "", bool draw = true,
                bool stream = true, stream_params stream_param = {},
                int cam_core = -1)   // 預設不保留核心,理由見下
{
    (void)out_file;
    std::signal(SIGINT, signalHandler);
    if (const char* e = std::getenv("PIPE_CAM_CORE")) cam_core = std::atoi(e);

    // OpenCV 的執行緒設定維持預設,不做任何調整。
    // 平行化交給 OpenCV 自己決定;這個版本只靠「擷取獨立一條執行緒」
    // 來避免 V4L2 佇列積壓,其餘全部串列跑在主執行緒。

    // ---- 模型。單一處理執行緒,一個 context 就夠 ----
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

    // ---- 相機。raw 模式讓色彩轉換留給處理端做 ----
    cam_conf.raw_output = true;
    if (cam_conf.buffer_count <= 0) cam_conf.buffer_count = 3;
    if (const char* e = std::getenv("PIPE_V4L2_BUFS"))
        cam_conf.buffer_count = std::max(2, std::atoi(e));

    Camera cam(cam_conf);
    if (!cam.open()) return;
    const int cam_w = cam.actualWidth();
    const int cam_h = cam.actualHeight();
    const int cvt_code = cam.conversionCode();
    std::cout << "[Camera] 色彩轉換位置: "
              << (cvt_code >= 0 ? "處理執行緒" : "cam.read() 內部") << "\n";

    // ---- resize IP。只需要一份全幀 BGR(處理端專用)----
    const size_t need = static_cast<size_t>(cam_w) * cam_h * 3 * 3
                      + static_cast<size_t>(in_w) * in_w * 3 * 2
                      + 8u * 1024 * 1024;
    hls::use_dma_heap("auto", ((need >> 20) + 16) << 20);
    hls::resize::use_devmem();
    bool ip_ready = hls::resize::available();
    std::cout << "[resize IP] " << (ip_ready ? hls::resize::device_info()
                                             : "不可用:" + hls::resize::last_error())
              << "\n";

    // 三塊全幀 BGR 緩衝,交給 FrameGrabber 當作發布緩衝。
    //
    // 放在 DMA 記憶體是必要的:色彩轉換改在擷取執行緒做,輸出直接落在
    // 這裡,主執行緒的 letterbox 才能 zero-copy 讀它。若用一般記憶體,
    // letterbox 內部就得再複製一次全幀。
    std::vector<cv::Mat> bgr_bufs(3);
    for (int i = 0; i < 3; ++i) {
        if (ip_ready) {
            bgr_bufs[i] = hls::resize::input_buffer(cam_w, cam_h, i);
            if (bgr_bufs[i].empty()) ip_ready = false;
        }
        if (bgr_bufs[i].empty()) bgr_bufs[i].create(cam_h, cam_w, CV_8UC3);
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

    // DPU 那一段拆開量:
    //   tm_dpu_hw   submit + wait_hw —— 硬體執行時間,不受 CPU 競爭影響
    //   tm_dpu_cpu  finish() —— memcpy + NHWC→NCHW 轉置,純 CPU,會被搶
    // 混在一起的話,看到數字上升也分不出是硬體變慢還是 CPU 被搶。
    fpipe::StageTimer tm_rsz, tm_pre, tm_dpu_hw, tm_dpu_cpu, tm_post, tm_out, tm_lat;
    std::atomic<long long> n_proc{0};
    std::atomic<double> cur_fps{0.0};
    const double t_start = fpipe::now_ms();

    // ===================== 擷取:交給 FrameGrabber =====================
    // 執行緒、三緩衝、去積壓都在 camera.h 裡,這邊只要 start / acquire / stop。
    FrameGrabber grabber(cam);

    if (!grabber.setBuffers(bgr_bufs))
        std::cerr << "[FrameGrabber] setBuffers 失敗,改用內部配置的緩衝\n";

    if (cvt_code >= 0) {
        // 色彩轉換搬到擷取執行緒。
        //
        // 為什麼划算:擷取端每輪約一個影格間隔,其中絕大部分是阻塞在
        // 等相機,CPU 閒著。把轉換挪過來等於填進那段空檔 —— 只要
        // 「retrieve + 轉換」不超過影格間隔,擷取端的週期就不會變長,
        // 而處理端實質少掉整整一段。
        grabber.setTransform([cvt_code](const cv::Mat& src, cv::Mat& dst) {
            cv::cvtColor(src, dst, cvt_code);
        });
        std::cout << "[排程] 色彩轉換在擷取執行緒\n";
    }

    grabber.start();
    if (cam_core >= 0) {
        const bool ok = grabber.pinThread(cam_core);
        std::cout << "[排程] 擷取執行緒綁 core " << cam_core
                  << (ok ? " 成功" : " 失敗") << std::endl;
    }

    // ===================== 主執行緒:其餘全部 =====================
    //
    // 不另開 worker:處理本來就是一條串列的流程,交給別的執行緒只是
    // 多一次交接與同步。留在主執行緒還有個好處 —— Ctrl-C 之後這裡
    // 直接跳出迴圈收尾,不用等別人。
    {
        ResizeResult rr;
        hls::resize::Result hres;
        cv::Mat rgb_buf, drawn;
        std::vector<cv::Mat> float_outputs(engine.num_outputs());
        std::vector<bytetrack::Box> boxes;
        long long frames = 0, prev = 0;
        double t_prev = fpipe::now_ms();
        fpipe::Ema fps_ema(0.3);
        bool first = true;

        while (g_running) {
            // Handle 是 RAII:離開作用域自動歸還緩衝。
            // 它持有期間該緩衝不會被擷取端覆蓋,所以要儘早放掉。
            FrameGrabber::Handle f = grabber.acquire(200);
            if (!f.valid()) continue;              // 逾時,回頭再檢查 g_running
            const double t_cap_stamp = f.timestamp();
            const cv::Mat& raw = f.mat();

            // 色彩轉換已在擷取執行緒完成,這裡拿到的就是 BGR
            // ---- letterbox ----
            double t0 = fpipe::now_ms();
            if (ip_ready) {
                hls::resize::letterbox(raw, in_w, hres);
                to_project_result(hres, rr);
            } else {
                resize(raw, in_w, rr);
            }
            tm_rsz.add(fpipe::now_ms() - t0);

            // raw 在這裡就用不到了(letterbox 的輸出已經是另一塊記憶體)。
            // 提前把 Handle 換成空的,緩衝立刻歸還,擷取端就能重用它 ——
            // 不這麼做的話它會一直持有到迴圈結尾,等於少一個可用緩衝。
            f = FrameGrabber::Handle();

            // ---- 前處理 ----
            t0 = fpipe::now_ms();
            cv::cvtColor(rr.img, rgb_buf, cv::COLOR_BGR2RGB);
            cv::Mat dpu_in = engine.input_mat(0);
            norm_and_fix(rgb_buf, in_fix, dpu_in);
            tm_pre.add(fpipe::now_ms() - t0);

            // ---- DPU:硬體與 CPU 兩段分開計時 ----
            // submit + wait_hw 是純硬體時間,和核心數無關;
            // finish 是 memcpy + NHWC->NCHW 轉置,是 CPU 工作。
            // 混在一起看會誤以為「DPU 變慢了」。
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

            // ---- 繪製 + 串流 ----
            t0 = fpipe::now_ms();
            if (draw) draw_tracking(rr.img, drawn, tracks, cur_fps.load());
            else      drawn = rr.img;
            if (stream) streamer->send(drawn);
            tm_out.add(fpipe::now_ms() - t0);

            tm_lat.add(fpipe::now_ms() - t_cap_stamp * 1000.0);
            ++n_proc;

            if (first) { first = false; cv::imwrite("frame0.png", rr.img); }

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
                          << "  rsz " << w(tm_rsz)
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

    // 主迴圈結束 -> 停掉擷取執行緒
    g_running = false;
    grabber.stop();

    const double wall = (fpipe::now_ms() - t_start) / 1000.0;
    cam.close();
    if (streamer) streamer->close();

    const double proc = tm_rsz.avg() + tm_pre.avg() + tm_dpu_hw.avg()
                      + tm_dpu_cpu.avg() + tm_post.avg() + tm_out.avg();

    std::cout << "\n──────── 統計 ────────\n"
              << "處理 " << n_proc.load() << " 幀,"
              << "擷取 " << grabber.frameId() << " 幀,"
              << "推掉 V4L2 舊幀 " << grabber.staleSkipped() << ",略過 "
              << grabber.overwritten() << "\n\n"
              << std::fixed << std::setprecision(2)
              << "  擷取 + 色彩轉換     " << grabber.avgCaptureMs() << " ms\n"
              << "  (色彩轉換已併入擷取執行緒)\n"
              << "  letterbox           " << tm_rsz.avg()  << " ms\n"
              << "  前處理              " << tm_pre.avg()  << " ms\n"
              << "  DPU 硬體            " << tm_dpu_hw.avg()
              << " ms  (submit + wait_hw,硬體時間)\n"
              << "  DPU 輸出整理        " << tm_dpu_cpu.avg()
              << " ms  (finish:memcpy + NHWC→NCHW 轉置,CPU)\n"
              << "  後處理 + 追蹤       " << tm_post.avg() << " ms\n"
              << "  繪製 + 串流         " << tm_out.avg()  << " ms\n"
              << "  ── 處理端合計       " << proc << " ms\n"
              << "     其中硬體等待     " << (tm_dpu_hw.avg() + tm_rsz.avg())
              << " ms(DPU + resize IP,不受 CPU 競爭影響)\n"
              << "     其中 CPU 工作    "
              << (proc - tm_dpu_hw.avg() - tm_rsz.avg())
              << " ms(會與擷取執行緒搶核)\n\n"
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