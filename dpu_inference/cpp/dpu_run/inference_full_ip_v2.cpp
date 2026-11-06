// yolo_full.cpp — 兩條執行緒的低延遲版本(UYVY 直送 IP,IP 同時輸出 DPU 輸入)
//
// 流程:相機 UYVY → hls::uyvy_resize_quant::letterbox
//          (IP:UYVY→RGB + 2/3 倍縮小 + 量化,同時輸出 RGB 與 quant)
//          quant 由函式庫直接寫進 DPU 輸入緩衝
//      → DPU → 後處理(int8 直接解碼)→ 追蹤 → 繪圖 / 串流

// ==================== Includes ====================
#include "../utils/modelrunner_pipe.h"
#include "../utils/tracker.h"
#include "../utils/stream.h"
#include "../utils/drawer.h"            // 需包含 drawer_tracking.h 的宣告
#include "../utils/draw_predictor.h"
#include "../utils/yolopproc.h"
#include "../utils/camera.h"            // 只用到 Camera::Config(CLI 參數)
#include "../utils/v4l2_dmabuf_camera.h"
#include "../utils/cli_args.h"
#include "../utils/frame_pipeline.h"
#include "../utils/hls_uyvy_resize_quant.h"
#include "../utils/norm_fix_letterbox.h"
#include "../utils/data_struct.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace uq = hls::uyvy_resize_quant;


// ==================== Helper Functions ====================

/** @brief Set the x and y values of a pair of floats. */
inline void set_xy(std::pair<float, float>& d, float x, float y) {
    d.first = x; d.second = y;
}

/** @brief Convert the ip letterbox result to the project result. */
inline void to_project_result(const uq::LetterboxResult& s, ResizeResult& d) {
    d.img     = s.img;                 // RGB uint8,繪圖 / 串流用
    d.content = s.content;
    set_xy(d.ratio, s.ratio.x, s.ratio.y);
    set_xy(d.pad,   s.pad.x,   s.pad.y);
}

/**
 * @brief Print the arguments passed to the run_camera function
 */
void print_args(
    std::string xmodel_path,
    Camera::Config cam_conf,
    double conf_th = 0.1,
    double iou_th = 0.45,
    std::string out_file = "",
    bool draw = true,
    bool stream = true,
    stream_params stream_param = {},
    int cam_core = -1
) {
    std::cout << "=========================== Arguments ============================" << std::endl;
    std::cout << "xmodel_path: " << xmodel_path << "\n"
              << "cam_conf: index=" << cam_conf.index
              << ", width=" << cam_conf.width
              << ", height=" << cam_conf.height
              << ", fps=" << cam_conf.fps
              << ", fourcc=" << cam_conf.fourcc
              << ", buffer_count=" << cam_conf.buffer_count
              << ", raw_output=" << (cam_conf.raw_output ? "true" : "false") << "\n"
              << "conf_th: " << conf_th << "\n"
              << "iou_th: " << iou_th << "\n"
              << "out_file: " << out_file << "\n"
              << "draw: " << (draw ? "true" : "false") << "\n"
              << "stream: " << (stream ? "true" : "false") << "\n"
              << "stream_param: ip=" << stream_param.ip
              << ", port=" << stream_param.port
              << ", width=" << stream_param.width
              << ", height=" << stream_param.height
              << ", fps=" << stream_param.fps
              << ", quality=" << stream_param.quality
              << "\n"
              << "cam_core: " << cam_core
              << std::endl;
    std::cout << "===================================================================" << std::endl;
}

/**
 * @brief Print information about the Xmodel
 */
void print_xmodel_info(const XmodelPipelineEngine& engine) {
    std::cout << "=========================== Xmodel Info ===========================" << std::endl;
    std::cout << "Input shape: " << engine.in_h() << "x" << engine.in_w()
              << ", Channels: " << engine.in_c()
              << ", Scale: " << engine.input_scale() << std::endl;

    std::cout << "Output shapes and scales:" << "(" << engine.num_outputs() << ")" << std::endl;
    for (int head_idx = 0; head_idx < static_cast<int>(engine.num_outputs()); ++head_idx) {
        std::cout << "Channels: " << engine.output_channels(head_idx)
                  << ", Height: " << engine.output_h(head_idx)
                  << ", Width: " << engine.output_w(head_idx)
                  << ", Scale: " << engine.output_scale(head_idx) << std::endl;
    }
    std::cout << "===================================================================" << std::endl;
}

/**
 * @brief Format a timer as "recent_avg/recent_max"
 */
inline std::string format_timer(const fpipe::StageTimer& t) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(1)
      << t.recent_avg() << "/" << t.recent_max();
    return o.str();
}

/**
 * @brief Bind the DPU engine outputs to class maps and distance maps.
 *
 * Outputs are identified by shape (xmodel output order is not guaranteed):
 *   channel == 4*dist_ch → dist map, otherwise → cls map
 *   H / W               → which stride
 * Results are ordered by stride: cls_maps[s] / dist_maps[s] belong to dist_strides[s].
 *
 * @param cls_idx  [output] engine output index of cls_maps[s]  (for refresh_output_mats)
 * @param dist_idx [output] engine output index of dist_maps[s]
 * @return true on success
 */
bool bind_outputs(
    const XmodelPipelineEngine& engine,
    std::vector<OutputMap>& cls_maps,
    std::vector<OutputMap>& dist_maps,
    std::vector<int>& cls_idx,
    std::vector<int>& dist_idx,
    const int dist_ch = 16,
    const std::vector<int>& dist_strides = {8, 16, 32}
) {
    cls_maps.clear(); dist_maps.clear();
    cls_idx.clear();  dist_idx.clear();

    const int L = static_cast<int>(dist_strides.size());
    const int num_out = static_cast<int>(engine.num_outputs());
    if (num_out != 2 * L) {
        std::cerr << "Error: DPU engine has " << num_out << " outputs, expected " << 2 * L
                  << " (separate cls and dist maps per stride)." << std::endl;
        return false;
    }

    const int IN_H = engine.in_h(), IN_W = engine.in_w();

    // [stride index] → engine output index
    std::vector<int> cls_by_s(L, -1), dist_by_s(L, -1);
    int nc = 0;

    for (int head_idx = 0; head_idx < num_out; ++head_idx) {
        const int OUT_C = engine.output_channels(head_idx),
                  OUT_H = engine.output_h(head_idx),
                  OUT_W = engine.output_w(head_idx);

        // Which stride does this output belong to?
        int s_idx = -1;
        for (int s = 0; s < L; ++s) {
            if (OUT_H == IN_H / dist_strides[s] && OUT_W == IN_W / dist_strides[s]) {
                s_idx = s;
                break;
            }
        }
        if (s_idx < 0) {
            std::cerr << "Error: Output " << head_idx << " has size " << OUT_H << "x" << OUT_W
                      << ", which does not match any expected stride." << std::endl;
            return false;
        }

        const bool is_dist = (OUT_C == 4 * dist_ch);
        std::vector<int>& slot = is_dist ? dist_by_s : cls_by_s;
        if (slot[s_idx] != -1) {
            std::cerr << "Error: Duplicate " << (is_dist ? "dist" : "cls")
                      << " output for stride " << dist_strides[s_idx] << "." << std::endl;
            return false;
        }
        slot[s_idx] = head_idx;

        if (!is_dist) {
            if (nc != 0 && nc != OUT_C) {
                std::cerr << "Error: cls channel count differs between heads ("
                          << nc << " / " << OUT_C << ")." << std::endl;
                return false;
            }
            nc = OUT_C;
        }
    }

    for (int s = 0; s < L; ++s) {
        if (cls_by_s[s] < 0 || dist_by_s[s] < 0) {
            std::cerr << "Error: Stride " << dist_strides[s]
                      << " is missing a cls or dist output." << std::endl;
            return false;
        }
    }

    // Fill in stride order
    for (int s = 0; s < L; ++s) {
        const int ci = cls_by_s[s], di = dist_by_s[s];
        cls_idx.push_back(ci);
        dist_idx.push_back(di);
        cls_maps.push_back(OutputMap{engine.output_mat_nhwc(0, ci), dist_strides[s],
                                     static_cast<double>(engine.output_scale(ci))});
        dist_maps.push_back(OutputMap{engine.output_mat_nhwc(0, di), dist_strides[s],
                                      static_cast<double>(engine.output_scale(di))});
    }
    return nc > 0;
}

/**
 * @brief Re-acquire output Mat headers after finish() (only pointers, negligible cost).
 */
inline void refresh_output_mats(
    const XmodelPipelineEngine& engine,
    const std::vector<int>& cls_idx, const std::vector<int>& dist_idx,
    std::vector<cv::Mat>& cls_mats, std::vector<cv::Mat>& dist_mats
) {
    for (size_t s = 0; s < cls_idx.size(); ++s)
        cls_mats[s] = engine.output_mat_nhwc(0, cls_idx[s]);
    for (size_t s = 0; s < dist_idx.size(); ++s)
        dist_mats[s] = engine.output_mat_nhwc(0, dist_idx[s]);
}

// program is running or not
static std::atomic<bool> g_running{true};
// Interrupt handler (Ctrl+C) to set the running flag to false
static void signalHandler(int) { g_running = false; }

/**
 * @brief Camera Frame YOLO Full Pipeline with IP Acceleration
 */
void run_camera(
    std::string xmodel_path,
    Camera::Config cam_conf,
    double conf_th = 0.1,
    double iou_th = 0.45,
    std::string out_file = "",
    bool draw = true,
    bool stream = true,
    stream_params stream_param = {},
    int cam_core = -1
) {
    (void)out_file;
    std::signal(SIGINT, signalHandler);
    if (const char* e = std::getenv("PIPE_CAM_CORE")) cam_core = std::atoi(e);

    // ====================== V4L2 Camera (open only) ==================
    if (cam_conf.fourcc != "UYVY") {
        std::cerr << "Error: Camera fourcc must be UYVY for this pipeline." << std::endl;
        return;
    }

    DmabufCamera::Config dma_cam_conf;
    dma_cam_conf.device      = "/dev/video" + std::to_string(cam_conf.index);
    dma_cam_conf.width       = cam_conf.width;
    dma_cam_conf.height      = cam_conf.height;
    dma_cam_conf.fps         = cam_conf.fps;
    dma_cam_conf.num_buffers = 4;
    dma_cam_conf.heap        = "auto";
    if (const char* e = std::getenv("PIPE_V4L2_BUFS")) dma_cam_conf.num_buffers = std::max(4, std::atoi(e));
    if (const char* e = std::getenv("PIPE_DMA_HEAP"))  dma_cam_conf.heap = e;

    DmabufCamera cam(dma_cam_conf);
    if (!cam.open()) {
        std::cerr << "Error: Failed to open camera device: " << cam.error() << std::endl;
        return;
    }
    std::cout << "=========================== Camera Setup ==========================" << std::endl;
    std::cout << "Camera opened: " << dma_cam_conf.device
              << ", Width: " << cam.width()
              << ", Height: " << cam.height()
              << ", FPS: " << dma_cam_conf.fps
              << std::endl;
    std::cout << "===================================================================" << std::endl;

    DmabufCamera::Frame frame; // DmabufCamera Frame


    // ===================== Model Engine =============================
    XmodelPipelineEngine engine(xmodel_path, 1);
    const int IN_W = engine.in_w(), IN_H = engine.in_h();
    const int DIST_CH = 16;
    const std::vector<int> DIST_STRIDES = {8, 16, 32};

    print_xmodel_info(engine);

    // DPU input buffer (int8). Borders are filled by letterbox on the first frame.
    cv::Mat dpu_in = engine.input_mat(0);

    std::vector<OutputMap> cls_maps, dist_maps;   // ordered by stride
    std::vector<int>       cls_idx,  dist_idx;    // engine output index per stride
    if (!bind_outputs(engine, cls_maps, dist_maps, cls_idx, dist_idx, DIST_CH, DIST_STRIDES))
        return;

    // Mat views and scales of DPU output buffers
    std::vector<cv::Mat> cls_mats(cls_maps.size()), dist_mats(dist_maps.size());
    std::vector<float>   cls_scales(cls_maps.size()), dist_scales(dist_maps.size());
    for (size_t i = 0; i < cls_maps.size(); ++i) {
        cls_mats[i]   = cls_maps[i].map;
        cls_scales[i] = static_cast<float>(cls_maps[i].scale);
    }
    for (size_t i = 0; i < dist_maps.size(); ++i) {
        dist_mats[i]   = dist_maps[i].map;
        dist_scales[i] = static_cast<float>(dist_maps[i].scale);
    }

    // Number of classes = cls output channel count
    const int NC = engine.output_channels(cls_idx[0]);
    std::cout << "Model " << IN_W << "x" << IN_H << "  nc=" << NC << "\n";
    for (size_t s = 0; s < DIST_STRIDES.size(); ++s)
        std::cout << "  stride " << std::setw(2) << DIST_STRIDES[s]
                  << ": cls = out" << cls_idx[s] << " (scale " << cls_scales[s] << ")"
                  << "  dist = out" << dist_idx[s] << " (scale " << dist_scales[s] << ")\n";


    // ====================== IP Setup ================================
    // [letterbox] resize_ratio = min(1.0, model_w/cam_w, model_h/cam_h)
    const double resize_ratio = std::min({
        1.0,
        static_cast<double>(IN_W) / cam.width(),
        static_cast<double>(IN_H) / cam.height()
    });

    // check if the IP can be used for the given camera resolution and model input size
    uq::Plan plan = uq::plan_for(
        cam.width(), cam.height(),
        static_cast<uint32_t>(std::round(cam.width()  * resize_ratio)),
        static_cast<uint32_t>(std::round(cam.height() * resize_ratio))
    );

    if (!plan.use_ip) {
        std::cerr << "Error: The IP cannot be used for the given camera resolution and model input size: "
                  << plan.reason << std::endl;
        return;
    }
    std::cout << "=========================== IP Setup ==============================" << std::endl;
    std::cout << "IP will be used for resizing and quantization." << std::endl;
    std::cout << "Resize ratio: " << resize_ratio << std::endl;
    std::cout << "IP Plan: scale=" << plan.scale
              << ", in_w=" << cam.width()
              << ", in_h=" << cam.height()
              << ", out_w=" << plan.params.out_w
              << ", out_h=" << plan.params.out_h
              << std::endl;
    std::cout << "===================================================================" << std::endl;

    // IP DownScale Factor
    const size_t IP_SCALE = static_cast<size_t>(plan.scale);
    // IP output size (RGB888) based on camera resolution and scale factor
    const size_t IP_OUT = (static_cast<size_t>(cam.width()) / IP_SCALE)
                        * (static_cast<size_t>(cam.height()) / IP_SCALE) * 3;
    // required buffer size for IP input staging and IP outputs
    const size_t IP_NEED =
          static_cast<size_t>(cam.width()) * cam.height() * 2   // UYVY input (1 Pixel = 2 bytes)
        + IP_OUT * 2                                            // IP output (RGB + quant)
        + 8u * 1024 * 1024;                                     // 8MB margin

    hls::use_dma_heap("auto", ((IP_NEED >> 20) + 16) << 20);

    // IP control register physical address (optional)
    if (const char* e = std::getenv("PIPE_UYVY_PHYS"))
        uq::use_devmem(std::strtoull(e, nullptr, 16));

    // If the IP is not available, the pipeline will run without IP acceleration
    bool ip_enabled = uq::available();
    if (!ip_enabled)
        std::cerr << "Warning: The IP is not available (" << uq::last_error()
                  << "). The pipeline will run without IP acceleration." << std::endl;
    else
        std::cout << "IP is available: " << uq::device_info() << std::endl;


    // ====================== YOLO Post-Processing ====================
    YOLOPostProcessor yolo_postproc(1, IN_H, IN_W, NC, DIST_CH, DIST_STRIDES);
    // DPU output not cacheable so need to get the data from DDR
    yolo_postproc.set_cls_mem_hint(YOLOPostProcessor::MemHint::NonCacheable);


    // ====================== Streaming ===============================
    std::unique_ptr<RtpJpegStreamer> streamer;
    if (stream) {
        streamer = std::make_unique<RtpJpegStreamer>(
            stream_param.width, stream_param.height,
            stream_param.fps,
            stream_param.ip, stream_param.port,
            stream_param.quality
        );
        if (!streamer->isOpened()) {
            std::cerr << "Error: Failed to open RTP streamer." << std::endl;
            return;
        }
        std::cout << "=========================== Streaming Setup ========================" << std::endl;
        std::cout << "Streaming to: " << stream_param.ip << ":" << stream_param.port
                  << ", Width: " << stream_param.width
                  << ", Height: " << stream_param.height
                  << ", FPS: " << stream_param.fps
                  << ", Quality: " << stream_param.quality
                  << std::endl;
        std::cout << "===================================================================" << std::endl;
    }


    // ====================== Tracking ================================
    bytetrack::Params tp;
    tp.max_lost_seconds = 2; tp.class_aware = true;
    bytetrack::BYTETracker tracker(tp);


    // ====================== Pipeline Timing =========================
    fpipe::StageTimer tm_ip;                // IP time (inside letterbox)
    fpipe::StageTimer tm_rsz;               // letterbox (CPU+IP), quant written to DPU input here
    fpipe::StageTimer tm_pre;               // CPU work between letterbox and DPU submit
    fpipe::StageTimer tm_dpu_hw;            // submit → DPU done
    fpipe::StageTimer tm_dpu_sw;            // finish() (DPU output memcpy)
    fpipe::StageTimer tm_cpu_when_wait_dpu; // CPU work during DPU inference
    fpipe::StageTimer tm_wait_dpu;          // CPU wait DPU time
    fpipe::StageTimer tm_post;              // Post Process + tracking
    fpipe::StageTimer tm_draw;              // Draw
    fpipe::StageTimer tm_out;               // Draw + Stream
    fpipe::StageTimer tm_lat;               // Latency

    std::atomic<long long> n_proc{0};       // processed frame count
    std::atomic<double>    cur_fps{0.0};    // current FPS


    // ====================== Start Camera ============================
    if (!cam.start()) {
        std::cerr << "Error: Failed to start camera streaming: " << cam.error() << std::endl;
        return;
    }
    std::cout << "Camera streaming started." << std::endl;

    if (cam_core >= 0) {
        const bool ok = cam.pinThread(cam_core);
        std::cout << "Capture thread pinned to core " << cam_core
                  << (ok ? " (ok)" : " (failed)") << std::endl;
    }

    const double t_start = fpipe::now_ms();


    // ====================== Main Thread =============================
    {
        ResizeResult rr; // Resize Result
        cv::Mat* const canva_ptr = &rr.img; // draw image pointer to resize image (In-place)
        uq::LetterboxResult lb; // Resize Result (IP letterbox), reused across frames

        std::vector<bytetrack::Box> bounding_boxes; // YOLO prediction

        long long frames = 0, prev = 0; // frames since last report, n_proc at last report
        double t_prev = fpipe::now_ms(); // previous report time
        fpipe::Ema fps_ema(0.3); // EMA of FPS
        bool first_frame = true;

        while (g_running) {
            // =================== Get Frame ===================
            frame = cam.acquire(200); // Wait new frame (200ms timeout)
            if (!frame.valid()) continue; // No frame, skip

            const double t_cap_stamp = frame.timestamp(); // Capture time stamp
            // cam raw uyvy frame : directly use cam dma buffer
            const cv::Mat& raw_uyvy = frame.mat();


            // =================== Letterbox ===================
            double t0 = fpipe::now_ms(); // letterbox start time

            // IP (or CPU fallback inside the library) writes RGB to lb.img
            // and quant directly into dpu_in
            uq::letterbox(raw_uyvy, IN_W, lb, ip_enabled, frame.phys(), &dpu_in);
            to_project_result(lb, rr);

            tm_rsz.add(fpipe::now_ms() - t0);
            if (lb.used_ip)
                tm_ip.add(lb.ip_ms);

            if (first_frame) {
                if (!lb.used_ip)
                    std::cerr << "IP letterbox failed, use CPU letterbox: "
                              << lb.reason << std::endl;
                first_frame = false;
            }


            // =================== Preprocess ==================
            // quant is already in dpu_in; only release the camera buffer and prepare draw area
            t0 = fpipe::now_ms();
            frame.reset(); // Image Buffer already used, release it
            tm_pre.add(fpipe::now_ms() - t0);


            // =================== DPU =========================
            t0 = fpipe::now_ms(); // dpu start time
            engine.submit(0);

            const double t_cpu_process = fpipe::now_ms(); // CPU work during DPU inference start
            // TODO : do something here
            const double t_wait = fpipe::now_ms(); // CPU wait start time
            tm_cpu_when_wait_dpu.add(t_wait - t_cpu_process);

            engine.wait_hw(0); // wait DPU interrupt
            const double t_done = fpipe::now_ms();
            tm_wait_dpu.add(t_done - t_wait);
            tm_dpu_hw.add(t_done - t0);

            t0 = fpipe::now_ms();
            engine.finish(0, false); // do not copy DPU output (DMA) to cacheable buffer
            tm_dpu_sw.add(fpipe::now_ms() - t0);

            // re-acquire output Mat headers after finish()
            refresh_output_mats(engine, cls_idx, dist_idx, cls_mats, dist_mats);


            // =================== Post Process =================
            t0 = fpipe::now_ms(); // Post Process start time

            const std::vector<DetectionBatch>& nms = // NMS Result
                yolo_postproc.process(
                    cls_mats, dist_mats,
                    cls_scales, dist_scales,
                    static_cast<float>(conf_th), static_cast<float>(iou_th),
                    YOLOPostProcessor::Layout::NHWC
                );
            // YOLO output coordinates (model input space)
            map_detections(nms[0], bounding_boxes, 0.f, 0.f, 1.f, 1.f, cv::Size(IN_W, IN_H));
            // tracking results
            const std::vector<bytetrack::Track>& tracks = tracker.update(bounding_boxes, t_cap_stamp);

            tm_post.add(fpipe::now_ms() - t0);


            // =================== Draw and Stream ===============
            t0 = fpipe::now_ms(); // Draw and Stream start time

            if (draw) {
                const double t_draw = fpipe::now_ms();
                draw_tracking_in(*canva_ptr, rr.content, tracks, cur_fps.load());
                tm_draw.add(fpipe::now_ms() - t_draw);
            }

            if (stream)
                streamer->send(*canva_ptr);

            tm_out.add(fpipe::now_ms() - t0);


            // =================== FPS and Latency ===============
            tm_lat.add(fpipe::now_ms() - t_cap_stamp * 1000.0); // Latency
            ++n_proc;

            if (++frames >= 30) {
                const double now = fpipe::now_ms();
                const long long n_now = n_proc.load();
                const double inst = (n_now - prev) * 1000.0 / (now - t_prev);
                cur_fps = fps_ema.push(inst); // Calculate FPS
                t_prev = now; prev = n_now;
                frames = 0;

                std::cout << "Frame " << n_now
                          << "  FPS " << std::fixed << std::setprecision(1) << cur_fps.load()
                          << " | cam prep " << std::setprecision(2) << cam.avgPrepMs()
                          << "  lbox " << format_timer(tm_rsz)
                          << " (ip " << format_timer(tm_ip) << ")"
                          << "  pre " << format_timer(tm_pre)
                          << "  dpu " << format_timer(tm_dpu_hw)
                          << " (cpu " << format_timer(tm_cpu_when_wait_dpu)
                          << " / wait " << format_timer(tm_wait_dpu) << ")"
                          << "  memcpy " << format_timer(tm_dpu_sw)
                          << "  post " << format_timer(tm_post)
                          << "  draw " << std::setprecision(2)
                          << tm_draw.recent_avg() << "/" << tm_draw.recent_max()
                          << std::setprecision(1)
                          << "  out " << format_timer(tm_out)
                          << " | latency " << format_timer(tm_lat)
                          << "  stale " << cam.stale()
                          << "  skip " << cam.overwritten()
                          << "  broken " << cam.badFrames() << std::endl;
            }
        }
    }

    // ====================== Close Application ======================
    g_running = false;
    cam.stop();

    const double wall = (fpipe::now_ms() - t_start) / 1000.0;
    cam.close();
    if (streamer) streamer->close();

    const double proc = tm_rsz.avg() + tm_pre.avg() + tm_dpu_hw.avg()
                      + tm_dpu_sw.avg() + tm_post.avg() + tm_out.avg();
    const double hw = tm_wait_dpu.avg() + tm_ip.avg();  // waiting for hardware

    std::cout << "\n──────── Statistics ────────\n"
              << "Processed " << n_proc.load() << " frames, "
              << "captured " << cam.frames() << " frames, "
              << "dropped stale V4L2 frames " << cam.stale() << ", skipped "
              << cam.overwritten() << ", bad frames " << cam.badFrames() << "\n\n"
              << std::fixed << std::setprecision(2)
              << "  capture copy+clean  " << cam.avgPrepMs() << " ms  ("
              << (cam.zeroCopy() ? "DMABUF zero-copy, clean only" : "MMAP, includes one copy")
              << ", on capture thread)\n"
              << "  letterbox           " << tm_rsz.avg()
              << " ms  (of which uyvy_resize_quant IP " << tm_ip.avg() << " ms, "
              << (ip_enabled ? "IP" : "CPU") << ")\n"
              << "  preprocess          " << tm_pre.avg()
              << " ms  (quant written to DPU input in letterbox, should be ~0)\n"
              << "  DPU (submit→done)   " << tm_dpu_hw.avg() << " ms\n"
              << "    CPU overlap work  " << tm_cpu_when_wait_dpu.avg() << " ms\n"
              << "    remaining wait    " << tm_wait_dpu.avg() << " ms"
              << "  (near 0 means CPU work takes longer than DPU, adding latency)\n"
              << "  DPU output memcpy   " << tm_dpu_sw.avg() << " ms  (no copy, should be ~0)\n"
              << "  postprocess + track " << tm_post.avg()
              << " ms  (direct int8 decode, no transpose / fix2float)\n"
              << "  draw + stream       " << tm_out.avg()
              << " ms  (of which drawing only " << tm_draw.avg() << " ms)\n"
              << "  ── processing total " << proc << " ms\n"
              << "     hardware wait    " << hw << " ms\n"
              << "     CPU work         " << (proc - hw) << " ms\n\n"
              << "End-to-end latency  avg " << tm_lat.avg()
              << "  recent " << tm_lat.recent_avg()
              << "  max " << tm_lat.recent_max() << " ms\n"
              << "Throughput " << (wall > 0 ? n_proc.load() / wall : 0) << " FPS"
              << " (limit " << (proc > 0 ? 1000.0 / proc : 0) << ")\n";
}


// ==================== Main Function =====================
int main(int argc, char* argv[]) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) { print_usage(argv[0]); return -1; }

    stream_params sp{args.st_ip, args.st_port, args.st_width,
                     args.st_height, args.st_fps, args.st_quality};
    Camera::Config cc{args.cam_index, args.cam_width, args.cam_height,
                      args.cam_fps, args.cam_fourcc};

    print_args(
        args.model_path,
        cc, args.conf, args.iou,
        "", args.draw, args.stream, sp
    );

    run_camera(
        args.model_path,
        cc,
        args.conf,
        args.iou,
        "",
        args.draw,
        args.stream, sp
    );

    return 0;
}