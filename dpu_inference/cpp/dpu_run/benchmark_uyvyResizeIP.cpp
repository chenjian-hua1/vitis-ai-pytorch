// ============================================================================
//  benchmark_uyvyResizeIP.cpp — uyvy_resize letterbox:IP(PL)vs PS(CPU)效能量測
//                               + 後段全流程分段量測(int8 直接解碼的後處理)
//
//  ./benchmark ./model/yolo11n_int.xmodel
//  ./benchmark ./model/yolo11n_int.xmodel -n 300 --no-track
//  BENCH_LB=ip   ./benchmark ...     只跑 IP(純淨的全流程數字)
//  BENCH_LB=ps   ./benchmark ...     只跑 PS(純淨的全流程數字)
//  BENCH_LB=both ./benchmark ...     (預設)同一幀 IP / PS 各跑一次做 A/B
//  BENCH_OCV=0   ./benchmark ...     both 模式下不跑 OpenCV 舊路徑參考
//  BENCH_OUT_COPY=1 ./benchmark ...  finish() 整份 memcpy 到 cacheable 暫存區
//                                    (預設 0,同 inference 的 kDpuOutCopy=false)
//  BENCH_POST_REF=1 ./benchmark ...  每幀另跑舊後處理(轉置+反量化+合併+process)
//                                    當參考,量時間並比對結果(不計入主流程)
//
//  流程(對齊 inference_uyvyResizeIP.cpp):
//    DmabufCamera(UYVY 直接落在 DMA 記憶體)
//      → hls::uyvy_resize::letterbox (UYVY→RGB + 縮放 + 補黑邊)
//      → Norm+Fix → DPU 硬體 → DPU finish(是否 memcpy 可選)
//      → 後處理:NHWC int8 直接解碼(cls int8 閾值掃描,只對通過的 anchor
//               讀 dist 並反量化)+ NMS,不轉置、不做整張 fix2float
//      → Track → (RGB2BGR + Draw) → Stream
//
//  模型:box / cls 分開的 6 輸出 xmodel(split_output=True 量化的 best_split.pt),
//        輸出順序用形狀辨認(channel == 4*ch → dist,H/W → stride)
//
//  letterbox 比較的三個版本(both 模式,同一幀、同一塊來源記憶體):
//    IP          letterbox(..., use_ip=true)  PL 做 UYVY→RGB + 2/3 倍縮小,
//                                             CPU 只做零頭縮放 / 貼黑邊
//                  └ 拆成「IP 段(含 sync)」與「CPU 剩餘」兩段
//    PS          letterbox(..., use_ip=false) 同一套演算法,全部在 CPU
//                                             (就是 IP 失效時的 fallback)
//    PS-OpenCV   cvtColor(UYVY→RGB 全幀) + resize()   舊版的 CPU 路徑,當參考
//
//  公平性:
//    - 三個版本的執行順序每幀輪替,避免「後跑的吃到前一個的 cache」
//    - IP 與 PS 的輸出逐 byte 比對,確認兩邊在算同一件事
//    - 主流程只用一個版本的結果(both 時用 IP);另外兩個版本與比對的時間
//      從 Loop / 端到端延遲中扣掉,所以全流程數字仍代表「只跑主版本」的情況
//
//  環境變數(沿用 inference):
//    PIPE_V4L2_BUFS    V4L2 buffer 數(預設 6)
//    PIPE_DMA_HEAP     擷取緩衝用的 dma-heap 名稱
//    PIPE_UYVY_PHYS    uyvy_resize 控制暫存器實體位址(十六進位)
//    PIPE_CAM_CORE     擷取執行緒綁核
//    PIPE_CLS_MEM      不複製輸出時,DPU 輸出緩衝的記憶體屬性:
//                      uncached(預設)= cls 先整張 memcpy 再掃描
//                      cached         = 直接掃描(緩衝確定是 cacheable 才用)
// ============================================================================

#include "cli_args.h"
#include "modelrunner_pipe.h"
#include "tracker.h"
#include "stream.h"
#include "drawer.h"
#include "yolopproc.h"
#include "camera.h"              // 只用到 Camera::Config
#include "v4l2_dmabuf_camera.h"
#include "preproc.h"
#include "frame_pipeline.h"
#include "hls_uyvy_resize.h"
#include "norm_fix_letterbox.h"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

static std::atomic<bool> g_running{true};
static void signalHandler(int) { g_running = false; }

static double now_ms() { return fpipe::now_ms(); }


// ─────────────────────────────────────────────────────────────────────────────
//  分段計時器(保留全部樣本,算百分位數)
// ─────────────────────────────────────────────────────────────────────────────
struct BenchTimer {
    std::string         name;
    std::vector<double> samples;   // ms
    bool                is_hw = false;

    BenchTimer() {}
    BenchTimer(const std::string& n, bool hw = false) : name(n), is_hw(hw) {}

    void   add(double ms) { samples.push_back(ms); }
    bool   empty() const  { return samples.empty(); }
    size_t n() const      { return samples.size(); }

    double sum() const { double s = 0.0; for (double v : samples) s += v; return s; }
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

static void print_header(const std::string& title, bool with_share = true) {
    std::cout << "\n" << title << "\n"
              << std::left  << std::setw(26) << "Stage"
              << std::right << std::setw(9)  << "avg"
              << std::setw(9) << "min"
              << std::setw(9) << "p50"
              << std::setw(9) << "p95"
              << std::setw(9) << "max"
              << std::setw(9) << "std";
    if (with_share) std::cout << std::setw(8) << "share";
    std::cout << std::setw(6) << "型別"
              << "\n" << std::string(94, '-') << "\n";
}

static void print_row(const BenchTimer& s, double total_for_pct, bool with_share = true) {
    if (s.empty()) return;
    std::cout << std::left  << std::setw(26) << s.name
              << std::right << std::fixed   << std::setprecision(3)
              << std::setw(9) << s.avg()
              << std::setw(9) << s.mn()
              << std::setw(9) << s.pct(0.50)
              << std::setw(9) << s.pct(0.95)
              << std::setw(9) << s.mx()
              << std::setw(9) << s.stddev();
    if (with_share) {
        const double share = (total_for_pct > 0.0) ? (s.avg() / total_for_pct * 100.0) : 0.0;
        std::cout << std::setw(7) << std::setprecision(1) << share << "%";
    }
    std::cout << std::setw(6) << (s.is_hw ? "HW" : "CPU") << "\n";
}


// ─────────────────────────────────────────────────────────────────────────────
//  LetterboxResult → 專案的 ResizeResult
// ─────────────────────────────────────────────────────────────────────────────
template <class T>
static void set_xy(std::pair<T, T>& d, float x, float y) {
    d.first = static_cast<T>(x); d.second = static_cast<T>(y);
}
template <class P>
static auto set_xy(P& d, float x, float y) -> decltype(d.x, d.y, void()) {
    d.x = decltype(d.x)(x); d.y = decltype(d.y)(y);
}
static void to_project_result(const hls::uyvy_resize::LetterboxResult& s, ResizeResult& d) {
    d.img = s.img;  d.content = s.content;
    set_xy(d.ratio, s.ratio.x, s.ratio.y);
    set_xy(d.pad,   s.pad.x,   s.pad.y);
}


// ─────────────────────────────────────────────────────────────────────────────
//  6 個輸出的對應:每個 head(stride 8/16/32)各一個 cls、一個 dist
// ─────────────────────────────────────────────────────────────────────────────
struct SplitOutputMap {
    std::vector<int> cls_idx, dist_idx;      // [head] → engine 輸出編號
    int nc = 0;
};

static bool build_split_map(const XmodelPipelineEngine& engine, int in_h, int in_w, int ch,
                            const std::vector<int>& strides, SplitOutputMap& m)
{
    const size_t L = strides.size();
    m.cls_idx.assign(L, -1);
    m.dist_idx.assign(L, -1);
    m.nc = 0;

    if (engine.num_outputs() != 2 * L) {
        std::cerr << "模型有 " << engine.num_outputs() << " 個輸出,預期 " << 2 * L
                  << " 個(box / cls 分開)。\n"
                  << "請用 split_output=True 量化的 xmodel(best_split.pt)。\n";
        return false;
    }
    for (size_t i = 0; i < engine.num_outputs(); ++i) {
        const int C = engine.output_channels(i);
        const int H = engine.output_h(i);
        const int W = engine.output_w(i);

        int head = -1;
        for (size_t s = 0; s < L; ++s)
            if (in_h / strides[s] == H && in_w / strides[s] == W) head = static_cast<int>(s);
        if (head < 0) {
            std::cerr << "輸出 " << i << " 的大小 " << H << "x" << W << " 對不到任何 stride\n";
            return false;
        }
        std::vector<int>& slot = (C == 4 * ch) ? m.dist_idx : m.cls_idx;
        if (slot[head] != -1) {
            std::cerr << "stride " << strides[head] << " 有重複的"
                      << (C == 4 * ch ? " dist" : " cls") << " 輸出\n";
            return false;
        }
        slot[head] = static_cast<int>(i);
        if (C != 4 * ch) {
            if (m.nc != 0 && m.nc != C) {
                std::cerr << "各 head 的 cls channel 數不一致(" << m.nc << " / " << C << ")\n";
                return false;
            }
            m.nc = C;
        }
    }
    for (size_t s = 0; s < L; ++s)
        if (m.cls_idx[s] < 0 || m.dist_idx[s] < 0) {
            std::cerr << "stride " << strides[s] << " 缺少 cls 或 dist 輸出\n";
            return false;
        }
    return m.nc > 0;
}

// cls 讀取模式:finish() 有複製 → 直接掃描;沒複製 → 依 PIPE_CLS_MEM
static YOLOPostProcessor::MemHint cls_mem_hint(bool out_copy)
{
    if (out_copy) return YOLOPostProcessor::MemHint::Cacheable;
    const char* e = std::getenv("PIPE_CLS_MEM");
    if (e && std::string(e) == "cached") return YOLOPostProcessor::MemHint::Cacheable;
    return YOLOPostProcessor::MemHint::NonCacheable;
}

// 依 finish() 的選擇重新取得輸出 Mat header(只是指標)
static void bind_outputs(const XmodelPipelineEngine& engine, const SplitOutputMap& m,
                         std::vector<cv::Mat>& cls_maps, std::vector<cv::Mat>& dist_maps)
{
    for (size_t s = 0; s < m.cls_idx.size(); ++s) {
        cls_maps[s]  = engine.output_mat_nhwc(0, m.cls_idx[s]);
        dist_maps[s] = engine.output_mat_nhwc(0, m.dist_idx[s]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  舊後處理路徑(參考用):轉置 + 反量化 + 合併 cls/dist + process
//  緩衝由呼叫端持有,迴圈中重用,只有第一次會配置
// ─────────────────────────────────────────────────────────────────────────────
struct RefPost {
    std::vector<cv::Mat> merged;
    cv::Mat fd, fc;
    double tr_ms = 0, proc_ms = 0;
};

static const DetectionBatch& run_reference_post(XmodelPipelineEngine& engine,
                                                const SplitOutputMap& m, int ch,
                                                YOLOPostProcessor& ref_pp, RefPost& r,
                                                float conf, float iou)
{
    const double t0 = now_ms();
    r.merged.resize(m.cls_idx.size());
    for (size_t s = 0; s < m.cls_idx.size(); ++s) {
        engine.output_float_nchw(0, m.dist_idx[s], r.fd);      // (1, 64, H, W)
        engine.output_float_nchw(0, m.cls_idx[s],  r.fc);      // (1, nc, H, W)
        const int H = r.fd.size[2], W = r.fd.size[3], HW = H * W;
        const int sz[] = {1, 4 * ch + m.nc, H, W};
        r.merged[s].create(4, sz, CV_32F);
        std::memcpy(r.merged[s].ptr<float>(), r.fd.ptr<float>(),
                    sizeof(float) * 4 * ch * HW);
        std::memcpy(r.merged[s].ptr<float>() + static_cast<size_t>(4) * ch * HW,
                    r.fc.ptr<float>(), sizeof(float) * m.nc * HW);
    }
    const double t1 = now_ms();
    const DetectionBatch& ref = ref_pp.process(r.merged, conf, iou)[0];
    r.tr_ms   = t1 - t0;
    r.proc_ms = now_ms() - t1;
    return ref;
}

// 比對兩組偵測結果,回傳最大差;類別或數量不同回傳 -1
static double compare_detections(const DetectionBatch& ref, const DetectionBatch& got)
{
    if (ref.count != got.count) return -1.0;
    double max_diff = 0;
    for (int k = 0; k < ref.count; ++k) {
        const Detection &a = ref.data[k], &b = got.data[k];
        if (a.class_id != b.class_id) return -1.0;
        max_diff = std::max({max_diff,
            (double)std::fabs(a.x1 - b.x1), (double)std::fabs(a.y1 - b.y1),
            (double)std::fabs(a.x2 - b.x2), (double)std::fabs(a.y2 - b.y2),
            (double)std::fabs(a.score - b.score)});
    }
    return max_diff;
}


// ─────────────────────────────────────────────────────────────────────────────
//  主體
// ─────────────────────────────────────────────────────────────────────────────
enum class LbMode { Both, IP, PS };

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

    // ---- 0. 量測模式 ----
    LbMode mode = LbMode::Both;
    if (const char* e = std::getenv("BENCH_LB")) {
        if      (std::strcmp(e, "ip") == 0) mode = LbMode::IP;
        else if (std::strcmp(e, "ps") == 0) mode = LbMode::PS;
    }
    bool want_ocv = true;
    if (const char* e = std::getenv("BENCH_OCV")) want_ocv = std::atoi(e) != 0;

    // ---- 1. 模型 ----
    XmodelPipelineEngine engine(xmodel_path, 1);
    const int in_w = engine.in_w();
    const int in_h = engine.in_h();
    const int ch = 16;
    const std::vector<int> strides = {8, 16, 32};

    SplitOutputMap omap;
    if (!build_split_map(engine, in_h, in_w, ch, strides, omap)) return;
    const int nc = omap.nc;
    std::cout << "模型輸入: " << in_w << "x" << in_h
              << "  輸出數量: " << engine.num_outputs() << "  nc = " << nc << "\n";
    for (size_t s = 0; s < strides.size(); ++s)
        std::cout << "  stride " << std::setw(2) << strides[s]
                  << ":cls = out" << omap.cls_idx[s]
                  << " (" << engine.output_channels(omap.cls_idx[s]) << "ch, scale "
                  << engine.output_scale(omap.cls_idx[s]) << ")"
                  << "  dist = out" << omap.dist_idx[s]
                  << " (" << engine.output_channels(omap.dist_idx[s]) << "ch, scale "
                  << engine.output_scale(omap.dist_idx[s]) << ")\n";

    YOLOPostProcessor yolo_pp(1, in_h, in_w, nc, ch, strides);
    const int in_fix = static_cast<int>(std::round(std::log2(engine.input_scale())));

    std::vector<cv::Mat> cls_maps(strides.size()), dist_maps(strides.size());
    std::vector<float>   cls_scales(strides.size()), dist_scales(strides.size());
    for (size_t s = 0; s < strides.size(); ++s) {
        cls_scales[s]  = engine.output_scale(omap.cls_idx[s]);
        dist_scales[s] = engine.output_scale(omap.dist_idx[s]);
    }

    // DPU 輸出處理方式:預設同 inference(kDpuOutCopy=false),可用環境變數切換
    bool out_copy = false;
    if (const char* e = std::getenv("BENCH_OUT_COPY")) out_copy = std::atoi(e) != 0;
    yolo_pp.set_cls_mem_hint(cls_mem_hint(out_copy));
    const bool cls_direct = (yolo_pp.cls_mem_hint() == YOLOPostProcessor::MemHint::Cacheable);
    std::cout << "[DPU 輸出] finish():"
              << (out_copy ? "整份 memcpy 到 cacheable 暫存區(BENCH_OUT_COPY=1)"
                           : "不複製,後處理直接讀 DPU 輸出緩衝")
              << "\n           cls 掃描:"
              << (cls_direct ? "直接讀取,不 memcpy" : "先整張 memcpy 再掃描") << "\n";

    // 舊後處理參考(第一個統計幀一定跑一次做比對;BENCH_POST_REF=1 則每幀都跑)
    bool post_ref_every = false;
    if (const char* e = std::getenv("BENCH_POST_REF")) post_ref_every = std::atoi(e) != 0;
    YOLOPostProcessor ref_pp(1, in_h, in_w, nc, ch, strides);
    RefPost ref_buf;

    // ---- 2. 相機:V4L2 DMABUF,UYVY 直接落在 DMA 記憶體 ----
    if (cam_conf.fourcc != "UYVY")
        std::cout << "[Camera] fourcc " << cam_conf.fourcc
                  << " 改為 UYVY(uyvy_resize IP 只吃 UYVY)\n";

    DmabufCamera::Config dc;
    dc.device = "/dev/video" + std::to_string(cam_conf.index);
    dc.width  = cam_conf.width;
    dc.height = cam_conf.height;
    dc.fps    = cam_conf.fps;
    if (const char* e = std::getenv("PIPE_V4L2_BUFS")) dc.num_buffers = std::max(4, std::atoi(e));
    if (const char* e = std::getenv("PIPE_DMA_HEAP"))  dc.heap = e;

    DmabufCamera cam(dc);
    if (!cam.open()) { std::cerr << "[Camera] 開啟失敗:" << cam.error() << "\n"; return; }
    std::cout << "[Camera] " << cam.describe() << "\n";
    const int cam_w = cam.width();
    const int cam_h = cam.height();

    // ---- 3. letterbox 規劃 ----
    const float r0 = std::min(1.0f, std::min(float(in_w) / cam_h, float(in_w) / cam_w));
    const hls::uyvy_resize::Plan plan = hls::uyvy_resize::plan_for(
        cam_w, cam_h,
        static_cast<uint32_t>(std::round(cam_w * r0)),
        static_cast<uint32_t>(std::round(cam_h * r0)));
    const int sc = plan.use_ip ? static_cast<int>(plan.scale) : 1;

    // DMA pool:verify staging + IP 輸出(IP 與 PS 各一份)
    const size_t need =
          static_cast<size_t>(cam_w) * cam_h * 2
        + static_cast<size_t>(cam_w / sc) * (cam_h / sc) * 3 * 2
        + 8u * 1024 * 1024;
    hls::use_dma_heap("auto", ((need >> 20) + 16) << 20);

    if (const char* e = std::getenv("PIPE_UYVY_PHYS"))
        hls::uyvy_resize::use_devmem(std::strtoull(e, nullptr, 16));

    std::cout << "[letterbox] " << cam_w << "x" << cam_h << " -> " << in_w
              << "x" << in_w << ":" << plan.reason;
    if (plan.use_ip)
        std::cout << "(" << plan.scale << " 倍 -> " << plan.params.out_w
                  << "x" << plan.params.out_h << ")";
    std::cout << "\n";

    // ---- 4. IP 可用性 + 開機 verify ----
    bool ip_enabled = plan.use_ip;
    if (!plan.use_ip) {
        std::cout << "[uyvy_resize] 這個解析度 IP 不適用,只能量 PS\n";
    } else if (!hls::uyvy_resize::available()) {
        std::cerr << "[uyvy_resize] 不可用:" << hls::uyvy_resize::last_error()
                  << "\n             裝置:" << hls::uyvy_resize::device_info() << "\n";
        ip_enabled = false;
    } else {
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
            std::cerr << "\n";
            ip_enabled = false;
        }
    }
    if (!ip_enabled && mode != LbMode::PS) {
        std::cout << "[bench] IP 無法使用,模式改為 PS only\n";
        mode = LbMode::PS;
    }

    const bool do_ip  = (mode != LbMode::PS);
    const bool do_ps  = (mode != LbMode::IP);
    const bool do_ocv = (mode == LbMode::Both) && want_ocv;
    const bool main_is_ip = do_ip;

    std::cout << "===== uyvy_resize IP vs PS 量測 (warmup " << warmup
              << " 幀, 統計 " << iter << " 幀) =====\n"
              << "模式: "
              << (mode == LbMode::Both ? "both(同一幀 A/B,主流程用 IP 結果)"
                  : mode == LbMode::IP ? "IP only" : "PS only")
              << (do_ocv ? " + OpenCV 舊路徑參考" : "") << "\n";

    // ---- 5. 串流:尺寸對齊 letterbox 影像 ----
    std::unique_ptr<RtpJpegStreamer> streamer;
    if (stream) {
        stream_param.width  = in_w;
        stream_param.height = in_h;
        streamer = std::make_unique<RtpJpegStreamer>(
            stream_param.width, stream_param.height, stream_param.fps,
            stream_param.ip, stream_param.port, stream_param.quality);
        if (!streamer->isOpened()) {
            std::cerr << "錯誤:無法開啟 GStreamer 發送管線\n";
            streamer.reset(); stream = false;
        }
    }

    // ---- 6. 追蹤 ----
    bytetrack::Params tp;
    tp.max_lost_seconds = 2.0;
    tp.class_aware = true;
    bytetrack::BYTETracker tracker(tp);
    std::vector<bytetrack::Box> boxes;
    const std::vector<bytetrack::Track>* tracks = nullptr;

    // ---- 7. 計時器 ----
    // letterbox 比較
    BenchTimer s_ip_all  ("IP letterbox 總計", true);
    BenchTimer s_ip_hw   ("  └ IP 段 (含 sync)", true);
    BenchTimer s_ip_cpu  ("  └ CPU 剩餘(零頭/貼邊)");
    BenchTimer s_ip_copy ("  └ 輸入複製(非 zero-copy)");
    BenchTimer s_ps_all  ("PS letterbox (同演算法)");
    BenchTimer s_ocv_cvt ("PS-OpenCV cvtColor");
    BenchTimer s_ocv_rsz ("PS-OpenCV resize");
    BenchTimer s_ocv_all ("PS-OpenCV 總計");

    // 全流程(主版本)
    BenchTimer s_cap  ("Capture (等新幀)");
    BenchTimer s_lb   (main_is_ip ? "LetterBox (IP)" : "LetterBox (PS)", main_is_ip);
    BenchTimer s_quant("Norm+Fix (quant)");
    BenchTimer s_dpu  ("DPU 硬體", true);
    BenchTimer s_fin  (out_copy ? "DPU finish (memcpy)" : "DPU finish (不複製)");
    BenchTimer s_post ("後處理 int8解碼+NMS");
    BenchTimer s_trk  ("Track");
    BenchTimer s_draw ("RGB2BGR+Draw");
    BenchTimer s_send ("Stream Send");
    BenchTimer s_loop ("[Loop 淨值]");
    BenchTimer s_lat  ("[端到端延遲 淨值]");

    // IP vs PS 輸出比對
    long long cmp_frames = 0, cmp_bad_frames = 0, cmp_bad_px_sum = 0;
    double    cmp_max_diff = 0.0;
    bool      ip_zero_copy = true;
    bool      ip_failed_mid = false;

    // 後處理:新(int8 直接解碼)vs 舊(轉置+反量化+合併)
    BenchTimer s_ref_tr  ("舊:轉置+反量化+合併");
    BenchTimer s_ref_pp  ("舊:process (DFL+NMS)");
    BenchTimer s_ref_all ("舊:後處理總計");
    long long post_cmp_frames = 0, post_cmp_bad = 0;
    double    post_cmp_max = 0.0;
    bool      post_checked = false;

    // 緩衝:迴圈外重用,避免每幀重新配置
    hls::uyvy_resize::LetterboxResult lb_ip, lb_ps;
    ResizeResult rr, rr_ocv;
    cv::Mat rgb_full, diff, bgr_out, drawn;
    const std::vector<DetectionBatch>* nms_result = nullptr;
    const cv::Size draw_size(in_w, in_h);

    long long det_sum = 0, trk_sum = 0;
    int  done = 0;
    long long frame_idx = 0;
    bool aborted = false;

    // ---- 8. 開始擷取 ----
    if (!cam.start()) { std::cerr << "[Camera] 串流啟動失敗:" << cam.error() << "\n"; return; }
    if (const char* e = std::getenv("PIPE_CAM_CORE")) {
        const int core = std::atoi(e);
        std::cout << "[排程] 擷取執行緒綁 core " << core
                  << (cam.pinThread(core) ? " 成功" : " 失敗") << "\n";
    }

    // 取一幀,連續逾時太多次就放棄
    auto acquire = [&](DmabufCamera::Frame& f, double& wait_ms) -> bool {
        const double t0 = now_ms();
        for (int tries = 0; tries < 10 && g_running; ++tries) {
            f = cam.acquire(200);
            if (f.valid()) { wait_ms = now_ms() - t0; return true; }
        }
        return false;
    };

    // ---- 三種 letterbox ----
    // 回傳實際耗時(ms);measure = false 時不記錄
    auto run_ip = [&](DmabufCamera::Frame& f, bool measure) -> double {
        const double t0 = now_ms();
        hls::uyvy_resize::letterbox(f.mat(), in_w, lb_ip, true, f.phys());
        const double ms = now_ms() - t0;
        if (!lb_ip.used_ip) {
            std::cerr << "\n[uyvy_resize] 執行失敗:" << hls::uyvy_resize::last_error()
                      << "(花費 " << std::fixed << std::setprecision(1) << ms << " ms)\n";
            ip_failed_mid = true;
            return ms;
        }
        if (!lb_ip.zero_copy) ip_zero_copy = false;
        if (measure) {
            s_ip_all.add(ms);
            s_ip_hw.add(lb_ip.ip_ms);
            s_ip_cpu.add(std::max(0.0, ms - lb_ip.ip_ms));
            if (!lb_ip.zero_copy) s_ip_copy.add(lb_ip.timing.copy_ms);
        }
        return ms;
    };
    auto run_ps = [&](DmabufCamera::Frame& f, bool measure) -> double {
        const double t0 = now_ms();
        hls::uyvy_resize::letterbox(f.mat(), in_w, lb_ps, false, f.phys());
        const double ms = now_ms() - t0;
        if (measure) s_ps_all.add(ms);
        return ms;
    };
    auto run_ocv = [&](DmabufCamera::Frame& f, bool measure) -> double {
        const cv::Mat& uyvy = f.mat();
        if (uyvy.type() != CV_8UC2) return 0.0;
        const double t0 = now_ms();
        cv::cvtColor(uyvy, rgb_full, cv::COLOR_YUV2RGB_UYVY);
        const double t1 = now_ms();
        resize(rgb_full, in_w, rr_ocv);           // preproc.h 的 CPU letterbox
        const double t2 = now_ms();
        if (measure) {
            s_ocv_cvt.add(t1 - t0);
            s_ocv_rsz.add(t2 - t1);
            s_ocv_all.add(t2 - t0);
        }
        return t2 - t0;
    };

    // 一次完整處理。回傳「非主版本」花掉的時間,好從 loop/延遲扣掉
    auto process_one = [&](DmabufCamera::Frame& f, double t_cap_stamp, bool measure) -> double {
        double extra = 0.0;
        double t0;

        // ---- letterbox:每幀輪替執行順序 ----
        // 三個版本都在 f.reset() 之前跑完,讀的是同一塊 UYVY
        double main_ms = 0.0;
        const int rot = static_cast<int>(frame_idx++ % 3);
        for (int k = 0; k < 3; ++k) {
            const int which = (k + rot) % 3;      // 0 = IP, 1 = PS, 2 = OpenCV
            if (which == 0 && do_ip) {
                const double ms = run_ip(f, measure);
                if (main_is_ip) main_ms = ms; else extra += ms;
            } else if (which == 1 && do_ps) {
                const double ms = run_ps(f, measure);
                if (!main_is_ip) main_ms = ms; else extra += ms;
            } else if (which == 2 && do_ocv) {
                extra += run_ocv(f, measure);
            }
        }

        // IP 中途失敗:之後改成 PS only,這一幀用 PS 補
        if (main_is_ip && ip_failed_mid) return -1.0;

        // IP vs PS 逐 byte 比對(不計入任何時間)
        if (do_ip && do_ps) {
            t0 = now_ms();
            if (measure && lb_ip.img.size() == lb_ps.img.size()
                        && lb_ip.img.type() == lb_ps.img.type()) {
                cv::absdiff(lb_ip.img, lb_ps.img, diff);
                double mx = 0.0;
                cv::minMaxLoc(diff.reshape(1), nullptr, &mx);
                const int bad = cv::countNonZero(diff.reshape(1));
                ++cmp_frames;
                if (bad > 0) { ++cmp_bad_frames; cmp_bad_px_sum += bad; }
                cmp_max_diff = std::max(cmp_max_diff, mx);
            } else if (measure) {
                ++cmp_frames; ++cmp_bad_frames;    // 尺寸都不同,直接算不符
            }
            extra += now_ms() - t0;
        }

        to_project_result(main_is_ip ? lb_ip : lb_ps, rr);
        if (measure) s_lb.add(main_ms);

        // UYVY 讀完了,立刻歸還緩衝
        f.reset();

        // ---- Norm + Fix:已經是 RGB ----
        t0 = now_ms();
        cv::Mat dpu_in = engine.input_mat(0);
        norm_and_fix_letterbox(rr, in_fix, dpu_in);
        if (measure) s_quant.add(now_ms() - t0);

        // ---- DPU 硬體 ----
        t0 = now_ms();
        engine.submit(0);
        engine.wait_hw(0);
        if (measure) s_dpu.add(now_ms() - t0);

        // ---- DPU finish:是否 memcpy 由 out_copy 決定 ----
        t0 = now_ms();
        engine.finish(0, out_copy);
        bind_outputs(engine, omap, cls_maps, dist_maps);      // 只是取指標
        if (measure) s_fin.add(now_ms() - t0);

        // ---- 後處理:直接吃 NHWC int8,不轉置、不 fix2float ----
        t0 = now_ms();
        nms_result = &yolo_pp.process(cls_maps, dist_maps, cls_scales, dist_scales,
                                      static_cast<float>(conf_th),
                                      static_cast<float>(iou_th),
                                      YOLOPostProcessor::Layout::NHWC);
        if (measure) { s_post.add(now_ms() - t0); det_sum += (*nms_result)[0].count; }

        // ---- 舊後處理參考(不計入主流程,時間算進 extra 扣掉)----
        if (measure && (post_ref_every || !post_checked)) {
            t0 = now_ms();
            const DetectionBatch& ref = run_reference_post(
                engine, omap, ch, ref_pp, ref_buf,
                static_cast<float>(conf_th), static_cast<float>(iou_th));
            const double d = compare_detections(ref, (*nms_result)[0]);
            ++post_cmp_frames;
            if (d < 0 || d >= 1e-3) ++post_cmp_bad;
            if (d > post_cmp_max) post_cmp_max = d;
            if (!post_checked) {
                post_checked = true;
                std::cout << "[check] int8 直接解碼 vs 轉置+反量化:偵測數 "
                          << (*nms_result)[0].count << " / " << ref.count << "  最大差 "
                          << (d < 0 ? std::string("類別/數量不同") : std::to_string(d))
                          << (d >= 0 && d < 1e-3 ? "  OK" : "  不一致!") << "\n";
            }
            if (post_ref_every) {
                s_ref_tr.add(ref_buf.tr_ms);
                s_ref_pp.add(ref_buf.proc_ms);
                s_ref_all.add(ref_buf.tr_ms + ref_buf.proc_ms);
            }
            extra += now_ms() - t0;
        }

        // ---- 追蹤 ----
        if (do_track) {
            t0 = now_ms();
            map_detections((*nms_result)[0], boxes, 0.f, 0.f, 1.f, 1.f, draw_size);
            tracks = &tracker.update(boxes, t_cap_stamp);
            if (measure) { s_trk.add(now_ms() - t0);
                           trk_sum += static_cast<long long>(tracks->size()); }
        }

        // ---- 繪製:這裡才轉回 BGR ----
        if (draw || stream) {
            t0 = now_ms();
            cv::cvtColor(rr.img, bgr_out, cv::COLOR_RGB2BGR);
            if (draw) {
                if (do_track && tracks) {
                    draw_tracking(bgr_out, drawn, *tracks, 0.0);
                } else {
                    ResizeResult identity;
                    identity.img = bgr_out;
                    set_xy(identity.ratio, 1.f, 1.f);
                    set_xy(identity.pad,   0.f, 0.f);
                    draw_detection(bgr_out, drawn, (*nms_result)[0], identity, 0.0);
                }
            } else {
                drawn = bgr_out;
            }
            if (measure) s_draw.add(now_ms() - t0);
        }

        // ---- 串流 ----
        if (stream) {
            t0 = now_ms();
            streamer->send(drawn);
            if (measure) s_send.add(now_ms() - t0);
        }
        return extra;
    };

    // IP 執行中失敗:直接中止統計(混著 IP/PS 的數字沒有比較意義),
    // 並提示改用 BENCH_LB=ps
    auto handle_ip_fail = [&]() {
        std::cerr << "[bench] IP 執行中失敗,中止量測。裝置:"
                  << hls::uyvy_resize::device_info()
                  << "\n        可改用 BENCH_LB=ps 單獨量 PS\n";
        aborted = true;
    };

    // ---- 9. 暖機 ----
    std::cout << "\nwarmup ...\n";
    for (int i = 0; i < warmup && g_running; ++i) {
        DmabufCamera::Frame f;
        double wait_ms = 0;
        if (!acquire(f, wait_ms)) { aborted = true; break; }
        if (process_one(f, f.timestamp(), false) < 0) { handle_ip_fail(); break; }
    }
    if (nms_result && !aborted)
        std::cout << "warmup 後偵測到 " << (*nms_result)[0].count << " 個框\n";

    // ---- 10. 主迴圈 ----
    std::cout << "開始統計(Ctrl+C 可提早結束)...\n";
    const double t_bench0 = now_ms();
    double extra_total = 0.0;

    for (int i = 0; i < iter && g_running && !aborted; ++i) {
        const double t_loop0 = now_ms();

        DmabufCamera::Frame f;
        double wait_ms = 0;
        if (!acquire(f, wait_ms)) { aborted = true; break; }
        s_cap.add(wait_ms);
        const double t_cap_stamp = f.timestamp();     // 秒

        const double extra = process_one(f, t_cap_stamp, true);
        if (extra < 0) { handle_ip_fail(); break; }
        extra_total += extra;

        const double t_end = now_ms();
        s_loop.add(t_end - t_loop0 - extra);
        s_lat.add(t_end - t_cap_stamp * 1000.0 - extra);
        ++done;

        if (done % 30 == 0)
            std::cout << "\r  進度 " << done << "/" << iter << std::flush;
    }
    const double t_bench_total = now_ms() - t_bench0;
    std::cout << "\r  進度 " << done << "/" << iter << "\n";

    // ---- 11. 收尾 ----
    g_running = false;
    cam.stop();
    cam.close();
    if (streamer) streamer->close();

    if (done == 0) { std::cout << "沒有取得任何有效幀,無法統計。\n"; return; }

    // ---- 12. 報表一:letterbox IP vs PS ----
    print_header("===== letterbox:IP vs PS(ms," + std::to_string(cam_w) + "x"
                 + std::to_string(cam_h) + " UYVY -> " + std::to_string(in_w) + "x"
                 + std::to_string(in_w) + " RGB)=====", false);
    print_row(s_ip_all,  0, false);
    print_row(s_ip_hw,   0, false);
    print_row(s_ip_cpu,  0, false);
    print_row(s_ip_copy, 0, false);
    print_row(s_ps_all,  0, false);
    print_row(s_ocv_cvt, 0, false);
    print_row(s_ocv_rsz, 0, false);
    print_row(s_ocv_all, 0, false);

    std::cout << std::fixed << std::setprecision(2);
    if (!s_ip_all.empty() && !s_ps_all.empty() && s_ip_all.avg() > 0) {
        std::cout << "\nIP 相對 PS(同演算法)加速   : "
                  << s_ps_all.avg() / s_ip_all.avg() << "x  (avg), "
                  << s_ps_all.pct(0.95) / std::max(1e-9, s_ip_all.pct(0.95)) << "x  (p95)\n";
    }
    if (!s_ip_all.empty() && !s_ocv_all.empty() && s_ip_all.avg() > 0) {
        std::cout << "IP 相對 PS-OpenCV 加速       : "
                  << s_ocv_all.avg() / s_ip_all.avg() << "x  (avg)\n";
    }
    if (!s_ip_all.empty()) {
        std::cout << "IP letterbox 內 CPU 佔比      : "
                  << std::setprecision(1)
                  << (s_ip_all.avg() > 0 ? s_ip_cpu.avg() / s_ip_all.avg() * 100.0 : 0.0)
                  << " %  (其餘是等 IP)\n"
                  << "IP 輸入                      : "
                  << (ip_zero_copy ? "zero-copy(實體位址直讀)" : "有額外複製") << "\n";
    }
    if (cmp_frames > 0) {
        std::cout << "IP vs PS 輸出比對            : " << cmp_frames << " 幀中 "
                  << cmp_bad_frames << " 幀不一致,最大差 "
                  << std::setprecision(0) << cmp_max_diff;
        if (cmp_bad_frames > 0)
            std::cout << ",平均每幀不符 " << std::setprecision(1)
                      << double(cmp_bad_px_sum) / cmp_bad_frames << " 個值";
        std::cout << (cmp_bad_frames == 0 ? "  (bit-exact)" : "") << "\n";
    }

    // ---- 13. 報表二:後處理 新 vs 舊 ----
    print_header("===== 後處理:int8 直接解碼 vs 舊路徑(ms)=====", false);
    print_row(s_post,    0, false);
    print_row(s_ref_tr,  0, false);
    print_row(s_ref_pp,  0, false);
    print_row(s_ref_all, 0, false);
    std::cout << std::fixed << std::setprecision(2);
    if (!s_ref_all.empty() && s_post.avg() > 0) {
        std::cout << "\n新後處理相對舊路徑加速      : "
                  << s_ref_all.avg() / s_post.avg() << "x  (avg)\n";
    } else {
        std::cout << "\n(舊路徑只在第一幀跑一次做比對;設 BENCH_POST_REF=1 才會每幀計時)\n";
    }
    if (post_cmp_frames > 0) {
        std::cout << "新舊結果比對                : " << post_cmp_frames << " 幀中 "
                  << post_cmp_bad << " 幀不一致";
        if (post_cmp_max > 0) std::cout << ",最大差 " << std::setprecision(6) << post_cmp_max;
        std::cout << "\n";
    }

    // ---- 14. 報表三:全流程(主版本)----
    const BenchTimer* all[] = {&s_lb, &s_quant, &s_dpu, &s_fin, &s_post,
                               &s_trk, &s_draw, &s_send};
    double compute = 0, hw_ms = 0;
    for (const BenchTimer* s : all) compute += s->avg();
    hw_ms = s_dpu.avg() + (main_is_ip ? s_ip_hw.avg() : 0.0);
    const double loop_ms = s_loop.avg();

    print_header(std::string("===== 全流程分段(主版本 = ") + (main_is_ip ? "IP" : "PS")
                 + ";share 以計算總計為分母)=====");
    print_row(s_cap, loop_ms);
    std::cout << std::string(94, '-') << "\n";
    for (const BenchTimer* s : all) print_row(*s, compute);
    std::cout << std::string(94, '-') << "\n";
    print_row(s_loop, loop_ms);

    std::cout << std::fixed << std::setprecision(3)
              << "\n擷取端 copy+clean (擷取執行緒): " << cam.avgPrepMs() << " ms  ("
              << (cam.zeroCopy() ? "DMABUF 零複製" : "MMAP,含一次複製") << ")\n"
              << "計算總計 (不含擷取)         : " << compute << " ms  →  "
              << std::setprecision(2) << (compute > 0 ? 1000.0 / compute : 0.0)
              << " FPS (理論上限)\n"
              << std::setprecision(3)
              << "  其中硬體等待              : " << hw_ms << " ms  (DPU"
              << (main_is_ip ? " + uyvy_resize IP" : "") << ")\n"
              << "  其中 CPU 工作             : " << (compute - hw_ms) << " ms\n";

    if (main_is_ip && !s_ps_all.empty()) {
        const double compute_ps = compute - s_lb.avg() + s_ps_all.avg();
        std::cout << "若 letterbox 改用 PS        : " << compute_ps << " ms  →  "
                  << std::setprecision(2) << (compute_ps > 0 ? 1000.0 / compute_ps : 0.0)
                  << " FPS (以本次 PS 實測推算)\n" << std::setprecision(3);
    }

    std::cout << "\n迴圈淨值 (含擷取等待)       : " << loop_ms << " ms  →  "
              << std::setprecision(2) << (loop_ms > 0 ? 1000.0 / loop_ms : 0.0) << " FPS\n"
              << std::setprecision(3)
              << "端到端延遲淨值 avg/p95/max  : " << s_lat.avg() << " / "
              << s_lat.pct(0.95) << " / " << s_lat.mx() << " ms\n";
    if (extra_total > 0)
        std::cout << "(非主版本 letterbox、輸出比對與舊後處理參考共花 " << std::setprecision(2)
                  << extra_total / done << " ms/幀,已從上面兩項扣除;"
                  << "wall clock FPS 未扣除)\n";
    std::cout << std::setprecision(2)
              << "實測平均 FPS (wall clock)   : "
              << (t_bench_total > 0 ? done * 1000.0 / t_bench_total : 0.0)
              << "  (" << done << " 幀 / " << std::setprecision(1)
              << t_bench_total / 1000.0 << " s)\n"
              << std::setprecision(2)
              << "平均偵測框數 / 幀           : " << double(det_sum) / done << "\n";
    if (do_track)
        std::cout << "平均 track 數 / 幀          : " << double(trk_sum) / done << "\n";
    std::cout << "相機:擷取 " << cam.frames() << " 幀,推掉舊幀 " << cam.stale()
              << ",略過 " << cam.overwritten() << ",壞幀 " << cam.badFrames() << "\n";
    if (aborted)
        std::cout << "(測試被中斷,統計基於已完成的 " << done << " 幀)\n";

    // ---- 15. 存最後一幀(IP/PS 都有的話各存一張,方便目視比對)----
    if (!save_last.empty() && nms_result && !rr.img.empty()) {
        cv::cvtColor(rr.img, bgr_out, cv::COLOR_RGB2BGR);
        if (do_track && tracks) {
            draw_tracking(bgr_out, drawn, *tracks, 0.0);
        } else {
            ResizeResult identity;
            identity.img = bgr_out;
            set_xy(identity.ratio, 1.f, 1.f);
            set_xy(identity.pad,   0.f, 0.f);
            draw_detection(bgr_out, drawn, (*nms_result)[0], identity, 0.0);
        }
        if (!drawn.empty() && cv::imwrite(save_last, drawn))
            std::cout << "最後一幀結果已存至: " << save_last << "\n";
        else
            std::cerr << "imwrite 失敗: " << save_last << "\n";

        if (do_ip && do_ps && !lb_ip.img.empty() && !lb_ps.img.empty()) {
            cv::Mat a, b;
            cv::cvtColor(lb_ip.img, a, cv::COLOR_RGB2BGR);
            cv::cvtColor(lb_ps.img, b, cv::COLOR_RGB2BGR);
            cv::imwrite("benchmark_lb_ip.png", a);
            cv::imwrite("benchmark_lb_ps.png", b);
            std::cout << "letterbox 輸出已存至: benchmark_lb_ip.png / benchmark_lb_ps.png\n";
        }
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