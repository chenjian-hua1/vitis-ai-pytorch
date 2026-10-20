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
//  DPU 輸出(box / cls 分開的 6 輸出模型):
//    不再做 NHWC→NCHW 轉置,也不再做整張圖的 fix2float。
//    後處理直接吃 DPU 的 NHWC int8 輸出:
//      1. cls 閾值換算成 int8,直接比較(不做浮點運算)
//      2. 只有通過閾值的 anchor,才讀它的 64 個 dist 值並反量化
//    finish(ctx, copy) 要不要把整份輸出 memcpy 到暫存區,由下方 kDpuOutCopy 決定。
//
//  繪圖(本次修改):
//    - 不 clone:直接畫在 letterbox 後的影像上(draw_tracking_in)
//    - 只畫在 letterbox 的內容區(rr.content)內,黑邊完全不會被寫入,
//      所以重複使用的 lb.img 不會殘留上一幀的框 / FPS,也不會污染下一幀的 DPU 輸入
//    - 不需要串流 BGR 時,完全不做 RGB→BGR(PIPE_STREAM_RGB=1)
//    - 需要 BGR 時,轉換移到 DPU 執行期間(CPU 空檔),不佔延遲
//    - 可選:DPU 執行期間把「預測的框線區塊」載入 L2(PIPE_PREFETCH)
//
//  擷取改用 DmabufCamera(V4L2 DMABUF import),不再經過 cv::VideoCapture。
//  架構:一條擷取執行緒(DQBUF + clean)+ 主執行緒串列處理,只取最新一幀。
//
//  環境變數:
//    PIPE_CAM_CORE      擷取執行緒綁核
//    PIPE_V4L2_BUFS     V4L2 buffer 數(DMABUF 模式,預設 6)
//    PIPE_DMA_HEAP      擷取緩衝用的 dma-heap 名稱(預設自動挑,排除 system)
//    PIPE_UYVY_PHYS     uyvy_resize 控制暫存器實體位址(十六進位)
//    PIPE_CLS_MEM       kDpuOutCopy=false 時,DPU 輸出緩衝的記憶體屬性:
//                       uncached(預設)= cls 先整張 memcpy 再掃描
//                       cached         = 直接掃描
//
//    PIPE_PREFETCH      DPU 期間預載入繪圖區塊(預設 0)
//                       0 = 關閉   1 = 讀入快取   2 = 讀後寫回(取得獨佔權)
//                       只有 draw 開啟時才有作用
//    PIPE_STREAM_RGB    1 = 直接串流 RGB 影像(RtpJpegStreamer 的 appsrc caps 要是 RGB)
//                       0 = 串流前轉成 BGR(預設,和原本相容)
//    PIPE_STREAM_CLONE  1 = 每幀把影像複製一份再交給 streamer->send()
//                       (若 send() 是非同步且不複製,才需要;預設 0)
//    PIPE_CHECK_LB      1 = 第 2 幀檢查 letterbox 是否每幀完整重寫內容區
// ─────────────────────────────────────────────────────────────

#include "modelrunner_pipe.h"
#include "tracker.h"
#include "stream.h"
#include "drawer.h"            // 需包含 drawer_tracking.h 的宣告
// #include "prefetch_drawer.h"
#include "draw_predictor.h"
#include "yolopproc.h"
#include "camera.h"            // 只用到 Camera::Config(CLI 參數)
#include "v4l2_dmabuf_camera.h"
#include "preproc.h"
#include "cli_args.h"
#include "frame_pipeline.h"
#include "hls_uyvy_resize.h"
#include "norm_fix_letterbox.h"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ── DPU 輸出設定 ──────────────────────────────────────────────
//   true  : finish() 把整份輸出 memcpy 到 cacheable 暫存區,後處理讀暫存區
//   false : finish() 不複製,後處理直接讀 DPU 輸出緩衝
static constexpr bool kDpuOutCopy = false;

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

inline int env_int(const char* name, int def) {
    const char* e = std::getenv(name);
    return e ? std::atoi(e) : def;
}

// 繪圖範圍 = letterbox 的內容區;內容區為空(沒有黑邊)時用整張影像
inline cv::Rect draw_area(const ResizeResult& rr) {
    cv::Rect a = rr.content;
    a &= cv::Rect(0, 0, rr.img.cols, rr.img.rows);
    return a.empty() ? cv::Rect(0, 0, rr.img.cols, rr.img.rows) : a;
}

// 檢查 letterbox 是否每幀完整重寫內容區(含邊緣):
// 先後用白、黑兩種底色預填 lb.img 各做一次,內容區結果必須完全相同。
// 最後一次以黑色預填,所以檢查後 lb.img 仍是正確的(黑邊為黑)。
template <class Phys>
void check_letterbox_rewrite(const cv::Mat& uyvy, int in_w,
                             hls::uyvy_resize::LetterboxResult& lb,
                             bool ip_enabled, Phys phys)
{
    const cv::Rect c = cv::Rect(lb.content) & cv::Rect(0, 0, lb.img.cols, lb.img.rows);
    if (c.empty()) { std::cout << "[check] letterbox 內容區為空,略過\n"; return; }
    lb.img.setTo(cv::Scalar::all(255));
    hls::uyvy_resize::letterbox(uyvy, in_w, lb, ip_enabled, phys);
    const cv::Mat a = lb.img(c).clone();
    lb.img.setTo(cv::Scalar::all(0));
    hls::uyvy_resize::letterbox(uyvy, in_w, lb, ip_enabled, phys);
    const bool ok = cv::norm(a, lb.img(c), cv::NORM_INF) == 0;
    std::cout << "[check] letterbox 內容區每幀完整重寫:"
              << (ok ? "是" : "否!(請把 draw_area 縮小 1~2 px)") << "\n";
}

// ─────────────────────────────────────────────────────────────
//  6 個輸出的對應關係:每個 head(stride 8/16/32)各一個 cls、一個 dist
//  xmodel 的輸出順序不一定和 Python return 的順序相同,所以用形狀辨認:
//    channel == 4*ch → dist(DFL),其他 → cls
//    H / W           → 屬於哪一個 stride
// ─────────────────────────────────────────────────────────────
struct SplitOutputMap {
    std::vector<int> cls_idx, dist_idx;      // [head] → engine 輸出編號
    int nc = 0;
};

bool build_split_map(const XmodelPipelineEngine& engine, int in_h, int in_w, int ch,
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
            std::cerr << "輸出 " << i << " 的大小 " << H << "x" << W
                      << " 對不到任何 stride\n";
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

// cls 讀取模式:由使用者明確指定,不使用自動判斷
YOLOPostProcessor::MemHint cls_mem_hint(bool out_copy)
{
    if (out_copy) return YOLOPostProcessor::MemHint::Cacheable;
    const char* e = std::getenv("PIPE_CLS_MEM");
    if (e && std::string(e) == "cached") return YOLOPostProcessor::MemHint::Cacheable;
    return YOLOPostProcessor::MemHint::NonCacheable;
}

// 依上一次 finish() 的選擇,重新取得輸出的 Mat header(只是指標,成本可忽略)
void bind_outputs(const XmodelPipelineEngine& engine, const SplitOutputMap& m,
                  std::vector<cv::Mat>& cls_maps, std::vector<cv::Mat>& dist_maps)
{
    for (size_t s = 0; s < m.cls_idx.size(); ++s) {
        cls_maps[s]  = engine.output_mat_nhwc(0, m.cls_idx[s]);
        dist_maps[s] = engine.output_mat_nhwc(0, m.dist_idx[s]);
    }
}

// 第一幀:用舊路徑(轉置 + 反量化 + 合併輸出)算一次,和新路徑的結果比對
void check_against_reference(XmodelPipelineEngine& engine, const SplitOutputMap& m,
                             int in_h, int in_w, int ch, float conf, float iou,
                             const DetectionBatch& got)
{
    std::vector<cv::Mat> merged;
    cv::Mat fd, fc;
    for (size_t s = 0; s < m.cls_idx.size(); ++s) {
        engine.output_float_nchw(0, m.dist_idx[s], fd);      // (1, 64, H, W)
        engine.output_float_nchw(0, m.cls_idx[s],  fc);      // (1, nc, H, W)
        const int H = fd.size[2], W = fd.size[3], HW = H * W;
        const int sz[] = {1, 4 * ch + m.nc, H, W};
        cv::Mat x(4, sz, CV_32F);
        std::memcpy(x.ptr<float>(), fd.ptr<float>(), sizeof(float) * 4 * ch * HW);
        std::memcpy(x.ptr<float>() + static_cast<size_t>(4) * ch * HW, fc.ptr<float>(),
                    sizeof(float) * m.nc * HW);
        merged.push_back(x);
    }

    YOLOPostProcessor ref_pp(1, in_h, in_w, m.nc, ch);
    const DetectionBatch& ref = ref_pp.process(merged, conf, iou)[0];

    double max_diff = 0;
    bool ok = (ref.count == got.count);
    for (int k = 0; ok && k < ref.count; ++k) {
        const Detection &a = ref.data[k], &b = got.data[k];
        if (a.class_id != b.class_id) { ok = false; break; }
        max_diff = std::max({max_diff,
            (double)std::fabs(a.x1 - b.x1), (double)std::fabs(a.y1 - b.y1),
            (double)std::fabs(a.x2 - b.x2), (double)std::fabs(a.y2 - b.y2),
            (double)std::fabs(a.score - b.score)});
    }
    std::cout << "[check] int8 直接解碼 vs 轉置+反量化:偵測數 " << got.count << " / "
              << ref.count << "  最大差 " << max_diff
              << (ok && max_diff < 1e-3 ? "  OK" : "  不一致!") << "\n";
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

    // ---- 繪圖 / 串流 / 預存取設定 ----
    const int  pf_mode      = env_int("PIPE_PREFETCH", 0);
    const bool prefetch_on  = draw && pf_mode > 0;
    const bool stream_rgb   = env_int("PIPE_STREAM_RGB", 0) != 0;
    const bool stream_clone = env_int("PIPE_STREAM_CLONE", 0) != 0;
    const bool check_lb     = env_int("PIPE_CHECK_LB", 0) != 0;
    // 串流要 BGR 才需要轉換;不串流、或串流直接吃 RGB 時完全不轉
    // const bool need_bgr     = stream && !stream_rgb;
    // stream 已經是 rgb 格式
    const bool need_bgr = 0;

    set_draw_prefetch(!prefetch_on ? DrawPrefetch::Off
                      : pf_mode == 1 ? DrawPrefetch::Read : DrawPrefetch::Write);
    DrawPredictor draw_pred;

    std::cout << "[繪圖] 就地繪製於 letterbox 內容區  預存取:"
              << (!prefetch_on ? "關閉" : pf_mode == 1 ? "讀入快取" : "讀後寫回")
              << "  串流格式:" << (!stream ? "無串流" : stream_rgb ? "RGB(不轉換)" : "BGR(DPU 期間轉換)")
              << "\n";

    // ---- 模型 ----
    XmodelPipelineEngine engine(xmodel_path, 1);
    const int in_w = engine.in_w();
    const int in_h = engine.in_h();

    const int ch = 16;
    const std::vector<int> strides = {8, 16, 32};

    // 6 個輸出 → [head] 的 cls / dist 對應
    SplitOutputMap omap;
    if (!build_split_map(engine, in_h, in_w, ch, strides, omap)) return;
    const int nc = omap.nc;
    std::cout << "模型 " << in_w << "x" << in_h << "  nc=" << nc << "\n";
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

    // ---- 後處理直接讀的 NHWC int8 輸出 ----
    std::vector<cv::Mat> cls_maps(strides.size()), dist_maps(strides.size());
    std::vector<float>   cls_scales(strides.size()), dist_scales(strides.size());
    for (size_t s = 0; s < strides.size(); ++s) {
        cls_scales[s]  = engine.output_scale(omap.cls_idx[s]);    // = 2^-fix_point
        dist_scales[s] = engine.output_scale(omap.dist_idx[s]);
    }

    // ---- DPU 輸出的處理方式:由使用者決定,不自動判斷 ----
    const bool out_copy = kDpuOutCopy;
    yolo_pp.set_cls_mem_hint(cls_mem_hint(out_copy));
    const bool cls_direct = (yolo_pp.cls_mem_hint() == YOLOPostProcessor::MemHint::Cacheable);
    std::cout << "[DPU 輸出] finish():"
              << (out_copy ? "整份 memcpy 到 cacheable 暫存區(kDpuOutCopy=true)"
                           : "不複製,後處理直接讀 DPU 輸出緩衝(kDpuOutCopy=false)")
              << "\n           cls 掃描:"
              << (cls_direct ? "直接讀取,不 memcpy" : "先整張 memcpy 再掃描")
              << "\n";

    // ---- 相機:V4L2 DMABUF,影格直接落在 DMA 記憶體 ----
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
    if (!cam.open()) {
        std::cerr << "[Camera] 開啟失敗:" << cam.error() << "\n";
        return;
    }
    std::cout << "[Camera] " << cam.describe() << "\n";
    const int cam_w = cam.width();
    const int cam_h = cam.height();

    // ---- letterbox 規劃:跟每一幀用的是同一個函式,開機時先看一次 ----
    const float r0 = std::min(1.0f, std::min(float(in_w) / cam_h, float(in_w) / cam_w));
    const hls::uyvy_resize::Plan plan = hls::uyvy_resize::plan_for(
        cam_w, cam_h,
        static_cast<uint32_t>(std::round(cam_w * r0)),
        static_cast<uint32_t>(std::round(cam_h * r0)));
    const int sc = plan.use_ip ? static_cast<int>(plan.scale) : 1;

    // ---- DMA pool ----
    const size_t need =
          static_cast<size_t>(cam_w) * cam_h * 2                // verify 的 staging
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
    // tm_dpu_hw = submit → DPU 完成(包含 DPU 期間做的 CPU 工作)
    //   tm_ovl  = DPU 期間的 CPU 工作(BGR 轉換 + 預存取)
    //   tm_wait = CPU 工作做完後,還要再等 DPU 多久(應 > 0,否則 CPU 工作拖慢了延遲)
    // tm_draw 只有繪圖本身,方便比較預存取開關的差異
    fpipe::StageTimer tm_ip, tm_rsz, tm_pre, tm_dpu_hw, tm_ovl, tm_wait,
                      tm_dpu_cpu, tm_post, tm_draw, tm_out, tm_lat;
    std::atomic<long long> n_proc{0};
    std::atomic<double> cur_fps{0.0};
    const double t_start = fpipe::now_ms();

    // ===================== 擷取:DmabufCamera =====================
    if (!cam.start()) {
        std::cerr << "[Camera] 串流啟動失敗:" << cam.error() << "\n";
        return;
    }
    if (cam_core >= 0) {
        const bool ok = cam.pinThread(cam_core);
        std::cout << "[排程] 擷取執行緒綁 core " << cam_core
                  << (ok ? " 成功" : " 失敗") << std::endl;
    }

    // ===================== 主執行緒 =====================
    {
        ResizeResult rr;
        hls::uyvy_resize::LetterboxResult lb;     // 迴圈外重用,lb.img 不會每幀重新配置
        cv::Mat bgr_out;                          // 需要 BGR 時才使用,迴圈外重用
        std::vector<bytetrack::Box> boxes;
        long long frames = 0, prev = 0;
        double t_prev = fpipe::now_ms();
        fpipe::Ema fps_ema(0.3);
        bool first = true;
        bool warned_copy = false;

        while (g_running) {
            DmabufCamera::Frame f = cam.acquire(200);
            if (!f.valid()) continue;
            const double t_cap_stamp = f.timestamp();
            const cv::Mat& uyvy = f.mat();              // 就在 DMA 記憶體裡

            // ---- letterbox:UYVY 進,RGB 正方形出 ----
            double t0 = fpipe::now_ms();
            hls::uyvy_resize::letterbox(uyvy, in_w, lb, ip_enabled, f.phys());
            if (check_lb && frames == 1)
                check_letterbox_rewrite(uyvy, in_w, lb, ip_enabled, f.phys());
            to_project_result(lb, rr);
            tm_rsz.add(fpipe::now_ms() - t0);

            // UYVY 已經讀完(lb.img 是另一塊記憶體),立刻歸還緩衝給擷取端
            f.reset();

            if (lb.used_ip) {
                tm_ip.add(lb.ip_ms);
                if (!lb.zero_copy && !warned_copy) {
                    warned_copy = true;
                    std::cerr << "[uyvy_resize] 輸入沒有走實體位址,每幀多一次複製 "
                              << std::fixed << std::setprecision(2) << lb.timing.copy_ms
                              << " ms\n";
                }
            } else if (ip_enabled) {
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
            norm_and_fix_letterbox(rr, in_fix, dpu_in);
            tm_pre.add(fpipe::now_ms() - t0);

            // ---- DPU ----
            // rr.img 在 submit 前已完成、DPU 讀的是 dpu_in,所以 DPU 執行期間
            // 可以安全地讀 rr.img(轉 BGR)並預載入繪圖區塊
            const cv::Rect area = draw_area(rr);
            cv::Mat* canvas_ptr = &rr.img;                // 實際要畫、要送出的影像

            t0 = fpipe::now_ms();
            engine.submit(0);

            // ===== DPU 執行期間:CPU 空檔 =====
            const double t_ovl = fpipe::now_ms();
            if (need_bgr) {
                cv::cvtColor(rr.img, bgr_out, cv::COLOR_RGB2BGR);
                canvas_ptr = &bgr_out;
            }
            if (prefetch_on)       // 轉完再預載入,框線區塊才會是 L2 裡最新的資料
                prefetch_track_regions_in(*canvas_ptr, area, draw_pred.predict(t_cap_stamp));
            const double t_wait = fpipe::now_ms();
            tm_ovl.add(t_wait - t_ovl);

            engine.wait_hw(0);
            const double t_done = fpipe::now_ms();
            tm_wait.add(t_done - t_wait);
            tm_dpu_hw.add(t_done - t0);

            // ---- DPU 輸出:是否 memcpy 由 kDpuOutCopy 決定 ----
            t0 = fpipe::now_ms();
            engine.finish(0, out_copy);
            tm_dpu_cpu.add(fpipe::now_ms() - t0);
            bind_outputs(engine, omap, cls_maps, dist_maps);

            // ---- 後處理 + 追蹤:直接吃 NHWC int8,不轉置、不 fix2float ----
            t0 = fpipe::now_ms();
            const std::vector<DetectionBatch>& nms =
                yolo_pp.process(cls_maps, dist_maps, cls_scales, dist_scales,
                                static_cast<float>(conf_th), static_cast<float>(iou_th),
                                YOLOPostProcessor::Layout::NHWC);
            map_detections(nms[0], boxes, 0.f, 0.f, 1.f, 1.f, cv::Size(in_w, in_w));
            const std::vector<bytetrack::Track>& tracks =
                tracker.update(boxes, t_cap_stamp);
            if (prefetch_on) draw_pred.update(tracks, t_cap_stamp);   // 只讀 tracks,不動 tracker
            tm_post.add(fpipe::now_ms() - t0);

            // 第一幀:和舊路徑(轉置 + 反量化)比對一次
            if (first) {
                check_against_reference(engine, omap, in_h, in_w, ch,
                                        static_cast<float>(conf_th),
                                        static_cast<float>(iou_th), nms[0]);

                // 第一幀要存的乾淨影像(繪圖是就地進行,要在畫之前先留一份 BGR)
                cv::Mat frame0;
                
                if (canvas_ptr == &bgr_out) frame0 = bgr_out.clone();
                else                    cv::cvtColor(rr.img, frame0, cv::COLOR_RGB2BGR);

                cv::imwrite("frame0.png", frame0); 
                frame0.release(); 
                
                first = false;
            }

            // ---- 繪製 + 串流:就地、限定在內容區,沒有 clone、沒有整張 cvtColor ----
            t0 = fpipe::now_ms();
            if (draw) {
                const double td = fpipe::now_ms();
                draw_tracking_in(*canvas_ptr, area, tracks, cur_fps.load());
                tm_draw.add(fpipe::now_ms() - td);
            }
            if (stream) {
                if (stream_clone) {
                    cv::Mat c = canvas_ptr->clone();
                    streamer->send(c);
                } else {
                    streamer->send(*canvas_ptr);
                }
            }
            tm_out.add(fpipe::now_ms() - t0);

            tm_lat.add(fpipe::now_ms() - t_cap_stamp * 1000.0);
            ++n_proc;

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
                          << " | cam prep " << std::setprecision(2) << cam.avgPrepMs()
                          << std::setprecision(1)
                          << "  lbox " << w(tm_rsz)
                          << " (ip " << w(tm_ip) << ")"
                          << "  pre " << w(tm_pre)
                          << "  dpu " << w(tm_dpu_hw)
                          << " (cpu " << w(tm_ovl) << " / wait " << w(tm_wait) << ")"
                          << "  memcpy " << w(tm_dpu_cpu)
                          << "  post " << w(tm_post)
                          << "  draw " << std::setprecision(2)
                          << tm_draw.recent_avg() << "/" << tm_draw.recent_max()
                          << std::setprecision(1)
                          << "  out " << w(tm_out)
                          << "  | 延遲 " << w(tm_lat)
                          << "  過期 " << cam.stale()
                          << "  略過 " << cam.overwritten()
                          << "  壞幀 " << cam.badFrames() << std::endl;
            }
        }
    }

    g_running = false;
    cam.stop();

    const double wall = (fpipe::now_ms() - t_start) / 1000.0;
    cam.close();
    if (streamer) streamer->close();

    const double proc = tm_rsz.avg() + tm_pre.avg() + tm_dpu_hw.avg()
                      + tm_dpu_cpu.avg() + tm_post.avg() + tm_out.avg();
    const double hw = tm_wait.avg() + tm_ip.avg();      // 真正在等硬體的時間

    std::cout << "\n──────── 統計 ────────\n"
              << "處理 " << n_proc.load() << " 幀,"
              << "擷取 " << cam.frames() << " 幀,"
              << "推掉 V4L2 舊幀 " << cam.stale() << ",略過 "
              << cam.overwritten() << ",壞幀 " << cam.badFrames() << "\n\n"
              << std::fixed << std::setprecision(2)
              << "  擷取端 copy+clean   " << cam.avgPrepMs() << " ms  ("
              << (cam.zeroCopy() ? "DMABUF 零複製,只有 clean" : "MMAP,含一次複製")
              << ",在擷取執行緒)\n"
              << "  letterbox           " << tm_rsz.avg()
              << " ms  (其中 uyvy_resize IP " << tm_ip.avg() << " ms)\n"
              << "  前處理              " << tm_pre.avg()  << " ms\n"
              << "  DPU(submit→完成)   " << tm_dpu_hw.avg() << " ms\n"
              << "    其中 CPU 空檔工作 " << tm_ovl.avg() << " ms"
              << (need_bgr ? "(BGR 轉換" : "(無 BGR 轉換")
              << (prefetch_on ? " + 預存取)" : ",無預存取)") << "\n"
              << "    其中剩餘等待      " << tm_wait.avg() << " ms"
              << "  (接近 0 代表 CPU 工作比 DPU 久,會拖慢延遲)\n"
              << "  DPU 輸出 memcpy     " << tm_dpu_cpu.avg() << " ms"
              << (out_copy ? "  (kDpuOutCopy=true)" : "  (kDpuOutCopy=false,應接近 0)") << "\n"
              << "  後處理 + 追蹤       " << tm_post.avg()
              << " ms  (int8 直接解碼,無轉置 / fix2float)\n"
              << "  繪製 + 串流         " << tm_out.avg()
              << " ms  (其中純繪圖 " << tm_draw.avg() << " ms)\n"
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