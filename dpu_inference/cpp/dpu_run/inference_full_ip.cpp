// yolo_full.cpp — 兩條執行緒的低延遲版本(UYVY 直送 IP,IP 同時輸出 DPU 輸入)
//
// ─────────────────────────────────────────────────────────────
//  與前一版(hls_uyvy_resize)的差異
//
//  前一版:相機 UYVY → hls::uyvy_resize::letterbox(RGB)
//          → CPU norm_and_fix_letterbox(RGB → int8 定點)→ DPU
//
//  這一版:相機 UYVY → hls::uyvy_resize_quant::letterbox
//              (IP:UYVY→RGB + 2/3 倍縮小 + 量化,同時輸出 RGB 與 quant)
//              quant 由函式庫直接寫進 DPU 輸入緩衝(不經過中間的 lb.quant 再 copy)
//          → DPU
//          RGB(lb.img)仍保留,只用來繪圖 / 串流。
//
//  本版修正(quant 判斷):
//    - 函式庫的 quant 參考已改成與 IP 相同的 kLut(fix_point = uq::ip_fix_point())。
//      但 DPU 真正要的是 xmodel 的 fix_point,兩者可能不同,
//      所以唯一的正確答案仍是 norm_and_fix_letterbox(與 DPU 輸入相同的公式與 fix_point):
//        * 開機 verify:RGB 錯才關閉 IP;只有 quant 不符時只提示,改由第一幀比對決定
//        * 第一幀:IP quant 內容區 vs norm_and_fix,印出每通道錯誤數與範例;
//          不一致時自動掃 fix_point 0..10,指出 IP 實際相當於哪個 fix_point
//        * 黑邊:由 norm_and_fix 算出每通道的值(例:fix_point=6 → -128/-128/-115),
//          不再用 quant_byte(0)
//        * 有 CPU 零頭縮放(quant 不是 IP 一次到位)時,一律用 norm_and_fix,
//          不使用函式庫依 quant_byte() 算出的 quant
//    - 開機檢查 xmodel 的輸入 fix_point 是否等於 IP 編譯時的 fix_point(PIPE_IP_FIX)
//
//  本版修改(效能):
//    - IP 的 quant 直接寫進 DPU 輸入(uq::letterbox 的 quant_dst),省掉 lb.quant → DPU 輸入的複製;
//      不用 IP quant 時讓函式庫完全不產生 quant(want_quant = false)
//    - 黑邊不再每幀重填:函式庫與 norm_and_fix_letterbox 都只在 buffer / 內容區改變時填一次
//    - 開機多一項檢查:函式庫的量化表(= IP 的 ROM)與 norm_and_fix(xmodel 的 fix_point)768 項逐一比對,
//      一致才讓函式庫把 quant 寫進 DPU 輸入(包括有 CPU 零頭、CPU 退路時由 CPU 查同一張表算出的 quant)
//
//  環境變數(新增 / 變更):
//    PIPE_UYVY_PHYS     uyvy_resize_quant 控制暫存器實體位址(十六進位)
//    PIPE_IP_FIX        IP 的量化 LUT 編譯時的 fix_point(預設 = uq::ip_fix_point(),即函式庫的 kFixPoint)
//                       只有 bitstream 與函式庫不是同一版 quantize_lut.h 時才需要手動設定
//                       與 xmodel 輸入 fix_point 不同時不使用 IP 的 quant
//    PIPE_QUANT_CHECK   0 = 略過第一幀 quant vs norm_and_fix 比對(預設 1)
//                       開機 verify 的 quant 不符時,這個比對一定會執行
//    PIPE_FORCE_QUANT   1 = 強制使用 IP 的 quant 當 DPU 輸入(預設 0)
//                         強制模式下:
//                           - IP 不可用 / 尺寸不支援 IP / DPU 輸入型別或 fix_point 不符 → 直接結束
//                           - 開機 verify 未通過、第一幀 quant 比對不一致 → 只警告,仍使用 IP
//                           - 執行中 IP 失敗 → 不關閉 IP,每幀照常重試(只印前 5 次警告)
//                           - quant 不是 IP 一次到位(IP 沒跑 / 有 CPU 零頭)的幀:由函式庫用同一張表在 CPU 量化
//                             (開機已確認與 norm_and_fix 逐項相同)
//    其餘同前一版(PIPE_CAM_CORE / PIPE_V4L2_BUFS / PIPE_DMA_HEAP / PIPE_CLS_MEM /
//                 PIPE_PREFETCH / PIPE_STREAM_RGB / PIPE_STREAM_CLONE / PIPE_CHECK_LB)
// ─────────────────────────────────────────────────────────────

#include "modelrunner_pipe.h"
#include "tracker.h"
#include "stream.h"
#include "drawer.h"            // 需包含 drawer_tracking.h 的宣告
#include "draw_predictor.h"
#include "yolopproc.h"
#include "camera.h"            // 只用到 Camera::Config(CLI 參數)
#include "v4l2_dmabuf_camera.h"
#include "preproc.h"
#include "cli_args.h"
#include "frame_pipeline.h"
#include "hls_uyvy_resize_quant.h"
#include "norm_fix_letterbox.h"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
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

namespace uq = hls::uyvy_resize_quant;

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

inline void to_project_result(const uq::LetterboxResult& s, ResizeResult& d) {
    d.img     = s.img;                 // RGB uint8,繪圖 / 串流用
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

// 兩張同型別、同尺寸的 Mat 逐 byte 比較(支援 8U / 8S)。回傳不同的 byte 數,max_diff 為最大差。
size_t compare_bytes(const cv::Mat& a, const cv::Mat& b, int& max_diff) {
    max_diff = 0;
    if (a.size() != b.size() || a.type() != b.type()) { max_diff = 255; return a.total() * a.elemSize(); }
    cv::Mat a16, b16, d;
    a.reshape(1).convertTo(a16, CV_16S);
    b.reshape(1).convertTo(b16, CV_16S);
    cv::absdiff(a16, b16, d);
    double mx = 0;
    cv::minMaxLoc(d, nullptr, &mx);
    max_diff = static_cast<int>(mx);
    return static_cast<size_t>(cv::countNonZero(d));
}

// ─────────────────────────────────────────────────────────────
//  quant(CV_8SC3)比對 / 黑邊:正確答案一律以 norm_and_fix_letterbox 為準
// ─────────────────────────────────────────────────────────────

struct QuantDiff {
    size_t n = 0;                          // 不同的 byte 數
    size_t ch[3] = {0, 0, 0};              // R / G / B 各自的錯誤數
    int    max_diff = 0;
    struct Sample { int x, y, c, got, ref; };
    std::vector<Sample> samples;           // 前幾個錯誤(座標相對於比對區域)
};

// a、b:同尺寸的 CV_8SC3(可以是 ROI)
QuantDiff compare_quant(const cv::Mat& a, const cv::Mat& b, size_t max_samples = 5) {
    QuantDiff r;
    CV_Assert(a.size() == b.size() && a.type() == CV_8SC3 && b.type() == CV_8SC3);
    for (int y = 0; y < a.rows; ++y) {
        const schar* pa = a.ptr<schar>(y);
        const schar* pb = b.ptr<schar>(y);
        for (int i = 0; i < a.cols * 3; ++i) {
            const int d = std::abs(int(pa[i]) - int(pb[i]));
            if (!d) continue;
            ++r.n;
            ++r.ch[i % 3];
            r.max_diff = std::max(r.max_diff, d);
            if (r.samples.size() < max_samples)
                r.samples.push_back({i / 3, y, i % 3, int(pa[i]), int(pb[i])});
        }
    }
    return r;
}

// 黑色 (0,0,0) 經 norm_and_fix 後的值:與 DPU 輸入公式完全相同(每個通道不同)
cv::Scalar dpu_pad_value(int fix_point) {
    const cv::Mat black(1, 1, CV_8UC3, cv::Scalar::all(0));
    cv::Mat out;
    norm_and_fix_letterbox(black, fix_point, cv::Rect(0, 0, 0, 0), out);   // 內容區為空 → 整張都是黑邊
    const cv::Vec<schar, 3> v = out.at<cv::Vec<schar, 3>>(0, 0);
    return cv::Scalar(v[0], v[1], v[2]);
}

// letterbox 的內容區(與影像取交集)
inline cv::Rect content_rect(const uq::LetterboxResult& lb) {
    return cv::Rect(lb.content) & cv::Rect(0, 0, lb.img.cols, lb.img.rows);
}

// 函式庫的量化表(= IP 的 ROM)與 norm_and_fix_letterbox(fix_point)逐項比對,回傳不同的項數(0..768)
int lut_mismatch_vs_normfix(int fix_point) {
    cv::Mat img(1, 256, CV_8UC3);
    for (int v = 0; v < 256; ++v) img.at<cv::Vec3b>(0, v) = cv::Vec3b(v, v, v);
    cv::Mat out;
    norm_and_fix_letterbox(img, fix_point, cv::Rect(0, 0, 256, 1), out);
    int n = 0;
    for (int v = 0; v < 256; ++v)
        for (int c = 0; c < 3; ++c)
            if (out.at<cv::Vec<schar, 3>>(0, v)[c] != uq::quant_byte(static_cast<uint8_t>(v), c)) ++n;
    return n;
}

// 掃 fix_point 0..10,找出和 IP quant 最接近的那一個(用來判斷 IP 的 LUT 是用哪個 fix_point 編的)
int detect_ip_fix_point(const ResizeResult& rr, const cv::Mat& ip_quant, const cv::Rect& c,
                        size_t& best_err)
{
    int best = -1;
    best_err = SIZE_MAX;
    cv::Mat ref;
    for (int fp = 0; fp <= 10; ++fp) {
        norm_and_fix_letterbox(rr, fp, ref);
        const QuantDiff d = compare_quant(ip_quant(c), ref(c), 0);
        if (d.n < best_err) { best_err = d.n; best = fp; }
    }
    return best;
}

// 檢查 letterbox 是否每幀完整重寫內容區(RGB 與 quant 都要):
// 先後用兩種底色把整張(含黑邊)預填各做一次,內容區結果必須完全相同。
// 預填會蓋掉黑邊,所以每次都先 letterbox_invalidate_borders(),讓函式庫重填黑邊;
// 檢查結束時 lb(與 quant_dst)仍是正確的結果。
template <class Phys>
void check_letterbox_rewrite(const cv::Mat& uyvy, int in_w, uq::LetterboxResult& lb,
                             bool ip_enabled, Phys phys, cv::Mat* qdst, bool want_q)
{
    const cv::Rect c = content_rect(lb);
    if (c.empty()) { std::cout << "[check] letterbox 內容區為空,略過\n"; return; }

    auto run = [&](int rgb_fill, int q_fill) {
        lb.img.setTo(cv::Scalar::all(rgb_fill));
        if (want_q) lb.quant.setTo(cv::Scalar::all(q_fill));
        uq::letterbox_invalidate_borders();
        uq::letterbox(uyvy, in_w, lb, ip_enabled, phys, qdst, want_q);
    };

    run(255, 100);
    const cv::Mat a  = lb.img(c).clone();
    const cv::Mat qa = want_q ? lb.quant(c).clone() : cv::Mat();
    run(17, -3);

    int d1 = 0, d2 = 0;
    const size_t n1 = compare_bytes(a, lb.img(c), d1);
    const size_t n2 = want_q ? compare_bytes(qa, lb.quant(c), d2) : 0;
    std::cout << "[check] letterbox 內容區每幀完整重寫:RGB "
              << (n1 == 0 ? "是" : "否!") << "  quant "
              << (!want_q ? "(未產生)" : n2 == 0 ? "是" : "否!")
              << ((n1 || n2) ? "(請把 draw_area 縮小 1~2 px)" : "") << "\n";
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
    const bool check_quant  = env_int("PIPE_QUANT_CHECK", 1) != 0;
    const bool force_quant  = env_int("PIPE_FORCE_QUANT", 0) != 0;
    const int  ip_fix       = env_int("PIPE_IP_FIX", uq::ip_fix_point());   // 預設 = 函式庫編譯時的 kFixPoint
    if (force_quant)
        std::cout << "[強制 IP quant] PIPE_FORCE_QUANT=1:IP 一次到位的幀一律使用 IP 的 quant\n";
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

    // DPU 輸入的 fix_point:input_scale = 2^fix_point(Vitis AI 的輸入慣例)
    const double in_scale = engine.input_scale();
    const int in_fix = static_cast<int>(std::round(std::log2(in_scale)));
    if (in_fix <= 0)
        std::cerr << "[DPU 輸入] input_scale=" << in_scale << " → fix_point=" << in_fix
                  << ",不像是正常值;請確認 engine.input_scale() 回傳的是 2^fix_point\n";

    // 黑邊值:與 DPU 輸入公式相同,每個通道不同
    const cv::Scalar qpad = dpu_pad_value(in_fix);
    std::cout << "[DPU 輸入] 黑邊 (0,0,0) 量化後 = (" << qpad[0] << ", " << qpad[1]
              << ", " << qpad[2] << ")\n";

    // ---- DPU 輸入能不能直接吃 IP 的 quant ----
    // 需要:輸入是 CV_8SC3、尺寸 = 模型輸入、letterbox 是正方形(in_w == in_h),
    //       且 xmodel 輸入的 fix_point = IP 的 LUT 編譯時的 fix_point
    bool quant_usable = false;
    {
        const cv::Mat d0 = engine.input_mat(0);
        const bool fmt_ok = d0.type() == CV_8SC3 && d0.cols == in_w && d0.rows == in_h && in_w == in_h;
        const bool fix_ok = in_fix == ip_fix;
        const int  lut_bad = fix_ok ? lut_mismatch_vs_normfix(in_fix) : -1;
        const bool lut_ok  = lut_bad == 0;
        quant_usable = fmt_ok && fix_ok && lut_ok;
        std::cout << "[DPU 輸入] fix_point=" << in_fix << "(IP 為 " << ip_fix << ")  input_mat 型別="
                  << d0.type() << " " << d0.cols << "x" << d0.rows;
        if (quant_usable)
            std::cout << "  → 可直接使用 IP 的 quant\n";
        else if (!fmt_ok)
            std::cout << "  → 型別 / 尺寸不符,維持 CPU norm_and_fix\n";
        else if (!fix_ok)
            std::cout << "  → fix_point 不同,IP 的 quant 不能用,維持 CPU norm_and_fix\n"
                      << "             (請用 kFixPoint=" << in_fix << " 重新合成 IP,或設定 PIPE_IP_FIX)\n";
        else
            std::cout << "  → 函式庫量化表與 norm_and_fix 有 " << lut_bad
                      << "/768 項不同,維持 CPU norm_and_fix\n";
        if (quant_usable)
            std::cout << "[DPU 輸入] 函式庫量化表 vs norm_and_fix(fix_point=" << in_fix
                      << "):768 項完全相同;IP 的 quant 直接寫入 DPU 輸入\n";
    }
    if (force_quant && !quant_usable) {
        std::cerr << "[強制 IP quant] DPU 輸入不是 CV_8SC3 " << in_w << "x" << in_h
                  << "(或非正方形、fix_point " << in_fix << " ≠ IP 的 " << ip_fix
                  << "、或量化表不一致),無法直接使用 IP 的 quant,結束\n";
        return;
    }
    bool quant_trusted = quant_usable;     // 第一個 IP 幀比對失敗會被改成 false(強制模式除外)
    bool quant_checked = !check_quant;

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
                  << " 改為 UYVY(uyvy_resize_quant IP 只吃 UYVY)\n";

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
    const uq::Plan plan = uq::plan_for(
        cam_w, cam_h,
        static_cast<uint32_t>(std::round(cam_w * r0)),
        static_cast<uint32_t>(std::round(cam_h * r0)));
    const int sc = plan.use_ip ? static_cast<int>(plan.scale) : 1;
    if (force_quant && !plan.use_ip) {
        std::cerr << "[強制 IP quant] " << cam_w << "x" << cam_h << " -> " << in_w
                  << " 無法使用 IP:" << plan.reason << ",結束\n";
        return;
    }

    // ---- DMA pool(IP 現在有兩份輸出:RGB + quant)----
    const size_t ip_out = static_cast<size_t>(cam_w / sc) * (cam_h / sc) * 3;
    const size_t need =
          static_cast<size_t>(cam_w) * cam_h * 2                // verify 的 staging
        + ip_out * 2                                            // IP 輸出:RGB + quant
        + 8u * 1024 * 1024;
    hls::use_dma_heap("auto", ((need >> 20) + 16) << 20);

    // ---- uyvy_resize_quant IP ----
    if (const char* e = std::getenv("PIPE_UYVY_PHYS"))
        uq::use_devmem(std::strtoull(e, nullptr, 16));

    std::cout << "[letterbox] " << cam_w << "x" << cam_h << " -> " << in_w
              << "x" << in_w << ":" << plan.reason;
    if (plan.use_ip)
        std::cout << "(" << plan.scale << " 倍 -> " << plan.params.out_w
                  << "x" << plan.params.out_h << ")";
    std::cout << "\n";

    // IP 是否啟用:開機檢查 RGB 沒過、或執行中出錯,就關掉,之後每幀直接走 CPU
    bool ip_enabled = plan.use_ip;
    if (plan.use_ip) {
        if (!uq::available()) {
            std::cerr << "[uyvy_resize_quant] 不可用:" << uq::last_error()
                      << "\n             裝置:" << uq::device_info();
            if (force_quant) {
                std::cerr << "\n             強制 IP quant 模式,結束\n";
                return;
            }
            std::cerr << "\n             letterbox 改用 CPU\n";
            ip_enabled = false;
        } else {
            const auto rep = uq::verify(uq::make_test_uyvy(cam_w, cam_h),
                                        static_cast<int>(plan.scale));
            std::cout << "[uyvy_resize_quant] " << uq::device_info()
                      << "  verify: " << rep.reason << "\n";

            // RGB 不符 → 關閉 IP;只有 quant 不符 → 不立刻關閉,改由第一幀和 norm_and_fix 比對決定
            //(函式庫的 quant 參考是 IP 編譯時的 kFixPoint,不一定等於 xmodel 的 fix_point)
            const bool rgb_ok = rep.ran && rep.mismatches == 0 && rep.pad_dirty == 0;
            const bool q_ok   = rep.ran && rep.q_mismatches == 0 && rep.q_pad_dirty == 0;

            if (!rgb_ok) {
                std::cerr << "[uyvy_resize_quant] verify 的 RGB 未通過";
                if (rep.ran)
                    std::cerr << ":\n    RGB   不符 " << rep.mismatches << "/" << rep.total
                              << "  R/G/B 錯 " << rep.ch_err[0] << "/" << rep.ch_err[1]
                              << "/" << rep.ch_err[2] << "  最大差 " << rep.max_diff
                              << "  補位非 0 " << rep.pad_dirty;
                if (force_quant) {
                    std::cerr << "\n             強制 IP quant 模式:仍使用 IP(結果可能不正確)\n";
                } else {
                    std::cerr << "\n             letterbox 改用 CPU\n";
                    ip_enabled = false;
                }
            } else if (!q_ok) {
                std::cout << "[uyvy_resize_quant] RGB 正確;quant 與函式庫參考(kLut)不符 "
                          << rep.q_mismatches << "/" << rep.q_total
                          << "(R/G/B 錯 " << rep.q_ch_err[0] << "/" << rep.q_ch_err[1]
                          << "/" << rep.q_ch_err[2] << ",最大差 " << rep.q_max_diff << ")\n"
                          << "             原因:" << rep.reason << "\n"
                          << "             IP 照常使用,quant 是否可用改由第一幀和 norm_and_fix 比對決定\n";
                quant_checked = false;     // 不管 PIPE_QUANT_CHECK,一定要比對一次
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
    // tm_pre 是 CPU norm_and_fix(退路);IP 路徑的 quant 已在 letterbox(tm_rsz)內寫入 DPU 輸入
    // tm_dpu_hw = submit → DPU 完成(包含 DPU 期間做的 CPU 工作)
    //   tm_ovl  = DPU 期間的 CPU 工作(BGR 轉換 + 預存取)
    //   tm_wait = CPU 工作做完後,還要再等 DPU 多久
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
        uq::LetterboxResult lb;                   // 迴圈外重用,lb.img / lb.quant 不會每幀重新配置
        cv::Mat bgr_out;                          // 需要 BGR 時才使用,迴圈外重用
        std::vector<bytetrack::Box> boxes;
        long long frames = 0, prev = 0;
        double t_prev = fpipe::now_ms();
        fpipe::Ema fps_ema(0.3);
        bool first = true;
        bool warned_copy = false;
        int  ip_fail_msgs = 0;

        while (g_running) {
            DmabufCamera::Frame f = cam.acquire(200);
            if (!f.valid()) continue;
            const double t_cap_stamp = f.timestamp();
            const cv::Mat& uyvy = f.mat();              // 就在 DMA 記憶體裡

            // ---- letterbox:UYVY 進,RGB 正方形出;quant 直接寫進 DPU 輸入 ----
            //   ip_quant = true :quant(內容區 + 黑邊)由函式庫直接寫進 dpu_in,lb.quant 與 dpu_in 共用記憶體
            //   ip_quant = false:函式庫不產生 quant,下面用 norm_and_fix 寫 dpu_in
            //   黑邊只在第一幀(或內容區改變時)寫入,之後每幀只寫內容區
            cv::Mat dpu_in = engine.input_mat(0);
            const bool ip_quant = quant_trusted;
            cv::Mat* qdst = ip_quant ? &dpu_in : nullptr;

            double t0 = fpipe::now_ms();
            uq::letterbox(uyvy, in_w, lb, ip_enabled, f.phys(), qdst, ip_quant);
            if (check_lb && frames == 1)
                check_letterbox_rewrite(uyvy, in_w, lb, ip_enabled, f.phys(), qdst, ip_quant);
            to_project_result(lb, rr);
            tm_rsz.add(fpipe::now_ms() - t0);

            // UYVY 已經讀完(lb.img / DPU 輸入是另一塊記憶體),立刻歸還緩衝給擷取端
            f.reset();

            if (lb.used_ip) {
                tm_ip.add(lb.ip_ms);
                if (!lb.zero_copy && !warned_copy) {
                    warned_copy = true;
                    std::cerr << "[uyvy_resize_quant] 輸入沒有走實體位址,每幀多一次複製 "
                              << std::fixed << std::setprecision(2) << lb.timing.copy_ms
                              << " ms\n";
                }
            } else if (ip_enabled) {
                if (!force_quant) ip_enabled = false;
                if (!force_quant || ip_fail_msgs++ < 5)
                    std::cerr << "[uyvy_resize_quant] 執行失敗,"
                              << (force_quant ? "強制 IP quant 模式:不關閉 IP,下一幀繼續重試"
                                              : "之後改用 CPU") << "\n"
                              << "             原因:" << uq::last_error() << "\n"
                              << "             裝置:" << uq::device_info() << "\n"
                              << "             花費:" << std::fixed << std::setprecision(1)
                              << lb.ip_ms << " ms\n";
            }

            // ---- 前處理:IP 路徑的 quant 已在 DPU 輸入裡;否則用 CPU norm_and_fix ----
            t0 = fpipe::now_ms();

            // 第一個「IP 一次到位」的幀:DPU 輸入的內容區(IP 的 quant)和 norm_and_fix(同一張 RGB)逐 byte 比對
            const bool ip_exact = lb.used_ip &&
                             lb.mid_w == lb.content.width && lb.mid_h == lb.content.height;
            const cv::Rect qc = content_rect(lb);
            if (ip_quant && ip_exact && !quant_checked) {
                cv::Mat ref;
                norm_and_fix_letterbox(rr, in_fix, ref);
                const QuantDiff d = compare_quant(dpu_in(qc), ref(qc));
                std::cout << "[check] IP quant vs CPU norm_and_fix(內容區 " << qc.width << "x"
                          << qc.height << ",fix_point=" << in_fix << "):不符 " << d.n << "/"
                          << static_cast<size_t>(qc.area()) * 3
                          << "  R/G/B 錯 " << d.ch[0] << "/" << d.ch[1] << "/" << d.ch[2]
                          << "  最大差 " << d.max_diff
                          << (d.n == 0 ? "  OK,之後直接使用 IP 的 quant"
                                       : force_quant ? "  不一致!強制模式,仍使用 IP 的 quant"
                                                     : "  不一致!之後改用 CPU norm_and_fix") << "\n";
                if (d.n != 0) {
                    for (const auto& s : d.samples) {
                        const int px = qc.x + s.x, py = qc.y + s.y;
                        const cv::Vec3b in = rr.img.at<cv::Vec3b>(py, px);
                        std::cerr << "    (" << px << "," << py << ") " << "RGB"[s.c]
                                  << "  輸入=" << int(in[s.c]) << "  IP=" << s.got
                                  << "  正確=" << s.ref << "\n";
                    }
                    size_t best_err = 0;
                    const int fp = detect_ip_fix_point(rr, dpu_in, qc, best_err);
                    std::cerr << "    → IP 的 quant 最接近 fix_point=" << fp << "(不符 " << best_err
                              << ")。";
                    if (best_err == 0 && fp != in_fix)
                        std::cerr << "IP 的 LUT 是用 fix_point=" << fp << " 編的,xmodel 要 " << in_fix
                                  << ":請用 kFixPoint=" << in_fix << " 重新合成\n";
                    else if (best_err == 0)
                        std::cerr << "\n";
                    else if (d.ch[1] == 0 && d.ch[0] && d.ch[2])
                        std::cerr << "只有 R/B 錯、G 正確:IP 的 R/B 查表對調(檢查 quantize_core 的位元位置)\n";
                    else
                        std::cerr << "沒有任何 fix_point 完全吻合:檢查板子上的 bitstream 是否為最新版\n";
                    if (!force_quant) quant_trusted = false;   // 這一幀起改用 norm_and_fix
                }
                quant_checked = true;
            }

            // IP quant 不可用(或剛判定不可用):整張由 norm_and_fix 寫入(它也只在第一次填黑邊)
            if (!quant_trusted)
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
                else                        cv::cvtColor(rr.img, frame0, cv::COLOR_RGB2BGR);

                cv::imwrite("frame0.png", frame0);
                frame0.release();

                first = false;
            }

            // ---- 繪製 + 串流:就地、限定在內容區,沒有 clone、沒有整張 cvtColor ----
            // 繪圖只動 rr.img(RGB),不影響 lb.quant / DPU 輸入
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
              << " ms  (其中 uyvy_resize_quant IP " << tm_ip.avg() << " ms)\n"
              << "  前處理              " << tm_pre.avg()  << " ms  ("
              << (quant_trusted ? (force_quant ? "IP quant 已在 letterbox 直接寫入 DPU 輸入,強制模式"
                                               : "IP quant 已在 letterbox 直接寫入 DPU 輸入,應接近 0")
                                : "CPU norm_and_fix") << ")\n"
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