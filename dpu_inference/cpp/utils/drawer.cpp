// ============================================================================
//  Tracking 繪圖（取代 drawer.cpp 中原本的 draw_tracks 與 draw_tracking）
//
//  - 函式名稱與參數和原版相同，主程式不用改
//  - 不 clone：直接畫在輸入影像的像素上（輸入影像的內容會被改變）
//  - 可用 set_draw_prefetch() 開關預載入
// ============================================================================
#include <drawer.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

// 單次預載入的位元組上限，可在編譯時以 -DDRAWER_PREFETCH_BUDGET=... 覆寫
// KV260 的 L2 為 1 MB（4 核共用），需保留空間給後處理與其他資料
#ifndef DRAWER_PREFETCH_BUDGET
#define DRAWER_PREFETCH_BUDGET (640u << 10)
#endif

#ifndef DRAWER_TOUCH_HOOK
#define DRAWER_TOUCH_HOOK(p) ((void)0)   // 單元測試用，可記錄被預載入的位址
#endif

namespace {

// ---- 繪圖參數（與原版相同） ----
constexpr int    kFont       = cv::FONT_HERSHEY_SIMPLEX;
constexpr double kLabelScale = 0.5;
constexpr int    kLabelThick = 1;
constexpr int    kBoxThick   = 2;
constexpr double kFpsScale   = 0.8;
constexpr int    kFpsThick   = 2;
const cv::Point  kFpsOrg(10, 30);

// ---- 預載入參數（KV260：L2 1 MB 共用） ----
constexpr int    kLine   = 64;          // cache line 大小
constexpr int    kMargin = 4;           // 預測誤差容許範圍（px）
constexpr size_t kBudget = DRAWER_PREFETCH_BUDGET;

std::atomic<DrawPrefetch> g_prefetch{DrawPrefetch::Off};
std::vector<std::string>  g_class_names;
volatile unsigned         g_touch_sink;   // 防止預載入的讀取被最佳化掉

inline cv::Scalar track_color(int id)
{
    return cv::Scalar((id * 67  + 100) % 255,
                      (id * 113 +  50) % 255,
                      (id * 179 + 150) % 255);
}

// 與原版字串完全一致："ID<id> <name>: " + std::to_string(score).substr(0, 4)
// 寫入 out，重複使用其容量；長度不受限制
inline void format_label(std::string& out, int id, int cls, float score,
                         const std::vector<std::string>& names)
{
    char sc[64];
    std::snprintf(sc, sizeof(sc), "%f", score);           // 同 std::to_string
    const bool known = cls >= 0 && static_cast<size_t>(cls) < names.size();
    auto fmt = [&](char* b, size_t n) {
        return known ? std::snprintf(b, n, "ID%d %s: %.4s", id, names[cls].c_str(), sc)
                     : std::snprintf(b, n, "ID%d Class %d: %.4s", id, cls, sc);
    };
    out.resize(std::max<size_t>(out.capacity(), 63));
    const int len = fmt(out.data(), out.size() + 1);
    if (len < 0) { out.clear(); return; }
    if (static_cast<size_t>(len) > out.size()) {
        out.resize(static_cast<size_t>(len));
        fmt(out.data(), out.size() + 1);
    } else {
        out.resize(static_cast<size_t>(len));
    }
}

// 框貼在畫面頂端時，標籤翻到框內側
inline int label_top_of(int y1, int text_h)
{
    return (y1 - text_h - 6 >= 0) ? y1 - text_h - 6 : y1;
}

// ---------------------------------------------------------------------------
//  就地繪製
// ---------------------------------------------------------------------------
void draw_tracks_impl(cv::Mat&                             img,
                      const std::vector<bytetrack::Track>& tracks,
                      const std::vector<std::string>&      names)
{
    thread_local std::string label;      // 重複使用，第一次之後不再配置

    for (const bytetrack::Track& t : tracks) {
        const int x1 = static_cast<int>(t.x1), y1 = static_cast<int>(t.y1);
        const int x2 = static_cast<int>(t.x2), y2 = static_cast<int>(t.y2);
        const cv::Scalar color = track_color(t.track_id);

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, kBoxThick);

        format_label(label, t.track_id, t.cls, t.score, names);

        int baseline = 0;
        const cv::Size ts = cv::getTextSize(label, kFont, kLabelScale, kLabelThick, &baseline);
        const int top = label_top_of(y1, ts.height);

        cv::rectangle(img, cv::Point(x1, top),
                      cv::Point(x1 + ts.width, top + ts.height + 6),
                      color, cv::FILLED);
        cv::putText(img, label, cv::Point(x1, top + ts.height + 1),
                    kFont, kLabelScale, cv::Scalar(255, 255, 255), kLabelThick);
    }
}

// ---------------------------------------------------------------------------
//  預載入
// ---------------------------------------------------------------------------
template <bool ForWrite>
struct Toucher {
    cv::Mat& img;
    size_t   used = 0;
    unsigned sink = 0;

    explicit Toucher(cv::Mat& m) : img(m) {}
    ~Toucher() { g_touch_sink = sink; }

    inline void touch(uchar* p)
    {
        DRAWER_TOUCH_HOOK(p);
        if constexpr (ForWrite) { volatile uchar* v = p; *v = *v; }
        else                    { sink += *p; }
    }

    // 載入第 y 列 [xa, xb] 所在的 cache line，每條只碰一次、不越界
    bool span(int y, int xa, int xb)
    {
        if (used >= kBudget) return false;
        if (y < 0 || y >= img.rows) return true;
        xa = std::max(xa, 0);
        xb = std::min(xb, img.cols - 1);
        if (xa > xb) return true;

        const int       es  = static_cast<int>(img.elemSize());
        uchar*          row = img.ptr<uchar>(y);
        uchar*          s   = row + xa * es;
        const uintptr_t sa  = reinterpret_cast<uintptr_t>(s);
        const uintptr_t ea  = reinterpret_cast<uintptr_t>(row + xb * es + es - 1);

        touch(s);
        for (uintptr_t a = (sa | (kLine - 1)) + 1; a <= ea; a += kLine)
            touch(reinterpret_cast<uchar*>(a));

        used += ((ea / kLine) - (sa / kLine) + 1) * kLine;
        return true;
    }

    bool rect(int xa, int ya, int xb, int yb)
    {
        ya = std::max(ya, 0);
        yb = std::min(yb, img.rows - 1);
        for (int y = ya; y <= yb; ++y)
            if (!span(y, xa, xb)) return false;
        return true;
    }
};

template <bool ForWrite>
void prefetch_impl(cv::Mat&                             img,
                   const std::vector<bytetrack::Track>& pred,
                   const std::vector<std::string>&      names)
{
    Toucher<ForWrite> T(img);

    // FPS 文字：位置固定，最優先
    static int      fps_bl = 0;
    static cv::Size fps_ts = cv::getTextSize("FPS: 999.99   Tracks: 999",
                                             kFont, kFpsScale, kFpsThick, &fps_bl);
    if (!T.rect(kFpsOrg.x - 2, kFpsOrg.y - fps_ts.height - 2,
                kFpsOrg.x + fps_ts.width + 2, kFpsOrg.y + fps_bl + 2)) return;

    const int e = kBoxThick / 2 + 1 + kMargin;     // 邊線涵蓋範圍
    thread_local std::string label;

    // 預測框可能發散或為 NaN：非有限值跳過，其餘限制在影像外一小段範圍內，
    // 避免整數溢位與空轉（超出影像的部分本來就不會被畫到）
    const float pad = 64.f;
    auto to_px = [&](float v, int dim, int& out) {
        if (!std::isfinite(v)) return false;
        out = static_cast<int>(std::clamp(v, -pad, static_cast<float>(dim) + pad));
        return true;
    };

    for (const bytetrack::Track& p : pred) {
        int x1, y1, x2, y2;
        if (!to_px(p.x1, img.cols, x1) || !to_px(p.y1, img.rows, y1) ||
            !to_px(p.x2, img.cols, x2) || !to_px(p.y2, img.rows, y2)) continue;
        if (x2 <= x1 || y2 <= y1) continue;

        // 標籤：用與繪圖相同的字串計算實際寬度
        format_label(label, p.track_id, p.cls, p.score, names);
        int bl = 0;
        const cv::Size ts = cv::getTextSize(label, kFont, kLabelScale, kLabelThick, &bl);
        const int top = label_top_of(y1, ts.height);
        if (!T.rect(x1 - kMargin, top - kMargin,
                    x1 + ts.width + kMargin, top + ts.height + 6 + kMargin)) return;

        // 上下邊
        if (!T.rect(x1 - e, y1 - e, x2 + e, y1 + e)) return;
        if (!T.rect(x1 - e, y2 - e, x2 + e, y2 + e)) return;

        // 左右邊（上下邊已涵蓋的列跳過）
        const int ya = std::max(y1 + e + 1, 0);
        const int yb = std::min(y2 - e, img.rows);
        for (int y = ya; y < yb; ++y) {
            if (!T.span(y, x1 - e, x1 + e)) return;
            if (!T.span(y, x2 - e, x2 + e)) return;
        }
    }
}

} // namespace

// ============================================================================
//  設定
// ============================================================================
void set_draw_prefetch(DrawPrefetch mode)
{
    g_prefetch.store(mode, std::memory_order_relaxed);
}

DrawPrefetch get_draw_prefetch()
{
    return g_prefetch.load(std::memory_order_relaxed);
}

void set_draw_class_names(const std::vector<std::string>& names)
{
    g_class_names = names;
}

void prefetch_track_regions(cv::Mat& frame,
                            const std::vector<bytetrack::Track>& predicted)
{
    if (frame.empty()) return;
    switch (g_prefetch.load(std::memory_order_relaxed)) {
        case DrawPrefetch::Off:   return;
        case DrawPrefetch::Read:  prefetch_impl<false>(frame, predicted, g_class_names); break;
        case DrawPrefetch::Write: prefetch_impl<true >(frame, predicted, g_class_names); break;
    }
}

// ============================================================================
//  與原版相同名稱與參數的介面
// ============================================================================

// 原版回傳 clone；此版回傳與 img 共用像素的 Mat，img 本身的內容會被畫上
cv::Mat draw_tracks(const cv::Mat&                       img,
                    const std::vector<bytetrack::Track>& tracks,
                    const std::vector<std::string>&      class_names)
{
    cv::Mat out = img;                   // 只複製 header，不複製像素
    draw_tracks_impl(out, tracks, class_names);
    return out;
}

// 原版 out 是 in 的複本；此版 out 與 in 共用像素，主程式照常使用 out 即可
// void draw_tracking(const cv::Mat&                       in,
//                    cv::Mat&                             out,
//                    const std::vector<bytetrack::Track>& tracks,
//                    double                               fps)
// {
//     out = in;                            // 只複製 header，不複製像素
//     if (!tracks.empty())
//         draw_tracks_impl(out, tracks, g_class_names);

//     char buf[64];
//     std::snprintf(buf, sizeof(buf), "FPS: %.2f   Tracks: %zu", fps, tracks.size());
//     cv::putText(out, buf, kFpsOrg, kFont, kFpsScale, cv::Scalar(0, 255, 0), kFpsThick);
// }

void draw_tracking(const cv::Mat& in,
                   cv::Mat& out,
                   const std::vector<bytetrack::Track>& tracks,
                   double fps)
{
    // Use the latest camera frame.
    out = in;

    // Draw tracking boxes.
    if (!tracks.empty())
        draw_tracks_impl(out, tracks, g_class_names);

    // 用來存放快取 Box 資訊的結構
    struct BoxInfo {
        cv::Mat mat;         // 整個黑底白字的影像區塊
        cv::Rect val_rect;   // 只有「數字部分」的範圍 (用來局部塗黑)
        int text_y;          // 繪製文字的基準 Y 座標
    };

    static BoxInfo fps_info;
    static BoxInfo tracks_info;

    static double cached_fps = -1.0;
    static size_t cached_tracks = static_cast<size_t>(-1);
    static bool initialized = false;

    const int padding = 8;

    // 只在第一次執行時分配與初始化空間
    if (!initialized)
    {
        // 建立預分配 Box 的 Lambda
        auto init_box = [&](const std::string& title, const std::string& dummy_val, BoxInfo& info)
        {
            int base_t = 0, base_v = 0;
            cv::Size t_size = cv::getTextSize(title, kFont, kFpsScale, kFpsThick, &base_t);
            cv::Size v_size = cv::getTextSize(dummy_val, kFont, kFpsScale, kFpsThick, &base_v);

            int max_h = std::max(t_size.height, v_size.height);
            int max_base = std::max(base_t, base_v);
            info.text_y = padding + max_h;

            // 1. 一次性分配足夠大的空間 (Title寬度 + 預留的最大數字寬度)
            info.mat.create(
                max_h + max_base + padding * 2,
                t_size.width + v_size.width + padding * 2,
                CV_8UC3
            );
            info.mat = cv::Scalar(0, 0, 0); // 填滿黑色底

            // 2. 畫上永遠不會變動的 Title (例如 "FPS: ")
            cv::putText(
                info.mat,
                title,
                cv::Point(padding, info.text_y),
                kFont, kFpsScale, cv::Scalar(255, 255, 255), kFpsThick, cv::LINE_AA
            );

            // 3. 紀錄「數值區塊」的 ROI (X起點在 title 後面，一直到 box 邊緣)
            info.val_rect = cv::Rect(
                padding + t_size.width, 
                0, 
                info.mat.cols - (padding + t_size.width), 
                info.mat.rows
            );
        };

        // 預留足夠寬的 dummy_val 以確保空間夠大不溢位
        init_box("FPS: ", "88.88", fps_info);
        init_box("Tracks: ", "88", tracks_info);
        
        initialized = true;
    }

    // Update FPS box
    const double displayed_fps = std::round(fps * 100.0) / 100.0;
    if (displayed_fps != cached_fps)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.2f", displayed_fps);

        // 只把數字區塊清空 (塗黑)
        fps_info.mat(fps_info.val_rect) = cv::Scalar(0, 0, 0);

        // 在數字區塊畫上新數字
        cv::putText(
            fps_info.mat,
            buf,
            cv::Point(fps_info.val_rect.x, fps_info.text_y),
            kFont, kFpsScale, cv::Scalar(255, 255, 255), kFpsThick, cv::LINE_AA
        );

        cached_fps = displayed_fps;
    }

    // Update Tracks box
    if (tracks.size() != cached_tracks)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%zu", tracks.size());

        // 只把數字區塊清空 (塗黑)
        tracks_info.mat(tracks_info.val_rect) = cv::Scalar(0, 0, 0);

        // 在數字區塊畫上新數字
        cv::putText(
            tracks_info.mat,
            buf,
            cv::Point(tracks_info.val_rect.x, tracks_info.text_y),
            kFont, kFpsScale, cv::Scalar(255, 255, 255), kFpsThick, cv::LINE_AA
        );

        cached_tracks = tracks.size();
    }

    // Paste both cached boxes side by side at the top-left corner.
    const int margin = 8;
    const int gap = 4;

    const int fps_x = margin;
    const int fps_y = margin;

    const int tracks_x = fps_x + fps_info.mat.cols + gap;
    const int tracks_y = margin;

    // Paste FPS box.
    if (fps_x + fps_info.mat.cols <= out.cols &&
        fps_y + fps_info.mat.rows <= out.rows)
    {
        fps_info.mat.copyTo(
            out(cv::Rect(
                fps_x, fps_y,
                fps_info.mat.cols, fps_info.mat.rows
            ))
        );
    }

    // Paste Tracks box on the same row.
    if (tracks_x + tracks_info.mat.cols <= out.cols &&
        tracks_y + tracks_info.mat.rows <= out.rows)
    {
        tracks_info.mat.copyTo(
            out(cv::Rect(
                tracks_x, tracks_y,
                tracks_info.mat.cols, tracks_info.mat.rows
            ))
        );
    }
}

// ============================================================================
//  限定範圍版本：以 ROI 繪圖，ROI 以外的像素完全不會被寫入
// ============================================================================
namespace {
// 把 tracks 的座標平移成 ROI 座標；重複使用同一塊 vector 容量
const std::vector<bytetrack::Track>&
shift_tracks(const std::vector<bytetrack::Track>& src, const cv::Rect& r)
{
    thread_local std::vector<bytetrack::Track> dst;
    dst.assign(src.begin(), src.end());
    const float dx = static_cast<float>(r.x), dy = static_cast<float>(r.y);
    for (bytetrack::Track& t : dst) { t.x1 -= dx; t.x2 -= dx; t.y1 -= dy; t.y2 -= dy; }
    return dst;
}
}  // namespace

void draw_tracking_in(cv::Mat& img, const cv::Rect& area,
                      const std::vector<bytetrack::Track>& tracks, double fps)
{
    const cv::Rect r = area & cv::Rect(0, 0, img.cols, img.rows);
    if (r.empty()) return;
    cv::Mat roi = img(r);                       // 只是 header，不複製像素
    draw_tracking(roi, roi, shift_tracks(tracks, r), fps);
}

void prefetch_track_regions_in(cv::Mat& img, const cv::Rect& area,
                               const std::vector<bytetrack::Track>& predicted)
{
    const cv::Rect r = area & cv::Rect(0, 0, img.cols, img.rows);
    if (r.empty()) return;
    cv::Mat roi = img(r);
    prefetch_track_regions(roi, shift_tracks(predicted, r));
}


// ============================================================================
//  Box 座標轉換
// ============================================================================

void scale_detections(const DetectionBatch& detections,
                      std::vector<bytetrack::Box> &out_boxes,
                      const ResizeResult&   resize_inf,
                      cv::Size              orig_shape)
{
    out_boxes.clear();                       // 先清空，避免提前 return 時殘留舊資料
    if (detections.count <= 0) return ;
    out_boxes.reserve(static_cast<size_t>(detections.count));

    const float inv_rx = 1.f / resize_inf.ratio.first;
    const float inv_ry = 1.f / resize_inf.ratio.second;
    const float dw     = resize_inf.pad.first;
    const float dh     = resize_inf.pad.second;
    const float w_max  = static_cast<float>(orig_shape.width);
    const float h_max  = static_cast<float>(orig_shape.height);

    for (int i = 0; i < detections.count; ++i) {
        const Detection& d = detections.data[i];
        bytetrack::Box b;
        b.x1    = std::clamp((d.x1 - dw) * inv_rx, 0.f, w_max);
        b.y1    = std::clamp((d.y1 - dh) * inv_ry, 0.f, h_max);
        b.x2    = std::clamp((d.x2 - dw) * inv_rx, 0.f, w_max);
        b.y2    = std::clamp((d.y2 - dh) * inv_ry, 0.f, h_max);
        b.score = d.score;
        b.cls   = d.class_id;
        out_boxes.push_back(b);
    }
}

void map_detections(const DetectionBatch& det,
                    std::vector<bytetrack::Box>& out,
                    float off_x, float off_y,   // 要扣掉的 letterbox padding
                    float sx, float sy,         // 額外縮放(不縮就傳 1)
                    cv::Size dst)               // 目標座標系,用來 clamp
{
    out.clear();
    if (det.count <= 0) return;
    out.reserve(static_cast<size_t>(det.count));

    const float w_max = dst.width  - 1.f;
    const float h_max = dst.height - 1.f;

    for (int i = 0; i < det.count; ++i) {
        const Detection& d = det.data[i];
        bytetrack::Box b;
        b.x1 = std::clamp((d.x1 - off_x) * sx, 0.f, w_max);
        b.y1 = std::clamp((d.y1 - off_y) * sy, 0.f, h_max);
        b.x2 = std::clamp((d.x2 - off_x) * sx, 0.f, w_max);
        b.y2 = std::clamp((d.y2 - off_y) * sy, 0.f, h_max);
        b.score = d.score;
        b.cls   = d.class_id;
        out.push_back(b);
    }
}