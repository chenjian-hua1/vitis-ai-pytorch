#ifndef BYTETRACK_H
#define BYTETRACK_H

// ============================================================================
//  ByteTrack (Zhang et al., ECCV 2022) — C++ 實作（時間戳版本）
//
//  設計重點：
//    * 不依賴 OpenCV / Eigen / OpenMP，只用 std（方便丟到 ARM / DPU 板子上）
//    * Kalman filter 為手寫 8 維狀態 (cx, cy, a, h, vcx, vcy, va, vh)，
//      矩陣運算針對 constant-velocity 的稀疏結構展開，沒有通用 matmul
//    * 時間以「秒」驅動：dt 由呼叫端傳入的時間戳算出，幀率飄動不會影響
//      預測距離；lost 的容忍度也直接以秒表示 (max_lost_seconds)
//    * 指派用 Jonker-Volgenant / e-maxx O(n^2 m) 版本，n,m 都是幾十個等級
//    * 所有 track 存在單一 std::vector，關聯階段只用 index，
//      不會有指標失效的問題
//
//  用法：
//      bytetrack::BYTETracker tracker;
//      const auto& tracks = tracker.update(boxes, capture_time_sec);
//      for (const auto& t : tracks) { t.track_id, t.x1 ... }
//
//  timestamp 省略（或給負數）時退回「每次呼叫 = 1 幀」的固定 dt 行為。
// ============================================================================

#include <utility>
#include <vector>

// Detection / DetectionBatch 的 adapter。若你的 util.h 欄位名不同，
// 只需要改 bytetrack.cpp 最上方 det_to_box() 那一個函式；
// 或是編譯時定義 BYTETRACK_NO_UTIL_ADAPTER 完全關掉這層。
#ifndef BYTETRACK_NO_UTIL_ADAPTER
#include <data_struct.h>
#endif

namespace bytetrack {

// ---------------------------------------------------------------------------
//  參數
// ---------------------------------------------------------------------------
struct Params {
    float track_thresh      = 0.50f;  // high / low detection 的分界
    float high_thresh       = 0.60f;  // 產生「新軌跡」所需的最低分數
    float low_thresh        = 0.10f;  // 低於此分數直接丟掉
    float match_thresh      = 0.80f;  // 第一階段 cost 上限 (cost = 1 - IoU)
    float match_thresh_low  = 0.50f;  // 第二階段（低分框救回）
    float match_thresh_new  = 0.70f;  // 第三階段（未確認軌跡）

    // ── 時間相關 ────────────────────────────────────────────────────────
    float max_lost_seconds  = 1.0f;   // 目標消失多久後放棄（秒）
    float nominal_fps       = 30.f;   // dt 正規化基準：dt = Δt * nominal_fps
                                      // 讓 dt 平常落在 1.0 附近，Kalman 的
                                      // 噪聲常數（調在「每幀」尺度）不用重調
    float dt_min            = 0.25f;  // dt 下限
    float dt_max            = 4.0f;   // dt 上限：防止卡頓/中斷後時間戳暴衝，
                                      //          把所有框推到畫面外導致全斷

    bool  class_aware       = true;   // true = 只允許同一個 class 互相配對
    bool  fuse_score        = false;  // true = cost 融合 detection 分數
};

// 輸入框（xyxy，原圖座標）
struct Box {
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
    float score = 0.f;
    int   cls   = 0;
};

// 每幀輸出
struct Track {
    int   track_id     = 0;
    int   cls          = 0;
    float score        = 0.f;
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;  // Kalman 平滑後的框
    int   tracklet_len = 0;                        // 已連續追蹤幀數
};

enum class TrackState { New = 0, Tracked, Lost, Removed };

// Kalman 狀態：mean = (cx, cy, a, h, vcx, vcy, va, vh)，a = w / h
// 速度單位為「每個標稱幀」，實際位移 = 速度 * dt
struct KalmanState {
    float mean[8]    = {0};
    float cov[8][8]  = {{0}};
};

// ---------------------------------------------------------------------------
//  單一軌跡
// ---------------------------------------------------------------------------
class STrack {
public:
    void init(const Box& b, int id, int frame, double time);
    void update(const Box& b, int frame, double time);
    void re_activate(const Box& b, int frame, double time);
    void predict(float dt);

    void mark_lost()    { state = TrackState::Lost; }
    void mark_removed() { state = TrackState::Removed; }

    void  tlwh(float out[4]) const;   // 由 Kalman mean 還原成 (x, y, w, h)
    Track to_track() const;

    int        track_id     = 0;
    int        cls          = 0;
    float      score        = 0.f;
    TrackState state        = TrackState::New;
    bool       is_activated = false;  // false = 只出現過一幀，還在觀察期
    int        frame_id     = 0;      // 最後一次被更新的幀
    int        start_frame  = 0;
    double     last_time    = 0.0;    // 最後一次被更新的時間（秒）
    int        tracklet_len = 0;
    KalmanState kf;
};

// ---------------------------------------------------------------------------
//  Tracker
// ---------------------------------------------------------------------------
class BYTETracker {
public:
    explicit BYTETracker(const Params& p = Params());

    // timestamp：擷取該幀影像的時刻（秒）。務必用「拿到影像的時間」，
    // 不要用推論結束的時間，否則會把推論延遲的抖動混進 dt。
    // 給負數 = 不使用時間戳，退回固定 dt = 1。
    const std::vector<Track>& update(const std::vector<Box>& dets,
                                     double timestamp = -1.0);

#ifndef BYTETRACK_NO_UTIL_ADAPTER
    // 直接吃 YOLOPostProcessor 的輸出
    // ⚠ 注意：座標必須先轉回原圖空間（去掉 letterbox 的 pad / ratio），
    //         否則 Kalman 的速度估計會落在錯誤的尺度上。
    const std::vector<Track>& update(const std::vector<Detection>& dets,
                                     double timestamp = -1.0);
    const std::vector<Track>& update(const DetectionBatch& batch,
                                     double timestamp = -1.0);
#endif

    void reset();

    int    frame_id() const { return frame_id_; }
    double last_dt()  const { return last_dt_; }   // 除錯用：上一幀的正規化 dt
    const std::vector<STrack>& tracks() const { return tracks_; }
    Params& params() { return p_; }

private:
    void build_cost(const std::vector<int>& track_idx,
                    const std::vector<Box>& dets,
                    bool fuse,
                    std::vector<float>& cost) const;

    void remove_duplicates();

    Params p_;
    int    frame_id_  = 0;
    int    next_id_   = 1;
    double last_time_ = -1.0;   // 上一幀的時間戳（-1 = 尚未有）
    double cur_time_  = 0.0;    // 本幀時間（秒）
    float  last_dt_   = 1.f;

    std::vector<STrack> tracks_;      // 只包含 Tracked / Lost
    std::vector<STrack> new_tracks_;
    std::vector<Track>  output_;

    // 每幀重複使用的暫存區（避免反覆配置記憶體）
    std::vector<Box>   high_, low_, rest_high_;
    std::vector<int>   pool_, unconfirmed_, r_tracked_;
    std::vector<float> cost_;
    std::vector<std::pair<int, int>> matches_;
    std::vector<int>   u_track_, u_det_, u_track2_, u_det2_;
    std::vector<Box>   det_buf_;
};

} // namespace bytetrack

#endif // BYTETRACK_H