<<<<<<< HEAD
// kalman_box_tracker.h
// 宣告檔：YOLO(NMS 後) bounding box 的等速模型 Kalman filter，
// 用來預測物件下一偵位置。實作在 kalman_box_tracker.cpp。
//
//   #include "kalman_box_tracker.h"
//   // 編譯時記得一起連結 kalman_box_tracker.cpp
//   using namespace kbt;
//   KalmanBoxTracker t(toCenter({100,100,150,200}));
//   CxCyWH pred = t.predict();      // 預測這一偵
//   t.update(toCenter(detBox));     // 拿 YOLO 偵測校正
=======
#ifndef BYTETRACK_H
#define BYTETRACK_H
>>>>>>> 327f7fd (update)

// ============================================================================
//  ByteTrack (Zhang et al., ECCV 2022) — C++ 實作
//
//  設計重點：
//    * 不依賴 OpenCV / Eigen / OpenMP，只用 std（方便丟到 ARM / DPU 板子上）
//    * Kalman filter 為手寫 8 維狀態 (cx, cy, a, h, vcx, vcy, va, vh)
//      矩陣運算針對 constant-velocity 的稀疏結構展開，沒有通用 matmul
//    * 指派用 Jonker-Volgenant / e-maxx O(n^2 m) 版本，n,m 都是幾十個等級
//    * 所有 track 存在單一 std::vector，關聯階段只用 index，
//      不會有指標失效的問題（原版官方 C++ 實作在這裡很容易踩雷）
//
//  用法：
//      bytetrack::BYTETracker tracker;                  // 預設 30 fps
//      const auto& tracks = tracker.update(dets[b]);    // 每幀呼叫一次
//      for (const auto& t : tracks) { t.track_id, t.x1 ... }
// ============================================================================

<<<<<<< HEAD
=======
#include <utility>
>>>>>>> 327f7fd (update)
#include <vector>

// Detection / DetectionBatch 的 adapter。若你的 util.h 欄位名不同，
// 只需要改 bytetrack.cpp 最上方 det_to_box() 那一個函式；
// 或是編譯時定義 BYTETRACK_NO_UTIL_ADAPTER 完全關掉這層。
#ifndef BYTETRACK_NO_UTIL_ADAPTER
#include "util.h"
#endif

<<<<<<< HEAD
// ======================= 極簡矩陣 =======================
struct Mat {
    int r = 0, c = 0;
    std::vector<double> d;

    Mat() {}
    Mat(int r, int c) : r(r), c(c), d(static_cast<size_t>(r) * c, 0.0) {}

    // 存取子是一行、且在迴圈裡大量呼叫，留在 header inline
    double&       operator()(int i, int j)       { return d[i * c + j]; }
    double        operator()(int i, int j) const { return d[i * c + j]; }

    static Mat I(int n);   // 單位矩陣，實作在 .cpp
};

// 矩陣運算（實作在 .cpp）
Mat operator*(const Mat& a, const Mat& b);
Mat operator+(const Mat& a, const Mat& b);
Mat operator-(const Mat& a, const Mat& b);
Mat transpose(const Mat& a);
Mat inverse(Mat a);          // Gauss-Jordan 求逆
=======
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
    int   track_buffer      = 30;     // Lost 狀態最多保留幾幀 (@30fps)
    float frame_rate        = 30.f;   // 實際 fps，用來換算 track_buffer
    bool  class_aware       = true;   // true = 只允許同一個 class 互相配對
    bool  fuse_score        = false;  // true = cost 融合 detection 分數
};

// 輸入框（xyxy）
struct Box {
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
    float score = 0.f;
    int   cls   = 0;
};
>>>>>>> 327f7fd (update)

// 每幀輸出
struct Track {
    int   track_id     = 0;
    int   cls          = 0;
    float score        = 0.f;
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;  // Kalman 平滑後的框
    int   tracklet_len = 0;                        // 已連續追蹤幀數
};

<<<<<<< HEAD
CxCyWH toCenter(const Box& b);   // xyxy -> 中心點
Box    toXYXY(const CxCyWH& c);  // 中心點 -> xyxy
=======
enum class TrackState { New = 0, Tracked, Lost, Removed };
>>>>>>> 327f7fd (update)

// Kalman 狀態：mean = (cx, cy, a, h, vcx, vcy, va, vh)，a = w / h
struct KalmanState {
    float mean[8]    = {0};
    float cov[8][8]  = {{0}};
};

// ---------------------------------------------------------------------------
//  單一軌跡
// ---------------------------------------------------------------------------
class STrack {
public:
<<<<<<< HEAD
    // 用第一次偵測框初始化。dt = 兩偵間隔(用「偵」為單位時填 1)。
    // 預設參數寫在宣告端。
    explicit KalmanBoxTracker(const CxCyWH& init, double dt = 1.0);

    CxCyWH predict();                    // 預測步：往前推一偵，回傳預測框
    void   update(const CxCyWH& meas);   // 更新步：拿當偵偵測框校正

    CxCyWH currentBox() const;           // 目前估計框(中心點格式)
    void   velocity(double& vx, double& vy, double& vw, double& vh) const;

    // 可選：外部微調雜訊
    void setMeasurementNoise(double pos, double size); // 大→平滑
    void setProcessNoise(double pos, double vel);      // 大→靈敏
=======
    void init(const Box& b, int id, int frame);          // 建立新軌跡
    void update(const Box& b, int frame);               // 配對成功（Tracked）
    void re_activate(const Box& b, int frame);          // 從 Lost 救回來
    void predict();                                     // Kalman 預測一步

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
    int        tracklet_len = 0;
    KalmanState kf;
};

// ---------------------------------------------------------------------------
//  Tracker
// ---------------------------------------------------------------------------
class BYTETracker {
public:
    explicit BYTETracker(const Params& p = Params());

    // 核心介面
    const std::vector<Track>& update(const std::vector<Box>& dets);

#ifndef BYTETRACK_NO_UTIL_ADAPTER
    // 直接吃 YOLOPostProcessor 的輸出
    const std::vector<Track>& update(const std::vector<Detection>& dets);
    const std::vector<Track>& update(const DetectionBatch& batch);
#endif

    void reset();

    int  frame_id()      const { return frame_id_; }
    int  max_time_lost() const { return max_time_lost_; }
    const std::vector<STrack>& tracks() const { return tracks_; }
    Params& params() { return p_; }
>>>>>>> 327f7fd (update)

private:
    // cost = 1 - IoU（含 class / score 融合），rows = tracks, cols = dets
    void build_cost(const std::vector<int>& track_idx,
                    const std::vector<Box>& dets,
                    bool fuse,
                    std::vector<float>& cost) const;

    void remove_duplicates();

    Params p_;
    int    frame_id_      = 0;
    int    next_id_       = 1;
    int    max_time_lost_ = 30;

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

<<<<<<< HEAD
#endif // KALMAN_BOX_TRACKER_H
=======
#endif // BYTETRACK_H
>>>>>>> 327f7fd (update)
