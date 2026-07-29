#include "tracker.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace bytetrack {

// ============================================================================
//  Detection adapter —— 唯一需要對照你的 util.h 的地方
// ============================================================================
#ifndef BYTETRACK_NO_UTIL_ADAPTER
namespace {
inline Box det_to_box(const Detection& d)
{
    Box b;
    b.x1    = d.x1;
    b.y1    = d.y1;
    b.x2    = d.x2;
    b.y2    = d.y2;
    b.score = d.score;
    b.cls   = d.class_id;
    return b;
}
} // namespace
#endif

namespace {

constexpr float kLarge  = 1e5f;      // 不可行配對的 cost
constexpr float kStdPos = 1.f / 20.f;
constexpr float kStdVel = 1.f / 160.f;

// ---------------------------------------------------------------------------
//  座標轉換
// ---------------------------------------------------------------------------
inline void tlwh_to_xyah(const float t[4], float o[4])
{
    o[0] = t[0] + t[2] * 0.5f;
    o[1] = t[1] + t[3] * 0.5f;
    o[2] = (t[3] > 1e-6f) ? (t[2] / t[3]) : 0.f;
    o[3] = t[3];
}

inline void box_to_tlwh(const Box& b, float t[4])
{
    t[0] = b.x1;
    t[1] = b.y1;
    t[2] = b.x2 - b.x1;
    t[3] = b.y2 - b.y1;
}

inline float iou_tlwh(const float a[4], const float b[4])
{
    const float ax2 = a[0] + a[2], ay2 = a[1] + a[3];
    const float bx2 = b[0] + b[2], by2 = b[1] + b[3];

    const float ix = std::min(ax2, bx2) - std::max(a[0], b[0]);
    if (ix <= 0.f) return 0.f;
    const float iy = std::min(ay2, by2) - std::max(a[1], b[1]);
    if (iy <= 0.f) return 0.f;

    const float inter = ix * iy;
    const float uni   = a[2] * a[3] + b[2] * b[3] - inter;
    return (uni > 0.f) ? (inter / uni) : 0.f;
}

// ---------------------------------------------------------------------------
//  Kalman filter（constant velocity，稀疏結構手動展開）
//    F = [[I, dt·I], [0, I]]，H = [I, 0]
//    dt 為「標稱幀的倍數」，等速時 dt=1 即退化回原論文的每幀模型
// ---------------------------------------------------------------------------
void kf_initiate(KalmanState& s, const float xyah[4])
{
    for (int i = 0; i < 4; ++i) s.mean[i] = xyah[i];
    for (int i = 4; i < 8; ++i) s.mean[i] = 0.f;

    const float h = xyah[3];
    const float sd[8] = {
        2.f  * kStdPos * h, 2.f  * kStdPos * h, 1e-2f, 2.f  * kStdPos * h,
        10.f * kStdVel * h, 10.f * kStdVel * h, 1e-5f, 10.f * kStdVel * h
    };
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            s.cov[i][j] = (i == j) ? sd[i] * sd[i] : 0.f;
}

void kf_predict(KalmanState& s, bool zero_h_velocity, float dt)
{
    // Lost 的軌跡不要讓框繼續長大 / 縮小
    if (zero_h_velocity) s.mean[7] = 0.f;

    const float h = s.mean[3];
    // 過程噪聲隨 dt 放大：間隔越久，預測越不可信
    const float sd[8] = {
        kStdPos * h * dt, kStdPos * h * dt, 1e-2f * dt, kStdPos * h * dt,
        kStdVel * h * dt, kStdVel * h * dt, 1e-5f * dt, kStdVel * h * dt
    };

    // x = F x
    for (int i = 0; i < 4; ++i) s.mean[i] += dt * s.mean[i + 4];

    // P = F P Fᵀ：先做列 (F P)，再做行 (· Fᵀ)，兩步互不重疊所以可就地運算
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j)
            s.cov[i][j] += dt * s.cov[i + 4][j];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j)
            s.cov[i][j] += dt * s.cov[i][j + 4];

    // P += Q
    for (int i = 0; i < 8; ++i) s.cov[i][i] += sd[i] * sd[i];
}

void kf_update(KalmanState& s, const float z[4])
{
    const float h = s.mean[3];
    const float rsd[4] = { kStdPos * h, kStdPos * h, 1e-1f, kStdPos * h };

    // S = H P Hᵀ + R  (4×4)，B = P Hᵀ  (8×4)
    float S[4][4], B[8][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            S[i][j] = s.cov[i][j] + ((i == j) ? rsd[i] * rsd[i] : 0.f);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j)
            B[i][j] = s.cov[i][j];

    // Cholesky：S = L Lᵀ（S 為對稱正定，比直接求逆穩定）
    float L[4][4] = {{0}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = S[i][j];
            for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
            if (i == j) L[i][i] = std::sqrt(std::max(sum, 1e-12f));
            else        L[i][j] = sum / L[j][j];
        }
    }

    // 解 S Kᵀ = Bᵀ 得到 Kalman gain K (8×4)
    float K[8][4];
    for (int r = 0; r < 8; ++r) {
        float y[4], x[4];
        for (int i = 0; i < 4; ++i) {                  // forward:  L y = b
            float sum = B[r][i];
            for (int k = 0; k < i; ++k) sum -= L[i][k] * y[k];
            y[i] = sum / L[i][i];
        }
        for (int i = 3; i >= 0; --i) {                 // backward: Lᵀ x = y
            float sum = y[i];
            for (int k = i + 1; k < 4; ++k) sum -= L[k][i] * x[k];
            x[i] = sum / L[i][i];
        }
        for (int i = 0; i < 4; ++i) K[r][i] = x[i];
    }

    // x += K (z − H x)
    float innov[4];
    for (int i = 0; i < 4; ++i) innov[i] = z[i] - s.mean[i];
    for (int r = 0; r < 8; ++r) {
        float acc = 0.f;
        for (int i = 0; i < 4; ++i) acc += K[r][i] * innov[i];
        s.mean[r] += acc;
    }

    // P −= K S Kᵀ = K Bᵀ
    float newcov[8][8];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            float acc = 0.f;
            for (int k = 0; k < 4; ++k) acc += K[i][k] * B[j][k];
            newcov[i][j] = s.cov[i][j] - acc;
        }
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            s.cov[i][j] = newcov[i][j];
}

// ---------------------------------------------------------------------------
//  Jonker-Volgenant 指派（e-maxx 版本，要求 nr <= nc）
//  回傳 row2col[i] = 配到的 column，未配到為 -1
// ---------------------------------------------------------------------------
void jv_assign(const std::vector<float>& cost, int nr, int nc,
               std::vector<int>& row2col)
{
    const float INF = std::numeric_limits<float>::max();
    std::vector<float> u(nr + 1, 0.f), v(nc + 1, 0.f), minv(nc + 1);
    std::vector<int>   p(nc + 1, 0), way(nc + 1, 0);
    std::vector<char>  used(nc + 1);

    for (int i = 1; i <= nr; ++i) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv.begin(), minv.end(), INF);
        std::fill(used.begin(), used.end(), 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            int   j1    = -1;
            float delta = INF;
            for (int j = 1; j <= nc; ++j) {
                if (used[j]) continue;
                const float cur = cost[(i0 - 1) * nc + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            if (j1 < 0) break;                       // 保險，理論上不會發生
            for (int j = 0; j <= nc; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else           minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);

        while (j0) { const int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; }
    }

    row2col.assign(nr, -1);
    for (int j = 1; j <= nc; ++j)
        if (p[j] > 0 && p[j] <= nr) row2col[p[j] - 1] = j - 1;
}

// 帶 cost 上限的線性指派：cost >= thresh 的配對會被拆掉
void linear_assignment(const std::vector<float>& cost, int nr, int nc,
                       float thresh,
                       std::vector<std::pair<int, int>>& matches,
                       std::vector<int>& u_row,
                       std::vector<int>& u_col)
{
    matches.clear();
    u_row.clear();
    u_col.clear();

    if (nr == 0 || nc == 0) {
        for (int i = 0; i < nr; ++i) u_row.push_back(i);
        for (int j = 0; j < nc; ++j) u_col.push_back(j);
        return;
    }

    std::vector<int> row2col;
    if (nr <= nc) {
        jv_assign(cost, nr, nc, row2col);
    } else {
        std::vector<float> t(static_cast<size_t>(nc) * nr);
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nc; ++j)
                t[static_cast<size_t>(j) * nr + i] = cost[static_cast<size_t>(i) * nc + j];
        std::vector<int> col2row;
        jv_assign(t, nc, nr, col2row);
        row2col.assign(nr, -1);
        for (int j = 0; j < nc; ++j)
            if (col2row[j] >= 0) row2col[col2row[j]] = j;
    }

    std::vector<char> col_taken(nc, 0);
    for (int i = 0; i < nr; ++i) {
        const int j = row2col[i];
        if (j >= 0 && cost[static_cast<size_t>(i) * nc + j] < thresh) {
            matches.emplace_back(i, j);
            col_taken[j] = 1;
        } else {
            u_row.push_back(i);
        }
    }
    for (int j = 0; j < nc; ++j)
        if (!col_taken[j]) u_col.push_back(j);
}

} // anonymous namespace

// ============================================================================
//  STrack
// ============================================================================
void STrack::tlwh(float out[4]) const
{
    const float w = kf.mean[2] * kf.mean[3];
    const float h = kf.mean[3];
    out[0] = kf.mean[0] - w * 0.5f;
    out[1] = kf.mean[1] - h * 0.5f;
    out[2] = w;
    out[3] = h;
}

void STrack::init(const Box& b, int id, int frame, double time)
{
    float t[4], xyah[4];
    box_to_tlwh(b, t);
    tlwh_to_xyah(t, xyah);
    kf_initiate(kf, xyah);

    track_id     = id;
    cls          = b.cls;
    score        = b.score;
    state        = TrackState::Tracked;
    is_activated = (frame == 1);   // 第一幀直接輸出，其餘要再確認一次
    frame_id     = frame;
    start_frame  = frame;
    last_time    = time;
    tracklet_len = 0;
}

void STrack::update(const Box& b, int frame, double time)
{
    float t[4], xyah[4];
    box_to_tlwh(b, t);
    tlwh_to_xyah(t, xyah);
    kf_update(kf, xyah);

    state        = TrackState::Tracked;
    is_activated = true;
    cls          = b.cls;
    score        = b.score;
    frame_id     = frame;
    last_time    = time;
    ++tracklet_len;
}

void STrack::re_activate(const Box& b, int frame, double time)
{
    float t[4], xyah[4];
    box_to_tlwh(b, t);
    tlwh_to_xyah(t, xyah);
    kf_update(kf, xyah);

    state        = TrackState::Tracked;
    is_activated = true;
    cls          = b.cls;
    score        = b.score;
    frame_id     = frame;
    last_time    = time;
    tracklet_len = 0;
}

void STrack::predict(float dt)
{
    kf_predict(kf, state != TrackState::Tracked, dt);
}

Track STrack::to_track() const
{
    float t[4];
    tlwh(t);
    Track r;
    r.track_id     = track_id;
    r.cls          = cls;
    r.score        = score;
    r.x1           = t[0];
    r.y1           = t[1];
    r.x2           = t[0] + t[2];
    r.y2           = t[1] + t[3];
    r.tracklet_len = tracklet_len;
    return r;
}

// ============================================================================
//  BYTETracker
// ============================================================================
BYTETracker::BYTETracker(const Params& p)
    : p_(p)
{
    reset();
}

void BYTETracker::reset()
{
    frame_id_  = 0;
    next_id_   = 1;
    last_time_ = -1.0;
    cur_time_  = 0.0;
    last_dt_   = 1.f;
    tracks_.clear();
    new_tracks_.clear();
    output_.clear();
}

void BYTETracker::build_cost(const std::vector<int>& track_idx,
                             const std::vector<Box>& dets,
                             bool fuse,
                             std::vector<float>& cost) const
{
    const int nr = static_cast<int>(track_idx.size());
    const int nc = static_cast<int>(dets.size());
    cost.assign(static_cast<size_t>(nr) * nc, 0.f);
    if (nr == 0 || nc == 0) return;

    float tb[4], db[4];
    for (int i = 0; i < nr; ++i) {
        const STrack& tr = tracks_[track_idx[i]];
        tr.tlwh(tb);
        for (int j = 0; j < nc; ++j) {
            const Box& d = dets[j];
            if (p_.class_aware && d.cls != tr.cls) {
                cost[static_cast<size_t>(i) * nc + j] = kLarge;
                continue;
            }
            box_to_tlwh(d, db);
            const float iou = iou_tlwh(tb, db);
            cost[static_cast<size_t>(i) * nc + j] =
                fuse ? (1.f - iou * d.score) : (1.f - iou);
        }
    }
}

void BYTETracker::remove_duplicates()
{
    // Tracked 與 Lost 高度重疊時，保留歷史較長的那條
    const int n = static_cast<int>(tracks_.size());
    float a[4], b[4];
    for (int i = 0; i < n; ++i) {
        if (tracks_[i].state == TrackState::Removed) continue;
        for (int j = i + 1; j < n; ++j) {
            if (tracks_[j].state == TrackState::Removed) continue;
            const bool ti = (tracks_[i].state == TrackState::Tracked);
            const bool tj = (tracks_[j].state == TrackState::Tracked);
            if (ti == tj) continue;                      // 同狀態不比

            tracks_[i].tlwh(a);
            tracks_[j].tlwh(b);
            if (iou_tlwh(a, b) < 0.85f) continue;

            const int li = tracks_[i].frame_id - tracks_[i].start_frame;
            const int lj = tracks_[j].frame_id - tracks_[j].start_frame;
            if (li > lj) tracks_[j].mark_removed();
            else         tracks_[i].mark_removed();
        }
    }
}

const std::vector<Track>& BYTETracker::update(const std::vector<Box>& dets,
                                              double timestamp)
{
    ++frame_id_;

    // ── 0. 由時間戳算出這一幀的 dt（正規化成「標稱幀的倍數」）────────────
    float dt = 1.f;
    if (timestamp >= 0.0) {
        if (last_time_ >= 0.0) {
            dt = static_cast<float>((timestamp - last_time_) * p_.nominal_fps);
            dt = std::clamp(dt, p_.dt_min, p_.dt_max);
        }
        last_time_ = timestamp;
        cur_time_  = timestamp;
    } else {
        // 沒有時間戳：用幀計數推算時間，行為等同原版固定 dt
        cur_time_ = frame_id_ / static_cast<double>(p_.nominal_fps);
    }
    last_dt_ = dt;

    // ── 1. 依分數切成 high / low ──────────────────────────────────────────
    high_.clear();
    low_.clear();
    for (const Box& d : dets) {
        if (d.score < p_.low_thresh)    continue;
        if (d.score >= p_.track_thresh) high_.push_back(d);
        else                            low_.push_back(d);
    }

    // ── 2. Kalman 預測 ────────────────────────────────────────────────────
    for (STrack& t : tracks_) t.predict(dt);

    pool_.clear();
    unconfirmed_.clear();
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
        const STrack& t = tracks_[i];
        if (t.state == TrackState::Tracked && !t.is_activated)
            unconfirmed_.push_back(i);          // 上一幀才出生，還在觀察期
        else
            pool_.push_back(i);                 // Tracked(已確認) 或 Lost
    }

    // ── 3. 第一階段：pool ↔ 高分框 ────────────────────────────────────────
    build_cost(pool_, high_, p_.fuse_score, cost_);
    linear_assignment(cost_, static_cast<int>(pool_.size()),
                      static_cast<int>(high_.size()),
                      p_.match_thresh, matches_, u_track_, u_det_);

    for (const auto& m : matches_) {
        STrack&    t = tracks_[pool_[m.first]];
        const Box& d = high_[m.second];
        if (t.state == TrackState::Tracked) t.update(d, frame_id_, cur_time_);
        else                                t.re_activate(d, frame_id_, cur_time_);
    }

    // ── 4. 第二階段：沒配到的 Tracked ↔ 低分框（救遮擋 / 模糊的框）───────
    r_tracked_.clear();
    for (int i : u_track_) {
        const int gi = pool_[i];
        if (tracks_[gi].state == TrackState::Tracked) r_tracked_.push_back(gi);
        // 原本就 Lost 且沒配到的，維持 Lost
    }

    build_cost(r_tracked_, low_, false, cost_);
    linear_assignment(cost_, static_cast<int>(r_tracked_.size()),
                      static_cast<int>(low_.size()),
                      p_.match_thresh_low, matches_, u_track2_, u_det2_);

    for (const auto& m : matches_)
        tracks_[r_tracked_[m.first]].update(low_[m.second], frame_id_, cur_time_);

    for (int i : u_track2_)
        tracks_[r_tracked_[i]].mark_lost();

    // ── 5. 第三階段：未確認軌跡 ↔ 剩下的高分框 ────────────────────────────
    rest_high_.clear();
    for (int j : u_det_) rest_high_.push_back(high_[j]);

    build_cost(unconfirmed_, rest_high_, p_.fuse_score, cost_);
    linear_assignment(cost_, static_cast<int>(unconfirmed_.size()),
                      static_cast<int>(rest_high_.size()),
                      p_.match_thresh_new, matches_, u_track2_, u_det2_);

    for (const auto& m : matches_)
        tracks_[unconfirmed_[m.first]].update(rest_high_[m.second],
                                              frame_id_, cur_time_);

    for (int i : u_track2_)
        tracks_[unconfirmed_[i]].mark_removed();   // 只出現一幀 → 判定為雜訊

    // ── 6. 產生新軌跡 ─────────────────────────────────────────────────────
    new_tracks_.clear();
    for (int j : u_det2_) {
        const Box& d = rest_high_[j];
        if (d.score < p_.high_thresh) continue;
        STrack t;
        t.init(d, next_id_++, frame_id_, cur_time_);
        new_tracks_.push_back(t);
    }

    // ── 7. Lost 過期 → Removed（以秒為單位，與幀率無關）───────────────────
    for (STrack& t : tracks_)
        if (t.state == TrackState::Lost &&
            cur_time_ - t.last_time > static_cast<double>(p_.max_lost_seconds))
            t.mark_removed();

    // ── 8. 整理容器（先去重再刪除，避免索引失效）─────────────────────────
    remove_duplicates();
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const STrack& t) {
                                     return t.state == TrackState::Removed;
                                 }),
                  tracks_.end());
    tracks_.insert(tracks_.end(), new_tracks_.begin(), new_tracks_.end());

    // ── 9. 輸出 ───────────────────────────────────────────────────────────
    output_.clear();
    for (const STrack& t : tracks_)
        if (t.state == TrackState::Tracked && t.is_activated)
            output_.push_back(t.to_track());

    return output_;
}

#ifndef BYTETRACK_NO_UTIL_ADAPTER
const std::vector<Track>& BYTETracker::update(const std::vector<Detection>& dets,
                                              double timestamp)
{
    det_buf_.clear();
    det_buf_.reserve(dets.size());
    for (const Detection& d : dets) det_buf_.push_back(det_to_box(d));
    return update(det_buf_, timestamp);
}

const std::vector<Track>& BYTETracker::update(const DetectionBatch& batch,
                                              double timestamp)
{
    det_buf_.clear();
    det_buf_.reserve(static_cast<size_t>(batch.count));
    for (int i = 0; i < batch.count; ++i)
        det_buf_.push_back(det_to_box(batch.data[i]));
    return update(det_buf_, timestamp);
}
#endif

} // namespace bytetrack