// draw_predictor.h — 預測「這一幀」繪圖時各 track 的框位置，供 DPU 期間預載入使用
//
//  - 只讀取 tracker.update() 的輸出，不改動 tracker 內部狀態（不會重複做 Kalman 預測）
//  - 等速模型：用前後兩次 update 的位置差與時間差估速度，再以本幀擷取時間外插
//  - 速度做指數平滑，降低偵測框抖動造成的誤差
#pragma once
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <drawer.h>            // bytetrack::Track

class DrawPredictor {
public:
    // alpha  ：速度平滑係數（越大越跟隨最新速度）
    // max_dt ：兩幀間隔超過此值（秒）就不使用速度，避免掉幀後外插過頭
    explicit DrawPredictor(float alpha = 0.5f, double max_dt = 0.2)
        : alpha_(alpha), max_dt_(max_dt) {}

    // 每幀 tracker.update() 之後呼叫；t = 該幀擷取時間（秒）
    void update(const std::vector<bytetrack::Track>& tracks, double t)
    {
        next_.clear();
        for (const bytetrack::Track& tr : tracks) {
            State s;
            s.trk = tr;
            s.t   = t;
            auto it = cur_.find(tr.track_id);
            if (it != cur_.end()) {
                const State& o  = it->second;
                const double dt = t - o.t;
                if (dt > 1e-4 && dt <= max_dt_) {
                    const float inv = static_cast<float>(1.0 / dt);
                    const float v[4] = { (tr.x1 - o.trk.x1) * inv, (tr.y1 - o.trk.y1) * inv,
                                         (tr.x2 - o.trk.x2) * inv, (tr.y2 - o.trk.y2) * inv };
                    for (int k = 0; k < 4; ++k)
                        s.v[k] = alpha_ * v[k] + (1.f - alpha_) * o.v[k];
                }
            }
            next_.emplace(tr.track_id, s);
        }
        cur_.swap(next_);
    }

    // DPU 執行期間呼叫；t = 目前這一幀的擷取時間（秒）
    const std::vector<bytetrack::Track>& predict(double t)
    {
        out_.clear();
        for (const auto& kv : cur_) {
            const State& s  = kv.second;
            const float  dt = static_cast<float>(std::clamp(t - s.t, 0.0, max_dt_));
            bytetrack::Track p = s.trk;
            p.x1 += s.v[0] * dt;  p.y1 += s.v[1] * dt;
            p.x2 += s.v[2] * dt;  p.y2 += s.v[3] * dt;
            out_.push_back(p);
        }
        return out_;
    }

private:
    struct State {
        bytetrack::Track trk{};
        double           t = 0.0;
        float            v[4] = {0.f, 0.f, 0.f, 0.f};
    };
    std::unordered_map<int, State> cur_, next_;
    std::vector<bytetrack::Track>  out_;
    float  alpha_;
    double max_dt_;
};