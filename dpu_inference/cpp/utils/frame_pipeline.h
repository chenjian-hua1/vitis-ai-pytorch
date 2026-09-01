// frame_pipeline.h — 多階段 pipeline 的 slot 環
//
// 設計重點:slot 的「所有權」在階段之間流轉,同一時刻只有一個階段
// 持有某個 slot,所以影像資料本身完全不需要上鎖 —— 鎖只保護那幾個
// 小小的 index 佇列。這比 FrameGrabber 那種「共用一張 latest + clone」
// 的做法省掉每幀一次全幀複製。
//
//   free --> [階段1] --> q1 --> [階段2] --> q2 --> [階段3] --> free
//
// 拿不到空 slot 時丟棄當前輸入,而不是等待。對即時影像來說這是對的:
// 等待會讓延遲不斷累積,最後看到的是好幾秒前的畫面。

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

// 命名為 fpipe 而非 pipe:POSIX 的 <unistd.h> 有一個全域函式 pipe(int*),
// 兩者同時可見時編譯器會報 "redeclared as different kind of entity"。
namespace fpipe {

// 單向的 slot index 佇列
class Stage {
public:
    void push(int slot) {
        { std::lock_guard<std::mutex> lk(m_);
          q_.push_back(slot);
          if (q_.size() > peak_) peak_ = q_.size(); }
        cv_.notify_one();
    }

    // wait_ms < 0 無限等;== 0 不等待(拿不到回 -1)
    int pop(int wait_ms = 200) {
        std::unique_lock<std::mutex> lk(m_);
        if (wait_ms == 0) {
            if (q_.empty()) return -1;
        } else if (wait_ms < 0) {
            cv_.wait(lk, [&] { return !q_.empty() || stop_; });
        } else {
            cv_.wait_for(lk, std::chrono::milliseconds(wait_ms),
                         [&] { return !q_.empty() || stop_; });
        }
        if (q_.empty()) return -1;
        const int s = q_.front();
        q_.pop_front();
        return s;
    }

    // 只取最新的一個,較舊的透過 on_stale 回收。
    //
    // 用在「監看型」的末端階段(串流、顯示):那裡的目的是看到當前畫面,
    // 不是每一幀都要送出。若讓它逐幀處理,它的速度就會透過 slot 回收
    // 反壓整條 pipeline,把吞吐拖到它的水準。
    template <class F>
    int pop_latest(int wait_ms, F&& on_stale) {
        std::vector<int> stale;
        int newest = -1;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait_for(lk, std::chrono::milliseconds(wait_ms),
                         [&] { return !q_.empty() || stop_; });
            if (q_.empty()) return -1;
            while (q_.size() > 1) { stale.push_back(q_.front()); q_.pop_front(); }
            newest = q_.front();
            q_.pop_front();
        }
        // 在鎖外回收,避免在持有本佇列的鎖時去碰另一個佇列的鎖
        for (int s : stale) on_stale(s);
        return newest;
    }

    void stop() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
    }

    bool stopped() const { std::lock_guard<std::mutex> lk(m_); return stop_; }
    size_t depth() const { std::lock_guard<std::mutex> lk(m_); return q_.size(); }
    size_t peak()  const { std::lock_guard<std::mutex> lk(m_); return peak_; }

private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<int> q_;
    size_t peak_ = 0;
    bool stop_ = false;
};

// 各階段的耗時統計。
//
// 同時保留「累計平均」與「最近 N 筆的視窗統計」:
// 累計平均在跑了幾千幀之後幾乎不再變動(每個新樣本只推動 1/n),
// 看起來像凍住,完全反映不出當下的抖動。要判斷波動來源,
// 得看視窗內的平均與最大值。
class StageTimer {
public:
    explicit StageTimer(size_t window = 60) : win_(window) { buf_.reserve(window); }

    void add(double ms) {
        std::lock_guard<std::mutex> lk(m_);
        sum_ += ms; ++n_;
        if (buf_.size() < win_) buf_.push_back(ms);
        else { buf_[pos_] = ms; }
        pos_ = (pos_ + 1) % win_;
    }

    double avg() const {                     // 累計平均
        std::lock_guard<std::mutex> lk(m_);
        return n_ ? sum_ / static_cast<double>(n_) : 0.0;
    }

    // 最近 N 筆的平均 / 最大 —— 抖動看這個
    double recent_avg() const {
        std::lock_guard<std::mutex> lk(m_);
        if (buf_.empty()) return 0.0;
        double s = 0; for (double v : buf_) s += v;
        return s / static_cast<double>(buf_.size());
    }
    double recent_max() const {
        std::lock_guard<std::mutex> lk(m_);
        double mx = 0; for (double v : buf_) if (v > mx) mx = v;
        return mx;
    }
    double recent_min() const {
        std::lock_guard<std::mutex> lk(m_);
        if (buf_.empty()) return 0.0;
        double mn = buf_[0]; for (double v : buf_) if (v < mn) mn = v;
        return mn;
    }

private:
    mutable std::mutex m_;
    double sum_ = 0;
    long long n_ = 0;
    size_t win_, pos_ = 0;
    std::vector<double> buf_;
};

// 指數移動平均。瞬時 FPS 在有丟幀與反壓的系統裡本來就會抖,
// 用它平滑之後比較看得出趨勢。
class Ema {
public:
    explicit Ema(double alpha = 0.2) : a_(alpha) {}
    double push(double v) {
        v_ = init_ ? (a_ * v + (1 - a_) * v_) : v;
        init_ = true;
        return v_;
    }
    double get() const { return v_; }
private:
    double a_, v_ = 0;
    bool init_ = false;
};

// 把執行緒綁到指定的核心。
//
// 為什麼需要:pipeline 的階段耗時本來是固定的,但被搶佔或 cache 被
// 其他執行緒逐出時會出現數倍的抖動。把重的階段釘在專屬核心上,
// 它的工作集比較不會被踩,耗時也就穩定。
//
// 限制「整個行程」可用的核心。之後建立的執行緒會繼承這個設定,
// 所以只要在建立 pipeline 執行緒之前呼叫一次,就能把某顆核心空出來
// 保留給特定用途,其餘執行緒仍由 OS 自由排程於剩下的核心之間。
//
// 這比「逐條綁核」溫和:我們只保證某顆核不被別人用,
// 至於其他階段要怎麼分配,交給排程器決定。
inline bool restrict_process_to(const std::vector<int>& cpus) {
#if defined(__linux__)
    if (cpus.empty()) return true;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) if (c >= 0) CPU_SET(c, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
    (void)cpus;
    return false;
#endif
}

// cpu < 0 表示不綁。回傳 false 表示平台不支援或設定失敗。
inline bool pin_thread(std::thread& t, int cpu) {
#if defined(__linux__)
    if (cpu < 0) return true;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(t.native_handle(), sizeof(set), &set) == 0;
#else
    (void)t; (void)cpu;
    return false;
#endif
}

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace fpipe