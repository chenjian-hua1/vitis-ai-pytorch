// parallel_cvt.h — 以固定工作執行緒做分條色彩轉換
//
// 為什麼不直接用 cv::cvtColor 的內建平行化:
//
//   cvtColor 內部走 parallel_for_,執行緒數由「全域」的 cv::setNumThreads
//   決定。但那個設定會同時影響所有 OpenCV 運算 —— copyTo、resize、
//   draw、imwrite 全部跟著開執行緒,於是它們的工作執行緒和 pipeline 的
//   階段互相搶核、踩 cache,症狀是各階段出現數倍抖動。
//
//   把全域設成 1 可以消掉抖動(實測轉置從 17.9 ms 降到 6.2 ms),
//   代價是 1080p 的 UYVY→BGR 從約 9 ms 變成約 26 ms。
//
//   這個類別讓兩者兼得:全域維持 1,只有「這一個大運算」用自己的
//   固定執行緒池平行化。數量與綁核都由我們決定,不會外溢到其他運算。
//
// 用法:
//   ParallelCvt cvt(3);                      // 繼承行程的 affinity
//   ParallelCvt cvt(3, {1, 2, 3});           // 或明確指定核心
//   cvt.convert(uyvy, bgr, cv::COLOR_YUV2BGR_UYVY);
//
// 限制:只做「逐列獨立」的轉換。YUV422 的次取樣是水平方向的,不跨列,
//       所以依列切分安全;若換成 NV12 這類垂直次取樣的格式,切點必須
//       對齊偶數列。

#pragma once

#include <opencv2/opencv.hpp>

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

class ParallelCvt {
public:
    // n_workers <= 1:直接在呼叫端執行緒做,不開任何 worker。
    // cpus:每個「額外」worker 要綁的核心(worker 0 是呼叫端自己)。
    //       留空表示不綁,繼承行程的 affinity。
    explicit ParallelCvt(int n_workers = 3, std::vector<int> cpus = {})
        : n_(n_workers < 1 ? 1 : n_workers)
    {
        if (n_ <= 1) return;
        workers_.reserve(static_cast<size_t>(n_ - 1));
        for (int i = 1; i < n_; ++i) {
            workers_.emplace_back([this, i] { loop(i); });
            if (static_cast<size_t>(i - 1) < cpus.size())
                pin(workers_.back(), cpus[static_cast<size_t>(i - 1)]);
        }
    }

    ~ParallelCvt() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++gen_;
        }
        cv_start_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }

    ParallelCvt(const ParallelCvt&)            = delete;
    ParallelCvt& operator=(const ParallelCvt&) = delete;

    int workers() const { return n_; }

    // dst 若尺寸型別已相符就不會重新配置 —— 這點很重要,
    // 因為 dst 通常是 DMA 記憶體,重新配置會讓實體位址失效。
    void convert(const cv::Mat& src, cv::Mat& dst, int code) {
        if (n_ <= 1 || src.rows < n_ * 8) {      // 太小就不值得分工
            cv::cvtColor(src, dst, code);
            return;
        }

        // 先把 dst 配置好,worker 才能安全地各寫各的列
        if (dst.rows != src.rows || dst.cols != src.cols ||
            dst.type() != CV_8UC3) {
            dst.create(src.rows, src.cols, CV_8UC3);
        }

        {
            std::lock_guard<std::mutex> lk(m_);
            src_ = &src; dst_ = &dst; code_ = code;
            remaining_ = n_ - 1;
            ++gen_;
        }
        cv_start_.notify_all();

        run_stripe(0);                           // 呼叫端也負責一條

        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [&] { return remaining_ == 0; });
    }

private:
    static void pin(std::thread& t, int cpu) {
#if defined(__linux__)
        if (cpu < 0) return;
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
        pthread_setaffinity_np(t.native_handle(), sizeof(set), &set);
#else
        (void)t; (void)cpu;
#endif
    }

    void run_stripe(int idx) {
        const int rows = src_->rows;
        const int y0 = static_cast<int>((int64_t)rows * idx / n_);
        const int y1 = static_cast<int>((int64_t)rows * (idx + 1) / n_);
        if (y1 <= y0) return;

        const cv::Range r(y0, y1);
        cv::Mat s = src_->rowRange(r);
        cv::Mat d = dst_->rowRange(r);
        cv::cvtColor(s, d, code_);
    }

    void loop(int idx) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_start_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
            }
            run_stripe(idx);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--remaining_ == 0) cv_done_.notify_one();
            }
        }
    }

    int n_;
    std::vector<std::thread> workers_;

    std::mutex m_;
    std::condition_variable cv_start_, cv_done_;
    uint64_t gen_ = 0;
    int  remaining_ = 0;
    bool stop_ = false;

    const cv::Mat* src_ = nullptr;
    cv::Mat*       dst_ = nullptr;
    int            code_ = 0;
};