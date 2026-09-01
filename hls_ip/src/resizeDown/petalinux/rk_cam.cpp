// rk_camera.cpp — 相機擷取與 resize 並行的雙執行緒 pipeline
//
// 架構:
//   [擷取執行緒] grab -> retrieve(原始 UYVY) -> cvtColor 直接寫進 DMA slot
//                                      |
//                                   filled 佇列
//                                      v
//   [處理執行緒] letterbox()(zero-copy)-> IP
//
// UYVY 相對 MJPEG 的差別:
//   MJPEG  省頻寬,但 CPU 要解 JPEG(1080p 在 A53 約 25~30 ms,擋住 30fps)
//   UYVY   免解碼,但很吃頻寬:1920x1080x2x60 = 248.8 MB/s,必須 USB 3.0
//
// 色彩轉換的處理:
//   設 CAP_PROP_CONVERT_RGB=0,拿到原始的 CV_8UC2 影格,自己呼叫 cvtColor
//   直接輸出到 DMA slot。這樣「轉換」與「搬進 DMA」合而為一,
//   比起讓 OpenCV 轉好再複製一次,省下整整一次全幀搬移。
//
// 用法:
//   ./rk_camera [裝置] [寬] [高] [input_size] [幀數] [slot數]
//               [--fourcc UYVY|YUYV|MJPG] [--fps 60] [--convert]
//   ./rk_camera 0 1920 1080 640 600 3 --fourcc UYVY --fps 60

#include "hls_resize.hpp"

#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_running{true};
void on_signal(int) { g_running = false; }

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t).count();
}

std::string fourcc_str(int f) {
    char b[5] = {char(f & 0xFF), char((f >> 8) & 0xFF),
                 char((f >> 16) & 0xFF), char((f >> 24) & 0xFF), 0};
    return b;
}

// 原始 YUV 422 影格 -> BGR 的轉換碼。回傳 -1 表示不需要 / 不支援。
int conversion_for(const std::string& fcc) {
    if (fcc == "UYVY" || fcc == "HDYC") return cv::COLOR_YUV2BGR_UYVY;
    if (fcc == "YUYV" || fcc == "YUY2") return cv::COLOR_YUV2BGR_YUY2;
    return -1;
}

// slot 的所有權在兩條執行緒間流轉:
//   free -> 擷取端寫入 -> filled -> 處理端讀取 -> free
// 任何時刻一個 slot 只屬於一方,影像資料本身不需要上鎖。
class SlotQueue {
public:
    explicit SlotQueue(int n) { for (int i = 0; i < n; ++i) free_.push_back(i); }

    int acquire_free(int wait_ms = -1) {
        std::unique_lock<std::mutex> lk(m_);
        if (wait_ms == 0) { if (free_.empty()) return -1; }
        else if (wait_ms < 0) cv_free_.wait(lk, [&] { return !free_.empty() || stop_; });
        else cv_free_.wait_for(lk, std::chrono::milliseconds(wait_ms),
                               [&] { return !free_.empty() || stop_; });
        if (free_.empty()) return -1;
        const int s = free_.front(); free_.pop_front(); return s;
    }
    void publish(int slot) {
        { std::lock_guard<std::mutex> lk(m_);
          filled_.push_back(slot);
          if (filled_.size() > peak_) peak_ = filled_.size(); }
        cv_filled_.notify_one();
    }
    int acquire_filled(int wait_ms = 200) {
        std::unique_lock<std::mutex> lk(m_);
        cv_filled_.wait_for(lk, std::chrono::milliseconds(wait_ms),
                            [&] { return !filled_.empty() || stop_; });
        if (filled_.empty()) return -1;
        const int s = filled_.front(); filled_.pop_front(); return s;
    }
    void recycle(int slot) {
        { std::lock_guard<std::mutex> lk(m_); free_.push_back(slot); }
        cv_free_.notify_one();
    }
    void stop() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_free_.notify_all(); cv_filled_.notify_all();
    }
    size_t peak() const { std::lock_guard<std::mutex> lk(m_); return peak_; }

private:
    mutable std::mutex m_;
    std::condition_variable cv_free_, cv_filled_;
    std::deque<int> free_, filled_;
    size_t peak_ = 0;
    bool stop_ = false;
};

struct Stats {
    std::atomic<long long> captured{0}, processed{0}, dropped{0}, read_fail{0};
    std::mutex m;
    double acc_grab = 0, acc_retrieve = 0, acc_convert = 0, acc_proc = 0;
};

}  // namespace


int main(int argc, char** argv) {
    // ---- 參數 ----
    std::vector<const char*> pos;
    std::string want_fcc = "UYVY";
    double want_fps = 60.0;
    bool let_opencv_convert = false;    // --convert:讓 OpenCV 自己轉,較慢
    int  bufsize = 4;                   // V4L2 buffer 數,見下方說明
    int  cv_threads = 0;                // 0 = 不動 OpenCV 的執行緒數

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--fourcc") && i + 1 < argc)      want_fcc = argv[++i];
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc)    want_fps = atof(argv[++i]);
        else if (!strcmp(argv[i], "--convert"))                let_opencv_convert = true;
        else if (!strcmp(argv[i], "--bufsize") && i + 1 < argc) bufsize = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) cv_threads = atoi(argv[++i]);
        else pos.push_back(argv[i]);
    }
    const char* dev   = pos.size() > 0 ? pos[0] : "0";
    const int   w     = pos.size() > 1 ? atoi(pos[1]) : 1920;
    const int   h     = pos.size() > 2 ? atoi(pos[2]) : 1080;
    const int   isize = pos.size() > 3 ? atoi(pos[3]) : 640;
    const int   nfrm  = pos.size() > 4 ? atoi(pos[4]) : 600;
    const int   nslot = pos.size() > 5 ? atoi(pos[5]) : 3;

    std::signal(SIGINT, on_signal);

    hls::use_dma_heap("reserved", 64u * 1024 * 1024);
    hls::resize::use_devmem();

    printf("Pool: %s\n", hls::pool_info().c_str());
    if (!hls::resize::available()) {
        fprintf(stderr, "IP 不可用: %s\n", hls::resize::last_error().c_str());
        return 1;
    }

    // ---- 相機 ----
    cv::VideoCapture cap;
    const bool numeric = (strlen(dev) == 1 && dev[0] >= '0' && dev[0] <= '9');
    if (numeric) cap.open(dev[0] - '0', cv::CAP_V4L2);
    else         cap.open(dev, cv::CAP_V4L2);
    if (!cap.isOpened()) { fprintf(stderr, "開不了相機 %s\n", dev); return 1; }

    // 順序有講究:先 FOURCC,再解析度,最後 fps。
    // 反過來設的話,驅動可能因為當下格式不支援該解析度而拒絕。
    if (want_fcc.size() == 4)
        cap.set(cv::CAP_PROP_FOURCC,
                cv::VideoWriter::fourcc(want_fcc[0], want_fcc[1], want_fcc[2], want_fcc[3]));
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    cap.set(cv::CAP_PROP_FPS, want_fps);
    // *** 這個值決定能不能跑滿幀率 ***
    // 只有 1 個 buffer 時,我們持有它的期間相機無法擷取下一幀,
    // 週期變成「處理時間 + 一個影格間隔」,直接砍半幀率。
    // v4l2-ctl --stream-mmap 預設用 4 個,所以它量得到 60fps。
    // 代價是佇列變深、延遲增加,但 4 個仍在可接受範圍。
    if (bufsize > 0) cap.set(cv::CAP_PROP_BUFFERSIZE, bufsize);

    if (cv_threads > 0) cv::setNumThreads(cv_threads);

    // 關掉 OpenCV 的自動轉換,拿原始影格自己處理
    if (!let_opencv_convert) cap.set(cv::CAP_PROP_CONVERT_RGB, 0);

    const int aw  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int ah  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const std::string fcc = fourcc_str(static_cast<int>(cap.get(cv::CAP_PROP_FOURCC)));
    const double cam_fps  = cap.get(cv::CAP_PROP_FPS);

    printf("相機: %dx%d  %s  %.1f fps  (V4L2 buffer %d,OpenCV 執行緒 %d)\n",
           aw, ah, fcc.c_str(), cam_fps, bufsize, cv::getNumThreads());
    if (fcc != want_fcc)
        printf("  [!] 要求 %s 但拿到 %s —— 驅動不支援,已退回它選的格式\n",
               want_fcc.c_str(), fcc.c_str());

    const double mbps = aw * (double)ah * 2 * cam_fps / 1e6;
    if (fcc == "UYVY" || fcc == "YUYV")
        printf("  頻寬需求約 %.0f MB/s(%s)\n", mbps,
               mbps > 45 ? "USB 2.0 撐不住,必須接 USB 3.0 埠" : "USB 2.0 可行");

    const int cvt = let_opencv_convert ? -1 : conversion_for(fcc);
    if (!let_opencv_convert && cvt < 0 && fcc != "BGR3" && fcc != "RGB3") {
        fprintf(stderr,
                "\n格式 %s 沒有對應的直接轉換路徑。\n"
                "加上 --convert 讓 OpenCV 自己轉(較慢),或改用 UYVY / YUYV。\n",
                fcc.c_str());
        return 1;
    }

    // ---- DMA 輸入緩衝 ----
    std::vector<cv::Mat> slots(nslot);
    for (int i = 0; i < nslot; ++i) {
        slots[i] = hls::resize::input_buffer(aw, ah, i);
        if (slots[i].empty()) {
            fprintf(stderr, "slot %d 配置失敗,每個需要 %.1f MB\n",
                    i, aw * ah * 3 / 1048576.0);
            return 1;
        }
    }
    printf("已配置 %d 個 DMA 輸入緩衝,共 %.1f MB\n",
           nslot, nslot * aw * ah * 3 / 1048576.0);
    // 每個 slot 各自一份原始影格緩衝。
    // retrieve 本來就會把資料複製到我們給的 Mat,所以用 N 份就能讓
    // 轉換移到處理執行緒去做,不需要額外再複製一次。
    std::vector<cv::Mat> raws(nslot);

    printf("色彩轉換: %s(在處理執行緒)\n\n",
           cvt >= 0 ? "cvtColor 直接輸出到 DMA(免額外複製)"
                    : "OpenCV 內建轉換 + 複製到 DMA");

    SlotQueue q(nslot);
    Stats st;
    std::atomic<bool> direct_ok{true};
    const auto t_start = std::chrono::steady_clock::now();

    // ================= 擷取執行緒 =================
    std::thread producer([&] {
        while (g_running && st.captured < nfrm) {
            const int slot = q.acquire_free(0);   // 拿不到就丟幀,避免延遲累積

            auto t0 = std::chrono::steady_clock::now();
            const bool grabbed = cap.grab();
            const double grab_ms = ms_since(t0);

            if (!grabbed) { ++st.read_fail; if (slot >= 0) q.recycle(slot); continue; }
            if (slot < 0) { ++st.dropped; continue; }

            // retrieve 到這個 slot 專屬的緩衝。色彩轉換留給處理執行緒,
            // 擷取端才能儘快把 V4L2 buffer 還給驅動去拍下一幀。
            t0 = std::chrono::steady_clock::now();
            const bool got = cap.retrieve(raws[slot]);
            const double retrieve_ms = ms_since(t0);
            if (!got || raws[slot].empty()) { ++st.read_fail; q.recycle(slot); continue; }

            { std::lock_guard<std::mutex> lk(st.m);
              st.acc_grab     += grab_ms;
              st.acc_retrieve += retrieve_ms; }
            ++st.captured;
            q.publish(slot);
        }
        q.stop();
    });

    // ================= 處理執行緒 =================
    std::thread consumer([&] {
        hls::resize::Result res;
        bool first = true;

        while (true) {
            const int slot = q.acquire_filled(200);
            if (slot < 0) {
                if (!g_running || st.captured >= nfrm) break;
                continue;
            }

            // 色彩轉換,輸出直接落在 DMA slot
            auto t0 = std::chrono::steady_clock::now();
            const uint8_t* before = slots[slot].data;
            if (cvt >= 0)
                cv::cvtColor(raws[slot], slots[slot], cvt);
            else if (raws[slot].size() == slots[slot].size() && raws[slot].type() == CV_8UC3)
                raws[slot].copyTo(slots[slot]);
            else
                cv::resize(raws[slot], slots[slot], slots[slot].size());
            const double convert_ms = ms_since(t0);

            if (slots[slot].data != before) {   // 被重新配置,DMA 位址失效
                direct_ok = false;
                slots[slot] = hls::resize::input_buffer(aw, ah, slot);
            }

            t0 = std::chrono::steady_clock::now();
            hls::resize::letterbox(slots[slot], isize, res);
            const double proc_ms = ms_since(t0);

            q.recycle(slot);        // 儘早歸還,讓擷取端繼續

            { std::lock_guard<std::mutex> lk(st.m);
              st.acc_convert += convert_ms;
              st.acc_proc    += proc_ms; }
            ++st.processed;

            if (first) {
                first = false;
                printf("第一幀:\n");
                printf("  加速 : %s", res.used_ip ? "IP" : "CPU");
                if (res.used_ip) printf(" %dx -> %dx%d", res.ip_scale, res.mid_w, res.mid_h);
                printf("\n  說明 : %s\n", res.reason);
                printf("  明細 : 複製 %.3f%s | cache %.3f | IP %.3f | 後處理 %.3f ms\n",
                       res.timing.copy_ms, res.zero_copy ? "(zero-copy)" : "",
                       res.timing.sync_ms, res.timing.run_ms, res.timing.post_ms);
                cv::imwrite("frame0.png", res.img);
                printf("  已存 frame0.png\n\n");
            }

            // 這裡接後續處理(推論、顯示、編碼…)
        }
    });

    producer.join();
    consumer.join();

    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t_start).count();
    const long long cn = st.captured.load(), pn = st.processed.load();

    printf("擷取 %lld 幀,處理 %lld 幀\n", cn, pn);
    printf("  丟棄(無空 slot)  %lld\n", st.dropped.load());
    printf("  讀取失敗           %lld\n", st.read_fail.load());
    printf("  佇列最深           %zu\n", q.peak());
    if (!direct_ok)
        printf("  [!] 轉換過程重新配置了緩衝,zero-copy 已失效\n");

    if (cn && pn) {
        const double g = st.acc_grab / cn, r = st.acc_retrieve / cn;
        const double c = st.acc_convert / cn, p = st.acc_proc / pn;

        const double cap_total  = g + r;
        const double proc_total = c + p;
        const double budget = cam_fps > 0 ? 1000.0 / cam_fps : 0;

        printf("\n擷取執行緒(合計 %.3f ms):\n", cap_total);
        printf("  cap.grab()      %.3f ms   等待相機 + 出佇列\n", g);
        printf("  cap.retrieve()  %.3f ms   取出影格\n", r);
        printf("處理執行緒(合計 %.3f ms):\n", proc_total);
        printf("  色彩轉換 -> DMA %.3f ms   %s\n", c,
               cvt >= 0 ? "(轉換與搬移合一)" : "");
        printf("  letterbox       %.3f ms\n", p);

        printf("\n端到端吞吐:%.1f FPS(%.3f ms/幀)\n", pn / wall, wall * 1000.0 / pn);

        printf("\n瓶頸判讀(每幀預算 %.2f ms @%.0f fps):\n", budget, cam_fps);
        if (budget > 0 && proc_total > budget)
            printf("  處理端 %.2f ms 超過預算 -> 色彩轉換是瓶頸。\n"
                   "  試 --threads 4 讓 cvtColor 用多核,或降解析度。\n", proc_total);
        else if (budget > 0 && cap_total > budget)
            printf("  擷取端 %.2f ms 超過預算。retrieve 若偏高,那是 OpenCV\n"
                   "  把影格複製給我們的成本;grab 偏高則是在等相機。\n", cap_total);
        else
            printf("  兩端都在預算內。若 FPS 仍不足,提高 --bufsize 或查 USB 連線速率。\n");
    }
    return 0;
}