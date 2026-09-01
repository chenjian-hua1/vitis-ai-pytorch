#pragma once

#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// ============================================================================
//  Camera — cv::VideoCapture 的輕量包裝
// ============================================================================
//
//  Config 的兩個選項:
//
//  1. buffer_count — V4L2 的 buffer 數,預設 4(與 OpenCV 相同)。
//     絕對不要設成 1:只有一個 buffer 時,我們持有它的期間相機無法
//     擷取下一幀,週期會變成「處理時間 + 一個影格間隔」,幀率砍半。
//     設 0 表示不動,交給 OpenCV 決定。
//
//  2. raw_output — 關掉 OpenCV 內建的色彩轉換,直接拿原始影格。
//     UYVY/YUYV → BGR 的轉換若留在 read() 裡,整段時間都算在擷取上。
//     關掉之後由呼叫端自己轉,可以搬到別的階段做。
//     只有 UYVY / YUYV 適用;MJPG 的原始資料是壓縮位元流,不能這樣用。
//
class Camera {
public:
    struct Config {
        int         index  = 0;
        int         width  = 640;
        int         height = 480;
        double      fps    = 60.0;
        std::string fourcc = "MJPG";   // 空字串 = 不設定
        int         buffer_count = 4;  // 0 = 不設定
        bool        raw_output   = false;
    };

    explicit Camera(int index = 0);
    explicit Camera(const Config& cfg);
    ~Camera();

    Camera(const Camera&)            = delete;
    Camera& operator=(const Camera&) = delete;

    bool open();

    // 取下一幀 —— V4L2 佇列中「最舊」的那一幀
    bool nextFrame(cv::Mat& frame);

    // 取「目前最新」的一幀,先把佇列裡積壓的舊幀丟掉。
    //
    // V4L2 的 buffer 佇列是 FIFO,read() 永遠給最舊的一幀。下游一慢
    // 就會排滿,之後每次讀到的都是 N 幀之前的畫面 —— 內容正確,但
    // 一直落後。這個函式主動把積壓推掉,讓你永遠拿到剛拍到的那一幀。
    //
    // 怎麼知道該推幾幀:用「離開的時間」估算。上次取幀到現在若過了
    // D 毫秒,期間大約產出 floor(D / 影格間隔) 幀,其中只有最後一幀
    // 值得留 —— 所以推掉 n-1 幀再取。
    //
    // 不用「grab() 是否立刻返回」來判斷,是因為那個規則有個致命缺陷:
    // 我們忙碌期間產出的那一幀會讓下一次 grab() 立刻返回,於是被判定
    // 成「舊的」而丟掉,接著再空等一整個間隔。但那幀其實只有「忙碌
    // 時間」那麼舊,遠比再等一個間隔新鮮。實測會讓擷取週期變成兩倍。
    //
    // @param skipped  傳回這次推掉幾幀(可為 nullptr)
    bool nextFrameLatest(cv::Mat& frame, int* skipped = nullptr);

    // 只推進一幀,不取出資料。用於「這幀不要了」的情況 —— 比讀到
    // 暫存 Mat 省下一次全幀複製,同時讓 buffer 回到佇列,不會把相機卡住。
    bool skipFrame();

    void close();
    bool isOpened() const;

    // --- open() 後才有效 ---
    int    actualWidth()  const { return m_actualWidth;  }
    int    actualHeight() const { return m_actualHeight; }
    double actualFps()    const { return m_actualFps;    }
    double frameIntervalMs() const { return m_frameIntervalMs; }
    const std::string& actualFourcc() const { return m_actualFourcc; }

    // raw_output 生效時,影格的型別(UYVY/YUYV 為 CV_8UC2)
    int rawType() const { return m_rawMode ? CV_8UC2 : CV_8UC3; }

    // 把 rawType() 的影格轉成 BGR 所需的 cv::COLOR_* 代碼;
    // -1 表示不需要轉換(已經是 BGR)
    int conversionCode() const { return m_cvtCode; }

    bool rawMode() const { return m_rawMode; }

private:
    Config           m_cfg;
    cv::VideoCapture m_cap;
    int              m_actualWidth  = 0;
    int              m_actualHeight = 0;
    double           m_actualFps    = 0.0;
    std::string      m_actualFourcc;
    bool             m_rawMode = false;
    int              m_cvtCode = -1;
    double           m_frameIntervalMs = 16.7;   // 由 actualFps 推得
    std::chrono::steady_clock::time_point m_lastCapture{};
    bool             m_haveLastCapture = false;
};


// ============================================================================
//  FrameGrabber — 背景擷取,只保留「最新一幀」
// ============================================================================
//
//  設計重點:
//
//  1. 三個緩衝,零複製交接。
//     一個處理端正在讀、一個擷取端正在寫、一個已寫好等著被取。
//     取用時只是拿到緩衝的參照,不做任何複製 —— 舊版的 getLatest()
//     在鎖裡做 clone(),1080p 每次就是 6 MB,而且會卡住擷取執行緒。
//
//  2. 擷取端永不阻塞。
//     前一幀若還沒被取走就直接覆蓋。這是刻意的:對即時影像來說
//     舊幀沒有保留價值,積壓只會讓延遲愈來愈大。
//
//  3. 內部用 Camera::nextFrameLatest(),所以連 V4L2 驅動那層的
//     積壓也一併清掉。兩層加起來,處理端拿到的永遠是最新畫面。
//
//  4. 可在擷取執行緒裡做一次轉換(setTransform)。
//     擷取端大部分時間阻塞在等相機,CPU 是閒著的 —— 把色彩轉換這類
//     固定成本挪過來,等於填進空等的時間,處理端則實質變快。
//     只要「retrieve + 轉換」不超過一個影格間隔,擷取端就不會變慢。
//
//  5. 緩衝可由外部提供(setBuffers)。
//     需要輸出落在 DMA 記憶體時很有用 —— 例如轉換完要直接餵給
//     PL 上的 resize IP,緩衝就得是 DMA 配出來的,否則會失去 zero-copy。
//
//  用法:
//     FrameGrabber grabber(cam);
//     grabber.setBuffers({dma0, dma1, dma2});          // 選用
//     grabber.setTransform([&](const cv::Mat& s, cv::Mat& d){
//         cv::cvtColor(s, d, code);                    // 選用
//     });
//     grabber.start();
//     while (running) {
//         auto f = grabber.acquire(200);      // RAII,離開作用域自動歸還
//         if (!f.valid()) continue;
//         process(f.mat(), f.timestamp());
//     }
//     grabber.stop();
//
class FrameGrabber {
public:
    // 取得的一幀。生命週期內緩衝不會被覆蓋;請儘早讓它離開作用域。
    class Handle {
    public:
        Handle() = default;
        Handle(FrameGrabber* g, int idx, double t, long long id)
            : g_(g), idx_(idx), t_(t), id_(id) {}
        ~Handle() { if (g_) g_->release(); }

        Handle(Handle&& o) noexcept { swap(o); }
        Handle& operator=(Handle&& o) noexcept { swap(o); return *this; }
        Handle(const Handle&)            = delete;
        Handle& operator=(const Handle&) = delete;

        bool valid() const { return g_ != nullptr && idx_ >= 0; }
        const cv::Mat& mat() const { return g_->m_buf[idx_]; }
        double    timestamp() const { return t_; }   // steady_clock 的秒數
        long long id()        const { return id_; }

    private:
        void swap(Handle& o) {
            std::swap(g_, o.g_); std::swap(idx_, o.idx_);
            std::swap(t_, o.t_); std::swap(id_, o.id_);
        }
        FrameGrabber* g_ = nullptr;
        int           idx_ = -1;
        double        t_ = 0;
        long long     id_ = 0;
    };

    explicit FrameGrabber(Camera& cam) : m_cam(cam) {}
    ~FrameGrabber() { stop(); }

    FrameGrabber(const FrameGrabber&)            = delete;
    FrameGrabber& operator=(const FrameGrabber&) = delete;

    // 以下兩個必須在 start() 之前呼叫。

    // 由外部提供三個緩衝(需已配置好、尺寸型別一致)。
    // 不呼叫的話 start() 會依相機的尺寸自行配置。
    bool setBuffers(const std::vector<cv::Mat>& bufs);

    // 在擷取執行緒裡對每一幀做的轉換。
    // src 是相機的原始影格,dst 是要發布出去的緩衝。
    // dst 已配置好,轉換函式不可讓它重新配置(cvtColor 尺寸型別相符時不會)。
    void setTransform(std::function<void(const cv::Mat& src, cv::Mat& dst)> fn) {
        m_transform = std::move(fn);
    }

    void start();
    void stop();

    // 等到有新的一幀為止。逾時或已停止時回傳無效的 Handle。
    Handle acquire(int wait_ms = 200);

    // 相容用:複製一份出來。會多一次全幀複製,新程式請用 acquire()。
    bool getLatest(cv::Mat& frame);
    bool getLatest(cv::Mat& frame, long long& outFrameId);

    // 擷取執行緒單次取幀的平均耗時。主要用來判斷是否純粹在等相機:
    // 若接近一個影格間隔,代表相機是瓶頸,軟體端沒得優化。
    double avgCaptureMs() const {
        const long long n = m_capCount.load();
        return n ? m_capSumMs.load() / static_cast<double>(n) : 0.0;
    }
    double maxCaptureMs() const { return m_capMaxMs.load(); }

    // 把擷取執行緒綁到指定核心。必須在 start() 之後呼叫。
    // cpu < 0 表示不綁。回傳 false 表示平台不支援或設定失敗。
    bool pinThread(int cpu);

    long long frameId()      const { return m_frameId.load(); }
    // 處理端來不及、被擷取端直接覆蓋掉的幀數
    long long overwritten()  const { return m_overwritten.load(); }
    // 為了取到最新而從 V4L2 佇列推掉的舊幀數
    long long staleSkipped() const { return m_staleSkipped.load(); }

private:
    friend class Handle;

    void captureLoop();
    void release();
    int  pickFree() const;      // 呼叫端須持有 m_mutex

    Camera&           m_cam;
    std::thread       m_thread;
    std::atomic<bool> m_running{false};

    mutable std::mutex      m_mutex;
    std::condition_variable m_cv;
    cv::Mat   m_buf[3];
    cv::Mat   m_scratch;     // setTransform 時,存放相機的原始影格
    std::function<void(const cv::Mat&, cv::Mat&)> m_transform;
    bool      m_extBuffers = false;
    double    m_lastWorkMs = 0.0;   // 上一輪轉換花的時間,用來判斷是否落後
    int       m_write = 0, m_ready = -1, m_reading = -1;
    double    m_tCap = 0;
    long long m_readyId = 0;

    std::atomic<double>    m_capSumMs{0.0};
    std::atomic<double>    m_capMaxMs{0.0};
    std::atomic<long long> m_capCount{0};

    std::atomic<long long> m_frameId{0};
    std::atomic<long long> m_overwritten{0};
    std::atomic<long long> m_staleSkipped{0};
};