#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>

// ============================================================================
//  Camera Process
// ============================================================================

/**
 * Camera — 封裝 cv::VideoCapture 的輕量包裝類別
 *
 * 預設行為：open() 時自動要求驅動使用最高解析度與最高 FPS，
 * 實際值由硬體決定，可透過 actualWidth() / actualHeight() / actualFps() 查詢。
 *
 * 使用方式：
 *   Camera cam(0);
 *   cam.open();
 *   cv::Mat frame;
 *   while (cam.nextFrame(frame)) {
 *       // 使用 frame ...
 *   }
 *   cam.close();
 */
class Camera {
public:
    struct Config {
        int    index  = 0;
        int    width  = 640;
        int    height = 480;
        double fps    = 60.0;
        std::string fourcc = "MJPG";   // 新增；空字串 = 不設定
    };
 
    explicit Camera(int index = 0);
    explicit Camera(const Config& cfg);
    ~Camera();
 
    // 禁止複製（VideoCapture 不可複製）
    Camera(const Camera&)            = delete;
    Camera& operator=(const Camera&) = delete;
 
    /** 開啟攝影機，成功回傳 true */
    bool open();
 
    /** 擷取下一幀；成功回傳 true，frame 內含影像資料 */
    bool nextFrame(cv::Mat& frame);
 
    /** 關閉攝影機並釋放資源 */
    void close();
 
    /** 查詢攝影機是否已開啟 */
    bool isOpened() const;
 
    // --- 實際套用的參數（open() 後才有效） ---
    int    actualWidth()  const { return m_actualWidth;  }
    int    actualHeight() const { return m_actualHeight; }
    double actualFps()    const { return m_actualFps;    }
 
private:
    Config             m_cfg;
    cv::VideoCapture   m_cap;
    int                m_actualWidth  = 0;
    int                m_actualHeight = 0;
    double             m_actualFps    = 0.0;
    std::string        m_actualFourcc = "MJPG";
};

/**
 * FrameGrabber — 在背景執行緒持續擷取攝影機影像的雙緩衝包裝
 *
 * 用途：解耦「攝影機擷取」與「主執行緒推理」，避免 cap.read() 的
 *       阻塞時間直接累加到主迴圈總耗時上。
 *
 * 使用方式：
 *   FrameGrabber grabber(cam);
 *   grabber.start();
 *
 *   cv::Mat frame;
 *   while (g_running) {
 *       if (!grabber.getLatest(frame)) {
 *           continue;  // 還沒有任何一幀，稍等
 *       }
 *       // 用 frame 做推理 ...
 *   }
 *
 *   grabber.stop();
 *
 * 注意事項：
 *   - getLatest() 永遠回傳「目前為止抓到的最新一幀」的複製，
 *     若主執行緒處理速度比擷取慢，會自動跳過（drop）中間的舊幀。
 *   - 若擷取執行緒尚未取得任何畫面，getLatest() 回傳 false。
 *   - frameId() 可用來判斷主執行緒是否拿到「新的」一幀
 *     （避免處理重複幀），用法見下方範例。
 */
class FrameGrabber {
public:
    explicit FrameGrabber(Camera& cam) : m_cam(cam) {}
 
    ~FrameGrabber() { stop(); }
 
    FrameGrabber(const FrameGrabber&)            = delete;
    FrameGrabber& operator=(const FrameGrabber&) = delete;
 
    /** 啟動背景擷取執行緒 */
    void start()
    {
        if (m_running.exchange(true)) {
            return;  // 已經在跑了
        }
        m_thread = std::thread(&FrameGrabber::captureLoop, this);
    }
 
    /** 停止背景執行緒並等待結束 */
    void stop()
    {
        if (!m_running.exchange(false)) {
            return;  // 本來就沒在跑
        }
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }
 
    /**
     * 取得最新一幀的複製。
     * @return true 表示 frame 有效；false 表示尚未擷取到任何畫面。
     */
    bool getLatest(cv::Mat& frame)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_latest.empty()) {
            return false;
        }
        frame = m_latest.clone();
        return true;
    }
 
    /**
     * 取得最新一幀的複製，並回傳該幀的序號（frame id）。
     * 可搭配外部變數比較，避免重複處理同一幀。
     *
     * 範例：
     *   long long lastId = -1, curId;
     *   while (...) {
     *       if (!grabber.getLatest(frame, curId)) continue;
     *       if (curId == lastId) continue;  // 還是同一幀，跳過
     *       lastId = curId;
     *       // 處理 frame ...
     *   }
     */
    bool getLatest(cv::Mat& frame, long long& outFrameId)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_latest.empty()) {
            return false;
        }
        frame = m_latest.clone();
        outFrameId = m_frameId;
        return true;
    }
 
    /** 擷取執行緒目前抓到的總幀數 */
    long long frameId() const { return m_frameId; }
 
private:
    void captureLoop()
    {
        cv::Mat tmp;
        while (m_running) {
            if (!m_cam.nextFrame(tmp)) {
                // 讀取失敗：稍微等待後重試，避免 busy loop 洗版錯誤訊息
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
 
            std::lock_guard<std::mutex> lock(m_mutex);
            m_latest = std::move(tmp);
            ++m_frameId;
        }
    }
 
    Camera&                 m_cam;
    std::thread             m_thread;
    std::atomic<bool>       m_running{false};
 
    std::mutex              m_mutex;
    cv::Mat                 m_latest;
    long long               m_frameId = 0;
};
 