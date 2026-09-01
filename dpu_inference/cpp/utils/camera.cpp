#include "camera.h"

#include <iostream>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

// ============================================================================
//  Camera
// ============================================================================

Camera::Camera(int index)          { m_cfg.index = index; }
Camera::Camera(const Config& cfg) : m_cfg(cfg) {}
Camera::~Camera()                  { close(); }

static std::string fourcc_to_str(double v)
{
    const int f = static_cast<int>(v);
    char s[5] = { char(f & 0xFF), char((f >> 8) & 0xFF),
                  char((f >> 16) & 0xFF), char((f >> 24) & 0xFF), '\0' };
    return std::string(s);
}

bool Camera::open()
{
    m_cap.open(m_cfg.index, cv::CAP_V4L2);
    if (!m_cap.isOpened()) {
        std::cerr << "[Camera] 無法開啟攝影機 (index=" << m_cfg.index << ")\n";
        return false;
    }

    // FOURCC 要「先」設定:V4L2 換格式時會重置可用的解析度/FPS 清單,
    // 放在 width/height 之後設會把前面設好的值蓋掉。
    if (m_cfg.fourcc.size() == 4) {
        m_cap.set(cv::CAP_PROP_FOURCC,
                  cv::VideoWriter::fourcc(m_cfg.fourcc[0], m_cfg.fourcc[1],
                                          m_cfg.fourcc[2], m_cfg.fourcc[3]));
    }
    m_cap.set(cv::CAP_PROP_FRAME_WIDTH,  m_cfg.width);
    m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, m_cfg.height);
    m_cap.set(cv::CAP_PROP_FPS,          m_cfg.fps);

    if (m_cfg.buffer_count > 0)
        m_cap.set(cv::CAP_PROP_BUFFERSIZE, m_cfg.buffer_count);

    m_actualWidth  = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    m_actualHeight = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_actualFps    = m_cap.get(cv::CAP_PROP_FPS);
    m_actualFourcc = fourcc_to_str(m_cap.get(cv::CAP_PROP_FOURCC));
    m_frameIntervalMs = (m_actualFps > 1.0) ? 1000.0 / m_actualFps : 16.7;

    // 決定要不要走 raw 模式
    m_rawMode = false;
    m_cvtCode = -1;
    if (m_cfg.raw_output) {
        if (m_actualFourcc == "UYVY")
            m_cvtCode = cv::COLOR_YUV2BGR_UYVY;
        else if (m_actualFourcc == "YUYV" || m_actualFourcc == "YUY2")
            m_cvtCode = cv::COLOR_YUV2BGR_YUY2;

        if (m_cvtCode >= 0) {
            m_cap.set(cv::CAP_PROP_CONVERT_RGB, 0);
            m_rawMode = (m_cap.get(cv::CAP_PROP_CONVERT_RGB) == 0);
            if (!m_rawMode) {
                m_cvtCode = -1;
                std::cerr << "[Camera] 驅動不接受 CONVERT_RGB=0,仍由 OpenCV 轉換\n";
            }
        } else {
            std::cerr << "[Camera] " << m_actualFourcc
                      << " 沒有對應的手動轉換路徑,維持 OpenCV 內建轉換\n";
        }
    }

    std::cout << "[Camera] 開啟成功  " << m_actualWidth << "x" << m_actualHeight
              << " @ " << m_actualFps << " fps  [" << m_actualFourcc << "]"
              << "  buffer=" << (m_cfg.buffer_count > 0
                                 ? std::to_string(m_cfg.buffer_count) : "預設")
              << "  " << (m_rawMode ? "raw(呼叫端自行轉換)" : "OpenCV 轉換為 BGR")
              << std::endl;

    if (m_cfg.fourcc.size() == 4 && m_actualFourcc != m_cfg.fourcc)
        std::cerr << "[Camera] 警告: 要求 " << m_cfg.fourcc
                  << " 但驅動實際給 " << m_actualFourcc << std::endl;

    return true;
}

bool Camera::nextFrame(cv::Mat& frame)
{
    if (!m_cap.isOpened()) {
        std::cerr << "[Camera] 尚未開啟\n";
        return false;
    }
    if (!m_cap.read(frame) || frame.empty()) {
        std::cerr << "[Camera] 無法擷取影像\n";
        return false;
    }
    return true;
}

bool Camera::skipFrame()
{
    if (!m_cap.isOpened()) return false;
    // grab() 只把 buffer 出佇列再還回去,不做 retrieve 的資料複製
    return m_cap.grab();
}

bool Camera::nextFrameLatest(cv::Mat& frame, int* skipped)
{
    if (!m_cap.isOpened()) {
        std::cerr << "[Camera] 尚未開啟\n";
        if (skipped) *skipped = 0;
        return false;
    }

    const auto now = std::chrono::steady_clock::now();

    // 依「離開的時間」估算佇列裡積了幾幀。
    // 只有在我們慢於相機時才需要推 —— 快於相機時佇列是空的,
    // grab() 會直接等下一幀,那才是最新的,不該再推。
    int to_skip = 0;
    if (m_haveLastCapture) {
        const double away = std::chrono::duration<double, std::milli>(
                                now - m_lastCapture).count();
        const int arrived = static_cast<int>(away / m_frameIntervalMs);
        to_skip = arrived - 1;                 // 留最後一幀
        if (to_skip < 0) to_skip = 0;

        const int cap = (m_cfg.buffer_count > 0 ? m_cfg.buffer_count : 4) - 1;
        if (to_skip > cap) to_skip = cap;      // 佇列不可能比 buffer 數還深
    }

    for (int i = 0; i < to_skip; ++i) {
        if (!m_cap.grab()) {
            std::cerr << "[Camera] grab 失敗\n";
            if (skipped) *skipped = i;
            return false;
        }
    }
    if (skipped) *skipped = to_skip;

    if (!m_cap.grab()) {
        std::cerr << "[Camera] grab 失敗\n";
        return false;
    }
    if (!m_cap.retrieve(frame) || frame.empty()) {
        std::cerr << "[Camera] retrieve 失敗\n";
        return false;
    }

    m_lastCapture = std::chrono::steady_clock::now();
    m_haveLastCapture = true;
    return true;
}

void Camera::close()
{
    if (m_cap.isOpened()) {
        m_cap.release();
        std::cout << "[Camera] 已關閉" << std::endl;
    }
}

bool Camera::isOpened() const { return m_cap.isOpened(); }


// ============================================================================
//  FrameGrabber
// ============================================================================

bool FrameGrabber::setBuffers(const std::vector<cv::Mat>& bufs)
{
    if (m_running.load()) return false;      // 必須在 start() 之前
    if (bufs.size() != 3) return false;
    for (const auto& b : bufs)
        if (b.empty() || b.size() != bufs[0].size() || b.type() != bufs[0].type())
            return false;

    std::lock_guard<std::mutex> lk(m_mutex);
    for (int i = 0; i < 3; ++i) m_buf[i] = bufs[static_cast<size_t>(i)];
    m_extBuffers = true;
    return true;
}

void FrameGrabber::start()
{
    if (m_running.exchange(true)) return;    // 已經在跑了

    {
        std::lock_guard<std::mutex> lk(m_mutex);
        // 外部沒給緩衝的話,依相機的尺寸與型別自行配置
        if (!m_extBuffers) {
            for (int i = 0; i < 3; ++i)
                m_buf[i].create(m_cam.actualHeight(), m_cam.actualWidth(),
                                m_cam.rawType());
        }
        // 有轉換時,原始影格需要一塊暫存;它只被擷取執行緒使用,一份就夠
        if (m_transform)
            m_scratch.create(m_cam.actualHeight(), m_cam.actualWidth(),
                             m_cam.rawType());
        m_write = 0; m_ready = -1; m_reading = -1;
    }
    m_thread = std::thread(&FrameGrabber::captureLoop, this);
}

void FrameGrabber::stop()
{
    if (!m_running.exchange(false)) return;  // 本來就沒在跑
    m_cv.notify_all();                       // 叫醒可能卡在 acquire 的等待
    if (m_thread.joinable()) m_thread.join();
}

int FrameGrabber::pickFree() const
{
    // 選一個既不是處理端在讀、也不是已就緒的緩衝。
    // 三個緩衝最多兩個被佔,所以一定找得到。
    for (int i = 0; i < 3; ++i)
        if (i != m_reading && i != m_ready) return i;
    return 0;
}

void FrameGrabber::captureLoop()
{
    while (m_running) {
        cv::Mat* target = nullptr;
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            target = &m_buf[m_write];
        }

        int skipped = 0;
        const auto tc0 = std::chrono::steady_clock::now();

        // 要不要清佇列,是自適應決定的。
        //
        // nextFrameLatest 會把「已經在佇列裡」的幀當成舊幀推掉,再等
        // 一張全新的。這在單執行緒時代是對的,但有專屬擷取執行緒之後
        // 就變成負擔:轉換完回來時佇列剛好有一張新幀,卻被丟掉再等一輪,
        // 週期變成「影格間隔 + 工作時間」而不是兩者取大。
        //
        // 只有「上一輪的工作時間超過影格間隔」時才需要清 —— 那才表示
        // 我們真的落後、佇列真的積了東西。
        const double interval = m_cam.frameIntervalMs();
        const bool behind = (m_lastWorkMs > interval);

        bool ok;
        if (m_transform) {
            ok = behind ? m_cam.nextFrameLatest(m_scratch, &skipped)
                        : m_cam.nextFrame(m_scratch);
            if (ok) {
                const uint8_t* before = target->data;
                m_transform(m_scratch, *target);
                if (target->data != before) {
                    // 轉換函式讓緩衝重新配置了 —— 外部給的 DMA 位址會失效
                    std::cerr << "[FrameGrabber] 轉換重新配置了緩衝,"
                                 "外部提供的記憶體已失效\n";
                    ok = false;
                }
            }
        } else {
            ok = behind ? m_cam.nextFrameLatest(*target, &skipped)
                        : m_cam.nextFrame(*target);
            m_lastWorkMs = 0.0;   // 沒有轉換,擷取端幾乎不佔 CPU
        }

        const double capMs = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - tc0).count();
        if (ok) {
            m_capSumMs = m_capSumMs.load() + capMs;
            ++m_capCount;
            if (capMs > m_capMaxMs.load()) m_capMaxMs = capMs;
        }

        if (!ok) {
            // 讀取失敗:稍等再試,避免 busy loop 洗版
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (skipped) m_staleSkipped += skipped;

        const double t = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        {
            std::lock_guard<std::mutex> lk(m_mutex);
            // 前一幀還沒被取走就直接覆蓋 —— 刻意的:舊幀沒有保留價值
            if (m_ready >= 0) ++m_overwritten;
            m_ready   = m_write;
            m_tCap    = t;
            m_readyId = ++m_frameId;
            m_write   = pickFree();
        }
        m_cv.notify_one();
    }
}

FrameGrabber::Handle FrameGrabber::acquire(int wait_ms)
{
    std::unique_lock<std::mutex> lk(m_mutex);
    m_cv.wait_for(lk, std::chrono::milliseconds(wait_ms),
                  [&] { return m_ready >= 0 || !m_running; });
    if (m_ready < 0) return Handle();

    m_reading = m_ready;
    m_ready   = -1;
    return Handle(this, m_reading, m_tCap, m_readyId);
}

void FrameGrabber::release()
{
    std::lock_guard<std::mutex> lk(m_mutex);
    m_reading = -1;
}

bool FrameGrabber::pinThread(int cpu)
{
#if defined(__linux__)
    if (cpu < 0) return true;
    if (!m_thread.joinable()) return false;    // 還沒 start()
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(m_thread.native_handle(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

bool FrameGrabber::getLatest(cv::Mat& frame)
{
    long long id = 0;
    return getLatest(frame, id);
}

bool FrameGrabber::getLatest(cv::Mat& frame, long long& outFrameId)
{
    Handle h = acquire(0);
    if (!h.valid()) return false;
    h.mat().copyTo(frame);      // 相容介面:這裡就是那次額外的全幀複製
    outFrameId = h.id();
    return true;
}