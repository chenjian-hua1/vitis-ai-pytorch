#include <camera.h>

// ============================================================================
//  Camera Process
// ============================================================================
// ── 建構子 ──────────────────────────────────────────────────────────────────
 
Camera::Camera(int index)
{
    m_cfg.index = index;
}
 
Camera::Camera(const Config& cfg)
    : m_cfg(cfg)
{}
 
Camera::~Camera()
{
    close();
}
 
// ── 公開介面 ─────────────────────────────────────────────────────────────────
 
bool Camera::open()
{
    m_cap.open(m_cfg.index, cv::CAP_V4L2);
 
    if (!m_cap.isOpened()) {
        std::cerr << "[Camera] 無法開啟攝影機 (index="
                  << m_cfg.index << ")" << std::endl;
        return false;
    }
 
    // 設定鏡頭參數
    m_cap.set(cv::CAP_PROP_FRAME_WIDTH,  m_cfg.width);
    m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, m_cfg.height);
    m_cap.set(cv::CAP_PROP_FPS,          m_cfg.fps);

    m_cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
 
    // 讀回驅動實際套用的值
    m_actualWidth  = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    m_actualHeight = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_actualFps    = m_cap.get(cv::CAP_PROP_FPS);
 
    std::cout << "[Camera] 開啟成功  "
              << m_actualWidth << "x" << m_actualHeight
              << " @ " << m_actualFps << " fps" << std::endl;
    return true;
}
 
bool Camera::nextFrame(cv::Mat& frame)
{
    if (!m_cap.isOpened()) {
        std::cerr << "[Camera] 攝影機尚未開啟，請先呼叫 open()" << std::endl;
        return false;
    }
 
    if (!m_cap.read(frame) || frame.empty()) {
        std::cerr << "[Camera] 無法擷取影像" << std::endl;
        return false;
    }
 
    return true;
}
 
void Camera::close()
{
    if (m_cap.isOpened()) {
        m_cap.release();
        std::cout << "[Camera] 已關閉" << std::endl;
    }
}
 
bool Camera::isOpened() const
{
    return m_cap.isOpened();
}