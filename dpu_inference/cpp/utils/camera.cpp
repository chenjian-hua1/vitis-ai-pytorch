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
 
static std::string fourcc_to_str(double v)
{
    int f = static_cast<int>(v);
    char s[5] = { char(f & 0xFF), char((f >> 8) & 0xFF),
                  char((f >> 16) & 0xFF), char((f >> 24) & 0xFF), '\0' };
    return std::string(s);
}

bool Camera::open()
{
    m_cap.open(m_cfg.index, cv::CAP_V4L2);

    if (!m_cap.isOpened()) {
        std::cerr << "[Camera] 無法開啟攝影機 (index="
                  << m_cfg.index << ")" << std::endl;
        return false;
    }

    // FOURCC 要「先」設定：V4L2 換格式時會重置可用的解析度/FPS 清單，
    // 放在 width/height 之後設會把前面設好的值蓋掉。
    if (m_cfg.fourcc.size() == 4) {
        m_cap.set(cv::CAP_PROP_FOURCC,
                  cv::VideoWriter::fourcc(m_cfg.fourcc[0], m_cfg.fourcc[1],
                                          m_cfg.fourcc[2], m_cfg.fourcc[3]));
    }

    m_cap.set(cv::CAP_PROP_FRAME_WIDTH,  m_cfg.width);
    m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, m_cfg.height);
    m_cap.set(cv::CAP_PROP_FPS,          m_cfg.fps);

    // 讀回驅動實際套用的值
    m_actualWidth  = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    m_actualHeight = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_actualFps    = m_cap.get(cv::CAP_PROP_FPS);
    m_actualFourcc = fourcc_to_str(m_cap.get(cv::CAP_PROP_FOURCC));

    std::cout << "[Camera] 開啟成功  "
              << m_actualWidth << "x" << m_actualHeight
              << " @ " << m_actualFps << " fps"
              << "  [" << m_actualFourcc << "]" << std::endl;

    if (m_cfg.fourcc.size() == 4 && m_actualFourcc != m_cfg.fourcc) {
        std::cerr << "[Camera] 警告: 要求 " << m_cfg.fourcc
                  << " 但驅動實際給 " << m_actualFourcc << std::endl;
    }
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