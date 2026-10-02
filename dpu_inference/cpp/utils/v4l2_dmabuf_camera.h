// v4l2_dmabuf_camera.h — 相機影格直接落進 IP 可讀的 DMA 記憶體
//
// ─────────────────────────────────────────────────────────────
//  為什麼不用 cv::VideoCapture
//
//  VideoCapture 的路徑(UVC 相機):
//    USB → [核心 memcpy] → V4L2 的 vmalloc 緩衝
//        → [OpenCV memcpy] → cv::Mat
//        → [stage_input memcpy,若 Mat 不在 pool] → DMA pool → IP
//  1080p UYVY 每次 4 MB,光複製就 2~3 份。
//
//  這個類別用 V4L2 的 DMABUF import:
//    自己從 /dev/dma_heap 配「實體連續」的緩衝 → 把 fd 交給 uvcvideo
//    USB → [核心 memcpy] → 直接寫進我們的 DMA 緩衝 → clean cache → IP
//  核心那一次是 UVC 組幀必要的,之後完全零複製。
//
//  驅動不支援 DMABUF 時自動退回 MMAP:核心寫進自己的緩衝,
//  我們 memcpy 一次到 DMA 緩衝(仍比 VideoCapture 少一到兩次)。
//
//  Cache:核心用 CPU 寫入(經 cached 映射),IP 走非 coherent 的 HP port,
//  所以交給 IP 前要把整塊 clean 到 DDR。dma-buf 的 SYNC ioctl 在沒有
//  「mapped attachment」時什麼都不做,因此這裡直接用 dc cvac。
//  clean 在擷取執行緒做,處理端不付這個成本。
//
//  「只取最新一幀」的語意與 FrameGrabber 相同:處理端來不及時,
//  舊幀直接還給驅動,不會積壓。
//
//  需要 root(/proc/self/pagemap 取實體位址)。
// ─────────────────────────────────────────────────────────────
#pragma once

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class DmabufCamera {
public:
    struct Config {
        std::string device      = "/dev/video0";
        int         width       = 1920;
        int         height      = 1080;
        double      fps         = 60.0;
        int         num_buffers = 6;       // DMABUF 模式的 V4L2 緩衝數(至少 4)
        std::string heap        = "auto";  // /dev/dma_heap 底下的名稱;auto = 自動挑
        bool        allow_mmap_fallback = true;
    };

    // 取得的一幀。生命週期內緩衝不會被覆寫;請儘早讓它離開作用域。
    class Frame {
    public:
        Frame() = default;
        ~Frame() { reset(); }
        Frame(Frame&& o) noexcept { swap(o); }
        Frame& operator=(Frame&& o) noexcept { reset(); swap(o); return *this; }
        Frame(const Frame&)            = delete;
        Frame& operator=(const Frame&) = delete;

        bool           valid()     const { return cam_ != nullptr; }
        const cv::Mat& mat()       const;          // CV_8UC2,UYVY,連續
        uint64_t       phys()      const;          // mat().data 的實體位址
        double         timestamp() const { return t_; }   // steady_clock 秒
        long long      id()        const { return id_; }
        void           reset();                    // 提早歸還

    private:
        friend class DmabufCamera;
        Frame(DmabufCamera* c, int slot, double t, long long id)
            : cam_(c), slot_(slot), t_(t), id_(id) {}
        void swap(Frame& o) {
            std::swap(cam_, o.cam_); std::swap(slot_, o.slot_);
            std::swap(t_, o.t_);     std::swap(id_, o.id_);
        }
        DmabufCamera* cam_  = nullptr;
        int           slot_ = -1;
        double        t_    = 0;
        long long     id_   = 0;
    };

    explicit DmabufCamera(Config cfg) : m_cfg(std::move(cfg)) {}
    ~DmabufCamera() { close(); }
    DmabufCamera(const DmabufCamera&)            = delete;
    DmabufCamera& operator=(const DmabufCamera&) = delete;

    bool open();           // 設格式、配置緩衝;失敗時 error() 說明原因
    bool start();          // 開始串流 + 擷取執行緒
    void stop();
    void close();

    // 等到有新的一幀。逾時或已停止時回傳無效的 Frame。
    Frame acquire(int wait_ms = 200);

    bool pinThread(int cpu);   // start() 之後呼叫

    // --- open() 後有效 ---
    int    width()  const { return m_w; }
    int    height() const { return m_h; }
    double fps()    const { return m_fps; }
    bool   zeroCopy() const { return m_mode == Mode::Dmabuf; }
    const std::string& heapName() const { return m_heapName; }
    const std::string& error()    const { return m_err; }
    std::string describe() const;

    // --- 統計 ---
    long long frames()      const { return m_frameId.load(); }
    long long overwritten() const { return m_overwritten.load(); }  // 處理端來不及而被丟掉
    long long stale()       const { return m_stale.load(); }        // 驅動佇列裡積壓被跳過
    long long badFrames()   const { return m_bad.load(); }          // 驅動標記錯誤 / 不完整(USB -71 等)
    double    avgPrepMs()   const {                                 // 擷取端每幀的 copy + clean
        const long long n = m_prepCount.load();
        return n ? m_prepSumMs.load() / double(n) : 0.0;
    }

private:
    enum class Mode { None, Dmabuf, Mmap };

    struct DmaBuf {                 // 我們配的 DMA 緩衝(IP 讀這個)
        int      fd   = -1;
        uint8_t* virt = nullptr;
        uint64_t phys = 0;
        size_t   size = 0;
        cv::Mat  mat;
    };
    struct DrvBuf {                 // MMAP 模式:驅動自己的緩衝
        void*  virt = nullptr;
        size_t size = 0;
    };

    bool fail(const std::string& msg);
    bool setupFormat();
    bool allocDma(DmaBuf& b, size_t bytes);
    void freeDma(DmaBuf& b);
    bool requestBuffers(int count, uint32_t memory, int& got);
    bool queue(int index);          // DMABUF: index == slot;MMAP: 驅動緩衝索引
    void captureLoop();
    void recycle(int slot);         // 呼叫端須持有 m_mutex
    int  pickFreeSlot() const;      // MMAP 模式,呼叫端須持有 m_mutex
    void release(int slot);

    Config      m_cfg;
    int         m_fd = -1;
    Mode        m_mode = Mode::None;
    int         m_w = 0, m_h = 0;
    double      m_fps = 0;
    size_t      m_sizeimage = 0;
    size_t      m_frameBytes = 0;   // w * h * 2
    std::string m_heapName, m_err;

    std::vector<DmaBuf> m_slots;
    std::vector<DrvBuf> m_drv;

    std::thread       m_thread;
    std::atomic<bool> m_running{false};
    bool              m_streaming = false;

    mutable std::mutex      m_mutex;
    std::condition_variable m_cv;
    int       m_ready = -1, m_reading = -1;
    double    m_tReady = 0;
    long long m_readyId = 0;

    std::atomic<long long> m_frameId{0}, m_overwritten{0}, m_stale{0}, m_bad{0};
    std::atomic<double>    m_prepSumMs{0.0};
    std::atomic<long long> m_prepCount{0};
};