#include "v4l2_dmabuf_camera.h"
#include "hls_common.h"          // hls::detail::list_dma_heaps()

#include <linux/videodev2.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>

namespace {

// ---- DMA-BUF heap ABI(自己定義,避免舊 sysroot 沒有 linux/dma-heap.h)----
struct HeapAlloc {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
const unsigned long kHeapIoctlAlloc = _IOWR('H', 0x0, HeapAlloc);

constexpr size_t kPage = 4096;
inline size_t page_align(size_t n) { return (n + kPage - 1) & ~(kPage - 1); }

int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do { r = ::ioctl(fd, req, arg); } while (r < 0 && errno == EINTR);
    return r;
}

double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- 把 CPU cache 裡的資料寫回 DDR(不 invalidate)----
// 核心經 cached 映射寫入影格;IP 走非 coherent 的 HP port,看不到 cache。
// ARM 的 D-cache 是 PIPT,用任何一個虛擬映射做 clean 都作用在同一塊實體記憶體。
void clean_dcache(const void* p, size_t n) {
#if defined(__aarch64__)
    static const size_t line = [] {
        uint64_t ctr;
        __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
        return size_t(4) << ((ctr >> 16) & 0xF);     // DminLine
    }();
    uintptr_t a = reinterpret_cast<uintptr_t>(p) & ~(line - 1);
    const uintptr_t e = reinterpret_cast<uintptr_t>(p) + n;
    for (; a < e; a += line)
        __asm__ volatile("dc cvac, %0" :: "r"(a) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
#else
    (void)p; (void)n;
#endif
}

// ---- 虛擬 → 實體,並確認整塊連續 ----
bool resolve_phys(const void* virt, size_t size, uint64_t& phys, std::string& err) {
    const int pm = ::open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    if (pm < 0) { err = "open /proc/self/pagemap 失敗(需要 root)"; return false; }

    auto pfn_at = [&](size_t i, uint64_t& pfn) -> bool {
        const uint64_t va = reinterpret_cast<uint64_t>(virt) + i * kPage;
        uint64_t e = 0;
        if (::pread(pm, &e, sizeof e, off_t((va / kPage) * sizeof e)) != sizeof e) return false;
        if (!(e & (1ull << 63))) return false;
        pfn = e & ((1ull << 55) - 1);
        return pfn != 0;
    };

    uint64_t first = 0;
    bool ok = pfn_at(0, first);
    const size_t pages = (size + kPage - 1) / kPage;
    for (size_t i = 1; ok && i < pages; ++i) {
        uint64_t pfn = 0;
        ok = pfn_at(i, pfn) && pfn == first + i;
    }
    ::close(pm);
    if (!ok) { err = "緩衝不是實體連續(或無法讀 pagemap)"; return false; }
    phys = first * kPage;
    return true;
}

}  // namespace


// ============================================================================
//  Frame
// ============================================================================
const cv::Mat& DmabufCamera::Frame::mat() const { return cam_->m_slots[slot_].mat; }
uint64_t       DmabufCamera::Frame::phys() const { return cam_->m_slots[slot_].phys; }

void DmabufCamera::Frame::reset() {
    if (cam_) cam_->release(slot_);
    cam_ = nullptr; slot_ = -1;
}


// ============================================================================
//  配置
// ============================================================================
bool DmabufCamera::fail(const std::string& msg) {
    m_err = msg;
    std::cerr << "[DmabufCamera] " << msg << "\n";
    return false;
}

bool DmabufCamera::allocDma(DmaBuf& b, size_t bytes) {
    bytes = page_align(bytes);

    std::vector<std::string> heaps;
    if (!m_heapName.empty())            heaps = {m_heapName};    // 第一塊成功後就固定
    else if (m_cfg.heap != "auto")      heaps = {m_cfg.heap};
    else                                heaps = hls::detail::list_dma_heaps();

    std::string why = "找不到 /dev/dma_heap";
    for (const auto& h : heaps) {
        if (m_cfg.heap == "auto" && m_heapName.empty() && h == "system")
            continue;                                   // system heap 不連續

        const std::string dev = "/dev/dma_heap/" + h;
        const int hfd = ::open(dev.c_str(), O_RDWR | O_CLOEXEC);
        if (hfd < 0) { why = dev + ": " + std::strerror(errno); continue; }

        HeapAlloc req{};
        req.len      = bytes;
        req.fd_flags = O_RDWR | O_CLOEXEC;
        const int r  = xioctl(hfd, kHeapIoctlAlloc, &req);
        const int e  = errno;
        ::close(hfd);
        if (r < 0) { why = h + " 配置失敗: " + std::strerror(e); continue; }

        void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, int(req.fd), 0);
        if (p == MAP_FAILED) { ::close(int(req.fd)); why = h + " mmap 失敗"; continue; }

        // 讀過每一頁讓 pagemap 有 PFN。用讀不用寫,避免弄髒 cache line。
        ::mlock(p, bytes);
        {
            volatile const uint8_t* q = static_cast<const uint8_t*>(p);
            uint8_t sink = 0;
            for (size_t off = 0; off < bytes; off += kPage) sink ^= q[off];
            (void)sink;
        }

        uint64_t phys = 0;
        std::string perr;
        if (!resolve_phys(p, bytes, phys, perr)) {
            ::munmap(p, bytes); ::close(int(req.fd));
            why = h + ": " + perr;
            continue;
        }

        b.fd   = int(req.fd);
        b.virt = static_cast<uint8_t*>(p);
        b.phys = phys;
        b.size = bytes;
        b.mat  = cv::Mat(m_h, m_w, CV_8UC2, b.virt);   // step = w*2,連續
        m_heapName = h;
        return true;
    }
    return fail("DMA 緩衝配置失敗:" + why + "(CMA 夠大嗎?)");
}

void DmabufCamera::freeDma(DmaBuf& b) {
    if (b.virt) { ::munlock(b.virt, b.size); ::munmap(b.virt, b.size); }
    if (b.fd >= 0) ::close(b.fd);
    b = DmaBuf{};
}

bool DmabufCamera::requestBuffers(int count, uint32_t memory, int& got) {
    v4l2_requestbuffers rb{};
    rb.count  = static_cast<uint32_t>(count);
    rb.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = memory;
    if (xioctl(m_fd, VIDIOC_REQBUFS, &rb) < 0) return false;
    got = static_cast<int>(rb.count);
    return true;
}

bool DmabufCamera::setupFormat() {
    v4l2_format fmt{};
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = static_cast<uint32_t>(m_cfg.width);
    fmt.fmt.pix.height      = static_cast<uint32_t>(m_cfg.height);
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_UYVY;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (xioctl(m_fd, VIDIOC_S_FMT, &fmt) < 0)
        return fail(std::string("VIDIOC_S_FMT 失敗: ") + std::strerror(errno));

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_UYVY) {
        const uint32_t f = fmt.fmt.pix.pixelformat;
        const char s[5] = {char(f), char(f >> 8), char(f >> 16), char(f >> 24), 0};
        return fail(std::string("相機不支援 UYVY(驅動給 ") + s + ")");
    }
    m_w = static_cast<int>(fmt.fmt.pix.width);
    m_h = static_cast<int>(fmt.fmt.pix.height);
    m_frameBytes = size_t(m_w) * m_h * 2;

    if (fmt.fmt.pix.bytesperline && fmt.fmt.pix.bytesperline != uint32_t(m_w) * 2)
        return fail("bytesperline=" + std::to_string(fmt.fmt.pix.bytesperline) +
                    " 不等於寬度*2,IP 需要連續的影像");
    m_sizeimage = std::max<size_t>(fmt.fmt.pix.sizeimage, m_frameBytes);

    if (m_w != m_cfg.width || m_h != m_cfg.height)
        std::cerr << "[DmabufCamera] 警告:要求 " << m_cfg.width << "x" << m_cfg.height
                  << ",驅動給 " << m_w << "x" << m_h << "\n";

    // 幀率(驅動不支援就算了)
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator   = 1000;
    parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(m_cfg.fps * 1000 + 0.5);
    xioctl(m_fd, VIDIOC_S_PARM, &parm);
    if (xioctl(m_fd, VIDIOC_G_PARM, &parm) == 0 && parm.parm.capture.timeperframe.numerator)
        m_fps = double(parm.parm.capture.timeperframe.denominator) /
                parm.parm.capture.timeperframe.numerator;
    return true;
}

bool DmabufCamera::open() {
    close();
    m_err.clear();

    m_fd = ::open(m_cfg.device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (m_fd < 0) return fail("開啟 " + m_cfg.device + " 失敗: " + std::strerror(errno));

    v4l2_capability cap{};
    if (xioctl(m_fd, VIDIOC_QUERYCAP, &cap) < 0) return fail("VIDIOC_QUERYCAP 失敗");
    const uint32_t caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps
                                                                    : cap.capabilities;
    if (!(caps & V4L2_CAP_VIDEO_CAPTURE) || !(caps & V4L2_CAP_STREAMING))
        return fail(m_cfg.device + " 不是支援 streaming 的擷取裝置");

    if (!setupFormat()) return false;

    // ---- 優先:DMABUF import(零複製)----
    int got = 0;
    const int want = std::max(4, m_cfg.num_buffers);
    if (requestBuffers(want, V4L2_MEMORY_DMABUF, got) && got > 0) {
        m_slots.resize(size_t(got));
        for (auto& s : m_slots)
            if (!allocDma(s, m_sizeimage)) { close(); return false; }
        m_mode = Mode::Dmabuf;
        return true;
    }
    const int dmabuf_errno = errno;

    if (!m_cfg.allow_mmap_fallback)
        return fail(std::string("驅動不接受 DMABUF: ") + std::strerror(dmabuf_errno));

    // ---- 退路:MMAP + 一次 memcpy 到 DMA 緩衝 ----
    std::cerr << "[DmabufCamera] 驅動不接受 DMABUF(" << std::strerror(dmabuf_errno)
              << "),改用 MMAP + 一次複製\n";
    if (!requestBuffers(4, V4L2_MEMORY_MMAP, got) || got < 2)
        return fail(std::string("VIDIOC_REQBUFS(MMAP) 失敗: ") + std::strerror(errno));

    m_drv.resize(size_t(got));
    for (int i = 0; i < got; ++i) {
        v4l2_buffer b{};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = uint32_t(i);
        if (xioctl(m_fd, VIDIOC_QUERYBUF, &b) < 0) return fail("VIDIOC_QUERYBUF 失敗");
        void* p = ::mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, b.m.offset);
        if (p == MAP_FAILED) return fail("mmap V4L2 緩衝失敗");
        m_drv[size_t(i)] = {p, b.length};
    }
    m_slots.resize(3);
    for (auto& s : m_slots)
        if (!allocDma(s, m_frameBytes)) { close(); return false; }
    m_mode = Mode::Mmap;
    return true;
}

std::string DmabufCamera::describe() const {
    std::ostringstream o;
    o << m_cfg.device << "  " << m_w << "x" << m_h << " @ " << m_fps << " fps  UYVY  "
      << (m_mode == Mode::Dmabuf ? "DMABUF 零複製" :
          m_mode == Mode::Mmap   ? "MMAP + 1 次複製" : "未開啟")
      << "  緩衝 " << m_slots.size() << " 塊 x " << (m_slots.empty() ? 0 : m_slots[0].size >> 10)
      << " KB  heap=" << m_heapName;
    if (!m_slots.empty()) o << "  phys[0]=0x" << std::hex << m_slots[0].phys << std::dec;
    return o.str();
}


// ============================================================================
//  串流
// ============================================================================
bool DmabufCamera::queue(int index) {
    v4l2_buffer b{};
    b.type  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.index = static_cast<uint32_t>(index);
    if (m_mode == Mode::Dmabuf) {
        b.memory = V4L2_MEMORY_DMABUF;
        b.m.fd   = m_slots[size_t(index)].fd;
        b.length = static_cast<uint32_t>(m_slots[size_t(index)].size);
    } else {
        b.memory = V4L2_MEMORY_MMAP;
    }
    return xioctl(m_fd, VIDIOC_QBUF, &b) == 0;
}

bool DmabufCamera::start() {
    if (m_mode == Mode::None) return fail("尚未 open()");
    if (m_running.load()) return true;

    const size_t n = (m_mode == Mode::Dmabuf) ? m_slots.size() : m_drv.size();
    for (size_t i = 0; i < n; ++i)
        if (!queue(int(i))) return fail(std::string("VIDIOC_QBUF 失敗: ") + std::strerror(errno));

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(m_fd, VIDIOC_STREAMON, &type) < 0)
        return fail(std::string("VIDIOC_STREAMON 失敗: ") + std::strerror(errno));
    m_streaming = true;

    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_ready = -1; m_reading = -1;
    }
    m_running = true;
    m_thread = std::thread(&DmabufCamera::captureLoop, this);
    return true;
}

void DmabufCamera::stop() {
    if (m_running.exchange(false)) {
        m_cv.notify_all();
        if (m_thread.joinable()) m_thread.join();
    }
    if (m_streaming) {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(m_fd, VIDIOC_STREAMOFF, &type);     // 驅動會收回所有緩衝
        m_streaming = false;
    }
    std::lock_guard<std::mutex> lk(m_mutex);
    m_ready = -1; m_reading = -1;
}

void DmabufCamera::close() {
    stop();
    if (m_fd >= 0 && m_mode != Mode::None) {
        int got = 0;
        requestBuffers(0, m_mode == Mode::Dmabuf ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP, got);
    }
    for (auto& d : m_drv) if (d.virt) ::munmap(d.virt, d.size);
    m_drv.clear();
    for (auto& s : m_slots) freeDma(s);
    m_slots.clear();
    if (m_fd >= 0) { ::close(m_fd); m_fd = -1; }
    m_mode = Mode::None;
}

bool DmabufCamera::pinThread(int cpu) {
    if (cpu < 0) return true;
    if (!m_thread.joinable()) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(m_thread.native_handle(), sizeof(set), &set) == 0;
}

// 呼叫端須持有 m_mutex
void DmabufCamera::recycle(int slot) {
    if (m_mode == Mode::Dmabuf && m_streaming) queue(slot);   // 還給驅動
    // MMAP 模式:slot 是我們自己的緩衝,不用做任何事
}

int DmabufCamera::pickFreeSlot() const {
    for (int i = 0; i < int(m_slots.size()); ++i)
        if (i != m_reading && i != m_ready) return i;
    return 0;
}

void DmabufCamera::release(int slot) {
    std::lock_guard<std::mutex> lk(m_mutex);
    if (m_reading == slot) m_reading = -1;
    recycle(slot);
}

void DmabufCamera::captureLoop() {
    const uint32_t memory = (m_mode == Mode::Dmabuf) ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;

    while (m_running) {
        pollfd pfd{m_fd, POLLIN, 0};
        const int pr = ::poll(&pfd, 1, 200);
        if (pr <= 0) continue;

        // 把驅動佇列裡「已完成」的全部取出,只留最新的一幀
        int      newest = -1;
        timeval  ts{};
        uint32_t flags = 0;
        for (;;) {
            v4l2_buffer b{};
            b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            b.memory = memory;
            if (xioctl(m_fd, VIDIOC_DQBUF, &b) < 0) {
                if (errno != EAGAIN) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                break;
            }
            // 驅動標記錯誤、或資料不完整(USB 傳輸出錯 -71 常見)→ 直接還回去
            if ((b.flags & V4L2_BUF_FLAG_ERROR) || b.bytesused < m_frameBytes) {
                ++m_bad;
                queue(int(b.index));
                continue;
            }
            if (newest >= 0) { queue(newest); ++m_stale; }
            newest = int(b.index);
            ts     = b.timestamp;
            flags  = b.flags;
        }
        if (newest < 0) continue;

        // V4L2 的 MONOTONIC 時間戳 = CLOCK_MONOTONIC = steady_clock,
        // 延遲量測因此包含 USB 傳輸與驅動時間,比「取出時才打時間」準。
        const double t =
            ((flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC)
                ? double(ts.tv_sec) + double(ts.tv_usec) * 1e-6
                : now_s();

        const auto p0 = std::chrono::steady_clock::now();
        int slot;
        if (m_mode == Mode::Dmabuf) {
            slot = newest;                              // 影格已經在我們的 DMA 緩衝裡
        } else {
            {
                std::lock_guard<std::mutex> lk(m_mutex);
                slot = pickFreeSlot();
            }
            std::memcpy(m_slots[size_t(slot)].virt, m_drv[size_t(newest)].virt, m_frameBytes);
            queue(newest);                              // 驅動緩衝立刻還回去
        }
        clean_dcache(m_slots[size_t(slot)].virt, m_frameBytes);
        m_prepSumMs = m_prepSumMs.load() + std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - p0).count();
        ++m_prepCount;

        {
            std::lock_guard<std::mutex> lk(m_mutex);
            if (m_ready >= 0) { recycle(m_ready); ++m_overwritten; }   // 舊的沒人拿,丟掉
            m_ready   = slot;
            m_tReady  = t;
            m_readyId = ++m_frameId;
        }
        m_cv.notify_one();
    }
}

DmabufCamera::Frame DmabufCamera::acquire(int wait_ms) {
    std::unique_lock<std::mutex> lk(m_mutex);
    m_cv.wait_for(lk, std::chrono::milliseconds(wait_ms),
                  [&] { return m_ready >= 0 || !m_running; });
    if (m_ready < 0) return Frame();
    m_reading = m_ready;
    m_ready   = -1;
    return Frame(this, m_reading, m_tReady, m_readyId);
}