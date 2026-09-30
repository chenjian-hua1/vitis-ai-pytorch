// hls_common.cpp — 共用層的實作
//
// 所有 Linux 系統標頭(mmap、ioctl、UIO、dma_heap)只出現在這裡,
// 不會漏到應用程式那一側。

#include "hls_common.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <dirent.h>
#include <fcntl.h>
#include <linux/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

// CV_AUTOSTEP 定義在舊版 C API 的 opencv2/core/core_c.h,OpenCV 4 不再引入。
// 自己補一個,值取自 OpenCV 原始碼,避免把整個 C API 拉進來。
#ifndef CV_AUTOSTEP
#define CV_AUTOSTEP 0x7fffffff
#endif

namespace hls {

// 本檔的對外函式與匿名 namespace 都會用到內部型別
using namespace detail;

namespace {

// ============================================================
//  ARM64 使用者空間 cache 維護
//
//  dma_heap 的 begin/end_cpu_access 是走訪「已 attach 的裝置」來做同步。
//  我們從 userspace 直接操作 IP,沒有 driver attach 到這個 dmabuf,
//  DMA_BUF_IOCTL_SYNC 實際上什麼都沒做。症狀是小圖幾乎全錯、大圖幾乎全對。
//
//  DC CIVAC 在 EL0 可用(Linux 會設 SCTLR_EL1.UCI);若核心關掉會收到 SIGILL。
// ============================================================
namespace arm_cache {

#if defined(__aarch64__)

size_t line_size() {
    static const size_t sz = [] {
        uint64_t ctr = 0;
        __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
        return static_cast<size_t>(4u << ((ctr >> 16) & 0xF));   // DminLine
    }();
    return sz;
}

// clean & invalidate:對兩個方向都安全的超集合,
// 不會有「invalidate 掉尚未寫回的髒資料」的風險。
void flush(const void* addr, size_t size) {
    if (!addr || !size) return;
    const size_t line = line_size();
    uintptr_t p = reinterpret_cast<uintptr_t>(addr) & ~(line - 1);
    const uintptr_t end = reinterpret_cast<uintptr_t>(addr) + size;
    for (; p < end; p += line)
        __asm__ volatile("dc civac, %0" :: "r"(p) : "memory");
    __asm__ volatile("dsb ish" ::: "memory");
    __asm__ volatile("isb" ::: "memory");
}

constexpr bool supported() { return true; }

#else

void flush(const void*, size_t) {}
constexpr bool supported() { return false; }

#endif

}  // namespace arm_cache

// DMA-BUF Heaps 的 ABI,直接寫在這裡以免相依於特定版本的 kernel header
namespace dma_heap_abi {

struct allocation_data {
    __u64 len;
    __u32 fd;
    __u32 fd_flags;
    __u64 heap_flags;
};
struct buf_sync { __u64 flags; };

constexpr unsigned long IOCTL_ALLOC = _IOWR('H', 0x0, struct allocation_data);
constexpr unsigned long IOCTL_SYNC  = _IOW('b', 0x0, struct buf_sync);

constexpr __u64 SYNC_READ  = 1ull << 0;
constexpr __u64 SYNC_WRITE = 2ull << 0;
constexpr __u64 SYNC_RW    = SYNC_READ | SYNC_WRITE;
constexpr __u64 SYNC_START = 0ull << 2;   // 開始 CPU 存取 -> invalidate
constexpr __u64 SYNC_END   = 1ull << 2;   // 結束 CPU 存取 -> flush

}  // namespace dma_heap_abi

std::string to_hex(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "%llx", static_cast<unsigned long long>(v));
    return b;
}

uint64_t read_sysfs_val(const std::string& p, bool hex) {
    FILE* f = ::fopen(p.c_str(), "r");
    if (!f) return 0;
    unsigned long long v = 0;
    if (::fscanf(f, hex ? "%llx" : "%llu", &v) != 1) v = 0;
    ::fclose(f);
    return v;
}

// 讀 /proc/self/pagemap 取得實體位址,並確認整塊連續
uint64_t resolve_phys(void* virt, size_t size) {
    const size_t page = 4096;
    int pm = ::open("/proc/self/pagemap", O_RDONLY);
    if (pm < 0)
        throw std::runtime_error("open /proc/self/pagemap 失敗(需要 root)");

    auto pfn_at = [&](size_t idx) -> uint64_t {
        const uint64_t vaddr = reinterpret_cast<uint64_t>(virt) + idx * page;
        uint64_t entry = 0;
        if (::pread(pm, &entry, sizeof entry,
                    static_cast<off_t>((vaddr / page) * sizeof entry)) != sizeof entry)
            throw std::runtime_error("讀 pagemap 失敗");
        if (!(entry & (1ull << 63)))
            throw std::runtime_error("頁面不在記憶體中");
        const uint64_t pfn = entry & ((1ull << 55) - 1);
        if (pfn == 0)
            throw std::runtime_error("PFN 為 0 —— 權限不足,無法取得實體位址");
        return pfn;
    };

    uint64_t first = 0;
    try {
        first = pfn_at(0);
        const size_t pages = (size + page - 1) / page;
        for (size_t i = 1; i < pages; ++i) {
            if (pfn_at(i) != first + i)
                throw std::runtime_error(
                    "dma_heap 配到的記憶體不是實體連續的"
                    "(換用 linux,cma heap,或改用 u-dma-buf)");
        }
    } catch (...) { ::close(pm); throw; }

    ::close(pm);
    return first * page;
}

}  // namespace


namespace detail {

// ============================================================
//  list_dma_heaps
// ============================================================
std::vector<std::string> list_dma_heaps() {
    std::vector<std::string> cma, carveout, other, system;
    DIR* d = ::opendir("/dev/dma_heap");
    if (!d) return {};
    while (dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        if      (n.find("cma") != std::string::npos)      cma.push_back(n);
        else if (n.find("reserved") != std::string::npos ||
                 n.find("carveout") != std::string::npos) carveout.push_back(n);
        else if (n == "system")                           system.push_back(n);
        else                                              other.push_back(n);
    }
    ::closedir(d);

    std::vector<std::string> out;
    for (auto* v : {&cma, &carveout, &other, &system})
        out.insert(out.end(), v->begin(), v->end());
    return out;
}


// ============================================================
//  DmaPool
// ============================================================

// 模式一:ikwzm u-dma-buf(cached 映射,效能好,需手動 sync)
DmaPool::DmaPool(const std::string& name) : name_(name) {
    const std::string sys = "/sys/class/u-dma-buf/" + name + "/";
    phys_ = read_sysfs_val(sys + "phys_addr", true);
    size_ = static_cast<size_t>(read_sysfs_val(sys + "size", false));
    if (!phys_ || !size_)
        throw std::runtime_error("讀不到 " + name + " 的 phys_addr/size"
                                 "(u-dma-buf 模組載入了嗎?)");

    fd_ = ::open(("/dev/" + name).c_str(), O_RDWR);
    if (fd_ < 0)
        throw std::runtime_error("open /dev/" + name + " 失敗: " + strerror(errno));

    void* p = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) { ::close(fd_); throw std::runtime_error("mmap " + name + " 失敗"); }
    virt_    = static_cast<uint8_t*>(p);
    cached_  = true;
    backend_ = Backend::UdmaBuf;

    free_[0] = size_;
}

// 模式二:/dev/mem 直接映射一塊 no-map 的 reserved memory。
// O_SYNC 讓映射是 uncached,sync 變成 no-op;代價是 CPU 端運算慢一個數量級。
DmaPool::DmaPool(uint64_t phys, size_t size) : name_("devmem") {
    if (!phys || !size) throw std::runtime_error("reserved memory 的位址或大小為 0");

    fd_ = ::open("/dev/mem", O_RDWR | O_SYNC);
    if (fd_ < 0)
        throw std::runtime_error(std::string("open /dev/mem 失敗: ") + strerror(errno));

    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd_, static_cast<off_t>(phys));
    if (p == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error("mmap /dev/mem @0x" + to_hex(phys) +
                                 " 失敗(這塊有在裝置樹保留嗎?)");
    }
    virt_    = static_cast<uint8_t*>(p);
    phys_    = phys;
    size_    = size;
    cached_  = false;
    backend_ = Backend::DevMem;

    free_[0] = size_;
}

// 模式三:DMA-BUF Heaps。
// dma_heap 刻意不把實體位址交給 userspace,所以改用 /proc/self/pagemap
// 做虛擬->實體轉換,並逐頁確認確實連續。需要 root。
DmaPool::DmaPool(const DmaHeapTag& tag, size_t size) : name_("dma_heap:" + tag.heap) {
    const std::string dev = "/dev/dma_heap/" + tag.heap;
    int heap_fd = ::open(dev.c_str(), O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        std::string avail;
        for (const auto& h : list_dma_heaps()) avail += (avail.empty() ? "" : ", ") + h;
        throw std::runtime_error("open " + dev + " 失敗: " + strerror(errno) +
                                 "(可用的 heap: " + (avail.empty() ? "無" : avail) + ")");
    }

    dma_heap_abi::allocation_data req{};
    req.len        = size;
    req.fd_flags   = O_RDWR | O_CLOEXEC;
    req.heap_flags = 0;

    if (::ioctl(heap_fd, dma_heap_abi::IOCTL_ALLOC, &req) < 0) {
        const int e = errno;
        ::close(heap_fd);
        throw std::runtime_error("dma_heap 配置 " + std::to_string(size) +
                                 " bytes 失敗: " + strerror(e) +
                                 "(CMA 夠大嗎?試試 cma=256M)");
    }
    ::close(heap_fd);
    fd_ = static_cast<int>(req.fd);

    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) { ::close(fd_); throw std::runtime_error("mmap dmabuf 失敗"); }
    virt_ = static_cast<uint8_t*>(p);
    size_ = size;

    // 鎖住並碰過每一頁,pagemap 才會有有效的 PFN。
    // 用「讀」而非「寫」—— 寫會把整個 pool 的 cache line 弄髒,
    // 之後不定時寫回會蓋掉 IP 的輸出。
    ::mlock(virt_, size_);
    {
        volatile const uint8_t* probe = virt_;
        uint8_t sink = 0;
        for (size_t off = 0; off < size_; off += 4096) sink ^= probe[off];
        (void)sink;
    }

    try {
        phys_ = resolve_phys(virt_, size_);
    } catch (...) {
        ::munmap(virt_, size_); ::close(fd_);
        throw;
    }

    backend_ = Backend::DmaHeap;
    cached_  = true;
    // dma_heap 的 ioctl 在沒有 attach 裝置時是 no-op,一律自己來
    manual_cache_ = arm_cache::supported();
    free_[0] = size_;
}

// 模式四:外部提供(XRT 等)。只負責配置與位址換算,不管生命週期。
DmaPool::DmaPool(ExternalMem mem) : name_(mem.label), external_(std::move(mem)) {
    if (!external_.virt || !external_.phys || !external_.size)
        throw std::runtime_error("ExternalMem 的欄位不完整");

    virt_    = static_cast<uint8_t*>(external_.virt);
    phys_    = external_.phys;
    size_    = external_.size;
    cached_  = static_cast<bool>(external_.sync);
    backend_ = Backend::External;
    fd_      = -1;

    free_[0] = size_;
}

DmaPool::~DmaPool() {
    if (backend_ != Backend::External && virt_) ::munmap(virt_, size_);
    if (fd_ >= 0) ::close(fd_);
}

void* DmaPool::alloc(size_t bytes, size_t align) {
    std::lock_guard<std::mutex> lk(alloc_mtx_);
    for (auto it = free_.begin(); it != free_.end(); ++it) {
        const size_t off = it->first, len = it->second;
        const size_t aligned = (off + align - 1) & ~(align - 1);
        const size_t pad = aligned - off;
        if (len < pad + bytes) continue;

        free_.erase(it);
        if (pad) free_[off] = pad;
        const size_t tail_off = aligned + bytes;
        const size_t tail_len = len - pad - bytes;
        if (tail_len) free_[tail_off] = tail_len;

        used_[aligned] = bytes;
        if (aligned + bytes > used_end_) used_end_ = aligned + bytes;
        return virt_ + aligned;
    }
    throw std::runtime_error("DmaPool 空間不足");
}

void DmaPool::free(void* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(alloc_mtx_);
    const size_t off = static_cast<uint8_t*>(p) - virt_;
    auto it = used_.find(off);
    if (it == used_.end()) return;
    free_[off] = it->second;
    used_.erase(it);
    coalesce();
}

void DmaPool::coalesce() {
    for (auto it = free_.begin(); it != free_.end(); ) {
        auto nx = std::next(it);
        if (nx != free_.end() && it->first + it->second == nx->first) {
            it->second += nx->second;
            free_.erase(nx);
        } else ++it;
    }
}

uint64_t DmaPool::phys_of(const void* p) const {
    auto* u = static_cast<const uint8_t*>(p);
    if (u < virt_ || u >= virt_ + size_)
        throw std::runtime_error("指標不在 DMA pool 內 —— 這個 Mat 不是從 pool 配出來的");
    return phys_ + static_cast<uint64_t>(u - virt_);
}

bool DmaPool::contains(const void* p) const {
    auto* u = static_cast<const uint8_t*>(p);
    return u >= virt_ && u < virt_ + size_;
}

bool DmaPool::contains(const void* p, size_t n) const {
    if (!n) return contains(p);
    auto* u = static_cast<const uint8_t*>(p);
    return contains(u) && contains(u + n - 1);
}

size_t DmaPool::active_bytes() const {
    std::lock_guard<std::mutex> lk(alloc_mtx_);
    return used_end_;
}

void DmaPool::set_manual_cache(bool on) { manual_cache_ = on && arm_cache::supported(); }

void DmaPool::sync_for_device() { sync_range_for_device(virt_, active_bytes()); }
void DmaPool::sync_for_cpu()    { sync_range_for_cpu(virt_, active_bytes()); }

// 範圍式同步對效能影響很大:輸入 6 MB、輸出 0.7 MB 時,
// 全範圍做兩次 = 14 MB;只 flush 輸入、只 invalidate 輸出 = 6.9 MB。
void DmaPool::sync_range_for_device(const void* p, size_t n) {
    if (!n) return;
    if (manual_cache_) arm_cache::flush(p, n);
    switch (backend_) {
        case Backend::UdmaBuf:  poke_range(p, n, "sync_for_device"); break;
        case Backend::DmaHeap:  dmabuf_sync(dma_heap_abi::SYNC_END | dma_heap_abi::SYNC_RW); break;
        case Backend::External: if (external_.sync) external_.sync(true); break;
        case Backend::DevMem:   break;   // uncached,不需要
    }
}

void DmaPool::sync_range_for_cpu(const void* p, size_t n) {
    if (!n) return;
    if (manual_cache_) arm_cache::flush(p, n);
    switch (backend_) {
        case Backend::UdmaBuf:  poke_range(p, n, "sync_for_cpu"); break;
        case Backend::DmaHeap:  dmabuf_sync(dma_heap_abi::SYNC_START | dma_heap_abi::SYNC_RW); break;
        case Backend::External: if (external_.sync) external_.sync(false); break;
        case Backend::DevMem:   break;
    }
}

// u-dma-buf 的範圍同步要寫三次 sysfs(offset、size、觸發),必須成組,
// 否則兩顆 IP 同時同步時會互相蓋掉範圍。
void DmaPool::poke_range(const void* p, size_t n, const char* attr) {
    std::lock_guard<std::mutex> lk(sync_mtx_);
    const size_t off = static_cast<const uint8_t*>(p) - virt_;
    poke("sync_offset", static_cast<long long>(off));
    poke("sync_size",   static_cast<long long>(n));
    poke(attr, 1);
}

void DmaPool::poke(const char* attr, long long v) {
    const std::string p = "/sys/class/u-dma-buf/" + name_ + "/" + attr;
    FILE* f = ::fopen(p.c_str(), "w");
    if (!f) return;
    ::fprintf(f, "%lld", v);
    ::fclose(f);
}

void DmaPool::dmabuf_sync(uint64_t flags) {
    dma_heap_abi::buf_sync s{};
    s.flags = flags;
    ::ioctl(fd_, dma_heap_abi::IOCTL_SYNC, &s);
}


// ============================================================
//  DmaMat
// ============================================================
DmaMat::DmaMat(DmaPool& pool, int rows, int cols, int type, size_t row_align)
    : pool_(&pool) {
    const size_t elem = CV_ELEM_SIZE(type);
    step_       = align_up(static_cast<size_t>(cols) * elem, row_align);
    data_bytes_ = step_ * static_cast<size_t>(rows);
    bytes_      = align_up(data_bytes_, kCacheLine);   // 尾端不與下一塊共用 cache line
    ptr_        = pool.alloc(bytes_, kPageAlign);      // 起點對 page,AXI burst 開得滿
    mat_        = cv::Mat(rows, cols, type, ptr_, step_);
}

DmaMat::~DmaMat() { if (pool_ && ptr_) pool_->free(ptr_); }

DmaMat::DmaMat(DmaMat&& o) noexcept { swap(o); }
DmaMat& DmaMat::operator=(DmaMat&& o) noexcept { swap(o); return *this; }

void DmaMat::swap(DmaMat& o) noexcept {
    std::swap(pool_, o.pool_); std::swap(ptr_, o.ptr_);
    std::swap(step_, o.step_); std::swap(bytes_, o.bytes_);
    std::swap(data_bytes_, o.data_bytes_);
    std::swap(mat_, o.mat_);
}

uint64_t DmaMat::phys() const { return pool_->phys_of(ptr_); }

bool DmaMat::is_packed() const {
    return step_ == static_cast<size_t>(mat_.cols) * mat_.elemSize();
}

void DmaMat::import_from(const cv::Mat& src) {
    CV_Assert(src.rows == mat_.rows && src.cols == mat_.cols && src.type() == mat_.type());
    src.copyTo(mat_);      // Mat 已綁定外部記憶體,copyTo 不會重新配置
}


// ============================================================
//  DmaAllocator
// ============================================================
cv::UMatData* DmaAllocator::allocate(int dims, const int* sizes, int type,
                                     void* data0, size_t* step,
                                     cv::AccessFlag, cv::UMatUsageFlags) const {
    size_t total = CV_ELEM_SIZE(type);
    for (int i = dims - 1; i >= 0; --i) {
        if (step) {
            if (data0 && step[i] != static_cast<size_t>(CV_AUTOSTEP))
                total = step[i];
            else step[i] = total;
        }
        total *= sizes[i];
    }
    const size_t padded = DmaMat::align_up(total, DmaMat::kCacheLine);
    uint8_t* data = data0 ? static_cast<uint8_t*>(data0)
                          : static_cast<uint8_t*>(pool_.alloc(padded, DmaMat::kPageAlign));

    auto* u = new cv::UMatData(this);
    u->data = u->origdata = data;
    u->size = total;
    if (data0) u->flags |= cv::UMatData::USER_ALLOCATED;
    return u;
}

bool DmaAllocator::allocate(cv::UMatData* u, cv::AccessFlag, cv::UMatUsageFlags) const {
    if (!u) return false;
    u->urefcount++;
    return true;
}

void DmaAllocator::deallocate(cv::UMatData* u) const {
    if (!u) return;
    CV_Assert(u->urefcount >= 0 && u->refcount >= 0);
    if (u->refcount == 0) {
        if (!(u->flags & cv::UMatData::USER_ALLOCATED)) {
            pool_.free(u->origdata);
            u->origdata = nullptr;
        }
        delete u;
    }
}

}  // namespace detail


// ============================================================
//  共用 pool
// ============================================================
namespace {

class PoolHolder {
public:
    static PoolHolder& get() {
        static PoolHolder h;
        return h;
    }

    void set_name(const std::string& name) {
        std::lock_guard<std::mutex> lk(mtx_);
        mode_ = Mode::UdmaBuf;
        name_ = name;
        reset_locked();
    }
    void set_reserved(uint64_t phys, size_t size) {
        std::lock_guard<std::mutex> lk(mtx_);
        mode_ = Mode::Reserved;
        res_phys_ = phys; res_size_ = size;
        reset_locked();
    }
    void set_factory(std::function<std::shared_ptr<DmaPool>()> f) {
        std::lock_guard<std::mutex> lk(mtx_);
        mode_ = Mode::Factory;
        factory_ = std::move(f);
        reset_locked();
    }
    void set_dma_heap(const std::string& heap, size_t size) {
        std::lock_guard<std::mutex> lk(mtx_);
        mode_ = Mode::DmaHeap;
        heap_ = heap; res_size_ = size;
        reset_locked();
    }

    // 用 shared_ptr 讓 pool 活得比所有使用者久:各 IP 的 BufferCache 也是
    // static,解構順序不固定。每個 cache 各持一份 shared_ptr,順序就不再重要。
    std::shared_ptr<DmaPool> try_open() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (tried_) return pool_;
        tried_ = true;
        try {
            switch (mode_) {
                case Mode::Factory:
                    pool_ = factory_ ? factory_() : nullptr; break;
                case Mode::Reserved:
                    pool_ = std::make_shared<DmaPool>(res_phys_, res_size_); break;
                case Mode::DmaHeap:
                    pool_ = open_dma_heap(); break;
                case Mode::UdmaBuf:
                default:
                    pool_ = std::make_shared<DmaPool>(name_); break;
            }
            if (!pool_) error_ = "pool factory 回傳空指標";
        } catch (const std::exception& e) {
            error_ = e.what();
            pool_.reset();
        }
        return pool_;
    }

    std::string error() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return error_;
    }

private:
    PoolHolder() = default;

    void reset_locked() {
        pool_.reset();
        tried_ = false;
        error_.clear();
    }

    // "auto" 時逐一嘗試,直到某個 heap 真的配得出連續記憶體
    std::shared_ptr<DmaPool> open_dma_heap() {
        if (heap_ != "auto")
            return std::make_shared<DmaPool>(DmaHeapTag{heap_}, res_size_);

        const auto heaps = list_dma_heaps();
        if (heaps.empty())
            throw std::runtime_error("/dev/dma_heap 底下沒有任何 heap");

        std::string tried;
        for (const auto& h : heaps) {
            try {
                auto p = std::make_shared<DmaPool>(DmaHeapTag{h}, res_size_);
                heap_ = h;
                return p;
            } catch (const std::exception& e) {
                tried += (tried.empty() ? "" : "; ") + h + ": " + e.what();
            }
        }
        throw std::runtime_error("所有 dma_heap 都失敗 —— " + tried);
    }

    enum class Mode { UdmaBuf, Reserved, DmaHeap, Factory };
    Mode        mode_ = Mode::UdmaBuf;
    std::string name_ = "udmabuf0";
    std::string heap_ = "linux,cma";
    std::function<std::shared_ptr<DmaPool>()> factory_;
    uint64_t    res_phys_ = 0;
    size_t      res_size_ = 0;
    std::string error_;
    bool        tried_ = false;
    std::shared_ptr<DmaPool> pool_;
    mutable std::mutex mtx_;
};

std::vector<void (*)()>& cleanup_hooks() {
    static std::vector<void (*)()> v;
    return v;
}

}  // namespace

std::shared_ptr<DmaPool> detail::pool_ptr() { return PoolHolder::get().try_open(); }

void detail::set_pool_factory(std::function<std::shared_ptr<DmaPool>()> f) {
    PoolHolder::get().set_factory(std::move(f));
}

// ---- 對外(hls_common.h)----

void set_pool(const std::string& name) { PoolHolder::get().set_name(name); }

void use_dma_heap(const std::string& heap, size_t size) {
    PoolHolder::get().set_dma_heap(heap, size);
}

void use_reserved_memory(uint64_t phys, size_t size) {
    PoolHolder::get().set_reserved(phys, size);
}

void use_external_memory(std::function<ExternalMem()> provider) {
    set_pool_factory([provider = std::move(provider)]() -> std::shared_ptr<DmaPool> {
        if (!provider) return nullptr;
        return std::make_shared<DmaPool>(provider());
    });
}

bool pool_available() { return static_cast<bool>(pool_ptr()); }

std::string pool_error() { return PoolHolder::get().error(); }

std::string pool_info() {
    auto p = pool_ptr();
    if (!p) return "不可用: " + pool_error();
    char b[200];
    std::snprintf(b, sizeof b, "%s  phys=0x%llx  size=%.1f MB  (%s%s)",
                  p->name().c_str(), static_cast<unsigned long long>(p->phys()),
                  p->size() / 1048576.0,
                  p->is_cached() ? "cached" : "uncached,免 sync 但較慢",
                  p->manual_cache() ? ",自行做 cache 維護" :
                  (p->is_cached() ? ",由驅動 sync" : ""));
    return b;
}

detail::CleanupRegistrar::CleanupRegistrar(void (*fn)()) { cleanup_hooks().push_back(fn); }

void clear_buffers() {
    for (auto fn : cleanup_hooks()) fn();
}


namespace detail {

// ============================================================
//  BufferCache
// ============================================================
DmaMat* BufferCache::get(const std::shared_ptr<DmaPool>& pool, int rows, int cols,
                         int type, int slot) {
    if (!pool) return nullptr;
    if (keepalive_ != pool) {      // 換了 pool,舊 buffer 一律作廢
        map_.clear();
        keepalive_ = pool;
    }

    const Key k{rows, cols, type, slot};
    auto it = map_.find(k);
    if (it != map_.end()) return it->second.get();

    try {
        auto m = std::make_unique<DmaMat>(*pool, rows, cols, type);
        if (!m->is_packed()) return nullptr;
        DmaMat* raw = m.get();
        map_[k] = std::move(m);
        return raw;
    } catch (const std::exception&) {
        return nullptr;
    }
}

void BufferCache::clear() { map_.clear(); keepalive_.reset(); }


// ============================================================
//  AxiLiteIp
// ============================================================
namespace {

uint32_t read_hex_file(const std::string& path) {
    FILE* f = ::fopen(path.c_str(), "r");
    if (!f) return 0;
    unsigned v = 0;
    if (::fscanf(f, "0x%x", &v) != 1) v = 0;
    ::fclose(f);
    return v;
}

std::string uio_name(int num) {
    const std::string p = "/sys/class/uio/uio" + std::to_string(num) + "/name";
    FILE* f = ::fopen(p.c_str(), "r");
    if (!f) return {};
    char buf[128] = {0};
    if (!::fgets(buf, sizeof buf, f)) { ::fclose(f); return {}; }
    ::fclose(f);
    if (char* nl = ::strchr(buf, '\n')) *nl = 0;
    return buf;
}

int find_uio(const std::string& want) {
    DIR* d = ::opendir("/sys/class/uio");
    if (!d) return -1;
    int found = -1;
    while (dirent* e = ::readdir(d)) {
        if (::strncmp(e->d_name, "uio", 3) != 0) continue;
        const int num = ::atoi(e->d_name + 3);
        if (uio_name(num) == want) { found = num; break; }
    }
    ::closedir(d);
    return found;
}

// 依 map0 的實體位址尋找 UIO。位址是唯一不會變的識別 ——
// 名字取決於裝置樹節點怎麼命名(uio_pdrv_genirq 用 %pOFn,通常不等於 label)。
int find_uio_by_addr(uint64_t phys) {
    DIR* d = ::opendir("/sys/class/uio");
    if (!d) return -1;
    int found = -1;
    while (dirent* e = ::readdir(d)) {
        if (::strncmp(e->d_name, "uio", 3) != 0) continue;
        const std::string p = std::string("/sys/class/uio/") + e->d_name + "/maps/map0/addr";
        FILE* f = ::fopen(p.c_str(), "r");
        if (!f) continue;
        unsigned long long a = 0;
        const bool got = (::fscanf(f, "%llx", &a) == 1);
        ::fclose(f);
        if (got && a == phys) { found = ::atoi(e->d_name + 3); break; }
    }
    ::closedir(d);
    return found;
}

}  // namespace

AxiLiteIp::AxiLiteIp(const std::string& name, uint64_t ctrl_phys,
                     bool allow_devmem, size_t span)
    : ctrl_phys_(ctrl_phys) {
    int uio_num = find_uio(name);

    if (uio_num < 0 && ctrl_phys != 0) {
        uio_num = find_uio_by_addr(ctrl_phys);
        if (uio_num >= 0) matched_by_addr_ = true;
    }

    if (uio_num >= 0) {
        open_uio(uio_num);
        return;
    }

    if (!allow_devmem || ctrl_phys == 0)
        throw std::runtime_error(
            "找不到 UIO 裝置: 名稱 " + name +
            (ctrl_phys ? " 與位址 0x" + to_hex(ctrl_phys) + " 都" : std::string(" ")) +
            "沒有相符的 /dev/uioX"
            " (用 cat /sys/class/uio/uio*/name 確認實際名稱;檢查 device tree 的 "
            "compatible,以及 bootargs 是否有 uio_pdrv_genirq.of_id=generic-uio)" +
            (ctrl_phys ? std::string()
                       : std::string(" [未設定控制暫存器位址,無法依位址尋找或使用 /dev/mem]")));

    open_devmem(ctrl_phys, span);
}

AxiLiteIp::~AxiLiteIp() {
    if (base_) ::munmap(const_cast<uint32_t*>(base_), map_size_);
    if (fd_ >= 0) ::close(fd_);
}

void AxiLiteIp::open_uio(int uio_num) {
    actual_name_ = uio_name(uio_num);
    uio_index_   = uio_num;
    map_size_ = read_hex_file("/sys/class/uio/uio" + std::to_string(uio_num) +
                              "/maps/map0/size");
    if (map_size_ == 0) map_size_ = kDefaultSpan;

    const std::string dev = "/dev/uio" + std::to_string(uio_num);
    fd_ = ::open(dev.c_str(), O_RDWR);
    if (fd_ < 0)
        throw std::runtime_error("open " + dev + " 失敗: " + strerror(errno));

    void* p = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
        ::close(fd_); fd_ = -1;
        throw std::runtime_error("mmap 控制暫存器失敗");
    }
    base_    = static_cast<volatile uint32_t*>(p);
    has_irq_ = true;
}

void AxiLiteIp::open_devmem(uint64_t phys, size_t span) {
    map_size_ = span;
    fd_ = ::open("/dev/mem", O_RDWR | O_SYNC);
    if (fd_ < 0)
        throw std::runtime_error(std::string("open /dev/mem 失敗: ") + strerror(errno));

    void* p = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd_, static_cast<off_t>(phys));
    if (p == MAP_FAILED) {
        ::close(fd_); fd_ = -1;
        throw std::runtime_error("mmap /dev/mem 失敗(位址對嗎?需要 root)");
    }
    base_    = static_cast<volatile uint32_t*>(p);
    has_irq_ = false;
}

std::string AxiLiteIp::describe() const {
    if (!has_irq_)
        return "/dev/mem @0x" + to_hex(ctrl_phys_) + "(輪詢,無中斷)";
    std::string s = "/dev/uio" + std::to_string(uio_index_)
                  + " name=\"" + actual_name_ + "\"";
    if (matched_by_addr_)
        s += "  [名稱不符,靠位址找到 —— 建議改用這個名字呼叫 configure()]";
    return s;
}

// 寫 1 到 bit0,硬體會自動清掉;保留 auto_restart 位元
void AxiLiteIp::start() {
    const uint32_t c = rd(ap_ctrl::CTRL) & ap_ctrl::AUTO_RESTART;
    wr(ap_ctrl::CTRL, c | ap_ctrl::START);
}

void AxiLiteIp::irq_enable() {
    wr(ap_ctrl::GIE, 1);
    wr(ap_ctrl::IER, 1);   // 只開 ap_done
}

bool AxiLiteIp::wait_done_poll(int timeout_ms) {
    for (int i = 0; i < timeout_ms * 100; ++i) {
        if (is_done() || is_idle()) return true;
        ::usleep(10);
    }
    return false;
}

bool AxiLiteIp::wait_done_irq(int timeout_ms) {
    if (!has_irq_) return wait_done_poll(timeout_ms);   // /dev/mem 沒有中斷

    uint32_t unmask = 1;
    if (::write(fd_, &unmask, sizeof unmask) != static_cast<ssize_t>(sizeof unmask))
        return false;

    fd_set rs; FD_ZERO(&rs); FD_SET(fd_, &rs);
    timeval tv{ timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
    if (::select(fd_ + 1, &rs, nullptr, nullptr, &tv) <= 0) return false;

    uint32_t cnt = 0;
    if (::read(fd_, &cnt, sizeof cnt) != static_cast<ssize_t>(sizeof cnt)) return false;

    wr(ap_ctrl::ISR, rd(ap_ctrl::ISR));  // TOW:寫回同值清除
    return true;
}

bool AxiLiteIp::run(int timeout_ms) {
    if (has_irq_) irq_enable();
    start();
    return wait_done_irq(timeout_ms);
}


// ============================================================
//  資料搬移與 cache 同步
// ============================================================
bool stage_input(const std::shared_ptr<DmaPool>& pool, BufferCache& cache,
                 const cv::Mat& img, DmaInput& in, int slot) {
    in = DmaInput{};
    if (!pool || img.empty() || img.dims != 2) return false;

    const size_t bytes = img.total() * img.elemSize();
    Stopwatch sw;

    if (img.isContinuous() && pool->contains(img.data, bytes)) {
        in.phys      = pool->phys_of(img.data);
        in.data      = img.data;
        in.zero_copy = true;
    } else {
        DmaMat* m = cache.get(pool, img.rows, img.cols, img.type(), kStagingSlotBase + slot);
        if (!m) return false;
        img.copyTo(m->mat());          // 唯一一次複製:heap -> DMA 記憶體
        in.phys = m->phys();
        in.data = m->mat().data;
    }
    in.bytes   = bytes;
    in.copy_ms = in.zero_copy ? 0.0 : sw.ms();

    sw.reset();
    pool->sync_range_for_device(in.data, in.bytes);
    in.sync_ms = sw.ms();
    return true;
}

double finish_output(DmaPool& pool, const void* data, size_t bytes) {
    Stopwatch sw;
    pool.sync_range_for_cpu(data, bytes);
    return sw.ms();
}

double finish_output(DmaPool& pool, DmaMat& out) {
    return finish_output(pool, out.mat().data, out.data_bytes());
}

}  // namespace detail


// ============================================================
//  共用工具
// ============================================================
namespace {

// input_buffer 用的共用快取,不屬於任何一顆 IP
struct SharedInputs {
    std::mutex  mtx;
    BufferCache cache;
    static SharedInputs& get() { static SharedInputs s; return s; }
};

const CleanupRegistrar shared_inputs_cleanup{[] {
    auto& s = SharedInputs::get();
    std::lock_guard<std::mutex> lk(s.mtx);
    s.cache.clear();
}};

}  // namespace

cv::Mat input_buffer(int width, int height, int slot, int type) {
    auto& s = SharedInputs::get();
    std::lock_guard<std::mutex> lk(s.mtx);

    auto pool = pool_ptr();
    if (!pool) return cv::Mat();

    DmaMat* m = s.cache.get(pool, height, width, type, slot);
    return m ? m->mat() : cv::Mat();
}

cv::Mat make_test_pattern(int width, int height) {
    cv::Mat m(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = m.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            row[x * 3 + 0] = static_cast<uint8_t>(x * 255 / std::max(1, width - 1));
            row[x * 3 + 1] = static_cast<uint8_t>(y * 255 / std::max(1, height - 1));
            row[x * 3 + 2] = static_cast<uint8_t>((x + y) & 0xFF);
        }
    }
    return m;
}

}  // namespace hls