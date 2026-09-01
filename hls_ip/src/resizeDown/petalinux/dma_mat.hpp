// dma_mat.hpp — 讓 cv::Mat 直接落在 IP 能存取的實體連續記憶體上
//
// 核心概念:
//   cv::Mat 不是「配好再想辦法給 IP」,而是「先有 DMA buffer,Mat 去包它」。
//   cv::Mat 有一個建構子接受外部指標:
//       cv::Mat(rows, cols, type, void* data, size_t step)
//   這個建構子「不會複製、不會接管所有權」,Mat 只是一個 header。
//
// 硬體對應(resize_dpu.xsa):
//   gmem0/gmem1 是 128-bit → 一個 AXI word = 16 bytes
//   暫存器 out_words = 每列輸出佔幾個 word → 這就是 Mat 的 step / 16
//   所以把 Mat 的 step 對齊到 16 bytes,兩邊就能直接對上

#pragma once

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <string>
#include <map>
#include <vector>
#include <functional>
#include <dirent.h>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/types.h>

// CV_AUTOSTEP 定義在舊版 C API 的 opencv2/core/core_c.h,
// OpenCV 4 的 opencv.hpp 不再引入它,直接用會出現
//   identifier "CV_AUTOSTEP" is undefined
// 這裡自己補一個,值取自 OpenCV 原始碼,避免為了一個常數
// 把整個 C API 拉進來(core_c.h 會汙染一堆舊巨集)。
#ifndef CV_AUTOSTEP
#define CV_AUTOSTEP 0x7fffffff
#endif


// ============================================================
// ARM64 使用者空間 cache 維護
//
// 為什麼需要:dma_heap 的 begin/end_cpu_access 是走訪「已 attach 的
// 裝置」來做 cache 同步。我們從 userspace 直接操作 IP,沒有任何 driver
// attach 到這個 dmabuf,attachment list 是空的 —— DMA_BUF_IOCTL_SYNC
// 實際上什麼都沒做。症狀是小圖幾乎全錯、大圖幾乎全對(大的資料會自然
// 被擠出 cache)。
//
// DC CIVAC(clean & invalidate to PoC)在 EL0 可用,因為 Linux 會設定
// SCTLR_EL1.UCI。若某些核心關掉它,執行時會收到 SIGILL。
// ============================================================
namespace arm_cache {

#if defined(__aarch64__)

inline size_t line_size() {
    static const size_t sz = [] {
        uint64_t ctr = 0;
        __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
        return static_cast<size_t>(4u << ((ctr >> 16) & 0xF));   // DminLine
    }();
    return sz;
}

// clean & invalidate:對兩個方向都安全的超集合。
// 純 clean(DC CVAC)或純 invalidate(DC IVAC,EL1 限定)在這裡沒有
// 明顯好處,而 CIVAC 不會有「invalidate 掉尚未寫回的髒資料」的風險。
inline void flush(const void* addr, size_t size) {
    if (!addr || !size) return;
    const size_t line = line_size();
    uintptr_t p  = reinterpret_cast<uintptr_t>(addr) & ~(line - 1);
    const uintptr_t end = reinterpret_cast<uintptr_t>(addr) + size;
    for (; p < end; p += line)
        __asm__ volatile("dc civac, %0" :: "r"(p) : "memory");
    __asm__ volatile("dsb ish" ::: "memory");
    __asm__ volatile("isb" ::: "memory");
}

inline constexpr bool supported() { return true; }

#else

inline size_t line_size() { return 64; }
inline void flush(const void*, size_t) {}
inline constexpr bool supported() { return false; }

#endif

}  // namespace arm_cache

// ============================================================
// DMA-BUF Heaps 的 ABI(linux/dma-heap.h、linux/dma-buf.h)
// 直接寫在這裡,免得相依於特定版本的 kernel header。
// ============================================================
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

// 外部提供的記憶體來源。
// 讓 XRT、V4L2 之類的後端不必修改 DmaPool 本身 ——
// 只要交出「虛擬位址 + 實體位址 + 大小 + 同步函式」四樣東西即可。
struct ExternalMem {
    void*    virt = nullptr;
    uint64_t phys = 0;
    size_t   size = 0;
    std::string label = "external";
    // to_device = true 表示 CPU 寫完要 flush;false 表示 IP 寫完要 invalidate。
    // 若記憶體本來就 coherent 或 uncached,傳空的 function 即可。
    std::function<void(bool to_device)> sync;
};

// 給 DmaPool 建構子做多載用的標籤。
// heap = "auto" 表示自動挑一個(見 list_dma_heaps)。
struct DmaHeapTag {
    std::string heap = "auto";
};

// 列出 /dev/dma_heap 底下的 heap,依「配得到實體連續記憶體的可能性」排序:
//   1. 名稱含 cma      —— CMA heap,保證連續
//   2. reserved / carveout —— 裝置樹保留區,連續
//   3. 其他
//   4. system          —— 逐頁配置,幾乎不可能連續,排最後
inline std::vector<std::string> list_dma_heaps() {
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
// DmaPool:一塊實體連續記憶體 + 簡易配置器
// 後端用 ikwzm u-dma-buf(cached 映射,效能好,但要手動 sync)
// ============================================================
class DmaPool {
public:
    // 模式一:ikwzm u-dma-buf(cached 映射,效能好,需手動 sync)
    explicit DmaPool(const std::string& name) : name_(name) {
        const std::string sys = "/sys/class/u-dma-buf/" + name + "/";
        phys_ = read_val(sys + "phys_addr", true);
        size_ = static_cast<size_t>(read_val(sys + "size", false));
        if (!phys_ || !size_)
            throw std::runtime_error("讀不到 " + name + " 的 phys_addr/size"
                                     "(u-dma-buf 模組載入了嗎?)");

        fd_ = ::open(("/dev/" + name).c_str(), O_RDWR);
        if (fd_ < 0)
            throw std::runtime_error("open /dev/" + name + " 失敗: " + strerror(errno));

        void* p = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) { ::close(fd_); throw std::runtime_error("mmap " + name + " 失敗"); }
        virt_ = static_cast<uint8_t*>(p);
        cached_  = true;
        backend_ = Backend::UdmaBuf;

        free_[0] = size_;
    }

    // 模式四:由外部後端提供記憶體(XRT 等)。
    // DmaPool 只負責在這塊記憶體上做配置與位址換算,不管它從哪來。
    // 注意:記憶體的生命週期由提供者負責,DmaPool 不會 munmap 或釋放。
    explicit DmaPool(ExternalMem mem)
        : name_(mem.label), external_(std::move(mem)) {
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

    // 模式三:DMA-BUF Heaps(/dev/dma_heap/…)
    //
    // 這是 mainline 提供的 userspace 配置介面,不需要額外的 kernel module。
    // 從 "linux,cma" heap 配置保證實體連續。
    //
    // 障礙:dma_heap 刻意不把實體位址交給 userspace(正常用法是把 fd
    // 傳給另一個 driver)。但我們得把位址寫進 IP 的暫存器,所以改用
    // /proc/self/pagemap 做虛擬->實體轉換,並逐頁確認確實連續。
    // 需要 root;非特權行程讀到的 PFN 會被歸零。
    DmaPool(const DmaHeapTag& tag, size_t size) : name_("dma_heap:" + tag.heap) {
        const std::string dev = "/dev/dma_heap/" + tag.heap;
        int heap_fd = ::open(dev.c_str(), O_RDWR | O_CLOEXEC);
        if (heap_fd < 0) {
            std::string avail;
            for (const auto& h : list_dma_heaps()) avail += (avail.empty() ? "" : ", ") + h;
            throw std::runtime_error("open " + dev + " 失敗: " + strerror(errno) +
                                     "(可用的 heap: " +
                                     (avail.empty() ? "無" : avail) + ")");
        }

        dma_heap_abi::allocation_data req{};
        req.len       = size;
        req.fd_flags  = O_RDWR | O_CLOEXEC;
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

    // 模式二:/dev/mem 直接映射一塊 reserved memory
    //
    // 用於 u-dma-buf 還沒備妥時。O_SYNC 讓映射是 uncached(Device memory),
    // 所以 sync_for_* 變成 no-op —— CPU 寫下去直接進 DDR。
    //
    // 代價:uncached 記憶體上的 memcpy / OpenCV 運算會慢一個數量級。
    // 而且這塊位址必須是裝置樹裡 no-map 的 reserved-memory,
    // 否則會踩到 kernel 正在用的 RAM。
    DmaPool(uint64_t phys, size_t size) : name_("devmem") {
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

    ~DmaPool() {
        // 外部提供的記憶體由提供者自行釋放
        if (backend_ != Backend::External && virt_) ::munmap(virt_, size_);
        if (fd_ >= 0) ::close(fd_);
    }

    DmaPool(const DmaPool&) = delete;
    DmaPool& operator=(const DmaPool&) = delete;

    // 配置一塊對齊的區域,回傳虛擬指標
    void* alloc(size_t bytes, size_t align = 64) {
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            size_t off = it->first, len = it->second;
            size_t aligned = (off + align - 1) & ~(align - 1);
            size_t pad = aligned - off;
            if (len < pad + bytes) continue;

            free_.erase(it);
            if (pad) free_[off] = pad;
            size_t tail_off = aligned + bytes;
            size_t tail_len = len - pad - bytes;
            if (tail_len) free_[tail_off] = tail_len;

            used_[aligned] = bytes;
            if (aligned + bytes > used_end_) used_end_ = aligned + bytes;
            return virt_ + aligned;
        }
        throw std::runtime_error("DmaPool 空間不足");
    }

    void free(void* p) {
        if (!p) return;
        size_t off = static_cast<uint8_t*>(p) - virt_;
        auto it = used_.find(off);
        if (it == used_.end()) return;
        free_[off] = it->second;
        used_.erase(it);
        coalesce();
    }

    // *** 這是關鍵:虛擬指標 → IP 要的實體位址 ***
    uint64_t phys_of(const void* p) const {
        auto* u = static_cast<const uint8_t*>(p);
        if (u < virt_ || u >= virt_ + size_)
            throw std::runtime_error("指標不在 DMA pool 內 —— 這個 Mat 不是從 pool 配出來的");
        return phys_ + static_cast<uint64_t>(u - virt_);
    }

    bool contains(const void* p) const {
        auto* u = static_cast<const uint8_t*>(p);
        return u >= virt_ && u < virt_ + size_;
    }

    // gmem0/1 接 S_AXI_HP2/HP3,非 coherent → cached 模式一定要 sync。
    // uncached(/dev/mem)模式下資料本來就直接進 DDR,不需要同步。
    // ---- 全範圍(相容用)----
    void sync_for_device() { sync_range_for_device(virt_, active_bytes()); }
    void sync_for_cpu()    { sync_range_for_cpu(virt_, active_bytes()); }

    // ---- 範圍式:只維護真正用到的那一塊 ----
    //
    // 這件事對效能影響很大。輸入 6 MB、輸出 0.7 MB 的情況下,
    // 對整個 active 範圍做兩次維護 = 處理 14 MB;
    // 改成「啟動前只 flush 輸入、完成後只 invalidate 輸出」= 6.9 MB,
    // 少一半以上。
    void sync_range_for_device(const void* p, size_t n) {   // CPU 寫完 -> flush
        if (!n) return;
        if (manual_cache_) arm_cache::flush(p, n);
        switch (backend_) {
            case Backend::UdmaBuf: poke_range(p, n, "sync_for_device"); break;
            case Backend::DmaHeap: dmabuf_sync(dma_heap_abi::SYNC_END |
                                               dma_heap_abi::SYNC_RW); break;
            case Backend::External: if (external_.sync) external_.sync(true); break;
            case Backend::DevMem:  break;   // uncached,不需要
        }
    }
    void sync_range_for_cpu(const void* p, size_t n) {      // IP 寫完 -> invalidate
        if (!n) return;
        if (manual_cache_) arm_cache::flush(p, n);
        switch (backend_) {
            case Backend::UdmaBuf: poke_range(p, n, "sync_for_cpu"); break;
            case Backend::DmaHeap: dmabuf_sync(dma_heap_abi::SYNC_START |
                                               dma_heap_abi::SYNC_RW); break;
            case Backend::External: if (external_.sync) external_.sync(false); break;
            case Backend::DevMem:  break;
        }
    }

    bool is_cached() const { return cached_; }

    // 是否由本程式自行執行 cache 維護指令
    bool manual_cache() const { return manual_cache_; }
    void set_manual_cache(bool on) { manual_cache_ = on && arm_cache::supported(); }

    // 已配置出去的範圍。cache 維護只需涵蓋這一段,
    // 對整個 pool 做 flush 在 32MB 時會明顯拖慢。
    size_t active_bytes() const { return used_end_; }
    const std::string& name() const { return name_; }

    uint8_t* base() const { return virt_; }
    uint64_t phys() const { return phys_; }
    size_t   size() const { return size_; }

private:
    static uint64_t read_val(const std::string& p, bool hex) {
        FILE* f = ::fopen(p.c_str(), "r");
        if (!f) return 0;
        unsigned long long v = 0;
        if (::fscanf(f, hex ? "%llx" : "%llu", &v) != 1) v = 0;
        ::fclose(f);
        return v;
    }
    // u-dma-buf 支援指定範圍:先寫 sync_offset / sync_size,再觸發
    void poke_range(const void* p, size_t n, const char* attr) {
        const size_t off = static_cast<const uint8_t*>(p) - virt_;
        poke("sync_offset", static_cast<long long>(off));
        poke("sync_size",   static_cast<long long>(n));
        poke(attr, 1);
    }

    void poke(const char* attr, long long v) {
        std::string p = "/sys/class/u-dma-buf/" + name_ + "/" + attr;
        FILE* f = ::fopen(p.c_str(), "w");
        if (!f) return;
        ::fprintf(f, "%lld", v);
        ::fclose(f);
    }
    void coalesce() {
        for (auto it = free_.begin(); it != free_.end(); ) {
            auto nx = std::next(it);
            if (nx != free_.end() && it->first + it->second == nx->first) {
                it->second += nx->second;
                free_.erase(nx);
            } else ++it;
        }
    }

    void dmabuf_sync(__u64 flags) {
        dma_heap_abi::buf_sync s{};
        s.flags = flags;
        ::ioctl(fd_, dma_heap_abi::IOCTL_SYNC, &s);
    }

    // 讀 /proc/self/pagemap 取得實體位址,並確認整塊連續
    static uint64_t resolve_phys(void* virt, size_t size) {
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
                if (pfn_at(i) != first + i) {
                    ::close(pm);
                    throw std::runtime_error(
                        "dma_heap 配到的記憶體不是實體連續的"
                        "(換用 linux,cma heap,或改用 u-dma-buf)");
                }
            }
        } catch (...) { ::close(pm); throw; }

        ::close(pm);
        return first * page;
    }

    static std::string to_hex(uint64_t v) {
        char b[32]; std::snprintf(b, sizeof b, "%llx", (unsigned long long)v);
        return b;
    }

    enum class Backend { UdmaBuf, DevMem, DmaHeap, External };

    std::string name_;
    bool cached_ = true;
    Backend backend_ = Backend::UdmaBuf;
    ExternalMem external_;
    bool   manual_cache_ = false;
    size_t used_end_ = 0;
    int fd_ = -1;
    uint8_t* virt_ = nullptr;
    uint64_t phys_ = 0;
    size_t   size_ = 0;
    std::map<size_t, size_t> free_, used_;
};

// ============================================================
// DmaMat:一個「知道自己實體位址在哪」的 cv::Mat
// ============================================================
class DmaMat {
public:
    DmaMat() = default;

    // 起點對齊 page。AXI4 的兩條規則剛好在 4096 交會:
    //   1. 單一 burst 不得跨越 4KB 邊界(規格硬性要求)
    //   2. INCR burst 最長 256 beats;128-bit 埠下 256 × 16 = 4096 bytes
    // 所以 4KB 對齊時,每個 burst 都能開到最大長度而不被邊界切斷。
    static constexpr size_t kPageAlign = 4096;

    // Cortex-A53 的 cache line。長度必須補到它的倍數,
    // 否則尾端會與下一塊 buffer 共用同一條 line,sync 時互相破壞。
    static constexpr size_t kCacheLine = 64;

    static constexpr size_t align_up(size_t v, size_t a) {
        return (v + a - 1) & ~(a - 1);
    }

    // row_align 預設 1 = 緊密排列。
    //
    // *** 不要改成 16 ***
    // resize_kernel 的 compute_side 把輸入當成一條連續的 RGB888 bit 流,
    // 用 leftover 機制處理 pixel 跨越 128-bit 邊界,完全沒有 stride 概念。
    // 列的邊界是靠 ox >= out_w 推算的,所以每列一旦補了 padding,
    // 第二列開始就整個錯位。輸出側同理。
    DmaMat(DmaPool& pool, int rows, int cols, int type, size_t row_align = 1)
        : pool_(&pool) {
        const size_t elem = CV_ELEM_SIZE(type);

        step_ = align_up(static_cast<size_t>(cols) * elem, row_align);

        // 影像實際資料量。row_align=1 時就是 cols*elem*rows
        data_bytes_ = step_ * static_cast<size_t>(rows);

        // 長度補到 cache line:防尾端與後面共用同一條 line
        bytes_ = align_up(data_bytes_, kCacheLine);

        // 起點對到 page:讓 AXI burst 開得滿
        // 4096 是 64 的倍數,所以 cache line 對齊自動成立
        ptr_ = pool.alloc(bytes_, kPageAlign);

        mat_ = cv::Mat(rows, cols, type, ptr_, step_);
    }

    ~DmaMat() { if (pool_ && ptr_) pool_->free(ptr_); }

    DmaMat(DmaMat&& o) noexcept { swap(o); }
    DmaMat& operator=(DmaMat&& o) noexcept { swap(o); return *this; }
    DmaMat(const DmaMat&) = delete;
    DmaMat& operator=(const DmaMat&) = delete;

    cv::Mat&       mat()       { return mat_; }
    const cv::Mat& mat() const { return mat_; }
    operator cv::Mat&()        { return mat_; }     // 可直接餵給 OpenCV 函式

    // *** 寫進 in_ptr / out_ptr 的就是這個 ***
    uint64_t phys() const { return pool_->phys_of(ptr_); }

    size_t step()       const { return step_; }
    size_t data_bytes() const { return data_bytes_; }   // 真正的影像資料量
    size_t bytes()      const { return bytes_; }        // 含 cache line 補齊的配置量

    // 整張影像佔幾個 128-bit word。
    // 輸入側對應 total_words,輸出側對應 out_words —— 兩者都是「整張圖」的量,
    // 不是每列的量。
    uint32_t total_words() const { return static_cast<uint32_t>(data_bytes_ / 16); }

    // 緊密排列時應為 true。若為 false 表示 step 有 padding,
    // 這個 kernel 會算錯,務必檢查。
    bool is_packed() const {
        return step_ == static_cast<size_t>(mat_.cols) * mat_.elemSize();
    }

    // 從外面來的 Mat(imread / VideoCapture)搬進來,必要時自動轉尺寸型別
    void import_from(const cv::Mat& src) {
        CV_Assert(src.rows == mat_.rows && src.cols == mat_.cols && src.type() == mat_.type());
        src.copyTo(mat_);      // Mat 已綁定外部記憶體,copyTo 不會重新配置
    }

private:
    void swap(DmaMat& o) {
        std::swap(pool_, o.pool_); std::swap(ptr_, o.ptr_);
        std::swap(step_, o.step_); std::swap(bytes_, o.bytes_);
        std::swap(data_bytes_, o.data_bytes_);
        std::swap(mat_, o.mat_);
    }
    DmaPool* pool_ = nullptr;
    void*    ptr_  = nullptr;
    size_t   step_ = 0, bytes_ = 0, data_bytes_ = 0;
    cv::Mat  mat_;
};

// ============================================================
// 進階:自訂 MatAllocator
// 讓 cv::Mat::create() 自動從 DMA pool 配置,連 cv::resize() 這種
// 內部會自己配輸出的函式也能落在 DMA 記憶體上。
//
// 用法:
//   static DmaAllocator alloc(pool);
//   cv::Mat::setDefaultAllocator(&alloc);   // 全域切換,影響所有 Mat
//   ... 用完記得 setDefaultAllocator(nullptr) 換回來
//
// 注意:imread / imdecode 內部有自己的緩衝路徑,不保證走這裡。
// ============================================================
class DmaAllocator : public cv::MatAllocator {
public:
    explicit DmaAllocator(DmaPool& pool) : pool_(pool) {}

    cv::UMatData* allocate(int dims, const int* sizes, int type,
                           void* data0, size_t* step,
                           cv::AccessFlag, cv::UMatUsageFlags) const override {
        size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; --i) {
            if (step) {
                if (data0 && step[i] != static_cast<size_t>(CV_AUTOSTEP))
                    total = step[i];
                else step[i] = total;
            }
            total *= sizes[i];
        }
        // 長度補到 cache line,起點對到 page,與 DmaMat 一致
        const size_t padded = DmaMat::align_up(total, DmaMat::kCacheLine);
        uint8_t* data = data0 ? static_cast<uint8_t*>(data0)
                              : static_cast<uint8_t*>(
                                    pool_.alloc(padded, DmaMat::kPageAlign));

        auto* u = new cv::UMatData(this);
        u->data = u->origdata = data;
        u->size = total;
        if (data0) u->flags |= cv::UMatData::USER_ALLOCATED;
        return u;
    }

    bool allocate(cv::UMatData* u, cv::AccessFlag, cv::UMatUsageFlags) const override {
        if (!u) return false;
        u->urefcount++;
        return true;
    }

    void deallocate(cv::UMatData* u) const override {
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

private:
    DmaPool& pool_;
};