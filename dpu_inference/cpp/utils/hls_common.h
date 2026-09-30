// hls_common.h — 所有 IP 共用的介面(對外 API + 函式庫內部宣告)
//
// 檔案結構:
//   include/hls_common.h     共用介面(本檔)
//   include/hls_<ip>.h       每顆 IP 的 API
//   src/hls_common.cpp       共用層的實作
//   src/hls_<ip>.cpp         每顆 IP 的硬體位址、暫存器、driver、API 實作
//
// 本檔分兩段:
//   namespace hls           對外 API:pool 設定與查詢、Timing、input_buffer…
//   namespace hls::detail   函式庫內部:DmaPool、AxiLiteIp、IpHolder、stage_input…
//                           給 src/hls_<ip>.cpp 用;應用程式請不要直接呼叫。
//
// Linux 系統標頭(mmap、ioctl、UIO、dma_heap)只出現在 .cpp,不會漏到這裡。

#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace hls {

// ============================================================
//  共用 DMA pool 的設定
//
//  所有 IP 共用同一塊實體連續記憶體,IP 之間串接只需傳實體位址。
//  以下設定要在第一次用到任何 IP 之前呼叫;都不呼叫時預設用 u-dma-buf 的 udmabuf0。
// ============================================================

// 指定 u-dma-buf 的裝置名稱(cached 映射,效能最好)
void set_pool(const std::string& udmabuf_name);

// 改用 DMA-BUF Heaps(/dev/dma_heap/…)。mainline 內建,不需要 u-dma-buf 模組。
// heap = "auto" 時依序嘗試 cma → reserved/carveout → 其他 → system。
// 取得實體位址靠 /proc/self/pagemap,需要 root。
void use_dma_heap(const std::string& heap = "auto",
                  size_t size = 32u * 1024 * 1024);

// 改用 /dev/mem 映射裝置樹裡 no-map 的 reserved-memory。
// 映射是 uncached,不需 sync,但 CPU 端的搬移與運算會明顯變慢。
void use_reserved_memory(uint64_t phys, size_t size);

// 由外部後端提供記憶體(XRT、V4L2…)。
// 只要交出「虛擬位址 + 實體位址 + 大小 + 同步函式」即可。
// 記憶體的生命週期由提供者負責,函式庫不會 munmap 或釋放。
struct ExternalMem {
    void*    virt = nullptr;
    uint64_t phys = 0;
    size_t   size = 0;
    std::string label = "external";
    // to_device = true:CPU 寫完要 flush;false:IP 寫完要 invalidate。
    // 記憶體本來就 coherent 或 uncached 時留空即可。
    std::function<void(bool to_device)> sync;
};

// provider 會在第一次需要 pool 時被呼叫一次
void use_external_memory(std::function<ExternalMem()> provider);


// ============================================================
//  共用 DMA pool 的查詢
// ============================================================
bool        pool_available();
std::string pool_error();
std::string pool_info();     // 目前走哪一條路、位址、大小、cache 模式

// 釋放所有 IP 模組已快取的 buffer(切換工作尺寸、或想回收空間時用)
void clear_buffers();


// ============================================================
//  共用工具
// ============================================================

// 各階段耗時,用來判斷瓶頸在哪
struct Timing {
    double copy_ms = 0;   // 把資料搬進 DMA 記憶體(zero-copy 時為 0)
    double sync_ms = 0;   // cache 維護(進 + 出)
    double run_ms  = 0;   // IP 實際運算 + 等待
    double post_ms = 0;   // IP 之後的 CPU 後處理
    double total_ms = 0;
    bool   zero_copy = false;   // 輸入是否已在 DMA 記憶體,免去複製
};

// 取得一塊位於 DMA 記憶體的輸入緩衝。
//
// 把影像直接解碼 / 擷取到這裡,任何一顆 IP 都會走 zero-copy,
// 省掉整整一次全幀複製(1080p 約 3 ms)。
//
// slot 讓同一組尺寸能拿到多塊獨立的緩衝 —— 多執行緒 pipeline 必須用
// 不同的 slot,否則擷取端會覆寫 IP 正在讀的資料。
// 相同的 (width, height, slot, type) 每次回傳同一塊。
// pool 不可用時回傳空的 Mat。
cv::Mat input_buffer(int width, int height, int slot = 0, int type = CV_8UC3);

// 產生一張適合驗證硬體的 CV_8UC3 測試圖。
// 三個通道方向各異 —— 單色圖與純水平漸層在輸出欄錯位時值剛好相同,
// 完全測不出該類 bug。
cv::Mat make_test_pattern(int width, int height);


// ################################################################
//
//  以下是函式庫內部(hls::detail)
//  給 src/hls_<ip>.cpp 用,應用程式請不要直接呼叫。
//
//  新增 IP 時,src/hls_<ip>.cpp 的寫法照 src/hls_resize.cpp:
//    1. 匿名 namespace 裡寫硬體位址、暫存器 offset、繼承 AxiLiteIp 的 Kernel
//    2. device() 回傳 static IpHolder<Kernel>,並用 CleanupRegistrar 註冊清除
//    3. 實作 include/hls_<ip>.h 宣告的 API:stage_input → program → run → finish_output
//
// ################################################################
namespace detail {


// ============================================================
//  DmaPool:一塊實體連續記憶體 + 簡易配置器 + cache 同步
//  配置/釋放與 u-dma-buf 的範圍同步都自己上鎖,多顆 IP 可同時使用。
// ============================================================
struct DmaHeapTag {
    std::string heap = "auto";
};

// 列出 /dev/dma_heap 底下的 heap,依「配得到實體連續記憶體的可能性」排序
std::vector<std::string> list_dma_heaps();

class DmaPool {
public:
    explicit DmaPool(const std::string& udmabuf_name);   // u-dma-buf
    DmaPool(uint64_t phys, size_t size);                  // /dev/mem reserved-memory
    DmaPool(const DmaHeapTag& tag, size_t size);          // DMA-BUF Heaps
    explicit DmaPool(ExternalMem mem);                    // 外部提供
    ~DmaPool();

    DmaPool(const DmaPool&) = delete;
    DmaPool& operator=(const DmaPool&) = delete;

    void* alloc(size_t bytes, size_t align = 64);
    void  free(void* p);

    // 虛擬指標 → IP 要的實體位址。不在 pool 內會丟例外。
    uint64_t phys_of(const void* p) const;
    bool contains(const void* p) const;
    bool contains(const void* p, size_t n) const;   // [p, p+n) 整段都在 pool 內

    // 全範圍(相容用)
    void sync_for_device();
    void sync_for_cpu();
    // 範圍式:只維護真正用到的那一塊
    void sync_range_for_device(const void* p, size_t n);   // CPU 寫完 -> flush
    void sync_range_for_cpu(const void* p, size_t n);      // IP 寫完 -> invalidate

    bool is_cached() const    { return cached_; }
    bool manual_cache() const { return manual_cache_; }   // 是否自行執行 cache 維護指令
    void set_manual_cache(bool on);

    size_t active_bytes() const;                          // 已配置出去的範圍
    const std::string& name() const { return name_; }
    uint8_t* base() const { return virt_; }
    uint64_t phys() const { return phys_; }
    size_t   size() const { return size_; }

private:
    enum class Backend { UdmaBuf, DevMem, DmaHeap, External };

    void poke_range(const void* p, size_t n, const char* attr);
    void poke(const char* attr, long long v);
    void coalesce();                   // 呼叫端須持有 alloc_mtx_
    void dmabuf_sync(uint64_t flags);

    std::string name_;
    bool        cached_ = true;
    Backend     backend_ = Backend::UdmaBuf;
    ExternalMem external_;
    bool        manual_cache_ = false;
    size_t      used_end_ = 0;
    int         fd_ = -1;
    uint8_t*    virt_ = nullptr;
    uint64_t    phys_ = 0;
    size_t      size_ = 0;
    std::map<size_t, size_t> free_, used_;
    mutable std::mutex alloc_mtx_;   // 保護 free_ / used_ / used_end_
    std::mutex         sync_mtx_;    // 保護 u-dma-buf 的成組 sysfs 寫入
};


// ============================================================
//  DmaMat:一個「知道自己實體位址在哪」的 cv::Mat
//  先有 DMA buffer,Mat 只是包住它的 header(不複製、不接管所有權)。
// ============================================================
class DmaMat {
public:
    // 起點對齊 page:AXI4 burst 不得跨 4KB,且 128-bit × 256 beats 剛好 4KB
    static constexpr size_t kPageAlign = 4096;
    // Cortex-A53 cache line:長度補齊,避免尾端與下一塊共用 line
    static constexpr size_t kCacheLine = 64;

    static constexpr size_t align_up(size_t v, size_t a) {
        return (v + a - 1) & ~(a - 1);
    }

    DmaMat() = default;
    // row_align 預設 1 = 緊密排列。串流式 kernel 沒有 stride 概念,不要改。
    DmaMat(DmaPool& pool, int rows, int cols, int type, size_t row_align = 1);
    ~DmaMat();

    DmaMat(DmaMat&& o) noexcept;
    DmaMat& operator=(DmaMat&& o) noexcept;
    DmaMat(const DmaMat&) = delete;
    DmaMat& operator=(const DmaMat&) = delete;

    cv::Mat&       mat()       { return mat_; }
    const cv::Mat& mat() const { return mat_; }
    operator cv::Mat&()        { return mat_; }

    uint64_t phys() const;                               // 寫進 IP 指標暫存器的值
    size_t step()       const { return step_; }
    size_t data_bytes() const { return data_bytes_; }    // 真正的影像資料量
    size_t bytes()      const { return bytes_; }         // 含 cache line 補齊的配置量
    uint32_t total_words() const { return static_cast<uint32_t>(data_bytes_ / 16); }
    bool is_packed() const;
    void import_from(const cv::Mat& src);

private:
    void swap(DmaMat& o) noexcept;

    DmaPool* pool_ = nullptr;
    void*    ptr_  = nullptr;
    size_t   step_ = 0, bytes_ = 0, data_bytes_ = 0;
    cv::Mat  mat_;
};


// 讓 cv::Mat::create() 自動從 DMA pool 配置。
//   static DmaAllocator alloc(*pool_ptr());
//   cv::Mat::setDefaultAllocator(&alloc);   // 全域切換,用完記得換回 nullptr
// 注意:imread / imdecode 內部有自己的緩衝路徑,不保證走這裡。
class DmaAllocator : public cv::MatAllocator {
public:
    explicit DmaAllocator(DmaPool& pool) : pool_(pool) {}
    cv::UMatData* allocate(int dims, const int* sizes, int type, void* data0, size_t* step,
                           cv::AccessFlag, cv::UMatUsageFlags) const override;
    bool allocate(cv::UMatData* u, cv::AccessFlag, cv::UMatUsageFlags) const override;
    void deallocate(cv::UMatData* u) const override;
private:
    DmaPool& pool_;
};


// ============================================================
//  共用 pool 的內部存取
// ============================================================

// 回傳空的 shared_ptr 表示開不起來。
// IP 模組配置 buffer 時應持有這份 shared_ptr,確保 pool 比 buffer 晚死。
std::shared_ptr<DmaPool> pool_ptr();

// 由外部直接提供 DmaPool(比 use_external_memory 更底層)
void set_pool_factory(std::function<std::shared_ptr<DmaPool>()> f);


// ============================================================
//  BufferCache:依 (rows, cols, type, slot) 快取 DmaMat
//  本身不上鎖,由持有者(IpHolder)的 mutex 保護。
// ============================================================
class BufferCache {
public:
    // 尺寸相同就沿用,不同就配新的;失敗回傳 nullptr
    DmaMat* get(const std::shared_ptr<DmaPool>& pool, int rows, int cols, int type,
                int slot = 0);
    void clear();

private:
    using Key = std::tuple<int, int, int, int>;
    // *** 宣告順序有意義 ***:keepalive_ 必須在 map_ 之前,才會比 DmaMat 晚解構
    std::shared_ptr<DmaPool> keepalive_;
    std::map<Key, std::unique_ptr<DmaMat>> map_;
};

// 把清除函式掛進 hls::clear_buffers()。在 IP 的 .cpp 裡宣告一個 static 物件即可。
struct CleanupRegistrar {
    explicit CleanupRegistrar(void (*fn)());
};


// ============================================================
//  AxiLiteIp —— Vitis HLS ap_ctrl_hs IP 的通用控制
//  每顆 IP 的 Kernel 繼承它,只需補上自己的參數暫存器。
// ============================================================
namespace ap_ctrl {
inline constexpr uint32_t CTRL = 0x00;   // b0 start, b1 done, b2 idle, b3 ready, b7 auto_restart
inline constexpr uint32_t GIE  = 0x04;   // Global Interrupt Enable
inline constexpr uint32_t IER  = 0x08;   // b0 ap_done, b1 ap_ready
inline constexpr uint32_t ISR  = 0x0c;   // TOW:寫 1 清除

inline constexpr uint32_t START        = 1u << 0;
inline constexpr uint32_t DONE         = 1u << 1;
inline constexpr uint32_t IDLE         = 1u << 2;
inline constexpr uint32_t READY        = 1u << 3;
inline constexpr uint32_t AUTO_RESTART = 1u << 7;
}  // namespace ap_ctrl

class AxiLiteIp {
public:
    static constexpr size_t kDefaultSpan = 0x10000;

    // 尋找順序:UIO 名稱 → UIO map0 位址 → /dev/mem(僅 allow_devmem 時)
    // 走到 /dev/mem 就沒有中斷,等待會自動改成輪詢。
    AxiLiteIp(const std::string& name, uint64_t ctrl_phys = 0,
              bool allow_devmem = false, size_t span = kDefaultSpan);
    virtual ~AxiLiteIp();

    AxiLiteIp(const AxiLiteIp&) = delete;
    AxiLiteIp& operator=(const AxiLiteIp&) = delete;

    // ---- 裝置資訊 ----
    bool has_irq() const { return has_irq_; }
    const std::string& actual_name() const { return actual_name_; }
    int  uio_index() const { return uio_index_; }
    bool matched_by_addr() const { return matched_by_addr_; }   // true = 名字對不上
    std::string describe() const;

    // ---- 暫存器讀寫 ----
    void     wr(uint32_t off, uint32_t v) { base_[off / 4] = v; }
    uint32_t rd(uint32_t off) const       { return base_[off / 4]; }
    void     wr64(uint32_t off, uint64_t v) {
        wr(off,     static_cast<uint32_t>(v & 0xFFFFFFFFu));
        wr(off + 4, static_cast<uint32_t>(v >> 32));
    }

    // ---- ap_ctrl_hs ----
    bool is_idle()  const { return rd(ap_ctrl::CTRL) & ap_ctrl::IDLE; }
    bool is_ready() const { return rd(ap_ctrl::CTRL) & ap_ctrl::READY; }
    bool is_done()  const { return rd(ap_ctrl::CTRL) & ap_ctrl::DONE; }
    void start();
    void irq_enable();
    bool wait_done_poll(int timeout_ms = 2000);
    bool wait_done_irq(int timeout_ms = 2000);   // 沒有中斷時自動改輪詢

    // 啟動並等待完成。參數暫存器要先寫好。
    bool run(int timeout_ms = 2000);

private:
    void open_uio(int uio_num);
    void open_devmem(uint64_t phys, size_t span);

    int         fd_ = -1;
    volatile uint32_t* base_ = nullptr;
    size_t      map_size_ = 0;
    uint64_t    ctrl_phys_ = 0;
    bool        has_irq_ = false;
    bool        matched_by_addr_ = false;
    int         uio_index_ = -1;
    std::string actual_name_;
};


// ============================================================
//  IpHolder<Kernel> —— 每顆 IP 的單例
//  Kernel 必須能用 (name, ctrl_phys, allow_devmem) 建構。
//  樣板必須看得到實作,所以整個寫在這裡。
// ============================================================
template <class Kernel>
class IpHolder {
public:
    IpHolder(std::string name, uint64_t ctrl_phys)
        : default_name_(name), default_phys_(ctrl_phys),
          name_(std::move(name)), ctrl_phys_(ctrl_phys) {}

    IpHolder(const IpHolder&) = delete;
    IpHolder& operator=(const IpHolder&) = delete;

    const std::string& default_name() const { return default_name_; }
    uint64_t           default_phys() const { return default_phys_; }

    // ---- 設定(自行上鎖)----
    void set_name(const std::string& n) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (n == name_) return;
        name_ = n;
        reset_locked();
        cache_.clear();
    }
    void set_ctrl_phys(uint64_t phys, bool allow_devmem) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (phys == ctrl_phys_ && allow_devmem == allow_devmem_) return;
        ctrl_phys_    = phys;
        allow_devmem_ = allow_devmem;
        reset_locked();
    }

    // ---- 查詢(自行上鎖)----
    bool open() {
        std::lock_guard<std::mutex> lk(mtx_);
        return try_open() != nullptr;
    }
    bool using_irq() {
        std::lock_guard<std::mutex> lk(mtx_);
        Kernel* ip = try_open();
        return ip && ip->has_irq();
    }
    std::string info() {
        std::lock_guard<std::mutex> lk(mtx_);
        Kernel* ip = try_open();
        return ip ? ip->describe() : "未開啟: " + error_;
    }
    std::string error() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return error_;
    }
    void clear_cache() {
        std::lock_guard<std::mutex> lk(mtx_);
        cache_.clear();
    }

    // ---- 以下呼叫端必須先持有 mutex() ----
    Kernel* try_open() {
        if (tried_) return ip_.get();
        tried_ = true;
        try {
            ip_ = std::make_unique<Kernel>(name_, ctrl_phys_, allow_devmem_);
        } catch (const std::exception& e) {
            error_ = e.what();
            ip_.reset();
        }
        return ip_.get();
    }
    BufferCache& cache() { return cache_; }
    std::mutex&  mutex() { return mtx_; }

private:
    void reset_locked() {
        ip_.reset();
        tried_ = false;
        error_.clear();
    }

    const std::string default_name_;
    const uint64_t    default_phys_;
    std::string name_;
    uint64_t    ctrl_phys_ = 0;
    bool        allow_devmem_ = false;
    std::string error_;
    bool        tried_ = false;
    std::unique_ptr<Kernel> ip_;
    BufferCache cache_;
    mutable std::mutex mtx_;
};


// ============================================================
//  資料搬移與 cache 同步
//
//  每顆 IP 的執行流程:
//    stage_input()    輸入搬進 DMA 記憶體(已在裡面就 zero-copy)+ flush
//    寫參數暫存器、run()
//    finish_output()  invalidate 輸出
// ============================================================
class Stopwatch {
public:
    Stopwatch() : t0_(std::chrono::steady_clock::now()) {}
    void reset() { t0_ = std::chrono::steady_clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0_).count();
    }
private:
    std::chrono::steady_clock::time_point t0_;
};

struct DmaInput {
    uint64_t    phys = 0;        // 寫進 IP 輸入指標暫存器的值
    const void* data = nullptr;  // 對應的虛擬位址
    size_t      bytes = 0;
    bool        zero_copy = false;
    double      copy_ms = 0;
    double      sync_ms = 0;
};

// staging 用的 slot 從這裡開始,跟 IP 的輸出 buffer 錯開 ——
// 否則輸入與輸出同尺寸的 IP 會拿到同一塊記憶體。
inline constexpr int kStagingSlotBase = 1 << 16;

// 只支援 2 維 Mat。失敗回傳 false。
bool stage_input(const std::shared_ptr<DmaPool>& pool, BufferCache& cache,
                 const cv::Mat& img, DmaInput& in, int slot = 0);

// IP 寫完之後 invalidate 輸出,回傳耗時(ms)
double finish_output(DmaPool& pool, const void* data, size_t bytes);
double finish_output(DmaPool& pool, DmaMat& out);

}  // namespace detail

}  // namespace hls