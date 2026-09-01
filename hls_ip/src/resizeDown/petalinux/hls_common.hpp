// hls_common.hpp — 所有 IP 模組共用的基礎設施
//
// 分層原則:
//   hls_common.hpp   共用的東西:DMA pool、buffer 快取、錯誤訊息
//   hls_<ip>.hpp     每顆 IP 一個檔案,各自一個 namespace hls::<ip>
//
// 為什麼 pool 要共用:
//   u-dma-buf 是一塊固定大小的實體連續記憶體。若每顆 IP 各自開一個
//   udmabuf 節點,裝置樹要切死每塊多大,用不到的就浪費;而且 IP 之間
//   要串接時(resize 的輸出直接餵給下一顆),資料還得跨 pool 複製。
//   共用一個 pool,串接就只是傳一個實體位址。

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <functional>

#include "dma_mat.hpp"

namespace hls {

namespace detail {

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
        res_phys_ = 0;
        pool_.reset();
        tried_ = false;
    }

    // 改用 /dev/mem 映射一塊 reserved memory
    void set_reserved(uint64_t phys, size_t size) {
        std::lock_guard<std::mutex> lk(mtx_);
        mode_ = Mode::Reserved;
        res_phys_ = phys; res_size_ = size;
        pool_.reset(); tried_ = false;
    }

    // 由外部提供 pool 的建立方式(XRT 等)
    void set_factory(std::function<std::shared_ptr<DmaPool>()> f) {
        std::lock_guard<std::mutex> lk(mtx_);
        mode_ = Mode::Factory;
        factory_ = std::move(f);
        pool_.reset(); tried_ = false;
    }

    // 改用 DMA-BUF Heaps
    void set_dma_heap(const std::string& heap, size_t size) {
        std::lock_guard<std::mutex> lk(mtx_);
        mode_ = Mode::DmaHeap;
        heap_ = heap; res_size_ = size;
        pool_.reset(); tried_ = false;
    }

    // 回傳空的 shared_ptr 表示開不起來,呼叫端應退回 CPU 路徑。
    //
    // 用 shared_ptr 而非裸指標,是為了讓 pool 活得比所有使用者久:
    // 各 IP 模組的 BufferCache 也是 static,而 static 物件的解構順序
    // 是建立順序的反序 —— 誰先被碰到誰就後死。若 pool 先死,
    // DmaMat 解構時呼叫 pool_->free() 就會踩到已銷毀的物件(segfault)。
    // 讓每個 cache 各持一份 shared_ptr,順序就不再重要。
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
        } catch (const std::exception& e) {
            error_ = e.what();
            pool_.reset();
        }
        return pool_;
    }

    const std::string& error() const { return error_; }
    std::mutex& mutex() { return io_mtx_; }

private:
    // heap_ 為 "auto" 時逐一嘗試,直到某個 heap 真的配得出連續記憶體。
    // system heap 一定會在連續性檢查失敗,所以排最後、當作最終備案。
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
                heap_ = h;                 // 記住成功的那個
                return p;
            } catch (const std::exception& e) {
                tried += (tried.empty() ? "" : "; ") + h + ": " + e.what();
            }
        }
        throw std::runtime_error("所有 dma_heap 都失敗 —— " + tried);
    }

public:

private:
    PoolHolder() = default;
    enum class Mode { UdmaBuf, Reserved, DmaHeap, Factory };
    Mode        mode_ = Mode::UdmaBuf;
    std::string name_ = "udmabuf0";
    std::string heap_ = "linux,cma";
    std::function<std::shared_ptr<DmaPool>()> factory_;
    uint64_t    res_phys_ = 0;
    size_t      res_size_ = 0;
    std::string error_;
    bool tried_ = false;
    std::shared_ptr<DmaPool> pool_;
    std::mutex mtx_;      // 保護 pool_ 本身的建立
    std::mutex io_mtx_;   // 保護 pool 的 sync 與配置,跨 IP 共用
};

// 依尺寸快取 DmaMat,避免逐幀重配造成碎片。
// 每顆 IP 各自持有一份,但配出來的記憶體都來自同一個 pool。
class BufferCache {
public:
    // 尺寸不同就重配;相同則直接沿用。
    // 持有一份 pool 的 shared_ptr,確保 pool 不會比快取的 DmaMat 早死。
    // slot 讓同一組尺寸能有多塊獨立的 buffer。
    // 多執行緒 pipeline 需要它:擷取執行緒在寫 slot N+1 的同時,
    // IP 還在讀 slot N。共用一塊的話會互相覆寫。
    DmaMat* get(const std::shared_ptr<DmaPool>& pool, int rows, int cols, int type,
                int slot = 0) {
        if (!pool) return nullptr;
        if (keepalive_ != pool) {      // 換了 pool,舊 buffer 一律作廢
            map_.clear();
            keepalive_ = pool;
        }

        const Key k{rows, cols, type, slot};
        auto it = map_.find(k.hash());
        if (it != map_.end()) return it->second.get();

        try {
            auto m = std::make_unique<DmaMat>(*pool, rows, cols, type);
            if (!m->is_packed()) return nullptr;
            DmaMat* raw = m.get();
            map_[k.hash()] = std::move(m);
            return raw;
        } catch (const std::exception&) {
            return nullptr;
        }
    }

    void clear() { map_.clear(); keepalive_.reset(); }

private:
    struct Key {
        int rows, cols, type, slot;
        uint64_t hash() const {
            return (static_cast<uint64_t>(rows) << 40)
                 ^ (static_cast<uint64_t>(cols) << 16)
                 ^ (static_cast<uint64_t>(type) << 8)
                 ^ static_cast<uint64_t>(slot);
        }
    };
    // *** 宣告順序有意義 ***
    // 成員是反序解構,所以 keepalive_ 必須宣告在 map_ 之前,
    // 它才會比那些 DmaMat 晚死。反過來的話,pool 會先被銷毀,
    // 接著 DmaMat 解構時呼叫 pool_->free() 就踩到已釋放的物件。
    std::shared_ptr<DmaPool> keepalive_;
    std::unordered_map<uint64_t, std::unique_ptr<DmaMat>> map_;
};

}  // namespace detail


// ---- 共用 pool 的對外介面 ----

// 指定 u-dma-buf 的裝置名稱。要在第一次用到任何 IP 之前呼叫。
inline void set_pool(const std::string& name) {
    detail::PoolHolder::get().set_name(name);
}

// pool 開得起來嗎
inline bool pool_available() {
    return static_cast<bool>(detail::PoolHolder::get().try_open());
}

// 取得共用 pool。IP 模組配置 buffer 時應持有這份 shared_ptr。
inline std::shared_ptr<DmaPool> pool_ptr() {
    return detail::PoolHolder::get().try_open();
}

// 取得共用 pool。開不起來會丟例外,不確定時先問 pool_available()
inline DmaPool& pool() {
    auto p = detail::PoolHolder::get().try_open();
    if (!p) throw std::runtime_error("DMA pool 不可用: "
                                     + detail::PoolHolder::get().error());
    return *p;
}

inline std::string pool_error() {
    return detail::PoolHolder::get().error();
}

// 改用 /dev/mem 映射裝置樹裡 no-map 的 reserved-memory。
// u-dma-buf 還沒備妥時的替代方案 —— 映射是 uncached,不需 sync,
// 但 CPU 端的搬移與運算會明顯變慢。
// 改用 DMA-BUF Heaps(/dev/dma_heap/…)。這是 mainline 內建的介面,
// 不需要 u-dma-buf 模組。"linux,cma" heap 保證實體連續;
// 取得實體位址靠 /proc/self/pagemap,需要 root。
// 由外部後端提供 pool(見 hls_xrt.hpp)
inline void set_pool_factory(std::function<std::shared_ptr<DmaPool>()> f) {
    detail::PoolHolder::get().set_factory(std::move(f));
}

inline void use_dma_heap(const std::string& heap = "auto",
                         size_t size = 32u * 1024 * 1024) {
    detail::PoolHolder::get().set_dma_heap(heap, size);
}

inline void use_reserved_memory(uint64_t phys, size_t size) {
    detail::PoolHolder::get().set_reserved(phys, size);
}

// 目前 pool 的說明,用於確認走的是哪一條路
inline std::string pool_info() {
    auto p = detail::PoolHolder::get().try_open();
    if (!p) return "不可用: " + detail::PoolHolder::get().error();
    char b[160];
    std::snprintf(b, sizeof b, "%s  phys=0x%llx  size=%.1f MB  (%s%s)",
                  p->name().c_str(), (unsigned long long)p->phys(),
                  p->size() / 1048576.0,
                  p->is_cached() ? "cached" : "uncached,免 sync 但較慢",
                  p->manual_cache() ? ",自行做 cache 維護" :
                  (p->is_cached() ? ",由驅動 sync" : ""));
    return b;
}

// 釋放已快取的 buffer(切換工作尺寸、或想回收空間時用)
inline void clear_buffers();   // 由各 IP 模組註冊,定義在下方

namespace detail {
// 每個 IP 模組把自己的 cache 清除函式掛進來
inline std::vector<void(*)()>& cleanup_hooks() {
    static std::vector<void(*)()> v;
    return v;
}
struct CleanupRegistrar {
    explicit CleanupRegistrar(void (*fn)()) { cleanup_hooks().push_back(fn); }
};
}  // namespace detail

inline void clear_buffers() {
    for (auto fn : detail::cleanup_hooks()) fn();
}

}  // namespace hls