// resize_kernel.hpp
// PetaLinux 上以 UIO + mmap 控制 resize_kernel_0 (HLS ap_ctrl_hs IP)
// 硬體資訊來自 resize_dpu.xsa:
//   control base 0xB000_0000, size 0x10000, s_axi_control (32-bit)
//   m_axi_gmem0 -> S_AXI_HP2_FPD, m_axi_gmem1 -> S_AXI_HP3_FPD (非 coherent)
//   interrupt   -> xlconcat In0 -> pl_ps_irq0[0]
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <string>
#include <stdexcept>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/select.h>

// ---- 暫存器位移(來自 xresize_kernel_hw.h)----
namespace rk_reg {
constexpr uint32_t AP_CTRL       = 0x00;  // b0 start, b1 done, b2 idle, b3 ready, b7 auto_restart
constexpr uint32_t GIE           = 0x04;
constexpr uint32_t IER           = 0x08;
constexpr uint32_t ISR           = 0x0c;
constexpr uint32_t IN_PTR        = 0x10;  // 64-bit: 0x10 / 0x14
constexpr uint32_t OUT_PTR       = 0x1c;  // 64-bit: 0x1c / 0x20
constexpr uint32_t TOTAL_WORDS   = 0x28;
constexpr uint32_t TOTAL_RESULTS = 0x30;
constexpr uint32_t OUT_WORDS     = 0x38;
constexpr uint32_t OUT_W         = 0x40;
constexpr uint32_t SCALE_MODE    = 0x48;  // 1 bit
constexpr uint32_t INV_SCALE     = 0x50;  // 16 bit
}

class ResizeKernel {
public:
    // 尋找順序:
    //   1) /sys/class/uio/uioX/name 等於 name
    //   2) uioX 的 map0 位址等於 ctrl_phys(名字對不上時的可靠備案)
    //   3) /dev/mem 直接映射 ctrl_phys(僅在 allow_devmem 時)
    //
    // 走到第 3 步就沒有中斷,wait_done_irq 會自動改成輪詢。那是裝置樹
    // 還沒設好時的暫時手段:需要 root、空轉燒 CPU、無存取保護。
    explicit ResizeKernel(const std::string& name = "resize_kernel_0",
                          uint64_t ctrl_phys = 0,
                          bool allow_devmem = false) {
        // 1) 先用名字找
        int uio_num = find_uio(name);

        // 2) 名字對不上就用控制暫存器的實體位址找 —— 更可靠,
        //    因為裝置樹節點的命名方式常與預期不同
        if (uio_num < 0 && ctrl_phys != 0) {
            uio_num = find_uio_by_addr(ctrl_phys);
            if (uio_num >= 0) matched_by_addr_ = true;
        }

        if (uio_num >= 0) {
            actual_name_ = uio_name(uio_num);
            uio_index_   = uio_num;
            map_size_ = read_hex(("/sys/class/uio/uio" + std::to_string(uio_num) +
                                  "/maps/map0/size").c_str());
            if (map_size_ == 0) map_size_ = 0x10000;

            const std::string dev = "/dev/uio" + std::to_string(uio_num);
            fd_ = ::open(dev.c_str(), O_RDWR);
            if (fd_ < 0)
                throw std::runtime_error("open " + dev + " 失敗: " + strerror(errno));

            void* p = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE,
                             MAP_SHARED, fd_, 0);
            if (p == MAP_FAILED) {
                ::close(fd_);
                throw std::runtime_error("mmap 控制暫存器失敗");
            }
            base_ = static_cast<volatile uint32_t*>(p);
            has_irq_ = true;
            return;
        }

        if (!allow_devmem || ctrl_phys == 0)
            throw std::runtime_error(
                "找不到 UIO 裝置: 名稱 " + name +
                " 與位址 0x" + to_hex(ctrl_phys) + " 都沒有相符的 /dev/uioX"
                " (檢查 device tree 的 compatible,以及 bootargs 是否有 "
                "uio_pdrv_genirq.of_id=generic-uio)");

        // ---- /dev/mem 後路 ----
        map_size_ = 0x10000;
        fd_ = ::open("/dev/mem", O_RDWR | O_SYNC);
        if (fd_ < 0)
            throw std::runtime_error(std::string("open /dev/mem 失敗: ") + strerror(errno));

        void* p = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd_, static_cast<off_t>(ctrl_phys));
        if (p == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("mmap /dev/mem 失敗(位址對嗎?需要 root)");
        }
        base_ = static_cast<volatile uint32_t*>(p);
        has_irq_ = false;
    }

    ~ResizeKernel() {
        if (base_) ::munmap((void*)base_, map_size_);
        if (fd_ >= 0) ::close(fd_);
    }

    // 目前是走 UIO(有中斷)還是 /dev/mem(只能輪詢)
    bool has_irq() const { return has_irq_; }

    // 實際綁到的 /sys/class/uio/uioX/name(走 /dev/mem 時為空)
    const std::string& actual_name() const { return actual_name_; }
    int  uio_index()      const { return uio_index_; }
    // true 表示名字對不上、是靠位址找到的 —— 建議把名稱改對
    bool matched_by_addr() const { return matched_by_addr_; }

    ResizeKernel(const ResizeKernel&) = delete;
    ResizeKernel& operator=(const ResizeKernel&) = delete;

    // ---- 基本讀寫 ----
    void wr(uint32_t off, uint32_t v) { base_[off / 4] = v; }
    uint32_t rd(uint32_t off) const   { return base_[off / 4]; }
    void wr64(uint32_t off, uint64_t v) {
        wr(off,     static_cast<uint32_t>(v & 0xFFFFFFFFu));
        wr(off + 4, static_cast<uint32_t>(v >> 32));
    }

    // ---- 參數設定(in_ptr/out_ptr 必須是「實體位址」)----
    void set_in_ptr(uint64_t phys)      { wr64(rk_reg::IN_PTR,  phys); }
    void set_out_ptr(uint64_t phys)     { wr64(rk_reg::OUT_PTR, phys); }
    void set_total_words(uint32_t v)    { wr(rk_reg::TOTAL_WORDS,   v); }
    void set_total_results(uint32_t v)  { wr(rk_reg::TOTAL_RESULTS, v); }
    void set_out_words(uint32_t v)      { wr(rk_reg::OUT_WORDS,     v); }
    void set_out_w(uint32_t v)          { wr(rk_reg::OUT_W,         v); }
    void set_scale_mode(uint32_t v)     { wr(rk_reg::SCALE_MODE,    v & 1); }
    void set_inv_scale(uint32_t v)      { wr(rk_reg::INV_SCALE,     v & 0xFFFF); }

    bool is_idle()  const { return (rd(rk_reg::AP_CTRL) >> 2) & 1; }
    bool is_ready() const { return (rd(rk_reg::AP_CTRL) >> 3) & 1; }
    bool is_done()  const { return (rd(rk_reg::AP_CTRL) >> 1) & 1; }

    // ap_ctrl_hs:寫 1 到 bit0,硬體會自動清掉
    void start() { uint32_t c = rd(rk_reg::AP_CTRL) & 0x80; wr(rk_reg::AP_CTRL, c | 0x1); }

    // ---- 方式 A:輪詢等待(最省事,適合短工作)----
    bool wait_done_poll(int timeout_ms = 2000) {
        for (int i = 0; i < timeout_ms * 100; ++i) {
            if (is_done() || is_idle()) return true;
            ::usleep(10);
        }
        return false;
    }

    // ---- 方式 B:中斷等待(長工作不燒 CPU)----
    void irq_enable() {
        wr(rk_reg::GIE, 1);   // Global Interrupt Enable
        wr(rk_reg::IER, 1);   // 只開 ap_done
    }
    bool wait_done_irq(int timeout_ms = 2000) {
        if (!has_irq_) return wait_done_poll(timeout_ms);   // /dev/mem 沒有中斷

        uint32_t unmask = 1;
        if (::write(fd_, &unmask, sizeof(unmask)) != (ssize_t)sizeof(unmask)) return false;

        fd_set rs; FD_ZERO(&rs); FD_SET(fd_, &rs);
        timeval tv{ timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
        int r = ::select(fd_ + 1, &rs, nullptr, nullptr, &tv);
        if (r <= 0) return false;

        uint32_t cnt = 0;
        if (::read(fd_, &cnt, sizeof(cnt)) != (ssize_t)sizeof(cnt)) return false;

        wr(rk_reg::ISR, rd(rk_reg::ISR));  // TOW:寫回同值清除
        return true;
    }

private:
    static uint32_t read_hex(const char* path) {
        FILE* f = ::fopen(path, "r");
        if (!f) return 0;
        unsigned v = 0;
        if (::fscanf(f, "0x%x", &v) != 1) v = 0;
        ::fclose(f);
        return v;
    }

    // 依 map0 的實體位址尋找 UIO。位址是唯一不會變的識別 ——
    // 名字取決於裝置樹節點怎麼命名(uio_pdrv_genirq 用 %pOFn,
    // 也就是節點名去掉 unit address,通常不等於 label)。
    static int find_uio_by_addr(uint64_t phys) {
        DIR* d = ::opendir("/sys/class/uio");
        if (!d) return -1;
        int found = -1;
        while (dirent* e = ::readdir(d)) {
            if (::strncmp(e->d_name, "uio", 3) != 0) continue;
            const std::string p = std::string("/sys/class/uio/") + e->d_name
                                + "/maps/map0/addr";
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

    static std::string uio_name(int num) {
        const std::string p = "/sys/class/uio/uio" + std::to_string(num) + "/name";
        FILE* f = ::fopen(p.c_str(), "r");
        if (!f) return {};
        char buf[128] = {0};
        if (!::fgets(buf, sizeof buf, f)) { ::fclose(f); return {}; }
        ::fclose(f);
        if (char* nl = ::strchr(buf, '\n')) *nl = 0;
        return buf;
    }

    static int find_uio(const std::string& want) {
        DIR* d = ::opendir("/sys/class/uio");
        if (!d) return -1;
        int found = -1;
        while (dirent* e = ::readdir(d)) {
            if (::strncmp(e->d_name, "uio", 3) != 0) continue;
            std::string p = std::string("/sys/class/uio/") + e->d_name + "/name";
            FILE* f = ::fopen(p.c_str(), "r");
            if (!f) continue;
            char buf[128] = {0};
            if (::fgets(buf, sizeof(buf), f)) {
                char* nl = ::strchr(buf, '\n'); if (nl) *nl = 0;
                if (want == buf) found = ::atoi(e->d_name + 3);
            }
            ::fclose(f);
            if (found >= 0) break;
        }
        ::closedir(d);
        return found;
    }

    int fd_ = -1;
    volatile uint32_t* base_ = nullptr;
    size_t map_size_ = 0;
    bool        has_irq_ = false;
    bool        matched_by_addr_ = false;
    int         uio_index_ = -1;
    std::string actual_name_;

    static std::string to_hex(uint64_t v) {
        char b[32]; std::snprintf(b, sizeof b, "%llx", (unsigned long long)v);
        return b;
    }
};