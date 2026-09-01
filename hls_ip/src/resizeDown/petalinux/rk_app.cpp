// rk_app.cpp — 應用層範例
//
// main() 完全不碰硬體:沒有暫存器、沒有實體位址、沒有 DMA buffer。
// 全部藏在 hls::resize:: 後面。IP 不可用時會自動退回 CPU,
// 所以這個程式在沒有 FPGA 的機器上也能正常執行。
//
// 用法:
//   ./rk_app input.jpg output.png [input_size]
//   ./rk_app --selftest                    用內建測試圖驗證硬體
//   ./rk_app --bench input.jpg [次數]      量測 letterbox 吞吐

#include "hls_resize.hpp"

// 選用:用 XRT BO 當 DMA pool(-DHLS_USE_XRT)
#ifdef HLS_USE_XRT
#include "hls_xrt.hpp"
#endif

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void print_status() {
    printf("Pool: %s\n", hls::pool_info().c_str());
    if (hls::resize::available())
        printf("IP  : 可用 -> %s\n", hls::resize::device_info().c_str());
    else
        printf("IP  : 不可用(%s)—— 將全程使用 CPU\n",
               hls::resize::last_error().c_str());
}

void print_result(const hls::resize::Result& r) {
    printf("  加速: %s", r.used_ip ? "IP" : "CPU");
    if (r.used_ip) printf(" %dx -> 中間 %dx%d,耗時 %.3f ms",
                          r.ip_scale, r.mid_w, r.mid_h, r.ip_ms);
    printf("\n  說明: %s\n", r.reason);
    if (r.used_ip) {
        const auto& t = r.timing;
        printf("  明細: 複製 %.3f%s | cache %.3f | IP %.3f | 後處理 %.3f ms\n",
               t.copy_ms, r.zero_copy ? "(zero-copy)" : "",
               t.sync_ms, t.run_ms, t.post_ms);
    }
}

// ---- 一般用法 ----
int cmd_convert(const char* in_path, const char* out_path, int input_size) {
    cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR);
    if (img.empty()) { fprintf(stderr, "讀不到 %s\n", in_path); return 1; }

    printf("輸入 %dx%d -> letterbox %d\n", img.cols, img.rows, input_size);

    hls::resize::Result res;
    hls::resize::letterbox(img, input_size, res);

    print_result(res);
    printf("  content: (%d,%d) %dx%d  ratio: %.4f\n",
           res.content.x, res.content.y, res.content.width, res.content.height,
           res.ratio.x);

    cv::imwrite(out_path, res.img);
    printf("已輸出 %s\n", out_path);
    return 0;
}

// ---- 硬體驗證:用內建 pattern 比對軟體模型 ----
int cmd_selftest() {
    if (!hls::resize::available()) {
        fprintf(stderr, "IP 不可用: %s\n", hls::resize::last_error().c_str());
        return 1;
    }

    struct { int w, h, s; } cases[] = {
        {  64,  64, 2}, {  96,  96, 3},
        { 640, 480, 2}, { 960, 720, 3},
        {1920,1080, 2}, {1920,1080, 3},
    };

    int failed = 0;
    for (const auto& c : cases) {
        cv::Mat img = hls::resize::make_test_pattern(c.w, c.h);
        hls::resize::VerifyReport rep = hls::resize::verify(img, c.s);

        printf("%4dx%-4d /%d : ", c.w, c.h, c.s);
        if (!rep.ran) {
            printf("略過(%s)\n", rep.reason);
            continue;
        }
        printf("%s  不符 %llu/%llu,最大差 %d,%.3f ms\n",
               rep.passed() ? "[通過]" : "[失敗]",
               (unsigned long long)rep.mismatches,
               (unsigned long long)rep.total, rep.max_diff, rep.ms);
        if (!rep.passed()) ++failed;
    }

    printf("\n%s\n", failed ? "有案例未通過,檢查參數或 bitstream" : "全部通過");
    return failed ? 1 : 0;
}

// ---- 吞吐量測 ----
void bench_one(const char* label, const cv::Mat& img, int iters, int input_size) {
    hls::resize::Result res;
    hls::resize::letterbox(img, input_size, res);   // 暖機:讓 buffer 配好

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        hls::resize::letterbox(img, input_size, res);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();

    printf("\n[%s]\n", label);
    print_result(res);
    printf("  %d 次:平均 %.3f ms,%.1f FPS\n",
           iters, ms / iters, 1000.0 * iters / ms);
}

int cmd_bench(const char* in_path, int iters, int input_size) {
    cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR);
    if (img.empty()) { fprintf(stderr, "讀不到 %s\n", in_path); return 1; }

    printf("%dx%d -> letterbox %d,各 %d 次\n", img.cols, img.rows, input_size, iters);

    // 一般路徑:影像在一般 heap,每次都要複製進 DMA 記憶體
    bench_one("一般:每次複製", img, iters, input_size);

    // zero-copy:影像本來就放在 DMA 記憶體
    cv::Mat dma_in = hls::resize::input_buffer(img.cols, img.rows);
    if (!dma_in.empty()) {
        img.copyTo(dma_in);                 // 只做一次
        bench_one("zero-copy:影像已在 DMA 記憶體", dma_in, iters, input_size);
    } else {
        printf("\n(取不到 DMA 輸入緩衝,略過 zero-copy 對照)\n");
    }
    return 0;
}

}  // namespace

void usage(const char* prog) {
    printf("用法:\n"
           "  %s [選項] <輸入圖> [輸出圖] [input_size]\n"
           "  %s [選項] --selftest\n"
           "  %s [選項] --bench <輸入圖> [次數] [input_size]\n"
           "\n選項:\n"
           "  --devmem                UIO 找不到時改用 /dev/mem 存取暫存器\n"
           "  --heap <MB>             改用 DMA-BUF Heaps(免安裝模組,需 root)\n"
           "  --heap <名稱>:<MB>      指定 heap;不指定則自動挑一個\n"
           "  --xrt <MB>              改用 XRT BO 當 pool(需 -DHLS_USE_XRT)\n"
           "  --reserved <phys> <MB>  改用 /dev/mem 映射保留記憶體當 DMA pool\n"
           "  --pool <name>           指定 u-dma-buf 名稱(預設 udmabuf0)\n"
           "  --uio <name>            指定 UIO 名稱(預設 resize_kernel_0)\n",
           prog, prog, prog);
}

int main(int argc, char** argv) {
    // ---- 先抽掉選項,剩下的才是位置參數 ----
    std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!strcmp(a, "--devmem")) {
            hls::resize::use_devmem();
        } else if (!strcmp(a, "--pool") && i + 1 < argc) {
            hls::set_pool(argv[++i]);
        } else if (!strcmp(a, "--uio") && i + 1 < argc) {
            hls::resize::configure(argv[++i]);
        } else if (!strcmp(a, "--xrt") && i + 1 < argc) {
#ifdef HLS_USE_XRT
            hls::xrt_backend::use(static_cast<size_t>(atoi(argv[++i])) * 1024 * 1024);
#else
            ++i;
            fprintf(stderr, "此版本未啟用 XRT,請用 make rk_app_xrt 重建\n");
            return 1;
#endif
        } else if (!strcmp(a, "--heap") && i + 1 < argc) {
            // --heap <MB> 或 --heap <名稱>:<MB>
            const char* v = argv[++i];
            const char* colon = strchr(v, ':');
            if (colon) hls::use_dma_heap(std::string(v, colon),
                                         static_cast<size_t>(atoi(colon + 1)) * 1024 * 1024);
            else       hls::use_dma_heap("auto",
                                         static_cast<size_t>(atoi(v)) * 1024 * 1024);
        } else if (!strcmp(a, "--reserved") && i + 2 < argc) {
            const uint64_t phys = strtoull(argv[++i], nullptr, 0);
            const size_t   size = static_cast<size_t>(atoi(argv[++i])) * 1024 * 1024;
            hls::use_reserved_memory(phys, size);
        } else if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            pos.push_back(a);
        }
    }

    print_status();

    if (!pos.empty() && !strcmp(pos[0], "--selftest"))
        return cmd_selftest();

    if (pos.size() >= 2 && !strcmp(pos[0], "--bench"))
        return cmd_bench(pos[1],
                         pos.size() > 2 ? atoi(pos[2]) : 100,
                         pos.size() > 3 ? atoi(pos[3]) : 640);

    if (pos.empty()) { usage(argv[0]); return 1; }

    const char* in_path  = pos[0];
    const char* out_path = pos.size() > 1 ? pos[1] : "output.png";
    const int   size     = pos.size() > 2 ? atoi(pos[2]) : 640;
    return cmd_convert(in_path, out_path, size);
}