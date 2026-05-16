// bench_layout_arm.cpp
//
// Layout 轉換效能 benchmark：三種策略對比
//
//   Path A：DPU 內做 NHWC→NCHW int8 轉置 + PostProcessor NCHW memcpy concat
//           (目前生產用的設計，naive triple-nested loop)
//
//   Path B：DPU 不轉置直接給 NHWC + PostProcessor NHWC→NCHW float scatter concat
//           (省掉 int8 轉置，但 concat 變慢)
//
//   Path C：DPU 用 block-tiled int8 轉置 + PostProcessor NCHW memcpy concat
//           (A 的優化版，看看 tiling 能不能壓 int8 轉置時間)
//
// 編譯：
//   make
//
// 執行（預設 80×80, 40×40, 20×20, C=144）：
//   ./bench_layout_arm
//
// 自訂尺寸（例如 96×96 + 48×48 + 24×24, C=85）：
//   ./bench_layout_arm --scales=96,48,24 --channels=85
//
// 增加 iteration（雜訊大時可加）：
//   ./bench_layout_arm --iter=2000 --trials=30
//

#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <time.h>

// ============================================================================
//  CPU 資訊（讓你看清楚跑在哪台機器上）
// ============================================================================
static void print_cpu_info() {
    std::printf("===== Host CPU Info =====\n");

#if defined(__aarch64__)
    std::printf("Architecture: aarch64 (ARM 64-bit)\n");
#elif defined(__arm__)
    std::printf("Architecture: armv7 (ARM 32-bit)\n");
#elif defined(__x86_64__)
    std::printf("Architecture: x86_64\n");
#else
    std::printf("Architecture: unknown\n");
#endif

    std::ifstream f("/proc/cpuinfo");
    std::string line;
    int printed = 0;
    while (std::getline(f, line) && printed < 20) {
        if (line.find("model name")   != std::string::npos ||
            line.find("Hardware")     != std::string::npos ||
            line.find("Processor")    != std::string::npos ||
            line.find("CPU implementer") != std::string::npos ||
            line.find("CPU part")     != std::string::npos ||
            line.find("CPU MHz")      != std::string::npos ||
            line.find("Features")     != std::string::npos) {
            std::printf("  %s\n", line.c_str());
            ++printed;
            if (printed > 6) break;
        }
    }
    std::printf("\n");
}

// ============================================================================
//  高精度計時：clock_gettime(CLOCK_MONOTONIC) 直接給 ns
// ============================================================================
static inline double ns_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// ============================================================================
//  Path A: naive NHWC→NCHW int8 轉置
// ============================================================================
static void nhwc_to_nchw_int8_naive(const int8_t* src, int8_t* dst,
                                    int H, int W, int C) {
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            for (int c = 0; c < C; ++c)
                dst[c * H * W + h * W + w] = src[h * W * C + w * C + c];
}

// ============================================================================
//  Path C: block-tiled NHWC→NCHW int8 轉置
//
//  關鍵想法：以 (HW_TILE) × (C_TILE) 為單位處理。
//  每個 tile 內：
//    - src 是 HW_TILE 列 (跨度 C，連續 C_TILE 個 byte)
//    - dst 是 C_TILE 列 (跨度 H*W，連續 HW_TILE 個 byte)
//  C_TILE 選 32 或 64 byte (cache line size) 讓 dst 寫入是整條 cache line
//  HW_TILE 選 8 或 16 讓 src 跨度小、有 prefetch 命中機會
// ============================================================================
template <int HW_TILE, int C_TILE>
static void nhwc_to_nchw_int8_tiled(const int8_t* src, int8_t* dst,
                                    int H, int W, int C) {
    const int HW = H * W;
    int hw = 0;
    for (; hw + HW_TILE <= HW; hw += HW_TILE) {
        int c = 0;
        for (; c + C_TILE <= C; c += C_TILE) {
            // 處理一個 HW_TILE × C_TILE 的方塊
            for (int dc = 0; dc < C_TILE; ++dc) {
                int8_t* d = dst + (c + dc) * HW + hw;
                const int8_t* s = src + hw * C + c + dc;
                // 內層 HW_TILE 步：dst 連續寫、src 跨 C 跳
                for (int dhw = 0; dhw < HW_TILE; ++dhw) {
                    d[dhw] = s[dhw * C];
                }
            }
        }
        // 處理 C 方向的尾巴 (C % C_TILE)
        for (; c < C; ++c) {
            int8_t* d = dst + c * HW + hw;
            const int8_t* s = src + hw * C + c;
            for (int dhw = 0; dhw < HW_TILE; ++dhw) {
                d[dhw] = s[dhw * C];
            }
        }
    }
    // 處理 HW 方向的尾巴 (HW % HW_TILE)
    for (; hw < HW; ++hw) {
        for (int c = 0; c < C; ++c) {
            dst[c * HW + hw] = src[hw * C + c];
        }
    }
}

// ============================================================================
//  Concat 邏輯
// ============================================================================
static void nchw_concat(const float* src, float* dst_xcat, int C, int H, int W,
                        int A_total, int col_offset) {
    const int hw = H * W;
    for (int c = 0; c < C; ++c)
        std::memcpy(dst_xcat + c * A_total + col_offset,
                    src + c * hw,
                    hw * sizeof(float));
}

static void nhwc_concat_scatter(const float* src, float* dst_xcat,
                                int C, int H, int W,
                                int A_total, int col_offset) {
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w) {
            const float* s_anchor = src + (h * W + w) * C;
            const int idx = h * W + w;
            for (int c = 0; c < C; ++c)
                dst_xcat[c * A_total + col_offset + idx] = s_anchor[c];
        }
}

static void fix2float_inplace(const int8_t* src, float* dst, size_t n, float scale) {
    for (size_t i = 0; i < n; ++i)
        dst[i] = src[i] * scale;
}

// ============================================================================
//  統計工具
// ============================================================================
struct Stats {
    double mean, stddev, min_, max_;
};

static Stats summarize(const std::vector<double>& v) {
    double sum = 0;
    for (double x : v) sum += x;
    double mean = sum / v.size();
    double sq = 0;
    for (double x : v) sq += (x - mean) * (x - mean);
    return {mean, std::sqrt(sq / v.size()),
            *std::min_element(v.begin(), v.end()),
            *std::max_element(v.begin(), v.end())};
}

static void print_stat(const char* name, const Stats& s) {
    std::printf("  %-30s  mean=%.4f ± %.4f ms  (min %.4f, max %.4f)\n",
                name, s.mean, s.stddev, s.min_, s.max_);
}

// ============================================================================
//  Arg parsing
// ============================================================================
struct Scale { int H, W; };

struct Args {
    std::vector<Scale> scales = {{80,80},{40,40},{20,20}};
    int C = 144;
    int trials = 20;
    int iter = 500;
};

static bool parse_args(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--scales=", 0) == 0) {
            out.scales.clear();
            std::string v = a.substr(9);
            std::stringstream ss(v);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                int s = std::stoi(tok);
                out.scales.push_back({s, s});
            }
        } else if (a.rfind("--channels=", 0) == 0) {
            out.C = std::stoi(a.substr(11));
        } else if (a.rfind("--trials=", 0) == 0) {
            out.trials = std::stoi(a.substr(9));
        } else if (a.rfind("--iter=", 0) == 0) {
            out.iter = std::stoi(a.substr(7));
        } else if (a == "-h" || a == "--help") {
            std::printf("Usage: %s [--scales=H1,H2,...] [--channels=C] [--trials=N] [--iter=M]\n",
                        argv[0]);
            return false;
        }
    }
    return true;
}

// ============================================================================

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 0;

    print_cpu_info();

    const auto& scales = args.scales;
    const int C = args.C;
    const float SCALE = 1.0f / 64.0f;

    int A_total = 0;
    std::vector<int> col_offsets;
    for (auto& s : scales) {
        col_offsets.push_back(A_total);
        A_total += s.H * s.W;
    }

    std::printf("===== Bench config =====\n");
    std::printf("Scales:");
    for (auto& s : scales) std::printf(" %dx%d", s.H, s.W);
    std::printf("\nChannels: %d\n", C);
    std::printf("A_total : %d anchors\n", A_total);
    std::printf("Trials  : %d × %d iter\n\n", args.trials, args.iter);

    // ── 配置 buffer ──
    std::vector<std::vector<int8_t>> nhwc_int8(scales.size());
    std::vector<std::vector<float>>  nhwc_float(scales.size());
    std::vector<std::vector<int8_t>> nchw_int8(scales.size());
    std::vector<std::vector<float>>  nchw_float(scales.size());
    for (size_t k = 0; k < scales.size(); ++k) {
        size_t n = static_cast<size_t>(scales[k].H) * scales[k].W * C;
        nhwc_int8[k].resize(n);
        nhwc_float[k].resize(n);
        nchw_int8[k].resize(n);
        nchw_float[k].resize(n);
    }
    std::vector<float> x_cat_A(static_cast<size_t>(C) * A_total);
    std::vector<float> x_cat_B(static_cast<size_t>(C) * A_total);
    std::vector<float> x_cat_C(static_cast<size_t>(C) * A_total);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (auto& v : nhwc_int8)
        for (auto& x : v) x = static_cast<int8_t>(dist(rng));

    // ── 大 warmup（避免冷 cache 干擾結果）──
    for (int i = 0; i < 50; ++i) {
        for (size_t k = 0; k < scales.size(); ++k) {
            nhwc_to_nchw_int8_naive(nhwc_int8[k].data(), nchw_int8[k].data(),
                                    scales[k].H, scales[k].W, C);
            fix2float_inplace(nhwc_int8[k].data(), nhwc_float[k].data(),
                              nhwc_int8[k].size(), SCALE);
        }
    }

    auto bench = [&](auto&& fn) {
        std::vector<double> times;
        times.reserve(args.trials);
        for (int t = 0; t < args.trials; ++t) {
            double t0 = ns_now();
            for (int i = 0; i < args.iter; ++i) fn();
            times.push_back((ns_now() - t0) * 1e-6 / args.iter);
        }
        return summarize(times);
    };

    // ──────────── Path A ────────────
    auto s_A_transpose = bench([&]{
        for (size_t k = 0; k < scales.size(); ++k)
            nhwc_to_nchw_int8_naive(nhwc_int8[k].data(), nchw_int8[k].data(),
                                    scales[k].H, scales[k].W, C);
    });
    auto s_A_dequant = bench([&]{
        for (size_t k = 0; k < scales.size(); ++k)
            fix2float_inplace(nchw_int8[k].data(), nchw_float[k].data(),
                              nchw_int8[k].size(), SCALE);
    });
    auto s_A_concat = bench([&]{
        for (size_t k = 0; k < scales.size(); ++k)
            nchw_concat(nchw_float[k].data(), x_cat_A.data(),
                        C, scales[k].H, scales[k].W, A_total, col_offsets[k]);
    });

    // ──────────── Path B ────────────
    auto s_B_dequant = bench([&]{
        for (size_t k = 0; k < scales.size(); ++k)
            fix2float_inplace(nhwc_int8[k].data(), nhwc_float[k].data(),
                              nhwc_int8[k].size(), SCALE);
    });
    auto s_B_concat = bench([&]{
        for (size_t k = 0; k < scales.size(); ++k)
            nhwc_concat_scatter(nhwc_float[k].data(), x_cat_B.data(),
                                C, scales[k].H, scales[k].W,
                                A_total, col_offsets[k]);
    });

    // ──────────── Path C：block-tiled 轉置 ────────────
    auto s_C_transpose = bench([&]{
        for (size_t k = 0; k < scales.size(); ++k)
            nhwc_to_nchw_int8_tiled<16, 32>(nhwc_int8[k].data(),
                                            nchw_int8[k].data(),
                                            scales[k].H, scales[k].W, C);
    });
    // C 的 dequant 和 concat 跟 A 一樣，直接重用 A 的數字

    // 跑一輪 C 路徑把 x_cat_C 算出來做正確性檢查
    for (size_t k = 0; k < scales.size(); ++k) {
        nhwc_to_nchw_int8_tiled<16, 32>(nhwc_int8[k].data(), nchw_int8[k].data(),
                                        scales[k].H, scales[k].W, C);
        fix2float_inplace(nchw_int8[k].data(), nchw_float[k].data(),
                          nchw_int8[k].size(), SCALE);
        nchw_concat(nchw_float[k].data(), x_cat_C.data(),
                    C, scales[k].H, scales[k].W, A_total, col_offsets[k]);
    }

    bool ok_AB = (x_cat_A == x_cat_B);
    bool ok_AC = (x_cat_A == x_cat_C);

    // ── 輸出 ──
    std::printf("===== Path A: DPU naive 轉置 + NCHW memcpy concat =====\n");
    print_stat("Xmodel transpose (naive)",  s_A_transpose);
    print_stat("fix2float (NCHW)",           s_A_dequant);
    print_stat("Concat (NCHW memcpy)",       s_A_concat);
    double A_mean = s_A_transpose.mean + s_A_dequant.mean + s_A_concat.mean;
    double A_std  = std::sqrt(
        s_A_transpose.stddev*s_A_transpose.stddev +
        s_A_dequant.stddev*s_A_dequant.stddev +
        s_A_concat.stddev*s_A_concat.stddev);
    std::printf("  %-30s  mean=%.4f ± %.4f ms\n\n", "TOTAL A", A_mean, A_std);

    std::printf("===== Path B: 不轉置 + NHWC scatter concat =====\n");
    print_stat("(transpose skipped)", {0,0,0,0});
    print_stat("fix2float (NHWC)",     s_B_dequant);
    print_stat("Concat (NHWC scatter)", s_B_concat);
    double B_mean = s_B_dequant.mean + s_B_concat.mean;
    double B_std  = std::sqrt(
        s_B_dequant.stddev*s_B_dequant.stddev +
        s_B_concat.stddev*s_B_concat.stddev);
    std::printf("  %-30s  mean=%.4f ± %.4f ms\n\n", "TOTAL B", B_mean, B_std);

    std::printf("===== Path C: DPU tiled 轉置 (16×32) + NCHW memcpy concat =====\n");
    print_stat("Xmodel transpose (tiled)",  s_C_transpose);
    print_stat("fix2float (NCHW)",           s_A_dequant);
    print_stat("Concat (NCHW memcpy)",       s_A_concat);
    double C_mean = s_C_transpose.mean + s_A_dequant.mean + s_A_concat.mean;
    double C_std  = std::sqrt(
        s_C_transpose.stddev*s_C_transpose.stddev +
        s_A_dequant.stddev*s_A_dequant.stddev +
        s_A_concat.stddev*s_A_concat.stddev);
    std::printf("  %-30s  mean=%.4f ± %.4f ms\n\n", "TOTAL C", C_mean, C_std);

    // ── 比較表 ──
    std::printf("===== 比較 =====\n");
    auto report = [&](const char* name, double mean, double std_,
                      double base_mean, double base_std) {
        double diff = mean - base_mean;
        double diff_std = std::sqrt(std_*std_ + base_std*base_std);
        std::printf("  %s: %.4f ± %.4f ms  (vs A: %+.4f ms, %+.1f%%)",
                    name, mean, std_, diff, 100.0 * diff / base_mean);
        if (std::abs(diff) < 2 * diff_std) std::printf("  [n.s.]");
        std::printf("\n");
    };
    report("A (baseline)", A_mean, A_std, A_mean, A_std);
    report("B (scatter)",  B_mean, B_std, A_mean, A_std);
    report("C (tiled)",    C_mean, C_std, A_mean, A_std);
    std::printf("[n.s.] = 差異未達 2σ，統計上無顯著差異\n");

    std::printf("\n正確性: A==B: %s   A==C: %s\n",
                ok_AB ? "PASS" : "FAIL",
                ok_AC ? "PASS" : "FAIL");

    return 0;
}