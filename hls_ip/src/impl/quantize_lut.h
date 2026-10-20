#pragma once

#include <ap_int.h>

// ------------ Constants ------------
constexpr double kMean[3] = {0.485, 0.456, 0.406};
constexpr double kStd[3]  = {0.229, 0.224, 0.225};

constexpr int kFixPoint = 6;      // 固定的 fix_point
constexpr int kClampMin = -128;
constexpr int kClampMax = 127;

// ------------ constexpr 輔助函式 ------------
constexpr int ce_floor(double v) {
    return (int)v - ((double)(int)v > v ? 1 : 0);
}

// 四捨五入 (round half away from zero)
constexpr int ce_round(double v) {
    return v >= 0 ? (int)(v + 0.5) : -(int)(-v + 0.5);
}

constexpr int ce_clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

constexpr unsigned char quant_one(int x, int ch) {
    // norm = x * scale + bias = (x/255 - mean) / std
    return (unsigned char)(ce_clamp(
        ce_round(((x / 255.0 - kMean[ch]) / kStd[ch]) * (1 << kFixPoint)),
        kClampMin, kClampMax) & 0xFF);
}

// ------------ 編譯期產生的表 ------------
struct QuantLut {
    unsigned char r[256];
    unsigned char g[256];
    unsigned char b[256];

    constexpr QuantLut() : r(), g(), b() {
        for (int i = 0; i < 256; i++) {
            r[i] = quant_one(i, 0);
            g[i] = quant_one(i, 1);
            b[i] = quant_one(i, 2);
        }
    }
};

static constexpr QuantLut kLut{};   // 編譯期完成