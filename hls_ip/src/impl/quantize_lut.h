
#pragma once

// #include <ap_int.h>

// ------------ Constants ------------
constexpr double kMean[3] = {0.485, 0.456, 0.406};
constexpr double kStd[3]  = {0.229, 0.224, 0.225};

constexpr int kFixPoint = 5;
constexpr int kClampMin = -128;
constexpr int kClampMax = 127;

// ------------ scale / bias ------------
// Follow norm_fix_letterbox.cpp:
//
// scale = (1 / (std * 255)) * 2^fix_point
// bias  = (-mean / std) * 2^fix_point
//
// The f suffix preserves the float-rounded constants used
// by norm_fix_letterbox.cpp.

constexpr double kU8ScaleR = 1.0 / (kStd[0] * 255.0);
constexpr double kU8ScaleG = 1.0 / (kStd[1] * 255.0);
constexpr double kU8ScaleB = 1.0 / (kStd[2] * 255.0);

constexpr double kU8BiasR = -kMean[0] / kStd[0];
constexpr double kU8BiasG = -kMean[1] / kStd[1];
constexpr double kU8BiasB = -kMean[2] / kStd[2];

constexpr double kScale[3] = {
    kU8ScaleR * (1 << kFixPoint),
    kU8ScaleG * (1 << kFixPoint),
    kU8ScaleB * (1 << kFixPoint)
};

constexpr double kBias[3] = {
    kU8BiasR * (1 << kFixPoint),
    kU8BiasG * (1 << kFixPoint),
    kU8BiasB * (1 << kFixPoint)
};

// ------------ constexpr helpers ------------

constexpr int ce_clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Match double_kernel:
// 1. y = x * scale + bias
// 2. clamp to [-128, 127]
// 3. convert to integer by truncation toward zero
constexpr int quant_one(int x, int ch) {
    const double y = x * kScale[ch] + kBias[ch];

    // Clamp in double precision before integer conversion.
    const double clipped =
        y < static_cast<double>(kClampMin) ? kClampMin :
        y > static_cast<double>(kClampMax) ? kClampMax : y;

    return static_cast<int>(clipped);
}

// Store the signed int8 result as its equivalent 8-bit pattern.
constexpr unsigned char quant_one_u8(int x, int ch) {
    return static_cast<unsigned char>(quant_one(x, ch) & 0xFF);
}

// ------------ Compile-time LUT ------------

struct QuantLut {
    unsigned char r[256];
    unsigned char g[256];
    unsigned char b[256];

    constexpr QuantLut() : r(), g(), b() {
        for (int i = 0; i < 256; ++i) {
            r[i] = quant_one_u8(i, 0);
            g[i] = quant_one_u8(i, 1);
            b[i] = quant_one_u8(i, 2);
        }
    }
};

// Generated at compile time.
static constexpr QuantLut kLut{};
