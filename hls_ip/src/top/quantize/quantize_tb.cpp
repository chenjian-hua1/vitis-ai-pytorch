#include <ap_int.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "../../impl/quantize_lut.h"
#include "../../impl/quantize_impl.h"
#include "quantize_top.h"

static int ref_round(double v) {
    return v >= 0.0 ? static_cast<int>(std::floor(v + 0.5))
                    : -static_cast<int>(std::floor(-v + 0.5));
}

static uint8_t ref_quant(int x, int channel) {
    const double normalized =
        (x / 255.0 - kMean[channel]) / kStd[channel];
    int q = ref_round(normalized * (1 << kFixPoint));
    if (q < kClampMin) q = kClampMin;
    if (q > kClampMax) q = kClampMax;
    return static_cast<uint8_t>(q & 0xff);
}

static bool test_top() {
    constexpr int pixel_count = NUM_PIXEL;
    ap_uint<NUM_PIXEL * 24> input = 0;
    ap_uint<NUM_PIXEL * 24> output = 0;

    // Give every channel and pixel a different value to catch packing,
    // ordering, and final single-pixel handling errors.
    for (int p = 0; p < pixel_count; ++p) {
        const int r = (p * 37 + 11) & 0xff;
        const int g = (p * 73 + 29) & 0xff;
        const int b = (p * 109 + 47) & 0xff;
        const int base = p * 24;
        input.range(base + 23, base + 16) = r;
        input.range(base + 15, base + 8) = g;
        input.range(base + 7, base) = b;
    }

    quantize_top(input, output);

    int errors = 0;
    for (int p = 0; p < pixel_count; ++p) {
        const int r = (p * 37 + 11) & 0xff;
        const int g = (p * 73 + 29) & 0xff;
        const int b = (p * 109 + 47) & 0xff;
        const int base = p * 24;
        const uint8_t got_r = output.range(base + 23, base + 16).to_uint();
        const uint8_t got_g = output.range(base + 15, base + 8).to_uint();
        const uint8_t got_b = output.range(base + 7, base).to_uint();
        const uint8_t exp_r = ref_quant(r, 0);
        const uint8_t exp_g = ref_quant(g, 1);
        const uint8_t exp_b = ref_quant(b, 2);

        if (got_r != exp_r || got_g != exp_g || got_b != exp_b) {
            if (errors < 10) {
                std::printf("[quantize_top] pixel %d: expected (%02X,%02X,%02X), got (%02X,%02X,%02X)\n",
                            p, exp_r, exp_g, exp_b, got_r, got_g, got_b);
            }
            ++errors;
        }
    }

    std::printf("quantize_top (%d pixels): %s\n", pixel_count,
                errors == 0 ? "PASS" : "FAIL");
    return errors == 0;
}

int main() {
    if (test_top()) {
        std::puts("*** ALL TESTS PASSED ***");
        return 0;
    }
    std::puts("*** TEST FAILED ***");
    return 1;
}
