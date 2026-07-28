#!/bin/bash
set -e

# ===== input =====
SRC_FILE="$1"
EXTRA_PATH="$2"

# ===== output =====
OUT_DIR="$(dirname "$SRC_FILE")/build"
OUT_FILE="$OUT_DIR/$(basename "${SRC_FILE%.*}")"
mkdir -p "$OUT_DIR"

# ===== detect architecture =====
ARCH=$(uname -m)
OS=$(uname -s)
echo "🖥️  Detected architecture: $ARCH  OS: $OS"

# ===== choose optimization flags =====
#
# 通用的 SIMD 友善 flag（所有平台都開）：
#   -fopenmp-simd    解鎖 #pragma omp simd，不拉 OpenMP runtime
#   -ffast-math      允許 expf / 浮點重排向量化
#   -funroll-loops   小迴圈展開，配合 SIMD 效果好
#
# 驗證向量化成功可加 -fopt-info-vec-optimized 觀察輸出。
#
COMMON_SIMD_FLAGS="-fopenmp-simd -funroll-loops"

if [[ "$ARCH" == "x86_64" ]]; then
    echo "⚙️  Target: PC (x86_64)"
    # -march=native 自動啟用 SSE/AVX/AVX2/FMA（依 CPU 而定）
    OPT_FLAGS="-O3 -DNDEBUG -march=native -flto -ffast-math $COMMON_SIMD_FLAGS"
    INCLUDE_FLAGS="-I."
    EXTRA_LDFLAGS=""

elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    if [[ "$OS" == "Darwin" ]]; then
        echo "⚙️  Target: Apple Silicon (arm64)"
        # Apple M1/M2 的 NEON 是 ARMv8.4-A，不需要額外 +simd 修飾
        OPT_FLAGS="-O3 -DNDEBUG -mcpu=apple-m1 -ffast-math $COMMON_SIMD_FLAGS"
        # Homebrew 在 Apple Silicon 裝在 /opt/homebrew
        HOMEBREW_PREFIX="/opt/homebrew"
        export PKG_CONFIG_PATH="$HOMEBREW_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
        INCLUDE_FLAGS="-I. -I$HOMEBREW_PREFIX/include"
        EXTRA_LDFLAGS="-L$HOMEBREW_PREFIX/lib"
    else
        echo "⚙️  Target: ARM (aarch64, Cortex-A53 / KV260)"
        # Cortex-A53 是 ARMv8-A + NEON + crypto + crc
        # +simd 明確啟用 NEON（雖然 A53 預設就有，寫出來更安全）
        # -mtune=cortex-a53 針對 A53 的 pipeline 排程（in-order 雙發射）
        OPT_FLAGS="-O3 -DNDEBUG \
                   -march=armv8-a+simd+crc \
                   -mcpu=cortex-a53 -mtune=cortex-a53 \
                   -ffast-math $COMMON_SIMD_FLAGS"
        INCLUDE_FLAGS="-I."
        EXTRA_LDFLAGS=""
    fi
else
    echo "⚠️  Unknown architecture, fallback generic"
    OPT_FLAGS="-O3 -DNDEBUG $COMMON_SIMD_FLAGS"
    INCLUDE_FLAGS="-I."
    EXTRA_LDFLAGS=""
fi

# ===== OpenCV flags =====
CXXFLAGS="$(pkg-config --cflags opencv4)"
# LIBS="$(pkg-config --libs opencv4)"
LIBS="$(pkg-config --libs opencv4)"

# ===== include paths =====
if [ -n "$EXTRA_PATH" ]; then
    INCLUDE_FLAGS="$INCLUDE_FLAGS -I$EXTRA_PATH"
fi

# ===== collect sources =====
SRC_FILES="$SRC_FILE"
if [ -n "$EXTRA_PATH" ]; then
    EXTRA_FILES=$(find "$EXTRA_PATH" -name "*.cpp")
    SRC_FILES="$SRC_FILES $EXTRA_FILES"
fi

echo "📦 Source files: $SRC_FILES"
echo "🔍 Include paths: $INCLUDE_FLAGS $CXXFLAGS"
echo "⚡ Optimization: $OPT_FLAGS"

# ===== verbose vectorization report (optional) =====
# 加上 VERBOSE=1 環境變數會印出哪些迴圈成功 / 失敗向量化
VEC_REPORT=""
if [ "$VERBOSE" = "1" ]; then
    echo "📊 Vectorization report enabled"
    VEC_REPORT="-fopt-info-vec-optimized=vec_report.txt -fopt-info-vec-missed=vec_missed.txt"
fi

# ===== compile =====
/usr/bin/g++ \
    -std=c++17 \
    $OPT_FLAGS \
    $VEC_REPORT \
    $INCLUDE_FLAGS \
    $CXXFLAGS \
    $SRC_FILES \
    $LIBS \
    $EXTRA_LDFLAGS \
    -o "$OUT_FILE"

echo "✅ Build success: $OUT_FILE"

if [ "$VERBOSE" = "1" ]; then
    echo ""
    echo "────────── Vectorization Summary ──────────"
    if [ -f vec_report.txt ]; then
        echo "Vectorized loops:"
        grep -c "loop vectorized" vec_report.txt || echo "  (none)"
    fi
    if [ -f vec_missed.txt ]; then
        echo "Failed loops (top 10 reasons):"
        grep "couldn't vectorize\|not vectorized" vec_missed.txt | head -10
    fi
fi
