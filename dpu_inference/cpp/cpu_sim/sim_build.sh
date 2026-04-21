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
if [[ "$ARCH" == "x86_64" ]]; then
    echo "⚙️  Target: PC (x86_64)"
    OPT_FLAGS="-O3 -DNDEBUG -march=native -flto"

elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then

    if [[ "$OS" == "Darwin" ]]; then
        echo "⚙️  Target: Apple Silicon (arm64)"
        OPT_FLAGS="-O3 -DNDEBUG -mcpu=apple-m1 -ffast-math"

        # Homebrew 在 Apple Silicon 裝在 /opt/homebrew
        HOMEBREW_PREFIX="/opt/homebrew"
        export PKG_CONFIG_PATH="$HOMEBREW_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
        INCLUDE_FLAGS="-I. -I$HOMEBREW_PREFIX/include"
        EXTRA_LDFLAGS="-L$HOMEBREW_PREFIX/lib"
    else
        echo "⚙️  Target: ARM (aarch64)"
        # 👉 你是 KV260（Cortex-A72）
        OPT_FLAGS="-O3 -DNDEBUG -mcpu=cortex-a72 -ffast-math"
        INCLUDE_FLAGS="-I."
        EXTRA_LDFLAGS=""
    fi

else
    echo "⚠️  Unknown architecture, fallback generic"
    OPT_FLAGS="-O3 -DNDEBUG"
    INCLUDE_FLAGS="-I."
    EXTRA_LDFLAGS=""
fi

# ===== OpenCV flags =====
CXXFLAGS="$(pkg-config --cflags opencv4)"
# LIBS="$(pkg-config --libs opencv4)"
LIBS="$(pkg-config --libs opencv4) -lonnxruntime"

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

# ===== compile =====
/usr/bin/g++ \
    -std=c++17 \
    $OPT_FLAGS \
    $INCLUDE_FLAGS \
    $CXXFLAGS \
    $SRC_FILES \
    $LIBS \
    $EXTRA_LDFLAGS \
    -o "$OUT_FILE"

echo "✅ Build success: $OUT_FILE"