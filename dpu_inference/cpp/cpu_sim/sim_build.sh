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
echo "🖥️  Detected architecture: $ARCH"

# ===== choose optimization flags =====
if [[ "$ARCH" == "x86_64" ]]; then
    echo "⚙️  Target: PC (x86_64)"
    OPT_FLAGS="-O3 -DNDEBUG -march=native -flto"

elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    echo "⚙️  Target: ARM (aarch64)"

    # 👉 你是 KV260（Cortex-A72）
    OPT_FLAGS="-O3 -DNDEBUG -mcpu=cortex-a72 -ffast-math"

else
    echo "⚠️  Unknown architecture, fallback generic"
    OPT_FLAGS="-O3 -DNDEBUG"
fi

# ===== OpenCV flags =====
CXXFLAGS="$(pkg-config --cflags opencv4)"
LIBS="$(pkg-config --libs opencv4)"

# ===== include paths =====
INCLUDE_FLAGS="-I."
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
    -o "$OUT_FILE"

echo "✅ Build success: $OUT_FILE"