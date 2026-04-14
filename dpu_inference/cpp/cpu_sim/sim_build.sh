#!/bin/bash
set -e

# ===== input =====
SRC_FILE="$1"
EXTRA_PATH="$2"   # 🔥 額外 implementation 資料夾

# ===== output =====
OUT_DIR="$(dirname "$SRC_FILE")/build"
OUT_FILE="$OUT_DIR/$(basename "${SRC_FILE%.*}")"

mkdir -p "$OUT_DIR"

# ===== OpenCV flags =====
CXXFLAGS=$(pkg-config --cflags opencv4)
LIBS=$(pkg-config --libs opencv4)

# ===== collect sources =====
SRC_FILES="$SRC_FILE"

# 如果有 extra_path，就把裡面的 cpp 都加進來
if [ -n "$EXTRA_PATH" ]; then
    EXTRA_FILES=$(find "$EXTRA_PATH" -name "*.cpp")
    SRC_FILES="$SRC_FILES $EXTRA_FILES"
fi

echo "📦 Source files:"
echo "$SRC_FILES"

# ===== compile =====
/usr/bin/g++ \
    -std=c++17 \
    -g \
    $SRC_FILES \
    $CXXFLAGS \
    $LIBS \
    -o "$OUT_FILE"

echo "✅ Build success: $OUT_FILE"