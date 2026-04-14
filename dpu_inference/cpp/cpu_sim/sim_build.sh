#!/bin/bash

SRC_FILE="$1"
OUT_DIR="$(dirname "$SRC_FILE")/build"
OUT_FILE="$OUT_DIR/$(basename "${SRC_FILE%.*}")"

mkdir -p "$OUT_DIR"

# 透過 pkg-config 取得 flags
CXXFLAGS=$(pkg-config --cflags opencv4)
LIBS=$(pkg-config --libs opencv4)

/usr/bin/g++ \
    -fcolor-diagnostics \
    -fansi-escape-codes \
    -g \
    "$SRC_FILE" \
    $CXXFLAGS \
    $LIBS \
    -o "$OUT_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Build success: $OUT_FILE"
else
    echo "❌ Build failed"
fi