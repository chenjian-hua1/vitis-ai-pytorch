#!/bin/bash
set -e

# ===== input =====
SRC_FILE="$1"
EXTRA_PATH="$2"

# ===== output =====
OUT_DIR="$(dirname "$SRC_FILE")/build"
OUT_FILE="$OUT_DIR/$(basename "${SRC_FILE%.*}")"
mkdir -p "$OUT_DIR"

# ===== detect build mode =====
HOST_ARCH=$(uname -m)
HOST_OS=$(uname -s)

# 強制 cross compile 用 CROSS_COMPILE=1，否則照 host 跑 native
IS_CROSS=0
if [[ "$CROSS_COMPILE" == "1" ]]; then
    IS_CROSS=1
elif [[ -n "$CXX" && "$CXX" == *aarch64*xilinx*linux* ]]; then
    IS_CROSS=1
fi

echo "🖥️  Host: $HOST_ARCH  OS: $HOST_OS"

# ===== common SIMD-friendly flags =====
COMMON_SIMD_FLAGS="-fopenmp-simd -funroll-loops"

# ===== choose compiler & flags =====
if [[ "$IS_CROSS" == "1" ]]; then
    echo "⚙️  Target: Cross-compile to aarch64 (KV260, PetaLinux 2022.2 + Vitis AI)"

    # ============================================================
    # 🔥 全部寫死，不靠環境變數
    # ============================================================
    PETALINUX_ROOT="/home/jianhua/petalinux/cross_compile"
    SYSROOT="$PETALINUX_ROOT/sysroots/cortexa72-cortexa53-xilinx-linux"
    HOST_TOOLS="$PETALINUX_ROOT/sysroots/x86_64-petalinux-linux"
    COMPILER="$HOST_TOOLS/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-g++"
    # COMPILER = ${CXX}

    # ---- 基本檢查 ----
    if [ ! -x "$COMPILER" ]; then
        echo "❌ Compiler not found: $COMPILER"
        exit 1
    fi
    if [ ! -d "$SYSROOT" ]; then
        echo "❌ Sysroot not found: $SYSROOT"
        exit 1
    fi
    echo "🔧 Compiler: $COMPILER"
    echo "📂 Sysroot:  $SYSROOT"

    # ---- preflight: 檢查 VAI / XRT / boost 在不在 ----
    MISSING=0
    for h in xir/graph/graph.hpp vart/runner.hpp UniLog/UniLog.hpp; do
        [ -f "$SYSROOT/usr/include/$h" ] || { echo "❌ Missing header: $h"; MISSING=1; }
    done
    # 必要的 VAI lib (libvart-dpu-controller 在某些版本可能沒有獨立 .so，改成 optional)
    for l in libxir libvart-runner libunilog; do
        ls "$SYSROOT/usr/lib/${l}".so* >/dev/null 2>&1 \
            || { echo "❌ Missing VAI lib: $l"; MISSING=1; }
    done
    # optional VAI libs (有就連，沒有就跳過)
    OPTIONAL_VART_LIBS=""
    for l in vart-dpu-controller vart-util vart-xrt-device-handle vart-buffer-object vart-runner-assistant; do
        if ls "$SYSROOT/usr/lib/lib${l}".so* >/dev/null 2>&1; then
            OPTIONAL_VART_LIBS="$OPTIONAL_VART_LIBS -l${l}"
        else
            echo "ℹ️  optional vart lib not found, skip: lib${l}"
        fi
    done
    for l in libxrt_core libxrt_coreutil; do
        ls "$SYSROOT/usr/lib/${l}".so* >/dev/null 2>&1 \
            || { echo "❌ Missing XRT lib: $l"; MISSING=1; }
    done
    for l in libboost_filesystem libboost_system; do
        ls "$SYSROOT/usr/lib/${l}".so* >/dev/null 2>&1 \
            || { echo "❌ Missing boost: $l"; MISSING=1; }
    done
    # 檢查 OpenCV dnn 是否存在 (你的 code 用到 cv::dnn::NMSBoxes)
    if ! ls "$SYSROOT/usr/lib/libopencv_dnn".so* >/dev/null 2>&1; then
        echo "❌ Missing OpenCV: libopencv_dnn (cv::dnn::NMSBoxes 需要)"
        MISSING=1
    fi
    if [ "$MISSING" == "1" ]; then
        echo ""
        echo "💡 從 KV260 板子拷缺的 lib："
        echo "   sudo rsync -av root@<KV260_IP>:'/usr/lib/libxrt*'        $SYSROOT/usr/lib/"
        echo "   sudo rsync -av root@<KV260_IP>:'/usr/lib/libboost*'      $SYSROOT/usr/lib/"
        echo "   sudo rsync -av root@<KV260_IP>:'/usr/lib/libopencv*'     $SYSROOT/usr/lib/"
        echo "   sudo rsync -av root@<KV260_IP>:/usr/include/xrt          $SYSROOT/usr/include/"
        exit 1
    fi
    echo "✅ Sysroot deps OK"

    # ---- arch flags ----
    OPT_FLAGS="-O3 -DNDEBUG \
               --sysroot=$SYSROOT \
               -march=armv8-a+simd+crc \
               -mcpu=cortex-a53 -mtune=cortex-a53 \
               -ffast-math $COMMON_SIMD_FLAGS"

    # ---- pkg-config 走 target sysroot（local 設定，不污染外部 shell）----
    export PKG_CONFIG_PATH="$SYSROOT/usr/lib/pkgconfig:$SYSROOT/usr/share/pkgconfig"
    export PKG_CONFIG_SYSROOT_DIR="$SYSROOT"
    export PKG_CONFIG_LIBDIR="$SYSROOT/usr/lib/pkgconfig:$SYSROOT/usr/share/pkgconfig"

    INCLUDE_FLAGS="-I. -I$SYSROOT/usr/include"

    # ---- linker flags ----
    EXTRA_LDFLAGS="--sysroot=$SYSROOT \
                   -L$SYSROOT/usr/lib \
                   -Wl,-rpath-link,$SYSROOT/usr/lib \
                   -Wl,--copy-dt-needed-entries"

else
    # ===== native build =====
    COMPILER="/usr/bin/g++"
    ARCH="$HOST_ARCH"
    OS="$HOST_OS"
    OPTIONAL_VART_LIBS=""

    if [[ "$ARCH" == "x86_64" ]]; then
        echo "⚙️  Target: PC (x86_64) native"
        OPT_FLAGS="-O3 -DNDEBUG -march=native -flto -ffast-math $COMMON_SIMD_FLAGS"
        INCLUDE_FLAGS="-I."
        EXTRA_LDFLAGS=""
    elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
        if [[ "$OS" == "Darwin" ]]; then
            echo "⚙️  Target: Apple Silicon (arm64) native"
            OPT_FLAGS="-O3 -DNDEBUG -mcpu=apple-m1 -ffast-math $COMMON_SIMD_FLAGS"
            HOMEBREW_PREFIX="/opt/homebrew"
            export PKG_CONFIG_PATH="$HOMEBREW_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
            INCLUDE_FLAGS="-I. -I$HOMEBREW_PREFIX/include"
            EXTRA_LDFLAGS="-L$HOMEBREW_PREFIX/lib"
        else
            echo "⚙️  Target: ARM (aarch64) native"
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
fi

# 🔥 XMODEL_MODE 巨集 + 消掉 GCC 10 的 ABI note
OPT_FLAGS="$OPT_FLAGS -DXMODEL_MODE -Wno-psabi"

# ===== OpenCV flags =====
# ⚠️ 重要: 不用 `pkg-config --libs opencv4` (會帶出 -lopencv_ts，板子沒裝)
#         改用 `--libs-only-L` 只取 -L 路徑，library 名稱手動列
# 順序: 上層 module 在前 (dnn)，底層在後 (imgproc, core)
CXXFLAGS_OPENCV="$(pkg-config --cflags opencv4)"
OPENCV_LIBS="$(pkg-config --libs-only-L opencv4) \
             -lopencv_videoio \
             -lopencv_imgcodecs \
             -lopencv_highgui \
             -lopencv_dnn \
             -lopencv_imgproc \
             -lopencv_core"

# ===== Vitis AI / XRT / boost 完整連結 =====
VITIS_LIBS="-lvart-runner $OPTIONAL_VART_LIBS \
            -lxir -lunilog -lglog \
            -lxrt_core -lxrt_coreutil \
            -lboost_filesystem -lboost_system \
            -lpthread -ldl -lrt"

LIBS="$OPENCV_LIBS $VITIS_LIBS"
CXXFLAGS="$CXXFLAGS_OPENCV"

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
echo "🔗 Libraries: $LIBS"
echo "🔗 LDFLAGS:   $EXTRA_LDFLAGS"

# ===== verbose vectorization report (optional) =====
VEC_REPORT=""
if [ "$VERBOSE" = "1" ]; then
    echo "📊 Vectorization report enabled"
    VEC_REPORT="-fopt-info-vec-optimized=vec_report.txt -fopt-info-vec-missed=vec_missed.txt"
fi

# ===== compile =====
"$COMPILER" \
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

# ===== confirm ELF arch =====
if command -v file >/dev/null 2>&1; then
    echo "📋 $(file "$OUT_FILE")"
fi

# ===== 顯示 binary 依賴的 .so (debug 用) =====
# if [[ "$IS_CROSS" == "1" ]] && command -v "$HOST_TOOLS/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-readelf" >/dev/null 2>&1; then
#    echo ""
#    echo "────────── NEEDED libraries (檢查板子上要有的 .so) ──────────"
#    "$HOST_TOOLS/usr/bin/aarch64-xilinx-linux/aarch64-xilinx-linux-readelf" -d "$OUT_FILE" | grep NEEDED || true
# fi

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
