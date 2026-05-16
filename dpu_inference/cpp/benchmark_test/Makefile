# Makefile for layout benchmark
#
# 用法：
#   make              # 預設 -O3 + 自動偵測 host 架構的最佳指令集
#   make O2           # -O2 不向量化（接近 vendor toolchain 預設）
#   make debug        # -O0 -g
#   make clean
#
# 在 ARM 板子上 cross-compile（如果是直接 native compile 就用 `make` 即可）：
#   make CXX=aarch64-linux-gnu-g++

CXX      ?= g++
TARGET   := build/bench_layout_arm
SRC      := bench_layout_arm.cpp

CXXFLAGS_COMMON := -std=c++17 -Wall -Wextra

# 自動偵測架構，加上對應的優化旗標
UNAME_M  := $(shell uname -m)
ifeq ($(UNAME_M),aarch64)
  ARCH_FLAGS := -march=armv8-a
else ifeq ($(UNAME_M),armv7l)
  ARCH_FLAGS := -march=armv7-a -mfpu=neon
else
  ARCH_FLAGS := -march=native
endif

CXXFLAGS_O3 := $(CXXFLAGS_COMMON) -O3 $(ARCH_FLAGS)
CXXFLAGS_O2 := $(CXXFLAGS_COMMON) -O2 $(ARCH_FLAGS)
CXXFLAGS_DB := $(CXXFLAGS_COMMON) -O0 -g

.PHONY: all O2 debug clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS_O3) $(SRC) -o $(TARGET)
	@echo "Built $(TARGET) with: $(CXXFLAGS_O3)"

O2: $(SRC)
	$(CXX) $(CXXFLAGS_O2) $(SRC) -o $(TARGET)_O2
	@echo "Built $(TARGET)_O2 with: $(CXXFLAGS_O2)"

debug: $(SRC)
	$(CXX) $(CXXFLAGS_DB) $(SRC) -o $(TARGET)_debug
	@echo "Built $(TARGET)_debug with: $(CXXFLAGS_DB)"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET)_O2 $(TARGET)_debug