cmake_minimum_required(VERSION 3.12)
project(yolo_custom_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# ── OpenCV ────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── Vitis AI VART / XIR (installed at /usr on board) ─────────────────────
find_path(XIR_INCLUDE_DIR   xir/graph/graph.hpp PATHS /usr/include)
find_path(VART_INCLUDE_DIR  vart/runner.hpp     PATHS /usr/include)

find_library(XIR_LIB    xir   PATHS /usr/lib /usr/local/lib)
find_library(VART_LIB   vart  PATHS /usr/lib /usr/local/lib)

# ── Executable ────────────────────────────────────────────────────────────
add_executable(yolo_infer
    src/main.cpp
    src/yolo_postprocess.cpp
)

target_include_directories(yolo_infer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${XIR_INCLUDE_DIR}
    ${VART_INCLUDE_DIR}
)

target_link_libraries(yolo_infer PRIVATE
    ${OpenCV_LIBS}
    ${XIR_LIB}
    ${VART_LIB}
    pthread
)

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS yolo_infer RUNTIME DESTINATION bin)

message(STATUS "Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV      : ${OpenCV_VERSION}")
message(STATUS "XIR lib     : ${XIR_LIB}")
message(STATUS "VART lib    : ${VART_LIB}")
