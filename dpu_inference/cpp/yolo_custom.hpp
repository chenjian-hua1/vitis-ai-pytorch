#pragma once
/*
 * yolo_custom.hpp
 * ───────────────────────────────────────────────────────────────────────────
 * Type definitions and utility function declarations for
 * custom anchor-free DFL YOLO inference.
 *
 * Architecture:
 *   Output format : (B, H, W, 68)  NHWC  = box(64) + cls(4)
 *   3 scales      : 80x80 / 40x40 / 20x20  (stride 8 / 16 / 32)
 *   DFL           : ch=16,  4x16=64 channels for box regression
 *   num_classes   : 4
 *   Activation    : ReLU
 *
 * Backend selection (--device flag):
 *   dpu  — run on FPGA DPU via VART "run"      (board only)
 *   cpu  — run on x86/ARM CPU via VART "run_sim" (PC testing)
 */

#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <opencv2/core.hpp>

// ─────────────────────────────────────────────
//  User configuration (edit here)
// ─────────────────────────────────────────────
namespace cfg {

constexpr int   INPUT_W     = 640;
constexpr int   INPUT_H     = 640;
constexpr int   NC          = 4;            // num_classes
constexpr int   DFL_CH      = 16;           // Head self.ch
constexpr int   OUT_CH      = DFL_CH * 4 + NC;  // 68
constexpr float CONF_THRESH = 0.25f;
constexpr float IOU_THRESH  = 0.45f;

// Strides corresponding to each output scale (large -> small)
constexpr std::array<int, 3> STRIDES = {8, 16, 32};

// Class names — replace with your actual label names
inline const std::vector<std::string> CLASS_NAMES = {
    "class0", "class1", "class2", "class3"
};

// BGR color palette, one entry per class
inline const std::vector<cv::Scalar> PALETTE = {
    {56,  56,  255}, {151, 157, 255},
    {31,  112, 255}, {29,  178, 255}
};

} // namespace cfg


// ─────────────────────────────────────────────
//  Backend selector
// ─────────────────────────────────────────────
enum class RunDevice {
    DPU,   // FPGA DPU  — VART runner key: "run"
    CPU    // CPU sim   — VART runner key: "run_sim"  (PC testing)
};

// Returns the VART runner key string for the given device
inline const char* runner_key(RunDevice dev) {
    return (dev == RunDevice::DPU) ? "run" : "run_sim";
}

// Parses "dpu" / "cpu" string (case-insensitive) to RunDevice
// Throws std::invalid_argument on unknown value
inline RunDevice parse_device(const std::string& s) {
    if (s == "dpu" || s == "DPU") return RunDevice::DPU;
    if (s == "cpu" || s == "CPU") return RunDevice::CPU;
    throw std::invalid_argument("Unknown device: \"" + s +
                                "\". Valid options: dpu, cpu");
}


// ─────────────────────────────────────────────
//  Data structures
// ─────────────────────────────────────────────
struct Detection {
    float x1, y1, x2, y2;   // pixel coordinates in original image space
    float conf;
    int   cls_id;
};

struct LetterboxInfo {
    float scale;
    int   pad_left;
    int   pad_top;
    int   orig_w;
    int   orig_h;
};


// ─────────────────────────────────────────────
//  Function declarations
// ─────────────────────────────────────────────

// Letterbox resize — maintains aspect ratio and pads with gray (114)
cv::Mat letterbox(const cv::Mat& bgr, int target_w, int target_h,
                  LetterboxInfo& info);

// Decode a single output scale
// feat_ptr: float* pointing to (H, W, OUT_CH) data (NHWC, batch dim removed)
void decode_scale(const float* feat_ptr,
                  int feat_h, int feat_w,
                  int stride,
                  float conf_thresh,
                  std::vector<Detection>& out_dets);

// Non-maximum suppression
std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thresh);

// Restore coordinates from letterbox space to original image space
void restore_coords(std::vector<Detection>& dets, const LetterboxInfo& info);

// Draw bounding boxes and labels on frame
void draw_detections(cv::Mat& frame, const std::vector<Detection>& dets);