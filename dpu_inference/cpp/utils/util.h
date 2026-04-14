#pragma once

#include <vector>
#include <string>
#include <tuple>

#include <opencv2/opencv.hpp>

// ============================================================================
//  Data Structures
// ============================================================================

/**
 * @brief Result of make_anchors(): anchor (x,y) coordinates and stride values.
 */
struct AnchorResult {
    cv::Mat anchors;       ///< (A, 2) float32 — anchor (x, y) in feature-map space
    cv::Mat stride_tensor; ///< (A, 1) float32 — stride per anchor
};

/**
 * @brief Result of resize(): resized image plus metadata.
 */
struct ResizeResult {
    cv::Mat img;           ///< Resized + padded image
    std::pair<float,float> ratio; ///< (r, r) scale ratio
    std::pair<float,float> pad;   ///< (w, h) padding in pixels
};

/**
 * @brief Single detection: (x1, y1, x2, y2, score, class_id).
 */
struct Detection {
    float x1, y1, x2, y2;
    float score;
    int   class_id;
};

// ============================================================================
//  Fix / Float Conversion
// ============================================================================

/**
 * @brief Convert int8 fixed-point array to float32.
 *
 * @param data      CV_8S Mat of fixed-point values
 * @param fix_point Exponent (scale = 2^(-fix_point))
 * @return          CV_32F Mat
 */
cv::Mat fix2float(const cv::Mat& data, int fix_point);

/**
 * @brief Convert float32 array to int8 fixed-point.
 *
 * @param data      CV_32F Mat
 * @param fix_point Exponent (scale = 2^fix_point)
 * @return          CV_8S Mat clipped to [-128, 127]
 */
cv::Mat float2fix(const cv::Mat& data, int fix_point);

// ============================================================================
//  Bounding Box Utilities
// ============================================================================

/**
 * @brief Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2).
 *
 * @param x CV_32F Mat of shape (N, 4+)
 * @return  CV_32F Mat of same shape, first 4 columns converted
 */
cv::Mat wh2xy(const cv::Mat& x);

/**
 * @brief Convert boxes from (x1, y1, x2, y2) to (x, y, w, h).
 *
 * @param box CV_32F Mat of shape (N, 4)
 * @return    CV_32F Mat of shape (N, 4)
 */
cv::Mat xyxy2xywh(const cv::Mat& box);

// ============================================================================
//  Anchor Generation
// ============================================================================

/**
 * @brief Generate anchor points for each feature-map scale.
 *
 * @param feature_maps List of feature maps, each CV_32F (B, C, H, W) stored
 *                     as a 4-dim Mat or equivalent shape vector {B,C,H,W}.
 * @param strides      Downsampling factors, e.g. {8, 16, 32}
 * @param offset       Anchor offset from grid origin (default 0.5)
 * @return             AnchorResult with anchors (A,2) and stride_tensor (A,1)
 */
AnchorResult make_anchors(const std::vector<cv::Mat>& feature_maps,
                           const std::vector<int>&     strides,
                           float                       offset = 0.5f);

// ============================================================================
//  Pre-Processing
// ============================================================================

/**
/**
 * @brief Normalize an image using ImageNet mean/std statistics.
 *
 * Performs per-channel linear transformation:
 *   out[c] = (x[c] / (std[c] * 255)) - (mean[c] / std[c])   (uint8 input)
 *   out[c] =  x[c] / std[c]          - (mean[c] / std[c])   (float32 input)
 *
 * ImageNet constants:
 *   mean = [0.485, 0.456, 0.406]  (R, G, B)
 *   std  = [0.229, 0.224, 0.225]  (R, G, B)
 *
 * @param x Input image with the following constraints:
 *            - Shape  : (H, W, 3), must be continuous in memory
 *            - Depth  : CV_8U  [0, 255]  or CV_32F [0.0, 1.0]
 *            - Channel: 3 (BGR or RGB, channel order must match
 *                          the order used during model training)
 *
 * @return cv::Mat with:
 *            - Depth  : CV_32F
 *            - Shape  : same as input (H, W, 3)
 *            - Range  : approximately [-2.5, 2.5] per channel
 *
 * @throws cv::Exception if x is not 3-channel or not continuous
 *
 * @note Channel order (BGR vs RGB) is NOT checked internally.
 *       Caller is responsible for ensuring correct channel ordering
 *       before passing to this function.
 */
cv::Mat norm(const cv::Mat& x);

/**
 * @brief Letterbox-resize an image to a square of side `input_size`.
 *
 * Only downscales (r <= 1.0). Pads with black borders to reach exact size.
 *
 * @param img        Input BGR image
 * @param input_size Target side length (e.g. 640)
 * @return           ResizeResult containing image, ratio, and padding
 */
ResizeResult resize(const cv::Mat& img, int input_size);
ResizeResult resize_zero_copy(const cv::Mat& img, int input_size);

// ============================================================================
//  Post-Processing
// ============================================================================

/**
 * @brief DFL (Distribution Focal Loss) decoding.
 *
 * @param x   CV_32F Mat shaped (B, 4*ch, A)
 * @param ch  Number of DFL bins (default 16)
 * @return    CV_32F Mat shaped (B, 4, A)
 */
cv::Mat dfl_decode(const cv::Mat& x, int ch = 16);

/**
 * @brief Run Non-Maximum Suppression on raw YOLO output.
 *
 * @param outputs              CV_32F Mat of shape (B, 4+nc, A)
 * @param confidence_threshold Minimum objectness threshold (default 0.001)
 * @param iou_threshold        IoU threshold for NMS (default 0.65)
 * @return                     Vector (length B) of Detection vectors
 */
std::vector<std::vector<Detection>> non_max_suppression(
    const cv::Mat& outputs,
    float          confidence_threshold = 0.001f,
    float          iou_threshold        = 0.65f);

// ============================================================================
//  YOLO Post-Processor Class
// ============================================================================

/**
 * @brief Decode raw YOLO multi-scale feature maps into box + class predictions.
 *
 * Usage:
 *   YOLOPostProcessor pp(80, 16, {8, 16, 32});
 *   cv::Mat result = pp(feature_maps);
 */
class YOLOPostProcessor {
public:
    /**
     * @param nc      Number of classes (e.g. 80 for COCO). Default: 80
     * @param ch      DFL bins per coordinate. Default: 16
     * @param strides Downsampling strides for each scale. Default: {8,16,32}
     */
    explicit YOLOPostProcessor(int                nc      = 80,
                               int                ch      = 16,
                               std::vector<int>   strides = {8, 16, 32});

    /**
     * @brief Decode feature maps.
     *
     * @param x           List of feature maps, each CV_32F (B, no, H*W) or (B, no, H, W)
     * @param conf_thresh Confidence threshold (informational; NMS applies filtering)
     * @return            CV_32F Mat (B, 4+nc, A) — [cx,cy,w,h, cls_score...]
     */
    cv::Mat operator()(const std::vector<cv::Mat>& x,
                       float                        conf_thresh = 0.25f) const;

private:
    int              nc_;
    int              ch_;
    int              no_;
    std::vector<int> strides_;
};

// ============================================================================
//  Visualization
// ============================================================================

/**
 * @brief Scale boxes from resized/padded image back to original coordinates.
 *
 * @param boxes      (N, 4) CV_32F in (x1, y1, x2, y2)
 * @param ratio      (r, r) from ResizeResult
 * @param pad        (w, h) padding from ResizeResult
 * @param orig_shape Original image size as cv::Size (width, height)
 * @return           (N, 4) CV_32F boxes in original image coordinates
 */
cv::Mat scale_boxes(const cv::Mat&              boxes,
                    std::pair<float,float>       ratio,
                    std::pair<float,float>       pad,
                    cv::Size                     orig_shape);

/**
 * @brief Draw bounding boxes on an image.
 *
 * @param img          BGR image (H, W, 3) CV_8UC3
 * @param detections   Vector of Detection structs
 * @param class_names  Optional class name list
 * @return             Copy of image with boxes drawn
 */
cv::Mat draw_boxes(const cv::Mat&                    img,
                   const std::vector<Detection>&     detections,
                   const std::vector<std::string>&   class_names = {});