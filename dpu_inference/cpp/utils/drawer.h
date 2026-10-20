// ============================================================================
//  drawer.h 追加內容
//  原本 draw_tracks / draw_tracking 的宣告「保持不變」，只需把下面這段加進去
// ============================================================================
#pragma once
#include <tracker.h>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// 預載入模式
//   Off   ：不預載入（預設），prefetch_track_regions 直接返回
//   Read  ：一般讀取，載入 cache line（同一執行緒擷取 + 繪圖時建議用這個）
//   Write ：讀後寫回原值，同時取得獨佔權（frame 由其他核心寫入時可考慮）
enum class DrawPrefetch { Off, Read, Write };

void         set_draw_prefetch(DrawPrefetch mode);
DrawPrefetch get_draw_prefetch();

// draw_tracking 使用的類別名稱（預設為空，行為與原版相同：顯示 "Class N"）
// 請在初始化時設定一次，執行期間不要再改
void set_draw_class_names(const std::vector<std::string>& names);

// 在 DPU 執行期間呼叫：依 ByteTrack 預測框載入 frame 上即將繪圖的區塊
// predicted 的座標必須與 frame 相同；模式為 Off 時不做任何事
void prefetch_track_regions(cv::Mat& frame,
                            const std::vector<bytetrack::Track>& predicted);

// 只在 area 範圍內繪圖與預載入（座標仍使用整張影像的座標系，內部會平移）
//   - 所有線條、標籤都被裁切在 area 內，area 以外的像素（例如 letterbox 黑邊）不會被寫入
//   - FPS 文字出現在 area 的左上角
void draw_tracking_in(cv::Mat& img, const cv::Rect& area,
                      const std::vector<bytetrack::Track>& tracks, double fps);
void prefetch_track_regions_in(cv::Mat& img, const cv::Rect& area,
                               const std::vector<bytetrack::Track>& predicted);

                               // letterbox 座標 → 原圖座標，並轉成 tracker 的輸入格式
void scale_detections(const DetectionBatch& detections,
                      std::vector<bytetrack::Box> &out_boxes,
                      const ResizeResult&   resize_inf,
                      cv::Size              orig_shape);

void map_detections(const DetectionBatch& det,
                    std::vector<bytetrack::Box>& out,
                    float off_x, float off_y,   // 要扣掉的 letterbox padding
                    float sx, float sy,         // 額外縮放(不縮就傳 1)
                    cv::Size dst);               // 目標座標系,用來 clamp

                    
cv::Mat draw_tracks(const cv::Mat& img,
                    const std::vector<bytetrack::Track>& tracks,
                    const std::vector<std::string>& class_names = {});

void draw_tracking(const cv::Mat& in, cv::Mat& out,
                   const std::vector<bytetrack::Track>& tracks,
                   double fps);
