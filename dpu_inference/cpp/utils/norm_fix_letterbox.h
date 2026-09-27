// norm_letterbox_lut.hpp
// Letterbox 影像的 norm_and_fix，NEON 查表版
//
// 呼叫端只需要：
//   norm_and_fix_letterbox(img, fix_point, y0, y1, out);
//
// 其餘全部在 norm_letterbox_lut.cpp 內處理：
//   - 查表：fix_point 改變時自動重建（結果與原本 float 版逐位元一致）
//   - 黑邊：同一個 out buffer、同樣的 y0/y1/fix_point/尺寸 → 只在第一次填入
//   - 只計算影像內容列 [y0, y1)，NEON TBL/TBX 一次處理 16 pixel
//
// 注意：
//   - out 請每幀重複使用同一個 cv::Mat；最多同時記住 4 個 out buffer（例如 double buffering）。
//   - 不要自己寫入 out 的黑邊區域；若真的寫了，呼叫 norm_letterbox_reset() 讓下一次重填。
//   - 快取是 thread_local，每個執行緒各自一份；reset 也只作用在呼叫它的執行緒。

#pragma once

#include <opencv2/core.hpp>

// x:         CV_8UC3、連續記憶體的 letterbox 後影像
// fix_point: 同原本 norm_and_fix
// y0, y1:    影像內容所在的列範圍 [y0, y1)，其餘列為黑邊
// out:       輸出 CV_8SC3；每幀重複使用同一個 Mat
void norm_and_fix_letterbox(const cv::Mat& x, int fix_point, int y0, int y1, cv::Mat& out);

// 清除目前執行緒的快取：下一次呼叫會重建查表並重填黑邊，同時釋放持有的舊 buffer
void norm_letterbox_reset();
