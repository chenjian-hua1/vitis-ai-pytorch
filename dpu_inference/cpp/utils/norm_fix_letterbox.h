// norm_fix_letterbox.h
// Letterbox 影像的 norm_and_fix（AArch64 用 NEON 查表，其他平台用一般計算）
//
// 呼叫端只需要：
//   ResizeResult r = letterbox(...);
//   norm_and_fix_letterbox(r, fix_point, out);
//
// 其餘全部在 norm_fix_letterbox.cpp 內處理：
//   - 參數：fix_point 改變時自動重建查表／係數（結果與原本 float 版逐位元一致）
//   - 黑邊：依 r.content 判斷，上下左右都支援；
//           同一個 out buffer、同樣的 content/fix_point/尺寸 → 只在第一次填入
//   - 只計算 r.content 內的像素
//       AArch64（含 KV260）：NEON TBL/TBX 查表，一次處理 16 pixel
//       其他平台          ：一般 float 計算（原本 norm_and_fix 的公式）
//
// 注意：
//   - out 請每幀重複使用同一個 cv::Mat；最多同時記住 4 個 out buffer（例如 double buffering）。
//   - 不要自己寫入 out 的黑邊區域；若真的寫了，呼叫 norm_letterbox_reset() 讓下一次重填。
//   - 快取是 thread_local，每個執行緒各自一份；reset 也只作用在呼叫它的執行緒。
//   - 黑邊範圍只看 r.content；r.ratio / r.pad 不會被使用（content 已經是整數像素，最準確）。

#pragma once

#include "data_struct.h"

#include <opencv2/core.hpp>
#include <utility>

// r:         letterbox 的結果；使用 r.img 與 r.content
// fix_point: 同原本 norm_and_fix
// out:       輸出 CV_8SC3；每幀重複使用同一個 Mat
void norm_and_fix_letterbox(const ResizeResult& r, int fix_point, cv::Mat& out);

// 底層版本：直接給影像與內容區域（上面那個就是呼叫這個）
void norm_and_fix_letterbox(const cv::Mat& x, int fix_point, const cv::Rect& content, cv::Mat& out);

// 清除目前執行緒的快取：下一次呼叫會重建查表並重填黑邊，同時釋放持有的舊 buffer
void norm_letterbox_reset();