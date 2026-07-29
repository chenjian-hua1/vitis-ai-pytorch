#pragma once

#include <opencv2/opencv.hpp>

// ============================================================================
//  Data Structures
// ============================================================================


#if defined(__GNUC__) || defined(__clang__)
// 指標有可能指到同一塊記憶體 如果做層層運算時可能會互相影響只能照順序
// 需要都設成指到的記憶體空間都是獨立不同 不會有相依性才能平行化
#  define RESTRICT __restrict__
#else
#  define RESTRICT
#endif

struct ResizeResult {
    cv::Mat img;
    std::pair<float,float> ratio;
    std::pair<float,float> pad;
};

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int   class_id;
};

struct DetectionBatch {
    std::vector<Detection> data;
    int                    count;

    Detection*       begin()       { return data.data(); }
    Detection*       end()         { return data.data() + count; }
    const Detection* begin() const { return data.data(); }
    const Detection* end()   const { return data.data() + count; }
    int              size()  const { return count; }
};