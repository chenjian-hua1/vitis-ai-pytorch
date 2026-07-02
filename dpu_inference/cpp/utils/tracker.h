// kalman_box_tracker.h
// 宣告檔：YOLO(NMS 後) bounding box 的等速模型 Kalman filter，
// 用來預測物件下一偵位置。實作在 kalman_box_tracker.cpp。
//
//   #include "kalman_box_tracker.h"
//   // 編譯時記得一起連結 kalman_box_tracker.cpp
//   using namespace kbt;
//   KalmanBoxTracker t(toCenter({100,100,150,200}));
//   CxCyWH pred = t.predict();      // 預測這一偵
//   t.update(toCenter(detBox));     // 拿 YOLO 偵測校正

#ifndef KALMAN_BOX_TRACKER_H
#define KALMAN_BOX_TRACKER_H

#include <vector>

namespace kbt {

// ======================= 極簡矩陣 =======================
struct Mat {
    int r = 0, c = 0;
    std::vector<double> d;

    Mat() {}
    Mat(int r, int c) : r(r), c(c), d(static_cast<size_t>(r) * c, 0.0) {}

    // 存取子是一行、且在迴圈裡大量呼叫，留在 header inline
    double&       operator()(int i, int j)       { return d[i * c + j]; }
    double        operator()(int i, int j) const { return d[i * c + j]; }

    static Mat I(int n);   // 單位矩陣，實作在 .cpp
};

// 矩陣運算（實作在 .cpp）
Mat operator*(const Mat& a, const Mat& b);
Mat operator+(const Mat& a, const Mat& b);
Mat operator-(const Mat& a, const Mat& b);
Mat transpose(const Mat& a);
Mat inverse(Mat a);          // Gauss-Jordan 求逆

// ======================= bbox 格式 =======================
struct Box    { double x1, y1, x2, y2; };   // YOLO 常見輸出
struct CxCyWH { double cx, cy, w, h; };

CxCyWH toCenter(const Box& b);   // xyxy -> 中心點
Box    toXYXY(const CxCyWH& c);  // 中心點 -> xyxy

// ======================= Kalman Box Tracker =======================
// 狀態 x(8): [cx, cy, w, h, vx, vy, vw, vh]，量測 z(4): [cx, cy, w, h]
class KalmanBoxTracker {
public:
    // 用第一次偵測框初始化。dt = 兩偵間隔(用「偵」為單位時填 1)。
    // 預設參數寫在宣告端。
    explicit KalmanBoxTracker(const CxCyWH& init, double dt = 1.0);

    CxCyWH predict();                    // 預測步：往前推一偵，回傳預測框
    void   update(const CxCyWH& meas);   // 更新步：拿當偵偵測框校正

    CxCyWH currentBox() const;           // 目前估計框(中心點格式)
    void   velocity(double& vx, double& vy, double& vw, double& vh) const;

    // 可選：外部微調雜訊
    void setMeasurementNoise(double pos, double size); // 大→平滑
    void setProcessNoise(double pos, double vel);      // 大→靈敏

private:
    Mat x_, P_, F_, H_, Q_, R_;
};

} // namespace kbt

#endif // KALMAN_BOX_TRACKER_H
