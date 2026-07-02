// kalman_box_tracker.cpp
// kalman_box_tracker.h 的實作。
// 編譯範例: g++ -O2 -std=c++17 example_main.cpp kalman_box_tracker.cpp -o demo

#include "tracker.h"
#include <cmath>
#include <utility>   // std::swap

namespace kbt {

// ======================= 矩陣運算 =======================
Mat Mat::I(int n) {
    Mat m(n, n);
    for (int i = 0; i < n; ++i) m(i, i) = 1.0;
    return m;
}

Mat operator*(const Mat& a, const Mat& b) {
    Mat o(a.r, b.c);
    for (int i = 0; i < a.r; ++i)
        for (int k = 0; k < a.c; ++k) {
            double v = a(i, k);
            if (v == 0.0) continue;
            for (int j = 0; j < b.c; ++j) o(i, j) += v * b(k, j);
        }
    return o;
}

Mat operator+(const Mat& a, const Mat& b) {
    Mat o(a.r, a.c);
    for (size_t i = 0; i < a.d.size(); ++i) o.d[i] = a.d[i] + b.d[i];
    return o;
}

Mat operator-(const Mat& a, const Mat& b) {
    Mat o(a.r, a.c);
    for (size_t i = 0; i < a.d.size(); ++i) o.d[i] = a.d[i] - b.d[i];
    return o;
}

Mat transpose(const Mat& a) {
    Mat o(a.c, a.r);
    for (int i = 0; i < a.r; ++i)
        for (int j = 0; j < a.c; ++j) o(j, i) = a(i, j);
    return o;
}

// Gauss-Jordan (高斯橋登消去法) 求反矩陣 (KF 只會對 4x4 的 S 求逆)
Mat inverse(Mat a) {
    int n = a.r;
    Mat inv = Mat::I(n);
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::fabs(a(col, col));
        for (int r2 = col + 1; r2 < n; ++r2) {
            double v = std::fabs(a(r2, col));
            if (v > best) { best = v; piv = r2; }
        }
        if (piv != col)
            for (int j = 0; j < n; ++j) {
                std::swap(a(col, j), a(piv, j));
                std::swap(inv(col, j), inv(piv, j));
            }
        double pv = a(col, col);
        for (int j = 0; j < n; ++j) { a(col, j) /= pv; inv(col, j) /= pv; }
        for (int r2 = 0; r2 < n; ++r2) {
            if (r2 == col) continue;
            double f = a(r2, col);
            for (int j = 0; j < n; ++j) { a(r2, j) -= f * a(col, j); inv(r2, j) -= f * inv(col, j); }
        }
    }
    return inv;
}

// ======================= bbox 轉換 =======================
CxCyWH toCenter(const Box& b) {
    double w = b.x2 - b.x1, h = b.y2 - b.y1;
    return { b.x1 + w / 2.0, b.y1 + h / 2.0, w, h };
}

Box toXYXY(const CxCyWH& c) {
    return { c.cx - c.w / 2.0, c.cy - c.h / 2.0,
             c.cx + c.w / 2.0, c.cy + c.h / 2.0 };
}

// ======================= Kalman Box Tracker =======================
// 注意：預設參數只寫在 .h 的宣告端，這裡不再寫 = 1.0
KalmanBoxTracker::KalmanBoxTracker(const CxCyWH& init, double dt) {
    x_ = Mat(8, 1);
    x_(0, 0) = init.cx; x_(1, 0) = init.cy;
    x_(2, 0) = init.w;  x_(3, 0) = init.h;      // 速度初始 0

    F_ = Mat::I(8);                              // 等速模型
    F_(0, 4) = dt; F_(1, 5) = dt; F_(2, 6) = dt; F_(3, 7) = dt;

    H_ = Mat(4, 8);                              // 只觀測 cx,cy,w,h
    H_(0, 0) = H_(1, 1) = H_(2, 2) = H_(3, 3) = 1.0;

    P_ = Mat::I(8);                              // 速度未知 → 設大
    for (int i = 0; i < 4; ++i) P_(i, i) = 10.0;
    for (int i = 4; i < 8; ++i) P_(i, i) = 1000.0;

    Q_ = Mat::I(8);                              // 過程雜訊
    for (int i = 0; i < 4; ++i) Q_(i, i) = 1.0;
    for (int i = 4; i < 8; ++i) Q_(i, i) = 0.01;

    R_ = Mat::I(4);                              // 量測雜訊
    R_(0, 0) = R_(1, 1) = 1.0;
    R_(2, 2) = R_(3, 3) = 10.0;                  // w,h 較抖
}

CxCyWH KalmanBoxTracker::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * transpose(F_) + Q_;
    return currentBox();
}

void KalmanBoxTracker::update(const CxCyWH& meas) {
    Mat z(4, 1);
    z(0, 0) = meas.cx; z(1, 0) = meas.cy;
    z(2, 0) = meas.w;  z(3, 0) = meas.h;

    Mat y = z - H_ * x_;                         // 殘差 innovation
    Mat S = H_ * P_ * transpose(H_) + R_;        // 殘差協方差
    Mat K = P_ * transpose(H_) * inverse(S);     // Kalman gain
    x_ = x_ + K * y;
    Mat I = Mat::I(8);
    P_ = (I - K * H_) * P_;
}

CxCyWH KalmanBoxTracker::currentBox() const {
    return { x_(0, 0), x_(1, 0), x_(2, 0), x_(3, 0) };
}

void KalmanBoxTracker::velocity(double& vx, double& vy, double& vw, double& vh) const {
    vx = x_(4, 0); vy = x_(5, 0); vw = x_(6, 0); vh = x_(7, 0);
}

void KalmanBoxTracker::setMeasurementNoise(double pos, double size) {
    R_(0, 0) = R_(1, 1) = pos;
    R_(2, 2) = R_(3, 3) = size;
}

void KalmanBoxTracker::setProcessNoise(double pos, double vel) {
    for (int i = 0; i < 4; ++i) Q_(i, i) = pos;
    for (int i = 4; i < 8; ++i) Q_(i, i) = vel;
}

} // namespace kbt


/* 
using namespace kbt;
 
int main() {
    // 模擬每偵 YOLO+NMS 的框 (x1,y1,x2,y2)
    std::vector<Box> dets = {
        {100, 100, 150, 200}, {104, 103, 155, 204}, {109, 105, 161, 209},
        {113, 108, 166, 213}, {118, 111, 172, 218},
    };
 
    KalmanBoxTracker tracker(toCenter(dets[0]));  // 第 0 偵初始化
 
    std::cout << std::fixed << std::setprecision(1);
    for (size_t i = 1; i < dets.size(); ++i) {
        Box pb = toXYXY(tracker.predict());       // 1) 預測這一偵
        Box gt = dets[i];
        std::cout << "frame " << i
                  << "  預測=(" << pb.x1 << "," << pb.y1 << "," << pb.x2 << "," << pb.y2 << ")"
                  << "  實際=(" << gt.x1 << "," << gt.y1 << "," << gt.x2 << "," << gt.y2 << ")\n";
        tracker.update(toCenter(dets[i]));         // 2) 拿偵測校正
    }
 
    Box nb = toXYXY(tracker.predict());            // 預測未來下一偵
    double vx, vy, vw, vh; tracker.velocity(vx, vy, vw, vh);
    std::cout << "\n>>> 下一偵預測 (x1,y1,x2,y2) = ("
              << nb.x1 << ", " << nb.y1 << ", " << nb.x2 << ", " << nb.y2 << ")\n"
              << "    速度 vx=" << vx << " vy=" << vy << " vw=" << vw << " vh=" << vh << "\n";
    return 0;
}
*/