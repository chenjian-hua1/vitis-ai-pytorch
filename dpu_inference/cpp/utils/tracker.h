// kalman_box_tracker.h
// Header-only：YOLO(NMS 後) bounding box 的等速模型 Kalman filter，
// 用來預測物件下一偵位置。零外部依賴，#include 即可使用。
//
//   #include "kalman_box_tracker.h"
//   using namespace kbt;
//   KalmanBoxTracker t(toCenter({100,100,150,200}));
//   CxCyWH pred = t.predict();      // 預測這一偵
//   t.update(toCenter(detBox));     // 拿 YOLO 偵測校正
//
// 注意：因為是 header-only，所有自由函式都標 inline，
// 多個 .cpp 同時 include 也不會有 multiple definition。

#ifndef KALMAN_BOX_TRACKER_H
#define KALMAN_BOX_TRACKER_H

#include <vector>
#include <cmath>
#include <utility>

namespace kbt {

// ======================= 極簡矩陣工具 =======================
struct Mat {
    int r = 0, c = 0;
    std::vector<double> d;
    Mat() {}
    Mat(int r, int c) : r(r), c(c), d(static_cast<size_t>(r) * c, 0.0) {}
    double&       operator()(int i, int j)       { return d[i * c + j]; }
    double        operator()(int i, int j) const { return d[i * c + j]; }
    static Mat I(int n) { Mat m(n, n); for (int i = 0; i < n; ++i) m(i, i) = 1.0; return m; }
};

inline Mat operator*(const Mat& a, const Mat& b) {
    Mat o(a.r, b.c);
    for (int i = 0; i < a.r; ++i)
        for (int k = 0; k < a.c; ++k) {
            double v = a(i, k);
            if (v == 0.0) continue;
            for (int j = 0; j < b.c; ++j) o(i, j) += v * b(k, j);
        }
    return o;
}
inline Mat operator+(const Mat& a, const Mat& b) {
    Mat o(a.r, a.c);
    for (size_t i = 0; i < a.d.size(); ++i) o.d[i] = a.d[i] + b.d[i];
    return o;
}
inline Mat operator-(const Mat& a, const Mat& b) {
    Mat o(a.r, a.c);
    for (size_t i = 0; i < a.d.size(); ++i) o.d[i] = a.d[i] - b.d[i];
    return o;
}
inline Mat transpose(const Mat& a) {
    Mat o(a.c, a.r);
    for (int i = 0; i < a.r; ++i)
        for (int j = 0; j < a.c; ++j) o(j, i) = a(i, j);
    return o;
}
// Gauss-Jordan 求反矩陣 (KF 只會對 4x4 的 S 求逆)
inline Mat inverse(Mat a) {
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

// ======================= bbox 格式 =======================
struct Box    { double x1, y1, x2, y2; };   // YOLO 常見輸出
struct CxCyWH { double cx, cy, w, h; };

inline CxCyWH toCenter(const Box& b) {
    double w = b.x2 - b.x1, h = b.y2 - b.y1;
    return { b.x1 + w / 2.0, b.y1 + h / 2.0, w, h };
}
inline Box toXYXY(const CxCyWH& c) {
    return { c.cx - c.w / 2.0, c.cy - c.h / 2.0,
             c.cx + c.w / 2.0, c.cy + c.h / 2.0 };
}

// ======================= Kalman Box Tracker =======================
// 狀態 x(8): [cx, cy, w, h, vx, vy, vw, vh]，量測 z(4): [cx, cy, w, h]
class KalmanBoxTracker {
public:
    // 用第一次偵測框初始化。dt = 兩偵間隔(用「偵」為單位時填 1)。
    explicit KalmanBoxTracker(const CxCyWH& init, double dt = 1.0) {
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

    // 預測步：狀態往前推一偵，回傳預測框。每偵拿到新偵測「之前」呼叫。
    CxCyWH predict() {
        x_ = F_ * x_;
        P_ = F_ * P_ * transpose(F_) + Q_;
        return currentBox();
    }

    // 更新步：拿到當偵 YOLO 偵測框後校正。
    void update(const CxCyWH& meas) {
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

    CxCyWH currentBox() const { return { x_(0,0), x_(1,0), x_(2,0), x_(3,0) }; }
    void velocity(double& vx, double& vy, double& vw, double& vh) const {
        vx = x_(4,0); vy = x_(5,0); vw = x_(6,0); vh = x_(7,0);
    }

    // 可選：允許外部微調雜訊。gain>1 更相信量測(靈敏)，<1 更平滑。
    void setMeasurementNoise(double pos, double size) {
        R_(0,0) = R_(1,1) = pos; R_(2,2) = R_(3,3) = size;
    }
    void setProcessNoise(double pos, double vel) {
        for (int i = 0; i < 4; ++i) Q_(i,i) = pos;
        for (int i = 4; i < 8; ++i) Q_(i,i) = vel;
    }

private:
    Mat x_, P_, F_, H_, Q_, R_;
};

} // namespace kbt

#endif // KALMAN_BOX_TRACKER_H