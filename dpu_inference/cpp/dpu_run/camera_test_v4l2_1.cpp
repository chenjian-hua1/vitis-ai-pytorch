// test_busyloop_check.cpp
//
// 用途：
//   1. 確認 cap.read() 是否 busy-wait（每次呼叫耗時是否穩定 ~33ms，
//      還是忽快忽慢、甚至接近 0ms 狂跑）
//   2. 量測 MJPEG → BGR888 解碼的純解碼耗時
//      （透過比較「raw read」vs「retrieve+decode」分離兩階段）
//
// 編譯：
//   g++ test_busyloop_check.cpp -o test_busyloop_check \
//       $(pkg-config --cflags --libs opencv4) -std=c++17
//
// 執行：
//   ./test_busyloop_check [camera_index] [num_frames]
//
// 輸出說明：
//   - 逐幀印出 read() 耗時，可觀察分佈是否穩定
//   - 統計 min/max/avg/stddev，stddev 過大代表不穩定（可能有 busy-wait 或 buffer 累積問題）
//   - grab() vs retrieve() 分離測試：
//       grab()    : 只觸發硬體擷取一幀到內部 buffer（不解碼）
//       retrieve(): 把 buffer 中的資料解碼成 cv::Mat (BGR888)
//     兩者相加 ≈ read()，藉此分離出「等待硬體」vs「解碼」各佔多少時間

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <limits>

static double time_now()
{
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
               steady_clock::now().time_since_epoch())
        .count();
}

struct Stats {
    double avg = 0, mn = 1e18, mx = 0, stddev = 0;
};

Stats computeStats(const std::vector<double>& v)
{
    Stats s;
    if (v.empty()) return s;
    double sum = 0;
    for (double x : v) {
        sum += x;
        s.mn = std::min(s.mn, x);
        s.mx = std::max(s.mx, x);
    }
    s.avg = sum / v.size();

    double sq = 0;
    for (double x : v) sq += (x - s.avg) * (x - s.avg);
    s.stddev = std::sqrt(sq / v.size());
    return s;
}

void printStats(const std::string& label, const Stats& s)
{
    std::cout << std::setw(18) << label << " : "
              << "avg=" << std::fixed << std::setprecision(2) << std::setw(7) << s.avg << "ms"
              << "  min=" << std::setw(7) << s.mn << "ms"
              << "  max=" << std::setw(7) << s.mx << "ms"
              << "  stddev=" << std::setw(6) << s.stddev << "ms"
              << std::endl;
}

int main(int argc, char* argv[])
{
    int camIndex   = (argc > 1) ? std::atoi(argv[1]) : 0;
    int numFrames  = (argc > 2) ? std::atoi(argv[2]) : 100;

    std::cout << "=== Busy-wait 檢測 + MJPEG解碼耗時測試 ===" << std::endl;
    std::cout << "camera=" << camIndex << "  frames=" << numFrames << "\n" << std::endl;

    cv::VideoCapture cap(camIndex, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 2);   // 只保留 1 個 buffer,逼近「拿最新幀」
    if (!cap.isOpened()) {
        std::cerr << "無法開啟攝影機" << std::endl;
        return -1;
    }

    // cap.set(cv::CAP_PROP_FRAME_WIDTH,  std::numeric_limits<int>::max());
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, std::numeric_limits<int>::max());
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS,          std::numeric_limits<int>::max());

    int actualW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int actualH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actualFps = cap.get(cv::CAP_PROP_FPS);
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    char fourccStr[5] = {
        static_cast<char>(fourcc & 0xFF),
        static_cast<char>((fourcc >> 8) & 0xFF),
        static_cast<char>((fourcc >> 16) & 0xFF),
        static_cast<char>((fourcc >> 24) & 0xFF),
        '\0'
    };

    std::cout << "解析度: " << actualW << "x" << actualH
              << "  driver fps=" << actualFps
              << "  fourcc=" << fourccStr << "\n" << std::endl;

    cv::Mat frame;

    // ── warm-up ──────────────────────────────────────────────
    for (int i = 0; i < 5; ++i) cap.read(frame);

    // ── 測試 1: read() 整體耗時分佈 ──────────────────────────
    std::vector<double> readTimes;
    readTimes.reserve(numFrames);

    std::cout << "[測試 1] read() 整體耗時 (前10幀逐一列出，觀察是否穩定/規律):" << std::endl;
    for (int i = 0; i < numFrames; ++i) {
        double t0 = time_now();
        bool ok = cap.read(frame);
        double t1 = time_now();
        if (!ok || frame.empty()) { --i; continue; }

        double dt = t1 - t0;
        readTimes.push_back(dt);

        if (i < 10) {
            std::cout << "  frame " << std::setw(2) << i
                      << " : read()=" << std::fixed << std::setprecision(2) << dt << "ms"
                      << std::endl;
        }
    }
    std::cout << std::endl;

    // ── 測試 2: grab() vs retrieve() 分離 ────────────────────
    std::vector<double> grabTimes, retrieveTimes;
    grabTimes.reserve(numFrames);
    retrieveTimes.reserve(numFrames);

    std::cout << "[測試 2] grab() / retrieve() 分離 (前10幀逐一列出):" << std::endl;
    for (int i = 0; i < numFrames; ++i) {
        double t0 = time_now();
        bool grabOk = cap.grab();
        double t1 = time_now();

        if (!grabOk) { --i; continue; }

        bool retOk = cap.retrieve(frame);
        double t2 = time_now();

        if (!retOk || frame.empty()) { --i; continue; }

        double grabMs     = t1 - t0;
        double retrieveMs = t2 - t1;
        grabTimes.push_back(grabMs);
        retrieveTimes.push_back(retrieveMs);

        if (i < 10) {
            std::cout << "  frame " << std::setw(2) << i
                      << " : grab()=" << std::fixed << std::setprecision(2) << std::setw(6) << grabMs << "ms"
                      << "  retrieve()=" << std::setw(6) << retrieveMs << "ms"
                      << "  total=" << (grabMs + retrieveMs) << "ms"
                      << std::endl;
        }
    }
    std::cout << std::endl;

    cap.release();

    // ── 統計總結 ──────────────────────────────────────────────
    std::cout << "===================== 統計總結 =====================" << std::endl;
    printStats("read() (整體)", computeStats(readTimes));
    printStats("grab() (擷取)", computeStats(grabTimes));
    printStats("retrieve()(解碼)", computeStats(retrieveTimes));

    Stats grabS = computeStats(grabTimes);
    Stats retS  = computeStats(retrieveTimes);
    std::cout << std::setw(18) << "grab+retrieve" << " : avg="
              << std::fixed << std::setprecision(2)
              << (grabS.avg + retS.avg) << "ms" << std::endl;

    // ── 判讀建議 ──────────────────────────────────────────────
    std::cout << "\n===================== 判讀建議 =====================" << std::endl;

    Stats readS = computeStats(readTimes);

    // Busy-wait 判斷
    if (readS.avg < 5.0) {
        std::cout << "[!] read() 平均耗時 < 5ms，遠低於 1000/" << actualFps
                  << "=" << (1000.0/actualFps) << "ms" << std::endl;
        std::cout << "    → 強烈懷疑是 busy-wait 或驅動有自己的緩衝佇列在快速吐舊幀。" << std::endl;
    } else if (readS.stddev > readS.avg * 0.5) {
        std::cout << "[!] read() 耗時波動很大 (stddev=" << readS.stddev
                  << "ms, avg=" << readS.avg << "ms)" << std::endl;
        std::cout << "    → 可能不是穩定的 busy-wait，但時序不規律，"
                  << "有些幀等很久、有些幀立刻拿到（buffer堆積後一次吐出）。" << std::endl;
    } else {
        std::cout << "[OK] read() 耗時穩定且接近 1000/fps ("
                  << (1000.0/actualFps) << "ms)，行為正常（阻塞等待硬體）。" << std::endl;
    }


    std::cout << cv::getBuildInformation() << std::endl;
    // 解碼成本判斷
    std::cout << std::endl;
    if (retS.avg > grabS.avg) {
        std::cout << "[資訊] retrieve()(解碼) 耗時 (" << retS.avg
                  << "ms) > grab()(等硬體) 耗時 (" << grabS.avg << "ms)" << std::endl;
        std::cout << "    → MJPEG→BGR888 解碼是主要 CPU 成本，"
                  << "降低解析度或改用 YUYV 應該有顯著效果。" << std::endl;
    } else {
        std::cout << "[資訊] grab()(等硬體) 耗時 (" << grabS.avg
                  << "ms) >= retrieve()(解碼) 耗時 (" << retS.avg << "ms)" << std::endl;
        std::cout << "    → 主要時間花在等待硬體出圖，受限於攝影機 fps 上限，"
                  << "降解析度對速度提升有限（但仍可能降低CPU負載）。" << std::endl;
    }

    

    return 0;
}