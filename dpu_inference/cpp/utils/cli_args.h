// ============================================================================
//  cli_args.hpp — 無外部依賴的命令列參數解析
//  支援 "--key value" 與 "--key=value" 兩種寫法、短選項、以及 positional model path
// ============================================================================
#pragma once

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
//  參數集合（預設值就是原本 main 裡的固定數字）
// ─────────────────────────────────────────────────────────────────────────────
struct CliArgs {
    // 必要
    std::string model_path;

    // Camera: {index, width, height, fps, fourcc}
    int    cam_index   = 0;
    int    cam_width   = 1920; // 640
    int    cam_height  = 1080; // 480
    double cam_fps     = 60.0;
    // MJPG / YUYV / UYVY / H264
    std::string cam_fourcc = "UYVY";   // 空字串 = 不設定，沿用驅動預設

    // Benchmark
    int    warmup      = 10;
    int    iter        = 1000;
    float  conf        = 0.4f;
    float  iou         = 0.5f;
    bool   track       = true;
    bool   draw        = true;
    bool   stream      = true;

    // Stream: {ip, port, width, height, fps, quality}
    std::string st_ip  = "192.168.1.100";
    int    st_port     = 5000;
    int    st_width    = 640;
    int    st_height   = 640;
    double st_fps      = 60.0;
    int    st_quality  = 60;

    // 輸出
    std::string save   = "benchmark_last_frame.jpg";
};

// ─────────────────────────────────────────────────────────────────────────────
//  內部小工具
// ─────────────────────────────────────────────────────────────────────────────
namespace cli_detail {

inline bool to_int(const std::string& s, int& out, const char* opt) {
    if (s.empty()) { std::cerr << "錯誤: " << opt << " 缺少數值\n"; return false; }
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') {
        std::cerr << "錯誤: " << opt << " 需要整數，收到 \"" << s << "\"\n";
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

inline bool to_double(const std::string& s, double& out, const char* opt) {
    if (s.empty()) { std::cerr << "錯誤: " << opt << " 缺少數值\n"; return false; }
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0') {
        std::cerr << "錯誤: " << opt << " 需要浮點數，收到 \"" << s << "\"\n";
        return false;
    }
    out = v;
    return true;
}

// FOURCC 必須剛好 4 個可見 ASCII 字元（例如 MJPG / YUYV / H264）。
// 空字串代表「不主動設定」。順便統一轉成大寫。
inline bool to_fourcc(std::string& s, const char* opt) {
    if (s.empty()) return true;
    if (s.size() != 4) {
        std::cerr << "錯誤: " << opt << " 需要剛好 4 個字元（例如 MJPG、YUYV），收到 \""
                  << s << "\"\n";
        return false;
    }
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 32 || c > 126) {
            std::cerr << "錯誤: " << opt << " 只能包含可見的 ASCII 字元\n";
            return false;
        }
        s[i] = static_cast<char>(std::toupper(c));
    }
    return true;
}

inline bool file_readable(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    return f.good();
}

} // namespace cli_detail

// ─────────────────────────────────────────────────────────────────────────────
//  使用說明
// ─────────────────────────────────────────────────────────────────────────────
inline void print_usage(const char* prog) {
    const CliArgs d;   // 用預設值印出來，改預設值不必改這段文字
    std::cout
    << "\n用法: " << prog << " <xmodel_path> [選項]\n"
    << "\n必要參數:\n"
    << "  <xmodel_path>            量化後的 .xmodel 路徑（也可用 -m/--model 指定）\n"
    << "\nCamera:\n"
    << "  --cam-index   <int>      相機索引            (預設 " << d.cam_index  << ")\n"
    << "  --cam-width   <int>      擷取寬度            (預設 " << d.cam_width  << ")\n"
    << "  --cam-height  <int>      擷取高度            (預設 " << d.cam_height << ")\n"
    << "  --cam-fps     <float>    擷取 FPS            (預設 " << d.cam_fps    << ")\n"
    << "  --cam-fourcc  <4chars>   擷取像素格式        (預設 " << d.cam_fourcc << ")\n"
    << "                           常見: MJPG / YUYV / H264；傳空字串 \"\" 則不設定\n"
    << "\nBenchmark:\n"
    << "  -w, --warmup  <int>      暖機幀數            (預設 " << d.warmup << ")\n"
    << "  -n, --iter    <int>      統計幀數            (預設 " << d.iter   << ")\n"
    << "  -c, --conf    <float>    信心門檻            (預設 " << d.conf   << ")\n"
    << "  -i, --iou     <float>    NMS IoU 門檻        (預設 " << d.iou    << ")\n"
    << "      --track / --no-track     是否量測 ByteTrack (預設 "
                                     << (d.track ? "on" : "off") << ")\n"
    << "      --draw  / --no-draw      是否量測繪圖       (預設 "
                                     << (d.draw ? "on" : "off") << ")\n"
    << "      --stream / --no-stream   是否量測 RTP 送出  (預設 "
                                     << (d.stream ? "on" : "off") << ")\n"
    << "\nStream (--stream 才會用到):\n"
    << "  --ip          <string>   目標 IP             (預設 " << d.st_ip     << ")\n"
    << "  --port        <int>      目標 port           (預設 " << d.st_port   << ")\n"
    << "  --st-width    <int>      串流寬度            (預設 " << d.st_width  << ")\n"
    << "  --st-height   <int>      串流高度            (預設 " << d.st_height << ")\n"
    << "  --st-fps      <float>    串流 FPS            (預設 " << d.st_fps    << ")\n"
    << "  --quality     <int>      JPEG 品質 0-100     (預設 " << d.st_quality<< ")\n"
    << "\n其他:\n"
    << "  -o, --save    <path>     最後一幀存檔路徑     (預設 " << d.save << ")\n"
    << "                           傳空字串 \"\" 則不存檔\n"
    << "  -h, --help               顯示本說明\n"
    << "\n範例:\n"
    << "  " << prog << " ./model/yolo11n_int.xmodel\n"
    << "  " << prog << " ./model/yolo11n_int.xmodel -n 300 --cam-fps 30 --no-track\n"
    << "  " << prog << " ./model/yolo11n_int.xmodel --stream --ip 192.168.1.50 --port 5000\n"
    << std::endl;
}

// ─────────────────────────────────────────────────────────────────────────────
//  解析：成功回傳 true；--help 或參數錯誤回傳 false
// ─────────────────────────────────────────────────────────────────────────────
inline bool parse_args(int argc, char** argv, CliArgs& a) {
    using namespace cli_detail;

    for (int i = 1; i < argc; ++i) {
        std::string tok(argv[i]);

        // --key=value → 拆成 key / value
        std::string key = tok, inline_val;
        bool has_inline = false;
        if (tok.compare(0, 2, "--") == 0) {
            size_t eq = tok.find('=');
            if (eq != std::string::npos) {
                key        = tok.substr(0, eq);
                inline_val = tok.substr(eq + 1);
                has_inline = true;
            }
        }

        // 取下一個 token 當數值
        auto next_val = [&](std::string& out) -> bool {
            if (has_inline) { out = inline_val; return true; }
            if (i + 1 >= argc) {
                std::cerr << "錯誤: " << key << " 後面缺少數值\n";
                return false;
            }
            out = argv[++i];
            return true;
        };

        std::string v;

        if (key == "-h" || key == "--help") {
            return false;
        }
        // ── model ────────────────────────────────────────────────
        else if (key == "-m" || key == "--model") {
            if (!next_val(v)) return false;
            a.model_path = v;
        }
        // ── camera ───────────────────────────────────────────────
        else if (key == "--cam-index") {
            if (!next_val(v) || !to_int(v, a.cam_index, "--cam-index")) return false;
        }
        else if (key == "--cam-width") {
            if (!next_val(v) || !to_int(v, a.cam_width, "--cam-width")) return false;
        }
        else if (key == "--cam-height") {
            if (!next_val(v) || !to_int(v, a.cam_height, "--cam-height")) return false;
        }
        else if (key == "--cam-fps") {
            if (!next_val(v) || !to_double(v, a.cam_fps, "--cam-fps")) return false;
        }
        else if (key == "--cam-fourcc") {
            if (!next_val(v) || !to_fourcc(v, "--cam-fourcc")) return false;
            a.cam_fourcc = v;
        }
        // ── benchmark ────────────────────────────────────────────
        else if (key == "-w" || key == "--warmup") {
            if (!next_val(v) || !to_int(v, a.warmup, "--warmup")) return false;
        }
        else if (key == "-n" || key == "--iter") {
            if (!next_val(v) || !to_int(v, a.iter, "--iter")) return false;
        }
        else if (key == "-c" || key == "--conf") {
            double d;
            if (!next_val(v) || !to_double(v, d, "--conf")) return false;
            a.conf = static_cast<float>(d);
        }
        else if (key == "-i" || key == "--iou") {
            double d;
            if (!next_val(v) || !to_double(v, d, "--iou")) return false;
            a.iou = static_cast<float>(d);
        }
        else if (key == "--track")     { a.track  = true;  }
        else if (key == "--no-track")  { a.track  = false; }
        else if (key == "--draw")      { a.draw   = true;  }
        else if (key == "--no-draw")   { a.draw   = false; }
        else if (key == "--stream")    { a.stream = true;  }
        else if (key == "--no-stream") { a.stream = false; }
        // ── stream ───────────────────────────────────────────────
        else if (key == "--ip") {
            if (!next_val(v)) return false;
            a.st_ip = v;
        }
        else if (key == "--port") {
            if (!next_val(v) || !to_int(v, a.st_port, "--port")) return false;
        }
        else if (key == "--st-width") {
            if (!next_val(v) || !to_int(v, a.st_width, "--st-width")) return false;
        }
        else if (key == "--st-height") {
            if (!next_val(v) || !to_int(v, a.st_height, "--st-height")) return false;
        }
        else if (key == "--st-fps") {
            if (!next_val(v) || !to_double(v, a.st_fps, "--st-fps")) return false;
        }
        else if (key == "--quality") {
            if (!next_val(v) || !to_int(v, a.st_quality, "--quality")) return false;
        }
        // ── 其他 ─────────────────────────────────────────────────
        else if (key == "-o" || key == "--save") {
            if (!next_val(v)) return false;
            a.save = v;
        }
        // ── positional: 第一個非選項當 model path ────────────────
        else if (!tok.empty() && tok[0] != '-') {
            if (a.model_path.empty()) {
                a.model_path = tok;
            } else {
                std::cerr << "錯誤: 多餘的參數 \"" << tok << "\"\n";
                return false;
            }
        }
        else {
            std::cerr << "錯誤: 未知選項 \"" << tok << "\"\n";
            return false;
        }
    }

    // ── 驗證 ─────────────────────────────────────────────────────
    if (a.model_path.empty()) {
        std::cerr << "錯誤: 未指定 xmodel 路徑\n";
        return false;
    }
    if (!file_readable(a.model_path)) {
        std::cerr << "錯誤: 無法讀取 xmodel \"" << a.model_path << "\"\n";
        return false;
    }
    if (a.iter <= 0) {
        std::cerr << "錯誤: --iter 必須 > 0\n";
        return false;
    }
    if (a.warmup < 0) {
        std::cerr << "錯誤: --warmup 不可為負\n";
        return false;
    }
    if (a.conf < 0.0f || a.conf > 1.0f) {
        std::cerr << "錯誤: --conf 需在 0~1 之間\n";
        return false;
    }
    if (a.iou < 0.0f || a.iou > 1.0f) {
        std::cerr << "錯誤: --iou 需在 0~1 之間\n";
        return false;
    }
    if (a.cam_width <= 0 || a.cam_height <= 0 || a.cam_fps <= 0.0) {
        std::cerr << "錯誤: camera 寬/高/fps 必須為正數\n";
        return false;
    }
    if (a.stream) {
        if (a.st_port <= 0 || a.st_port > 65535) {
            std::cerr << "錯誤: --port 需在 1~65535 之間\n";
            return false;
        }
        if (a.st_quality < 0 || a.st_quality > 100) {
            std::cerr << "錯誤: --quality 需在 0~100 之間\n";
            return false;
        }
        if (a.st_width <= 0 || a.st_height <= 0 || a.st_fps <= 0.0) {
            std::cerr << "錯誤: stream 寬/高/fps 必須為正數\n";
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  印出實際生效的設定（跑 benchmark 前對一下很有用）
// ─────────────────────────────────────────────────────────────────────────────
inline void print_args(const CliArgs& a) {
    std::cout << "----- 設定 -----\n"
              << "xmodel   : " << a.model_path << "\n"
              << "camera   : index " << a.cam_index << "  "
              << a.cam_width << "x" << a.cam_height << " @" << a.cam_fps << "fps  "
              << "fourcc " << (a.cam_fourcc.empty() ? "(driver default)" : a.cam_fourcc)
              << "\n"
              << "bench    : warmup " << a.warmup << ", iter " << a.iter
              << ", conf " << a.conf << ", iou " << a.iou << "\n"
              << "stages   : track " << (a.track  ? "on" : "off")
              << ", draw "           << (a.draw   ? "on" : "off")
              << ", stream "         << (a.stream ? "on" : "off") << "\n";
    if (a.stream)
        std::cout << "stream   : " << a.st_ip << ":" << a.st_port << "  "
                  << a.st_width << "x" << a.st_height << " @" << a.st_fps
                  << "fps  q=" << a.st_quality << "\n";
    std::cout << "----------------" << std::endl;
}