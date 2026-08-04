// eval.cpp --------------------------------------------------------------------
// 離線評測用的 run_eval()：與 run_camera() 同一套前處理 / 推理 / 後處理 / 追蹤，
// 只是輸入從 Camera 換成 Python 產生的 image_manifest.tsv，
// 並在 tracker.update() 之後把結果寫成 TrackEval / pycocotools 吃得下的檔案。
//
// 用法:
//   ./eval model.onnx --manifest trackeval_io/image_manifest.tsv \
//                --out-dir trackeval_io --conf 0.2 --iou 0.5 --fps 30 \
//                --cls-map 1,2,3,6
//
// --cls-map 是「模型 class_id -> val.json category_id」的對照表，
// 由 notebook 讀 data.yaml 後自動產生並帶進來，不必改 C++ 重編。
// 填 -1 代表該 class 在 val.json 沒有對應類別，寫檔時會跳過。
// -----------------------------------------------------------------------------
#include <opencv2/opencv.hpp>

#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "eval.h"
// 以下是你原本的 header，路徑依專案調整
#include "modelrunner.h"
#include "tracker.h"
#include "stream.h"
#include "drawer.h"
#include "yolopproc.h"
#include "camera.h"
#include "preproc.h"


static volatile std::sig_atomic_t g_running = 1;
static void signalHandler(int) { g_running = 0; }

using Clock = std::chrono::high_resolution_clock;

double time_now() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}


// =============================================================================
void run_eval(std::string onnx_path,
              std::string manifest_path,
              std::string out_dir   = "trackeval_io",
              double      conf_th   = 0.2,
              double      iou_th    = 0.5,
              double      seq_fps   = 30.0,
              std::vector<int> cls_to_coco_cat = {},
              std::string video_out = "")
{
    std::signal(SIGINT, signalHandler);

    // ─────────────────────────────────────────────────────────────
    //  0. 讀取影像清單（已依 video_id / frame_index 排序）
    // ─────────────────────────────────────────────────────────────
    std::vector<evalio::ManifestItem> items = evalio::load_manifest(manifest_path);
    if (items.empty()) {
        std::cerr << "manifest 是空的: " << manifest_path << "\n";
        return;
    }

    // ─────────────────────────────────────────────────────────────
    //  1. 載入 ONNX_MODEL
    // ─────────────────────────────────────────────────────────────
    OnnxInferenceEngine engine(
        onnx_path,
        [](Ort::SessionOptions& opts) {
            // 離線評測不趕即時性，但要跑八千多張，開多執行緒會快很多
            // opts.SetIntraOpNumThreads(4);
            // opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            (void)opts;
        });

    const int in_w = static_cast<int>(engine.in_w());
    const int in_h = static_cast<int>(engine.in_h());
    std::cout << "模型輸入: " << in_w << "x" << in_h
              << "  輸出數量: " << engine.num_outputs() << "\n";

    // ─────────────────────────────────────────────────────────────
    //  2. 根據模型輸出決定後處理參數
    // ─────────────────────────────────────────────────────────────
    const int ch = 16;

    {
        cv::Mat dummy = cv::Mat::zeros(in_h, in_w, CV_32FC3);
        engine.run(dummy);
    }

    const std::vector<cv::Mat>& fmaps = engine.output_mats();
    const int no = fmaps[0].size[1];
    const int nc = no - 4 * ch;

    if (nc <= 0) {
        std::cerr << "模型輸出 channel 數 (" << no
                  << ") 與 YOLO DFL 頭假設不符 (ch=" << ch << ")\n";
        return;
    }
    std::cout << "推導出的 nc = " << nc << "\n";

    YOLOPostProcessor yolo_pp(1, in_h, in_w, nc, ch);

    // ─────────────────────────────────────────────────────────────
    //  3. 前處理 / 推理 / 後處理 的中間 buffer
    // ─────────────────────────────────────────────────────────────
    ResizeResult resize_result;
    cv::Mat norm_img;

    const std::vector<DetectionBatch>* nms_result = nullptr;

    // ─────────────────────────────────────────────────────────────
    //  4. 結果輸出設定
    // ─────────────────────────────────────────────────────────────
    // 模型 class_id -> val.json 的 category_id，由 --cls-map 帶進來。
    // 沒給的話退回這份預設值（SeaDronesSee：
    // 1=swimmer, 2=swimmer with life jacket, 3=boat, 6=life jacket）
    if (cls_to_coco_cat.empty()) {
        cls_to_coco_cat = {1, 2, 3, 6};
        std::cerr << "⚠️ 沒帶 --cls-map，使用內建預設值\n";
    }

    std::cout << "class 對照表 (模型 cls -> category_id): ";
    for (size_t k = 0; k < cls_to_coco_cat.size(); ++k)
        std::cout << k << "->" << cls_to_coco_cat[k]
                  << (k + 1 < cls_to_coco_cat.size() ? ", " : "\n");

    if (static_cast<int>(cls_to_coco_cat.size()) != nc)
        std::cerr << "⚠️ --cls-map 有 " << cls_to_coco_cat.size()
                  << " 項，但模型 nc = " << nc << "，請確認對應表\n";

    evalio::WriterOptions wopt;
    wopt.min_tracklet_len = 0;      // 想過濾短軌跡再調高
    wopt.min_score        = 0.0f;   // 門檻交給 conf_th 控，這裡不再過濾
    wopt.clamp_to_image   = true;

    evalio::TrackResultWriter writer(out_dir + "/predict.txt",
                                     out_dir + "/track_coco.json",
                                     out_dir + "/det_coco.json",
                                     cls_to_coco_cat, wopt);

    // ─────────────────────────────────────────────────────────────
    //  5. Tracking Setting
    // ─────────────────────────────────────────────────────────────
    bytetrack::Params p;
    p.max_lost_seconds = 2.;
    p.class_aware      = true;   // 只讓同 class 配對
    std::unique_ptr<bytetrack::BYTETracker> tracker(new bytetrack::BYTETracker(p));

    std::vector<bytetrack::Box> boxes;

    // ─────────────────────────────────────────────────────────────
    //  6. 主迴圈 : 逐張讀圖進行推理
    // ─────────────────────────────────────────────────────────────
    const int PROGRESS_EVERY = 5;   // 每幾幀更新一次進度

    cv::Mat frame, rgb_frame;
    cv::VideoWriter vw;
    long long frameCount = 0;
    int       last_video_id = -1;
    const double t_start = time_now();

    for (size_t i = 0; i < items.size() && g_running; ++i) {
        const evalio::ManifestItem& it = items[i];

        // ── 換影片就重建追蹤器（否則 track id 會跨片延續，AssA 會崩）──────
        if (it.video_id != last_video_id) {
            tracker.reset(new bytetrack::BYTETracker(p));
            last_video_id = it.video_id;
            std::cout << "\n[video " << it.video_id << "] start" << std::endl;
        }

        frame = cv::imread(it.path, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "\n⚠️ 讀不到影像，略過: " << it.path << "\n";
            continue;
        }

        // ── 時間戳 ─────────────────────────────────────────────────────
        // ★ 這裡「不能」沿用 run_camera 的 steady_clock。
        //   離線推論一幀可能要 0.3 秒，用真實時間的話 max_lost_seconds = 2s
        //   只等得到 6~7 幀，追蹤行為會跟實機完全不同。
        //   換算成影片內的時間軸才對；影片中若有抽幀斷點
        //   （val.json 的 video 15 就有一段 753 幀的空隙）也會被正確反映。
        const double t_cap = static_cast<double>(it.frame_index) / seq_fps;

        // ── 前處理 ──────────────────────────────────────────────────────
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        resize(rgb_frame, in_w, resize_result);
        norm(resize_result.img, norm_img);

        // ── ONNX 推理 ──────────────────────────────────────────────────
        engine.run(norm_img);

        // ── 後處理 ──────────────────────────────────────────────────────
        nms_result = &yolo_pp.process(engine.output_mats(), conf_th, iou_th);

        // ── Track ──────────────────────────────────────────────────────
        scale_detections((*nms_result)[0], boxes, resize_result,
                         cv::Size(frame.cols, frame.rows));

        // 追蹤前的偵測結果 → 算「偵測 mAP」
        writer.add_detections(it.image_id, boxes, frame.cols, frame.rows);

        const std::vector<bytetrack::Track>& tracks = tracker->update(boxes, t_cap);

        // ★★★ 要補在 update 後面的就是這行 ★★★
        writer.add_tracks(it.image_id, tracks, frame.cols, frame.rows);

        // ── 計數 / 進度 ────────────────────────────────────────────────
        ++frameCount;
        if (frameCount % PROGRESS_EVERY == 0 || i + 1 == items.size()) {
            const double elapsed = (time_now() - t_start) / 1000.0;   // time_now() 回傳毫秒
            const double fps     = elapsed > 0 ? frameCount / elapsed : 0.0;
            // PROGRESS 這個 token 是給 notebook 解析畫進度條用的，格式別亂改。
            // 用 \r 讓終端機原地更新；輸出接到 pipe 時 notebook 會自己切開處理。
            std::printf("\rPROGRESS %zu/%zu det=%d trk=%zu fps=%.2f   ",
                        i + 1, items.size(), (*nms_result)[0].size(),
                        tracks.size(), fps);
            std::fflush(stdout);   // ★ 接到 pipe 時是全緩衝，沒 flush 就看不到即時進度
        }

        // ── 繪圖（可選，只是拿來人眼確認追蹤是否正常）────────────────────
        if (!video_out.empty()) {
            cv::Mat drawn, small;
            draw_tracking(frame, drawn, tracks, 0.0);
            cv::resize(drawn, small,
                       cv::Size(1280, 1280 * drawn.rows / drawn.cols));
            if (!vw.isOpened())
                vw.open(video_out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        seq_fps, small.size());
            vw.write(small);
        }
    }

    std::cout << "\n";
    if (vw.isOpened()) {
        vw.release();
        std::cout << "🎬 追蹤影片: " << video_out << "\n";
    }

    // ── 收尾：一次把結果寫出去 ─────────────────────────────────────────
    if (!writer.save()) {
        std::cerr << "❌ 結果寫檔失敗，確認 " << out_dir << " 是否可寫\n";
        return;
    }
    std::cout << "共處理 " << frameCount << " 幀"
              << "，track rows = " << writer.track_rows()
              << "，det rows = "   << writer.det_rows() << "\n";
}

// =============================================================================
// "1,2,3,6" -> {1, 2, 3, 6}
static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::string tok;
    std::istringstream ss(s);
    while (std::getline(ss, tok, ',')) {
        // 去掉可能的空白 / 大括號，讓 "{1, 2, 3, 6}" 也能吃
        std::string t;
        for (char c : tok)
            if (!std::isspace(static_cast<unsigned char>(c)) && c != '{' && c != '}')
                t += c;
        if (!t.empty()) out.push_back(std::stoi(t));
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0]
                  << " model.onnx --manifest image_manifest.tsv"
                     " [--out-dir trackeval_io] [--conf 0.2] [--iou 0.5]"
                     " [--fps 30] [--cls-map 1,2,3,6] [--video out.mp4]\n";
        return 1;
    }

    std::string onnx     = argv[1];
    std::string manifest = "trackeval_io/image_manifest.tsv";
    std::string out_dir  = "trackeval_io";
    std::string video;
    std::vector<int> cls_map;
    double conf = 0.2, iou = 0.5, fps = 30.0;

    for (int i = 2; i + 1 < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--manifest") manifest = argv[++i];
        else if (k == "--out-dir")  out_dir  = argv[++i];
        else if (k == "--conf")     conf     = std::stod(argv[++i]);
        else if (k == "--iou")      iou      = std::stod(argv[++i]);
        else if (k == "--fps")      fps      = std::stod(argv[++i]);
        else if (k == "--cls-map")  cls_map  = parse_int_list(argv[++i]);
        else if (k == "--video")    video    = argv[++i];
    }

    run_eval(onnx, manifest, out_dir, conf, iou, fps, cls_map, video);
    return 0;
}
