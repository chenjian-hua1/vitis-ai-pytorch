/*
 * main.cpp
 * ───────────────────────────────────────────────────────────────────────────
 * Custom anchor-free DFL YOLO inference — Vitis AI VART C++ API
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j$(nproc)
 *
 * Run:
 *   # On FPGA board (default)
 *   ./yolo_infer --xmodel model.xmodel --source dog.jpg
 *   ./yolo_infer --xmodel model.xmodel --source 0 --device dpu   # webcam
 *   ./yolo_infer --xmodel model.xmodel --source video.mp4
 *
 *   # On PC for testing (CPU simulation — slow, for logic validation only)
 *   ./yolo_infer --xmodel model.xmodel --source dog.jpg --device cpu
 *   ./yolo_infer --xmodel model.xmodel --source dog.jpg --device cpu --bench 10
 */

#include "yolo_custom.hpp"

// Vitis AI VART
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <sstream>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iomanip>


// ═══════════════════════════════════════════════════════════════════════════
//  DPURunner — VART wrapper
// ═══════════════════════════════════════════════════════════════════════════
class DPURunner {
public:
    explicit DPURunner(const std::string& xmodel_path,
                       RunDevice device = RunDevice::DPU)
        : device_(device)
    {
        const char* key = runner_key(device_);
        std::cout << "[INFO] Loading xmodel : " << xmodel_path << "\n"
                  << "[INFO] Backend        : "
                  << (device_ == RunDevice::DPU ? "DPU (FPGA)"
                                                : "CPU (simulation)")
                  << "  [runner key: \"" << key << "\"]\n";

        graph_    = xir::Graph::deserialize(xmodel_path);
        subgraph_ = get_dpu_subgraph(graph_.get());
        runner_   = vart::Runner::create_runner(subgraph_, key);

        // Get input / output tensors
        in_tensors_  = runner_->get_input_tensors();
        out_tensors_ = runner_->get_output_tensors();

        // Verify input shape [B, H, W, C]
        auto in_shape = in_tensors_[0]->get_shape();
        input_h_ = in_shape[1];
        input_w_ = in_shape[2];
        std::cout << "[INFO] Input size     : "
                  << input_w_ << "x" << input_h_ << "\n";

        // Print output tensors and sort by H*W descending (80x80 -> 40x40 -> 20x20)
        std::cout << "[INFO] Output tensors :\n";
        for (auto* t : out_tensors_) {
            auto s = t->get_shape();
            std::cout << "       " << t->get_name()
                      << "  [" << s[0] << "," << s[1]
                      << "," << s[2] << "," << s[3] << "]\n";
        }
        build_sorted_out_indices();
        alloc_buffers();
    }

    // ── Full inference pipeline ───────────────────────────────────────────
    std::vector<Detection> run(const cv::Mat& bgr)
    {
        // 1. Letterbox preprocess
        LetterboxInfo lb_info;
        cv::Mat canvas = letterbox(bgr, input_w_, input_h_, lb_info);

        // Copy into input buffer (uint8, NHWC)
        std::memcpy(in_buf_.data(), canvas.data,
                    input_h_ * input_w_ * 3 * sizeof(uint8_t));

        // 2. DPU execute
        auto in_tb  = dynamic_cast<vart::RunnerExt*>(runner_.get())
                      ->get_inputs();
        auto out_tb = dynamic_cast<vart::RunnerExt*>(runner_.get())
                      ->get_outputs();

        copy_to_tensor_buffer(in_buf_.data(), in_tb[0]);

        auto [job_id, status] = runner_->execute_async(in_tb, out_tb);
        runner_->wait(job_id, -1);

        // 3. Read outputs, dequantize int8 -> float, decode each scale
        std::vector<Detection> all_dets;
        for (int rank = 0; rank < 3; ++rank) {
            int orig_idx = sorted_out_idx_[rank];
            auto* t      = out_tensors_[orig_idx];
            auto  shape  = t->get_shape();   // [B, H, W, C]
            int   H = shape[1], W = shape[2], C = shape[3];
            int   n = H * W * C;

            // Dequantize: int8 * 2^(-fix_point) -> float
            std::vector<float> feat(n);
            dequantize_output(out_tb[orig_idx], feat.data(), n,
                              get_output_scale(t));

            int stride = cfg::STRIDES[rank];
            decode_scale(feat.data(), H, W, stride,
                         cfg::CONF_THRESH, all_dets);
        }

        // 4. NMS
        auto result = nms(all_dets, cfg::IOU_THRESH);

        // 5. Restore coordinates to original image space
        restore_coords(result, lb_info);
        return result;
    }

private:
    // ── Find DPU subgraph ─────────────────────────────────────────────────
    static xir::Subgraph* get_dpu_subgraph(xir::Graph* g)
    {
        auto* root    = g->get_root_subgraph();
        auto  children = root->toposort_child_subgraph();
        for (auto* c : children) {
            if (c->has_attr("device") &&
                c->get_attr<std::string>("device") == "DPU")
                return c;
        }
        throw std::runtime_error(
            "DPU subgraph not found — verify the .xmodel was compiled correctly");
    }

    // ── Sort output tensor indices by H*W descending ──────────────────────
    void build_sorted_out_indices()
    {
        int n = static_cast<int>(out_tensors_.size());
        sorted_out_idx_.resize(n);
        std::iota(sorted_out_idx_.begin(), sorted_out_idx_.end(), 0);
        std::sort(sorted_out_idx_.begin(), sorted_out_idx_.end(),
                  [&](int a, int b) {
                      auto sa = out_tensors_[a]->get_shape();
                      auto sb = out_tensors_[b]->get_shape();
                      return (sa[1] * sa[2]) > (sb[1] * sb[2]);  // descending
                  });
    }

    // ── Allocate input buffer ─────────────────────────────────────────────
    void alloc_buffers()
    {
        in_buf_.resize(input_h_ * input_w_ * 3);
    }

    // ── Get dequantization scale from tensor attribute ────────────────────
    // Vitis AI quantized output is int8; scale = 2^(-fix_point)
    static float get_output_scale(const xir::Tensor* t)
    {
        if (t->has_attr("fix_point")) {
            int fp = t->get_attr<int>("fix_point");
            return std::pow(2.f, -fp);
        }
        return 1.f;  // float model, no scaling needed
    }

    // ── Copy uint8 data into TensorBuffer ────────────────────────────────
    static void copy_to_tensor_buffer(const uint8_t* src,
                                      vart::TensorBuffer* tb)
    {
        uint64_t data_ptr = 0;
        size_t   size     = 0;
        auto idx = std::vector<int>(tb->get_tensor()->get_shape().size(), 0);
        std::tie(data_ptr, size) = tb->data(idx);
        std::memcpy(reinterpret_cast<void*>(data_ptr), src, size);
    }

    // ── Dequantize int8 TensorBuffer -> float array ───────────────────────
    static void dequantize_output(vart::TensorBuffer* tb,
                                   float* dst, int n, float scale)
    {
        uint64_t data_ptr = 0;
        size_t   size     = 0;
        auto idx = std::vector<int>(tb->get_tensor()->get_shape().size(), 0);
        std::tie(data_ptr, size) = tb->data(idx);

        const int8_t* src = reinterpret_cast<const int8_t*>(data_ptr);
        for (int i = 0; i < n; ++i)
            dst[i] = src[i] * scale;
    }

    // ── Member variables ──────────────────────────────────────────────────
    RunDevice                      device_;
    std::unique_ptr<xir::Graph>    graph_;
    xir::Subgraph*                 subgraph_  = nullptr;
    std::unique_ptr<vart::Runner>  runner_;

    std::vector<const xir::Tensor*> in_tensors_;
    std::vector<const xir::Tensor*> out_tensors_;
    std::vector<int>                sorted_out_idx_;

    int input_w_ = cfg::INPUT_W;
    int input_h_ = cfg::INPUT_H;

    std::vector<uint8_t> in_buf_;
};


// ═══════════════════════════════════════════════════════════════════════════
//  Run modes
// ═══════════════════════════════════════════════════════════════════════════

void run_image(DPURunner& model,
               const std::string& src,
               const std::string& out_path)
{
    cv::Mat frame = cv::imread(src);
    if (frame.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << src << "\n";
        return;
    }

    auto t0   = std::chrono::steady_clock::now();
    auto dets = model.run(frame);
    auto ms   = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();

    std::cout << "[INFO] Inference: " << ms << " ms  "
              << dets.size() << " object(s) detected\n";

    for (const auto& d : dets) {
        const std::string& name =
            (d.cls_id < (int)cfg::CLASS_NAMES.size())
            ? cfg::CLASS_NAMES[d.cls_id]
            : "cls" + std::to_string(d.cls_id);
        std::cout << "       " << std::left << std::setw(12) << name
                  << " conf=" << std::fixed << std::setprecision(3) << d.conf
                  << "  xyxy=[" << (int)d.x1 << "," << (int)d.y1
                  << "," << (int)d.x2 << "," << (int)d.y2 << "]\n";
    }

    draw_detections(frame, dets);
    cv::imwrite(out_path, frame);
    std::cout << "[INFO] Result saved to: " << out_path << "\n";
}


void run_video(DPURunner& model,
               const std::string& src,
               const std::string& out_path)
{
    cv::VideoCapture cap;
    if (src == "0") cap.open(0);
    else            cap.open(src);

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open source: " << src << "\n";
        return;
    }

    int    W   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int    H   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;

    std::cout << "[INFO] Source: " << W << "x" << H
              << " @ " << fps << " fps\n";

    cv::VideoWriter writer(out_path,
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           fps, {W, H});

    long long frame_cnt = 0;
    double    total_ms  = 0.0;
    cv::Mat   frame;

    while (cap.read(frame)) {
        auto t0   = std::chrono::steady_clock::now();
        auto dets = model.run(frame);
        total_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        ++frame_cnt;

        draw_detections(frame, dets);

        double avg_fps = 1000.0 / (total_ms / frame_cnt);
        cv::putText(frame,
                    "FPS:" + std::to_string(static_cast<int>(avg_fps)),
                    {8, 30}, cv::FONT_HERSHEY_SIMPLEX,
                    1.0, {0, 255, 0}, 2, cv::LINE_AA);
        writer.write(frame);

        if (frame_cnt % 30 == 0)
            std::cout << "[INFO] frame=" << frame_cnt
                      << "  avg_fps=" << avg_fps << "\n";
    }
    std::cout << "[INFO] Processed " << frame_cnt
              << " frames. Result saved to: " << out_path << "\n";
}


void run_bench(DPURunner& model,
               const std::string& src,
               int repeat)
{
    cv::Mat frame = cv::imread(src);
    if (frame.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << src << "\n";
        return;
    }

    std::cout << "[BENCH] Warming up (10 runs)...\n";
    for (int i = 0; i < 10; ++i) model.run(frame);

    std::cout << "[BENCH] Running benchmark (" << repeat << " iterations)...\n";
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) model.run(frame);
    double elapsed_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();

    std::cout << "[BENCH] Avg latency : "
              << elapsed_ms / repeat << " ms\n"
              << "[BENCH] Throughput  : "
              << repeat / (elapsed_ms / 1000.0) << " FPS\n";
}


// ═══════════════════════════════════════════════════════════════════════════
//  Argument parsing
// ═══════════════════════════════════════════════════════════════════════════
struct Args {
    std::string xmodel = "model.xmodel";
    std::string source = "test.jpg";
    std::string output = "result.jpg";
    float       conf   = cfg::CONF_THRESH;
    float       iou    = cfg::IOU_THRESH;
    int         bench  = 0;
    RunDevice   device = RunDevice::DPU;   // --device dpu|cpu
};

static Args parse_args(int argc, char* argv[])
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      ((k == "--xmodel" || k == "-m") && i+1 < argc) a.xmodel = argv[++i];
        else if ((k == "--source" || k == "-s") && i+1 < argc) a.source = argv[++i];
        else if ((k == "--output" || k == "-o") && i+1 < argc) a.output = argv[++i];
        else if  (k == "--conf"                 && i+1 < argc) a.conf   = std::stof(argv[++i]);
        else if  (k == "--iou"                  && i+1 < argc) a.iou    = std::stof(argv[++i]);
        else if  (k == "--bench"                && i+1 < argc) a.bench  = std::stoi(argv[++i]);
        else if  (k == "--device" || k == "-d") {
            if (i+1 >= argc) {
                std::cerr << "[ERROR] --device requires an argument: dpu or cpu\n";
                std::exit(1);
            }
            a.device = parse_device(argv[++i]);
        }
        else if  (k == "--help" || k == "-h") {
            std::cout <<
                "Usage: yolo_infer [options]\n"
                "  --xmodel  <path>       Path to .xmodel file        (default: model.xmodel)\n"
                "  --source  <path>       Image / video / camera      (default: test.jpg)\n"
                "  --output  <path>       Output path                 (default: result.jpg)\n"
                "  --conf    <float>      Confidence threshold         (default: 0.25)\n"
                "  --iou     <float>      NMS IoU threshold            (default: 0.45)\n"
                "  --bench   <int>        Benchmark iterations, 0=off  (default: 0)\n"
                "  --device  <dpu|cpu>    Backend: dpu=FPGA, cpu=sim   (default: dpu)\n"
                "\n"
                "Examples:\n"
                "  On FPGA board : ./yolo_infer --xmodel m.xmodel --source img.jpg\n"
                "  On PC (test)  : ./yolo_infer --xmodel m.xmodel --source img.jpg --device cpu\n";
            std::exit(0);
        }
    }
    return a;
}


// ═══════════════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[])
{
    Args args = parse_args(argc, argv);

    DPURunner model(args.xmodel, args.device);

    if (args.bench > 0) {
        run_bench(model, args.source, args.bench);
    }
    else {
        // Determine whether source is a video or image
        bool is_video = (args.source == "0");
        if (!is_video) {
            std::string ext = args.source.substr(
                args.source.find_last_of('.') + 1);
            for (auto& c : ext) c = static_cast<char>(std::tolower(c));
            is_video = (ext == "mp4" || ext == "avi" ||
                        ext == "mov" || ext == "mkv");
        }

        if (is_video)
            run_video(model, args.source, args.output);
        else
            run_image(model, args.source, args.output);
    }

    return 0;
}