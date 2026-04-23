#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace OnnxInference {

    struct VisionModelInfo {
        int64_t ch = 0;
        int64_t in_h = 0;
        int64_t in_w = 0;

        std::vector<std::vector<int64_t>> output_shapes;
        std::vector<std::vector<float>> output_list; 
        std::vector<float> input_tensor_values;

        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        std::vector<Ort::AllocatedStringPtr> allocated_strings; 
    };

    inline size_t GetTotalElements(const std::vector<int64_t>& shape) {
        size_t total = 1;
        for (auto dim : shape) total *= (dim < 0) ? 1 : dim;
        return total;
    }

    // --- Function 1: 初始化 ---
    inline VisionModelInfo InitializeModelInfo(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
        VisionModelInfo info;

        auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
        info.input_names.push_back(input_name_ptr.get());
        info.allocated_strings.push_back(std::move(input_name_ptr));

        auto in_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (in_shape.size() == 4) {
            info.ch = in_shape[1];
            info.in_h = in_shape[2];
            info.in_w = in_shape[3];
        } else {
            throw std::runtime_error("Input shape is not 4D (NCHW).");
        }

        // 確保模型真的是 3 通道，才配得上我們的 Vec3f
        assert(info.ch == 3 && "Model input must be 3 channels for Vec3f!");
        info.input_tensor_values.resize(info.ch * info.in_h * info.in_w, 0.0f);

        size_t num_outputs = session.GetOutputCount();
        info.output_list.resize(num_outputs);

        for (size_t i = 0; i < num_outputs; i++) {
            auto out_name_ptr = session.GetOutputNameAllocated(i, allocator);
            info.output_names.push_back(out_name_ptr.get());
            info.allocated_strings.push_back(std::move(out_name_ptr));

            auto out_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            for (auto& dim : out_shape) { if (dim == -1) dim = 1; } 
            info.output_shapes.push_back(out_shape);

            size_t tensor_size = GetTotalElements(out_shape);
            info.output_list[i].resize(tensor_size, 0.0f);
        }

        return info;
    }

    // --- Function 2: 將 HWC 轉換為 NCHW ---
    // 直接接收強型別 cv::Mat_<cv::Vec3f>
    inline void ConvertHwcToNchw(const cv::Mat_<cv::Vec3f>& src, std::vector<float>& dst, int in_h, int in_w) {
        std::vector<cv::Mat> ch_channels;
        // 因為型別鎖死了 Vec3f，通道數固定為 3
        for (int i = 0; i < 3; ++i) {
            ch_channels.push_back(cv::Mat(in_h, in_w, CV_32FC1, 
                                          dst.data() + i * in_h * in_w));
        }
        cv::split(src, ch_channels); 
    }

    // --- Function 3: 推理核心邏輯 ---
    // 參數直接鎖死為 cv::Mat_<cv::Vec3f>
    inline void RunInference(Ort::Session& session, VisionModelInfo& info, const cv::Mat_<cv::Vec3f>& input_img) {
        
        // 1. 記憶體防呆：現在只需要檢查長寬，因為型別與通道已經被編譯器保證了！
        assert(input_img.cols == info.in_w && input_img.rows == info.in_h && 
               "Memory Error: Image size does not match pre-allocated buffer!");

        // 2. 執行 HWC -> NCHW 轉換
        ConvertHwcToNchw(input_img, info.input_tensor_values, info.in_h, info.in_w);

        // 3. 建立 Input Tensor
        std::vector<int64_t> input_shape = {1, 3, info.in_h, info.in_w};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, 
            info.input_tensor_values.data(), info.input_tensor_values.size(), 
            input_shape.data(), input_shape.size()
        ));

        // 4. 綁定 Output Tensor
        std::vector<Ort::Value> output_tensors;
        for (size_t i = 0; i < info.output_list.size(); i++) {
            output_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, 
                info.output_list[i].data(), info.output_list[i].size(), 
                info.output_shapes[i].data(), info.output_shapes[i].size()
            ));
        }

        // 5. 執行推理
        session.Run(Ort::RunOptions{nullptr}, 
                    info.input_names.data(), input_tensors.data(), 1, 
                    info.output_names.data(), output_tensors.data(), info.output_names.size());
    }

} // namespace OnnxInference

// --- 測試使用 ---
int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_StrongType_Test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "/home/jianhua/Desktop/vitis-ai-pytorch/dpu_inference/model/YOLO_int.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    OnnxInference::VisionModelInfo model_info = OnnxInference::InitializeModelInfo(session, allocator);
    
    cv::Mat frame(1080, 1920, CV_8UC3, cv::Scalar(0, 255, 0)); 

    // =========================================================
    // 前處理：最後一步將結果放入強型別的 cv::Mat_<cv::Vec3f> 中
    // =========================================================
    cv::Mat resized_img;
    cv::resize(frame, resized_img, cv::Size(model_info.in_w, model_info.in_h));
    // cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

    // 宣告強型別變數
    cv::Mat_<cv::Vec3f> float_img;
    // convertTo 會確保轉換為 CV_32FC3
    resized_img.convertTo(float_img, CV_32FC3, 1.0f / 255.0f); 

    std::cout << "\nRunning Inference..." << std::endl;
    // 傳入強型別引數
    OnnxInference::RunInference(session, model_info, float_img);

    std::cout << "Inference completed. First value of Output 0: " 
              << model_info.output_list[0][0] << std::endl;

    return 0;
}