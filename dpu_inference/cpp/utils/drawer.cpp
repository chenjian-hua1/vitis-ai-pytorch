#include <drawer.h>

cv::Mat scale_boxes(const cv::Mat&        boxes,
                    std::pair<float,float> ratio,
                    std::pair<float,float> pad,
                    cv::Size               orig_shape)
{
    cv::Mat out;
    boxes.convertTo(out, CV_32F);

    float w_max = static_cast<float>(orig_shape.width);
    float h_max = static_cast<float>(orig_shape.height);

    {
        cv::Mat col = (out.col(0) - pad.first) / ratio.first;
        cv::min(col, w_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(0));
    }
    {
        cv::Mat col = (out.col(1) - pad.second) / ratio.second;
        cv::min(col, h_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(1));
    }
    {
        cv::Mat col = (out.col(2) - pad.first) / ratio.first;
        cv::min(col, w_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(2));
    }
    {
        cv::Mat col = (out.col(3) - pad.second) / ratio.second;
        cv::min(col, h_max, col);
        cv::max(col, 0.f, col);
        col.copyTo(out.col(3));
    }

    return out;
}

cv::Mat draw_boxes(const cv::Mat&                  img,
                   const std::vector<Detection>&   detections,
                   const std::vector<std::string>& class_names)
{
    cv::Mat out = img.clone();

    for (const Detection& det : detections) {
        int x1 = static_cast<int>(det.x1), y1 = static_cast<int>(det.y1);
        int x2 = static_cast<int>(det.x2), y2 = static_cast<int>(det.y2);
        int id = det.class_id;

        cv::Scalar color(
            (id * 67  + 100) % 255,
            (id * 113 +  50) % 255,
            (id * 179 + 150) % 255
        );

        cv::rectangle(out, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        std::string label = (class_names.size() > static_cast<size_t>(id))
            ? class_names[id] + ": " + std::to_string(det.score).substr(0, 4)
            : "Class " + std::to_string(id) + ": " +
              std::to_string(det.score).substr(0, 4);

        int      baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                       0.5, 1, &baseline);

        cv::rectangle(out,
                       cv::Point(x1, y1 - ts.height - 6),
                       cv::Point(x1 + ts.width, y1),
                       color, cv::FILLED);

        cv::putText(out, label, cv::Point(x1, y1 - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }

    return out;
}

void draw_detection(const cv::Mat& in, cv::Mat& out, const DetectionBatch& detections, const ResizeResult& resize_inf, double fps) {
    if (detections.count > 0) {
        cv::Mat boxes_padded(detections.count, 4, CV_32F);
        for (int i = 0; i < detections.count; ++i) {
            const Detection& d = detections.data[i];
            float* r = boxes_padded.ptr<float>(i);
            r[0] = d.x1;  r[1] = d.y1;  r[2] = d.x2;  r[3] = d.y2;
        }

        cv::Mat boxes_orig = scale_boxes(
            boxes_padded, resize_inf.ratio, resize_inf.pad,
            cv::Size(in.cols, in.rows));

        std::vector<Detection> dets_drawable(detections.count);
        for (int i = 0; i < detections.count; ++i) {
            const float* r = boxes_orig.ptr<float>(i);
            dets_drawable[i] = Detection{
                r[0], r[1], r[2], r[3],
                detections.data[i].score, detections.data[i].class_id
            };
        }

        out = draw_boxes(in, dets_drawable);
    } else {
        out = in;
    }

    std::ostringstream ss;
    ss << "FPS: " << std::fixed << std::setprecision(2) << fps;
    cv::putText(out, ss.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);
}

// ============================================================================
//  Tracking 繪圖
// ============================================================================

std::vector<bytetrack::Box> scale_detections(const DetectionBatch& detections,
                                             const ResizeResult&   resize_inf,
                                             cv::Size              orig_shape)
{
    std::vector<bytetrack::Box> boxes;
    if (detections.count <= 0) return boxes;
    boxes.reserve(static_cast<size_t>(detections.count));

    const float inv_rx = 1.f / resize_inf.ratio.first;
    const float inv_ry = 1.f / resize_inf.ratio.second;
    const float dw     = resize_inf.pad.first;
    const float dh     = resize_inf.pad.second;
    const float w_max  = static_cast<float>(orig_shape.width);
    const float h_max  = static_cast<float>(orig_shape.height);

    for (int i = 0; i < detections.count; ++i) {
        const Detection& d = detections.data[i];
        bytetrack::Box b;
        b.x1    = std::clamp((d.x1 - dw) * inv_rx, 0.f, w_max);
        b.y1    = std::clamp((d.y1 - dh) * inv_ry, 0.f, h_max);
        b.x2    = std::clamp((d.x2 - dw) * inv_rx, 0.f, w_max);
        b.y2    = std::clamp((d.y2 - dh) * inv_ry, 0.f, h_max);
        b.score = d.score;
        b.cls   = d.class_id;
        boxes.push_back(b);
    }
    return boxes;
}


cv::Mat draw_tracks(const cv::Mat&                       img,
                    const std::vector<bytetrack::Track>& tracks,
                    const std::vector<std::string>&      class_names)
{
    cv::Mat out = img.clone();

    for (const bytetrack::Track& t : tracks) {
        int x1 = static_cast<int>(t.x1), y1 = static_cast<int>(t.y1);
        int x2 = static_cast<int>(t.x2), y2 = static_cast<int>(t.y2);

        // 顏色以 track_id 為基準,同一個目標整段影片顏色固定
        int id = t.track_id;
        cv::Scalar color(
            (id * 67  + 100) % 255,
            (id * 113 +  50) % 255,
            (id * 179 + 150) % 255
        );

        cv::rectangle(out, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        std::string cname = (class_names.size() > static_cast<size_t>(t.cls))
            ? class_names[t.cls]
            : "Class " + std::to_string(t.cls);

        std::string label = "ID" + std::to_string(id) + " " + cname +
                            ": " + std::to_string(t.score).substr(0, 4);

        int      baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                       0.5, 1, &baseline);

        // 框貼在畫面頂端時,標籤翻到框內側,避免被裁掉
        int label_top = (y1 - ts.height - 6 >= 0) ? y1 - ts.height - 6 : y1;
        cv::rectangle(out,
                       cv::Point(x1, label_top),
                       cv::Point(x1 + ts.width, label_top + ts.height + 6),
                       color, cv::FILLED);

        cv::putText(out, label, cv::Point(x1, label_top + ts.height + 1),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }

    return out;
}


void draw_tracking(const cv::Mat&                       in,
                   cv::Mat&                             out,
                   const std::vector<bytetrack::Track>& tracks,
                   double                               fps)
{
    out = tracks.empty() ? in : draw_tracks(in, tracks);

    std::ostringstream ss;
    ss << "FPS: " << std::fixed << std::setprecision(2) << fps
       << "   Tracks: " << tracks.size();
    cv::putText(out, ss.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);
}