// eval.h ----------------------------------------------------------------------
// Header-only：把 C++ 端的 detection / tracking 結果輸出成
//   1) MOT Challenge txt  -> 給 TrackEval 算 HOTA / MOTA / IDF1
//   2) COCO results json  -> 給 pycocotools 算 mAP
//
// 用法（重點）：frame 欄位一律填「該張圖在 val.json 裡的 image_id」，
// 不要用自己累加的 frame counter，Python 端會再依 video_id 重新編號。
//
//   evalio::TrackResultWriter writer("predict.txt",
//                                    "track_coco.json",
//                                    "det_coco.json",
//                                    {1, 2, 3, 6});   // 模型 cls -> COCO category_id
//   ...
//   writer.add_detections(image_id, boxes);          // 追蹤前的 NMS 結果（算 mAP 用）
//   const auto& tracks = tracker.update(boxes, t_cap);
//   writer.add_tracks(image_id, tracks, frame.cols, frame.rows);
//   ...
//   writer.save();
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#if __cplusplus >= 201703L && __has_include(<filesystem>)
#include <filesystem>
#define EVALIO_HAS_FS 1
#endif

namespace evalio {

struct WriterOptions {
    int   min_tracklet_len = 0;      // 只輸出已連續追蹤 >= N 幀的 track（0 = 全輸出）
    float min_score        = 0.0f;   // 分數門檻
    float min_box_wh       = 1.0f;   // 太小的框直接丟掉（避免 w/h = 0 讓 TrackEval 出事）
    bool  clamp_to_image   = true;   // 把框裁進影像範圍
};

class TrackResultWriter {
public:
    // cls_to_coco_cat: index = 模型輸出的 class_id，value = val.json 裡的 category_id。
    // 以這份 SeaDronesSee val.json 來說是 {1, 2, 3, 6}
    // （1=swimmer, 2=swimmer with life jacket, 3=boat, 6=life jacket）。
    // 傳空 vector 代表 class_id 直接當 category_id 用。
    TrackResultWriter(std::string mot_txt_path,
                      std::string track_json_path = "",
                      std::string det_json_path   = "",
                      std::vector<int> cls_to_coco_cat = {},
                      WriterOptions opt = WriterOptions())
        : mot_txt_path_(std::move(mot_txt_path)),
          track_json_path_(std::move(track_json_path)),
          det_json_path_(std::move(det_json_path)),
          cls_to_coco_cat_(std::move(cls_to_coco_cat)),
          opt_(opt) {}

    // ---- 每幀呼叫：追蹤結果 -------------------------------------------------
    // TrackT 只要有 track_id / cls / score / x1,y1,x2,y2 / tracklet_len 就能吃，
    // 所以這個 header 不需要 include 你的 bytetrack 標頭。
    template <typename TrackT>
    void add_tracks(int image_id, const std::vector<TrackT>& tracks,
                    int img_w = 0, int img_h = 0) {
        for (const auto& t : tracks) {
            if (t.tracklet_len < opt_.min_tracklet_len) continue;
            if (t.score < opt_.min_score) continue;

            float x1 = t.x1, y1 = t.y1, x2 = t.x2, y2 = t.y2;
            if (!normalize_box(x1, y1, x2, y2, img_w, img_h)) continue;

            mot_.push_back(Row{image_id, t.track_id, x1, y1, x2 - x1, y2 - y1,
                               t.score, t.cls});
            if (!track_json_path_.empty())
                track_json_.push_back(Row{image_id, t.track_id, x1, y1,
                                          x2 - x1, y2 - y1, t.score, t.cls});
        }
    }

    // ---- 每幀呼叫（可選）：追蹤前的偵測結果，用來算「純偵測」的 mAP ---------
    // BoxT 需要有 x1,y1,x2,y2 / score / cls（bytetrack::Box 就符合）。
    template <typename BoxT>
    void add_detections(int image_id, const std::vector<BoxT>& boxes,
                        int img_w = 0, int img_h = 0) {
        if (det_json_path_.empty()) return;
        for (const auto& b : boxes) {
            if (b.score < opt_.min_score) continue;
            float x1 = b.x1, y1 = b.y1, x2 = b.x2, y2 = b.y2;
            if (!normalize_box(x1, y1, x2, y2, img_w, img_h)) continue;
            det_json_.push_back(Row{image_id, -1, x1, y1, x2 - x1, y2 - y1,
                                    b.score, b.cls});
        }
    }

    // Detection / DetectionBatch（xyxy 但通常是 letterbox 座標）如果你已經
    // scale 回原圖，也可以直接丟進 add_detections。這個多載是給 DetectionBatch 用的。
    template <typename BatchT>
    void add_detection_batch(int image_id, const BatchT& batch,
                             int img_w = 0, int img_h = 0) {
        if (det_json_path_.empty()) return;
        for (const auto& d : batch) {
            if (d.score < opt_.min_score) continue;
            float x1 = d.x1, y1 = d.y1, x2 = d.x2, y2 = d.y2;
            if (!normalize_box(x1, y1, x2, y2, img_w, img_h)) continue;
            det_json_.push_back(Row{image_id, -1, x1, y1, x2 - x1, y2 - y1,
                                    d.score, d.class_id});
        }
    }

    // ---- 全部跑完再呼叫一次 ------------------------------------------------
    bool save() const {
        make_parent_dir(mot_txt_path_);
        make_parent_dir(track_json_path_);
        make_parent_dir(det_json_path_);
        bool ok = write_mot(mot_txt_path_, mot_);
        if (!track_json_path_.empty()) ok &= write_json(track_json_path_, track_json_);
        if (!det_json_path_.empty())   ok &= write_json(det_json_path_, det_json_);
        return ok;
    }

    size_t track_rows() const { return mot_.size(); }
    size_t det_rows()   const { return det_json_.size(); }

private:
    struct Row {
        int   image_id;
        int   track_id;
        float x, y, w, h;   // top-left + wh，原圖座標
        float score;
        int   cls;
    };

    bool normalize_box(float& x1, float& y1, float& x2, float& y2,
                       int img_w, int img_h) const {
        if (x2 < x1) std::swap(x1, x2);
        if (y2 < y1) std::swap(y1, y2);
        if (opt_.clamp_to_image && img_w > 0 && img_h > 0) {
            x1 = std::max(0.f, std::min(x1, static_cast<float>(img_w - 1)));
            y1 = std::max(0.f, std::min(y1, static_cast<float>(img_h - 1)));
            x2 = std::max(0.f, std::min(x2, static_cast<float>(img_w)));
            y2 = std::max(0.f, std::min(y2, static_cast<float>(img_h)));
        }
        return (x2 - x1) >= opt_.min_box_wh && (y2 - y1) >= opt_.min_box_wh;
    }

    static void make_parent_dir(const std::string& path) {
#ifdef EVALIO_HAS_FS
        if (path.empty()) return;
        std::filesystem::path p(path);
        if (p.has_parent_path() && !p.parent_path().empty())
            std::filesystem::create_directories(p.parent_path());
#else
        (void)path;   // C++14 以下請自行確保輸出資料夾已存在
#endif
    }

    int to_coco_cat(int cls) const {
        if (cls_to_coco_cat_.empty()) return cls;
        if (cls < 0 || cls >= static_cast<int>(cls_to_coco_cat_.size())) return -1;
        return cls_to_coco_cat_[cls];
    }

    // MOT 格式: frame,id,x,y,w,h,conf,class,vis
    // frame 這裡放 image_id，class 放模型原始 class_id（Python 端再對應）
    static bool write_mot(const std::string& path, const std::vector<Row>& rows) {
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "[evalio] cannot open %s\n", path.c_str()); return false; }
        for (const auto& r : rows)
            std::fprintf(f, "%d,%d,%.2f,%.2f,%.2f,%.2f,%.6f,%d,1\n",
                         r.image_id, r.track_id, r.x, r.y, r.w, r.h, r.score, r.cls);
        std::fclose(f);
        std::printf("[evalio] wrote %zu rows -> %s\n", rows.size(), path.c_str());
        return true;
    }

    bool write_json(const std::string& path, const std::vector<Row>& rows) const {
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "[evalio] cannot open %s\n", path.c_str()); return false; }
        std::fputc('[', f);
        bool first = true;
        for (const auto& r : rows) {
            int cat = to_coco_cat(r.cls);
            if (cat < 0) continue;
            if (!first) std::fputs(",\n", f); else std::fputc('\n', f);
            first = false;
            std::fprintf(f,
                "{\"image_id\":%d,\"category_id\":%d,"
                "\"bbox\":[%.2f,%.2f,%.2f,%.2f],\"score\":%.6f,\"track_id\":%d}",
                r.image_id, cat, r.x, r.y, r.w, r.h, r.score, r.track_id);
        }
        std::fputs("\n]\n", f);
        std::fclose(f);
        std::printf("[evalio] wrote %zu rows -> %s\n", rows.size(), path.c_str());
        return true;
    }

    std::string mot_txt_path_, track_json_path_, det_json_path_;
    std::vector<int> cls_to_coco_cat_;
    WriterOptions opt_;
    std::vector<Row> mot_, track_json_, det_json_;
};

// ---------------------------------------------------------------------------
// manifest 讀取：Python 端會產生一份 image_manifest.tsv
//   image_id \t video_id \t frame_index \t /abs/path/to/img.jpg
// 已依 video_id、frame_index 排好序，C++ 照順序跑即可。
// ---------------------------------------------------------------------------
struct ManifestItem {
    int         image_id;
    int         video_id;
    int         frame_index;
    std::string path;
};

inline std::vector<ManifestItem> load_manifest(const std::string& path) {
    std::vector<ManifestItem> items;
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { std::fprintf(stderr, "[evalio] cannot open %s\n", path.c_str()); return items; }
    char line[4096];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int a, b, c; char p[3072];
        if (std::sscanf(line, "%d\t%d\t%d\t%3071[^\n]", &a, &b, &c, p) == 4) {
            std::string sp(p);
            // 去掉尾端的 \r 或空白（Windows 換行、編輯器留白都可能混進來）
            while (!sp.empty() && (sp.back() == '\r' || sp.back() == ' ' || sp.back() == '\t'))
                sp.pop_back();
            items.push_back(ManifestItem{a, b, c, sp});
        }
    }
    std::fclose(f);
    std::printf("[evalio] manifest: %zu images\n", items.size());
    return items;
}

}  // namespace evalio
