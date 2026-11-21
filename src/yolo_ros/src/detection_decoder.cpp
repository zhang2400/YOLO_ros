// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/detection_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "yolo_ros/visualization.hpp"

namespace yolo_ros
{

namespace
{

/// Non-owning view over [C, N] model output channels.
struct TensorView
{
  const float * data{nullptr};
  int64_t channels{0};
  int64_t anchors{0};
  bool channels_first{true};

  float at(int64_t channel, int64_t anchor) const
  {
    return channels_first ?
           data[channel * anchors + anchor] :
           data[anchor * channels + channel];
  }
};

TensorView make_view(const Tensor & tensor)
{
  if (tensor.shape.size() < 2) {
    throw std::runtime_error("unexpected model output rank");
  }
  const int64_t dim_a = tensor.shape[tensor.shape.size() - 2];
  const int64_t dim_b = tensor.shape[tensor.shape.size() - 1];

  TensorView view;
  view.data = tensor.data.data();
  view.channels_first = dim_a < dim_b;
  view.channels = view.channels_first ? dim_a : dim_b;
  view.anchors = view.channels_first ? dim_b : dim_a;
  return view;
}

struct Candidate
{
  cv::Rect2f box;
  float cx{0.0F};
  float cy{0.0F};
  float w{0.0F};
  float h{0.0F};
  float score{0.0F};
  int class_id{-1};
  int64_t anchor{-1};
};

std::vector<Candidate> collect_candidates(
  const TensorView & view,
  const Letterbox & geometry,
  int num_classes,
  bool has_objectness,
  float conf_threshold)
{
  if (view.channels < 4 + num_classes + (has_objectness ? 1 : 0)) {
    throw std::runtime_error("model output has too few channels for the class count");
  }

  const float inv_scale = geometry.scale > 0.0F ? 1.0F / geometry.scale : 0.0F;
  const float img_w = static_cast<float>(geometry.image_width);
  const float img_h = static_cast<float>(geometry.image_height);
  const int64_t score_offset = 4 + (has_objectness ? 1 : 0);

  std::vector<Candidate> candidates;
  candidates.reserve(256);

  for (int64_t i = 0; i < view.anchors; ++i) {
    const float objectness = has_objectness ? view.at(4, i) : 1.0F;
    if (has_objectness && objectness < conf_threshold) {
      continue;
    }

    int best_class = -1;
    float best_score = 0.0F;
    for (int c = 0; c < num_classes; ++c) {
      const float score = view.at(score_offset + c, i) * objectness;
      if (score > best_score) {
        best_score = score;
        best_class = c;
      }
    }
    if (best_class < 0 || best_score < conf_threshold) {
      continue;
    }

    const float cx = view.at(0, i);
    const float cy = view.at(1, i);
    const float w = view.at(2, i);
    const float h = view.at(3, i);

    const float x = (cx - geometry.pad_x) * inv_scale;
    const float y = (cy - geometry.pad_y) * inv_scale;
    const float width = w * inv_scale;
    const float height = h * inv_scale;

    const float left = std::max(0.0F, x - width * 0.5F);
    const float top = std::max(0.0F, y - height * 0.5F);
    const float right = std::min(img_w, x + width * 0.5F);
    const float bottom = std::min(img_h, y + height * 0.5F);
    if (right <= left || bottom <= top) {
      continue;
    }

    Candidate cand;
    cand.box = cv::Rect2f(left, top, right - left, bottom - top);
    cand.cx = x;
    cand.cy = y;
    cand.w = width;
    cand.h = height;
    cand.score = best_score;
    cand.class_id = best_class;
    cand.anchor = i;
    candidates.push_back(cand);
  }
  return candidates;
}

/// Deterministic ordering: score descending, anchor index as a stable tie-break.
bool higher_score(const Candidate & a, const Candidate & b)
{
  if (a.score != b.score) {
    return a.score > b.score;
  }
  return a.anchor < b.anchor;
}

/// Class-aware greedy NMS, returns kept candidates sorted by score.
///
/// Candidates are sorted before grouping so that the per-class pass and the
/// final max_detections truncation never depend on unordered_map iteration
/// order: the same input always yields the same output.
std::vector<Candidate> run_nms(
  std::vector<Candidate> & candidates,
  float conf_threshold,
  float nms_threshold,
  int max_detections)
{
  std::sort(candidates.begin(), candidates.end(), higher_score);

  std::unordered_map<int, std::vector<size_t>> per_class;
  for (size_t i = 0; i < candidates.size(); ++i) {
    per_class[candidates[i].class_id].push_back(i);
  }

  std::vector<Candidate> kept;
  for (auto & entry : per_class) {
    const std::vector<size_t> & indices = entry.second;
    // Keep sub-pixel coordinates so IoU does not drift from integer truncation.
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    boxes.reserve(indices.size());
    scores.reserve(indices.size());
    for (size_t idx : indices) {
      const cv::Rect2f & b = candidates[idx].box;
      boxes.emplace_back(b.x, b.y, b.width, b.height);
      scores.push_back(candidates[idx].score);
    }

    std::vector<int> selected;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, selected);
    for (int k : selected) {
      kept.push_back(candidates[indices[static_cast<size_t>(k)]]);
    }
  }

  std::sort(kept.begin(), kept.end(), higher_score);
  if (static_cast<int>(kept.size()) > max_detections) {
    kept.resize(static_cast<size_t>(std::max(0, max_detections)));
  }
  return kept;
}

std::string label_for(int class_id, const std::vector<std::string> & class_names)
{
  return (class_id >= 0 && static_cast<size_t>(class_id) < class_names.size()) ?
         class_names[class_id] : std::to_string(class_id);
}

float sigmoid(float x)
{
  return 1.0F / (1.0F + std::exp(-x));
}

}  // namespace

std::vector<Detection> decode_detections(
  const Tensor & output,
  const Letterbox & geometry,
  const std::vector<std::string> & class_names,
  float conf_threshold,
  float nms_threshold,
  int max_detections)
{
  const TensorView view = make_view(output);

  int num_classes = static_cast<int>(class_names.size());
  bool has_objectness = false;
  if (num_classes <= 0) {
    num_classes = static_cast<int>(view.channels) - 4;
  } else if (view.channels == num_classes + 5) {
    has_objectness = true;
  } else if (view.channels < num_classes + 4) {
    throw std::runtime_error(
            "model output channels (" + std::to_string(view.channels) +
            ") do not match class count (" + std::to_string(num_classes) + ")");
  }
  if (num_classes <= 0) {
    throw std::runtime_error("could not infer class count from model output");
  }

  std::vector<Candidate> candidates =
    collect_candidates(view, geometry, num_classes, has_objectness, conf_threshold);
  std::vector<Candidate> kept =
    run_nms(candidates, conf_threshold, nms_threshold, max_detections);

  std::vector<Detection> detections;
  detections.reserve(kept.size());
  for (const auto & cand : kept) {
    Detection det;
    det.box = cand.box;
    det.score = cand.score;
    det.class_id = cand.class_id;
    det.label = label_for(cand.class_id, class_names);
    det.anchor = cand.anchor;
    detections.push_back(std::move(det));
  }
  return detections;
}

std::vector<PoseDetection> decode_pose(
  const Tensor & output,
  const Letterbox & geometry,
  const std::vector<std::string> & class_names,
  float conf_threshold,
  float nms_threshold,
  int num_keypoints,
  float keypoint_threshold,
  int max_detections)
{
  (void)keypoint_threshold;
  const TensorView view = make_view(output);
  const int num_classes = class_names.empty() ? 1 : static_cast<int>(class_names.size());

  std::vector<Candidate> candidates =
    collect_candidates(view, geometry, num_classes, false, conf_threshold);
  std::vector<Candidate> kept =
    run_nms(candidates, conf_threshold, nms_threshold, max_detections);

  const float inv_scale = geometry.scale > 0.0F ? 1.0F / geometry.scale : 0.0F;
  const int64_t kpt_offset = 4 + num_classes;

  std::vector<PoseDetection> poses;
  poses.reserve(kept.size());
  for (const auto & cand : kept) {
    PoseDetection pose;
    pose.det.box = cand.box;
    pose.det.score = cand.score;
    pose.det.class_id = cand.class_id;
    pose.det.label = label_for(cand.class_id, class_names);
    pose.det.anchor = cand.anchor;

    for (int k = 0; k < num_keypoints; ++k) {
      KeyPoint keypoint;
      keypoint.id = k;
      keypoint.x = (view.at(kpt_offset + 3 * k, cand.anchor) - geometry.pad_x) * inv_scale;
      keypoint.y = (view.at(kpt_offset + 3 * k + 1, cand.anchor) - geometry.pad_y) * inv_scale;
      keypoint.confidence = view.at(kpt_offset + 3 * k + 2, cand.anchor);
      pose.keypoints.push_back(keypoint);
    }
    poses.push_back(std::move(pose));
  }
  return poses;
}

std::vector<ObbDetection> decode_obb(
  const Tensor & output,
  const Letterbox & geometry,
  const std::vector<std::string> & class_names,
  float conf_threshold,
  float nms_threshold,
  int max_detections)
{
  const TensorView view = make_view(output);

  int num_classes = static_cast<int>(class_names.size());
  if (num_classes <= 0) {
    num_classes = static_cast<int>(view.channels) - 5;  // 4 box + angle
  }
  if (num_classes <= 0) {
    throw std::runtime_error("could not infer OBB class count from model output");
  }

  std::vector<Candidate> candidates =
    collect_candidates(view, geometry, num_classes, false, conf_threshold);
  std::vector<Candidate> kept =
    run_nms(candidates, conf_threshold, nms_threshold, max_detections);

  const int64_t angle_channel = 4 + num_classes;

  std::vector<ObbDetection> detections;
  detections.reserve(kept.size());
  for (const auto & cand : kept) {
    ObbDetection obb;
    obb.center_x = cand.cx;
    obb.center_y = cand.cy;
    obb.width = cand.w;
    obb.height = cand.h;
    obb.angle = view.at(angle_channel, cand.anchor);
    obb.score = cand.score;
    obb.class_id = cand.class_id;
    obb.label = label_for(cand.class_id, class_names);
    detections.push_back(std::move(obb));
  }
  return detections;
}

cv::Mat build_instance_masks(
  const Tensor & seg_output,
  const Tensor & prototypes,
  const std::vector<Detection> & detections,
  const Letterbox & geometry,
  float mask_threshold)
{
  const TensorView view = make_view(seg_output);
  if (view.channels < 33 || prototypes.shape.size() < 3) {
    throw std::runtime_error("invalid segmentation tensors");
  }

  const int64_t num_masks = prototypes.shape[prototypes.shape.size() - 3];
  const int64_t proto_h = prototypes.shape[prototypes.shape.size() - 2];
  const int64_t proto_w = prototypes.shape[prototypes.shape.size() - 1];
  const int64_t coeff_base = view.channels - num_masks;

  const float input_to_proto_x =
    static_cast<float>(proto_w) / static_cast<float>(geometry.input_width);
  const float input_to_proto_y =
    static_cast<float>(proto_h) / static_cast<float>(geometry.input_height);

  cv::Mat mask_image(
    geometry.image_height, geometry.image_width, CV_8UC3, cv::Scalar(0, 0, 0));

  for (const auto & det : detections) {
    if (det.anchor < 0) {
      continue;
    }

    // Box in model input coordinates, then in prototype coordinates.
    const float x_in = det.box.x * geometry.scale + geometry.pad_x;
    const float y_in = det.box.y * geometry.scale + geometry.pad_y;
    const int x0 = std::max(0, static_cast<int>(x_in * input_to_proto_x));
    const int y0 = std::max(0, static_cast<int>(y_in * input_to_proto_y));
    const int x1 = std::min(
      static_cast<int>(proto_w),
      static_cast<int>((x_in + det.box.width * geometry.scale) * input_to_proto_x));
    const int y1 = std::min(
      static_cast<int>(proto_h),
      static_cast<int>((y_in + det.box.height * geometry.scale) * input_to_proto_y));
    if (x1 <= x0 || y1 <= y0) {
      continue;
    }

    cv::Mat mask_small(y1 - y0, x1 - x0, CV_32F);
    for (int y = y0; y < y1; ++y) {
      float * row = mask_small.ptr<float>(y - y0);
      for (int x = x0; x < x1; ++x) {
        float value = 0.0F;
        for (int64_t k = 0; k < num_masks; ++k) {
          value += view.at(coeff_base + k, det.anchor) *
            prototypes.data[k * proto_h * proto_w + y * proto_w + x];
        }
        row[x - x0] = sigmoid(value);
      }
    }

    const int bw = std::max(1, static_cast<int>(det.box.width));
    const int bh = std::max(1, static_cast<int>(det.box.height));
    cv::Mat mask_full;
    cv::resize(mask_small, mask_full, cv::Size(bw, bh), 0, 0, cv::INTER_LINEAR);

    const cv::Scalar color = class_color(det.class_id);
    const int left = static_cast<int>(det.box.x);
    const int top = static_cast<int>(det.box.y);
    for (int y = 0; y < bh; ++y) {
      const int image_y = top + y;
      if (image_y < 0 || image_y >= mask_image.rows) {
        continue;
      }
      const float * row = mask_full.ptr<float>(y);
      cv::Vec3b * out = mask_image.ptr<cv::Vec3b>(image_y);
      for (int x = 0; x < bw; ++x) {
        const int image_x = left + x;
        if (image_x < 0 || image_x >= mask_image.cols) {
          continue;
        }
        if (row[x] >= mask_threshold) {
          out[image_x] = cv::Vec3b(
            static_cast<uchar>(color[0]), static_cast<uchar>(color[1]),
            static_cast<uchar>(color[2]));
        }
      }
    }
  }
  return mask_image;
}

}  // namespace yolo_ros
