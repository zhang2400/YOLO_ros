// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/classifier_node.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "yolo_ros/messages.hpp"
#include "yolo_ros/visualization.hpp"

namespace yolo_ros
{

ClassifierNode::ClassifierNode(const rclcpp::NodeOptions & options)
: YoloNodeBase("yolos_classifier", options)
{
}

bool ClassifierNode::configure_task()
{
  classification_pub_ =
    create_publisher<vision_msgs::msg::Classification>("~/classification", 10);
  return true;
}

void ClassifierNode::activate_task()
{
  classification_pub_->on_activate();
}

void ClassifierNode::deactivate_task()
{
  classification_pub_->on_deactivate();
}

void ClassifierNode::cleanup_task()
{
  classification_pub_.reset();
}

cv::Mat ClassifierNode::preprocess(const cv::Mat & image, Letterbox & geometry)
{
  geometry.image_width = image.cols;
  geometry.image_height = image.rows;
  geometry.input_width = engine_->input_width();
  geometry.input_height = engine_->input_height();
  geometry.pad_x = 0.0F;
  geometry.pad_y = 0.0F;

  const int side = std::min(image.cols, image.rows);
  const int x = (image.cols - side) / 2;
  const int y = (image.rows - side) / 2;
  const cv::Rect crop_rect(x, y, side, side);

  cv::Mat resized;
  cv::resize(
    image(crop_rect), resized,
    cv::Size(engine_->input_width(), engine_->input_height()));
  geometry.scale =
    static_cast<float>(engine_->input_width()) / static_cast<float>(side);
  return resized;
}

void ClassifierNode::process(
  const cv::Mat & /*image*/,
  const std::vector<Tensor> & outputs,
  const Letterbox & /*geometry*/,
  const std_msgs::msg::Header & header,
  cv::Mat & debug_canvas)
{
  if (outputs.empty() || outputs.front().data.empty()) {
    return;
  }

  std::vector<float> scores = outputs.front().data;

  // YOLO classification exports usually already apply softmax; detect the
  // difference by checking the value range and the sum.
  bool looks_like_probabilities = true;
  float sum = 0.0F;
  for (float value : scores) {
    if (value < 0.0F || value > 1.0F) {
      looks_like_probabilities = false;
      break;
    }
    sum += value;
  }
  looks_like_probabilities =
    looks_like_probabilities && std::abs(sum - 1.0F) < 0.05F;

  if (!looks_like_probabilities) {
    const float max_value = *std::max_element(scores.begin(), scores.end());
    float denominator = 0.0F;
    for (float & value : scores) {
      value = std::exp(value - max_value);
      denominator += value;
    }
    for (float & value : scores) {
      value /= denominator;
    }
  }

  const int total = static_cast<int>(scores.size());
  const int top_k = std::min(5, total);
  std::vector<int> indices(static_cast<size_t>(total));
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(
    indices.begin(), indices.begin() + top_k, indices.end(),
    [&scores](int a, int b) {return scores[a] > scores[b];});

  std::vector<std::pair<int, float>> results;
  results.reserve(static_cast<size_t>(top_k));
  for (int i = 0; i < top_k; ++i) {
    results.emplace_back(indices[static_cast<size_t>(i)],
      scores[indices[static_cast<size_t>(i)]]);
  }

  classification_pub_->publish(
    to_classification(results, class_names_, header));

  if (!debug_canvas.empty() && !results.empty()) {
    const int class_id = results.front().first;
    const std::string label =
      (class_id >= 0 && static_cast<size_t>(class_id) < class_names_.size()) ?
      class_names_[class_id] : std::to_string(class_id);
    draw_classification(debug_canvas, label, results.front().second);
  }
}

}  // namespace yolo_ros
