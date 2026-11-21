// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/visualization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include <opencv2/imgproc.hpp>

namespace yolo_ros
{

namespace
{

constexpr int kCocoSkeleton[][2] = {
  {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8},
  {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}};

void draw_label(cv::Mat & image, const cv::Rect & rect, const std::string & text,
  const cv::Scalar & color)
{
  constexpr double kFontScale = 0.5;
  int baseline = 0;
  const cv::Size text_size =
    cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, kFontScale, 1, &baseline);
  const int y = std::max(rect.y, text_size.height + 4);
  const cv::Rect label_rect(
    rect.x, y - text_size.height - 4, text_size.width + 4, text_size.height + 6);
  cv::rectangle(image, label_rect, color, cv::FILLED);
  cv::putText(
    image, text, cv::Point(rect.x + 2, y - 2),
    cv::FONT_HERSHEY_SIMPLEX, kFontScale, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

}  // namespace

cv::Scalar class_color(int class_id)
{
  // Deterministic palette keyed by class id (golden-ratio hue stepping).
  const double hue = std::fmod(class_id * 0.618033988749895 * 180.0, 180.0);
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 220, 240));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(c[0], c[1], c[2]);
}

void draw_detections(cv::Mat & image, const std::vector<Detection> & detections)
{
  for (const auto & det : detections) {
    const cv::Scalar color = class_color(det.class_id);
    const cv::Rect rect(
      static_cast<int>(det.box.x), static_cast<int>(det.box.y),
      static_cast<int>(det.box.width), static_cast<int>(det.box.height));
    cv::rectangle(image, rect, color, 2);

    char text[128];
    std::snprintf(
      text, sizeof(text), "%s %.2f", det.label.c_str(),
      static_cast<double>(det.score));
    draw_label(image, rect, text, color);
  }
}

void draw_poses(
  cv::Mat & image, const std::vector<PoseDetection> & poses, float keypoint_threshold)
{
  for (const auto & pose : poses) {
    const cv::Scalar color = class_color(pose.det.class_id);
    const cv::Rect rect(
      static_cast<int>(pose.det.box.x), static_cast<int>(pose.det.box.y),
      static_cast<int>(pose.det.box.width), static_cast<int>(pose.det.box.height));
    cv::rectangle(image, rect, color, 1);

    for (const auto & pair : kCocoSkeleton) {
      const KeyPoint & a = pose.keypoints[static_cast<size_t>(pair[0])];
      const KeyPoint & b = pose.keypoints[static_cast<size_t>(pair[1])];
      if (a.confidence >= keypoint_threshold && b.confidence >= keypoint_threshold) {
        cv::line(
          image, cv::Point(static_cast<int>(a.x), static_cast<int>(a.y)),
          cv::Point(static_cast<int>(b.x), static_cast<int>(b.y)),
          cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
      }
    }

    for (const auto & keypoint : pose.keypoints) {
      if (keypoint.confidence < keypoint_threshold) {
        continue;
      }
      cv::circle(
        image, cv::Point(static_cast<int>(keypoint.x), static_cast<int>(keypoint.y)),
        3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }

    char text[64];
    std::snprintf(
      text, sizeof(text), "%s %.2f", pose.det.label.c_str(),
      static_cast<double>(pose.det.score));
    draw_label(image, rect, text, color);
  }
}

void draw_obb_detections(cv::Mat & image, const std::vector<ObbDetection> & detections)
{
  for (const auto & det : detections) {
    const cv::Scalar color = class_color(det.class_id);
    const cv::RotatedRect rotated(
      cv::Point2f(det.center_x, det.center_y), cv::Size2f(det.width, det.height),
      static_cast<float>(det.angle * 180.0 / CV_PI));
    cv::Point2f corners[4];
    rotated.points(corners);
    for (int i = 0; i < 4; ++i) {
      cv::line(
        image, corners[i], corners[(i + 1) % 4], color, 2, cv::LINE_AA);
    }

    char text[128];
    std::snprintf(
      text, sizeof(text), "%s %.2f", det.label.c_str(),
      static_cast<double>(det.score));
    const cv::Rect label_anchor(
      static_cast<int>(rotated.center.x), static_cast<int>(rotated.center.y), 1, 1);
    draw_label(image, label_anchor, text, color);
  }
}

void draw_classification(cv::Mat & image, const std::string & label, float score)
{
  char text[192];
  std::snprintf(
    text, sizeof(text), "%s %.1f%%", label.c_str(),
    static_cast<double>(score) * 100.0);
  const cv::Rect anchor(8, 8, 1, 1);
  draw_label(image, anchor, text, cv::Scalar(0, 200, 0));
}

}  // namespace yolo_ros
