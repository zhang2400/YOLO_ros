// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef YOLO_ROS__TYPES_HPP_
#define YOLO_ROS__TYPES_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

namespace yolo_ros
{

/// Dense float tensor produced by the inference backend.
struct Tensor
{
  std::vector<int64_t> shape;
  std::vector<float> data;
};

/// Letterbox geometry used to map model space back to image space.
struct Letterbox
{
  int input_width{0};
  int input_height{0};
  int image_width{0};
  int image_height{0};
  float scale{1.0F};
  float pad_x{0.0F};
  float pad_y{0.0F};
};

/// One decoded detection in original image coordinates.
struct Detection
{
  cv::Rect2f box;  // x, y, width, height (top-left origin)
  float score{0.0F};
  int class_id{-1};
  std::string label;
  int64_t anchor{-1};  // index in the raw output, used for auxiliary channels
};

/// One 2D keypoint in original image coordinates.
struct KeyPoint
{
  float x{0.0F};
  float y{0.0F};
  float confidence{0.0F};
  int id{0};
};

/// Pose detection: a person box plus its keypoints.
struct PoseDetection
{
  Detection det;
  std::vector<KeyPoint> keypoints;
};

/// Oriented (rotated) bounding box in original image coordinates.
struct ObbDetection
{
  float center_x{0.0F};
  float center_y{0.0F};
  float width{0.0F};
  float height{0.0F};
  float angle{0.0F};  // radians
  float score{0.0F};
  int class_id{-1};
  std::string label;
};

}  // namespace yolo_ros

#endif  // YOLO_ROS__TYPES_HPP_
