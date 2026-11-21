// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef YOLO_ROS__LETTERBOX_HPP_
#define YOLO_ROS__LETTERBOX_HPP_

#include <vector>

#include <opencv2/core.hpp>

#include "yolo_ros/types.hpp"

namespace yolo_ros
{

/// Resize keeping aspect ratio and pad to the target size with grey (114).
cv::Mat letterbox(
  const cv::Mat & src,
  const cv::Size & target_size,
  Letterbox & geometry,
  const cv::Scalar & pad_color = cv::Scalar(114, 114, 114));

/// Convert a BGR uint8 image (cv_bridge convention) to a normalized RGB float
/// blob, matching the Ultralytics training/export channel order.
std::vector<float> to_blob(const cv::Mat & image, bool channels_first);

}  // namespace yolo_ros

#endif  // YOLO_ROS__LETTERBOX_HPP_
