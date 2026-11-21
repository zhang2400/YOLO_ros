// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef YOLO_ROS__DETECTION_DECODER_HPP_
#define YOLO_ROS__DETECTION_DECODER_HPP_

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "yolo_ros/types.hpp"

namespace yolo_ros
{

/// Decode a YOLOv5/v8/v11 style detection head into image-space detections.
/// Supports [1, C, N] and [1, N, C] layouts, with or without objectness.
std::vector<Detection> decode_detections(
  const Tensor & output,
  const Letterbox & geometry,
  const std::vector<std::string> & class_names,
  float conf_threshold,
  float nms_threshold,
  int max_detections = 300);

/// Decode a YOLO pose head (4 box + 1 class + K * 3 keypoints).
std::vector<PoseDetection> decode_pose(
  const Tensor & output,
  const Letterbox & geometry,
  const std::vector<std::string> & class_names,
  float conf_threshold,
  float nms_threshold,
  int num_keypoints = 17,
  float keypoint_threshold = 0.25F,
  int max_detections = 300);

/// Decode a YOLO OBB head (4 box + C class + 1 angle).
std::vector<ObbDetection> decode_obb(
  const Tensor & output,
  const Letterbox & geometry,
  const std::vector<std::string> & class_names,
  float conf_threshold,
  float nms_threshold,
  int max_detections = 300);

/// Build a colorized instance mask (BGR, original image size) from the
/// segmentation head and prototype tensor.
cv::Mat build_instance_masks(
  const Tensor & seg_output,
  const Tensor & prototypes,
  const std::vector<Detection> & detections,
  const Letterbox & geometry,
  float mask_threshold = 0.5F);

}  // namespace yolo_ros

#endif  // YOLO_ROS__DETECTION_DECODER_HPP_
