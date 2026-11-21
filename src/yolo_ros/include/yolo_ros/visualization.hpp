// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef YOLO_ROS__VISUALIZATION_HPP_
#define YOLO_ROS__VISUALIZATION_HPP_

#include <vector>

#include <opencv2/core.hpp>

#include "yolo_ros/types.hpp"

namespace yolo_ros
{

cv::Scalar class_color(int class_id);

void draw_detections(cv::Mat & image, const std::vector<Detection> & detections);

void draw_poses(
  cv::Mat & image, const std::vector<PoseDetection> & poses,
  float keypoint_threshold = 0.3F);

void draw_obb_detections(cv::Mat & image, const std::vector<ObbDetection> & detections);

void draw_classification(cv::Mat & image, const std::string & label, float score);

}  // namespace yolo_ros

#endif  // YOLO_ROS__VISUALIZATION_HPP_
