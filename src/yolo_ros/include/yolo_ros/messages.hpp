// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef YOLO_ROS__MESSAGES_HPP_
#define YOLO_ROS__MESSAGES_HPP_

#include <vector>

#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/classification.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "yolo_ros/msg/key_point2_d_array.hpp"
#include "yolo_ros/msg/obb_detection2_d_array.hpp"
#include "yolo_ros/types.hpp"

namespace yolo_ros
{

vision_msgs::msg::Detection2DArray to_detection_array(
  const std::vector<Detection> & detections,
  const std_msgs::msg::Header & header);

vision_msgs::msg::Detection2DArray to_detection_array(
  const std::vector<PoseDetection> & poses,
  const std_msgs::msg::Header & header);

yolo_ros::msg::KeyPoint2DArray to_keypoint_array(
  const std::vector<PoseDetection> & poses,
  const std_msgs::msg::Header & header);

yolo_ros::msg::ObbDetection2DArray to_obb_array(
  const std::vector<ObbDetection> & detections,
  const std_msgs::msg::Header & header);

vision_msgs::msg::Classification to_classification(
  const std::vector<std::pair<int, float>> & top_k,
  const std::vector<std::string> & class_names,
  const std_msgs::msg::Header & header);

}  // namespace yolo_ros

#endif  // YOLO_ROS__MESSAGES_HPP_
