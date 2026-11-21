// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/messages.hpp"

#include <string>
#include <utility>

#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include "yolo_ros/msg/key_point2_d.hpp"
#include "yolo_ros/msg/obb_detection2_d.hpp"
#include "yolo_ros/msg/oriented_box2_d.hpp"

namespace yolo_ros
{

namespace
{

std::string label_for(int class_id, const std::vector<std::string> & class_names)
{
  return (class_id >= 0 && static_cast<size_t>(class_id) < class_names.size()) ?
         class_names[class_id] : std::to_string(class_id);
}

}  // namespace

vision_msgs::msg::Detection2DArray to_detection_array(
  const std::vector<Detection> & detections,
  const std_msgs::msg::Header & header)
{
  vision_msgs::msg::Detection2DArray message;
  message.header = header;
  message.detections.reserve(detections.size());

  for (const auto & det : detections) {
    vision_msgs::msg::Detection2D out;
    out.header = header;
    out.bbox.center.position.x = det.box.x + det.box.width / 2.0;
    out.bbox.center.position.y = det.box.y + det.box.height / 2.0;
    out.bbox.size_x = det.box.width;
    out.bbox.size_y = det.box.height;

    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id =
      det.label.empty() ? std::to_string(det.class_id) : det.label;
    hypothesis.hypothesis.score = det.score;
    out.results.push_back(hypothesis);

    message.detections.push_back(std::move(out));
  }
  return message;
}

vision_msgs::msg::Detection2DArray to_detection_array(
  const std::vector<PoseDetection> & poses,
  const std_msgs::msg::Header & header)
{
  std::vector<Detection> detections;
  detections.reserve(poses.size());
  for (const auto & pose : poses) {
    detections.push_back(pose.det);
  }
  return to_detection_array(detections, header);
}

yolo_ros::msg::KeyPoint2DArray to_keypoint_array(
  const std::vector<PoseDetection> & poses,
  const std_msgs::msg::Header & header)
{
  yolo_ros::msg::KeyPoint2DArray message;
  message.header = header;

  for (size_t person = 0; person < poses.size(); ++person) {
    for (const auto & keypoint : poses[person].keypoints) {
      yolo_ros::msg::KeyPoint2D out;
      out.x = keypoint.x;
      out.y = keypoint.y;
      out.confidence = keypoint.confidence;
      out.id = keypoint.id;
      message.keypoints.push_back(out);
      message.person_index.push_back(static_cast<int32_t>(person));
    }
  }
  return message;
}

yolo_ros::msg::ObbDetection2DArray to_obb_array(
  const std::vector<ObbDetection> & detections,
  const std_msgs::msg::Header & header)
{
  yolo_ros::msg::ObbDetection2DArray message;
  message.header = header;
  message.detections.reserve(detections.size());

  for (const auto & det : detections) {
    yolo_ros::msg::ObbDetection2D out;
    out.header = header;
    out.bbox.center.x = det.center_x;
    out.bbox.center.y = det.center_y;
    out.bbox.size_x = det.width;
    out.bbox.size_y = det.height;
    out.bbox.theta = det.angle;
    out.class_id = det.class_id;
    out.class_name = det.label;
    out.score = det.score;
    message.detections.push_back(std::move(out));
  }
  return message;
}

vision_msgs::msg::Classification to_classification(
  const std::vector<std::pair<int, float>> & top_k,
  const std::vector<std::string> & class_names,
  const std_msgs::msg::Header & header)
{
  vision_msgs::msg::Classification message;
  message.header = header;
  message.results.reserve(top_k.size());

  for (const auto & [class_id, score] : top_k) {
    vision_msgs::msg::ObjectHypothesis hypothesis;
    hypothesis.class_id = label_for(class_id, class_names);
    hypothesis.score = score;
    message.results.push_back(std::move(hypothesis));
  }
  return message;
}

}  // namespace yolo_ros
