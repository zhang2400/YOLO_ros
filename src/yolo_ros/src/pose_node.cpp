// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/pose_node.hpp"

#include "yolo_ros/detection_decoder.hpp"
#include "yolo_ros/messages.hpp"
#include "yolo_ros/visualization.hpp"

namespace yolo_ros
{

PoseNode::PoseNode(const rclcpp::NodeOptions & options)
: YoloNodeBase("yolos_pose", options)
{
  declare_parameter("keypoint_threshold", 0.3);
}

bool PoseNode::configure_task()
{
  keypoint_threshold_ =
    static_cast<float>(get_parameter("keypoint_threshold").as_double());
  detections_pub_ =
    create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
  keypoints_pub_ =
    create_publisher<yolo_ros::msg::KeyPoint2DArray>("~/keypoints", 10);
  return true;
}

void PoseNode::activate_task()
{
  detections_pub_->on_activate();
  keypoints_pub_->on_activate();
}

void PoseNode::deactivate_task()
{
  detections_pub_->on_deactivate();
  keypoints_pub_->on_deactivate();
}

void PoseNode::cleanup_task()
{
  detections_pub_.reset();
  keypoints_pub_.reset();
}

void PoseNode::process(
  const cv::Mat & /*image*/,
  const std::vector<Tensor> & outputs,
  const Letterbox & geometry,
  const std_msgs::msg::Header & header,
  cv::Mat & debug_canvas)
{
  const std::vector<PoseDetection> poses = decode_pose(
    outputs.front(), geometry, class_names_, conf_threshold_, nms_threshold_,
    17, keypoint_threshold_, max_detections_);

  detections_pub_->publish(to_detection_array(poses, header));
  keypoints_pub_->publish(to_keypoint_array(poses, header));

  if (!debug_canvas.empty()) {
    draw_poses(debug_canvas, poses, keypoint_threshold_);
  }
}

}  // namespace yolo_ros
