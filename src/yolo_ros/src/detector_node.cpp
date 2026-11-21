// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/detector_node.hpp"

#include "yolo_ros/detection_decoder.hpp"
#include "yolo_ros/messages.hpp"
#include "yolo_ros/visualization.hpp"

namespace yolo_ros
{

DetectorNode::DetectorNode(const rclcpp::NodeOptions & options)
: YoloNodeBase("yolos_detector", options)
{
}

bool DetectorNode::configure_task()
{
  detections_pub_ =
    create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
  return true;
}

void DetectorNode::activate_task()
{
  detections_pub_->on_activate();
}

void DetectorNode::deactivate_task()
{
  detections_pub_->on_deactivate();
}

void DetectorNode::cleanup_task()
{
  detections_pub_.reset();
}

void DetectorNode::process(
  const cv::Mat & /*image*/,
  const std::vector<Tensor> & outputs,
  const Letterbox & geometry,
  const std_msgs::msg::Header & header,
  cv::Mat & debug_canvas)
{
  const std::vector<Detection> detections = decode_detections(
    outputs.front(), geometry, class_names_, conf_threshold_, nms_threshold_,
    max_detections_);

  detections_pub_->publish(to_detection_array(detections, header));

  if (!debug_canvas.empty()) {
    draw_detections(debug_canvas, detections);
  }
}

}  // namespace yolo_ros
