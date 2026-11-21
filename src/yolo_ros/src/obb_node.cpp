// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/obb_node.hpp"

#include "yolo_ros/detection_decoder.hpp"
#include "yolo_ros/messages.hpp"
#include "yolo_ros/visualization.hpp"

namespace yolo_ros
{

ObbNode::ObbNode(const rclcpp::NodeOptions & options)
: YoloNodeBase("yolos_obb", options)
{
}

bool ObbNode::configure_task()
{
  detections_pub_ =
    create_publisher<yolo_ros::msg::ObbDetection2DArray>("~/detections", 10);
  return true;
}

void ObbNode::activate_task()
{
  detections_pub_->on_activate();
}

void ObbNode::deactivate_task()
{
  detections_pub_->on_deactivate();
}

void ObbNode::cleanup_task()
{
  detections_pub_.reset();
}

void ObbNode::process(
  const cv::Mat & /*image*/,
  const std::vector<Tensor> & outputs,
  const Letterbox & geometry,
  const std_msgs::msg::Header & header,
  cv::Mat & debug_canvas)
{
  const std::vector<ObbDetection> detections = decode_obb(
    outputs.front(), geometry, class_names_, conf_threshold_, nms_threshold_,
    max_detections_);

  detections_pub_->publish(to_obb_array(detections, header));

  if (!debug_canvas.empty()) {
    draw_obb_detections(debug_canvas, detections);
  }
}

}  // namespace yolo_ros
