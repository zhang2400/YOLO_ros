// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/segmentor_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

#include "yolo_ros/detection_decoder.hpp"
#include "yolo_ros/messages.hpp"
#include "yolo_ros/visualization.hpp"

namespace yolo_ros
{

SegmentorNode::SegmentorNode(const rclcpp::NodeOptions & options)
: YoloNodeBase("yolos_segmentor", options)
{
}

bool SegmentorNode::configure_task()
{
  detections_pub_ =
    create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
  mask_pub_ = create_publisher<sensor_msgs::msg::Image>("~/mask", 1);
  return true;
}

void SegmentorNode::activate_task()
{
  detections_pub_->on_activate();
  mask_pub_->on_activate();
}

void SegmentorNode::deactivate_task()
{
  detections_pub_->on_deactivate();
  mask_pub_->on_deactivate();
}

void SegmentorNode::cleanup_task()
{
  detections_pub_.reset();
  mask_pub_.reset();
}

void SegmentorNode::process(
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

  if (outputs.size() < 2) {
    if (!debug_canvas.empty()) {
      draw_detections(debug_canvas, detections);
    }
    return;
  }

  const cv::Mat masks =
    build_instance_masks(outputs[0], outputs[1], detections, geometry);
  mask_pub_->publish(*cv_bridge::CvImage(header, "bgr8", masks).toImageMsg());

  if (!debug_canvas.empty()) {
    cv::Mat mask_gray;
    cv::cvtColor(masks, mask_gray, cv::COLOR_BGR2GRAY);
    masks.copyTo(debug_canvas, mask_gray);
    draw_detections(debug_canvas, detections);
  }
}

}  // namespace yolo_ros
