// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef YOLO_ROS__CLASSIFIER_NODE_HPP_
#define YOLO_ROS__CLASSIFIER_NODE_HPP_

#include <memory>
#include <vector>

#include <vision_msgs/msg/classification.hpp>

#include "yolo_ros/node_base.hpp"

namespace yolo_ros
{

class ClassifierNode : public YoloNodeBase
{
public:
  explicit ClassifierNode(const rclcpp::NodeOptions & options);

protected:
  bool configure_task() override;
  void activate_task() override;
  void deactivate_task() override;
  void cleanup_task() override;
  cv::Mat preprocess(const cv::Mat & image, Letterbox & geometry) override;
  void process(
    const cv::Mat & image,
    const std::vector<Tensor> & outputs,
    const Letterbox & geometry,
    const std_msgs::msg::Header & header,
    cv::Mat & debug_canvas) override;

private:
  rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Classification>::SharedPtr
    classification_pub_;
};

}  // namespace yolo_ros

#endif  // YOLO_ROS__CLASSIFIER_NODE_HPP_
