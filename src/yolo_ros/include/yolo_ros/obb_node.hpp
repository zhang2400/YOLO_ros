// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef YOLO_ROS__OBB_NODE_HPP_
#define YOLO_ROS__OBB_NODE_HPP_

#include <memory>
#include <vector>

#include "yolo_ros/msg/obb_detection2_d_array.hpp"
#include "yolo_ros/node_base.hpp"

namespace yolo_ros
{

class ObbNode : public YoloNodeBase
{
public:
  explicit ObbNode(const rclcpp::NodeOptions & options);

protected:
  bool configure_task() override;
  void activate_task() override;
  void deactivate_task() override;
  void cleanup_task() override;
  void process(
    const cv::Mat & image,
    const std::vector<Tensor> & outputs,
    const Letterbox & geometry,
    const std_msgs::msg::Header & header,
    cv::Mat & debug_canvas) override;

private:
  rclcpp_lifecycle::LifecyclePublisher<yolo_ros::msg::ObbDetection2DArray>::SharedPtr
    detections_pub_;
};

}  // namespace yolo_ros

#endif  // YOLO_ROS__OBB_NODE_HPP_
