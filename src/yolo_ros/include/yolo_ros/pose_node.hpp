// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef YOLO_ROS__POSE_NODE_HPP_
#define YOLO_ROS__POSE_NODE_HPP_

#include <memory>
#include <vector>

#include <vision_msgs/msg/detection2_d_array.hpp>

#include "yolo_ros/msg/key_point2_d_array.hpp"
#include "yolo_ros/node_base.hpp"

namespace yolo_ros
{

class PoseNode : public YoloNodeBase
{
public:
  explicit PoseNode(const rclcpp::NodeOptions & options);

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
  rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Detection2DArray>::SharedPtr
    detections_pub_;
  rclcpp_lifecycle::LifecyclePublisher<yolo_ros::msg::KeyPoint2DArray>::SharedPtr
    keypoints_pub_;
  float keypoint_threshold_{0.3F};
};

}  // namespace yolo_ros

#endif  // YOLO_ROS__POSE_NODE_HPP_
