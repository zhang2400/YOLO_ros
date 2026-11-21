// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef YOLO_ROS__NODE_BASE_HPP_
#define YOLO_ROS__NODE_BASE_HPP_

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "yolo_ros/onnx_engine.hpp"
#include "yolo_ros/types.hpp"

namespace yolo_ros
{

/// Shared lifecycle + inference pipeline for all task nodes.
class YoloNodeBase : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit YoloNodeBase(
    const std::string & node_name, const rclcpp::NodeOptions & options);

  using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

protected:
  /// Create task-specific publishers. Return false to fail configure.
  virtual bool configure_task() {return true;}
  virtual void activate_task() {}
  virtual void deactivate_task() {}
  virtual void cleanup_task() {}

  /// Preprocess one BGR image into the model input (default: letterbox).
  virtual cv::Mat preprocess(const cv::Mat & image, Letterbox & geometry);

  /// Decode outputs, publish task messages, draw into the debug canvas.
  virtual void process(
    const cv::Mat & image,
    const std::vector<Tensor> & outputs,
    const Letterbox & geometry,
    const std_msgs::msg::Header & header,
    cv::Mat & debug_canvas) = 0;

  void declare_common_parameters();
  void load_common_parameters();
  void load_labels(const std::string & path);

  std::unique_ptr<OnnxEngine> engine_;
  std::vector<std::string> class_names_;

  std::string model_path_;
  std::string labels_path_;
  bool use_gpu_{false};
  bool publish_debug_image_{true};
  bool publish_timing_{false};
  float conf_threshold_{0.4F};
  float nms_threshold_{0.45F};
  int max_detections_{300};
  int num_threads_{0};
  int input_size_{0};  // 0 = use the model's static size (640 fallback if dynamic)

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr image_msg);

  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64MultiArray>::SharedPtr
    timing_pub_;
};

}  // namespace yolo_ros

#endif  // YOLO_ROS__NODE_BASE_HPP_
