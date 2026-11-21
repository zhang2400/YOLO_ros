// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/node_base.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>

#include "yolo_ros/letterbox.hpp"

namespace yolo_ros
{

using namespace std::chrono_literals;

YoloNodeBase::YoloNodeBase(
  const std::string & node_name, const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode(node_name, options)
{
  declare_common_parameters();
  RCLCPP_INFO(get_logger(), "Node created (lifecycle: unconfigured)");
}

void YoloNodeBase::declare_common_parameters()
{
  declare_parameter("model_path", std::string{""});
  declare_parameter("labels_path", std::string{""});
  declare_parameter("use_gpu", false);
  declare_parameter("conf_threshold", 0.4);
  declare_parameter("nms_threshold", 0.45);
  declare_parameter("max_detections", 300);
  declare_parameter("num_threads", 0);
  declare_parameter("input_size", 0);
  declare_parameter("publish_debug_image", true);
  declare_parameter("debug_image_rate", 5.0);
  declare_parameter("publish_timing", false);
}

void YoloNodeBase::load_common_parameters()
{
  model_path_ = get_parameter("model_path").as_string();
  labels_path_ = get_parameter("labels_path").as_string();
  use_gpu_ = get_parameter("use_gpu").as_bool();
  conf_threshold_ = static_cast<float>(get_parameter("conf_threshold").as_double());
  nms_threshold_ = static_cast<float>(get_parameter("nms_threshold").as_double());
  max_detections_ = static_cast<int>(get_parameter("max_detections").as_int());
  num_threads_ = static_cast<int>(get_parameter("num_threads").as_int());
  input_size_ = static_cast<int>(get_parameter("input_size").as_int());
  publish_debug_image_ = get_parameter("publish_debug_image").as_bool();
  debug_image_rate_ = get_parameter("debug_image_rate").as_double();
  publish_timing_ = get_parameter("publish_timing").as_bool();
}

void YoloNodeBase::load_labels(const std::string & path)
{
  class_names_.clear();
  if (path.empty()) {
    return;
  }

  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open labels file: " + path);
  }

  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    class_names_.push_back(line);
  }
  RCLCPP_INFO(get_logger(), "Loaded %zu class names", class_names_.size());
}

YoloNodeBase::CallbackReturn YoloNodeBase::on_configure(
  const rclcpp_lifecycle::State & /*state*/)
{
  load_common_parameters();

  if (model_path_.empty()) {
    RCLCPP_ERROR(get_logger(), "Parameter 'model_path' is required");
    return CallbackReturn::FAILURE;
  }

  try {
    load_labels(labels_path_);
    engine_ = std::make_unique<OnnxEngine>(
      model_path_, use_gpu_, num_threads_, input_size_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize inference engine: %s", e.what());
    engine_.reset();
    return CallbackReturn::FAILURE;
  }

  if (engine_->dynamic_input() && input_size_ <= 0) {
    RCLCPP_WARN(
      get_logger(),
      "Model has dynamic input dimensions; using the %dx%d fallback. "
      "Set the 'input_size' parameter for other export resolutions.",
      engine_->input_width(), engine_->input_height());
  }

  RCLCPP_INFO(
    get_logger(), "Model loaded: %s (input %dx%d, %s)",
    model_path_.c_str(), engine_->input_width(), engine_->input_height(),
    engine_->channels_first() ? "NCHW" : "NHWC");

  if (publish_debug_image_) {
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("~/debug_image", 1);
  }
  if (publish_timing_) {
    timing_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/timing", 10);
  }

  if (!configure_task()) {
    RCLCPP_ERROR(get_logger(), "Task configuration failed");
    return CallbackReturn::FAILURE;
  }
  return CallbackReturn::SUCCESS;
}

YoloNodeBase::CallbackReturn YoloNodeBase::on_activate(
  const rclcpp_lifecycle::State & /*state*/)
{
  if (debug_pub_) {
    debug_pub_->on_activate();
  }
  if (timing_pub_) {
    timing_pub_->on_activate();
  }
  activate_task();

  callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = callback_group_;
  // Drop stale frames instead of queueing them: with a 25-30 Hz camera and
  // ~70 ms inference, a deep queue only adds latency, not throughput.
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "~/image_raw", rclcpp::SensorDataQoS().keep_last(1),
    std::bind(&YoloNodeBase::image_callback, this, std::placeholders::_1),
    sub_options);

  RCLCPP_INFO(get_logger(), "Activated, waiting for images on ~/image_raw");
  return CallbackReturn::SUCCESS;
}

YoloNodeBase::CallbackReturn YoloNodeBase::on_deactivate(
  const rclcpp_lifecycle::State & /*state*/)
{
  image_sub_.reset();
  deactivate_task();
  if (debug_pub_) {
    debug_pub_->on_deactivate();
  }
  if (timing_pub_) {
    timing_pub_->on_deactivate();
  }
  RCLCPP_INFO(get_logger(), "Deactivated");
  return CallbackReturn::SUCCESS;
}

YoloNodeBase::CallbackReturn YoloNodeBase::on_cleanup(
  const rclcpp_lifecycle::State & /*state*/)
{
  image_sub_.reset();
  cleanup_task();
  debug_pub_.reset();
  timing_pub_.reset();
  engine_.reset();
  class_names_.clear();
  RCLCPP_INFO(get_logger(), "Cleaned up");
  return CallbackReturn::SUCCESS;
}

YoloNodeBase::CallbackReturn YoloNodeBase::on_shutdown(
  const rclcpp_lifecycle::State & state)
{
  return on_cleanup(state);
}

cv::Mat YoloNodeBase::preprocess(const cv::Mat & image, Letterbox & geometry)
{
  return letterbox(
    image, cv::Size(engine_->input_width(), engine_->input_height()), geometry);
}

void YoloNodeBase::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr image_msg)
{
  if (!engine_) {
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception &) {
    try {
      cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge conversion failed: %s", e.what());
      return;
    }
  }

  const auto t0 = std::chrono::steady_clock::now();

  Letterbox geometry;
  const cv::Mat input = preprocess(cv_ptr->image, geometry);
  const std::vector<float> blob = to_blob(input, engine_->channels_first());

  const auto t1 = std::chrono::steady_clock::now();

  std::vector<Tensor> outputs;
  try {
    outputs = engine_->infer_all(
      blob, {1, 3, engine_->input_height(), engine_->input_width()});
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000, "Inference failed: %s", e.what());
    return;
  }

  const auto t2 = std::chrono::steady_clock::now();

  std_msgs::msg::Header header;
  header.stamp = image_msg->header.stamp;
  header.frame_id = image_msg->header.frame_id;

  cv::Mat debug_canvas;
  if (publish_debug_image_ && debug_pub_ && debug_pub_->is_activated()) {
    // Debug images are for humans: publishing a 640x480 frame every inference
    // can cost more than the inference itself on a slow DDS transport, so
    // throttle it. debug_image_rate <= 0 restores the publish-every-frame mode.
    const int64_t now_ns = now().nanoseconds();
    const int64_t period_ns =
      debug_image_rate_ > 0.0 ? static_cast<int64_t>(1e9 / debug_image_rate_) : 0;
    if (last_debug_ns_ == 0 || period_ns == 0 || now_ns - last_debug_ns_ >= period_ns) {
      debug_canvas = cv_ptr->image.clone();
      last_debug_ns_ = now_ns;
    }
  }

  try {
    process(cv_ptr->image, outputs, geometry, header, debug_canvas);
  } catch (const std::exception & e) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000, "Postprocessing failed: %s", e.what());
    return;
  }

  const auto t3 = std::chrono::steady_clock::now();

  if (!debug_canvas.empty()) {
    debug_pub_->publish(*cv_bridge::CvImage(header, "bgr8", debug_canvas).toImageMsg());
  }

  if (timing_pub_ && timing_pub_->is_activated()) {
    std_msgs::msg::Float64MultiArray timing;
    timing.data = {
      std::chrono::duration<double, std::milli>(t1 - t0).count(),
      std::chrono::duration<double, std::milli>(t2 - t1).count(),
      std::chrono::duration<double, std::milli>(t3 - t2).count()};
    timing_pub_->publish(timing);
  }
}

}  // namespace yolo_ros
