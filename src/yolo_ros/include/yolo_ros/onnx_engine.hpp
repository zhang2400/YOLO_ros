// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#ifndef YOLO_ROS__ONNX_ENGINE_HPP_
#define YOLO_ROS__ONNX_ENGINE_HPP_

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "yolo_ros/types.hpp"

namespace yolo_ros
{

/// Thin RAII wrapper around a single-input ONNX Runtime session.
class OnnxEngine
{
public:
  OnnxEngine(
    const std::string & model_path,
    bool use_cuda = false,
    int intra_op_threads = 0,
    int input_size = 0);

  ~OnnxEngine();

  OnnxEngine(const OnnxEngine &) = delete;
  OnnxEngine & operator=(const OnnxEngine &) = delete;

  /// Run synchronous inference, returning every model output.
  std::vector<Tensor> infer_all(
    const std::vector<float> & input, const std::vector<int64_t> & shape);

  /// Convenience for single-output models.
  Tensor infer(
    const std::vector<float> & input, const std::vector<int64_t> & shape);

  const std::vector<int64_t> & input_shape() const { return input_shape_; }
  int input_width() const { return input_width_; }
  int input_height() const { return input_height_; }
  bool channels_first() const { return channels_first_; }
  /// True when the model declares symbolic spatial dimensions.
  bool dynamic_input() const { return dynamic_input_; }

private:
  void query_io_info();

  Ort::Env env_;
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<int64_t> input_shape_;
  int input_width_{640};
  int input_height_{640};
  int input_size_override_{0};
  bool channels_first_{true};
  bool dynamic_input_{false};
};

}  // namespace yolo_ros

#endif  // YOLO_ROS__ONNX_ENGINE_HPP_
