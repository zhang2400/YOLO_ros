// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/onnx_engine.hpp"

#include <stdexcept>
#include <string>

namespace yolo_ros
{

OnnxEngine::OnnxEngine(
  const std::string & model_path,
  bool use_cuda,
  int intra_op_threads,
  int input_size)
: env_(ORT_LOGGING_LEVEL_WARNING, "yolo_ros"), input_size_override_(input_size)
{
  options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (intra_op_threads > 0) {
    options_.SetIntraOpNumThreads(intra_op_threads);
  }

  if (use_cuda) {
    try {
      options_.AppendExecutionProvider("CUDA", {{"device_id", "0"}});
    } catch (const std::exception & e) {
      // GPU provider unavailable in this ONNX Runtime build; keep CPU execution.
      fprintf(stderr, "[yolo_ros] CUDA provider unavailable (%s), falling back to CPU\n", e.what());
    }
  }

  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options_);
  query_io_info();
}

OnnxEngine::~OnnxEngine() = default;

void OnnxEngine::query_io_info()
{
  Ort::AllocatorWithDefaultOptions allocator;

  const size_t num_inputs = session_->GetInputCount();
  const size_t num_outputs = session_->GetOutputCount();
  if (num_inputs != 1 || num_outputs < 1) {
    throw std::runtime_error("expected exactly one model input and at least one output");
  }

  input_names_.clear();
  output_names_.clear();
  for (size_t i = 0; i < num_inputs; ++i) {
    input_names_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    output_names_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
  }

  input_shape_ =
    session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

  // Symbolic spatial dimensions need an explicit size: exports differ per task
  // (640 detection, 1024 OBB, 224 classification), so 640 is only a fallback.
  const int fallback = input_size_override_ > 0 ? input_size_override_ : 640;
  int64_t channels = 3;
  if (input_shape_.size() == 4) {
    const int64_t d1 = input_shape_[1];
    const int64_t d3 = input_shape_[3];
    channels_first_ = (d1 == 3) || (d3 != 3 && d1 <= 4);
    if (channels_first_) {
      channels = d1 > 0 ? d1 : 3;
      dynamic_input_ = input_shape_[2] <= 0 || input_shape_[3] <= 0;
      input_height_ = input_shape_[2] > 0 ? static_cast<int>(input_shape_[2]) : fallback;
      input_width_ = input_shape_[3] > 0 ? static_cast<int>(input_shape_[3]) : fallback;
    } else {
      dynamic_input_ = input_shape_[1] <= 0 || input_shape_[2] <= 0;
      input_height_ = input_shape_[1] > 0 ? static_cast<int>(input_shape_[1]) : fallback;
      input_width_ = input_shape_[2] > 0 ? static_cast<int>(input_shape_[2]) : fallback;
    }
  }
  if (channels != 3) {
    throw std::runtime_error("only 3-channel image inputs are supported");
  }
  if (!dynamic_input_ && input_size_override_ > 0 &&
    (input_size_override_ != input_width_ || input_size_override_ != input_height_))
  {
    throw std::runtime_error(
            "input_size (" + std::to_string(input_size_override_) +
            ") does not match the model's fixed input (" +
            std::to_string(input_width_) + "x" + std::to_string(input_height_) + ")");
  }
}

Tensor OnnxEngine::infer(
  const std::vector<float> & input,
  const std::vector<int64_t> & shape)
{
  return infer_all(input, shape).front();
}

std::vector<Tensor> OnnxEngine::infer_all(
  const std::vector<float> & input,
  const std::vector<int64_t> & shape)
{
  Ort::MemoryInfo memory_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, const_cast<float *>(input.data()), input.size(),
    shape.data(), shape.size());

  const char * input_name = input_names_.front().c_str();
  std::vector<const char *> output_names;
  output_names.reserve(output_names_.size());
  for (const auto & name : output_names_) {
    output_names.push_back(name.c_str());
  }

  std::vector<Ort::Value> outputs = session_->Run(
    Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1,
    output_names.data(), output_names.size());

  if (outputs.empty()) {
    throw std::runtime_error("model produced no output");
  }

  std::vector<Tensor> results;
  results.reserve(outputs.size());
  for (auto & value : outputs) {
    Ort::TensorTypeAndShapeInfo info = value.GetTensorTypeAndShapeInfo();
    Tensor tensor;
    tensor.shape = info.GetShape();
    const size_t count = info.GetElementCount();
    const float * raw = value.GetTensorData<float>();
    tensor.data.assign(raw, raw + count);
    results.push_back(std::move(tensor));
  }
  return results;
}

}  // namespace yolo_ros
