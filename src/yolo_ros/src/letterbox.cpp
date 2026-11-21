// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include "yolo_ros/letterbox.hpp"

#include <algorithm>
#include <cstring>

#include <opencv2/imgproc.hpp>

namespace yolo_ros
{

cv::Mat letterbox(
  const cv::Mat & src,
  const cv::Size & target_size,
  Letterbox & geometry,
  const cv::Scalar & pad_color)
{
  geometry.input_width = target_size.width;
  geometry.input_height = target_size.height;
  geometry.image_width = src.cols;
  geometry.image_height = src.rows;

  const float scale = std::min(
    static_cast<float>(target_size.width) / static_cast<float>(src.cols),
    static_cast<float>(target_size.height) / static_cast<float>(src.rows));

  const int resized_w = std::max(1, static_cast<int>(src.cols * scale));
  const int resized_h = std::max(1, static_cast<int>(src.rows * scale));

  cv::Mat resized;
  cv::resize(src, resized, cv::Size(resized_w, resized_h));

  geometry.scale = scale;
  geometry.pad_x = (target_size.width - resized_w) / 2.0F;
  geometry.pad_y = (target_size.height - resized_h) / 2.0F;

  const int top = static_cast<int>(geometry.pad_y);
  const int bottom = target_size.height - resized_h - top;
  const int left = static_cast<int>(geometry.pad_x);
  const int right = target_size.width - resized_w - left;

  cv::Mat padded;
  cv::copyMakeBorder(
    resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, pad_color);
  return padded;
}

std::vector<float> to_blob(const cv::Mat & bgr, bool channels_first)
{
  const int h = bgr.rows;
  const int w = bgr.cols;
  std::vector<float> blob(static_cast<size_t>(3) * h * w);

  const int plane = h * w;
  for (int y = 0; y < h; ++y) {
    const cv::Vec3b * row = bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x) {
      const size_t pixel = static_cast<size_t>(y) * w + x;
      // cv_bridge delivers BGR; Ultralytics exports expect RGB, so swap here
      // instead of paying for an extra cvtColor on every frame.
      const float r = row[x][2] / 255.0F;
      const float g = row[x][1] / 255.0F;
      const float b = row[x][0] / 255.0F;
      if (channels_first) {
        blob[pixel] = r;
        blob[plane + pixel] = g;
        blob[2 * plane + pixel] = b;
      } else {
        blob[pixel * 3 + 0] = r;
        blob[pixel * 3 + 1] = g;
        blob[pixel * 3 + 2] = b;
      }
    }
  }
  return blob;
}

}  // namespace yolo_ros
