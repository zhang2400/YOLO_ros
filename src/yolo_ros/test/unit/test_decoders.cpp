// Copyright 2026
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "yolo_ros/detection_decoder.hpp"
#include "yolo_ros/letterbox.hpp"

namespace yolo_ros
{
namespace
{

Tensor make_tensor(std::vector<int64_t> shape)
{
  Tensor tensor;
  tensor.shape = std::move(shape);
  size_t count = 1;
  for (int64_t dim : tensor.shape) {
    count *= static_cast<size_t>(dim);
  }
  tensor.data.assign(count, 0.0F);
  return tensor;
}

void set_channel(Tensor & tensor, int64_t channel, int64_t anchor, float value)
{
  const int64_t channels = tensor.shape[1];
  const int64_t anchors = tensor.shape[2];
  tensor.data[static_cast<size_t>(channel * anchors + anchor)] = value;
}

Letterbox identity_geometry()
{
  Letterbox geometry;
  geometry.input_width = 640;
  geometry.input_height = 640;
  geometry.image_width = 640;
  geometry.image_height = 640;
  geometry.scale = 1.0F;
  geometry.pad_x = 0.0F;
  geometry.pad_y = 0.0F;
  return geometry;
}

TEST(Letterbox, ToBlobWritesRgbFromBgrInput)
{
  // cv_bridge hands us BGR: (B, G, R) = (10, 20, 30).
  const cv::Mat bgr(1, 2, CV_8UC3, cv::Scalar(10, 20, 30));

  const std::vector<float> planar = to_blob(bgr, true);
  ASSERT_EQ(planar.size(), 6u);
  EXPECT_NEAR(planar[0], 30.0F / 255.0F, 1e-6);  // R plane
  EXPECT_NEAR(planar[2], 20.0F / 255.0F, 1e-6);  // G plane
  EXPECT_NEAR(planar[4], 10.0F / 255.0F, 1e-6);  // B plane

  const std::vector<float> interleaved = to_blob(bgr, false);
  ASSERT_EQ(interleaved.size(), 6u);
  EXPECT_NEAR(interleaved[0], 30.0F / 255.0F, 1e-6);
  EXPECT_NEAR(interleaved[1], 20.0F / 255.0F, 1e-6);
  EXPECT_NEAR(interleaved[2], 10.0F / 255.0F, 1e-6);
}

TEST(DetectionDecoder, TruncationKeepsTopScoresDeterministically)
{
  const std::vector<std::string> names{"alpha", "beta"};
  Tensor output = make_tensor({1, 6, 20});
  const float scores[4] = {0.6F, 0.9F, 0.7F, 0.8F};
  const int classes[4] = {0, 1, 0, 1};
  for (int64_t a = 0; a < 4; ++a) {
    set_channel(output, 0, a, 100.0F + 100.0F * static_cast<float>(a));
    set_channel(output, 1, a, 100.0F);
    set_channel(output, 2, a, 10.0F);
    set_channel(output, 3, a, 10.0F);
    set_channel(output, 4 + classes[a], a, scores[a]);
  }

  // With max_detections smaller than the candidate count, the survivor set
  // must always be the globally highest scores, independent of hash order.
  for (int run = 0; run < 20; ++run) {
    const auto dets =
      decode_detections(output, identity_geometry(), names, 0.4F, 0.45F, 2);
    ASSERT_EQ(dets.size(), 2u);
    EXPECT_NEAR(dets[0].score, 0.9F, 1e-5);
    EXPECT_NEAR(dets[1].score, 0.8F, 1e-5);
    EXPECT_EQ(dets[0].class_id, 1);
    EXPECT_EQ(dets[1].class_id, 1);
  }
}

TEST(DetectionDecoder, DecodesSingleBox)
{
  const std::vector<std::string> names{"alpha", "beta"};
  // 2 classes -> 6 channels, 10 anchors.
  Tensor output = make_tensor({1, 6, 10});
  const int64_t anchor = 3;
  set_channel(output, 0, anchor, 100.0F);
  set_channel(output, 1, anchor, 120.0F);
  set_channel(output, 2, anchor, 40.0F);
  set_channel(output, 3, anchor, 20.0F);
  set_channel(output, 4, anchor, 0.1F);
  set_channel(output, 5, anchor, 0.9F);

  const auto dets = decode_detections(output, identity_geometry(), names, 0.4F, 0.45F);
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].class_id, 1);
  EXPECT_NEAR(dets[0].score, 0.9F, 1e-5);
  EXPECT_NEAR(dets[0].box.x, 80.0F, 1e-3);
  EXPECT_NEAR(dets[0].box.y, 110.0F, 1e-3);
  EXPECT_NEAR(dets[0].box.width, 40.0F, 1e-3);
  EXPECT_NEAR(dets[0].box.height, 20.0F, 1e-3);
  EXPECT_EQ(dets[0].anchor, anchor);
}

TEST(DetectionDecoder, SuppressesSameClassButKeepsDifferentClasses)
{
  const std::vector<std::string> names{"alpha", "beta"};
  Tensor output = make_tensor({1, 6, 10});

  for (int64_t anchor : {0, 1}) {
    set_channel(output, 0, anchor, 100.0F);
    set_channel(output, 1, anchor, 100.0F);
    set_channel(output, 2, anchor, 50.0F);
    set_channel(output, 3, anchor, 50.0F);
    set_channel(output, 4, anchor, 0.9F);  // same class
  }
  set_channel(output, 2, 1, 48.0F);  // heavily overlapping box

  const auto dets = decode_detections(output, identity_geometry(), names, 0.4F, 0.45F);
  EXPECT_EQ(dets.size(), 1u);

  set_channel(output, 4, 1, 0.1F);  // first box drops to class 1
  set_channel(output, 5, 1, 0.8F);
  const auto dets2 = decode_detections(output, identity_geometry(), names, 0.4F, 0.45F);
  EXPECT_EQ(dets2.size(), 2u);
}

TEST(PoseDecoder, ReadsKeypoints)
{
  const std::vector<std::string> names{"person"};
  const int64_t anchors = 100;
  // 4 box + 1 class + 17 * 3 keypoints = 56 channels.
  Tensor output = make_tensor({1, 56, anchors});
  set_channel(output, 0, 0, 300.0F);
  set_channel(output, 1, 0, 300.0F);
  set_channel(output, 2, 0, 100.0F);
  set_channel(output, 3, 0, 200.0F);
  set_channel(output, 4, 0, 0.95F);
  // keypoint 0 at (310, 280), keypoint 5 at (350, 400)
  set_channel(output, 5, 0, 310.0F);
  set_channel(output, 6, 0, 280.0F);
  set_channel(output, 7, 0, 0.9F);
  set_channel(output, 5 + 3 * 5, 0, 350.0F);
  set_channel(output, 5 + 3 * 5 + 1, 0, 400.0F);
  set_channel(output, 5 + 3 * 5 + 2, 0, 0.8F);

  const auto poses =
    decode_pose(output, identity_geometry(), names, 0.4F, 0.45F, 17, 0.3F);
  ASSERT_EQ(poses.size(), 1u);
  ASSERT_EQ(poses[0].keypoints.size(), 17u);
  EXPECT_NEAR(poses[0].keypoints[0].x, 310.0F, 1e-3);
  EXPECT_NEAR(poses[0].keypoints[0].y, 280.0F, 1e-3);
  EXPECT_NEAR(poses[0].keypoints[5].x, 350.0F, 1e-3);
  EXPECT_NEAR(poses[0].keypoints[5].y, 400.0F, 1e-3);
}

TEST(ObbDecoder, ReadsAngle)
{
  const std::vector<std::string> names{"plane", "ship"};
  // 4 box + 2 class + 1 angle = 7 channels.
  Tensor output = make_tensor({1, 7, 50});
  set_channel(output, 0, 0, 500.0F);
  set_channel(output, 1, 0, 400.0F);
  set_channel(output, 2, 0, 120.0F);
  set_channel(output, 3, 0, 60.0F);
  set_channel(output, 5, 0, 0.8F);  // class 1
  set_channel(output, 6, 0, 0.785F);  // 45 degrees

  const auto dets = decode_obb(output, identity_geometry(), names, 0.4F, 0.45F);
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].class_id, 1);
  EXPECT_NEAR(dets[0].center_x, 500.0F, 1e-3);
  EXPECT_NEAR(dets[0].angle, 0.785F, 1e-5);
}

}  // namespace
}  // namespace yolo_ros
