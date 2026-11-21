#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Publish a still image at a fixed rate for benchmarking.

image_publisher is convenient but its actual rate depends on DDS throughput;
this streamer pre-encodes the message once and is enough to saturate the
detector on CPU-only test machines.

Usage:
  python3 tools/image_stream.py --image test_data/dog_bike_car.jpg \
      --topic /yolos_detector/image_raw --rate 100
"""

import argparse
import sys

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
except ImportError as exc:  # pragma: no cover - environment check
    sys.exit(f"ROS 2 Python environment required: {exc}")


class ImageStream(Node):

    def __init__(self, image, topic, rate):
        super().__init__("image_stream")
        source = cv2.imread(image, cv2.IMREAD_COLOR)
        if source is None:
            sys.exit(f"cannot read image: {image}")
        self.message = Image()
        self.message.height, self.message.width = source.shape[:2]
        self.message.encoding = "bgr8"
        self.message.step = self.message.width * 3
        self.message.data = np.ascontiguousarray(source).tobytes()
        self.message.header.frame_id = "camera"

        self.publisher = self.create_publisher(
            Image, topic, qos_profile_sensor_data)
        self.published = 0
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f"publishing {image} at {rate:.1f} Hz on {topic}")

    def tick(self):
        self.message.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.message)
        self.published += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--topic", default="/yolos_detector/image_raw")
    parser.add_argument("--rate", type=float, default=100.0)
    args = parser.parse_args()

    rclpy.init()
    node = ImageStream(args.image, args.topic, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"published {node.published} frames")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
