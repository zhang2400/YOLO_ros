#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Publish a still image or a V4L2 camera stream for benchmarking.

image_publisher is convenient but its actual rate depends on DDS throughput;
this streamer pre-encodes the message once (still image) or reads frames from
the camera at capture rate, and stamps each frame at capture time so end-to-end
latency can be measured from the output message header.

Usage:
  python3 tools/image_stream.py --image test_data/dog_bike_car.jpg \
      --topic /yolos_detector/image_raw --rate 100
  python3 tools/image_stream.py --camera 0 --width 640 --height 480 \
      --topic /yolos_detector/image_raw --rate 25
"""

import argparse
import sys
import threading
import time

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

    def __init__(self, args):
        super().__init__("image_stream")
        self.capture = None
        self.latest = None
        self.lock = threading.Lock()
        self.running = True
        self.message = Image()
        self.message.encoding = "bgr8"
        self.message.header.frame_id = "camera"

        if args.camera >= 0:
            for _ in range(3):  # a just-released device can be briefly busy
                self.capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
                if self.capture.isOpened():
                    break
                time.sleep(0.5)
            if not self.capture.isOpened():
                sys.exit(f"cannot open camera /dev/video{args.camera}")
            if args.width > 0:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            if args.height > 0:
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            if args.rate > 0:
                self.capture.set(cv2.CAP_PROP_FPS, args.rate)
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_name = f"/dev/video{args.camera} ({width}x{height})"
            # Capture in its own thread: publishing blocks on DDS throughput and
            # must not throttle the camera. The timer only ever sends the newest
            # frame, so stale frames are dropped instead of queued.
            threading.Thread(target=self.capture_loop, daemon=True).start()
        else:
            source = cv2.imread(args.image, cv2.IMREAD_COLOR)
            if source is None:
                sys.exit(f"cannot read image: {args.image}")
            height, width = source.shape[:2]
            self.message.step = width * 3
            self.message.data = np.ascontiguousarray(source).tobytes()
            source_name = args.image

        self.message.height, self.message.width = height, width
        self.publisher = self.create_publisher(
            Image, args.topic, qos_profile_sensor_data)
        self.published = 0
        self.stats = args.stats
        self.report_start = time.time()
        self.create_timer(1.0 / args.rate, self.tick)
        self.get_logger().info(
            f"publishing {source_name} at {args.rate:.1f} Hz on {args.topic}")

    def capture_loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                continue
            stamp = self.get_clock().now().to_msg()  # capture time
            with self.lock:
                self.latest = (frame, stamp)
        self.capture.release()

    def tick(self):
        if self.capture is not None:
            with self.lock:
                latest, self.latest = self.latest, None
            if latest is None:
                return
            frame, self.message.header.stamp = latest
            self.message.height, self.message.width = frame.shape[:2]
            self.message.step = self.message.width * 3
            self.message.data = np.ascontiguousarray(frame).tobytes()
        else:
            self.message.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.message)
        self.published += 1
        if self.stats:
            now = time.time()
            if now - self.report_start >= 1.0:
                self.get_logger().info(
                    f"published {self.published / (now - self.report_start):.1f} FPS")
                self.report_start = now
                self.published = 0

    def destroy_node(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="", help="path to a still image")
    parser.add_argument("--camera", type=int, default=-1,
                        help="V4L2 camera index, e.g. 0 for /dev/video0")
    parser.add_argument("--topic", default="/yolos_detector/image_raw")
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--stats", action="store_true",
                        help="log the actual publish rate once per second")
    args = parser.parse_args()
    if args.camera < 0 and not args.image:
        parser.error("one of --image or --camera is required")

    rclpy.init()
    node = ImageStream(args)
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
