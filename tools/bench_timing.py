#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Collect the ~/timing topic published by the C++ nodes.

Each message is [preprocess_ms, inference_ms, postprocess_ms]. The first frames
are dropped as warm-up, then P50/P95/P99 and throughput are printed and the raw
samples are appended to CSV.

Usage:
  python3 tools/bench_timing.py --samples 300 --warmup 20 \
      --topic /yolos_detector/timing --csv benchmark/results.csv --label threads=8
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
except ImportError as exc:  # pragma: no cover - environment check
    sys.exit(f"ROS 2 Python environment required: {exc}")


class TimingBench(Node):

    def __init__(self, args):
        super().__init__("bench_timing")
        self.args = args
        self.rows = []
        self.stamp_ns = []
        self.start = self.get_clock().now().nanoseconds
        self.sub = self.create_subscription(
            Float64MultiArray, args.topic, self.callback, 50)
        self.get_logger().info(
            f"waiting for {args.samples} samples on {args.topic}")

    def callback(self, msg):
        values = list(msg.data)
        if len(values) < 3:
            return
        self.rows.append(values[:3])
        self.stamp_ns.append(self.get_clock().now().nanoseconds)
        if len(self.rows) >= self.args.samples:
            self.finish()

    def finish(self):
        self.report()
        raise KeyboardInterrupt  # leave rclpy.spin() cleanly

    def report(self):
        warmup = min(self.args.warmup, len(self.rows) - 1)
        if len(self.rows) <= warmup:
            print("not enough samples to report")
            return
        stages = np.asarray(self.rows[warmup:], dtype=np.float64)
        total = stages.sum(axis=1)

        def percentiles(values):
            return [float(np.percentile(values, q)) for q in (50, 95, 99)]

        stages_p = [percentiles(stages[:, i]) for i in range(3)]
        total_p = percentiles(total)
        stamps = np.asarray(self.stamp_ns[warmup:], dtype=np.float64)
        span_s = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0
        throughput = (len(total) - 1) / span_s if span_s > 0 else 0.0

        print(f"samples={len(total)} (warmup {warmup})  "
              f"mean={total.mean():.1f} ms  "
              f"P50/P95/P99={total_p[0]:.1f}/{total_p[1]:.1f}/{total_p[2]:.1f} ms  "
              f"FPS(mean)={1000.0 / total.mean():.1f}  "
              f"FPS(topic)={throughput:.1f}")
        for name, values in zip(("preprocess", "inference", "postprocess"), stages_p):
            print(f"  {name:<11} P50/P95/P99="
                  f"{values[0]:.1f}/{values[1]:.1f}/{values[2]:.1f} ms")

        if not self.args.csv:
            return
        directory = os.path.dirname(self.args.csv)
        if directory:
            os.makedirs(directory, exist_ok=True)
        new_file = not os.path.exists(self.args.csv)
        with open(self.args.csv, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow([
                    "label", "sample", "preprocess_ms", "inference_ms",
                    "postprocess_ms", "total_ms"])
            for index, row in enumerate(stages):
                writer.writerow([self.args.label, index] + [
                    f"{value:.3f}" for value in row] + [f"{row.sum():.3f}"])
        print(f"appended {len(stages)} rows to {self.args.csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/yolos_detector/timing")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--csv", default="")
    parser.add_argument("--label", default="run")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    rclpy.init()
    node = TimingBench(args)

    def on_timeout():
        node.get_logger().warn(
            f"timeout after {len(node.rows)} samples; "
            "is the node activated with publish_timing:=true?")
        node.finish()

    node.create_timer(args.timeout, on_timeout)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
