#!/usr/bin/env bash
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible detector benchmark: starts the lifecycle node, activates it,
# streams one still image at a fixed rate, and collects ~/timing samples.
#
# Usage:
#   tools/bench_detector.sh --threads 4 --samples 300 --rate 30 --label threads4
#   tools/bench_detector.sh --threads 1 --samples 300 --rate 30 \
#       --csv benchmark/results.csv --label threads1
#
# The node must be built first (colcon build --base-paths src). Results are
# printed and optionally appended to --csv as raw per-frame rows.

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS=0
SAMPLES=300
RATE=200.0
WARMUP=20
TIMEOUT=300
LABEL=""
CSV=""
DOMAIN=""
MODEL="$ROOT/models/yolo11n.onnx"
IMAGE="$ROOT/test_data/dog_bike_car.jpg"
LABELS="$ROOT/data/coco.names"
TOPIC=/yolos_detector

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) THREADS="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --csv) CSV="$2"; shift 2 ;;
    --domain) DOMAIN="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --topic) TOPIC="$2"; shift 2 ;;
    -h|--help) sed -n '3,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LABEL="${LABEL:-threads=$THREADS}"
[[ -n "$DOMAIN" ]] && export ROS_DOMAIN_ID="$DOMAIN"
source /opt/ros/humble/setup.bash
source "$ROOT/install/setup.bash"

if ! ros2 pkg prefix yolo_ros >/dev/null 2>&1; then
  echo "cannot find yolo_ros; build it first and/or check LD_LIBRARY_PATH" >&2
  exit 1
fi

cleanup() {
  # Kill the ros2 run wrapper *and* its node child; SIGKILL because lifecycle
  # nodes can otherwise linger and pollute the next run's measurements.
  if [[ -n "${NODE_PID:-}" ]]; then
    pkill -9 -P "$NODE_PID" 2>/dev/null || true
    kill -9 "$NODE_PID" 2>/dev/null || true
  fi
  [[ -n "${PUB_PID:-}" ]] && kill -9 "$PUB_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

if pgrep -x yolos_detector_node >/dev/null; then
  echo "another yolos_detector_node is already running; stop it first" >&2
  exit 1
fi

ros2 run yolo_ros yolos_detector_node --ros-args \
  -p model_path:="$MODEL" \
  -p labels_path:="$LABELS" \
  -p num_threads:="$THREADS" \
  -p publish_timing:=true > /tmp/yolo_bench_node.log 2>&1 &
NODE_PID=$!
sleep 2

ros2 lifecycle set "$TOPIC" configure >/dev/null
ros2 lifecycle set "$TOPIC" activate >/dev/null

python3 "$ROOT/tools/image_stream.py" \
  --image "$IMAGE" --topic "$TOPIC/image_raw" \
  --rate "$RATE" > /tmp/yolo_bench_publisher.log 2>&1 &
PUB_PID=$!
sleep 1

python3 "$ROOT/tools/bench_timing.py" \
  --topic "$TOPIC/timing" --samples "$SAMPLES" --warmup "$WARMUP" \
  --timeout "$TIMEOUT" --label "$LABEL" ${CSV:+--csv "$CSV"}
