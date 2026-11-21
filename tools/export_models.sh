#!/usr/bin/env bash
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible ONNX export for the models used by yolo_ros.
# Requires: pip install ultralytics==8.3.40 onnx onnxruntime
#
# The channel order matters: Ultralytics exports expect RGB, and to_blob()
# converts the BGR image from cv_bridge accordingly.

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/models"
mkdir -p "$OUT"

if ! command -v yolo >/dev/null 2>&1; then
  echo "ultralytics CLI not found: pip install ultralytics==8.3.40" >&2
  exit 1
fi
python3 -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

cd "$OUT"

# Detection: dynamic H/W so input_size can be chosen at runtime.
yolo export model=yolo11n.pt format=onnx opset=17 dynamic=True imgsz=640

# Segmentation and pose: fixed 640.
yolo export model=yolo11n-seg.pt format=onnx opset=17 imgsz=640
yolo export model=yolo11n-pose.pt format=onnx opset=17 imgsz=640

# OBB and classification use their own native resolutions.
yolo export model=yolo11n-obb.pt format=onnx opset=17 imgsz=1024
yolo export model=yolo11n-cls.pt format=onnx opset=17 imgsz=224

ls -lh "$OUT"/*.onnx
