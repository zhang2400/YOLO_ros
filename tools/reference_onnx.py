#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Python ONNX Runtime reference for the C++ nodes.

The point is regression, not a second implementation: preprocessing replicates
letterbox.cpp and to_blob() (BGR input, RGB blob, /255, NCHW), and the decoder
mirrors detection_decoder.cpp closely enough that boxes/scores can be compared
with the ROS topics at 1e-3.

Usage:
  python3 tools/reference_onnx.py --model models/yolo11n.onnx \
      --image test_data/dog_bike_car.jpg --labels data/coco.names
  python3 tools/reference_onnx.py --model models/yolo11n-cls.onnx \
      --image test_data/dog_bike_car.jpg --labels data/imagenet.names --mode classify
  # --color bgr reproduces the pre-fix behaviour for comparison
"""

import argparse
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(image, size):
    """Same geometry as letterbox.cpp (integer resize and padding)."""
    h, w = image.shape[:2]
    tw, th = size
    scale = min(tw / w, th / h)
    rw = max(1, int(w * scale))
    rh = max(1, int(h * scale))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_x = (tw - rw) / 2.0
    pad_y = (th - rh) / 2.0
    top, left = int(pad_y), int(pad_x)
    padded = cv2.copyMakeBorder(
        resized, top, th - rh - top, left, tw - rw - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, pad_x, pad_y


def to_blob(image, color):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if color == "rgb" else image
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def nms_class_aware(boxes, scores, classes, conf, iou, max_det):
    kept = []
    for class_id in sorted(set(int(c) for c in classes)):
        idx = np.where(np.asarray(classes) == class_id)[0]
        rects = [[float(v) for v in boxes[i]] for i in idx]
        sel = cv2.dnn.NMSBoxes(rects, [float(scores[i]) for i in idx], conf, iou)
        for k in np.asarray(sel).flatten():
            i = int(idx[int(k)])
            kept.append((float(scores[i]), int(classes[i]), boxes[i]))
    kept.sort(key=lambda item: (-item[0], item[2][1], item[2][0]))
    return kept[:max_det]


def decode_detect(output, scale, pad_x, pad_y, img_w, img_h, conf, iou, max_det):
    data = output
    if data.ndim == 3:
        data = data[0]
    if data.shape[0] < data.shape[1]:
        data = data.T  # [N, 4 + classes]
    inv = 1.0 / scale if scale > 0 else 0.0
    boxes, scores, classes = [], [], []
    for row in data:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < conf:
            continue
        cx, cy, bw, bh = (float(v) for v in row[:4])
        x = (cx - pad_x) * inv
        y = (cy - pad_y) * inv
        w, h = bw * inv, bh * inv
        left = max(0.0, x - w / 2.0)
        top = max(0.0, y - h / 2.0)
        right = min(float(img_w), x + w / 2.0)
        bottom = min(float(img_h), y + h / 2.0)
        if right <= left or bottom <= top:
            continue
        boxes.append([left, top, right - left, bottom - top])
        scores.append(score)
        classes.append(class_id)
    return nms_class_aware(boxes, scores, classes, conf, iou, max_det)


def softmax(values):
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.min() >= 0.0 and values.max() <= 1.0 and abs(values.sum() - 1.0) < 0.05:
        return values
    exp = np.exp(values - values.max())
    return exp / exp.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--mode", choices=["detect", "classify", "raw"], default="detect")
    parser.add_argument("--color", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=0,
                        help="square input for dynamic models (default 640)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--bench", type=int, default=0,
                        help="time N inference calls instead of printing results")
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    size = (args.input_size or 640, args.input_size or 640)
    if isinstance(shape[2], int) and shape[2] > 0:
        size = (int(shape[3]), int(shape[2]))
    elif args.input_size <= 0:
        size = (640, 640)

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"cannot read image: {args.image}")

    if args.mode == "classify":
        side = min(image.shape[:2])
        y = (image.shape[0] - side) // 2
        x = (image.shape[1] - side) // 2
        square = image[y:y + side, x:x + side]
        model_input = cv2.resize(square, size, interpolation=cv2.INTER_LINEAR)
        meta = {"scale": size[0] / side, "pad_x": 0.0, "pad_y": 0.0}
    else:
        model_input, scale, pad_x, pad_y = letterbox(image, size)
        meta = {"scale": scale, "pad_x": pad_x, "pad_y": pad_y}

    blob = to_blob(model_input, args.color)

    if args.bench:
        import time
        for _ in range(args.warmup):
            session.run(None, {input_meta.name: blob})
        samples = []
        for _ in range(args.bench):
            start = time.perf_counter()
            session.run(None, {input_meta.name: blob})
            samples.append((time.perf_counter() - start) * 1000.0)
        samples.sort()
        count = len(samples)
        pick = lambda q: samples[min(count - 1, int(count * q / 100.0))]
        print(f"python onnxruntime  samples={count}  "
              f"P50/P95/P99={pick(50):.1f}/{pick(95):.1f}/{pick(99):.1f} ms  "
              f"FPS={1000.0 / (sum(samples) / count):.1f}")
        return

    output = session.run(None, {input_meta.name: blob})[0]

    if args.mode == "classify":
        probs = softmax(output)
        order = np.argsort(-probs)[:args.top_k]
        names = []
        if args.labels:
            with open(args.labels, encoding="utf-8") as handle:
                names = [line.strip() for line in handle]
        results = [
            {"class_id": int(i), "label": names[i] if i < len(names) else str(i),
             "score": float(probs[i])}
            for i in order
        ]
    elif args.mode == "detect":
        dets = decode_detect(
            output, meta["scale"], meta["pad_x"], meta["pad_y"],
            image.shape[1], image.shape[0], args.conf, args.iou, args.max_det)
        dets = dets[:args.top_k]
        names = []
        if args.labels:
            with open(args.labels, encoding="utf-8") as handle:
                names = [line.strip() for line in handle]
        results = [
            {"score": round(score, 6), "class_id": class_id,
             "label": names[class_id] if class_id < len(names) else str(class_id),
             "box": [round(float(v), 3) for v in box]}
            for score, class_id, box in dets
        ]
    else:
        results = {"shape": list(output.shape), "min": float(output.min()),
                   "max": float(output.max()), "mean": float(output.mean())}

    payload = {
        "model": args.model, "image": args.image, "color": args.color,
        "input_size": list(size), "mode": args.mode, "results": results,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{args.model}  {args.image}  color={args.color}  input={size}")
        for item in results if isinstance(results, list) else [results]:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
