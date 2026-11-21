#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Integrated web UI for the yolo_ros task nodes.

It launches one task node, drives its lifecycle, publishes a test image and
streams the debug image / detection results to a browser dashboard.

Usage:
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 tools/yolo_ui.py --port 8080
Then open http://localhost:8080
"""

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from vision_msgs.msg import Classification, Detection2DArray

try:
    from yolo_ros.msg import ObbDetection2DArray
except ImportError:  # message package not sourced
    ObbDetection2DArray = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = {
    'detector': {
        'executable': 'yolos_detector_node',
        'node': '/yolos_detector',
        'model': 'yolo11n.onnx',
        'labels': 'coco.names',
        'image': 'dog_bike_car.jpg',
    },
    'segmentor': {
        'executable': 'yolos_segmentor_node',
        'node': '/yolos_segmentor',
        'model': 'yolo11n-seg.onnx',
        'labels': 'coco.names',
        'image': 'seg_bus.jpg',
    },
    'pose': {
        'executable': 'yolos_pose_node',
        'node': '/yolos_pose',
        'model': 'yolo11n-pose.onnx',
        'labels': 'person.names',
        'image': 'pose_person.jpg',
    },
    'obb': {
        'executable': 'yolos_obb_node',
        'node': '/yolos_obb',
        'model': 'yolo11n-obb.onnx',
        'labels': 'dota.names',
        'image': 'obb_street.png',
    },
    'classifier': {
        'executable': 'yolos_classifier_node',
        'node': '/yolos_classifier',
        'model': 'yolo11n-cls.onnx',
        'labels': 'imagenet.names',
        'image': 'dog_bike_car.jpg',
    },
}

TASK_TITLES = {
    'detector': 'Object Detection',
    'segmentor': 'Instance Segmentation',
    'pose': 'Pose Estimation',
    'obb': 'Oriented Bounding Box',
    'classifier': 'Image Classification',
}


class UiNode(Node):
    def __init__(self):
        super().__init__('yolo_ui')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.condition = threading.Condition()

        self.latest_jpeg = None
        self.frame_seq = 0
        self.frames_total = 0
        self.pub_count = 0
        self.frame_times = deque(maxlen=30)
        self.detections = []
        self.timing = []
        self.last_error = ''

        self.task = None
        self.process = None
        self.state = 'stopped'
        self.log = deque(maxlen=120)

        self.image_pub = None
        self.image_msg = None
        self.timer = None
        self.subs = []

        self.source = 'image'        # 'image' or 'camera'
        self.camera = None           # cv2.VideoCapture instance
        self.camera_path = ''
        self.publish_rate = 5.0

    # ------------------------------------------------------------------
    # process / lifecycle control
    # ------------------------------------------------------------------
    def _run_cli(self, args, timeout=15):
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=timeout, check=False)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            return None

    def _log_reader(self, process):
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            with self.lock:
                self.log.append(line.rstrip())

    def _wait_for_node(self, node_name, timeout=25.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self._run_cli(['ros2', 'lifecycle', 'get', node_name], timeout=5)
            if result is not None and result.returncode == 0:
                return True
            time.sleep(0.5)
        return False

    def _transition(self, transition, timeout=20):
        assert self.task is not None
        node_name = TASKS[self.task]['node']
        result = self._run_cli(['ros2', 'lifecycle', 'set', node_name, transition],
                               timeout=timeout)
        if result is None or 'successful' not in (result.stdout + result.stderr):
            detail = ''
            if result is not None:
                detail = (result.stdout + result.stderr).strip().splitlines()
                detail = detail[-1] if detail else ''
            self.last_error = f'lifecycle {transition} failed: {detail}'
            return False
        with self.lock:
            self.state = {
                'configure': 'inactive',
                'activate': 'active',
                'deactivate': 'inactive',
                'shutdown': 'finalized',
            }.get(transition, self.state)
        return True

    # ------------------------------------------------------------------
    # camera helpers
    # ------------------------------------------------------------------
    @staticmethod
    def list_cameras(timeout_per_device=2.0):
        """Enumerate usable V4L2 cameras on this machine.

        Returns a list of dicts: {device, name, width, height}. Only devices
        that can actually be opened and return a frame are reported, which
        filters out metadata-only /dev/video nodes. Each device is probed in a
        daemon thread bounded by ``timeout_per_device`` seconds so a hung or
        phantom node (opens but never delivers a frame) cannot block the UI.
        """
        import glob

        def probe(target, label):
            try:
                cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    return None
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None or frame.size == 0:
                    return None
                h, w = frame.shape[:2]
                return {
                    'device': str(label),
                    'name': f'{label}  ({w}x{h})',
                    'width': int(w),
                    'height': int(h),
                }
            except Exception:  # noqa: BLE001
                return None

        def probe_with_timeout(target, label):
            box = {}

            def worker():
                box['result'] = probe(target, label)

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(timeout_per_device)
            if t.is_alive():
                # Device hung; the daemon thread will finish its blocked read()
                # and release the capture on its own. Skip this device.
                return None
            return box.get('result')

        # Silence OpenCV's noisy V4L2 warnings (e.g. "can't open camera by
        # index", "select() timeout") while enumerating, then restore level.
        logging = getattr(getattr(cv2, 'utils', None), 'logging', None)
        prev_level = None
        if logging is not None:
            try:
                prev_level = logging.getLogLevel()
                logging.setLogLevel(logging.LOG_LEVEL_SILENT)
            except Exception:  # noqa: BLE001
                prev_level = None

        try:
            devices = []
            paths = sorted(glob.glob('/dev/video*'))
            for path in paths:
                dev = probe_with_timeout(path, path)
                if dev is not None:
                    devices.append(dev)
            # Only fall back to integer indices when the OS exposes no
            # /dev/video nodes at all (e.g. non-Linux); otherwise indices
            # would just re-probe the same broken nodes.
            if not devices and not paths:
                for idx in range(5):
                    dev = probe_with_timeout(idx, idx)
                    if dev is not None:
                        devices.append(dev)
            return devices
        finally:
            if logging is not None and prev_level is not None:
                try:
                    logging.setLogLevel(prev_level)
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # ROS interfaces
    # ------------------------------------------------------------------
    def _teardown_ros(self):
        if self.timer is not None:
            self.timer.cancel()
            self.destroy_timer(self.timer)
            self.timer = None
        if self.image_pub is not None:
            self.destroy_publisher(self.image_pub)
            self.image_pub = None
        for sub in self.subs:
            self.destroy_subscription(sub)
        self.subs = []
        self.image_msg = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.camera_path = ''
        self.source = 'image'

    def _setup_ros(self, task, source, image_path, camera_path):
        spec = TASKS[task]
        node_name = spec['node']

        self.source = source
        if source == 'camera':
            self.camera = cv2.VideoCapture(camera_path, cv2.CAP_V4L2)
            if not self.camera.isOpened():
                # fall back to default backend / integer index
                self.camera.release()
                try:
                    idx = int(camera_path)
                except (TypeError, ValueError):
                    idx = 0
                self.camera = cv2.VideoCapture(idx)
            if not self.camera.isOpened():
                self.camera.release()
                self.camera = None
                raise RuntimeError(f'cannot open camera: {camera_path}')
            self.camera_path = camera_path
            self.image_msg = None
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f'cannot read image: {image_path}')
            self.image_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')

        self.image_pub = self.create_publisher(
            Image, f'{node_name}/image_raw', qos_profile_sensor_data)
        period = 1.0 / max(float(self.publish_rate), 1.0)
        self.timer = self.create_timer(period, self._publish_image)

        self.subs.append(self.create_subscription(
            Image, f'{node_name}/debug_image', self._on_debug, 10))
        self.subs.append(self.create_subscription(
            Float64MultiArray, f'{node_name}/timing', self._on_timing, 10))

        if task in ('detector', 'segmentor', 'pose'):
            self.subs.append(self.create_subscription(
                Detection2DArray, f'{node_name}/detections', self._on_detections, 10))
        elif task == 'obb' and ObbDetection2DArray is not None:
            self.subs.append(self.create_subscription(
                ObbDetection2DArray, f'{node_name}/detections', self._on_obb, 10))
        elif task == 'classifier':
            self.subs.append(self.create_subscription(
                Classification, f'{node_name}/classification', self._on_classification, 10))

    def _publish_image(self):
        if self.image_pub is None:
            return
        if self.source == 'camera':
            if self.camera is None:
                return
            ok, frame = self.camera.read()
            if not ok or frame is None or frame.size == 0:
                return
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        else:
            if self.image_msg is None:
                return
            msg = self.image_msg
        self.pub_count += 1
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.image_pub.publish(msg)

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _on_debug(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return
            with self.condition:
                self.latest_jpeg = encoded.tobytes()
                self.frame_seq += 1
                self.frames_total += 1
                self.frame_times.append(time.time())
                self.condition.notify_all()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f'debug image: {exc}'

    def _on_timing(self, msg):
        with self.lock:
            self.timing = [round(float(v), 2) for v in msg.data]

    def _on_detections(self, msg):
        items = []
        for det in msg.detections:
            hyp = det.results[0].hypothesis if det.results else None
            items.append({
                'class': hyp.class_id if hyp else '?',
                'score': round(float(hyp.score), 4) if hyp else 0.0,
                'x': round(float(det.bbox.center.position.x), 1),
                'y': round(float(det.bbox.center.position.y), 1),
                'w': round(float(det.bbox.size_x), 1),
                'h': round(float(det.bbox.size_y), 1),
            })
        with self.lock:
            self.detections = items

    def _on_obb(self, msg):
        items = []
        for det in msg.detections:
            items.append({
                'class': det.class_name,
                'score': round(float(det.score), 4),
                'x': round(float(det.bbox.center.x), 1),
                'y': round(float(det.bbox.center.y), 1),
                'w': round(float(det.bbox.size_x), 1),
                'h': round(float(det.bbox.size_y), 1),
                'theta': round(float(det.bbox.theta), 3),
            })
        with self.lock:
            self.detections = items

    def _on_classification(self, msg):
        items = [{
            'class': result.class_id,
            'score': round(float(result.score), 4),
            'x': 0.0,
            'y': 0.0,
            'w': 0.0,
            'h': 0.0,
        } for result in msg.results]
        with self.lock:
            self.detections = items

    # ------------------------------------------------------------------
    # public API used by HTTP handlers
    # ------------------------------------------------------------------
    def start(self, task, model='', labels='', image='', conf=None,
              source='image', camera='', rate=5):
        if task not in TASKS:
            return False, f'unknown task: {task}'
        if source not in ('image', 'camera'):
            return False, f'unknown source: {source}'
        with self.lock:
            if self.process is not None:
                return False, 'a task is already running, stop it first'

        spec = TASKS[task]
        model_path = model or os.path.join(PROJECT_ROOT, 'models', spec['model'])
        labels_path = labels or os.path.join(PROJECT_ROOT, 'data', spec['labels'])
        image_path = image or os.path.join(PROJECT_ROOT, 'test_data', spec['image'])

        if not os.path.isfile(model_path):
            return False, f'model not found: {model_path}'
        if source == 'image' and not os.path.isfile(image_path):
            return False, f'image not found: {image_path}'
        if source == 'camera' and not camera:
            return False, 'no camera selected'
        try:
            self.publish_rate = max(1.0, min(60.0, float(rate)))
        except (TypeError, ValueError):
            return False, f'invalid rate: {rate}'

        if source == 'camera':
            cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                return False, f'cannot open camera: {camera}'
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return False, f'camera returned no frame: {camera}'

        args = [
            'ros2', 'run', 'yolo_ros', spec['executable'], '--ros-args',
            '-p', f'model_path:={model_path}',
            '-p', f'publish_debug_image:=true',
            '-p', f'publish_timing:=true',
        ]
        if os.path.isfile(labels_path):
            args += ['-p', f'labels_path:={labels_path}']
        if conf is not None:
            args += ['-p', f'conf_threshold:={float(conf)}']

        with self.lock:
            self.task = task
            self.state = 'starting'
            self.detections = []
            self.timing = []
            self.last_error = ''
            self.log.clear()
            self.latest_jpeg = None
            self.frames_total = 0
            self.pub_count = 0
            self.frame_times.clear()

        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)
        threading.Thread(
            target=self._log_reader, args=(self.process,), daemon=True).start()

        if not self._wait_for_node(spec['node']):
            self.stop()
            return False, 'task node did not start, see log'

        if not self._transition('configure') or not self._transition('activate'):
            self.stop()
            return False, self.last_error or 'lifecycle transition failed'

        try:
            self._setup_ros(task, source, image_path, camera)
        except Exception as exc:  # noqa: BLE001
            self.stop()
            return False, str(exc)

        with self.lock:
            self.state = 'active'
        return True, f'{task} started'

    def stop(self):
        if self.task is not None and self.process is not None:
            self._transition('deactivate', timeout=10)
            self._transition('shutdown', timeout=10)
        self._teardown_ros()
        process = self.process
        self.process = None
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        with self.lock:
            self.state = 'stopped'
            self.task = None
            self.detections = []
            self.timing = []
        return True, 'stopped'

    def state_dict(self):
        with self.lock:
            now = time.time()
            times = [t for t in self.frame_times if now - t <= 3.0]
            fps = 0.0
            if len(times) >= 2:
                fps = (len(times) - 1) / max(times[-1] - times[0], 1e-3)
            return {
                'task': self.task,
                'title': TASK_TITLES.get(self.task, '-'),
                'state': self.state,
                'source': self.source,
                'camera': self.camera_path,
                'rate': round(self.publish_rate, 1),
                'frames': self.frames_total,
                'published': self.pub_count,
                'fps': round(fps, 1),
                'timing': self.timing,
                'detections': list(self.detections),
                'error': self.last_error,
                'log': list(self.log)[-8:],
            }

    def wait_frame(self, last_seq, timeout=2.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_seq != last_seq, timeout=timeout)
            return self.latest_jpeg, self.frame_seq


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>yolo_ros 集成测试台</title>
<style>
  :root { --bg:#0f1220; --panel:#181c30; --line:#2a3050; --text:#e8eaf6;
          --muted:#98a0c0; --accent:#4dabf7; --ok:#37b24d; --warn:#f59f00; --err:#e03131; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:14px 22px;
           border-bottom:1px solid var(--line); background:#12162a; }
  header h1 { font-size:17px; margin:0; font-weight:600; letter-spacing:.5px; }
  .pill { font-size:12px; padding:4px 10px; border-radius:999px;
          background:#232a45; color:var(--muted); }
  .pill.active { background:rgba(55,178,77,.15); color:#69db7c; }
  .pill.starting { background:rgba(245,159,0,.15); color:#ffd43b; }
  .pill.stopped { background:#232a45; color:var(--muted); }
  main { display:grid; grid-template-columns:290px 1fr 330px; gap:14px; padding:14px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .card h2 { font-size:13px; margin:0 0 12px; color:var(--muted); font-weight:600;
             letter-spacing:.6px; text-transform:uppercase; }
  label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
  select, input { width:100%; padding:7px 9px; border-radius:7px; border:1px solid var(--line);
                  background:#11152a; color:var(--text); font-size:13px; }
  .row { display:flex; gap:8px; }
  button { cursor:pointer; border:none; border-radius:7px; padding:9px 12px;
           font-size:13px; background:#2a3150; color:var(--text); width:100%; }
  button:hover { background:#343c60; }
  button.primary { background:var(--accent); color:#08101f; font-weight:600; }
  button.primary:hover { background:#74c0fc; }
  button.danger { background:#5c1f28; color:#ffc9c9; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .viewer { position:relative; background:#0a0d18; border-radius:8px; overflow:hidden;
            min-height:420px; display:flex; align-items:center; justify-content:center; }
  .viewer img { max-width:100%; max-height:70vh; display:block; }
  .placeholder { color:#4a5273; font-size:14px; }
  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:12px; }
  .stat { background:#11152a; border:1px solid var(--line); border-radius:8px; padding:10px; }
  .stat .k { font-size:11px; color:var(--muted); }
  .stat .v { font-size:18px; margin-top:4px; font-variant-numeric:tabular-nums; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th { text-align:left; color:var(--muted); font-weight:500; padding:6px 4px;
       border-bottom:1px solid var(--line); }
  td { padding:6px 4px; border-bottom:1px solid #20263f; font-variant-numeric:tabular-nums; }
  .score { color:#69db7c; }
  .empty { color:#4a5273; font-size:12px; padding:10px 0; }
  .err { color:#ff8787; font-size:12px; margin-top:10px; word-break:break-all; }
  pre.log { background:#0a0d18; border:1px solid var(--line); border-radius:7px;
            padding:8px; font-size:11px; color:#8b93b5; max-height:130px; overflow:auto;
            white-space:pre-wrap; margin:8px 0 0; }
  .btns { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }
  .btns .wide { grid-column:1 / span 2; }
</style>
</head>
<body>
<header>
  <h1>yolo_ros 集成测试台</h1>
  <span id="pill" class="pill stopped">stopped</span>
  <span id="title" class="pill">-</span>
</header>

<main>
  <section class="card">
    <h2>任务配置</h2>
    <label>任务类型</label>
    <select id="task">
      <option value="detector">目标检测</option>
      <option value="segmentor">实例分割</option>
      <option value="pose">姿态估计</option>
      <option value="obb">旋转框 OBB</option>
      <option value="classifier">图像分类</option>
    </select>

    <label>模型</label>
    <select id="model"></select>

    <label>标签文件</label>
    <select id="labels"></select>

    <label>图像来源</label>
    <select id="source">
      <option value="image">测试图片</option>
      <option value="camera">摄像头设备</option>
    </select>

    <label id="labelImage">测试图片</label>
    <select id="image"></select>

    <div id="cameraBox" style="display:none">
      <label>摄像头设备 <button id="refreshCam" type="button" style="width:auto;padding:2px 8px;font-size:11px;margin-left:6px">刷新</button></label>
      <select id="camera"></select>
    </div>

    <label>发布帧率 (Hz)</label>
    <input id="rate" type="number" min="1" max="60" step="1" value="5"/>

    <label>置信度阈值</label>
    <input id="conf" type="number" min="0.01" max="0.99" step="0.01" value="0.4"/>

    <div class="btns">
      <button id="startBtn" class="primary wide">启动并激活</button>
      <button id="stopBtn" class="danger wide" disabled>停止</button>
      <button id="cfgBtn" disabled>configure</button>
      <button id="actBtn" disabled>activate</button>
    </div>
  </section>

  <section class="card">
    <h2>实时画面</h2>
    <div class="viewer">
      <img id="stream" alt="" style="display:none"/>
      <div id="placeholder" class="placeholder">选择任务后点击“启动并激活”</div>
    </div>
    <div class="stats">
      <div class="stat"><div class="k">FPS</div><div class="v" id="fps">0</div></div>
      <div class="stat"><div class="k">检测数量</div><div class="v" id="count">0</div></div>
      <div class="stat"><div class="k">推理耗时 (ms)</div><div class="v" id="infer">-</div></div>
      <div class="stat"><div class="k">总帧数</div><div class="v" id="frames">0</div></div>
    </div>
    <div id="error" class="err"></div>
  </section>

  <section class="card">
    <h2>检测结果</h2>
    <table>
      <thead><tr><th>类别</th><th>置信度</th><th>中心 x/y</th><th>宽高</th></tr></thead>
      <tbody id="tbody"><tr><td colspan="4" class="empty">暂无数据</td></tr></tbody>
    </table>
    <h2 style="margin-top:16px">节点日志</h2>
    <pre class="log" id="log">-</pre>
  </section>
</main>

<script>
const presets = {
  detector:  {model:'yolo11n.onnx',      labels:'coco.names',     image:'dog_bike_car.jpg'},
  segmentor: {model:'yolo11n-seg.onnx',  labels:'coco.names',     image:'seg_bus.jpg'},
  pose:      {model:'yolo11n-pose.onnx', labels:'person.names',   image:'pose_person.jpg'},
  obb:       {model:'yolo11n-obb.onnx',  labels:'dota.names',     image:'obb_street.png'},
  classifier:{model:'yolo11n-cls.onnx',  labels:'imagenet.names', image:'dog_bike_car.jpg'},
};
let files = {models:[], labels:[], images:[]};
let cameras = [];
let running = false;

function fill(id, items, preferred) {
  const el = document.getElementById(id);
  el.innerHTML = '';
  items.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    el.appendChild(opt);
  });
  if (preferred && items.includes(preferred)) el.value = preferred;
}

function fillCameras(preferred) {
  const el = document.getElementById('camera');
  el.innerHTML = '';
  if (!cameras.length) {
    const opt = document.createElement('option');
    opt.value = ''; opt.textContent = '未检测到摄像头';
    el.appendChild(opt);
    return;
  }
  cameras.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.device; opt.textContent = c.name;
    el.appendChild(opt);
  });
  if (preferred && cameras.some(c => c.device === preferred)) el.value = preferred;
}

function applyPreset() {
  const p = presets[document.getElementById('task').value];
  fill('model', files.models, p.model);
  fill('labels', files.labels, p.labels);
  fill('image', files.images, p.image);
}

function applySource() {
  const src = document.getElementById('source').value;
  const isCam = src === 'camera';
  document.getElementById('cameraBox').style.display = isCam ? 'block' : 'none';
  document.getElementById('image').style.display = isCam ? 'none' : 'block';
  document.getElementById('labelImage').style.display = isCam ? 'none' : 'block';
}

async function loadFiles() {
  const res = await fetch('/api/files');
  files = await res.json();
  applyPreset();
}

async function loadCameras() {
  try {
    const res = await fetch('/api/cameras');
    const data = await res.json();
    cameras = data.cameras || [];
    fillCameras();
  } catch (e) {
    cameras = []; fillCameras();
  }
}

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {})
  });
  return res.json();
}

document.getElementById('task').onchange = applyPreset;
document.getElementById('source').onchange = applySource;
document.getElementById('refreshCam').onclick = loadCameras;

document.getElementById('startBtn').onclick = async () => {
  const src = document.getElementById('source').value;
  const payload = {
    task: document.getElementById('task').value,
    model: 'models/' + document.getElementById('model').value,
    labels: 'data/' + document.getElementById('labels').value,
    image: 'test_data/' + document.getElementById('image').value,
    conf: parseFloat(document.getElementById('conf').value),
    source: src,
    camera: document.getElementById('camera').value,
    rate: parseFloat(document.getElementById('rate').value)
  };
  const btn = document.getElementById('startBtn');
  btn.disabled = true;
  btn.textContent = '启动中...';
  const out = await post('/api/start', payload);
  btn.textContent = '启动并激活';
  if (!out.ok) {
    btn.disabled = false;
    alert(out.message || '启动失败');
  }
};

document.getElementById('stopBtn').onclick = async () => { await post('/api/stop'); };
document.getElementById('cfgBtn').onclick = async () => { await post('/api/lifecycle', {action:'configure'}); };
document.getElementById('actBtn').onclick = async () => { await post('/api/lifecycle', {action:'activate'}); };

function render(s) {
  running = s.state === 'active' || s.state === 'starting';
  const pill = document.getElementById('pill');
  pill.textContent = s.state;
  pill.className = 'pill ' + s.state;
  let titleText = s.title || '-';
  if (running) {
    titleText += '  |  ' + (s.source === 'camera'
      ? ('摄像头 ' + (s.camera || '-'))
      : '测试图片');
  }
  document.getElementById('title').textContent = titleText;
  document.getElementById('fps').textContent = s.fps;
  document.getElementById('count').textContent = s.detections.length;
  document.getElementById('infer').textContent =
    s.timing && s.timing.length >= 2 ? s.timing[1] : '-';
  document.getElementById('frames').textContent = s.frames;
  document.getElementById('error').textContent = s.error || '';
  document.getElementById('log').textContent = (s.log || []).join('\\n') || '-';

  document.getElementById('startBtn').disabled = running;
  document.getElementById('stopBtn').disabled = !running;
  document.getElementById('cfgBtn').disabled = !running;
  document.getElementById('actBtn').disabled = s.state !== 'inactive';
  document.getElementById('task').disabled = running;
  document.getElementById('model').disabled = running;
  document.getElementById('labels').disabled = running;
  document.getElementById('source').disabled = running;
  document.getElementById('image').disabled = running;
  document.getElementById('camera').disabled = running;
  document.getElementById('rate').disabled = running;
  document.getElementById('refreshCam').disabled = running;

  const stream = document.getElementById('stream');
  const ph = document.getElementById('placeholder');
  if (s.frames > 0) {
    if (stream.style.display === 'none') {
      stream.src = '/stream?t=' + Date.now();
      stream.style.display = 'block';
      ph.style.display = 'none';
    }
  } else if (s.state === 'stopped') {
    stream.style.display = 'none';
    ph.style.display = 'block';
  }

  const tbody = document.getElementById('tbody');
  if (!s.detections.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty">暂无数据</td></tr>';
  } else {
    tbody.innerHTML = s.detections.map(d =>
      '<tr>' +
      '<td>' + d.class + '</td>' +
      '<td class="score">' + d.score.toFixed(3) + '</td>' +
      '<td>' + d.x + ' / ' + d.y + '</td>' +
      '<td>' + d.w + ' x ' + d.h + '</td>' +
      '</tr>').join('');
  }
}

async function poll() {
  try {
    const res = await fetch('/api/state');
    render(await res.json());
  } catch (e) { /* server restarting */ }
}

loadFiles();
loadCameras();
applySource();
poll();
setInterval(poll, 500);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    app = None  # type: UiNode

    def log_message(self, *args):  # silence default access logs
        return

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get('Content-Length', '0'))
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode('utf-8'))
        except json.JSONDecodeError:
            return {}

    def do_GET(self):  # noqa: N802
        if self.path in ('/', '/index.html'):
            body = INDEX_HTML.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == '/api/state':
            self._send_json(self.app.state_dict())
            return

        if self.path == '/api/files':
            def listing(subdir, suffix):
                directory = os.path.join(PROJECT_ROOT, subdir)
                if not os.path.isdir(directory):
                    return []
                return sorted(
                    name for name in os.listdir(directory) if name.endswith(suffix))

            self._send_json({
                'models': listing('models', '.onnx'),
                'labels': listing('data', '.names'),
                'images': listing('test_data', '.jpg') + listing('test_data', '.png'),
            })
            return

        if self.path == '/api/cameras':
            self._send_json({'cameras': UiNode.list_cameras()})
            return

        if self.path.startswith('/stream'):
            self._stream()
            return

        self._send_json({'error': 'not found'}, status=404)

    def _stream(self):
        self.send_response(200)
        self.send_header(
            'Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        last_seq = -1
        try:
            while True:
                jpeg, seq = self.app.wait_frame(last_seq, timeout=1.5)
                if jpeg is None or seq == last_seq:
                    continue
                last_seq = seq
                self.wfile.write(
                    b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                    + str(len(jpeg)).encode() + b'\r\n\r\n')
                self.wfile.write(jpeg)
                self.wfile.write(b'\r\n')
        except (BrokenPipeError, ConnectionResetError, ValueError):
            return

    def do_POST(self):  # noqa: N802
        body = self._read_json()

        if self.path == '/api/start':
            task = body.get('task', '')
            model = self._resolve(body.get('model', ''), 'models')
            labels = self._resolve(body.get('labels', ''), 'data')
            image = self._resolve(body.get('image', ''), 'test_data')
            source = body.get('source', 'image')
            camera = body.get('camera', '')
            rate = body.get('rate', 5)
            ok, message = self.app.start(
                task, model=model, labels=labels, image=image,
                conf=body.get('conf'), source=source, camera=camera, rate=rate)
            self._send_json({'ok': ok, 'message': message})
            return

        if self.path == '/api/stop':
            ok, message = self.app.stop()
            self._send_json({'ok': ok, 'message': message})
            return

        if self.path == '/api/lifecycle':
            action = body.get('action', '')
            if action not in ('configure', 'activate', 'deactivate', 'shutdown'):
                self._send_json({'ok': False, 'message': 'unknown action'}, status=400)
                return
            if self.app.task is None:
                self._send_json({'ok': False, 'message': 'no task running'})
                return
            ok = self.app._transition(action)
            self._send_json({'ok': ok, 'message': 'ok' if ok else self.app.last_error})
            return

        self._send_json({'error': 'not found'}, status=404)

    @staticmethod
    def _resolve(path, subdir):
        if not path:
            return ''
        if os.path.isabs(path):
            return path
        return os.path.join(PROJECT_ROOT, subdir, os.path.basename(path))


def main():
    parser = argparse.ArgumentParser(description='yolo_ros integrated web UI')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    rclpy.init()
    node = UiNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    Handler.app = node
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    def shutdown(*_):
        node.stop()
        server.shutdown()
        executor.shutdown()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f'[yolo_ui] dashboard ready: http://localhost:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        node.stop()


if __name__ == '__main__':
    main()
