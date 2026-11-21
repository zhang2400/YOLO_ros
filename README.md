# yolo_ros

ROS 2 Humble + ONNX Runtime 的 YOLO 推理封装。用 C++17 实现检测、实例分割、姿态估计、
旋转框检测和图像分类 5 个 lifecycle 组件节点，共享同一套预处理 / 解码 / 可视化代码，
不依赖 YOLOs-CPP 或 Ultralytics Python 运行时。

## 特性

- **5 个任务一套骨架**：`YoloNodeBase` 统一 lifecycle（configure 加载模型、activate 建订阅、
  deactivate 退订、cleanup 释放引擎），任务差异只在 `configure_task()` / `process()`。
- **直接封装 ONNX Runtime**：`OnnxEngine` 负责 session、I/O 名称、静态/动态输入尺寸。
- **统一解码**：`TensorView` 非拥有视图 + `[1,C,N]` / `[1,N,C]` 自适应，检测 / 分割 /
  姿态 / OBB 共用候选框与按类 NMS，输出顺序确定（相同输入必得相同结果）。
- **ROS 2 惯例**：输入用 `SensorDataQoS`（允许丢帧换低延迟），输出用 `vision_msgs`；
  OBB 与关键点使用 `msg/` 下的自定义消息；调试图与 `~/timing`
  （预处理 / 推理 / 后处理，毫秒）均为 lifecycle publisher。
- **可验证**：6 个 gtest 用例（含 RGB 通道顺序、NMS 截断确定性）、Python ONNX 参考脚本、
  300 帧延迟基准与线程数扫描脚本。

## 目录

```
src/yolo_ros/
  include/yolo_ros/   node_base / onnx_engine / letterbox / detection_decoder / ...
  src/                实现与 5 个组件注册文件
  msg/                自定义消息（KeyPoint2D、OrientedBox2D 等）
  launch/             5 个 launch 文件
  config/             默认参数
  test/unit/          gtest 用例
tools/
  yolo_ui.py          Web 控制台（启动 / configure / activate / 推图 / 看结果）
  image_stream.py     固定频率图片推流（基准用）
  bench_timing.py     订阅 ~/timing，统计 P50/P95/P99 并写 CSV
  bench_detector.sh   一键基准：起节点 → 激活 → 推流 → 采样
  reference_onnx.py   Python ONNX 参考实现（数值回归 / 后端对照）
benchmark/results.csv 300 帧 × 5 组线程配置的原始采样
models/               ONNX 模型（不入库，见下文）
data/                 类别标签
test_data/            测试图片
```

## 编译

```bash
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge ros-humble-image-transport \
  ros-humble-image-tools
cd <repo>
source /opt/ros/humble/setup.bash
colcon build --base-paths src --packages-select yolo_ros \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

`--base-paths src` 是必要的，否则 colcon 会把 `third_party/` 当成源码包扫描。

ONNX Runtime 放在 `third_party/onnxruntime/`（需含 `include/` 与 `lib/`），或用环境变量指定：

```bash
export ONNXRUNTIME_DIR=/path/to/onnxruntime
```

## 模型

仓库不提交模型二进制。用 Ultralytics 按固定版本导出到 `models/`：

```bash
pip install ultralytics==8.3.40
bash tools/export_models.sh          # opset 17，detect/seg/pose 640、obb 1024、cls 224
```

已有模型也可直接复制到 `models/`，文件名与 `tools/*.launch.py` 的默认值保持一致：

| 任务 | 文件 | 输入 |
|---|---|---|
| 检测 | `yolo11n.onnx` | 640×640（导出为动态 H/W） |
| 分割 | `yolo11n-seg.onnx` | 640×640 |
| 姿态 | `yolo11n-pose.onnx` | 640×640 |
| OBB | `yolo11n-obb.onnx` | 1024×1024 |
| 分类 | `yolo11n-cls.onnx` | 224×224 |

标签文件放在 `data/`（COCO 80 类、ImageNet 1000 类、DOTA、person）。

## 运行

```bash
# 终端 1：检测节点（lifecycle 默认 unconfigured）
ros2 run yolo_ros yolos_detector_node --ros-args \
  -p model_path:=$PWD/models/yolo11n.onnx \
  -p labels_path:=$PWD/data/coco.names

# 终端 2：激活并推图
ros2 lifecycle set /yolos_detector configure
ros2 lifecycle set /yolos_detector activate
ros2 run image_publisher image_publisher_node --ros-args \
  -p filename:=$PWD/test_data/dog_bike_car.jpg -p publish_rate:=10.0 \
  -r image_raw:=/yolos_detector/image_raw

# 终端 3：看结果
ros2 topic echo /yolos_detector/detections
ros2 topic echo /yolos_detector/timing      # 需要 publish_timing:=true
```

或使用 Web UI：`python3 tools/yolo_ui.py`，浏览器打开 `http://localhost:8080`。

### 摄像头实流

```bash
# 先确认设备号（内置摄像头常同时暴露 /dev/video0..2，且可能重新编号）
ls /dev/video*

# 用 C++ 的 cam2image 推流，避免 Python 采集线程的 GIL 竞争
ros2 run image_tools cam2image --ros-args \
  -p device_id:=0 -p width:=640 -p height:=480 -p frequency:=30.0 \
  -r image:=/yolos_detector/image_raw
```

节点已在 lifecycle 激活后订阅 `~/image_raw`，无需重启。无 `image_tools` 时可用
`tools/image_stream.py --camera 0`（Python 采集，帧率较低）。

调试预览默认限制到 5 Hz（`debug_image_rate`）。在 DDS 大图吞吐差的机器上，逐帧发布
640×480 调试图会比推理本身更耗时；关掉预览（`publish_debug_image:=false`）可把节点
吞吐从约 3 FPS 提到约 10 FPS。

关键参数（`config/default_params.yaml`）：

| 参数 | 说明 |
|---|---|
| `model_path` / `labels_path` | 必填模型路径；标签可选 |
| `conf_threshold` / `nms_threshold` | 置信度与 NMS IoU 阈值 |
| `num_threads` | ORT intra-op 线程数，0 = 自动 |
| `input_size` | 动态导出模型的方形输入；0 = 使用模型静态尺寸，动态时回退 640 |
| `publish_debug_image` / `publish_timing` | 调试图 / `~/timing` 开关 |
| `debug_image_rate` | 调试图发布频率（Hz），≤ 0 表示逐帧；默认 5 |

## 验证

单元测试：

```bash
colcon test --base-paths src --packages-select yolo_ros
colcon test-result --verbose        # 6 个 gtest 用例（汇总 7 项，含 ctest 包装），0 failures
```

数值回归（C++ 与 Python 参考必须逐框一致）：

```bash
python3 tools/reference_onnx.py --model models/yolo11n.onnx \
  --image test_data/dog_bike_car.jpg --labels data/coco.names
# 与 ros2 topic echo /yolos_detector/detections 对比
```

实测结果（Ryzen 7 8745H / 8 vCPU 虚拟机，ORT 1.23.2，CPU EP，yolo11n）：

| 对照项 | 结果 |
|---|---|
| 检测（dog_bike_car.jpg） | bicycle 0.9387 / dog 0.9269 / truck 0.5102 / car 0.4919，与 Python 参考误差 < 1e-6 |
| 分类（dog_bike_car.jpg） | malamute 0.2761 / Eskimo_dog 0.1166，与 Python 参考一致 |
| Python ORT 单次推理 | P50 / P95 / P99 = 50.2 / 55.2 / 62.0 ms |
| C++ 节点（自动线程） | 预处理 3.0 / 推理 61.5 / 后处理 2.0 ms（P50），P95 总延迟 74.3 ms，14.6 FPS |

> 说明：C++ 的 61.5 ms 是持续灌图（200 Hz 推流、best-effort 丢帧）下的实测值，包含 ROS
> 回调与 CPU 竞争；空闲单帧推理约 50 ms，与 Python 后端一致。

线程数扫描（每组 300 帧，丢弃前 20 帧 warmup，连续推流）：

| num_threads | P50 | P95 | P99 | 吞吐 |
|---|---|---|---|---|
| 1 | 163.6 ms | 170.6 ms | 179.2 ms | 6.1 FPS |
| 2 | 96.2 ms | 104.7 ms | 109.4 ms | 10.3 FPS |
| 4 | 75.7 ms | 79.8 ms | 81.3 ms | 13.2 FPS |
| 8 | 74.9 ms | 91.9 ms | 99.3 ms | 13.1 FPS |
| 0（自动） | 66.8 ms | 74.3 ms | 78.1 ms | 14.9 FPS |

结论：4 线程之后收益消失，8 线程反而让 P95 变差；默认 `num_threads: 0`（ORT 自动，
实测接近 8 线程）是这台机器上的最优选择。复现：

```bash
tools/bench_detector.sh --threads 0 --samples 300 --rate 200 \
  --csv benchmark/results.csv

# 摄像头：device 0，30 Hz 推流，实测端到端延迟
tools/bench_detector.sh --camera 0 --rate 30 --source-fps 30 --samples 120
```

摄像头实流（640×480 内置摄像头，`cam2image` 推流，`~/timing` 与输出 header 时间戳）：

| 配置 | 输出吞吐 | 端到端 P50 / P95 | 丢帧 |
|---|---|---|---|
| 调试图逐帧（旧默认） | 1.5 FPS | 241 / 369 ms | 94% |
| 调试图 5 Hz（新默认） | 3.0 FPS | 69 / 103 ms | 70% |
| 关闭调试图 | 4.8–9.7 FPS | 75–82 / 114 ms | 3–84%（取决于源帧率） |
| 静态图 200 Hz 灌图 | 14.2 FPS | 86 / 120 ms | 43% |

> 这台虚拟机的 Fast DDS 回环带宽是瓶颈：单个 640×480 图像消息的发布/接收占比很高，
> 30 Hz 原始图像会把节点压到个位数 FPS。关键结论是延迟不再随队列增长（P50 约 70 ms，
> 接近单帧计算 59 ms），即 `keep_last(1)` 的“最新帧”策略生效；要跑满 30 FPS 需要
> 压缩图像或共享内存传输，当前未实现。

## 已知限制

- 同步推理：一帧处理完才收下一帧；订阅端 `keep_last(1)` 丢弃旧帧，延迟不累积，但没有多流并行。
- 只有 CPU execution provider；CUDA 需要换 ORT GPU 构建，TensorRT 未接入。
- 没有 rosbag 回放式集成测试与 CI；当前验证手段是单测 + Python 数值回归 + 延迟基准。
- 分类 / OBB 的输入尺寸来自模型静态维度，动态导出时需显式给 `input_size`。
- 未接入压缩图像（`image_transport`），大分辨率原始图像的弱网/虚拟机吞吐受限。
- 摄像头设备号可能重新编号（`/dev/video0..2`），工具不自动探测。

## License

Apache-2.0，见 [LICENSE](LICENSE)。
