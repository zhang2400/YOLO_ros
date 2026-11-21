# Copyright 2026
# SPDX-License-Identifier: Apache-2.0

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch the YOLO oriented bounding box node."""

    model_path = DeclareLaunchArgument(
        'model_path', description='Absolute path to the YOLO OBB ONNX model')
    labels_path = DeclareLaunchArgument(
        'labels_path', default_value='', description='Path to class names file')
    image_topic = DeclareLaunchArgument(
        'image_topic', default_value='/camera/image_raw',
        description='Input sensor_msgs/Image topic')
    use_gpu = DeclareLaunchArgument(
        'use_gpu', default_value='false', description='Enable CUDA execution provider')
    conf_threshold = DeclareLaunchArgument(
        'conf_threshold', default_value='0.4', description='Confidence threshold')
    nms_threshold = DeclareLaunchArgument(
        'nms_threshold', default_value='0.45', description='NMS IoU threshold')

    container = ComposableNodeContainer(
        name='yolos_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package='yolo_ros',
                plugin='yolo_ros::ObbNode',
                name='yolos_obb',
                parameters=[{
                    'model_path': LaunchConfiguration('model_path'),
                    'labels_path': LaunchConfiguration('labels_path'),
                    'use_gpu': LaunchConfiguration('use_gpu'),
                    'conf_threshold': LaunchConfiguration('conf_threshold'),
                    'nms_threshold': LaunchConfiguration('nms_threshold'),
                    'publish_debug_image': True,
                    'publish_timing': True,
                }],
                remappings=[
                    ('~/image_raw', LaunchConfiguration('image_topic')),
                ],
            ),
        ],
    )

    return LaunchDescription([
        model_path, labels_path, image_topic, use_gpu, conf_threshold,
        nms_threshold, container,
    ])
