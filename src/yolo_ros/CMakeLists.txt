cmake_minimum_required(VERSION 3.16)
project(yolo_ros VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# BUILD_TESTING is provided by ament_cmake (default ON), so `colcon test` works
# out of the box without extra -D flags.

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ---------------------------------------------------------------------------
# ONNX Runtime (local build under third_party, or ONNXRUNTIME_DIR env var)
# ---------------------------------------------------------------------------
if(NOT ONNXRUNTIME_ROOT)
  if(DEFINED ENV{ONNXRUNTIME_DIR})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_DIR}")
  else()
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnxruntime")
  endif()
endif()

find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib" /usr/local/lib /opt/onnxruntime/lib
  NO_DEFAULT_PATH)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include" /usr/local/include/onnxruntime
        /opt/onnxruntime/include
  NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE_DIR)
  message(FATAL_ERROR
    "ONNX Runtime not found. Set ONNXRUNTIME_DIR or place it in "
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnxruntime "
    "(expected lib/libonnxruntime.so and include/onnxruntime_cxx_api.h)")
endif()

message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")
message(STATUS "ONNX Runtime headers: ${ONNXRUNTIME_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Custom messages (keypoints + oriented bounding boxes)
# ---------------------------------------------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeyPoint2D.msg"
  "msg/KeyPoint2DArray.msg"
  "msg/OrientedBox2D.msg"
  "msg/ObbDetection2D.msg"
  "msg/ObbDetection2DArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# ---------------------------------------------------------------------------
# Component library
# ---------------------------------------------------------------------------
add_library(${PROJECT_NAME}_components SHARED
  src/onnx_engine.cpp
  src/letterbox.cpp
  src/detection_decoder.cpp
  src/visualization.cpp
  src/messages.cpp
  src/node_base.cpp
  src/detector_node.cpp
  src/segmentor_node.cpp
  src/pose_node.cpp
  src/obb_node.cpp
  src/classifier_node.cpp
  src/register_detector.cpp
  src/register_segmentor.cpp
  src/register_pose.cpp
  src/register_obb.cpp
  src/register_classifier.cpp
)

target_include_directories(${PROJECT_NAME}_components PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp>
  ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}_components
  ${ONNXRUNTIME_LIB}
  ${OpenCV_LIBS}
)

ament_target_dependencies(${PROJECT_NAME}_components
  rclcpp
  rclcpp_lifecycle
  rclcpp_components
  sensor_msgs
  vision_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  image_transport
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_components "${cpp_typesupport_target}")

get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIB}" DIRECTORY)
set_target_properties(${PROJECT_NAME}_components PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
  INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "yolo_ros::DetectorNode"
  EXECUTABLE yolos_detector_node
)

rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "yolo_ros::SegmentorNode"
  EXECUTABLE yolos_segmentor_node
)

rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "yolo_ros::PoseNode"
  EXECUTABLE yolos_pose_node
)

rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "yolo_ros::ObbNode"
  EXECUTABLE yolos_obb_node
)

rclcpp_components_register_node(${PROJECT_NAME}_components
  PLUGIN "yolo_ros::ClassifierNode"
  EXECUTABLE yolos_classifier_node
)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME}_components
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_components)
ament_export_dependencies(
  rclcpp rclcpp_lifecycle rclcpp_components sensor_msgs vision_msgs
  std_msgs geometry_msgs cv_bridge)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_decoders test/unit/test_decoders.cpp)
  target_link_libraries(test_decoders ${PROJECT_NAME}_components)
  ament_target_dependencies(test_decoders rclcpp)
endif()

ament_package()
