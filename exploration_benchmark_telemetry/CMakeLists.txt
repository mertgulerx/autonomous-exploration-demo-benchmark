# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)
project(exploration_benchmark_telemetry)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(nav2_bt_navigator REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME} "msg/NavigationEvent.msg"
  DEPENDENCIES geometry_msgs std_msgs)
add_library(benchmark_navigator SHARED src/navigator.cpp)
target_compile_options(benchmark_navigator PRIVATE -Wall -Wextra -Wpedantic)
ament_target_dependencies(benchmark_navigator nav2_bt_navigator nav2_core nav2_msgs
  geometry_msgs std_msgs rclcpp rclcpp_lifecycle pluginlib)
rosidl_get_typesupport_target(typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(benchmark_navigator "${typesupport_target}")
pluginlib_export_plugin_description_file(nav2_core navigator.xml)
install(TARGETS benchmark_navigator LIBRARY DESTINATION lib)
install(FILES LICENSE DESTINATION share/${PROJECT_NAME})
ament_export_dependencies(rosidl_default_runtime)
ament_package()
