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
project(exploration_benchmark_insight)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(frontier_exploration_ros2 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# The analysis glue lives only in the explorer's debug observer executable, so it is compiled here from the
# explorer's unchanged source; every scoring function comes from the explorer's installed core library.
set(FRONTIER_EXPLORATION_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../exploration_packages/frontier_exploration_ros2"
  CACHE PATH "Source tree of frontier_exploration_ros2 (for src/debug/debug_analyzer.cpp)")
set(ANALYZER_SOURCE "${FRONTIER_EXPLORATION_SOURCE_DIR}/src/debug/debug_analyzer.cpp")
if(NOT EXISTS "${ANALYZER_SOURCE}")
  message(FATAL_ERROR "Explorer analysis source not found: ${ANALYZER_SOURCE}. "
    "Set FRONTIER_EXPLORATION_SOURCE_DIR to the frontier_exploration_ros2 source tree.")
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FrontierInsight.msg"
  "msg/FrontierInsightCandidate.msg"
  DEPENDENCIES builtin_interfaces std_msgs)

add_executable(frontier_insight src/frontier_insight_node.cpp "${ANALYZER_SOURCE}")
target_compile_options(frontier_insight PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(frontier_insight frontier_exploration_ros2::frontier_exploration_ros2_core)
ament_target_dependencies(frontier_insight builtin_interfaces geometry_msgs nav_msgs rclcpp std_msgs
  tf2 tf2_ros)
rosidl_get_typesupport_target(typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(frontier_insight "${typesupport_target}")
install(TARGETS frontier_insight RUNTIME DESTINATION lib/${PROJECT_NAME})
install(FILES LICENSE DESTINATION share/${PROJECT_NAME})
ament_export_dependencies(rosidl_default_runtime)
ament_package()
