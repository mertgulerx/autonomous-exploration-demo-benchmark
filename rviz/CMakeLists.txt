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

cmake_minimum_required(VERSION 3.8)
project(rviz_autonomous_exploration_benchmark)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(exploration_benchmark_interfaces REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_rendering REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(slam_toolbox REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_library(rviz_autonomous_exploration_benchmark_plugins SHARED
  src/exploration_control_panel.cpp
  src/comparison_view.cpp
  src/parameter_editor.cpp
)
target_include_directories(rviz_autonomous_exploration_benchmark_plugins PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(rviz_autonomous_exploration_benchmark_plugins PUBLIC cxx_std_17)
target_link_libraries(rviz_autonomous_exploration_benchmark_plugins Qt5::Widgets)
ament_target_dependencies(rviz_autonomous_exploration_benchmark_plugins
  ament_index_cpp
  geometry_msgs
  exploration_benchmark_interfaces
  nav2_msgs
  nav_msgs
  pluginlib
  rclcpp
  rcl_interfaces
  rclcpp_action
  rviz_common
  rviz_rendering
  rviz_default_plugins
  slam_toolbox
  std_msgs
  std_srvs
  visualization_msgs
)
pluginlib_export_plugin_description_file(rviz_common rviz_plugin_description.xml)

install(TARGETS rviz_autonomous_exploration_benchmark_plugins
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES
  rviz_plugin_description.xml
  LICENSE
  DESTINATION share/${PROJECT_NAME}
)

set(BENCHMARK_CONFIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../config")
if(IS_DIRECTORY "${BENCHMARK_CONFIG_DIR}")
  install(DIRECTORY
    "${BENCHMARK_CONFIG_DIR}/"
    DESTINATION share/${PROJECT_NAME}/config
  )
endif()

ament_export_include_directories(include)
ament_export_libraries(rviz_autonomous_exploration_benchmark_plugins)
ament_export_dependencies(
  ament_index_cpp
  geometry_msgs
  exploration_benchmark_interfaces
  nav2_msgs
  nav_msgs
  pluginlib
  rclcpp
  rcl_interfaces
  rclcpp_action
  rviz_common
  rviz_rendering
  rviz_default_plugins
  slam_toolbox
  std_msgs
  std_srvs
  visualization_msgs
)

ament_package()
