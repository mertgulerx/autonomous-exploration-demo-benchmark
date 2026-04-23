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
find_package(nav2_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rviz_common REQUIRED)
find_package(slam_toolbox REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_library(rviz_autonomous_exploration_benchmark_plugins SHARED
  src/exploration_control_panel.cpp
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
  nav2_msgs
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_action
  rviz_common
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
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  src/frontier_path_tracker.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  src/frontier_path_tracker.yaml
  DESTINATION share/${PROJECT_NAME}/config
)

set(BENCHMARK_CONFIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../config")
if(IS_DIRECTORY "${BENCHMARK_CONFIG_DIR}")
  install(DIRECTORY
    "${BENCHMARK_CONFIG_DIR}/"
    DESTINATION share/${PROJECT_NAME}/config
  )
endif()

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(rviz_autonomous_exploration_benchmark_plugins)
ament_export_dependencies(
  ament_index_cpp
  geometry_msgs
  nav2_msgs
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_action
  rviz_common
  slam_toolbox
  std_msgs
  std_srvs
  visualization_msgs
)

ament_package()
