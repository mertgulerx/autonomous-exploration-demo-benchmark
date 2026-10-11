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
project(exploration_benchmark_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rcl_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RunSpec.msg" "msg/RunState.msg" "msg/RunMetrics.msg" "msg/CoverageMilestone.msg"
  "msg/SessionState.msg" "msg/Profile.msg" "msg/ProfileCatalog.msg"
  "msg/RobotProfile.msg" "msg/RobotCatalog.msg"
  "msg/Scenario.msg" "msg/ScenarioCatalog.msg" "msg/ParameterDefinition.msg" "msg/ParameterCatalog.msg"
  "srv/StartSingleRun.srv" "srv/StartComparison.srv" "srv/StopRun.srv"
  "srv/StopComparison.srv" "srv/ReturnToConfiguration.srv" "srv/WorkerCommand.srv" "srv/ResolveParameters.srv"
  DEPENDENCIES rcl_interfaces)
ament_export_dependencies(rosidl_default_runtime)
install(FILES LICENSE DESTINATION share/${PROJECT_NAME})
ament_package()
