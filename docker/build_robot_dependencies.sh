#!/usr/bin/env bash
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

set -eo pipefail

source /opt/ros/jazzy/setup.bash
export PYTHONPATH="/opt/benchmark_ws/exploration_benchmark_robots:${PYTHONPATH:-}"
ROBOT_RESOURCES=/opt/benchmark_ws/exploration_benchmark_robots

for family in rosbot ugv; do
  python3 -m exploration_benchmark_robots.dependencies fetch --family "$family" \
    --directory /opt/vendor --resource-directory "$ROBOT_RESOURCES"
done

# The previous layer removed the package lists; the locked install needs them.
apt-get update
python3 -m exploration_benchmark_robots.dependencies install-binaries --resource-directory "$ROBOT_RESOURCES"

export HUSARION_ROS_BUILD_TYPE=simulation
export CMAKE_BUILD_PARALLEL_LEVEL=2
export MAKEFLAGS=-j2
for family in rosbot ugv; do
  cd "/opt/vendor/$family"
  case "$family" in
    rosbot) packages=(rosbot_gazebo) ;;
    ugv) packages=(husarion_ugv_gazebo husarion_ugv_description husarion_ugv_controller) ;;
  esac
  # Build native controllers/descriptions in separate underlays. Never source a
  # different vendor family's conflicting component/controller revision here.
  colcon build --merge-install --executor sequential --packages-up-to "${packages[@]}" \
    --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
    --parallel-workers 2
done
python3 -m exploration_benchmark_robots.dependencies verify-binaries --resource-directory "$ROBOT_RESOURCES"
python3 -m exploration_benchmark_robots.dependencies snapshot --directory /opt/vendor/provenance --resource-directory "$ROBOT_RESOURCES"
rm -rf /var/lib/apt/lists/*
