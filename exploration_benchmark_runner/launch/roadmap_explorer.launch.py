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
"""roadmap-explorer's exploration server and its lifecycle manager with the benchmark's resolved parameter file
(profiles.yaml roadmap_explorer); as the package's own tb3_exploration.launch.py, without RViz. The worker
starts a session with one Explore goal once the server is active."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("params_file"),
            Node(
                package="roadmap_explorer",
                executable="roadmap_exploration_server",
                name="roadmap_explorer_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file"), {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_exploration",
                output="screen",
                parameters=[
                    {"autostart": True},
                    {"node_names": ["roadmap_explorer_node"]},
                    {"use_sim_time": use_sim_time},
                ],
            ),
        ]
    )
