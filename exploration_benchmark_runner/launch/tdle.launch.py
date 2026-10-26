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
"""TDLE's four nodes with the benchmark's resolved parameter file (profiles.yaml tdle), as the repository's
explore_with_bcr_ws/src/explore_with_bcr/launch/explore_only.launch.py, plus the benchmark's known-pose odometry
adapter (decision X2) that their odom_topic reads. The resolved file holds one scope "tdle" with a section per
node; each node gets its own section. Any node's exit shuts the launch down, so the worker sees the explorer
exit (it watches the launch and the first listed process only)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

NODES = (
    "fpoint_detector",
    "fpoint_filter",
    "subregion_arrangement",
    "fpoint_selection",
)


def explorer(context):
    path = LaunchConfiguration("params_file").perform(context)
    with open(path, encoding="utf-8") as stream:
        values = yaml.safe_load(stream)["tdle"]["ros__parameters"]
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}
    return [
        Node(
            package="exploration_benchmark_runner",
            executable="known_pose_odometry",
            name="known_pose_odometry",
            output="screen",
            parameters=[use_sim_time],
            on_exit=Shutdown(),
        ),
        *(
            Node(
                package="tdle",
                executable=name,
                name=name,
                output="screen",
                parameters=[values[name], use_sim_time],
                on_exit=Shutdown(),
            )
            for name in NODES
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("params_file"),
            OpaqueFunction(function=explorer),
        ]
    )
