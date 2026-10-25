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

"""Spawn only the selected robot; no RViz, world, SLAM, or Nav2 launch."""

import json
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def spawn(context):
    config = json.loads(
        Path(LaunchConfiguration("configuration").perform(context)).read_text()
    )
    profile = config["profile"]
    os.environ.setdefault("ROS_HOME", str(Path(config["directory"]) / "ros_home"))
    resource_paths = config.get("resource_paths", [])
    os.environ["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
        resource_paths + [os.environ.get("GZ_SIM_RESOURCE_PATH", "")]
    )
    actions = []
    if "robot_description" in config:
        actions.append(
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[
                    {
                        "use_sim_time": True,
                        "robot_description": config["visualization_description"],
                    }
                ],
                output="screen",
            )
        )
    if "spawn_file" in config:
        x, y, z, yaw = config["spawn_pose"]
        actions.append(
            Node(
                package="ros_gz_sim",
                executable="create",
                output="screen",
                arguments=[
                    "-world",
                    config["scenario"]["world_name"],
                    "-name",
                    profile["model"],
                    "-file",
                    config["spawn_file"],
                    "-x",
                    str(x),
                    "-y",
                    str(y),
                    "-z",
                    str(z),
                    "-Y",
                    str(yaw),
                ],
            )
        )
    actions.append(
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            output="screen",
            parameters=[{"use_sim_time": True, "config_file": config["bridge"]}],
        )
    )
    if "ekf" in config:
        actions.append(
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                parameters=[config["ekf"], {"use_sim_time": True}],
                output="screen",
                remappings=(
                    [("odometry/filtered", "/odom")]
                    if profile.get("measurement_odom_topic")
                    else []
                ),
            )
        )
    if "scan_filter" in config:
        actions.append(
            Node(
                package="laser_filters",
                executable="scan_to_scan_filter_chain",
                parameters=[config["scan_filter"]],
                output="screen",
                remappings=[("scan", "/robot/scan/raw"), ("scan_filtered", "/scan")],
            )
        )
    # Known-pose mapping (mapping-known-pose-v1): ground-truth odom->base_footprint
    # and the mapping-only scan for SLAM Toolbox (after the body filter's /scan).
    for executable, key in (
        ("known_pose_tf", "known_pose"),
        ("mapping_scan_filter", "mapping_scan"),
    ):
        actions.append(
            Node(
                package="exploration_benchmark_robots",
                executable=executable,
                parameters=[config[key]["parameters"]],
                output="screen",
            )
        )
    if profile["adapter"] not in ("mogi", "turtlebot3"):
        from exploration_benchmark_robots.wheeled import wheeled_actions

        actions.extend(wheeled_actions(config))
    return actions


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument("configuration"), OpaqueFunction(function=spawn)]
    )
