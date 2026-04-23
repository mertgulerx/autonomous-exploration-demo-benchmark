#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    roadmap_share = get_package_share_directory("roadmap_explorer")
    benchmark_rviz_share = get_package_share_directory(
        "rviz_autonomous_exploration_benchmark"
    )
    default_params_file = os.path.join(
        roadmap_share, "params", "tb3_exploration_params.yaml"
    )
    default_rviz_file = os.path.join(roadmap_share, "rviz", "exploration.rviz")
    default_override_params_file = os.path.join(
        benchmark_rviz_share, "config", "roadmap_explorer", "roadmap_explorer_overrides.yaml"
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")
    override_params_file = LaunchConfiguration("override_params_file")
    rviz_file = LaunchConfiguration("rviz_file")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )
    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="roadmap_explorer parameter file",
    )
    declare_rviz_file = DeclareLaunchArgument(
        "rviz_file",
        default_value=default_rviz_file,
        description="roadmap_explorer rviz config file",
    )
    declare_override_params_file = DeclareLaunchArgument(
        "override_params_file",
        default_value=default_override_params_file,
        description="minimal fairness override params file",
    )

    roadmap_explorer_node = Node(
        package="roadmap_explorer",
        executable="roadmap_exploration_server",
        name="roadmap_explorer_node",
        output="screen",
        parameters=[params_file, override_params_file, {"use_sim_time": use_sim_time}],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_exploration",
        output="screen",
        parameters=[
            {"autostart": True},
            {"node_names": ["roadmap_explorer_node"]},
            {"use_sim_time": use_sim_time},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_roadmap_explorer",
        output="screen",
        arguments=["-d", rviz_file],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            declare_use_sim_time,
            declare_params_file,
            declare_override_params_file,
            declare_rviz_file,
            roadmap_explorer_node,
            lifecycle_manager,
            rviz_node,
        ]
    )
