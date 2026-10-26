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
"""rrt-explore's node with the benchmark's resolved parameter file (profiles.yaml rrt_explore); as the package's
own rrt.launch.py with namespace robot1, except that TF and navigate_to_pose stay the run's global names (the
shipped file remaps TF into the namespace for its multi-robot setup). The node takes its robot ID from the
namespace and stays idle until the worker publishes true on /robot1/rrt/enable. The file's section key rrt would
not match the namespaced node /robot1/rrt, so the section's values are passed to the node directly."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

NAMESPACE = "robot1"


def explorer(context):
    path = LaunchConfiguration("params_file").perform(context)
    with open(path, encoding="utf-8") as stream:
        values = yaml.safe_load(stream)["rrt"]["ros__parameters"]
    return [
        Node(
            package="rrt_explore",
            executable="rrt",
            name="rrt",
            namespace=NAMESPACE,
            output="screen",
            remappings=[("navigate_to_pose", "/navigate_to_pose")],
            parameters=[
                values,
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("params_file"),
            OpaqueFunction(function=explorer),
        ]
    )
