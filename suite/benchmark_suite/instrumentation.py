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

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import yaml

from .storage import file_hash


def prepare(generated: dict[str, Any]) -> None:
    from ament_index_python.packages import get_package_share_directory

    get_package_share_directory("exploration_benchmark_telemetry")
    navigation = Path(generated["navigation"])
    values = yaml.safe_load(navigation.read_text())
    values["bt_navigator"]["ros__parameters"]["navigate_to_pose"]["plugin"] = (
        "exploration_benchmark_telemetry/Navigator"
    )
    navigation.write_text(yaml.safe_dump(values, sort_keys=False))
    spawn = Path(generated["spawn_file"])
    tree = ET.parse(spawn)
    root = tree.getroot()
    model = ET.SubElement(root, "gazebo") if root.tag == "robot" else root.find("model")
    if model is None:
        raise ValueError("Diagnostic instrumentation requires a URDF robot or SDF model")
    plugin = ET.SubElement(
        model,
        "plugin",
        filename="gz-sim-pose-publisher-system",
        name="gz::sim::systems::PosePublisher",
    )
    settings = dict(
        topic="/benchmark/ground_truth/pose",
        publish_model_pose=True,
        publish_nested_model_pose=False,
        publish_link_pose=False,
        publish_sensor_pose=False,
        publish_collision_pose=False,
        publish_visual_pose=False,
        use_pose_vector_msg=True,
        static_publisher=False,
        update_frequency=20,
    )
    for name, value in settings.items():
        ET.SubElement(plugin, name).text = (
            str(value).lower() if isinstance(value, bool) else str(value)
        )
    tree.write(spawn, encoding="unicode", xml_declaration=True)
    bridge_file = Path(generated["bridge"])
    bridge = yaml.safe_load(bridge_file.read_text())
    bridge.append(
        dict(
            ros_topic_name="/benchmark/ground_truth/pose",
            gz_topic_name="/benchmark/ground_truth/pose",
            ros_type_name="tf2_msgs/msg/TFMessage",
            gz_type_name="gz.msgs.Pose_V",
            direction="GZ_TO_ROS",
        )
    )
    bridge_file.write_text(yaml.safe_dump(bridge, sort_keys=False))
    generated["instrumentation"] = dict(
        version=1,
        navigator="exploration_benchmark_telemetry/Navigator",
        goal_identity="benchmark_local_sequence",
        native_uuid_association="unavailable_in_callback_api",
        ground_truth_source="Gazebo PosePublisher model world pose; finite-difference diagnostic velocity",
        ground_truth_settings=settings,
        ground_truth_tf_forwarded=False,
        model_to_base_offset="measured from stationary preparation; diagnostic only",
        baseline_navigation_sha256=generated["comparison_profile"]["navigation_sha256"],
        instrumented_navigation_sha256=file_hash(navigation),
        instrumented_spawn_sha256=file_hash(spawn),
        instrumented_bridge_sha256=file_hash(bridge_file),
    )
    (Path(generated["directory"]) / "launch.json").write_text(json.dumps(generated, indent=2))
