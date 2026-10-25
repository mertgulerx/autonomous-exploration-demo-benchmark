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

import yaml

from exploration_benchmark_robots.known_pose import GROUND_TRUTH_RECORD_PERIOD_NS

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
    known_pose = generated["known_pose"]
    generated["instrumentation"] = dict(
        version=2,
        navigator="exploration_benchmark_telemetry/Navigator",
        goal_identity="benchmark_local_sequence",
        native_uuid_association="unavailable_in_callback_api",
        ground_truth_source="Gazebo PosePublisher model world pose; finite-difference diagnostic velocity",
        # The publisher and its bridge belong to every run (mapping-known-pose-v1); diagnostics
        # record its stream, decimated to GROUND_TRUTH_RECORD_PERIOD_NS (worker_runtime.py).
        ground_truth_settings=known_pose["ground_truth_settings"],
        ground_truth_recorded_period_ns=GROUND_TRUTH_RECORD_PERIOD_NS,
        ground_truth_tf_forwarded=True,
        model_to_base_offset=(
            "zero by condition (model frame = base_footprint); the measured value is the check "
            "ground_truth_alignment.measured_model_to_base"
        ),
        baseline_navigation_sha256=generated["comparison_profile"]["navigation_sha256"],
        instrumented_navigation_sha256=file_hash(navigation),
    )
    (Path(generated["directory"]) / "launch.json").write_text(json.dumps(generated, indent=2))
