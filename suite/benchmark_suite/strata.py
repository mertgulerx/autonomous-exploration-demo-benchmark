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

"""Comparison key: the stratum within which profiles are compared and paired.

Version 1 is the hash the report job computes (``report_data.comparison_identity``). Version 2
(docs/MEASUREMENTS.md, "Comparison key version 2"; D-31) hashes the same parts, but the runtime identity keys
checkout commits by repository role instead of by absolute path, so the same code run from another directory
or host forms the same stratum. Runtime identities recorded from 22 October 2026 already use roles; for them
both versions give the same key.
"""

from collections.abc import Mapping
from typing import Any

from .contracts import content_hash
from .provenance import CHECKOUT_ROLE, EXPLORER_ROLE

COMPARISON_KEY_VERSION = 2
PHYSICAL_FIELDS = (
    "id",
    "kinematics",
    "footprint",
    "max_linear_speed",
    "max_angular_speed",
    "max_lateral_speed",
    "linear_acceleration",
    "angular_acceleration",
    "angular_deceleration",
    "physics_step",
    "lidar",
    "dependencies",
    "source_revisions",
)
COVERAGE_FIELDS = ("id", "metric_version", "grid_sha256", "metadata_sha256", "target_area_m2")
GENERATED_FILES = ("navigation.yaml", "slam.yaml", "ekf.yaml")


def physical_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The robot's physical and sensor identity of one attempt."""
    robot = manifest.get("robot", {})
    physical = {name: robot.get(name) for name in PHYSICAL_FIELDS}
    physical["sensor_configuration"] = manifest.get("robot_configuration", {}).get(
        "sensor_configuration"
    )
    return physical


def coverage_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The coverage reference identity of one attempt."""
    reference = manifest.get("coverage", {}).get("reference", {})
    return {name: reference.get(name) for name in COVERAGE_FIELDS}


def key_parts(manifest: Mapping[str, Any], common: Any) -> dict[str, Any]:
    """Everything the key hashes, with the runtime identity ``common`` as the caller resolved it."""
    generated = manifest.get("robot_configuration", {})
    return dict(
        physical=physical_identity(manifest),
        coverage=coverage_identity(manifest),
        spawn=manifest.get("configuration", {}).get("scenario", {}).get("spawn"),
        policy=manifest.get("policy"),
        common=common,
        generated_common={name: generated.get("sha256", {}).get(name) for name in GENERATED_FILES},
    )


def runtime_by_role(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """The runtime identity with absolute checkout paths replaced by repository roles."""
    result = {name: value for name, value in runtime.items() if not name.startswith("/")}
    commits = dict(result.get("commits") or {})
    for name, value in runtime.items():
        if name.startswith("/"):
            role = (
                EXPLORER_ROLE if name.rstrip("/").endswith("/" + EXPLORER_ROLE) else CHECKOUT_ROLE
            )
            commits[role] = value
    if commits:
        result["commits"] = commits
    return result


def comparison_key(manifest: Mapping[str, Any], runtime: Mapping[str, Any]) -> str:
    """Comparison key version 2 of an attempt; ``runtime`` is its resolved runtime object."""
    common = dict(manifest.get("runtime") or {})
    common["object"] = content_hash(runtime_by_role(runtime))
    return content_hash(key_parts(manifest, common))
