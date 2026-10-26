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

"""Launch an explorer (``profiles.yaml``) from a validated, immutable parameter snapshot."""

from copy import deepcopy
import json
import math
from pathlib import Path
import yaml

from .parameters import nest

# Margin around the target cells farthest from the spawn in roadmap-explorer's exploration boundary (metres).
BOUNDARY_MARGIN_M = 1.0


def exploration_boundary(spawn, reference):
    """roadmap-explorer's exploration polygon in the run's map frame: a square centred on the spawn (the map frame's
    origin) whose half-size covers every target cell of the world reference plus BOUNDARY_MARGIN_M, so it holds the
    world for any spawn yaw. Corner order as the shipped file's (x, y pairs)."""
    half = BOUNDARY_MARGIN_M + max(
        math.hypot(x - spawn[0], y - spawn[1])
        for x, y in zip(reference.world_x, reference.world_y)
    )
    half = round(half, 2)
    return [half, half, half, -half, -half, -half, -half, half]


def default_document(profile):
    """The explorer's Default parameter file as a mapping (every node section)."""
    from ament_index_python.packages import get_package_share_directory

    path = Path(get_package_share_directory(profile["parameter_package"])) / profile["parameters"]
    return yaml.safe_load(path.read_text())


def resolve_parameters(spec, directory, profile, robot=None, spawn=None, reference=None):
    """Write requested_parameters.yaml (the profile's resolved values) and resolved_parameters.yaml (what the
    explorer reads: those values with the benchmark's integration settings). Returns the resolved file and the
    integration settings recorded in the manifest. Without ``spawn`` and ``reference`` (a row cancelled before it
    started) the generated interface values are left at their Defaults."""
    node = profile.get("node", "frontier_explorer")
    requested = {node: {"ros__parameters": nest(deepcopy(spec.overrides))}}
    Path(directory, "requested_parameters.yaml").write_text(
        yaml.safe_dump(requested, sort_keys=False)
    )
    if node == "frontier_explorer":
        integration = {
            "use_sim_time": True,
            "completion_event_enabled": True,
            "completion_event_topic": "exploration_complete",
            "autostart": True,
        }
    else:
        integration = {"use_sim_time": True}
    parameters = deepcopy(requested)
    values = parameters[node]["ros__parameters"]
    values.update(integration)
    if "exploration_boundary" in values.get("explorationBT", {}) and None not in (spawn, reference):
        boundary = exploration_boundary(spawn, reference)
        values["explorationBT"]["exploration_boundary"] = boundary
        integration["explorationBT.exploration_boundary"] = boundary
    # Further node sections of the Default file (roadmap-explorer's costmap) are copied, with the robot's footprint
    # in place of a shipped robot_radius (decision E4).
    for section, content in default_document(profile).items():
        if section == node:
            continue
        content = deepcopy(content)
        for scope in _ros_parameter_scopes(content):
            if "robot_radius" in scope and robot is not None:
                del scope["robot_radius"]
                scope["footprint"] = json.dumps(robot["footprint"])
                integration[f"{section}.footprint"] = scope["footprint"]
            scope["use_sim_time"] = True
        parameters[section] = content
    path = Path(directory) / "resolved_parameters.yaml"
    path.write_text(yaml.safe_dump(parameters, sort_keys=False))
    return str(path), integration


def _ros_parameter_scopes(content):
    """Every ``ros__parameters`` mapping inside a (possibly namespaced) node section."""
    if not isinstance(content, dict):
        return []
    if "ros__parameters" in content:
        return [content["ros__parameters"]]
    return [scope for value in content.values() for scope in _ros_parameter_scopes(value)]


def command(profile, parameters):
    return [
        "ros2",
        "launch",
        profile.get("launch_package", profile["package"]),
        profile["launch"],
        "use_sim_time:=true",
        f"params_file:={parameters}",
    ]
