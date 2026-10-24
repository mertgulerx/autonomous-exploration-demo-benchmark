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

"""Resolve and validate the complete batch before any process can start."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import os
import math
import yaml

GAZEBO_SEED = 1234


def robot_parameter_schema(schema, robot):
    from copy import deepcopy

    schema = deepcopy(schema)
    if robot.get("adapter") != "mogi":
        schema.defaults.update(max_linear_speed_vmax=float(robot["max_linear_speed"]),
                               max_angular_speed_wmax=float(robot["max_angular_speed"]))
    return schema


@dataclass(frozen=True)
class Specification:
    profile: str = "default"
    label: str = ""
    parameter_file: str = ""
    overrides: dict = field(default_factory=dict)

    def dictionary(self):
        return asdict(self)

    @classmethod
    def from_message(cls, message):
        from .parameters import parameter_values

        return cls(
            message.profile or "default",
            message.label,
            message.parameter_file,
            parameter_values(message.overrides),
        )

    def message(self):
        from exploration_benchmark_interfaces.msg import RunSpec
        from .parameters import ros_parameters

        return RunSpec(
            profile=self.profile,
            label=self.label,
            parameter_file=self.parameter_file,
            overrides=ros_parameters(self.overrides),
        )


class Catalog:
    def __init__(self, path=None, root=None):
        self.root = Path(root or os.environ.get("BENCHMARK_PROJECT_ROOT", os.getcwd()))
        if path is None:
            from ament_index_python.packages import get_package_share_directory

            path = (
                Path(get_package_share_directory("exploration_benchmark_runner"))
                / "config/profiles.yaml"
            )
        data = yaml.safe_load(Path(path).read_text())
        self.profiles = data["profiles"]
        self.default_comparison = data["default_comparison"]
        from .parameters import ParameterSchema

        baseline = self.parameters(Specification())
        self.schema = ParameterSchema(baseline, Path(path).parent / "parameters.yaml")

    def parameters(self, spec):
        if spec.parameter_file:
            return str((self.root / Path(spec.parameter_file).expanduser()).resolve())
        profile = self.profiles[spec.profile]
        if "parameters" not in profile:
            return ""
        from ament_index_python.packages import get_package_share_directory

        return str(
            Path(get_package_share_directory(profile["parameter_package"]))
            / profile["parameters"]
        )

    def schema_for_robot(self, robot):
        return robot_parameter_schema(self.schema, robot)

    def validate(self, specs, limit, comparison=True, robot=None):
        if comparison and not 2 <= len(specs) <= 4:
            raise ValueError("A comparison requires 2–4 runs")
        if not comparison and len(specs) != 1:
            raise ValueError("A single run requires exactly one profile")
        if not math.isfinite(limit) or limit < 0:
            raise ValueError(
                "Simulation limit must be finite and nonnegative; zero disables it"
            )
        resolved = []
        schema = self.schema_for_robot(robot) if robot else self.schema
        for spec in specs:
            if spec.profile not in self.profiles:
                raise ValueError(
                    f"Unknown profile: {spec.profile}; only default is supported"
                )
            if len(spec.label) > 160 or "\n" in spec.label or "\r" in spec.label:
                raise ValueError(
                    "Labels must be single-line strings of at most 160 characters"
                )
            filename = (
                str((self.root / Path(spec.parameter_file).expanduser()).resolve())
                if spec.parameter_file
                else ""
            )
            text = Path(filename).read_text() if filename else ""
            if not isinstance(spec.overrides, dict):
                raise ValueError("Parameter overrides must be a mapping")
            values = schema.resolve(text, spec.overrides)
            # Snapshot here. Workers never reopen mutable user parameter files.
            resolved.append(
                Specification(spec.profile, spec.label or "Default", filename, values)
            )
        return resolved


def installed_worlds(root):
    """The manual runner's world registry: the installed copy beside ``profiles.yaml``,
    with paths relative to the project ``root`` and the installed coverage references.
    """
    from ament_index_python.packages import get_package_share_directory
    from .worlds import WorldRegistry

    share = Path(get_package_share_directory("exploration_benchmark_runner")) / "config"
    return WorldRegistry.load(
        root, share / "worlds.yaml", references=share / "coverage"
    )


def scenario(root, world, spawn=None, registry=None):
    """The world's file, SDF world name and spawn ``[x, y, z, yaw]``.

    A registry ID resolves through the world registry (default: the checkout's), with
    its default spawn unless ``spawn`` is given. An absolute world file path is for
    manual use and needs an explicit ``spawn``. ``moving_objects`` (the registry's
    block as plain data) and ``source`` (the world.yaml of a collection world, its path
    absolute) are null for tier A and absolute paths.
    """
    path = Path(world)
    moving_objects = source = None
    if path.is_absolute():
        if spawn is None:
            raise ValueError(f"A world file outside the registry needs a spawn: {path}")
        pose = spawn
    else:
        from .worlds import WorldRegistry

        if registry is None:
            registry = WorldRegistry.load(root)
        entry = registry.world(world)
        path = registry.path(entry.world_file)
        pose = spawn or entry.default_spawn.pose()
        moving_objects = entry.moving_objects  # a fresh plain copy, or None
        if entry.source is not None:
            source = {
                "world_yaml": str(registry.path(entry.source.world_yaml).resolve()),
                "world_yaml_sha256": entry.source.world_yaml_sha256,
            }
    if not path.is_file():
        raise ValueError(f"World does not exist: {path}")
    import xml.etree.ElementTree as ET

    name = ET.parse(path).getroot().find("world").attrib["name"]
    if len(pose) != 4 or not all(math.isfinite(v) for v in pose):
        raise ValueError("Spawn requires finite x, y, z, yaw")
    return {
        "world": world,
        "world_file": str(path.resolve()),
        "world_name": name,
        "spawn": pose,
        "moving_objects": moving_objects,
        "source": source,
    }


def scenarios(root, registry=None):
    """Every registry world the runner offers (not development ones), in file order."""
    from .worlds import WorldRegistry

    if registry is None:
        registry = WorldRegistry.load(root)
    return [scenario(root, item.id, registry=registry) for item in registry.runnable()]
