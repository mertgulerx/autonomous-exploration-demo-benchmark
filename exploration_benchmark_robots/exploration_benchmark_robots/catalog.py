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

"""Immutable experiment profiles and dependency availability."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import yaml


def profile_hash(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()


def resolve_physical_settings(profile, overrides=None):
    profile = deepcopy(profile)
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError("robot_overrides must be a mapping")
    if overrides and profile["id"] != "mogi":
        raise ValueError("Robot settings can only be changed for MOGI")
    defaults = dict(movement_speed=profile["max_linear_speed"],
                    turning_speed=profile["max_angular_speed"],
                    lidar_range=profile["lidar"]["range_max"])
    # The course model's own values; only settings that differ from them change its files.
    native = dict(defaults,
                  movement_speed=profile.get("native_max_linear_speed", profile["max_linear_speed"]))
    requested = {}
    for name, value in overrides.items():
        if name not in defaults:
            raise ValueError(f"Unknown MOGI setting: {name}; use {', '.join(defaults)}")
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        if name == "lidar_range" and value <= profile["lidar"]["range_min"]:
            raise ValueError(f"lidar_range must exceed {profile['lidar']['range_min']} m")
        requested[name] = float(value)
    if profile["id"] == "mogi":
        effective = dict(defaults, **requested)
        profile["benchmark_settings"] = dict(
            defaults=defaults, requested=requested, effective=effective, native=native
        )
        profile["benchmark_overrides"] = {k: v for k, v in effective.items() if v != native[k]}
        profile["max_linear_speed"] = effective["movement_speed"]
        profile["max_angular_speed"] = effective["turning_speed"]
        profile["lidar"]["range_max"] = effective["lidar_range"]
    return profile


class RobotCatalog:
    def __init__(self, directory=None, package_lookup=None):
        from ament_index_python.packages import get_package_share_directory

        self.lookup = package_lookup or get_package_share_directory
        self.directory = Path(directory or self.lookup("exploration_benchmark_robots"))
        data = yaml.safe_load((self.directory / "config/robots.yaml").read_text())
        self.profiles = data["robots"]
        self.default = data["default_robot"]
        for robot_id, profile in self.profiles.items():
            if profile["id"] != robot_id:
                raise ValueError(f"Robot catalog ID mismatch: {robot_id}")
            for key in ("max_linear_speed", "max_angular_speed", "physics_step"):
                if not math.isfinite(profile[key]) or profile[key] <= 0:
                    raise ValueError(f"{robot_id}: {key} must be positive and finite")
            if len(profile["footprint"]) < 3:
                raise ValueError(
                    f"{robot_id}: footprint requires at least three vertices"
                )
            for world, height in profile.get("spawn_z_by_world", {}).items():
                if world not in profile["worlds"] or not math.isfinite(height):
                    raise ValueError(f"{robot_id}: invalid spawn height for {world}")

    def profile(self, robot_id="mogi"):
        if robot_id not in self.profiles:
            raise ValueError(
                f"Unknown robot: {robot_id}; choose an entry in the robot catalog"
            )
        return deepcopy(self.profiles[robot_id])

    def availability(self, robot_id):
        profile = self.profile(robot_id)
        from .runtime import package_lookup

        lookup = package_lookup(profile, self.lookup)
        missing = []
        for package in profile["packages"]:
            try:
                lookup(package)
            except (LookupError, OSError):
                missing.append(package)
        if missing:
            return False, "Install/build robot dependencies: " + ", ".join(missing)
        return True, ""

    def require(self, robot_id, world, overrides=None):
        profile = resolve_physical_settings(self.profile(robot_id), overrides)
        available, reason = self.availability(robot_id)
        if not available:
            raise ValueError(f"{profile['label']}: {reason}")
        worlds = profile["worlds"]
        if world not in worlds:
            raise ValueError(
                f"{profile['label']}: unsupported world {world}; use {', '.join(worlds)}"
            )
        from .dependencies import inventory
        from .runtime import package_lookup, sdk_prefix

        profile["dependencies"] = inventory(
            profile["packages"], package_lookup(profile, self.lookup)
        )
        profile["sdk_prefix"] = sdk_prefix(profile)
        profile["catalog_profile_sha256"] = profile_hash(self.profiles[robot_id])
        family = {"husarion_ugv": "ugv"}.get(profile["family"], profile["family"])
        lock = yaml.safe_load((self.directory / "config/source_lock.yaml").read_text())
        profile["source_revisions"] = {
            name: spec
            for name, spec in lock["sources"].items()
            if spec["family"] == family
        }
        return profile

    def message(self):
        from exploration_benchmark_interfaces.msg import (
            RobotCatalog as Message,
            RobotProfile,
        )

        rows = []
        for robot_id, profile in self.profiles.items():
            available, reason = self.availability(robot_id)
            rows.append(
                RobotProfile(
                    id=robot_id,
                    label=profile["label"],
                    family=profile["family"],
                    profile_version=profile["version"],
                    lidar_label=f"Lidar range: {profile['lidar']['range_max']:g} m",
                    lidar_range=profile["lidar"]["range_max"],
                    kinematics=profile["kinematics"],
                    max_linear_speed=profile["max_linear_speed"],
                    max_angular_speed=profile["max_angular_speed"],
                    physics_step=profile["physics_step"],
                    available=available,
                    reason=reason,
                    worlds=profile["worlds"],
                )
            )
        return Message(robots=rows, default_robot=self.default)
