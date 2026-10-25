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

# Version 2 leaves each robot's world fit and spawn heights to the world registry
# (exploration_benchmark_runner/config/worlds.yaml); version 3 holds the three equal
# robots (TurtleBot3 Waffle Pi, TurtleBot4 Standard, ROSbot 3 PRO) with their catalog
# limits only (no per-run robot settings).
ROBOT_CATALOG_VERSION = 3
REGISTRY_FILE = "exploration_benchmark_runner/config/worlds.yaml"


def footprint_circumradius(footprint, padding=0.0):
    """Largest distance of a footprint corner from the base frame, each axis padded."""
    return max(math.hypot(abs(x) + padding, abs(y) + padding) for x, y in footprint)


def profile_hash(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()


def fill_robot_profiles(catalog, registry):
    """The profiles of a ``robots.yaml`` mapping, world fit filled from the registry.

    ``registry`` is the world registry (``exploration_benchmark_runner.worlds``
    ``WorldRegistry``; only ``runnable()`` and each world's ``id`` and ``robots`` fits
    are read, so this package does not import the runner, which depends on it). Each
    profile gains ``worlds`` (the runnable registry worlds that fit it, in registry
    order, as ``scenarios`` and the suite catalog list them) and, where the registry
    sets a height, ``spawn_z_by_world``. The catalog must be version 3, which leaves
    both to the registry; a stale catalog that still carries them is refused.
    """
    if not isinstance(catalog, dict):
        raise ValueError("Robot catalog: expected a mapping")
    if catalog.get("catalog_version") != ROBOT_CATALOG_VERSION:
        raise ValueError(
            f"Robot catalog: catalog_version must be {ROBOT_CATALOG_VERSION} "
            "(world fit and spawn heights live in the world registry)"
        )
    profiles = deepcopy(catalog.get("robots"))
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Robot catalog: robots must be a non-empty mapping")
    for robot_id, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"{robot_id}: expected a mapping")
        for key in ("worlds", "spawn_z_by_world"):
            if key in profile:
                raise ValueError(
                    f"{robot_id}: {key} belongs in the world registry ({REGISTRY_FILE})"
                )
        worlds, heights = [], {}
        for world in registry.runnable():
            for fit in world.robots:
                if fit.robot_id == robot_id:
                    worlds.append(world.id)
                    if fit.spawn_z is not None:
                        heights[world.id] = fit.spawn_z
        profile["worlds"] = worlds
        if heights:
            profile["spawn_z_by_world"] = heights
    return profiles


class RobotCatalog:
    def __init__(self, directory=None, package_lookup=None, *, worlds):
        """``worlds``: the world registry (``exploration_benchmark_runner.worlds``) that
        fills each profile's world fit (``fill_robot_profiles``): the checkout's for the
        suite and tools, the installed one for the manual runner
        (``configuration.installed_worlds``). Callers load it, because the runner
        depends on this package."""
        from ament_index_python.packages import get_package_share_directory

        self.lookup = package_lookup or get_package_share_directory
        self.registry = worlds
        self.directory = Path(directory or self.lookup("exploration_benchmark_robots"))
        data = yaml.safe_load((self.directory / "config/robots.yaml").read_text())
        self.profiles = fill_robot_profiles(data, self.registry)
        self.default = data.get("default_robot")
        if self.default not in self.profiles:
            raise ValueError(
                f"Robot catalog: default_robot {self.default!r} is not a catalog robot"
            )
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

    def profile(self, robot_id):
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

    def require(self, robot_id, world):
        profile = self.profile(robot_id)
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
        lock = yaml.safe_load((self.directory / "config/source_lock.yaml").read_text())
        profile["source_revisions"] = {
            name: spec
            for name, spec in lock["sources"].items()
            if spec["family"] == profile["family"]
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
