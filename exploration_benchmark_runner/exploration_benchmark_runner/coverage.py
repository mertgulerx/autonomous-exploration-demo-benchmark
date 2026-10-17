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

"""Fixed-reference SLAM coverage, independent of ROS and explorer inputs.

Reference cell centres are queried in the current SLAM grid. This intentionally
does not union old maps: loop-closure corrections may reduce measured coverage.
"""

import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from zipfile import BadZipFile

import numpy as np

METRIC_VERSION = 1
MILESTONES = (25.0, 50.0, 75.0, 90.0, 95.0)  # T25 added 17 October 2026
SAMPLE_INTERVAL = 1.0
STALE_SECONDS = 5.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_directory():
    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory("exploration_benchmark_runner"))
        / "config/coverage"
    )


def resolve_asset(root, identity):
    """Portable provenance: project-relative files or explicitly versioned Fuel.

    No network requests during preflight. Fuel assets must already be available
    in the same resource cache Gazebo uses; errors include a preparation command.
    """
    if not identity.startswith("fuel:"):
        return Path(root) / identity
    uri, relative = identity[5:].split("#", 1)
    parsed = urlparse(uri)
    parts = [unquote(part).lower() for part in parsed.path.split("/") if part]
    if (
        len(parts) != 5
        or parts[0] != "1.0"
        or parts[2] != "models"
        or not parts[4].isdigit()
    ):
        raise ValueError(f"Coverage requires a versioned Fuel model: {uri}")
    cache = Path(os.environ.get("BENCHMARK_FUEL_CACHE", str(Path.home() / ".gz/fuel")))
    base = cache / parsed.netloc / parts[1] / "models"
    names = (parts[3], quote(parts[3], safe=""))
    path = next(
        (
            base / name / parts[4] / relative
            for name in names
            if (base / name / parts[4] / relative).is_file()
        ),
        base / parts[3] / parts[4] / relative,
    )
    if not path.is_file():
        raise ValueError(
            f"Coverage asset missing: {path}. Prepare it with: gz fuel download -u {uri}"
        )
    return path


class Reference:
    def __init__(self, metadata_file):
        self.file = Path(metadata_file)
        try:
            self.metadata_bytes = self.file.read_bytes()
            self.metadata = json.loads(self.metadata_bytes)
            data = self.metadata
            if data["metric_version"] != METRIC_VERSION:
                raise ValueError("unsupported metric version")
            self.grid_file = self.file.parent / data["grid_file"]
            self.grid_bytes = self.grid_file.read_bytes()
            if hashlib.sha256(self.grid_bytes).hexdigest() != data["grid_sha256"]:
                raise ValueError("grid hash mismatch")
            with np.load(BytesIO(self.grid_bytes), allow_pickle=False) as grid:
                self.collision = grid["collision"].astype(bool)
                self.inclusion = grid["inclusion"].astype(bool)
            self.resolution = float(data["resolution"])
            self.origin = tuple(float(v) for v in data["origin"])
            if len(self.origin) != 3 or not all(math.isfinite(v) for v in self.origin):
                raise ValueError("invalid reference origin")
            if not math.isfinite(self.resolution) or self.resolution <= 0:
                raise ValueError("invalid reference resolution")
            shape = (data["height"], data["width"])
            if self.collision.shape != shape or self.inclusion.shape != shape:
                raise ValueError("reference dimensions disagree")
            self.target = self.inclusion & ~self.collision
            rows, columns = np.nonzero(self.target)
            self.count = len(rows)
            if not self.count or self.count != data["target_cells"]:
                raise ValueError("empty or inconsistent target mask")
            x, y = (columns + 0.5) * self.resolution, (rows + 0.5) * self.resolution
            c, s = math.cos(self.origin[2]), math.sin(self.origin[2])
            self.world_x = self.origin[0] + c * x - s * y
            self.world_y = self.origin[1] + s * x + c * y
            self.area = self.count * self.resolution**2
            if not math.isclose(self.area, data["target_area_m2"], abs_tol=1e-8):
                raise ValueError("reference area disagrees")
            self.identity = data["id"]
            self.metadata_hash = hashlib.sha256(self.metadata_bytes).hexdigest()
        except (
            OSError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            ValueError,
            BadZipFile,
        ) as exc:
            raise ValueError(f"Invalid coverage reference {self.file}: {exc}") from exc

    def validate_scenario(self, root, scenario, *, allow_custom_spawn=False):
        root = Path(root).resolve()
        data = self.metadata
        if data["world"] != scenario["world"] or (not allow_custom_spawn and data["spawn"] != scenario["spawn"]):
            raise ValueError(
                "Coverage reference does not match the selected world/spawn"
            )
        for identity, expected in data["source_sha256"].items():
            path = resolve_asset(root, identity)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(
                    f"Coverage reference is outdated: {identity}. Regenerate and review it with benchmark_reference."
                )
        if sha256(scenario["world_file"]) != data["source_sha256"].get(
            str(Path(scenario["world_file"]).relative_to(root))
        ):
            raise ValueError("Coverage reference world hash does not match")

    def descriptor(self):
        return dict(
            id=self.identity,
            metric_version=METRIC_VERSION,
            metadata_file=str(self.file.resolve()),
            metadata_sha256=self.metadata_hash,
            grid_sha256=self.metadata["grid_sha256"],
            target_area_m2=self.area,
        )

    def save(self, directory):
        """Copy the exact loaded snapshot, even if source files changed later."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.file.name).write_bytes(self.metadata_bytes)
        target = directory / self.metadata["grid_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.grid_bytes)


def attach_reference(root, scenario, directory=None, *, validate=True):
    reference = Reference(
        Path(directory or reference_directory()) / (scenario["world"] + ".json")
    )
    if validate:
        reference.validate_scenario(root, scenario)
    return dict(scenario, coverage_reference=reference.descriptor())


def planar_pose(pose):
    q = pose.orientation
    values = (pose.position.x, pose.position.y, q.x, q.y, q.z, q.w)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite pose")
    if (
        abs(q.x) > 1e-5
        or abs(q.y) > 1e-5
        or not math.isclose(q.z**2 + q.w**2, 1.0, abs_tol=1e-3)
    ):
        raise ValueError("coverage requires a normalized planar pose")
    return pose.position.x, pose.position.y, math.atan2(2 * q.w * q.z, 1 - 2 * q.z**2)


class Calculator:
    def __init__(self, reference, world_from_map):
        self.reference = reference
        self.alignment = tuple(world_from_map)
        if len(self.alignment) != 3 or not all(
            math.isfinite(v) for v in self.alignment
        ):
            raise ValueError("invalid map-to-world alignment")
        x, y, yaw = self.alignment
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = reference.world_x - x, reference.world_y - y
        self.map_x, self.map_y = c * dx + s * dy, -s * dx + c * dy
        self.geometry_key = None

    def count(self, grid):
        if grid is None:
            raise ValueError("map missing")
        info = grid.info
        width, height, resolution = info.width, info.height, info.resolution
        if width <= 0 or height <= 0 or len(grid.data) != width * height:
            raise ValueError("map dimensions/data disagree")
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("invalid map resolution")
        if grid.header.frame_id.lstrip("/") != "map":
            raise ValueError("unexpected map frame")
        origin = planar_pose(info.origin)
        key = (width, height, resolution, origin)
        if key != self.geometry_key:
            dx, dy = self.map_x - origin[0], self.map_y - origin[1]
            c, s = math.cos(origin[2]), math.sin(origin[2])
            columns = np.floor((c * dx + s * dy) / resolution).astype(np.int64)
            rows = np.floor((-s * dx + c * dy) / resolution).astype(np.int64)
            inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
            self.inside = inside
            self.indices = rows[inside] * width + columns[inside]
            self.geometry_key = key
        values = np.asarray(grid.data)
        if np.any((values < -1) | (values > 100)):
            raise ValueError("invalid occupancy values")
        return int(np.count_nonzero(values[self.indices] == 0))

    def free_cells(self, grid):
        """Per reference target cell (reference order): whether count() counts it free."""
        self.count(grid)
        free = np.zeros(self.reference.count, dtype=bool)
        free[self.inside] = np.asarray(grid.data)[self.indices] == 0
        return free


class Coverage:
    """One worker's sampling policy. Call only from its supervision thread."""

    def __init__(self, calculator):
        self.calculator = calculator
        self.motion_sim = None
        self.last_sample_sim = None
        self.last_clock = None
        self.last_advance_wall = None
        self.frozen = False
        self.flags = set()
        self.baseline = None
        self.milestones = [
            dict(percent=p, reached=False, simulation_elapsed=None)
            for p in MILESTONES
        ]
        self.latest = dict(
            coverage_percent=float("nan"),
            coverage_covered_m2=0.0,
            coverage_target_m2=calculator.reference.area,
            coverage_status="pending",
            coverage_map_stamp=float("nan"),
            coverage_map_age=float("nan"),
        )

    def snapshot(self):
        return dict(
            self.latest,
            coverage_quality_flags=sorted(self.flags),
            coverage_milestones=[dict(m) for m in self.milestones],
        )

    def start_motion(self, simulation, wall, grid, received_wall, received_unix):
        if self.motion_sim is not None or self.frozen:
            return None
        self.motion_sim = simulation
        # Movement itself is evidence that the active simulation has advanced.
        # The captured baseline map must pass the same delivery-age policy.
        if self.last_advance_wall is None:
            self.last_advance_wall = wall
        row = self._evaluate(
            simulation, wall, grid, received_wall, received_unix, "baseline"
        )
        self.baseline = dict(row)
        self.last_sample_sim = simulation
        return row

    def update(
        self, simulation, wall, grid, received_wall, received_unix, *, force=False
    ):
        if self.frozen:
            return None
        if self.last_clock is not None and simulation < self.last_clock:
            self.flags.add("coverage_clock_reset")
        if self.last_clock is None or simulation > self.last_clock:
            self.last_advance_wall = wall
        self.last_clock = simulation
        if (
            not force
            and self.last_sample_sim is not None
            and simulation - self.last_sample_sim < SAMPLE_INTERVAL
        ):
            return None
        self.last_sample_sim = simulation
        return self._evaluate(
            simulation,
            wall,
            grid,
            received_wall,
            received_unix,
            "active" if self.motion_sim is not None else "pre_motion",
        )

    def _evaluate(self, simulation, wall, grid, received_wall, received_unix, stage):
        status, covered = "valid", 0
        try:
            covered = self.calculator.count(grid)
            source = grid.header.stamp.sec + grid.header.stamp.nanosec * 1e-9
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            status, source = "unavailable", float("nan")
            self.flags.add("coverage_map_invalid:" + str(exc))
        advancing = (
            self.last_advance_wall is not None and wall - self.last_advance_wall < 1.0
        )
        if (
            status == "valid"
            and received_wall is not None
            and wall - received_wall > STALE_SECONDS
            and advancing
        ):
            status = "stale"
            self.flags.add("coverage_map_delivery_gap")
        if "coverage_clock_reset" in self.flags:
            status = "unavailable"
        percent = (
            100 * covered / self.calculator.reference.count
            if status != "unavailable"
            else float("nan")
        )
        self.latest = dict(
            coverage_percent=percent,
            coverage_covered_m2=covered * self.calculator.reference.resolution**2,
            coverage_target_m2=self.calculator.reference.area,
            coverage_status=status,
            coverage_map_stamp=source,
            coverage_map_age=(
                max(0.0, simulation - source) if math.isfinite(source) else float("nan")
            ),
        )
        elapsed_sim = (
            max(0.0, simulation - self.motion_sim)
            if self.motion_sim is not None
            else None
        )
        crossed = []
        if status == "valid" and self.motion_sim is not None:
            for milestone in self.milestones:
                if not milestone["reached"] and percent >= milestone["percent"]:
                    milestone.update(
                        reached=True,
                        simulation_elapsed=elapsed_sim,
                    )
                    crossed.append(dict(milestone))
        return dict(
            stage=stage,
            simulation=simulation,
            wall_monotonic=wall,
            simulation_elapsed=elapsed_sim,
            map_received_wall=received_wall,
            map_received_unix=received_unix,
            **self.latest,
            milestones_crossed=crossed,
        )

    def freeze(self):
        self.frozen = True
        if self.motion_sim is None:
            self.flags.add("coverage_motion_not_observed")
        return self.snapshot()
