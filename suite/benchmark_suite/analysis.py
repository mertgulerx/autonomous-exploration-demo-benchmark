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

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
import math
from typing import Any

from .contracts import ContractError, known_keys, mapping, number

ANALYSIS_VERSION = "motion-coverage-v1"


@dataclass(frozen=True, slots=True)
class DetectorSettings:
    stationary_speed: float = 0.01
    stationary_angular_speed: float = 0.05
    interval_seconds: float = 5.0
    slow_limit_fraction: float = 0.25
    command_speed: float = 0.05
    command_angular_speed: float = 0.15
    maximum_motion_gap_seconds: float = 0.5
    plateau_seconds: float = 30.0
    plateau_gain_percent: float = 0.25
    maximum_coverage_gap_seconds: float = 1.5
    revisit_separation_seconds: float = 15.0
    loop_minimum_distance: float = 3.0
    navigation_progress_seconds: float = 10.0
    navigation_progress_epsilon_m: float = 0.1
    maximum_navigation_gap_seconds: float = 0.5

    @classmethod
    def accept(cls, raw: Any = None) -> "DetectorSettings":
        values = mapping(raw if raw is not None else {}, "detectors")
        known_keys(values, {field.name for field in fields(cls)}, "detectors")
        accepted = {
            name: number(value, "detectors." + name, minimum=0.000001)
            for name, value in values.items()
        }
        if accepted.get("slow_limit_fraction", 0.25) > 1:
            raise ContractError(
                "detectors.slow_limit_fraction", "Use a fraction no greater than one"
            )
        return cls(**accepted)


def elapsed(row: dict[str, Any]) -> float | None:
    value = row.get("simulation_elapsed_ns")
    return value / 1e9 if value is not None else None


def valid_motion(row: dict[str, Any]) -> bool:
    return bool(
        row.get("valid")
        and elapsed(row) is not None
        and all(
            isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
            for key in ("x", "y", "yaw", "speed", "wz")
        )
    )


def position(row: dict[str, Any]) -> tuple[float, float]:
    """World position of a trajectory row (odometry through the frozen alignment) or a mapped goal.

    ``world_x``/``world_y`` are the goal positions report_data places; this module retires with R7.
    """
    x = row.get("odom_aligned_x_m", row.get("world_x"))
    y = row.get("odom_aligned_y_m", row.get("world_y"))
    return float(x if x is not None else row["x"]), float(y if y is not None else row["y"])


@dataclass(slots=True)
class Segment:
    first: dict[str, Any]
    last: dict[str, Any]
    start_row: int
    end_row: int
    start_distance: float
    end_distance: float
    bbox: list[float]

    @classmethod
    def start(cls, row: dict[str, Any], index: int, distance: float) -> "Segment":
        x, y = position(row)
        return cls(row, row, index, index, distance, distance, [x, y, x, y])

    def update(self, row: dict[str, Any], index: int, distance: float) -> None:
        self.last, self.end_row, self.end_distance = row, index, distance
        x, y = position(row)
        self.bbox = [
            min(self.bbox[0], x),
            min(self.bbox[1], y),
            max(self.bbox[2], x),
            max(self.bbox[3], y),
        ]

    def incident(self, kind: str, evidence: dict[str, Any]) -> dict[str, Any]:
        import json

        first, last = self.first, self.last
        return dict(
            first,
            type=kind,
            detector_version=ANALYSIS_VERSION,
            start_ns=first["simulation_elapsed_ns"],
            end_ns=last["simulation_elapsed_ns"],
            trajectory_start_row=self.start_row,
            trajectory_end_row=self.end_row,
            duration_ns=last["simulation_elapsed_ns"] - first["simulation_elapsed_ns"],
            distance_m=max(0.0, self.end_distance - self.start_distance),
            x=self.bbox[0],
            y=self.bbox[1],
            bbox=self.bbox,
            evidence_json=json.dumps(evidence, allow_nan=False),
        )


class MotionDetector:
    def __init__(
        self, linear_limit: float, footprint: list[list[float]], settings: DetectorSettings
    ):
        self.settings = settings
        self.limit = linear_limit
        self.radius = max(0.1, max((math.hypot(*point) for point in footprint), default=0.25))
        self.visited: dict[tuple[int, int], tuple[dict[str, Any], int, float]] = {}
        self.previous: dict[str, Any] | None = None
        self.index = -1
        self.distance = 0.0
        self.last_spatial_incident = -math.inf
        self.segments: dict[str, Segment] = {}
        self.durations = {"moving": 0.0, "stationary": 0.0, "turning_in_place": 0.0}
        self.gaps = 0
        self.valid_samples = 0
        self.spatial_limit_reached = False

    def state(self, row: dict[str, Any]) -> str:
        if row["speed"] >= self.settings.stationary_speed:
            return "moving"
        return (
            "stationary"
            if abs(row["wz"]) < self.settings.stationary_angular_speed
            else "turning_in_place"
        )

    def close_segments(self) -> list[dict[str, Any]]:
        result = [
            self._incident(kind, segment)
            for kind, segment in self.segments.items()
            if (elapsed(segment.last) or 0) - (elapsed(segment.first) or 0)
            >= self.settings.interval_seconds
        ]
        self.segments.clear()
        return result

    def _incident(self, kind: str, segment: Segment) -> dict[str, Any]:
        return segment.incident(
            kind,
            dict(
                source=segment.first["source"],
                thresholds=asdict(self.settings),
                linear_limit_m_s=self.limit,
                goal_sequence=segment.first.get("active_goal_sequence"),
                position_source="world"
                if segment.first.get("odom_aligned_x_m", segment.first.get("world_x")) is not None
                or segment.first["source"].startswith("gazebo")
                else "odometry",
            ),
        )

    def feed(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        self.index += 1
        result: list[dict[str, Any]] = []
        if not valid_motion(row):
            if self.previous is not None:
                self.gaps += 1
            self.previous = None
            return self.close_segments()
        self.valid_samples += 1
        if self.previous is not None:
            dt = (elapsed(row) or 0) - (elapsed(self.previous) or 0)
            if dt <= 0 or dt > self.settings.maximum_motion_gap_seconds:
                self.gaps += 1
                result.extend(self.close_segments())
            else:
                self.durations[self.state(self.previous)] += dt
                self.distance += math.dist(position(row), position(self.previous))
        speed, angular = row["speed"], abs(row["wz"])
        predicates = {
            "stationary": self.state(row) == "stationary",
            "slow_motion": bool(row.get("active_goal_sequence"))
            and self.settings.stationary_speed
            <= speed
            < self.limit * self.settings.slow_limit_fraction,
            "command_motion_discrepancy": (
                math.hypot(row.get("command_vx") or 0, row.get("command_vy") or 0)
                >= self.settings.command_speed
                and speed < self.settings.stationary_speed
            )
            or (
                abs(row.get("command_wz") or 0) >= self.settings.command_angular_speed
                and angular < self.settings.stationary_angular_speed
            ),
        }
        for kind, active in predicates.items():
            segment = self.segments.get(kind)
            if active:
                if segment is None:
                    self.segments[kind] = Segment.start(row, self.index, self.distance)
                else:
                    segment.update(row, self.index, self.distance)
            elif segment is not None:
                segment.update(row, self.index, self.distance)
                if (elapsed(row) or 0) - (
                    elapsed(segment.first) or 0
                ) >= self.settings.interval_seconds:
                    result.append(self._incident(kind, segment))
                del self.segments[kind]
        x, y = position(row)
        cell = math.floor(x / self.radius), math.floor(y / self.radius)
        if not self.spatial_limit_reached:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    previous = self.visited.get((cell[0] + dx, cell[1] + dy))
                    if previous is None:
                        continue
                    start, index, distance = previous
                    now, then = elapsed(row) or 0, elapsed(start) or 0
                    separation, traveled = now - then, self.distance - distance
                    if (
                        separation >= self.settings.revisit_separation_seconds
                        and traveled >= 2 * self.radius
                        and math.dist(position(start), (x, y)) <= self.radius
                        and now - self.last_spatial_incident
                        >= self.settings.revisit_separation_seconds
                    ):
                        segment = Segment.start(start, index, distance)
                        segment.update(row, self.index, self.distance)
                        evidence = dict(
                            source=row["source"],
                            spatial_tolerance_m=self.radius,
                            time_separation_s=separation,
                            substantial_distance_m=max(
                                self.settings.loop_minimum_distance, 10 * self.radius
                            ),
                        )
                        result.append(segment.incident("revisit", evidence))
                        if traveled >= max(self.settings.loop_minimum_distance, 10 * self.radius):
                            result.append(segment.incident("closed_loop", evidence))
                        self.last_spatial_incident = now
            self.visited[cell] = row, self.index, self.distance
            if len(self.visited) > 250000:
                self.spatial_limit_reached = True
                self.visited.clear()
        self.previous = row
        return result


def with_navigation(
    motion: Iterable[dict[str, Any]], navigation: Iterable[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    events = iter(navigation)
    event = next(events, None)
    active = None
    for row in motion:
        clock = row.get("clock_ns")
        while event is not None and clock is not None and (event.get("clock_ns") or 0) <= clock:
            if event["event"] == "initialized":
                active = event["goal_sequence"]
            elif event["event"] == "terminal":
                active = None
            event = next(events, None)
        yield dict(row, active_goal_sequence=active)


def coverage_plateaus(
    rows: Iterable[dict[str, Any]], settings: DetectorSettings
) -> Iterator[dict[str, Any]]:
    window: deque[dict[str, Any]] = deque()
    segment: Segment | None = None
    previous: float | None = None
    for index, row in enumerate(rows):
        now = elapsed(row)
        valid = (
            row.get("valid")
            and row.get("status") == "valid"
            and now is not None
            and row.get("coverage_percent") is not None
        )
        gap = (
            previous is not None
            and now is not None
            and (now <= previous or now - previous > settings.maximum_coverage_gap_seconds)
        )
        if not valid or gap:
            if segment is not None:
                yield segment.incident(
                    "coverage_plateau",
                    dict(
                        source="slam_map",
                        gain_percent=segment.last["coverage_percent"]
                        - segment.first["coverage_percent"],
                        thresholds=asdict(settings),
                    ),
                )
                segment = None
            window.clear()
        previous = now if valid else None
        if not valid:
            continue
        assert now is not None
        row = dict(row, x=0.0, y=0.0)
        window.append(row)
        while len(window) > 1 and now - (elapsed(window[1]) or 0) >= settings.plateau_seconds:
            window.popleft()
        plateau = (
            now - (elapsed(window[0]) or 0) >= settings.plateau_seconds
            and row["coverage_percent"] - window[0]["coverage_percent"]
            < settings.plateau_gain_percent
        )
        if plateau:
            if segment is None:
                segment = Segment.start(window[0], -1, 0)
            segment.update(row, -1, 0)
        elif segment is not None:
            yield segment.incident(
                "coverage_plateau",
                dict(
                    source="slam_map",
                    gain_percent=segment.last["coverage_percent"]
                    - segment.first["coverage_percent"],
                    thresholds=asdict(settings),
                ),
            )
            segment = None
    if segment is not None:
        yield segment.incident(
            "coverage_plateau",
            dict(
                source="slam_map",
                gain_percent=segment.last["coverage_percent"] - segment.first["coverage_percent"],
                thresholds=asdict(settings),
            ),
        )


def navigation_no_progress(
    rows: Iterable[dict[str, Any]], settings: DetectorSettings
) -> Iterator[dict[str, Any]]:
    window: deque[dict[str, Any]] = deque()
    segment: Segment | None = None
    previous: dict[str, Any] | None = None

    def incident(value: Segment) -> dict[str, Any]:
        return value.incident(
            "navigation_no_progress",
            dict(
                source="native_action_feedback",
                native_uuid=value.first["native_uuid"],
                goal_sequence_association="unavailable",
                distance_remaining_change_m=value.first["distance_remaining"]
                - value.last["distance_remaining"],
                thresholds=asdict(settings),
            ),
        )

    for row in rows:
        if row["event"] != "feedback":
            continue
        now = elapsed(row)
        valid = (
            row.get("valid")
            and now is not None
            and row.get("distance_remaining") is not None
            and row.get("native_uuid")
        )
        broken = previous is not None and (
            row.get("native_uuid") != previous["native_uuid"]
            or now is None
            or now - (elapsed(previous) or 0) <= 0
            or now - (elapsed(previous) or 0) > settings.maximum_navigation_gap_seconds
        )
        if not valid or broken:
            if segment is not None:
                yield incident(segment)
            segment = None
            window.clear()
        previous = row if valid else None
        if not valid:
            continue
        assert now is not None
        window.append(row)
        while (
            len(window) > 1
            and now - (elapsed(window[1]) or 0) >= settings.navigation_progress_seconds
        ):
            window.popleft()
        stalled = (
            now - (elapsed(window[0]) or 0) >= settings.navigation_progress_seconds
            and window[0]["distance_remaining"] - row["distance_remaining"]
            < settings.navigation_progress_epsilon_m
        )
        if stalled:
            if segment is None:
                segment = Segment.start(window[0], -1, 0)
            segment.update(row, -1, 0)
        elif segment is not None:
            yield incident(segment)
            segment = None
    if segment is not None:
        yield incident(segment)


def tail_durations(
    milestones: Iterable[dict[str, Any]], completion: float | None
) -> dict[str, float | None]:
    times = {
        item["percent"]: item.get("simulation_elapsed")
        for item in milestones
        if item.get("reached")
    }
    start, end = times.get(90), times.get(95)
    return {
        "90_to_95_seconds": end - start if end is not None and start is not None else None,
        "90_to_completion_seconds": completion - start
        if start is not None and completion is not None
        else None,
    }
