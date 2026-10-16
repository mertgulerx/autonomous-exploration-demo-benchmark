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

"""Physical robot path shared by the live Monitor and report figures.

Vertices are measured ground-truth base positions only: a sample becomes a vertex when it starts a segment,
changes its reference classification or lies at least ``gate_m`` from the previous vertex, so the drawn
polyline stays within ``gate_m`` of every received sample. Segments break at invalid, late, reversed or
discontinuous samples. Coordinates are never smoothed, snapped or interpolated.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

SEGMENT_START = 1
IN_COLLISION = 2
OUTSIDE_REFERENCE = 4
BREAK_REASONS = ("invalid", "gap", "reversed", "jump")
MONITOR_LAYERS = ("explored", "navigation")


@dataclass(frozen=True, slots=True)
class PathSettings:
    gate_m: float = 0.05
    gap_seconds: float = 0.5
    jump_floor_m: float = 0.5
    jump_speed_factor: float = 2.0
    capacity: int = 200_000
    page: int = 5000

    def __post_init__(self) -> None:
        for name in ("gate_m", "gap_seconds", "jump_floor_m", "jump_speed_factor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.capacity < 1 or self.page < 1:
            raise ValueError("capacity and page must be positive")


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def base_pose(
    model: tuple[float, float, float], offset: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Compose a planar model pose in the world with the base pose expressed in the model frame."""
    x, y, yaw = model
    dx, dy, dyaw = offset
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * dx - s * dy, y + s * dx + c * dy, wrap(yaw + dyaw)


class ReferenceCells:
    """Classify world positions against the cells of a coverage reference."""

    def __init__(
        self,
        collision: np.ndarray[Any, Any],
        inclusion: np.ndarray[Any, Any],
        origin: tuple[float, float, float],
        resolution: float,
    ):
        self.collision = np.asarray(collision, dtype=bool)
        self.inclusion = np.asarray(inclusion, dtype=bool)
        if self.collision.ndim != 2 or self.collision.shape != self.inclusion.shape:
            raise ValueError("reference collision and inclusion grids disagree")
        x, y, yaw = (float(value) for value in origin)
        if not all(math.isfinite(value) for value in (x, y, yaw, resolution)) or resolution <= 0:
            raise ValueError("invalid reference origin or resolution")
        self.x, self.y, self.resolution = x, y, float(resolution)
        self.cos, self.sin = math.cos(yaw), math.sin(yaw)

    @classmethod
    def of(cls, reference: Any) -> "ReferenceCells":
        return cls(reference.collision, reference.inclusion, reference.origin, reference.resolution)

    def flags(self, x: float, y: float) -> int:
        dx, dy = x - self.x, y - self.y
        column = math.floor((self.cos * dx + self.sin * dy) / self.resolution)
        row = math.floor((-self.sin * dx + self.cos * dy) / self.resolution)
        height, width = self.collision.shape
        if not (0 <= row < height and 0 <= column < width):
            return OUTSIDE_REFERENCE
        if self.collision[row, column]:
            return IN_COLLISION
        return 0 if self.inclusion[row, column] else OUTSIDE_REFERENCE


class PhysicalPath:
    """Bounded, sequenced vertices of one attempt's physical path. Callers serialize access."""

    def __init__(
        self,
        maximum_speed: float,
        settings: PathSettings | None = None,
        reference: ReferenceCells | None = None,
    ):
        if not math.isfinite(maximum_speed) or maximum_speed <= 0:
            raise ValueError("maximum_speed must be a positive finite number")
        self.settings = settings or PathSettings()
        self.maximum_speed = float(maximum_speed)
        self.reference = reference
        capacity = self.settings.capacity
        self.x = np.empty(capacity, dtype=np.float64)
        self.y = np.empty(capacity, dtype=np.float64)
        self.flags = np.zeros(capacity, dtype=np.uint8)
        self.count = 0
        self.samples = 0
        self.collision_samples = 0
        self.outside_samples = 0
        self.breaks = dict.fromkeys(BREAK_REASONS, 0)
        self.pending: str | None = None
        self.previous: tuple[int, float, float] | None = None

    def add(self, stamp_ns: int, x: float, y: float) -> int | None:
        """Feed one received sample; return its sequence when it became a vertex."""
        self.samples += 1
        if not (math.isfinite(x) and math.isfinite(y)):
            self._interrupt("invalid")
            return None
        settings = self.settings
        if self.previous is not None:
            stamp, previous_x, previous_y = self.previous
            elapsed = (stamp_ns - stamp) / 1e9
            if elapsed < 0:
                self._interrupt("reversed")
            elif elapsed > settings.gap_seconds:
                self._interrupt("gap")
            elif math.hypot(x - previous_x, y - previous_y) > max(
                settings.jump_floor_m, settings.jump_speed_factor * self.maximum_speed * elapsed
            ):
                self._interrupt("jump")
        self.previous = stamp_ns, x, y
        cell = self.reference.flags(x, y) if self.reference is not None else 0
        self.collision_samples += bool(cell & IN_COLLISION)
        self.outside_samples += bool(cell & OUTSIDE_REFERENCE)
        start = self.pending is not None or self.count == 0
        if not start:
            last = (self.count - 1) % settings.capacity
            if (
                math.hypot(x - self.x[last], y - self.y[last]) < settings.gate_m
                and int(self.flags[last]) & ~SEGMENT_START == cell
            ):
                return None
        if self.pending is not None:
            self.breaks[self.pending] += 1
            self.pending = None
        index = self.count % settings.capacity
        self.x[index], self.y[index] = x, y
        self.flags[index] = cell | (SEGMENT_START if start else 0)
        self.count += 1
        return self.count

    def _interrupt(self, reason: str) -> None:
        if self.pending is None and self.count:
            self.pending = reason

    def tail(self) -> list[float] | None:
        """Latest sample when it continues the last vertex's segment without being a vertex."""
        if self.pending is not None or not self.count or self.previous is None:
            return None
        _, x, y = self.previous
        last = (self.count - 1) % self.settings.capacity
        return None if (x, y) == (self.x[last], self.y[last]) else [x, y]

    def page(self, after: int, limit: int | None = None) -> dict[str, Any]:
        """Vertices with sequence greater than ``after`` (oldest first) and the path counters."""
        capacity = self.settings.capacity
        limit = self.settings.page if limit is None else limit
        first = max(1, self.count - capacity + 1)
        start = max(after + 1, first)
        stop = min(self.count, start + limit - 1)
        points: list[list[float]] = []
        if stop >= start:
            indices = np.arange(start - 1, stop) % capacity
            points = [
                [x, y, flags]
                for x, y, flags in zip(
                    self.x[indices].tolist(),
                    self.y[indices].tolist(),
                    self.flags[indices].tolist(),
                    strict=True,
                )
            ]
        return dict(
            after=after,
            from_sequence=start,
            points=points,
            first_sequence=first,
            last_sequence=self.count,
            dropped=first - 1,
            samples=self.samples,
            breaks=dict(self.breaks),
            collision_samples=self.collision_samples,
            outside_samples=self.outside_samples,
            tail=self.tail(),
        )


def explored_pixels(
    target: np.ndarray[Any, Any], free: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Reference-shaped image: 255 where a target cell counts as free, 0 elsewhere."""
    rows, columns = np.nonzero(target)
    if free.shape != rows.shape:
        raise ValueError("free-cell mask does not match the reference target cells")
    pixels = np.zeros(target.shape, dtype=np.uint8)
    pixels[rows[free], columns[free]] = 255
    return pixels


def trim_reply(
    reply: Mapping[str, Any], after: int, revisions: Mapping[str, int]
) -> dict[str, Any]:
    """Adapt a cached worker reply to a requester's cursor and the layer revisions it holds."""
    result = dict(reply)
    if "from_sequence" in result:
        if result["after"] > after:
            result["points"] = []
            result["from_sequence"] = result["last_sequence"] + 1
        else:
            skip = max(0, after + 1 - result["from_sequence"])
            result["points"] = result["points"][skip:]
            result["from_sequence"] += skip
            result["after"] = after
    result["layers"] = {
        name: dict(layer, data=None) if layer.get("revision") == revisions.get(name) else layer
        for name, layer in result.get("layers", {}).items()
    }
    return result


@dataclass(slots=True)
class PathCursor:
    history: str | None = None
    applied: int = 0
    dropped: int = 0


def accept_page(
    cursor: PathCursor, reply: Mapping[str, Any]
) -> tuple[bool, list[tuple[float, float, int]]]:
    """Advance the cursor with a reply; return whether the history restarted and the new vertices.

    Vertices already applied are skipped, so duplicated, delayed and reordered replies change nothing. A
    page computed for a later cursor is ignored. A sequence gap means the worker evicted vertices before
    they were read: they are counted as dropped and the next vertex starts a segment.
    """
    history, after, start = reply.get("history"), reply.get("after"), reply.get("from_sequence")
    points = reply.get("points")
    if (
        not isinstance(history, str)
        or not isinstance(after, int)
        or not isinstance(start, int)
        or isinstance(after, bool)
        or isinstance(start, bool)
        or after < 0
        or start < 1
        or not isinstance(points, list)
    ):
        raise ValueError("Malformed monitor path page")
    reset = history != cursor.history
    if reset:
        cursor.history, cursor.applied, cursor.dropped = history, 0, 0
    if after > cursor.applied:
        return reset, []
    skip = max(0, cursor.applied + 1 - start)
    vertices: list[tuple[float, float, int]] = []
    for point in points[skip:]:
        if (
            not isinstance(point, list)
            or len(point) != 3
            or not all(isinstance(value, (int, float)) for value in point)
            or isinstance(point[2], bool)
            or not isinstance(point[2], int)
            or not all(math.isfinite(value) for value in point[:2])
        ):
            raise ValueError("Malformed monitor path vertex")
        vertices.append((float(point[0]), float(point[1]), point[2]))
    if not vertices:
        return reset, []
    start += skip
    if start > cursor.applied + 1:
        cursor.dropped += start - cursor.applied - 1
        x, y, flags = vertices[0]
        vertices[0] = (x, y, flags | SEGMENT_START)
    cursor.applied = start + len(vertices) - 1
    return reset, vertices
