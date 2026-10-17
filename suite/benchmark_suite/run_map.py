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

"""Coverage and path map of one recorded run: the report's map data and the Results run map.

The attempt report's map panel and the Results run map share these functions, so both show the same explored
mask and the same ground-truth vertices as the live Monitor (docs/SUITE_ARCHITECTURE.md, "Results run map").
"""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import uuid

import numpy as np

from exploration_benchmark_runner.coverage import Calculator, Reference

from .monitoring import (
    PhysicalPath,
    ReferenceCells,
    base_pose,
    explored_pixels,
    png,
    reference_pixels,
)

GROUND_TRUTH_OFFSET = ("model_to_base_x", "model_to_base_y", "model_to_base_yaw")
GROUND_TRUTH_COLUMNS = ["clock_ns", "valid", "x", "y", "yaw", "speed", *GROUND_TRUTH_OFFSET]


def explored_cells(
    reference: Any, directory: Path, manifest: Mapping[str, Any]
) -> tuple[np.ndarray[Any, Any] | None, str | None]:
    """Reference target cells the coverage metric counts as free in the final map."""
    alignment = ((manifest.get("coverage") or {}).get("alignment") or {}).get("world_from_map")
    if not alignment:
        return None, "Explored cells unavailable: no coverage alignment recorded"
    path = directory / "final_map.npz"
    if not path.exists():
        return None, "Explored cells unavailable: final map not recorded"
    try:
        with np.load(path, allow_pickle=False) as archive:
            occupancy = archive["occupancy"]
            metadata = json.loads(str(archive["metadata"]))
        x, y, z = metadata["origin"]
        qx, qy, qz, qw = metadata["orientation"]
        grid = SimpleNamespace(
            header=SimpleNamespace(frame_id=str(metadata["frame"])),
            info=SimpleNamespace(
                width=occupancy.shape[1],
                height=occupancy.shape[0],
                resolution=metadata["resolution"],
                origin=SimpleNamespace(
                    position=SimpleNamespace(x=x, y=y, z=z),
                    orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
                ),
            ),
            data=occupancy.ravel(),
        )
        free: np.ndarray[Any, Any] = Calculator(reference, alignment).free_cells(grid)
    except KeyError as exc:
        return None, f"Explored cells unavailable: final map has no {exc}"
    except (TypeError, ValueError) as exc:
        return None, f"Explored cells unavailable: {exc}"
    return free, None


@dataclass(frozen=True, slots=True)
class GroundTruthPath:
    reference: Any | None
    path: PhysicalPath | None
    speeds: tuple[float, ...]
    notes: tuple[str, ...]


@lru_cache(maxsize=4)
def cached_reference(metadata_file: str, metadata_sha256: str) -> Any:
    return Reference(metadata_file)


def load_reference(descriptor: Mapping[str, Any] | None) -> tuple[Any | None, str | None]:
    """The attempt's coverage reference when its files still match the recorded hashes."""
    if not descriptor:
        return None, "World reference not recorded"
    try:
        reference = cached_reference(
            str(descriptor["metadata_file"]), str(descriptor["metadata_sha256"])
        )
        matches = (
            reference.metadata_hash == descriptor["metadata_sha256"]
            and reference.metadata["grid_sha256"] == descriptor["grid_sha256"]
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return None, f"World reference unavailable: {exc}"
    if not matches:
        return None, "World reference changed since the attempt; not drawn"
    return reference, None


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def triple(values: Any) -> tuple[float, float, float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return None
    x, y, z = (finite(value) for value in values)
    return None if x is None or y is None or z is None else (x, y, z)


def ground_truth_path(
    rows: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> GroundTruthPath:
    """Report path built exactly like the live Monitor's: recorded rows in arrival order."""
    reference, note = load_reference((manifest.get("coverage") or {}).get("reference"))
    notes = [note] if note else []
    maximum = finite((manifest.get("robot") or {}).get("max_linear_speed"))
    if maximum is None or maximum <= 0:
        notes.append("Robot speed limit unavailable; ground-truth path not drawn")
        return GroundTruthPath(reference, None, (), tuple(notes))
    path = PhysicalPath(
        maximum, reference=ReferenceCells.of(reference) if reference is not None else None
    )
    fallback = triple((manifest.get("ground_truth_alignment") or {}).get("model_to_base"))
    speeds: list[float] = []
    for row in rows:
        offset = triple([row.get(key) for key in GROUND_TRUTH_OFFSET]) or fallback
        if offset is None:
            continue
        model = triple([row.get(key) for key in ("x", "y", "yaw")]) if row.get("valid") else None
        x, y = base_pose(model, offset)[:2] if model is not None else (math.nan, math.nan)
        if path.add(row["clock_ns"], x, y) is not None:
            speed = finite(row.get("speed"))
            speeds.append(math.nan if speed is None else speed)
    if not path.samples:
        notes.append(
            "No ground-truth samples with a model-to-base offset "
            "(diagnostics disabled or alignment missing)"
        )
        return GroundTruthPath(reference, None, (), tuple(notes))
    dropped = max(0, path.count - path.settings.capacity)
    if dropped:
        notes.append(f"Oldest {dropped:,} vertices not drawn (history bound)")
    return GroundTruthPath(reference, path, tuple(speeds[dropped:]), tuple(notes))


# Results run map: at most this many vertices per reply (about 0.8 MiB), and this many cached maps.
PAGE_VERTICES = 20_000
CACHED_MAPS = 8


def build(payload: dict[str, Any]) -> dict[str, Any]:
    """Map data of one finished attempt from its recorded files; runs in a spawned job."""
    import duckdb

    from .datasets import materialize_manifest
    from .datasets import observations as recorded
    from .storage import inside

    identity = str(payload["attempt_id"])
    directory = inside(Path(payload["workspace"]) / "attempts", identity)
    manifest = materialize_manifest(directory)
    cacheable = True
    try:
        rows = recorded(directory, "ground_truth", GROUND_TRUTH_COLUMNS).to_pylist()
        physical = ground_truth_path(rows, manifest)
    except (duckdb.Error, OSError) as exc:
        # A read error may be transient (compaction moves segments): report it once, never cache it.
        reference, note = load_reference((manifest.get("coverage") or {}).get("reference"))
        unreadable = f"Ground-truth stream unreadable: {exc}"
        physical = GroundTruthPath(reference, None, (), (*filter(None, [note]), unreadable))
        cacheable = False
    notes = list(physical.notes)
    reference = physical.reference
    base = explored = None
    if reference is not None:
        base = dict(
            world_id=(manifest.get("identity") or {}).get("world_id"),
            image_png=png(reference_pixels(reference.collision, reference.target)),
            origin=list(reference.origin),
            resolution=reference.resolution,
            width=reference.metadata["width"],
            height=reference.metadata["height"],
            reference_id=reference.identity,
            metadata_sha256=reference.metadata_hash,
        )
        free, note = explored_cells(reference, directory, manifest)
        if free is None:
            notes.append(note or "Explored cells unavailable")
        else:
            cells = int(free.sum())
            explored = dict(
                image_png=png(explored_pixels(reference.target, free)),
                origin=list(reference.origin),
                resolution=reference.resolution,
                cells=cells,
                reference_cells=int(reference.count),
                share_percent=100 * cells / reference.count,
            )
    path = physical.path
    page = path.page(0, path.settings.capacity) if path is not None else None
    points = np.asarray(
        [point[:2] for point in page["points"]] if page else [], dtype=float
    ).reshape(-1, 2)
    return dict(
        attempt_id=identity,
        x=points[:, 0].copy(),
        y=points[:, 1].copy(),
        flags=np.asarray([point[2] for point in page["points"]] if page else [], dtype=np.uint8),
        first_sequence=page["first_sequence"] if page else 1,
        samples=page["samples"] if page else 0,
        breaks=page["breaks"] if page else {},
        collision_samples=page["collision_samples"] if page else 0,
        outside_samples=page["outside_samples"] if page else 0,
        tail=page["tail"] if page else None,
        path_available=path is not None,
        notes=notes,
        reference=dict(id=reference.identity, metadata_sha256=reference.metadata_hash)
        if reference is not None
        else None,
        base=base,
        explored=explored,
        cacheable=cacheable,
    )


def remember(cache: OrderedDict[str, Any], identity: str, value: Any) -> None:
    """Store ``value`` as the newest entry and evict the oldest beyond ``CACHED_MAPS``."""
    cache[identity] = value
    cache.move_to_end(identity)
    while len(cache) > CACHED_MAPS:
        cache.popitem(last=False)


class RunMaps:
    """Engine-side run maps: built one at a time in a spawned job, cached, and served in pages.

    The engine loop only looks up the cache and slices pages; recorded files are read in the job. The
    latest request wins, so fast clicking through runs never queues a backlog.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.ready: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.once: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.failures: OrderedDict[str, str] = OrderedDict()
        self.wanted: str | None = None
        self.building: str | None = None
        self.job: Any = None

    def get(self, identity: str, after: int) -> dict[str, Any]:
        if identity in self.ready:
            self.ready.move_to_end(identity)
            return self.page(self.ready[identity], after)
        if identity in self.once:
            return self.page(self.once.pop(identity), after)
        if identity in self.failures:
            return dict(attempt_id=identity, state="failed", reason=self.failures.pop(identity))
        # The latest request wins: asking again for the map being built cancels a queued one.
        self.wanted = None if identity == self.building else identity
        return dict(attempt_id=identity, state="building")

    @staticmethod
    def page(entry: dict[str, Any], after: int) -> dict[str, Any]:
        """Vertices after ``after`` with the counters, as ``PhysicalPath.page`` and ``monitor.get``."""
        first, count = entry["first_sequence"], entry["first_sequence"] + len(entry["x"]) - 1
        start = max(after + 1, first)
        stop = min(count, start + PAGE_VERTICES - 1)
        points: list[list[Any]] = []
        if stop >= start:
            window = slice(start - first, stop - first + 1)
            points = [
                [x, y, flags]
                for x, y, flags in zip(
                    entry["x"][window].tolist(),
                    entry["y"][window].tolist(),
                    entry["flags"][window].tolist(),
                    strict=True,
                )
            ]
        reply = dict(
            attempt_id=entry["attempt_id"],
            state="ready",
            history=entry["history"],
            after=after,
            from_sequence=start,
            points=points,
            first_sequence=first,
            last_sequence=max(count, 0),
            dropped=first - 1,
            samples=entry["samples"],
            breaks=entry["breaks"],
            collision_samples=entry["collision_samples"],
            outside_samples=entry["outside_samples"],
            tail=entry["tail"],
            path_available=entry["path_available"],
            notes=entry["notes"],
            reference=entry["reference"],
        )
        if after == 0:
            reply.update(base=entry["base"], explored=entry["explored"])
        return reply

    def poll(self) -> None:
        if self.job is not None:
            result = self.job.poll()
            if result is None:
                return
            self.job.close()
            identity, self.job, self.building = self.building, None, None
            assert identity is not None
            if result["ok"]:
                entry = result["result"]
                entry["history"] = f"{identity}:{uuid.uuid4().hex}"
                remember(self.ready if entry.pop("cacheable") else self.once, identity, entry)
            else:
                remember(self.failures, identity, str(result["detail"]))
        if self.job is None and self.wanted is not None:
            identity, self.wanted = self.wanted, None
            if identity in self.ready:
                return
            from .jobs import Job

            try:
                self.job = Job(
                    "run_map",
                    {"workspace": str(self.workspace), "attempt_id": identity},
                    uuid.uuid4().hex,
                )
                self.building = identity
            except Exception as exc:
                remember(self.failures, identity, f"Map job could not start: {exc}")

    def close(self) -> None:
        if self.job is not None:
            self.job.close()
            self.job = None
