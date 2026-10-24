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

"""The export's tables and raw streams beyond the run records, read by the extended figures (FIGURE_EXTENSION FX0).

``data.load`` reads what the statistics need (runs, coverage, the ground-truth trajectory). The figures of the
extension also read the goal, gate and decision tables, the accounting, the final maps and the attempts' raw
streams (``raw/<attempt>/<stream>.parquet``). Every read goes through a ``Reads`` record, so each figure's sidecar
hashes exactly the files it used (FIGURE_STYLE §1). A table or stream an export does not hold reads as ``None``
(an older export). A run whose explorer wrote log format 1 has no rows in the gate table and no decision timings,
while the table itself exists: test the run's group, not ``None``; ``log_formats`` tells which runs.
"""

from dataclasses import dataclass, field
import io
import json
import math
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .data import TIME_EPSILON
from .export import Export

Row = dict[str, Any]


@dataclass(slots=True)
class Reads:
    """The export files one output read (relative paths, as listed in ``SHA256SUMS``)."""

    export: Export
    files: set[str] = field(default_factory=set)

    def table(self, name: str, columns: list[str] | None = None) -> pa.Table | None:
        """A top-level table, or ``None`` when the export holds none of that name."""
        path = f"{name}.parquet"
        if path not in self.export.files:
            return None
        self.files.add(path)
        return pq.read_table(self.export.directory / path, columns=columns)

    def rows(self, name: str, columns: list[str] | None = None) -> list[Row]:
        """A top-level table's rows, none when the export lacks the table."""
        table = self.table(name, columns)
        return [] if table is None else table.to_pylist()

    def raw(
        self, attempt_id: str, stream: str, columns: list[str] | None = None
    ) -> pa.Table | None:
        """One attempt's raw stream, or ``None`` when the export holds no such file."""
        path = f"raw/{attempt_id}/{stream}.parquet"
        if path not in self.export.files:
            return None
        self.files.add(path)
        return pq.read_table(self.export.directory / path, columns=columns)

    def file(self, path: str) -> bytes | None:
        """Any other listed file (a map's PGM or YAML, a reference's files), or ``None``."""
        if path not in self.export.files:
            return None
        self.files.add(path)
        return (self.export.directory / path).read_bytes()


def log_formats(reads: Reads) -> dict[str, int]:
    """Each run's explorer log format, from its decision rows or else its goal rows (MEASUREMENTS "Explorer log
    formats"); a run without either has none. Format 1 records no gate rows, cancel reasons or timings."""
    formats: dict[str, int] = {}
    for name in ("goals", "explorer_decisions"):
        for row in reads.rows(name, ["attempt_id", "log_format"]):
            if row["log_format"] is not None:
                formats[str(row["attempt_id"])] = int(row["log_format"])
    return formats


@dataclass(frozen=True, slots=True)
class ReferenceCells:
    """A coverage reference's target cells in reference order (the order of ``explored_cells`` masks):
    ``rows``/``columns`` index a grid whose row 0 is the smallest y; cell centres in metres follow from
    ``origin`` and ``resolution``."""

    rows: npt.NDArray[np.intp]
    columns: npt.NDArray[np.intp]
    shape: tuple[int, int]
    origin: tuple[float, float]
    resolution: float

    def centres(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        x0, y0 = self.origin
        x = x0 + (self.columns.astype(np.float64) + 0.5) * self.resolution
        y = y0 + (self.rows.astype(np.float64) + 0.5) * self.resolution
        return x, y


def reference_cells(reads: Reads, name: str) -> ReferenceCells | None:
    """The target cells of ``references/<name>/`` (``target.pgm``, row 0 the largest y, flipped here); None
    when the reference is not exported, its origin is rotated (the cell centres here assume yaw 0, as every
    reference so far has) or ``target.pgm`` holds another number of cells than ``target_cells``: callers say the
    reference is not readable and count it."""
    meta_bytes, target = (
        reads.file(f"references/{name}/reference.yaml"),
        reads.file(f"references/{name}/target.pgm"),
    )
    if meta_bytes is None or target is None:
        return None
    meta = yaml.safe_load(meta_bytes)
    if len(meta["origin"]) > 2 and float(meta["origin"][2]) != 0.0:
        return None
    grid = np.array(Image.open(io.BytesIO(target))) == 255
    rows, columns = np.nonzero(grid[::-1])
    if rows.size != int(meta["target_cells"]):
        return None
    origin = (float(meta["origin"][0]), float(meta["origin"][1]))
    return ReferenceCells(rows, columns, grid.shape, origin, float(meta["resolution"]))


Mask = npt.NDArray[np.bool_]
# an explored-cell snapshot: (simulation_elapsed_ns, None for a pre-motion one; mask in reference order)
Snapshot = tuple[int | None, Mask]


@dataclass(frozen=True, slots=True)
class SnapshotChain:
    """An attempt's decoded explored-cell snapshots in clock order with their clocks (``clock_ns``), the rows
    the stream holds, and the rows not decoded: ``skipped`` (a broken chain: a change row whose decoded count
    differs from ``free_cells``, a change row without a preceding keyframe, and the change rows after it until
    the next keyframe) and ``invalid`` (rows marked ``valid = false``, never decoded)."""

    snapshots: tuple[Snapshot, ...]
    clocks: tuple[int, ...]
    rows: int
    skipped: int
    invalid: int


def explored_chain(reads: Reads, attempt_id: str) -> SnapshotChain | None:
    """An attempt's explored-cell snapshots (the worker's coverage numerator over the reference's target cells,
    every 5 simulated s and at the end), decoded, with what was not: None without the stream. Keyframes hold
    the mask, change rows the XOR with the previous snapshot. A decoded mask whose count of free cells differs
    from the row's ``free_cells`` (a change row lost after it was queued) breaks the chain: it and the following
    change rows are skipped until the next keyframe, as is a change row without a preceding keyframe. A row
    marked invalid is not decoded and breaks the chain the same way."""
    table = reads.raw(
        attempt_id,
        "explored_cells",
        [
            "clock_ns",
            "simulation_elapsed_ns",
            "kind",
            "reference_cells",
            "free_cells",
            "mask",
            "valid",
        ],
    )
    if table is None:
        return None
    found: list[Snapshot] = []
    clocks: list[int] = []
    previous: npt.NDArray[np.bool_] | None = None
    skipped = invalid = 0
    rows = sorted(table.to_pylist(), key=lambda r: r["clock_ns"])
    for row in rows:
        if row["valid"] is False:
            invalid += 1
            previous = None
            continue
        bits = np.unpackbits(np.frombuffer(row["mask"], dtype=np.uint8))[: row["reference_cells"]]
        data = bits.astype(bool)
        if row["kind"] == "keyframe":
            mask = data
        elif previous is not None and previous.size == data.size:
            mask = previous ^ data
        else:
            skipped += 1
            continue
        if int(np.count_nonzero(mask)) != row["free_cells"]:
            previous = None
            skipped += 1
            continue
        previous = mask
        found.append((row["simulation_elapsed_ns"], mask))
        clocks.append(int(row["clock_ns"]))
    return SnapshotChain(tuple(found), tuple(clocks), len(rows), skipped, invalid)


def explored_snapshots(reads: Reads, attempt_id: str) -> list[Snapshot]:
    """An attempt's decoded explored-cell snapshots (``explored_chain``), in clock order; none without the
    stream."""
    chain = explored_chain(reads, attempt_id)
    return [] if chain is None else list(chain.snapshots)


Triple = tuple[float, float, float]


def finite_triple(values: Any) -> Triple | None:
    """Three finite numbers (booleans are not numbers), as written in YAML or JSON, or None."""
    if not isinstance(values, list) or len(values) != 3:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return None
    numbers = (float(values[0]), float(values[1]), float(values[2]))
    return numbers if all(math.isfinite(v) for v in numbers) else None


def world_from_map(reads: Reads, attempt_id: str) -> Triple | None:
    """The attempt's frozen map-to-world alignment (x, y, yaw), the one its coverage used: ``world_from_map`` of
    its exported final map (``maps/<attempt>.yaml``), else (no map, or a map without a usable key) of its raw
    manifest (``coverage.alignment``), the source the export copies it from; None without either. Goals and
    insight rows are in the map frame, the references and the ground truth in the world frame."""
    found = reads.file(f"maps/{attempt_id}.yaml")
    if found is not None:
        alignment = finite_triple((yaml.safe_load(found) or {}).get("world_from_map"))
        if alignment is not None:
            return alignment
    manifest = reads.file(f"raw/{attempt_id}/manifest.json")
    if manifest is None:
        return None
    coverage = json.loads(manifest).get("coverage") or {}
    return finite_triple((coverage.get("alignment") or {}).get("world_from_map"))


def to_world(alignment: Triple, x: float, y: float) -> tuple[float, float]:
    """A map-frame point in the world frame."""
    wx, wy, yaw = alignment
    c, s = math.cos(yaw), math.sin(yaw)
    return wx + c * x - s * y, wy + s * x + c * y


GROUND_TRUTH_COLUMNS = [
    "clock_ns",
    "simulation_elapsed_ns",
    "valid",
    "x",
    "y",
    "yaw",
    "model_to_base_x",
    "model_to_base_y",
    "model_to_base_yaw",
]


def model_to_base(reads: Reads, attempt: str) -> Triple | None:
    """The manifest's model-to-base offset, used for ground-truth rows that do not carry their own."""
    raw = reads.file(f"raw/{attempt}/manifest.json")
    if raw is None:
        return None
    return finite_triple((json.loads(raw).get("ground_truth_alignment") or {}).get("model_to_base"))


@dataclass(frozen=True, slots=True)
class BasePose:
    """One valid ground-truth row's robot base position in the world frame: its simulation clock, its time on the
    movement clock (None before movement onset) and x, y."""

    clock_ns: int
    t_s: float | None
    x: float
    y: float


def base_poses(
    reads: Reads, attempt: str, end_s: float | None, *, pre_motion: bool = False
) -> list[BasePose] | None:
    """A run's base positions from the 20 Hz raw ``ground_truth`` rows from movement onset to ``end_s`` (no end
    with None; with ``pre_motion``, the rows before onset too), in clock order: the Gazebo model pose composed
    with the row's model-to-base offset (else the manifest's), as the run record's ground-truth path is; invalid
    rows and rows without a pose or an offset are left out. None without the stream (F17, ``regions``)."""
    table = reads.raw(attempt, "ground_truth", GROUND_TRUTH_COLUMNS)
    if table is None:
        return None
    rows = sorted(
        (
            r
            for r in table.to_pylist()
            if (
                end_s is None or r["simulation_elapsed_ns"] / 1e9 <= end_s + TIME_EPSILON
                if r["simulation_elapsed_ns"] is not None
                else pre_motion
            )
            and r["valid"]
            and r["clock_ns"] is not None
        ),
        key=lambda r: r["clock_ns"],
    )
    fallback: Triple | None = None
    found: list[BasePose] = []
    for row in rows:
        model = finite_triple([row["x"], row["y"], row["yaw"]])
        offset = finite_triple([row[k] for k in GROUND_TRUTH_COLUMNS[-3:]])
        if offset is None:
            fallback = fallback or model_to_base(reads, attempt)
            offset = fallback
        if model is None or offset is None:
            continue
        # the base pose: the model pose composed with the offset (to_world turns it by the model's yaw)
        base_x, base_y = to_world(model, offset[0], offset[1])
        elapsed = row["simulation_elapsed_ns"]
        found.append(
            BasePose(
                int(row["clock_ns"]), None if elapsed is None else elapsed / 1e9, base_x, base_y
            )
        )
    return found
