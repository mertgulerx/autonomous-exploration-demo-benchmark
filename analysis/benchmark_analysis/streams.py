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


def explored_snapshots(
    reads: Reads, attempt_id: str
) -> list[tuple[int | None, npt.NDArray[np.bool_]]]:
    """An attempt's explored-cell snapshots (the worker's coverage numerator over the reference's target cells,
    every 5 simulated s and at the end), decoded: (``simulation_elapsed_ns``, None for the pre-motion one; mask in
    reference order). Keyframes hold the mask, change rows the XOR with the previous snapshot. A decoded mask
    whose count of free cells differs from the row's ``free_cells`` (a change row lost after it was queued) breaks
    the chain: it and the following change rows are skipped until the next keyframe, as is a change row without
    a preceding keyframe."""
    table = reads.raw(
        attempt_id,
        "explored_cells",
        ["clock_ns", "simulation_elapsed_ns", "kind", "reference_cells", "free_cells", "mask"],
    )
    if table is None:
        return []
    found: list[tuple[int | None, npt.NDArray[np.bool_]]] = []
    previous: npt.NDArray[np.bool_] | None = None
    for row in sorted(table.to_pylist(), key=lambda r: r["clock_ns"]):
        bits = np.unpackbits(np.frombuffer(row["mask"], dtype=np.uint8))[: row["reference_cells"]]
        data = bits.astype(bool)
        if row["kind"] == "keyframe":
            mask = data
        elif previous is not None and previous.size == data.size:
            mask = previous ^ data
        else:
            continue
        if int(np.count_nonzero(mask)) != row["free_cells"]:
            previous = None
            continue
        previous = mask
        found.append((row["simulation_elapsed_ns"], mask))
    return found


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
