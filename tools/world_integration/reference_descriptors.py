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

"""Official world descriptors and region raster of a coverage reference.

Runs worldgen's descriptor functions (`worldgen.descriptors.VERSION`, defined in docs/WORLDS.md "Descriptors")
unchanged on a coverage reference grid instead of the T0 raster. Free space F is the reference's free floor
`inclusion & ~collision` (the builder has already cut it to the spawn's 4-connected component). The robots are
`worldgen.descriptors.ROBOTS` (MOGI); adding robots is a descriptor-version decision, not an option here.

Writes two files into the output directory (format: docs/plans/worlds/EXECUTION.md "Shared contracts"):

- ``descriptors.json``: the descriptor values with ``grid.source: coverage_reference`` and the reference's
  ``metadata_sha256``, the label table of the region raster (``region_raster``) and ``regions_counted``;
- ``regions.png``: a 16-bit greyscale PNG with the reference grid's shape, row 0 the largest y (as the export's
  masks), 0 outside F and the grown regions 1 … n (T1's cores in ``scipy.ndimage.label`` order), each free cell
  assigned to its geodesically nearest core (`grow_regions`; a deliberate deviation from worldgen's
  watershed growth, recorded as ``region_growth``). A world without a region core gives an all-zero raster.

Run with the suite's development Python, which has worldgen's pinned numpy and scipy, one reference per process
(it is not meant for concurrent in-process use):

    .venv-suite/bin/python tools/world_integration/reference_descriptors.py \
        --reference exploration_benchmark_runner/config/coverage/corridor_v2.json --output DIR

Outside the images; it reads the reference and never writes next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

ROOT = Path(__file__).resolve().parents[2]
WORLDGEN_DIR = ROOT / "world_collection" / "worldgen"
sys.path.insert(0, str(WORLDGEN_DIR.parent))

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import scipy  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage, sparse  # noqa: E402
from scipy.sparse import csgraph  # noqa: E402

import worldgen  # noqa: E402
from worldgen import descriptors, raster  # noqa: E402
from worldgen.checks import Analysis, count_loops  # noqa: E402
from worldgen.world import BoolArray, Expected, Spawn, World  # noqa: E402

DESCRIPTION = "Official world descriptors and region raster of a coverage reference."
RESOLUTION_M = raster.RESOLUTION
# Cells of non-free border added around the reference grid (see `analysis`). Two, not one, so the 2 x 2 blocks
# of nav_complexity's 0.10 m grid for large worlds keep the reference grid's alignment.
PAD = 2
DESCRIPTORS_FILE = "descriptors.json"
REGIONS_FILE = "regions.png"
DEFAULT_SPAWN_ID = "Default"
FREE_SPACE = "inclusion & ~collision"
# worldgen sources whose content decides the values (WORLDGEN_VERSION alone misses a modified checkout)
WORLDGEN_SOURCES = ("descriptors.py", "raster.py", "checks.py", "rng.py")


class AdapterError(ValueError):
    """The reference or the inputs cannot give official descriptors."""


@dataclass(frozen=True, slots=True)
class CoverageReference:
    """A coverage reference: its metadata and its grids (row 0 = smallest y)."""

    path: Path
    identity: str
    metadata: dict[str, Any]
    metadata_sha256: str
    spawn: tuple[float, float]
    grid: raster.Grid
    collision: BoolArray
    inclusion: BoolArray


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_reference(path: Path) -> CoverageReference:
    """Read `<id>_v2.json` and its grid file.

    Checks: readable JSON with a string `id`, a finite `spawn` [x, y, …], `grid_file` whose SHA-256 equals
    `grid_sha256`, `collision` and `inclusion` arrays of shape (`height`, `width`), `resolution` 0.05 m and an
    `origin` [x, y, 0] (finite, yaw 0). It does not repeat the runner's target-cell and metric-version checks,
    which do not affect F.
    """
    try:
        metadata_bytes = path.read_bytes()
        metadata = json.loads(metadata_bytes)
        identity = metadata["id"]
        if not isinstance(identity, str) or not identity:
            raise AdapterError("id must be a non-empty string")
        spawn = (float(metadata["spawn"][0]), float(metadata["spawn"][1]))
        grid_bytes = (path.parent / str(metadata["grid_file"])).read_bytes()
        if _sha256(grid_bytes) != metadata["grid_sha256"]:
            raise AdapterError("grid hash mismatch")
        with np.load(io.BytesIO(grid_bytes), allow_pickle=False) as grid:
            collision = np.asarray(grid["collision"], dtype=bool)
            inclusion = np.asarray(grid["inclusion"], dtype=bool)
        resolution = float(metadata["resolution"])
        x0, y0, yaw = (float(v) for v in metadata["origin"])
        width, height = int(metadata["width"]), int(metadata["height"])
    except (OSError, KeyError, IndexError, TypeError, ValueError, BadZipFile) as exc:
        raise AdapterError(f"invalid coverage reference {path}: {exc}") from exc
    if not all(math.isfinite(v) for v in (*spawn, x0, y0)):
        raise AdapterError(f"{path}: spawn and origin must be finite")
    if not math.isclose(resolution, RESOLUTION_M, abs_tol=1e-12):
        raise AdapterError(
            f"{path}: resolution {resolution} m, the descriptors need {RESOLUTION_M} m"
        )
    if yaw != 0.0:
        raise AdapterError(f"{path}: origin yaw must be 0")
    if collision.shape != (height, width) or inclusion.shape != (height, width):
        raise AdapterError(f"{path}: grid shape disagrees with width and height")
    return CoverageReference(
        path=path,
        identity=identity,
        metadata=metadata,
        metadata_sha256=_sha256(metadata_bytes),
        spawn=spawn,
        grid=raster.Grid(x0, y0, width, height, RESOLUTION_M),
        collision=collision,
        inclusion=inclusion,
    )


def analysis(reference: CoverageReference) -> Analysis:
    """The T0 `Analysis` on the reference grid, padded by `PAD` non-free cells on every side.

    The construction follows `worldgen.checks.analyse` (worldgen 1.0.1, recorded in `software`) and must follow
    it when worldgen changes; `analyse` itself rasterises a `World`, so it cannot take a reference grid.

    The padding (collision true, inclusion false) closes the boundary where F reaches the grid edge (Warehouse):
    the distance transform does not count cells beyond the array as outside F, so clearance, skeleton, regions
    and F_rho would be overstated. It also makes `count_loops` see the boundary walls as not enclosed: they
    touch the padding, which lies outside the inclusion, as T0's inclusion polygon leaves the outer walls
    outside. Without it a reference whose inclusion covers the whole grid (Warehouse) counts its shell walls
    as loops. The padded ring is chosen over `binary_fill_holes(F)` (T0's no-polygon path) because it keeps the
    reference's own inclusion for every other world and fixes clearance and loops with one rule.
    """
    h = reference.grid.h
    collision = np.pad(reference.collision, PAD, constant_values=True)
    inclusion = np.pad(reference.inclusion, PAD, constant_values=False)
    grid = raster.Grid(
        round(reference.grid.x0 - PAD * h, 9),
        round(reference.grid.y0 - PAD * h, 9),
        reference.grid.nx + 2 * PAD,
        reference.grid.ny + 2 * PAD,
        h,
    )
    if (collision.size > descriptors.LARGE_GRID_CELLS) != (
        reference.collision.size > descriptors.LARGE_GRID_CELLS
    ):
        raise AdapterError(
            f"{reference.path}: the padding moves the grid across nav_complexity's large-grid size"
        )
    free = inclusion & ~collision
    if not free.any():
        raise AdapterError(f"{reference.path}: the reference has no free floor")
    _, components = ndimage.label(free)
    if components != 1:
        raise AdapterError(
            f"{reference.path}: the free floor has {components} 4-connected components, not one"
        )
    unpadded = free[PAD:-PAD, PAD:-PAD]
    border = bool(
        unpadded[0, :].any()
        or unpadded[-1, :].any()
        or unpadded[:, 0].any()
        or unpadded[:, -1].any()
    )
    dist = raster.clearance(free, h)
    skeleton = raster.skeleton(free, dist, h)
    if skeleton.any():
        masked = np.where(skeleton, dist, np.inf)
        index = np.unravel_index(int(np.argmin(masked)), masked.shape)
        cell: tuple[int, int] | None = (int(index[0]), int(index[1]))
        narrowest: float | None = round(2.0 * float(dist[index]), 4)
    else:
        cell, narrowest = None, None
    loops, unsized = count_loops(collision, inclusion, h)
    return Analysis(
        grid=grid,
        occupied=collision,
        inclusion=inclusion,
        inclusion_polygons=(),
        reachable=free,
        sealed=np.zeros_like(free),
        clearance=dist,
        skeleton=skeleton,
        narrowest_m=narrowest,
        narrowest_cell=cell,
        loops=loops,
        loops_unsized=unsized,
        border_touch=border,
    )


REGION_GROWTH = "geodesic-nearest-core"
REGION_CORE_MIN_M2 = 1.0  # maintainer decision 61 (docs/research/WORLD_COLLECTION.md §9)
# Per-label offset of the virtual source's edges: breaks exact distance ties towards the lower label. It must
# exceed float rounding of path sums (about 1e-12 m) and stay below the smallest gap between two different
# 8-connected path lengths at these distances (|m·√2 − n|·h, about 1e-5 m for paths of 10^3 cells).
TIE_EPSILON_M = 1e-9
MAX_TIE_LABELS = 5_000


def grow_regions(
    free: BoolArray, dist: npt.NDArray[np.float64], h: float
) -> tuple[int, np.ndarray[Any, np.dtype[np.int32]]]:
    """Regions with T1's cores and counting rule, grown by geodesic distance (`REGION_GROWTH`).

    Cores are `descriptors.regions`' cores: the components (`ndimage.label`, 4-connected, row-major order) of the
    free cells with clearance at least `descriptors.CORE_DEPTH_M`, of which only those of at least
    `REGION_CORE_MIN_M2` seed a region (decision 61). Every free cell joins the core with the
    shortest 8-connected path through F (steps h and √2·h); an exact tie goes to the lower label. The count is
    the number of grown regions of at least `descriptors.REGION_MIN_M2`.

    Deviation from worldgen: `descriptors.regions` grows the cores with `ndimage.watershed_ift` on the inverted
    clearance. Its near-wall band is a plateau of equal cost joined through doorways, so the first-queued
    marker floods the band of every room; the counts can be right while the ownership near walls is wrong.

    One multi-source Dijkstra (`scipy.sparse.csgraph`) from a virtual node with an edge of weight
    `label · TIE_EPSILON_M` to every core cell; each cell's label is the core at the end of its predecessor
    chain, found by pointer jumping. F must not touch the array edge (the caller pads it).
    """
    cores, n = ndimage.label(free & (dist >= descriptors.CORE_DEPTH_M))
    # Decision 61: only a core of at least REGION_CORE_MIN_M2 seeds a region; smaller cores (recess slivers)
    # are dropped and grown into like any other cell. The kept cores are renumbered in their label order.
    core_areas = np.bincount(cores.ravel(), minlength=n + 1) * h * h
    kept = core_areas >= REGION_CORE_MIN_M2
    kept[0] = False
    renumber = np.zeros(n + 1, dtype=np.int32)
    renumber[kept] = np.arange(1, int(kept.sum()) + 1, dtype=np.int32)
    cores = renumber[cores]
    n = int(kept.sum())
    if n == 0:
        return 0, np.zeros(free.shape, dtype=np.int32)
    if n > MAX_TIE_LABELS:
        raise AdapterError(f"{n} region cores: too many for the tie-break offset")
    if free[0, :].any() or free[-1, :].any() or free[:, 0].any() or free[:, -1].any():
        raise AdapterError("free space touches the array edge; pad it first")
    width = free.shape[1]
    flat_free = free.ravel()
    nodes = np.flatnonzero(flat_free)
    count_nodes = len(nodes)
    index = np.full(free.size, -1, dtype=np.int64)
    index[nodes] = np.arange(count_nodes, dtype=np.int64)
    rows, cols, weights = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        neighbour = nodes + dr * width + dc
        ok = flat_free[neighbour]
        a_side, b_side = index[nodes[ok]], index[neighbour[ok]]
        step = h * (math.sqrt(2.0) if dr and dc else 1.0)
        rows += [a_side, b_side]
        cols += [b_side, a_side]
        weights += [np.full(len(a_side), step)] * 2
    core_labels = cores.ravel()[nodes].astype(np.int64)
    core_nodes = np.flatnonzero(core_labels)
    rows.append(np.full(len(core_nodes), count_nodes, dtype=np.int64))
    cols.append(core_nodes)
    weights.append(core_labels[core_nodes] * TIE_EPSILON_M)
    graph = sparse.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(count_nodes + 1, count_nodes + 1),
    )
    del rows, cols, weights, index
    _, predecessors = csgraph.dijkstra(
        graph, directed=True, indices=count_nodes, return_predecessors=True
    )
    pointer = np.asarray(predecessors[:count_nodes], dtype=np.int64)
    if (pointer < 0).any():
        raise AdapterError("a free cell is not reachable from any region core")
    labels = core_labels.copy()
    pointer[core_nodes] = core_nodes  # a core cell's predecessor is the virtual node
    open_nodes = np.flatnonzero(labels == 0)
    while len(open_nodes):
        target = pointer[open_nodes]
        labels[open_nodes] = labels[target]
        pointer[open_nodes] = pointer[target]
        open_nodes = open_nodes[labels[open_nodes] == 0]
    grown = np.zeros(free.size, dtype=np.int32)
    grown[nodes] = labels
    grown_grid = grown.reshape(free.shape)
    areas = np.bincount(labels, minlength=n + 1)[1:] * h * h
    return int(np.count_nonzero(areas >= descriptors.REGION_MIN_M2)), grown_grid


def _check_spawns(spawns: tuple[Spawn, ...], a: Analysis) -> None:
    ids = [s.id for s in spawns]
    if not spawns or len(ids) != len(set(ids)):
        raise AdapterError("spawn ids must be present and unique")
    for s in spawns:
        if not (math.isfinite(s.x) and math.isfinite(s.y)):
            raise AdapterError(f"spawn {s.id} is not finite")
        r, c = a.grid.cell(s.x, s.y)
        if not (0 <= r < a.grid.ny and 0 <= c < a.grid.nx) or not a.reachable[r, c]:
            raise AdapterError(f"spawn {s.id} ({s.x}, {s.y}) is not on the reference's free floor")


def _finite(value: Any) -> Any:
    """Non-finite floats become null (unavailable values are null)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    return value


def _png16(labels: np.ndarray[Any, np.dtype[np.int32]]) -> bytes:
    """A 16-bit greyscale PNG, row 0 the largest y."""
    if int(labels.max()) > np.iinfo(np.uint16).max or int(labels.min()) < 0:
        raise AdapterError("region labels do not fit a 16-bit image")
    buffer = io.BytesIO()
    image = Image.fromarray(np.ascontiguousarray(np.flipud(labels).astype(np.uint16)))
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def compute(
    reference: CoverageReference, spawns: tuple[Spawn, ...]
) -> tuple[dict[str, Any], bytes]:
    """The descriptor document and the region PNG of a reference."""
    a = analysis(reference)
    _check_spawns(spawns, a)
    # `descriptors.compute` reads only `spawns` from the world; the other fields are placeholders.
    world = World(
        id=reference.identity,
        slug="coverage_reference",
        title=reference.identity,
        tier="",
        role="",
        primitives=(),
        spawns=spawns,
        expected=Expected(free_area_m2=a.free_area_m2, narrowest_passage_m=0.0),
    )
    values = descriptors.compute(world, a)
    h = a.grid.h
    # `compute` counts regions with worldgen's watershed growth; the official raster and count use the
    # geodesic growth (`grow_regions`), and worldgen's count is kept beside it for comparison with previews.
    count, padded = grow_regions(a.reachable, a.clearance, h)
    values["regions_worldgen_watershed"] = values["regions"]
    values["regions"] = count
    values["region_growth"] = REGION_GROWTH
    values["region_core_min_m2"] = REGION_CORE_MIN_M2
    if padded[:PAD].any() or padded[-PAD:].any() or padded[:, :PAD].any() or padded[:, -PAD:].any():
        raise AdapterError("the region raster labels the padding")
    labels = np.ascontiguousarray(padded[PAD:-PAD, PAD:-PAD])
    free = reference.inclusion & ~reference.collision
    has_cores = bool(labels.max() > 0)
    if (labels[~free] != 0).any() or (has_cores and (labels[free] <= 0).any()):
        raise AdapterError("the region raster breaks the label contract (0 outside F, 1 … n in F)")
    cells = np.bincount(labels[free].ravel(), minlength=int(labels.max()) + 1)
    table = []
    for label in range(1, len(cells)):
        area = float(cells[label]) * h * h
        table.append(
            {
                "label": label,
                "cells": int(cells[label]),
                "area_m2": round(area, 4),
                "counted": bool(area >= descriptors.REGION_MIN_M2),
            }
        )
    counted = [row["label"] for row in table if row["counted"]]
    if len(counted) != count:
        raise AdapterError("counted region labels disagree with the descriptor count")
    png = _png16(labels)
    values["grid"] = {
        "source": "coverage_reference",
        "reference_id": reference.identity,
        "metadata_sha256": reference.metadata_sha256,
        "grid_sha256": str(reference.metadata["grid_sha256"]),
        "resolution_m": h,
        "seed": descriptors.SEED,
        "free_space": FREE_SPACE,
        "padding_cells": PAD,
        "free_floor_touches_grid_edge": a.border_touch,
    }
    values["spawns"] = [{"id": s.id, "x": s.x, "y": s.y} for s in spawns]
    values["region_raster"] = {
        "file": REGIONS_FILE,
        "sha256": _sha256(png),
        "encoding": "16-bit greyscale PNG; row 0 = largest y; 0 outside F; 1 … n grown regions",
        "labels": table,
    }
    values["regions_counted"] = counted
    values["software"] = {
        "worldgen": worldgen.WORLDGEN_VERSION,
        "worldgen_sha256": {
            name: _sha256((WORLDGEN_DIR / name).read_bytes()) for name in WORLDGEN_SOURCES
        },
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "adapter_sha256": _sha256(Path(__file__).read_bytes()),
    }
    return _finite(values), png


def write(output: Path, document: dict[str, Any], png: bytes) -> None:
    """Write both files without overwriting: each is written to a temporary file and linked into place, so a
    failure leaves neither a partial file nor a lone PNG."""
    text = (json.dumps(document, indent=1, allow_nan=False, ensure_ascii=False) + "\n").encode()
    output.mkdir(parents=True, exist_ok=True)
    placed: list[Path] = []
    try:
        for name, data in ((REGIONS_FILE, png), (DESCRIPTORS_FILE, text)):
            with tempfile.NamedTemporaryFile(
                dir=output, prefix=f".{name}.", delete=False
            ) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            try:
                os.link(temporary, output / name)  # fails if the file exists
            except FileExistsError as exc:
                raise AdapterError(f"refusing to overwrite {output / name}") from exc
            finally:
                temporary.unlink()
            placed.append(output / name)
    except BaseException:
        for path in placed:
            path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--reference", type=Path, required=True, help="<id>_v2.json of the reference"
    )
    parser.add_argument("--output", type=Path, required=True, help="directory for the two files")
    parser.add_argument(
        "--spawn",
        nargs=3,
        action="append",
        metavar=("ID", "X", "Y"),
        help=f"a spawn for spawn_visible_fraction (repeat; default: the reference's spawn as "
        f"{DEFAULT_SPAWN_ID!r})",
    )
    args = parser.parse_args(argv)
    try:
        reference = load_reference(args.reference)
        if args.spawn:
            spawns = tuple(Spawn(i, float(x), float(y), 0.0) for i, x, y in args.spawn)
        else:
            spawns = (Spawn(DEFAULT_SPAWN_ID, *reference.spawn, 0.0),)
        document, png = compute(reference, spawns)
        write(args.output, document, png)
    except (AdapterError, ValueError) as exc:
        parser.error(str(exc))
    regions = document["regions"]
    note = " (no region core of clearance >= 0.8 m: all-zero raster)" if not regions else ""
    print(
        f"{reference.identity}: {document['area_m2']} m², {regions} regions "
        f"(labels {len(document['region_raster']['labels'])}){note} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
