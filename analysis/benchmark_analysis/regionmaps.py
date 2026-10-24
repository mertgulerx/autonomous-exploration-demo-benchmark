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

"""Region displays of ``region-measures-v1`` (plans/worlds/I8 §3.6; definitions in ``regions``): figure F27 and
table T12, and the region boundaries the atlas F12 draws (I8 §3.2).

- F27 region map, one figure per stratum (``figures.stratum_variants``: ``F27_<label slug>``, the label led by the
  world group when the study has several): one map per profile (Default first, at most three in a row) on the
  stratum's coverage reference at one scale, the T1 regions outlined. A counted region with more than
  ``FEW_RUNS`` runs with a value is filled with the Kaplan–Meier median of its T90_j on the shared ``cividis``
  ``Normalize(0, L)`` of FP-10, L the whole study's longest limit (``figures.path_norm``), or with the hatched
  "not reached" grey when the median is not reached by L; with at most ``FEW_RUNS`` runs it is not filled and a
  strip under its number shows each run's value (a cell per run, hatched grey when not reached by L). Under each
  map a key gives per counted region "k/n reached", T90_j (the individual values, or the median with its
  quartiles) and the median of the entries E_j, and the runs without a value by reason.
- T12 region table: per stratum, profile and counted region the runs, reached k/n, T90_j (individual values up to
  ``FEW_RUNS`` runs, else the Kaplan–Meier median and quartiles, RD-10), completed runs not reached, the median
  entries, the recoveries placed in the region and the runs with ``snapshots_incomplete``; recoveries outside
  every counted region or unplaced get rows of their own when there are any. Grouped like T3–T7
  (``tables.Sources.with_groups``).

Both are diagnostics (decision 56): never tested or ranked, with many regions × strata × profiles and no
multiplicity control; T90_j means mapped, not visited, at the 5 s snapshot resolution; E_j and K_j cover windows
that differ by outcome. Censoring follows the milestone rule (``regions``): a run not reaching 90 % of a region by L
is censored at L, also when it completed earlier. An export without region rasters (export version 1) writes
neither output.
"""

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
import math
import statistics
from typing import Any

from matplotlib.colors import ListedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import numpy.typing as npt
from scipy import ndimage

from . import stats
from .data import REPORTED, Stratum
from .export import Export
from .figures import (
    PROVENANCE,
    REFERENCE_PAD_M,
    TIME_LABEL,
    Extent,
    FigureContext,
    FigureOut,
    MapCode,
    draw_reference,
    export_reference_name,
    finish,
    group_inputs,
    load_reference,
    message_figure,
    panel_title,
    path_norm,
    reference_files,
    stratum_variants,
)
from .regions import REGION_MEASURES, RegionTime, RunRegions, run_regions, summary
from .streams import Reads
from .style import EDGE, MM_PER_INCH, PAGE_MAX_MM, PRESETS, SEQUENTIAL_CMAP, figure, page_note
from .tables import DASH, Sources, Table
from .worlds import Regions, descriptors, regions

FEW_RUNS = 3  # up to this many runs with a value: individual values, never a fill or a median
NOT_REACHED = "#BDBDBD"  # the fixed "not reached by L" grey, always hatched (never a numeric fill)
NOT_REACHED_HATCH = "\\\\\\\\"
# region boundaries: thin dashed lines between cells of different regions (walls are drawn by the map)
BOUNDARY_WIDTH = 0.5
BOUNDARY_DASHES = (0.0, (2.0, 1.2))
DIAGNOSTIC = (
    f"diagnostic ({REGION_MEASURES}, decision 56): not in stats.json, never tested, ranked or cited as an "
    "outcome"
)

# F27 layout in millimetres
GAP_MM, ROW_GAP_MM, TITLE_MM, KEY_GAP_MM = 4.0, 3.0, 6.0, 1.5
BAND_MM, FOOT_MM = 16.0, 6.0  # under the maps: colour bar and legend; the evidence label
BAR_H_MM = 2.5
MIN_SCALE = 0.5  # mm per metre: below it the 5 m bar is 2.5 mm and the regions no longer read
MAX_COLUMNS = 3
CELL_MM, CELL_GAP_MM = 1.8, 0.4  # a strip cell (one run) under a region's number
NUMBER_PT = 7.0


# Shared: region boundaries, the stratum's reference and measures ---------------------------------------------


def has_regions(export: Export) -> bool:
    """The export holds a region raster for some coverage reference (export version 2)."""
    return any(f.startswith("references/") and f.endswith("/regions.png") for f in export.files)


def _runs(edges: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.intp], ...]:
    """The runs of True along axis 0 of ``edges`` per column: (column, first row, row after the last)."""
    padded = np.zeros((edges.shape[0] + 2, edges.shape[1]), dtype=np.int8)
    padded[1:-1] = edges
    steps = np.diff(padded, axis=0).T  # column-major, so nonzero lists each column's runs in order
    start_columns, starts = np.nonzero(steps == 1)
    _, ends = np.nonzero(steps == -1)
    return start_columns, starts, ends


def boundary_segments(raster: Regions) -> list[npt.NDArray[np.float64]]:
    """The edges between 4-adjacent cells of two different regions (both labelled), merged along rows and
    columns into straight segments in metres; walls and the outside are not boundaries."""
    labels, (x0, y0), res = raster.labels, raster.origin, raster.resolution
    found: list[npt.NDArray[np.float64]] = []
    # vertical edges between columns c and c + 1, merged over consecutive rows
    vertical = (labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] > 0) & (labels[:, 1:] > 0)
    for column, first, after in zip(*_runs(vertical), strict=True):
        x = x0 + (column + 1) * res
        found.append(np.array([[x, y0 + first * res], [x, y0 + after * res]]))
    # horizontal edges between rows r and r + 1, merged over consecutive columns
    horizontal = (labels[:-1] != labels[1:]) & (labels[:-1] > 0) & (labels[1:] > 0)
    for row, first, after in zip(*_runs(horizontal.T), strict=True):
        y = y0 + (row + 1) * res
        found.append(np.array([[x0 + first * res, y], [x0 + after * res, y]]))
    return found


def draw_boundaries(ax: Any, raster: Regions, *, zorder: float = 2.5) -> None:
    """The region boundaries (``boundary_segments``) as thin dashed dark lines above fills and obstacle
    outlines."""
    segments = boundary_segments(raster)
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors=EDGE,
                linewidths=BOUNDARY_WIDTH,
                linestyles=[BOUNDARY_DASHES],
                capstyle="butt",
                zorder=zorder,
            )
        )


def boundary_handle() -> Line2D:
    return Line2D(
        [],
        [],
        color=EDGE,
        linewidth=BOUNDARY_WIDTH,
        linestyle=BOUNDARY_DASHES,
        label="Region boundary (T1)",
    )


@dataclass(frozen=True, slots=True)
class StratumRegions:
    """A stratum's region measures: its coverage reference (that of its first reported run with one) and
    raster, each profile's measures (study order, reported runs on that reference), the reported runs on
    another reference or without one (left out, counted), the files read and why nothing was measured
    (``reason``: "observed", "no_reference", "no_region_raster")."""

    reference: str | None
    raster: Regions | None
    measures: tuple[tuple[str, tuple[RunRegions, ...]], ...]
    other_reference: int
    no_reference: int
    reason: str
    inputs: frozenset[str] = frozenset()
    region_areas: dict[int, float] = field(default_factory=dict)

    @property
    def counted(self) -> list[int]:
        return [] if self.raster is None else sorted(self.raster.counted)

    @property
    def attempts(self) -> set[str]:
        return {m.attempt_id for _, runs in self.measures for m in runs}


def collect(export: Export, stratum: Stratum, order: Sequence[str]) -> StratumRegions:
    """The stratum's ``region-measures-v1`` per profile in ``order`` (``regions.run_regions`` per reported
    run, each reading its own files)."""
    by_reference = [
        (run, export_reference_name(export, str(run["attempt_id"]))) for run in stratum.reported
    ]
    reference = next((name for _, name in by_reference if name is not None), None)
    no_reference = sum(name is None for _, name in by_reference)
    other = sum(name is not None and name != reference for _, name in by_reference)
    files = {PROVENANCE} if PROVENANCE in export.files else set()
    if reference is None:
        return StratumRegions(None, None, (), other, no_reference, "no_reference", frozenset(files))
    reads = Reads(export)
    raster = regions(reads, reference)
    found = descriptors(reads, reference) or {}
    areas = {
        int(entry["label"]): float(entry["area_m2"])
        for entry in (found.get("region_raster") or {}).get("labels") or []
        if entry.get("area_m2") is not None
    }
    files |= reads.files
    if raster is None:
        return StratumRegions(
            reference, None, (), other, no_reference, "no_region_raster", frozenset(files)
        )
    chosen = [run for run, name in by_reference if name == reference]
    measures = []
    for profile in order:
        mine = tuple(
            run_regions(export, run, reference, stratum.limit_s)
            for run in chosen
            if run["profile_id"] == profile
        )
        if mine:
            measures.append((profile, mine))
            files |= {name for m in mine for name in m.inputs}
    return StratumRegions(
        reference,
        raster,
        tuple(measures),
        other,
        no_reference,
        "observed",
        frozenset(files),
        areas,
    )


@dataclass(frozen=True, slots=True)
class RegionSummary:
    """One counted region of one profile: ``times`` the runs' T90_j with a value (events first, by value;
    censored after), ``median`` and ``quartiles`` the Kaplan–Meier values (None when not reached; only above
    ``FEW_RUNS`` runs with a value), the median entries over the runs with entries, the recoveries placed in it
    (None when no run has K_j) and the runs without T90_j by reason."""

    region: int
    times: tuple[RegionTime, ...]
    completed_not_reached: int
    median: float | None
    quartiles: tuple[float | None, float | None]
    entries_median: float | None
    entries_runs: int
    recoveries: int | None
    no_value: dict[str, int]

    @property
    def n(self) -> int:
        return len(self.times)

    @property
    def reached(self) -> int:
        return sum(t.event is True for t in self.times)

    @property
    def few(self) -> bool:
        return self.n <= FEW_RUNS


def region_summary(measures: Sequence[RunRegions], region: int) -> RegionSummary:
    """A region's T90_j summary (individual values or Kaplan–Meier, RD-10), entries and recoveries."""
    times, no_value, completed_not_reached = [], Counter[str](), 0
    for m in measures:
        t = m.times.get(region)
        if t is None:
            no_value[m.reason] += 1
        elif t.event is None or t.value is None:
            no_value[t.reason] += 1
        else:
            times.append(t)
            completed_not_reached += t.event is False and m.outcome == "completed"
    ordered = tuple(sorted(times, key=lambda t: (t.event is False, t.value)))
    median: float | None = None
    quartiles: tuple[float | None, float | None] = (None, None)
    if len(ordered) > FEW_RUNS:
        curve = stats.kaplan_meier(
            [t.value or 0.0 for t in ordered], [bool(t.event) for t in ordered]
        )
        median = stats.km_quantile(curve, 0.5)
        quartiles = (stats.km_quantile(curve, 0.25), stats.km_quantile(curve, 0.75))
    entries = [
        m.entries.counts.get(region, 0)
        for m in measures
        if m.entries is not None and m.entries.counts is not None
    ]
    placed = [
        m.recoveries.counts.get(region, 0)
        for m in measures
        if m.recoveries is not None and m.recoveries.counts is not None
    ]
    return RegionSummary(
        region,
        ordered,
        completed_not_reached,
        median,
        quartiles,
        float(statistics.median(entries)) if entries else None,
        len(entries),
        sum(placed) if placed else None,
        dict(sorted(no_value.items())),
    )


def _seconds(value: float | None) -> str:
    """A region time at the snapshot resolution (5 s): whole seconds, "> L" when not reached by L."""
    return "> L" if value is None else f"{value:.0f}"


def time_text(region: RegionSummary) -> str:
    """T90_j as printed: the individual values (``FEW_RUNS`` or fewer), else the median [Q1, Q3]."""
    if not region.times:
        return DASH
    if region.few:
        return ", ".join(_seconds(t.value if t.event else None) for t in region.times)
    if region.median is None:
        return "not reached"
    return f"{_seconds(region.median)} [{_seconds(region.quartiles[0])}, {_seconds(region.quartiles[1])}]"


def _number(value: float | None) -> str:
    return DASH if value is None else f"{value:g}"


def _null_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{reason.replace('_', ' ')} {n}" for reason, n in counts.items())


def run_nulls(measures: Sequence[RunRegions]) -> dict[str, dict[str, int]]:
    """Runs without a value by measure and reason: T90_j (all regions alike: run reason or the first region's
    reason), entries and recoveries."""
    found: dict[str, Counter[str]] = {
        "t90": Counter(),
        "entries": Counter(),
        "recoveries": Counter(),
    }
    for m in measures:
        first = next(iter(m.times.values()), None)
        if m.reason != "observed":
            found["t90"][m.reason] += 1
        elif first is not None and first.reason in ("snapshots_incomplete", "coverage_ineligible"):
            found["t90"][first.reason] += 1
        if m.entries is not None and m.entries.counts is None:
            found["entries"][m.entries.reason] += 1
        if m.recoveries is not None and m.recoveries.counts is None:
            found["recoveries"][m.recoveries.reason] += 1
    return {key: dict(sorted(counts.items())) for key, counts in found.items()}


def limits_text() -> list[str]:
    """The limits every region output states (I8 §3.5, MEASUREMENTS "Region measures")."""
    return [
        f"Region measures {REGION_MEASURES} are diagnostics (decision 56): never tested or ranked, with many "
        "regions, strata and profiles and no multiplicity control, so differences between profiles are not "
        "findings. Regions are T1's (regions of at least 4.0 m² counted; boundaries at geodesic midpoints, "
        "roughly in openings narrower than 1.6 m), not room semantics.",
        "T90_j: the first explored-cell snapshot at or before L in which at least 90 % of the region's reference "
        "target cells are explored, stamped with its own time since first movement; snapshots every 5 s, so the "
        "resolution is 5 s and values are printed in whole seconds. It means mapped, not visited: a region can "
        "reach 90 % from the next room. Not reached by L: censored at L, also for a run that completed earlier "
        "(event-free through L, METHODOLOGY §4.5); a run without movement never reaches it. A run whose snapshot "
        "chain is incomplete (missing, broken, a gap above 5.5 s or none in the last 5.5 s of its window) has no value for any region "
        "(snapshots_incomplete), counted and never shown as late.",
        "Entries E_j: stays of at least 1.0 s reaching 0.25 m inside the region, from the 20 Hz ground truth "
        "between first movement and min(T_end, L); recoveries K_j: the recovery rows placed at the ground-truth "
        "pose within 50 ms (dataset schema 11). Both cover windows that differ by outcome (a completed run is "
        "counted over a shorter window), so outcomes are given beside them and differences are not behaviour "
        "effects. Simulation only.",
    ]


# F27 region map -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MapLayout:
    scale: float  # mm per metre
    map_mm: tuple[float, float]
    column_mm: float
    columns: int
    rows: int
    key_mm: float
    total_mm: float
    floored: bool


def _key_line_mm(columns: int) -> tuple[float, float, float]:
    """The key's font size (8 pt, 7 pt in three columns), its line height and an upper bound on its character
    width, in mm."""
    if columns >= MAX_COLUMNS:
        return 7.0, 3.0, 1.5
    return 8.0, 3.4, 1.7


def _layout(
    width_mm: float, size_m: tuple[float, float], profiles: int, key_lines: int
) -> MapLayout:
    """Profile maps in rows of up to three at one scale, each with its title above and its key below, shrunk
    until the figure fits the page, but not below ``MIN_SCALE``."""
    columns = max(1, min(MAX_COLUMNS, profiles))
    rows = max(1, math.ceil(profiles / columns))
    column = (width_mm - (columns - 1) * GAP_MM) / columns
    _, line, _ = _key_line_mm(columns)
    key = KEY_GAP_MM + key_lines * line
    fixed = rows * (TITLE_MM + key + ROW_GAP_MM) + BAND_MM + FOOT_MM
    fitted = min(column / size_m[0], (PAGE_MAX_MM - fixed) / (rows * size_m[1]))
    scale = max(fitted, MIN_SCALE)
    map_mm = (size_m[0] * scale, size_m[1] * scale)
    return MapLayout(
        scale, map_mm, column, columns, rows, key, fixed + rows * map_mm[1], fitted < MIN_SCALE
    )


def label_points(raster: Regions) -> dict[int, tuple[float, float]]:
    """Each counted region's label point: its cell farthest from another region and from the floor's edge
    (the first such cell in row order on a tie), in metres."""
    labels = raster.labels
    same = labels > 0
    for axis in (0, 1):
        for shift in (1, -1):
            same &= np.roll(labels, shift, axis=axis) == labels
    same[[0, -1], :] = False
    same[:, [0, -1]] = False
    depth = ndimage.distance_transform_edt(same)
    counted = sorted(raster.counted)
    positions = ndimage.maximum_position(depth, labels, index=counted)
    x0, y0 = raster.origin
    return {
        j: (x0 + (c + 0.5) * raster.resolution, y0 + (r + 0.5) * raster.resolution)
        for j, (r, c) in zip(counted, positions, strict=True)
    }


def _obstacles(ax: Any, codes: npt.NDArray[Any], extent: Extent) -> None:
    """Obstacles outlined above the fills (cividis's middle is close to the obstacle grey, FIGURE_STYLE §6)."""
    resolution = (extent[1] - extent[0]) / codes.shape[1]
    ax.contour(
        extent[0] + resolution * (np.arange(codes.shape[1]) + 0.5),
        extent[2] + resolution * (np.arange(codes.shape[0]) + 0.5),
        codes == MapCode.OCCUPIED,
        levels=[0.5],
        colors=EDGE,
        linewidths=0.4,
        zorder=2,
    )


def _fills(
    ax: Any, raster: Regions, found: Sequence[RegionSummary], norm: Normalize, extent: Extent
) -> None:
    """Regions with more than ``FEW_RUNS`` values: the median on cividis, or hatched grey when not reached."""
    labels = raster.labels
    medians = np.full(labels.shape, np.nan)
    missing = np.zeros(labels.shape, dtype=bool)
    for region in found:
        if region.few:
            continue
        cells = labels == region.region
        if region.median is None:
            missing |= cells
        else:
            medians[cells] = region.median
    common = dict(
        origin="lower", extent=extent, interpolation="nearest", interpolation_stage="rgba", zorder=1
    )
    if np.isfinite(medians).any():
        ax.imshow(medians, cmap=SEQUENTIAL_CMAP, norm=norm, **common)
    if missing.any():
        ax.imshow(
            np.where(missing, 1.0, np.nan),
            cmap=ListedColormap([NOT_REACHED]),
            vmin=0,
            vmax=2,
            **common,
        )
        resolution = raster.resolution
        ax.contourf(
            extent[0] + resolution * (np.arange(labels.shape[1]) + 0.5),
            extent[2] + resolution * (np.arange(labels.shape[0]) + 0.5),
            missing,
            levels=[0.5, 1.5],
            colors="none",
            hatches=[NOT_REACHED_HATCH],
            zorder=1.2,
        )


def _cell(
    fig: Figure, ax: Any, anchor: tuple[float, float], left_mm: float, top_mm: float, color: Any
) -> None:
    """One strip cell, ``left_mm``/``top_mm`` from the region's label point; hatched grey without a colour."""
    size = CELL_MM / MM_PER_INCH
    rect = Rectangle(
        (left_mm / MM_PER_INCH, top_mm / MM_PER_INCH - size),
        size,
        size,
        transform=fig.dpi_scale_trans + ScaledTranslation(*anchor, ax.transData),
        facecolor=NOT_REACHED if color is None else color,
        hatch=NOT_REACHED_HATCH if color is None else None,
        edgecolor=EDGE,
        linewidth=0.4,
        clip_on=False,
        zorder=5,
    )
    ax.add_artist(rect)  # not add_patch: the cell is in inches and must not move the data limits


def _labels(
    fig: Figure,
    ax: Any,
    points: dict[int, tuple[float, float]],
    found: Sequence[RegionSummary],
    norm: Normalize,
) -> None:
    """Each counted region's number on a white box at its label point, with the strip of its runs' values
    under it when it has ``FEW_RUNS`` or fewer."""
    cmap = plt.get_cmap(SEQUENTIAL_CMAP)
    box = dict(facecolor="#FFFFFF", edgecolor="none", pad=0.6)
    for region in found:
        anchor = points[region.region]
        strip = 0 < region.n and region.few
        ax.text(
            *anchor,
            str(region.region),
            fontsize=NUMBER_PT,
            ha="center",
            va="bottom" if strip else "center",
            bbox=box,
            clip_on=False,
            zorder=5,
        )
        if not strip:
            continue
        width = region.n * CELL_MM + (region.n - 1) * CELL_GAP_MM
        for index, t in enumerate(region.times):
            color = cmap(norm(t.value)) if t.event and t.value is not None else None
            _cell(fig, ax, anchor, -width / 2 + index * (CELL_MM + CELL_GAP_MM), -0.6, color)


def _key(
    fig: Figure,
    box: Any,
    left: float,
    top: float,
    column_mm: float,
    found: Sequence[RegionSummary],
    nulls: list[str],
    columns: int,
) -> None:
    """The key under one profile's map: a header, a line per counted region, then the runs without a value."""
    size, line, char = _key_line_mm(columns)
    places = (0.0, 7 * char, 15 * char)  # Region | Reached | T90_j; median entries right-aligned
    header = ("Region", "Reached", "T90,j (s)")
    lines: list[tuple[str, str, str, str]] = [(*header, "Median E")]
    lines += [
        (str(r.region), f"{r.reached}/{r.n}", time_text(r), _number(r.entries_median))
        for r in found
    ]
    for index, cells in enumerate(lines):
        y = top - (index + 1) * line
        colour = "#555555" if index == 0 else "#000000"
        for place, text in zip(places, cells[:3], strict=True):
            x, yy, _, _ = box(left + place, y, 0.0, 0.0)
            fig.text(x, yy, text, fontsize=size, ha="left", va="bottom", color=colour)
        x, yy, _, _ = box(left + column_mm, y, 0.0, 0.0)
        fig.text(x, yy, cells[3], fontsize=size, ha="right", va="bottom", color=colour)
    for index, text in enumerate(nulls):
        x, yy, _, _ = box(left, top - (len(lines) + index + 1) * line, 0.0, 0.0)
        fig.text(x, yy, text, fontsize=size, ha="left", va="bottom", color="#555555")


def _null_lines(measures: Sequence[RunRegions]) -> list[str]:
    nulls = run_nulls(measures)
    found = []
    if nulls["t90"]:
        found.append(f"No T90,j: {_null_text(nulls['t90'])}")
    if nulls["entries"]:
        found.append(f"No E: {_null_text(nulls['entries'])}")
    return found


def _legend(fig: Figure, box: Any, width_mm: float, norm: Normalize) -> None:
    """The colour bar (left) and the key to fills, strips and boundaries (right) in the band under the maps."""
    bar_width = width_mm * 0.55
    cax = fig.add_axes(box(2.0, BAND_MM + FOOT_MM - 5.0, bar_width - 4.0, BAR_H_MM))
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(SEQUENTIAL_CMAP))
    fig.colorbar(mappable, cax=cax, orientation="horizontal").set_label(
        f"T90,j: {TIME_LABEL[0].lower()}{TIME_LABEL[1:]}"
    )
    handles = [
        Patch(
            facecolor=NOT_REACHED,
            hatch=NOT_REACHED_HATCH,
            edgecolor=EDGE,
            linewidth=0.4,
            label="Median (fill) or run (cell) not reached by L",
        ),
        Patch(
            facecolor="#FFFFFF",
            edgecolor=EDGE,
            linewidth=0.4,
            label=f"n ≤ {FEW_RUNS}: no fill, a cell per run under the number",
        ),
        boundary_handle(),
    ]
    x, y, _, _ = box(bar_width + 2.0, BAND_MM + FOOT_MM - 1.0, 0.0, 0.0)
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x, y),
        ncols=1,
        fontsize=7,
        handlelength=1.6,
        handletextpad=0.4,
        borderaxespad=0.0,
    )


def _outcomes(measures: Sequence[RunRegions]) -> str:
    counts = Counter(m.outcome for m in measures)
    return ", ".join(f"{outcome.replace('_', ' ')} {n}" for outcome, n in sorted(counts.items()))


# T12's short outcome codes (its notes spell them out)
OUTCOME_CODES = {"completed": "c", "time_limit": "tl", "no_movement": "nm"}


def _outcome_codes(measures: Sequence[RunRegions]) -> str:
    counts = Counter(m.outcome for m in measures)
    return " ".join(f"{OUTCOME_CODES.get(o, o)} {n}" for o, n in sorted(counts.items()))


def _notes(
    ctx: FigureContext,
    stratum: Stratum,
    data: StratumRegions,
    found: dict[str, list[RegionSummary]],
    layout: MapLayout,
    norm: Normalize,
) -> list[str]:
    label = ctx.label(stratum)
    notes = [
        f"{label}, L = {stratum.limit_s:g} s of simulated time since first movement; all reported runs per "
        f"profile on the coverage reference {data.reference} (time limit and no movement included). Every map "
        f"at {layout.scale:.4g} mm per metre (5 m bar), cropped to the reference grid plus "
        f"{REFERENCE_PAD_M:g} m; map greys as FIGURE_STYLE §6, obstacles outlined, region boundaries dashed, "
        "counted regions numbered (the raster's labels, as in T12).",
        f"Fill: with more than {FEW_RUNS} runs with a value, the Kaplan–Meier median of T90_j (midpoint rule, "
        f"RD-10) on cividis over [0, {norm.vmax:g}] s, the whole study's longest L, the same in every F27; a "
        "median not reached by L hatched grey, never a numeric fill. With "
        f"{FEW_RUNS} or fewer, no fill: a cell per run under the region's number (sorted, not reached last). "
        "Key: k/n reached of the runs with a value, T90_j (the individual values, or the median [Q1, Q3]; "
        "> L not reached), the median entries E_j over the runs with entries.",
        *limits_text(),
    ]
    outcomes = "; ".join(
        f"{ctx.styles[profile].name}: {_outcomes(runs)}" for profile, runs in data.measures
    )
    notes.append(f"Outcomes (the windows of E_j differ by them): {outcomes}.")
    late = [
        f"{ctx.styles[profile].name} region {r.region} ({r.completed_not_reached})"
        for profile, regions_found in found.items()
        for r in regions_found
        if r.completed_not_reached
    ]
    if late:
        notes.append(f"Completed runs not reaching 90 % of a region by L: {', '.join(late)}.")
    for profile, runs in data.measures:
        nulls = run_nulls(runs)
        parts = [
            f"{key} {_null_text(counts)}"
            for key, counts in (("T90_j", nulls["t90"]), ("E_j", nulls["entries"]))
            if counts
        ]
        if parts:
            notes.append(f"{ctx.styles[profile].name}, runs without a value: {'; '.join(parts)}.")
    left_out = []
    for profile in ctx.profiles(stratum):
        counts = {k: v for k, v in stratum.counts(profile).items() if k != REPORTED}
        if counts:
            text = ", ".join(f"{k} {v}" for k, v in counts.items())
            left_out.append(f"{ctx.styles[profile].name}: {text}")
    if left_out:
        notes.append(f"Final runs not reported (not drawn, counted): {'; '.join(left_out)}.")
    if data.other_reference or data.no_reference:
        notes.append(
            f"Reported runs left out, counted: {data.other_reference} on another coverage reference, "
            f"{data.no_reference} without an exported reference."
        )
    uncounted = (
        0
        if data.raster is None
        else len(set(np.unique(data.raster.labels)) - {0}) - len(data.counted)
    )
    if uncounted:
        notes.append(
            f"{uncounted} region(s) under 4.0 m² are outlined but not counted, numbered or measured."
        )
    tall = page_note(layout.total_mm) if layout.floored else None
    if tall is not None:
        notes.append(tall)
    return notes


def _figure(ctx: FigureContext, stratum: Stratum, name: str, variant_label: str) -> FigureOut:
    """One stratum's region map, named and labelled by ``figures.stratum_variants``."""
    export = ctx.study.export
    data = collect(export, stratum, ctx.order)
    attempts = {str(r["attempt_id"]) for r in stratum.reported}
    label = ctx.label(stratum)
    inputs = set(data.inputs) | reference_files(ctx, data.reference)
    loaded = None
    first = next((m for _, runs in data.measures for m in runs), None)
    if first is not None:
        loaded = load_reference(ctx, first.attempt_id)
    shape_ok = (
        loaded is not None
        and data.raster is not None
        and loaded[0].shape == data.raster.labels.shape
    )
    message = {
        "no_reference": "no exported coverage reference",
        "no_region_raster": "no region raster for its coverage reference",
    }.get(data.reason)
    if message is None and not data.measures:
        message = "no reported run on the coverage reference"
    elif message is None and not data.counted:
        message = "no counted region (at least 4.0 m²) on its coverage reference"
    elif message is None and not shape_ok:
        message = "coverage reference not readable or its region raster has another shape"
    if message is not None or loaded is None or data.raster is None:
        built = message_figure(
            ctx, name, f"{label}: {message}", attempts=attempts, inputs=inputs, wrap=60
        )
        built.extra.update(stratum=variant_label, measure=REGION_MEASURES)
        return built
    codes, extent = loaded
    raster = data.raster
    norm = path_norm(ctx)
    found = {
        profile: [region_summary(runs, j) for j in data.counted] for profile, runs in data.measures
    }
    nulls = {profile: _null_lines(runs) for profile, runs in data.measures}
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    size_m = (
        extent[1] - extent[0] + 2 * REFERENCE_PAD_M,
        extent[3] - extent[2] + 2 * REFERENCE_PAD_M,
    )
    key_lines = 1 + len(data.counted) + max(len(n) for n in nulls.values())
    layout = _layout(width_mm, size_m, len(data.measures), key_lines)
    total = layout.total_mm
    # The panels are placed in millimetres (as F12 and F17); the empty grid axis keeps the layout engine quiet.
    fig, axes = figure(ctx.preset, "double", aspect=total / width_mm)
    axes[0][0].set_axis_off()

    def box(left: float, bottom: float, w: float, h: float) -> tuple[float, float, float, float]:
        return (left / width_mm, bottom / total, w / width_mm, h / total)

    points = label_points(raster)
    row_mm = TITLE_MM + layout.map_mm[1] + layout.key_mm + ROW_GAP_MM
    for index, (profile, runs) in enumerate(data.measures):
        row, column = divmod(index, layout.columns)
        left = column * (layout.column_mm + GAP_MM)
        top = total - row * row_mm
        bottom = top - TITLE_MM - layout.map_mm[1]
        inset = (layout.column_mm - layout.map_mm[0]) / 2
        ax = fig.add_axes(box(left + inset, bottom, *layout.map_mm))
        ax.set_axis_off()
        draw_reference(ax, codes, extent)
        _fills(ax, raster, found[profile], norm, extent)
        _obstacles(ax, codes, extent)
        draw_boundaries(ax, raster)
        _labels(fig, ax, points, found[profile], norm)
        panel_title(ax, index, f"{ctx.styles[profile].name} · n = {len(runs)}")
        _key(
            fig,
            box,
            left,
            bottom - KEY_GAP_MM,
            layout.column_mm,
            found[profile],
            nulls[profile],
            layout.columns,
        )
    _legend(fig, box, width_mm, norm)
    finish(fig, ctx, descriptive=True)
    return FigureOut(
        name,
        fig,
        "double",
        data.attempts,
        f"Region map for {label} (diagnostic, {REGION_MEASURES}): per profile, the T1 regions of the coverage "
        "reference coloured by the median time to 90 % mapped (or each run's time when there are at most three "
        "runs), with runs reached, times and median entries per region below each map.",
        _notes(ctx, stratum, data, found, layout, norm),
        dict(
            stratum=variant_label,
            measure=REGION_MEASURES,
            robot_id=stratum.robot_id,
            world_id=stratum.world_id,
            spawn_id=stratum.spawn_id,
            comparison_key=stratum.comparison_key,
            limit_s=stratum.limit_s,
            reference=data.reference,
            mm_per_m=round(layout.scale, 4),
            time_norm=[0.0, float(norm.vmax or 0.0)],
            few_runs=FEW_RUNS,
            display=DIAGNOSTIC,
            regions_counted=data.counted,
            region_areas_m2={str(j): a for j, a in sorted(data.region_areas.items())},
            left_out=dict(other_reference=data.other_reference, no_reference=data.no_reference),
            profiles={
                profile: dict(
                    summary(runs)[profile],
                    regions_shown=[_region_document(r) for r in found[profile]],
                    runs=[m.document() for m in runs],
                )
                for profile, runs in data.measures
            },
        ),
        inputs=inputs,
    )


def _region_document(region: RegionSummary) -> dict[str, Any]:
    return dict(
        region=region.region,
        n=region.n,
        reached=region.reached,
        values_s=[t.value for t in region.times] if region.few else None,
        events=[t.event for t in region.times] if region.few else None,
        km_median_s=region.median,
        km_quartiles_s=list(region.quartiles) if not region.few else None,
        completed_not_reached=region.completed_not_reached,
        entries_median=region.entries_median,
        entries_runs=region.entries_runs,
        recoveries=region.recoveries,
        no_value=region.no_value,
    )


def figures(ctx: FigureContext) -> Iterator[FigureOut]:
    """One F27 per stratum (``figures.stratum_variants``), each built when the caller asks for it; none when the
    export holds no region raster (export version 1)."""
    if not has_regions(ctx.study.export):
        return
    for stratum, name, label in stratum_variants(ctx, "F27"):
        built = _figure(ctx, stratum, name, label)
        built.inputs |= group_inputs(ctx, stratum)
        yield built


# T12 region table ----------------------------------------------------------------------------------------------

T12_COLUMNS = [
    "Stratum",
    "Profile",
    "Region",
    "Area (m²)",
    "Runs",
    "Outcomes",
    "Reached",
    "T90,j (s)",
    "Completed, not reached",
    "Entries, median",
    "Recoveries",
    "Snapshots incomplete",
]


def t12(src: Sources) -> Table:
    table = Table(
        "T12",
        f"Region measures ({REGION_MEASURES}, diagnostics): region coverage time, entries and recoveries",
        list(T12_COLUMNS),
    )
    export = src.study.export
    documents: dict[str, Any] = {}
    files: set[str] = set()
    without: list[str] = []
    extra_notes: list[str] = []
    for stratum in src.study.strata:
        data = collect(export, stratum, src.order)
        files |= data.inputs
        label = src.label(stratum)
        if data.reason != "observed" or not data.counted:
            reason = data.reason if data.reason != "observed" else "no_counted_regions"
            without.append(f"{label} ({reason.replace('_', ' ')})")
            continue
        documents[label] = dict(
            reference=data.reference,
            regions_counted=data.counted,
            left_out=dict(other_reference=data.other_reference, no_reference=data.no_reference),
            profiles={profile: summary(runs)[profile] for profile, runs in data.measures},
        )
        for profile, runs in data.measures:
            nulls = run_nulls(runs)
            incomplete = str(nulls["t90"].get("snapshots_incomplete", 0))
            outcomes = _outcome_codes(runs)
            for j in data.counted:
                region = region_summary(runs, j)
                table.rows.append(
                    [
                        label,
                        src.name(profile),
                        str(j),
                        _number(
                            None if j not in data.region_areas else round(data.region_areas[j], 1)
                        ),
                        str(len(runs)),
                        outcomes,
                        f"{region.reached}/{region.n}",
                        time_text(region),
                        str(region.completed_not_reached),
                        _number(region.entries_median),
                        _number(region.recoveries),
                        incomplete,
                    ]
                )
            placed = [
                m.recoveries
                for m in runs
                if m.recoveries is not None and m.recoveries.counts is not None
            ]
            for name, count in (
                ("outside counted regions", sum(k.outside for k in placed)),
                ("unplaced", sum(k.unplaced for k in placed)),
            ):
                if count:
                    table.rows.append(
                        [
                            label,
                            src.name(profile),
                            name,
                            DASH,
                            str(len(runs)),
                            outcomes,
                            DASH,
                            DASH,
                            DASH,
                            DASH,
                            str(count),
                            incomplete,
                        ]
                    )
            other = {k: v for k, v in nulls["t90"].items() if k != "snapshots_incomplete"}
            parts = [
                f"{key} {_null_text(counts)}"
                for key, counts in (
                    ("T90_j", other),
                    ("E_j", nulls["entries"]),
                    ("K_j", {k: v for k, v in nulls["recoveries"].items() if k != "not_recorded"}),
                )
                if counts
            ]
            mismatch = sum(1 for m in runs if m.recoveries is not None and m.recoveries.mismatch)
            if mismatch:
                parts.append(
                    f"recovery rows not matching the run's recoveries in {mismatch} run(s)"
                )
            if parts:
                extra_notes.append(f"{label}, {src.name(profile)}: {'; '.join(parts)}.")
            table.attempt_ids |= {m.attempt_id for m in runs}
    table.notes = [
        "Per stratum, profile and counted region: Runs = reported runs on the stratum's coverage reference; "
        "Outcomes = those runs by outcome (c completed, tl time limit, nm no movement; the windows of E_j and "
        "K_j differ by outcome); "
        "Reached = k/n of the runs with a T90_j value; T90_j = the individual values in whole seconds up to "
        f"{FEW_RUNS} runs with a value, else the Kaplan–Meier median [Q1, Q3] (midpoint rule, RD-10), > L or "
        "not reached when not reached by L; Completed, not reached = completed runs censored at L in the "
        "region; Entries, median = median E_j over the runs with entries; Recoveries = K_j summed over the runs "
        "with recovery rows (— when none has them: dataset schema below 11 records none); Snapshots incomplete "
        "= runs without any T90_j because their explored-cell chain is incomplete. Area: the region's free "
        "floor from the descriptors.",
        *limits_text(),
        *extra_notes,
    ]
    not_recorded = sum(
        p.get("recoveries_null", {}).get("not_recorded", 0)
        for document in documents.values()
        for p in document["profiles"].values()
    )
    if not_recorded:
        table.notes.append(
            f"Recoveries not recorded in {not_recorded} run(s) (dataset schema below 11): K_j is —."
        )
    if without:
        table.notes.append(f"Strata without region measures, left out: {', '.join(without)}.")
    table.extra.update(measure=REGION_MEASURES, regions=documents)
    table.inputs = set(files)
    return table


def tables(src: Sources) -> list[Table]:
    """T12, grouped like T3–T7 (``Sources.with_groups``); none when the export holds no region raster."""
    if not has_regions(src.study.export):
        return []
    return [src.with_groups(t12(src))]
