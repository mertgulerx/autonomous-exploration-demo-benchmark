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

"""Figure F17: map heat maps across runs (docs/plans/FIGURE_EXTENSION.md, FX3, FX-12, FX-20; FIGURE_STYLE §3, §6;
METHODOLOGY §3 "Display quantities").

One figure per stratum (``F17_<stratum>``), one row of maps per profile over all its reported runs (time limit,
stuck and no movement included), on the stratum's coverage reference at one scale (millimetres per metre the
same in every map of the figure):

- (a) the per-cell explored share at L: the share of the runs whose explored reference cells (the worker's
  ``explored_cells``, the last snapshot at or before L; the pre-motion snapshot counts from 0) include the cell,
  on cividis over [0, 1] (a row of fewer than ``min_n`` runs says its shares come in steps of 1/n);
- (b) bin entries per run on 0.25 m bins (aligned with the reference origin) from the 20 Hz raw ``ground_truth``
  base pose, rows from movement onset to min(T_end, L): an entry is a run's arrival in a bin (consecutive valid
  samples in one bin count once, the first sample's bin is the first entry), summed over the runs and divided by
  the number of runs with ground truth (a run that never moved counts with 0); log cividis, one norm in the
  figure, bins without target cells masked, bins never entered left as map;
- (c) the goal poses of the ``goals`` table placed in the world with the run's frozen ``world_from_map``
  alignment (``goals.place``) in the shared goal marks (``goals.map_style``): reached filled, cancelled open, a
  cancel by the explorer's close-enough guard half filled, other statuses with their own marks; the legend counts
  each;
- (d) a histogram of (b)'s values over the entered target bins per profile.

No profile − Default map: it would be an unpaired difference without uncertainty. Every value is a descriptive
display quantity, written to the sidecar's ``extra`` and never cited. A run without a usable explored-cell snapshot
(an older export, a run without a coverage alignment, a mask of another size) is left out of (a)'s denominator, and
a moving run without usable ground truth out of (b)'s; both are counted. The scale is one per figure, the largest
that fits the page (not below ``MIN_SCALE``).
"""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
import math
from typing import Any

from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import numpy.typing as npt

from .data import REPORTED, TIME_EPSILON, Row, Stratum, never_moved, run_end
from .figures import (
    REFERENCE_PAD_M,
    Extent,
    FigureContext,
    FigureOut,
    MapCode,
    draw_reference,
    finish,
    load_reference,
    message_figure,
    panel_message,
    panel_title,
    reference_files,
    reference_name,
    group_inputs,
    stratum_variants,
)
from .goals import (
    ARRIVED,
    Goal,
    PlacedGoal,
    close_enough_text,
    map_handles,
    map_style,
    place,
    read_goals,
)
from .streams import (
    Mask,
    Reads,
    ReferenceCells,
    base_poses,
    explored_snapshots,
    reference_cells,
    world_from_map,
)
from .style import EDGE, PAGE_MAX_MM, PRESETS, SEQUENTIAL_CMAP, figure, page_note

BIN_M = 0.25  # METHODOLOGY §3: bin entries on 0.25 m bins
MIN_SCALE = 0.5  # mm per metre: below it the 5 m bar is 2.5 mm and the heat layers no longer read
# Layout in millimetres: the rotated profile label left of each row, gaps, the column titles above the maps,
# the band under the maps for the colour bars and goal legend, (d) and the evidence label at the foot.
LABEL_MM, ROW_LABEL_GAP_MM = 6.0, 1.5
FEW_LINE_MM = (
    3.5  # the row label's second line ("n < min_n: ..."), when some row has fewer than min_n runs
)
GAP_MM, ROW_GAP_MM, TITLE_MM = 3.0, 3.0, 6.0
BAR_MM, BAR_DROP_MM, BAR_H_MM, BAR_INSET_MM, LEGEND_RAISE_MM = 16.0, 3.0, 2.5, 2.0, 1.0
LEGEND_LINE_MM = (
    4.2  # one 8 pt line of the goal legend with its spacing; the band grows with the legend
)
HIST_MM, HIST_LEFT_MM, HIST_RIGHT_MM, HIST_BOTTOM_MM, HIST_TITLE_MM = 40.0, 12.0, 4.0, 10.0, 8.0
FOOT_MM = 9.0
SHARE_LABEL = "Share of runs explored by L"
ENTRY_LABEL = "Bin entries per run (log scale)"
TITLES = (SHARE_LABEL, f"Entries per run, {BIN_M:g} m bins", "Goal poses")
DESCRIPTIVE = "descriptive display quantity (METHODOLOGY §3), not in stats.json, never cited"
# goal map keys (goals.Goal.map_key) in drawing order: the most frequent first, reached on top
GOAL_DRAW_ORDER = ("canceled", ARRIVED, "preempted", "failed", "open", "other", "succeeded")

Floats = npt.NDArray[np.float64]
Bin = tuple[int, int]  # bin row (row 0 at the smallest y), bin column


@dataclass(frozen=True, slots=True)
class RunLayers:
    """One reported run's contribution: its explored mask at L (None without a usable snapshot, ``snapshot_issue``
    says why), the snapshot's time (s on the movement clock, 0 for the pre-motion one), its arrivals as (row,
    column) bins in order (None for a moving run without usable ground truth; empty for a run that never moved),
    and its goals dispatched by min(T_end, L) placed in the world (``goals.place``), with those of them that
    cannot be placed counted by map key; goals dispatched later are left out, not counted."""

    attempt_id: str
    outcome: str
    coverage_eligible: bool
    explored: Mask | None
    snapshot_s: float | None
    snapshot_issue: (
        str | None
    )  # "none" (no snapshot by L) or "size" (a mask of another size than the reference)
    arrivals: npt.NDArray[np.intp] | None  # shape (k, 2)
    samples: (
        int  # ground-truth rows counted (movement onset to min(T_end, L), valid, with an offset)
    )
    goals: tuple[PlacedGoal, ...]
    unplaced: Counter[str]

    @property
    def goal_keys(self) -> Counter[str]:
        """The run's goals dispatched by min(T_end, L) by map key (``goals.Goal.map_key``), placed or not; goals
        dispatched later are not counted (``goals.place`` drops them)."""
        return Counter(g.key for g in self.goals) + self.unplaced


@dataclass(frozen=True, slots=True)
class ProfileLayers:
    profile: str
    runs: tuple[RunLayers, ...]
    share: Floats | None  # per target cell, reference order; None when no run has snapshots
    # per bin, arrivals summed over the runs with ground truth (and those that never moved) over their number;
    # 0 outside target bins
    entries_per_run: Floats
    arrivals_outside: int  # arrivals in bins without a target cell or off the grid (not drawn)

    @property
    def with_snapshots(self) -> int:
        return sum(run.explored is not None for run in self.runs)

    @property
    def with_ground_truth(self) -> int:
        return sum(run.arrivals is not None for run in self.runs)

    def snapshot_issues(self, issue: str) -> int:
        return sum(run.snapshot_issue == issue for run in self.runs)

    @property
    def goals_unplaced(self) -> int:
        return sum(run.unplaced.total() for run in self.runs)


@dataclass(frozen=True, slots=True)
class Bins:
    """The 0.25 m bin grid over the reference grid, aligned with its origin, and which bins hold target cells."""

    origin: tuple[float, float]
    shape: tuple[int, int]
    target: Mask

    @property
    def extent(self) -> Extent:
        x0, y0 = self.origin
        return (x0, x0 + self.shape[1] * BIN_M, y0, y0 + self.shape[0] * BIN_M)


def _bins(cells: ReferenceCells) -> Bins:
    per_bin = BIN_M / cells.resolution
    shape = (math.ceil(cells.shape[0] / per_bin), math.ceil(cells.shape[1] / per_bin))
    target = np.zeros(shape, dtype=bool)
    target[(cells.rows / per_bin).astype(np.intp), (cells.columns / per_bin).astype(np.intp)] = True
    return Bins(cells.origin, shape, target)


def _explored_at(
    reads: Reads, attempt: str, limit_s: float, cells: int
) -> tuple[Mask | None, float | None, str | None]:
    """The last snapshot at or before L (the pre-motion one at 0) with its time, or None and the reason: "none"
    without a snapshot by L, "size" when a snapshot's mask does not have the reference's size (another
    reference: the run has no usable snapshot)."""
    chosen: tuple[Mask | None, float | None, str | None] = (None, None, "none")
    for elapsed_ns, mask in explored_snapshots(reads, attempt):
        if mask.size != cells:
            return None, None, "size"
        seconds = 0.0 if elapsed_ns is None else elapsed_ns / 1e9
        if seconds <= limit_s + TIME_EPSILON:
            chosen = (mask, seconds, None)
    return chosen


def _arrivals(
    reads: Reads, attempt: str, end_s: float, bins: Bins
) -> tuple[npt.NDArray[np.intp] | None, int]:
    """A moving run's arrivals in bins from movement onset to ``end_s`` and the rows counted (``base_poses``);
    consecutive rows in one bin are one arrival. None without a usable row (no stream, every row invalid or
    without an offset): unavailable, not zero."""
    poses = base_poses(reads, attempt, end_s)
    if poses is None:
        return None, 0
    found: list[Bin] = []
    for pose in poses:
        bx, by = pose.x - bins.origin[0], pose.y - bins.origin[1]
        cell = (math.floor(by / BIN_M), math.floor(bx / BIN_M))
        if not found or found[-1] != cell:
            found.append(cell)
    return (np.array(found, dtype=np.intp) if found else None), len(poses)


def _run_layers(
    reads: Reads,
    run: Row,
    stratum: Stratum,
    cells: ReferenceCells,
    bins: Bins,
    goals: tuple[Goal, ...],
) -> RunLayers:
    attempt = str(run["attempt_id"])
    explored, snapshot_s, issue = _explored_at(reads, attempt, stratum.limit_s, cells.rows.size)
    end_s, _ = run_end(run, stratum.limit_s)
    arrivals, samples = (
        _arrivals(reads, attempt, end_s, bins)
        if not never_moved(run)
        else (np.zeros((0, 2), dtype=np.intp), 0)  # never moved: no entries, in the divisor
    )
    placed, unplaced = place(goals, world_from_map(reads, attempt) if goals else None, end_s)
    return RunLayers(
        attempt,
        str(run["outcome"]),
        bool(run["coverage_eligible"]),
        explored,
        snapshot_s,
        issue,
        arrivals,
        samples,
        placed,
        unplaced,
    )


def _profile_layers(profile: str, runs: list[RunLayers], bins: Bins) -> ProfileLayers:
    masks = [r.explored for r in runs if r.explored is not None]
    share = np.asarray(np.stack(masks).mean(axis=0), dtype=np.float64) if masks else None
    counts = np.zeros(bins.shape, dtype=np.float64)
    outside = 0
    tracked = [r.arrivals for r in runs if r.arrivals is not None]
    for arrivals in tracked:
        rows, columns = arrivals[:, 0], arrivals[:, 1]
        inside = (rows >= 0) & (rows < bins.shape[0]) & (columns >= 0) & (columns < bins.shape[1])
        np.add.at(counts, (rows[inside], columns[inside]), 1.0)
        outside += int(np.count_nonzero(~inside))
    outside += int(counts[~bins.target].sum())
    counts[~bins.target] = 0.0
    entries = counts / len(tracked) if tracked else counts
    return ProfileLayers(profile, tuple(runs), share, entries, outside)


@dataclass(frozen=True, slots=True)
class StratumLayers:
    """What one F17 figure draws: the stratum's reference (map codes, extent, target cells, bins) and the
    layers of each profile with reported runs on it; the runs left out because they used another reference or
    none was exported, and the profiles left without a row, are counted by profile."""

    reference: str
    codes: npt.NDArray[Any]
    extent: Extent
    cells: ReferenceCells
    bins: Bins
    layers: tuple[ProfileLayers, ...]
    other_reference: dict[str, int]
    no_reference: dict[str, int]
    without_rows: tuple[
        str, ...
    ]  # profiles with final runs in the stratum but none reported on the reference
    goals_exported: bool


def _collect(ctx: FigureContext, stratum: Stratum, reads: Reads) -> StratumLayers | None:
    """The layers of every profile of the stratum on the reference its first run used, or None without one."""
    references = {
        str(r["attempt_id"]): reference_name(ctx, r["attempt_id"]) for r in stratum.reported
    }
    first = next(((a, ref) for a, ref in references.items() if ref is not None), None)
    if first is None:
        return None
    attempt, reference = first
    loaded, cells = load_reference(ctx, attempt), reference_cells(reads, reference)
    if loaded is None or cells is None:
        return None
    bins = _bins(cells)
    goals = read_goals(reads, stratum.reported)
    layers, other, missing, without = [], Counter[str](), Counter[str](), []
    for profile in ctx.profiles(stratum):
        runs = []
        for run in stratum.runs_of(profile):
            run_id = str(run["attempt_id"])
            if references[run_id] is None:
                missing[profile] += 1
            elif references[run_id] != reference:
                other[profile] += 1
            else:
                record = goals.get(run_id)
                found = () if record is None else record.goals
                runs.append(_run_layers(reads, run, stratum, cells, bins, found))
        if runs:
            layers.append(_profile_layers(profile, runs, bins))
        else:
            without.append(profile)
    return StratumLayers(
        reference,
        *loaded,
        cells,
        bins,
        tuple(layers),
        dict(other),
        dict(missing),
        tuple(without),
        "goals.parquet" in reads.export.files,
    )


@dataclass(frozen=True, slots=True)
class Layout:
    scale: float  # mm per metre
    map_mm: tuple[float, float]
    column_mm: float
    total_mm: float
    floored: bool  # the scale is at MIN_SCALE and the figure may be taller than a page


def _layout(
    width_mm: float, size_m: tuple[float, float], profiles: int, band_mm: float, label_mm: float
) -> Layout:
    """Three map columns at one scale, shrunk until the figure fits the page, but not below ``MIN_SCALE``;
    ``band_mm`` under the maps for the colour bars and the goal legend, ``label_mm`` left of them for the row
    labels."""
    column = (width_mm - label_mm - 2 * GAP_MM) / 3
    fixed = TITLE_MM + band_mm + HIST_MM + FOOT_MM + profiles * ROW_GAP_MM
    fitted = min(column / size_m[0], (PAGE_MAX_MM - fixed) / (profiles * size_m[1]))
    scale = max(fitted, MIN_SCALE)
    map_mm = (size_m[0] * scale, size_m[1] * scale)
    return Layout(scale, map_mm, column, fixed + profiles * map_mm[1], fitted < MIN_SCALE)


def _draw_goals(ax: Any, goals: list[PlacedGoal]) -> None:
    for key in GOAL_DRAW_ORDER:
        mine = [g for g in goals if g.key == key]
        if mine:
            ax.plot([g.x for g in mine], [g.y for g in mine], zorder=4, **map_style(key))


def _map(ax: Any, codes: npt.NDArray[Any], extent: Extent) -> None:
    """The reference map (``draw_reference``) with the obstacles outlined above the heat layers, since cividis's
    middle is a grey close to the obstacle grey; its scale bar (z 5) stays above the goal marks (z 4)."""
    ax.set_axis_off()
    draw_reference(ax, codes, extent)
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


def _heat(ax: Any, values: Floats, norm: Normalize, extent: Extent) -> AxesImage:
    """A heat layer: NaN cells take the colour map's transparent "bad" colour (masked, §6 item 6)."""
    image: AxesImage = ax.imshow(
        values,
        cmap=SEQUENTIAL_CMAP,
        norm=norm,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        interpolation_stage="rgba",
        zorder=1,
    )
    return image


def _log_axis(axis: Any) -> None:
    """Log ticks at 1, 2 and 5 per decade written as plain numbers (no superscripts below 7 pt)."""
    axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _entry_norm(layers: tuple[ProfileLayers, ...]) -> LogNorm | None:
    """One log norm over every profile's positive bin values (FX-12), or None when no run entered a bin."""
    values = np.concatenate(
        [p.entries_per_run[p.entries_per_run > 0] for p in layers] or [np.zeros(0)]
    )
    if not values.size:
        return None
    low = float(values.min())
    # when every bin has the same value (one run, each bin entered once) the norm still needs a range
    return LogNorm(low, max(float(values.max()), 2 * low))


def _draw_row(
    panels: list[Any], data: StratumLayers, layer: ProfileLayers, entry_norm: LogNorm | None
) -> tuple[AxesImage | None, AxesImage | None]:
    """One profile's maps (a)–(c); returns the share and entry images (None where not drawn)."""
    share_ax, entry_ax, goal_ax = panels
    for ax in panels:
        _map(ax, data.codes, data.extent)
    share_image = entry_image = None
    if layer.share is not None:
        image = np.full(data.cells.shape, np.nan)
        image[data.cells.rows, data.cells.columns] = layer.share
        share_image = _heat(share_ax, image, Normalize(0.0, 1.0), data.extent)
    else:
        panel_message(share_ax, "no explored-cell snapshots", backdrop=True)
    entries = layer.entries_per_run
    if entry_norm is not None and np.any(entries > 0):
        values = np.where(entries > 0, entries, np.nan)
        entry_image = _heat(entry_ax, values, entry_norm, data.bins.extent)
    else:
        panel_message(entry_ax, "no bin entries", backdrop=True)
    goals = [g for run in layer.runs for g in run.goals]
    _draw_goals(goal_ax, goals)
    if not data.goals_exported:
        panel_message(goal_ax, "goals not exported", backdrop=True)
    elif not goals:
        unplaced = layer.goals_unplaced
        panel_message(
            goal_ax,
            f"no goal placed ({unplaced} without\na map-frame pose or alignment)"
            if unplaced
            else "no goals",
            backdrop=True,
        )
    for ax in panels:  # the reference limits, also where a bin layer reaches past them
        ax.set_xlim(data.extent[0] - REFERENCE_PAD_M, data.extent[1] + REFERENCE_PAD_M)
        ax.set_ylim(data.extent[2] - REFERENCE_PAD_M, data.extent[3] + REFERENCE_PAD_M)
    return share_image, entry_image


def _keys(
    fig: Figure,
    box: Any,
    lefts: list[float],
    column_mm: float,
    bar_top: float,
    images: list[AxesImage | None],
    goal_handles: list[Any],
) -> None:
    """The colour bars under (a) and (b) and the goal legend under (c), one entry per line, from ``bar_top``
    (mm) down."""
    for column, (image, text) in enumerate(zip(images, (SHARE_LABEL, ENTRY_LABEL), strict=True)):
        if image is not None:
            cax = fig.add_axes(
                box(
                    lefts[column] + BAR_INSET_MM,
                    bar_top - BAR_H_MM,
                    column_mm - 2 * BAR_INSET_MM,
                    BAR_H_MM,
                )
            )
            fig.colorbar(image, cax=cax, orientation="horizontal").set_label(text)
            if column == 1:
                _log_axis(cax.xaxis)
    x, y, _, _ = box(lefts[2] + column_mm / 2, bar_top + LEGEND_RAISE_MM, 0.0, 0.0)
    fig.legend(
        handles=goal_handles,
        loc="upper center",
        bbox_to_anchor=(x, y),
        ncols=1,
        fontsize=8,
        handletextpad=0.3,
    )


def _histogram(
    ax: Any, ctx: FigureContext, data: StratumLayers, entry_norm: LogNorm | None
) -> None:
    """(d): the profiles' bin values over the entered target bins, as steps on a log axis."""
    panel_title(ax, 3, "Entered target bins by entries per run")
    ax.set_xlabel(ENTRY_LABEL)
    ax.set_ylabel("Bins")
    if entry_norm is None:
        panel_message(ax, "no bin entries", boxed=True)
        return
    low, high = float(entry_norm.vmin or 1.0), float(entry_norm.vmax or 2.0)
    edges = np.geomspace(low / 1.05, high * 1.05, 25)
    for layer in data.layers:
        style = ctx.styles[layer.profile]
        values = layer.entries_per_run[data.bins.target]
        values = values[values > 0]
        if values.size:
            ax.hist(
                values,
                bins=edges.tolist(),
                histtype="step",
                color=style.color,
                linestyle=style.linestyle,
                linewidth=1.0,
                label=f"{style.name} ({values.size} bins)",
            )
    ax.set_xscale("log")
    ax.set_xlim(min(_tick(low, below=True), edges[0]), max(_tick(high, below=False), edges[-1]))
    _log_axis(ax.xaxis)
    ax.legend(loc="upper right")


def _tick(value: float, *, below: bool) -> float:
    """The nearest 1-2-5 tick below (or above) a positive value, so the log axis shows a label at each end."""
    decade = 10.0 ** math.floor(math.log10(value))
    ticks = [decade * m for m in (1.0, 2.0, 5.0, 10.0)]
    return max(t for t in ticks if t <= value) if below else min(t for t in ticks if t >= value)


def _summary(
    ctx: FigureContext, stratum: Stratum, data: StratumLayers, layer: ProfileLayers
) -> dict[str, Any]:
    """The sidecar numbers of one profile's panels (descriptive, FX-20)."""
    share, entered = layer.share, layer.entries_per_run[data.bins.target]
    return dict(
        profile=layer.profile,
        name=ctx.styles[layer.profile].name,
        runs=len(layer.runs),
        outcomes=dict(sorted(Counter(r.outcome for r in layer.runs).items())),
        final_not_reported={
            k: v for k, v in stratum.counts(layer.profile).items() if k != REPORTED
        },
        other_reference=data.other_reference.get(layer.profile, 0),
        reference_not_exported=data.no_reference.get(layer.profile, 0),
        coverage_ineligible_included=sum(not r.coverage_eligible for r in layer.runs),
        runs_with_snapshots=layer.with_snapshots,
        snapshot_size_mismatch=layer.snapshot_issues("size"),
        runs_without_ground_truth=len(layer.runs) - layer.with_ground_truth,
        target_cells=int(data.cells.rows.size),
        share_max=None if share is None else round(float(share.max()), 4),
        share_mean=None if share is None else round(float(share.mean()), 4),
        cells_share_1=None if share is None else int(np.count_nonzero(share >= 1.0)),
        cells_share_0=None if share is None else int(np.count_nonzero(share <= 0.0)),
        target_bins=int(data.bins.target.sum()),
        bins_entered=int(np.count_nonzero(entered > 0)),
        arrivals_total=int(sum(len(r.arrivals) for r in layer.runs if r.arrivals is not None)),
        arrivals_outside_target_bins=layer.arrivals_outside,
        entries_per_run_max=round(float(entered.max()), 4) if entered.size else None,
        goals=dict(sorted(sum((r.goal_keys for r in layer.runs), Counter[str]()).items())),
        goals_drawn=dict(sorted(Counter(g.key for r in layer.runs for g in r.goals).items())),
        goals_unplaced=layer.goals_unplaced,
        per_run=[
            dict(
                attempt_id=r.attempt_id,
                outcome=r.outcome,
                snapshot_s=r.snapshot_s,
                explored_cells=None if r.explored is None else int(r.explored.sum()),
                arrivals=None if r.arrivals is None else len(r.arrivals),
                ground_truth_rows=r.samples,
                goals=len(r.goals),
            )
            for r in layer.runs
        ],
    )


def _notes(
    ctx: FigureContext, stratum: Stratum, data: StratumLayers, label: str, layout: Layout
) -> list[str]:
    notes = [
        f"{label}, L = {stratum.limit_s:g} s of simulated time since first movement; all reported runs per "
        "profile (time limit, stuck and no movement included). Values are descriptive display quantities "
        f"(METHODOLOGY §3): sidecar only, never cited. Every map at {layout.scale:.4g} mm per metre (5 m bar), "
        f"cropped to the reference grid plus {REFERENCE_PAD_M:g} m; map greys as FIGURE_STYLE §6, obstacles "
        "outlined, outside the reference hatched.",
        "(a) per reference target cell, the share of the runs whose explored cells (explored_cells, the last "
        "snapshot at or before L; the pre-motion snapshot counts at 0) include it; cividis over [0, 1]. "
        f"(b) entries per run on {BIN_M:g} m bins aligned with the reference origin, from the 20 Hz raw ground "
        "truth (base pose) between movement onset and min(T_end, L): an entry is an arrival in a bin, staying "
        "counts once, a run that never moved has none; summed over the runs with ground truth and divided by their "
        "number; log "
        "cividis with one norm in the figure; bins without target cells masked, bins never entered left as map. "
        "(c) goal poses dispatched by the run's end, placed in the world with the run's frozen world_from_map "
        "alignment; filled circle reached by Nav2, half-filled circle ended by the explorer's close-enough guard "
        "(cancel reason completion_distance: the robot came within that distance of the goal), open circle "
        "cancelled otherwise, other statuses by their marks; the legend counts each (sidecar goals: placed or "
        f"not). A row of fewer than {ctx.settings.min_n} runs (min_n) says so: its shares in (a) come in steps "
        "of 1/n. (d) the values of (b) over the entered target bins, per profile.",
        "No profile − Default map: it would be an unpaired difference without uncertainty.",
    ]
    left_out = []
    for layer in data.layers:
        counts = {k: v for k, v in stratum.counts(layer.profile).items() if k != REPORTED}
        if counts:
            text = ", ".join(f"{k} {v}" for k, v in counts.items())
            left_out.append(f"{ctx.styles[layer.profile].name}: {text}")
    if left_out:
        notes.append(f"Final runs not reported (not drawn, counted): {'; '.join(left_out)}.")
    if data.without_rows:
        names = ", ".join(ctx.styles[p].name for p in data.without_rows)
        notes.append(f"No row for profiles without reported runs on this reference: {names}.")
    other = sum(data.other_reference.values())
    if other:
        notes.append(
            f"{other} reported run(s) on another coverage reference are left out, counted."
        )
    missing = sum(data.no_reference.values())
    if missing:
        notes.append(
            f"{missing} reported run(s) whose coverage reference was not exported are left out."
        )
    without = sum(len(layer.runs) - layer.with_snapshots for layer in data.layers)
    if without:
        sizes = sum(layer.snapshot_issues("size") for layer in data.layers)
        notes.append(
            f"{without} run(s) without a usable explored-cell snapshot by L ({sizes} with a mask of another "
            "size than the reference) are left out of (a)'s denominator, counted."
        )
    ineligible = sum(not r.coverage_eligible for layer in data.layers for r in layer.runs)
    if ineligible:
        notes.append(
            f"(a) includes {ineligible} reported run(s) that are not coverage-eligible (their explored cells "
            "are drawn as recorded; their coverage values are not in the statistics)."
        )
    untracked = sum(len(layer.runs) - layer.with_ground_truth for layer in data.layers)
    if untracked:
        notes.append(
            f"{untracked} moving run(s) without usable ground truth are left out of (b) and (d)'s divisor, "
            "counted."
        )
    if not data.goals_exported:
        notes.append("The export holds no goals table: (c) is empty.")
    unplaced = sum(layer.goals_unplaced for layer in data.layers)
    if unplaced:
        notes.append(f"{unplaced} goal(s) without a map-frame pose or alignment are not drawn.")
    tall = page_note(layout.total_mm) if layout.floored else None
    if tall is not None:
        notes.append(tall)
    return notes


def _figure(ctx: FigureContext, stratum: Stratum, name: str, variant_label: str) -> FigureOut:
    """One stratum's heat maps, named and labelled by ``figures.stratum_variants``."""
    reads = Reads(ctx.study.export)
    attempts = {str(r["attempt_id"]) for r in stratum.reported}
    label = ctx.label(stratum)
    data = _collect(ctx, stratum, reads)
    inputs = reads.files | reference_files(ctx, None if data is None else data.reference)
    if data is None or not data.layers:
        text = (
            "coverage reference not readable"
            if data is None
            else "no reported run on the coverage reference"
        )
        built = message_figure(ctx, name, f"{label}: {text}", attempts=attempts, inputs=inputs)
        built.extra["stratum"] = variant_label
        return built
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    extent = data.extent
    size_m = (
        extent[1] - extent[0] + 2 * REFERENCE_PAD_M,
        extent[3] - extent[2] + 2 * REFERENCE_PAD_M,
    )
    goal_counts = Counter(g.key for layer in data.layers for run in layer.runs for g in run.goals)
    close_enough = close_enough_text(ctx.study.export, [layer.profile for layer in data.layers])
    goal_handles = map_handles(goal_counts, close_enough)
    lines = sum(str(handle.get_label()).count("\n") + 1 for handle in goal_handles)
    band_mm = max(BAR_MM, BAR_DROP_MM - LEGEND_RAISE_MM + LEGEND_LINE_MM * lines + 2.0)
    few = any(len(layer.runs) < ctx.settings.min_n for layer in data.layers)
    label_mm = LABEL_MM + (FEW_LINE_MM if few else 0.0)
    layout = _layout(width_mm, size_m, len(data.layers), band_mm, label_mm)
    total = layout.total_mm
    # The panels are placed in millimetres (as in F12); the empty grid axis keeps the layout engine quiet.
    fig, axes = figure(ctx.preset, "double", aspect=total / width_mm)
    axes[0][0].set_axis_off()

    def box(left: float, bottom: float, w: float, h: float) -> tuple[float, float, float, float]:
        return (left / width_mm, bottom / total, w / width_mm, h / total)

    entry_norm = _entry_norm(data.layers)
    lefts = [label_mm + i * (layout.column_mm + GAP_MM) for i in range(3)]
    inset = (layout.column_mm - layout.map_mm[0]) / 2
    top = total - TITLE_MM
    images: list[AxesImage | None] = [None, None]
    for index, layer in enumerate(data.layers):
        bottom = top - layout.map_mm[1]
        n = len(layer.runs)
        steps = (
            f"\nn < {ctx.settings.min_n}: shares in steps of 1/{n}"
            if n < ctx.settings.min_n
            else ""
        )
        fig.text(
            (lefts[0] + inset - ROW_LABEL_GAP_MM) / width_mm,
            (bottom + layout.map_mm[1] / 2) / total,
            f"{ctx.styles[layer.profile].name} · n = {n}{steps}",
            rotation=90,
            ha="right",
            va="center",
            fontsize=8,
        )
        panels = [fig.add_axes(box(x + inset, bottom, *layout.map_mm)) for x in lefts]
        drawn = _draw_row(panels, data, layer, entry_norm)
        images = [new if new is not None else old for new, old in zip(drawn, images, strict=True)]
        if index == 0:
            for column, title in enumerate(TITLES):
                panel_title(panels[column], column, title)
        top = bottom - ROW_GAP_MM
    _keys(
        fig,
        box,
        lefts,
        layout.column_mm,
        top + ROW_GAP_MM - BAR_DROP_MM,
        images,
        goal_handles,
    )
    hist = fig.add_axes(
        box(
            lefts[0] + HIST_LEFT_MM,
            FOOT_MM + HIST_BOTTOM_MM,
            width_mm - lefts[0] - HIST_LEFT_MM - HIST_RIGHT_MM,
            HIST_MM - HIST_BOTTOM_MM - HIST_TITLE_MM,
        )
    )
    _histogram(hist, ctx, data, entry_norm)
    finish(fig, ctx, descriptive=True)

    return FigureOut(
        name,
        fig,
        "double",
        attempts,
        f"Map heat maps for {label} (descriptive): per profile, the share of runs that explored each reference "
        "cell by L, ground-truth bin entries per run, and goal poses by status, with a histogram of bin entries.",
        _notes(ctx, stratum, data, label, layout),
        dict(
            stratum=variant_label,
            robot_id=stratum.robot_id,
            world_id=stratum.world_id,
            spawn_id=stratum.spawn_id,
            comparison_key=stratum.comparison_key,
            limit_s=stratum.limit_s,
            reference=data.reference,
            bin_m=BIN_M,
            mm_per_m=round(layout.scale, 4),
            share_norm=[0.0, 1.0],
            entries_norm=None
            if entry_norm is None
            else [round(float(entry_norm.vmin or 0.0), 6), round(float(entry_norm.vmax or 0.0), 6)],
            display=DESCRIPTIVE,
            profiles=[_summary(ctx, stratum, data, layer) for layer in data.layers],
        ),
        # load_reference and draw_reference read the reference's files outside ``reads``
        inputs=inputs,
    )


def figures(ctx: FigureContext) -> Iterator[FigureOut]:
    """One F17 per stratum (``figures.stratum_variants``), each built when the caller asks for it."""
    for stratum, name, label in stratum_variants(ctx, "F17"):
        built = _figure(ctx, stratum, name, label)
        built.inputs |= group_inputs(ctx, stratum)
        yield built
