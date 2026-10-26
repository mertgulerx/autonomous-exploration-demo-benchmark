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

"""Region measures ``region-measures-v1`` (plans/worlds/I8 §3.5; decision 56): per reported run and counted
region of the world's reference grid, the region coverage time, the entries and the Nav2 recoveries placed in it.

Diagnostics, never outcomes or rankings; computed from the export, never stored in the run record. Times are
simulated seconds since movement onset t0 (the milestone clock), L the stratum's time limit, the window
[t0, min(T_end, L)].

- Regions (T1 §4.2): the label raster on the reference grid (``worlds.regions``); counted regions are those of
  at least 4.0 m² (``Regions.counted``); smaller ones keep their label and are not counted. R_j are the reference
  target cells with label j. A raster without a counted region gives no measure (``no_counted_regions``).
- Region coverage time T90_j (``coverage_times``), coverage-eligible runs only (else ``coverage_ineligible``):
  the time of the first explored-cell snapshot at or before L in which at least 90 % of R_j is explored. The
  t = 0 state is the last pre-motion snapshot only (a region at 90 % there has T90_j = 0); later snapshots are
  stamped with their own movement clock (every 5 s, phase-locked to activation, so up to one period after the
  crossing). Not reached by L: censored at L, also for a run that completed or stuck earlier (event-free
  through L, METHODOLOGY §4.5); the last snapshot used may lie up to one period before L. ``no_movement``: never
  reached, censored at L, with the last pre-motion snapshot's share (RD-8). The chain is incomplete, giving no
  value for any region (``snapshots_incomplete``), when the stream is missing or empty, a row was not decoded (a
  broken chain or a row marked invalid, ``streams.explored_chain``), two consecutive snapshots up to the terminal one
  lie more than 5.5 s apart (a dropped row: the worker restarts with a keyframe, so the chain itself stays whole),
  or, for a run that moved, the last decoded snapshot (a pre-motion one counts as t = 0) lies more than 5.5 s
  before the window's end min(T_end, L) (``data.run_end``): the terminal row that closes the window was lost;
  the largest spacing is recorded. |R_j| = 0 gives no value (``no_target_cells``).
- Entries E_j (``entries``), from the base positions of the 20 Hz ``ground_truth`` rows in the window,
  independent of odometry (``motion_eligible``). A sample's region is the label of its cell; samples outside every
  counted region are dropped (they neither start nor end a stay; their time counts for no region). A stay is a
  maximal run of samples in one region; its duration is the sum of the intervals from each of its samples to the
  next window sample, an interval longer than 0.25 s counted as a gap and left out. The window's leading
  (t0 to the first sample) and trailing (the last sample to its end) intervals are gaps too; any interval longer
  than 2 s, or no sample in the window, gives no value (``ground_truth_gap``, ``no_ground_truth_in_window``). A
  stay is an entry when it lasts at least 1.0 s and holds a sample at least 0.25 m inside its region (no 8-
  connected path of fewer than 5 cells in the region to a cell of another region: doorway hysteresis). Stays that
  are not entries are removed one at a time, shortest first (earliest on a tie), their equal neighbours joined
  (durations added), until every stay is an entry. The counted region of the first window sample in a counted
  region is the first entry even when its stay is not an entry. The run order is the regions by first entry.
  ``no_movement``: 1 for the region of the last valid pre-motion sample, 0 elsewhere; without a pre-motion sample
  no value.
- Recoveries K_j (``recoveries``), navigation-eligible runs only (else ``navigation_ineligible``): the
  ``navigation`` rows with ``event = "recovery"`` (dataset schema 11; ``not_recorded`` before, and
  ``no_navigation_stream`` for a schema-11 run without the stream). A row before t0 or after min(T_end, L) by its
  own clock is counted ``before_onset`` or ``after_end``; one in the window is placed at the base position of the
  ground-truth sample nearest in clock within 50 ms (any sample, not cut at the window), ``unplaced`` without
  one and ``outside`` when that position is in no counted region. The rows must add up to the run's
  ``recoveries`` (``goal_facts``); a difference is recorded, never corrected.

The pure functions take arrays and numbers; ``run_regions`` reads one run's streams from the export.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from .data import TIME_EPSILON, Row, never_moved, run_end
from .export import Export
from .streams import (
    Mask,
    Reads,
    ReferenceCells,
    Snapshot,
    SnapshotChain,
    base_poses,
    explored_chain,
    reference_cells,
)
from .worlds import Regions, regions

REGION_MEASURES = "region-measures-v1"
COVERED_SHARE = 0.9  # T90_j: share of R_j explored
# an entry: a stay of at least ENTRY_MIN_S holding a sample at least ENTRY_DEPTH_M inside its region
ENTRY_MIN_S = 1.0
ENTRY_DEPTH_M = 0.25
GAP_S = 0.25  # a ground-truth interval longer than this is a gap, left out of stay durations
GROUND_TRUTH_GAP_MAX_S = 2.0  # a longer interval in the window: no entries (as C(h)'s gap rule)
SNAPSHOT_SPACING_MAX_S = 5.5  # 5 s snapshots: a wider spacing means a dropped snapshot
# a recovery row takes the ground-truth sample nearest in clock within this
RECOVERY_MATCH_NS = 50_000_000
# the dataset schema from which runs record ``recovery`` rows (plans/worlds/I5 §3.4)
RECOVERY_SCHEMA = 11

Labels = npt.NDArray[np.uint16]


@dataclass(frozen=True, slots=True)
class RegionTime:
    """T90_j of one region in one run: ``value`` in seconds or None; ``event`` True when reached by L, False when
    censored at L, None without a value; ``reason`` "observed", "censored_at_limit", "no_movement",
    "snapshots_incomplete", "coverage_ineligible" or "no_target_cells"; ``share`` the explored share of R_j at
    the event (censored: at the last snapshot used), None without a value."""

    value: float | None
    event: bool | None
    reason: str
    share: float | None


@dataclass(frozen=True, slots=True)
class RegionEntries:
    """E_j of one run: ``counts`` per counted region (None without a value), ``order`` the regions by first
    entry, ``gaps`` and ``gap_s`` the window intervals longer than ``GAP_S`` and their total length, ``samples``
    used and ``reason`` ("observed", "no_movement", "no_sample_in_regions", "no_ground_truth",
    "no_ground_truth_in_window", "ground_truth_gap")."""

    counts: dict[int, int] | None
    order: tuple[int, ...]
    gaps: int
    gap_s: float
    samples: int
    reason: str


@dataclass(frozen=True, slots=True)
class RegionRecoveries:
    """K_j of one run: ``counts`` per counted region (None without a value) and the recovery rows not counted in
    a region: ``unplaced`` (no sample within 50 ms), ``outside`` (placed in no counted region), ``before_onset``
    and ``after_end`` (by the row's clock); ``rows`` the recovery rows, ``expected`` the run's ``recoveries``
    (None when unknown); ``reason``."""

    counts: dict[int, int] | None
    unplaced: int = 0
    outside: int = 0
    before_onset: int = 0
    after_end: int = 0
    rows: int = 0
    expected: int | None = None
    reason: str = "observed"

    @property
    def mismatch(self) -> bool | None:
        """The rows differ from the run's ``recoveries`` (None when either is unknown)."""
        if self.counts is None or self.expected is None:
            return None
        return self.rows != self.expected


@dataclass(slots=True)
class _Stay:
    region: int
    duration: float
    deep: bool  # holds a sample at least ENTRY_DEPTH_M inside the region


# Pure functions --------------------------------------------------------------------------------------------------


def target_labels(raster: Regions, cells: ReferenceCells) -> Labels | None:
    """The region label of every reference target cell, in reference order (the order of ``explored_cells``
    masks); None when the raster and the reference grid differ in shape."""
    if raster.labels.shape != cells.shape:
        return None
    return raster.labels[cells.rows, cells.columns]


def _shifted(grid: npt.NDArray[Any], dy: int, dx: int, fill: Any) -> npt.NDArray[Any]:
    """``grid`` moved by (dy, dx) cells, ``fill`` where nothing moved in."""
    moved = np.full_like(grid, fill)
    height, width = grid.shape
    moved[max(dy, 0) : height + min(dy, 0), max(dx, 0) : width + min(dx, 0)] = grid[
        max(-dy, 0) : height + min(-dy, 0), max(-dx, 0) : width + min(-dx, 0)
    ]
    return moved


NEIGHBOURS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def deep_cells(raster: Regions) -> Mask:
    """The raster cells at least ``ENTRY_DEPTH_M`` inside their region: labelled cells with no 8-connected path
    of fewer than ``ceil(ENTRY_DEPTH_M / resolution)`` steps inside their region to a cell of another region
    (an 8-connected step is never longer than the distance it covers, so the depth is at least that)."""
    labels = raster.labels
    labelled = labels > 0
    steps = math.ceil(ENTRY_DEPTH_M / raster.resolution - TIME_EPSILON)
    shifted = [_shifted(labels, dy, dx, 0) for dy, dx in NEIGHBOURS]
    shallow = labelled & np.logical_or.reduce([(s != labels) & (s > 0) for s in shifted])
    same = [s == labels for s in shifted]
    for _ in range(steps - 1):
        grown = np.logical_or.reduce(
            [
                _shifted(shallow, dy, dx, False) & m
                for (dy, dx), m in zip(NEIGHBOURS, same, strict=True)
            ]
        )
        shallow |= grown & labelled
    return np.asarray(labelled & ~shallow)


def values_at(
    raster: Regions, grid: npt.NDArray[Any], x: npt.ArrayLike, y: npt.ArrayLike, fill: Any
) -> npt.NDArray[Any]:
    """The value of ``grid`` (the raster's shape) at each world position's cell, ``fill`` off the raster."""
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    columns = np.floor((xs - raster.origin[0]) / raster.resolution).astype(np.intp)
    rows = np.floor((ys - raster.origin[1]) / raster.resolution).astype(np.intp)
    height, width = grid.shape
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    found = np.full(xs.shape, fill, dtype=grid.dtype)
    found[inside] = grid[rows[inside], columns[inside]]
    return found


def label_at(raster: Regions, x: npt.ArrayLike, y: npt.ArrayLike) -> Labels:
    """The region label of each world position's cell (0 off the raster)."""
    return values_at(raster, raster.labels, x, y, 0)


def snapshot_spacing(chain: SnapshotChain) -> float | None:
    """The largest clock spacing of consecutive decoded snapshots, in seconds (None below two)."""
    clocks = chain.clocks
    if len(clocks) < 2:
        return None
    return max(b - a for a, b in zip(clocks, clocks[1:], strict=False)) / 1e9


def chain_complete(chain: SnapshotChain | None, window_end_s: float | None = None) -> bool:
    """The snapshots can give T90_j: rows exist, every row decoded, no spacing above 5.5 s and, for a run that
    moved (``window_end_s`` = min(T_end, L) from ``data.run_end``; None for ``no_movement``, whose T90_j reads
    only the pre-motion state), the last decoded snapshot (a pre-motion one at t = 0) no more than 5.5 s before
    the window's end: a later terminal row was lost, so no row closes the window."""
    if chain is None or chain.rows == 0 or chain.skipped or chain.invalid:
        return False
    spacing = snapshot_spacing(chain)
    if spacing is not None and spacing > SNAPSHOT_SPACING_MAX_S + TIME_EPSILON:
        return False
    if window_end_s is None or not chain.snapshots:
        return True
    last_ns = chain.snapshots[-1][0]
    last_s = 0.0 if last_ns is None else last_ns / 1e9
    return window_end_s - last_s <= SNAPSHOT_SPACING_MAX_S + TIME_EPSILON


def coverage_times(
    labels: Labels,
    counted: frozenset[int],
    snapshots: Sequence[Snapshot],
    *,
    limit_s: float,
    outcome: str,
    complete: bool,
) -> dict[int, RegionTime]:
    """T90_j for every counted region (``labels``: per target cell, reference order; ``snapshots`` in clock
    order; ``complete`` per ``chain_complete``)."""
    if not complete or not snapshots:
        return {j: RegionTime(None, None, "snapshots_incomplete", None) for j in sorted(counted)}
    before = [mask for ns, mask in snapshots if ns is None]
    timed: list[tuple[float, Mask]] = [(0.0, before[-1])] if before else []
    if outcome != "no_movement":  # never reaches (RD-8): only the last pre-motion state counts
        timed += [
            (ns / 1e9, m)
            for ns, m in snapshots
            if ns is not None and ns / 1e9 <= limit_s + TIME_EPSILON
        ]
    found: dict[int, RegionTime] = {}
    for j in sorted(counted):
        members = labels == j
        size = int(np.count_nonzero(members))
        if size == 0:
            found[j] = RegionTime(None, None, "no_target_cells", None)
            continue
        share: float | None = None
        reached: float | None = None
        for t, mask in timed:
            share = int(np.count_nonzero(mask & members)) / size
            if outcome != "no_movement" and share >= COVERED_SHARE:
                reached = t
                break
        if reached is not None:
            found[j] = RegionTime(reached, True, "observed", share)
        else:
            reason = "no_movement" if outcome == "no_movement" else "censored_at_limit"
            found[j] = RegionTime(limit_s, False, reason, share)
    return found


def _stays(
    times: Sequence[float],
    regions_seen: Sequence[int],
    deep: Sequence[bool],
    counted: frozenset[int],
    end_s: float,
) -> tuple[list[_Stay], int, float, float]:
    """Stays in order, the gaps, their total seconds and the longest interval: each sample owns the interval to
    the next window sample (none when longer than ``GAP_S``); the leading and trailing intervals of the window
    are gaps when longer than ``GAP_S``; samples outside the counted regions are dropped, so they join stays."""
    edges = [times[0], end_s - times[-1]]
    steps = [b - a for a, b in zip(times, times[1:], strict=False)]
    gaps = [s for s in [*edges, *steps] if s > GAP_S]
    stays: list[_Stay] = []
    for index, j in enumerate(regions_seen):
        step = steps[index] if index < len(steps) else 0.0
        owned = step if step <= GAP_S else 0.0
        if j not in counted:
            continue
        if stays and stays[-1].region == j:
            stays[-1].duration += owned
            stays[-1].deep |= deep[index]
        else:
            stays.append(_Stay(j, owned, deep[index]))
    return stays, len(gaps), float(sum(gaps)), max([*edges, *steps], default=0.0)


def _is_entry(stay: _Stay) -> bool:
    return stay.deep and stay.duration >= ENTRY_MIN_S - TIME_EPSILON


def _entries_only(stays: list[_Stay]) -> list[_Stay]:
    """Remove the stays that are not entries shortest first (earliest on a tie), joining equal neighbours."""
    stays = [_Stay(s.region, s.duration, s.deep) for s in stays]
    while short := [i for i, s in enumerate(stays) if not _is_entry(s)]:
        index = min(short, key=lambda i: (stays[i].duration, i))
        del stays[index]
        if 0 < index < len(stays) and stays[index - 1].region == stays[index].region:
            stays[index - 1].duration += stays[index].duration
            stays[index - 1].deep |= stays[index].deep
            del stays[index]
    return stays


def entries(
    times: Sequence[float],
    regions_seen: Sequence[int],
    deep: Sequence[bool],
    counted: frozenset[int],
    *,
    end_s: float,
    outcome: str,
) -> RegionEntries:
    """E_j from the samples' movement-clock times, region labels and depth flags (``deep_cells``): the window
    samples in clock order, or for ``no_movement`` the pre-motion samples."""
    if outcome == "no_movement":
        if not times:
            return RegionEntries(None, (), 0, 0.0, 0, "no_ground_truth")
        start = regions_seen[-1] if regions_seen[-1] in counted else None
        counts = {j: int(j == start) for j in sorted(counted)}
        started = () if start is None else (start,)
        return RegionEntries(counts, started, 0, 0.0, len(times), "no_movement")
    if not times:
        return RegionEntries(None, (), 0, 0.0, 0, "no_ground_truth_in_window")
    stays, gaps, gap_s, longest = _stays(times, regions_seen, deep, counted, end_s)
    if longest > GROUND_TRUTH_GAP_MAX_S + TIME_EPSILON:
        return RegionEntries(None, (), gaps, gap_s, len(times), "ground_truth_gap")
    counts = {j: 0 for j in sorted(counted)}
    if not stays:
        return RegionEntries(counts, (), gaps, gap_s, len(times), "no_sample_in_regions")
    first = stays[0].region
    counts[first] += 1
    order = [first]
    for index, stay in enumerate(_entries_only(stays)):
        if index == 0 and stay.region == first:
            continue  # the start stay, already counted
        counts[stay.region] += 1
        if stay.region not in order:
            order.append(stay.region)
    return RegionEntries(counts, tuple(order), gaps, gap_s, len(times), "observed")


def recoveries(
    recovery_clocks: Sequence[int],
    *,
    onset_clock_ns: int | None,
    end_s: float,
    sample_clocks: Sequence[int],
    sample_regions: Sequence[int],
    counted: frozenset[int],
    expected: int | None,
) -> RegionRecoveries:
    """K_j of recorded recovery rows (clocks): before t0 or after ``end_s`` by the row's own clock (every row
    when the run never moved, ``onset_clock_ns`` None), else placed at the ground-truth sample nearest in clock
    within ``RECOVERY_MATCH_NS`` (``sample_clocks`` sorted, any sample of the run)."""
    clocks = np.asarray(sample_clocks, dtype=np.int64)
    counts = {j: 0 for j in sorted(counted)}
    tally: Counter[str] = Counter()
    for clock in recovery_clocks:
        if onset_clock_ns is None or clock < onset_clock_ns:
            tally["before_onset"] += 1
            continue
        if (clock - onset_clock_ns) / 1e9 > end_s + TIME_EPSILON:
            tally["after_end"] += 1
            continue
        position = int(np.searchsorted(clocks, clock))
        near = [i for i in (position - 1, position) if 0 <= i < len(clocks)]
        best = min(near, key=lambda i: (abs(int(clocks[i]) - clock), i), default=None)
        if best is None or abs(int(clocks[best]) - clock) > RECOVERY_MATCH_NS:
            tally["unplaced"] += 1
        elif sample_regions[best] in counted:
            counts[sample_regions[best]] += 1
        else:
            tally["outside"] += 1
    return RegionRecoveries(
        counts,
        tally["unplaced"],
        tally["outside"],
        tally["before_onset"],
        tally["after_end"],
        len(recovery_clocks),
        expected,
    )


# One run from the export -----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunRegions:
    """``region-measures-v1`` of one run; ``reason`` names why the run has no measures at all
    ("no_region_raster", "reference_not_readable", "raster_shape_differs", "no_counted_regions"), else
    "observed"; the snapshot chain's rows, skipped and invalid rows and largest spacing; ``inputs`` the export
    files this run's measures read."""

    attempt_id: str
    profile_id: str
    outcome: str
    reason: str
    times: dict[int, RegionTime]
    entries: RegionEntries | None
    recoveries: RegionRecoveries | None
    snapshot_rows: int = 0
    snapshot_skipped: int = 0
    snapshot_invalid: int = 0
    snapshot_spacing_s: float | None = None
    inputs: frozenset[str] = frozenset()

    def document(self) -> dict[str, Any]:
        """The measures as a sidecar holds them (``measure`` names the definition)."""
        e, k = self.entries, self.recoveries
        return dict(
            measure=REGION_MEASURES,
            attempt_id=self.attempt_id,
            profile_id=self.profile_id,
            outcome=self.outcome,
            reason=self.reason,
            snapshots=dict(
                rows=self.snapshot_rows,
                skipped=self.snapshot_skipped,
                invalid=self.snapshot_invalid,
                max_spacing_s=self.snapshot_spacing_s,
            ),
            t90_s={
                str(j): dict(value=t.value, event=t.event, reason=t.reason, share=t.share)
                for j, t in self.times.items()
            },
            entries=None
            if e is None
            else dict(
                counts=None if e.counts is None else {str(j): n for j, n in e.counts.items()},
                order=list(e.order),
                gaps=e.gaps,
                gap_s=e.gap_s,
                samples=e.samples,
                reason=e.reason,
            ),
            recoveries=None
            if k is None
            else dict(
                counts=None if k.counts is None else {str(j): n for j, n in k.counts.items()},
                unplaced=k.unplaced,
                outside=k.outside,
                before_onset=k.before_onset,
                after_end=k.after_end,
                rows=k.rows,
                expected=k.expected,
                rows_mismatch=k.mismatch,
                reason=k.reason,
            ),
        )


def recovery_rows(reads: Reads, run: Row) -> tuple[list[int] | None, str]:
    """The clocks of the run's ``recovery`` rows and why there are none: (None, "not_recorded") below dataset
    schema 11, (None, "no_navigation_stream") at 11 without the stream."""
    schema = run.get("dataset_schema")
    if schema is None or int(schema) < RECOVERY_SCHEMA:
        return None, "not_recorded"
    table = reads.raw(str(run["attempt_id"]), "navigation", ["event", "clock_ns"])
    if table is None:
        return None, "no_navigation_stream"
    clocks = sorted(
        int(r["clock_ns"])
        for r in table.to_pylist()
        if r["event"] == "recovery" and r["clock_ns"] is not None
    )
    return clocks, "observed"


def _in_window(t_s: float | None, end_s: float) -> bool | None:
    """True for a sample in [t0, end_s], False for a pre-motion one, None for one after the window."""
    if t_s is None:
        return False
    return True if -TIME_EPSILON <= t_s <= end_s + TIME_EPSILON else None


def run_regions(export: Export, run: Row, reference: str, limit_s: float) -> RunRegions:
    """``region-measures-v1`` of one reported run on its coverage reference ``reference`` (the provenance's
    ``reference_name``) with the stratum's L; reads through its own ``Reads``, so ``inputs`` are this run's."""
    reads = Reads(export)
    attempt = str(run["attempt_id"])
    # the measurement rule: an explorer failure before movement onset is measured as never moving (RD-8, C1);
    # the run keeps its own outcome label (``label``) for the counts (T12, F27)
    label = str(run["outcome"])
    outcome = "no_movement" if never_moved(run) else label
    profile = str(run["profile_id"])
    raster = regions(reads, reference)
    cells = reference_cells(reads, reference)

    def without(reason: str) -> RunRegions:
        return RunRegions(
            attempt, profile, label, reason, {}, None, None, inputs=frozenset(reads.files)
        )

    if raster is None:
        return without("no_region_raster")
    if cells is None:
        return without("reference_not_readable")
    labels = target_labels(raster, cells)
    if labels is None:
        return without("raster_shape_differs")
    counted = raster.counted
    if not counted:
        return without("no_counted_regions")
    chain = explored_chain(reads, attempt)
    snapshots = () if chain is None else chain.snapshots
    end_s, _ = run_end(run, limit_s)
    moving = outcome != "no_movement"
    complete = chain_complete(chain, end_s if moving else None) and all(
        m.size == labels.size for _, m in snapshots
    )
    if run.get("coverage_eligible") is False:
        times = {j: RegionTime(None, None, "coverage_ineligible", None) for j in sorted(counted)}
    else:
        times = coverage_times(
            labels, counted, snapshots, limit_s=limit_s, outcome=outcome, complete=complete
        )
    poses = base_poses(reads, attempt, None, pre_motion=True) or []
    seen = label_at(raster, [p.x for p in poses], [p.y for p in poses])
    deep = values_at(raster, deep_cells(raster), [p.x for p in poses], [p.y for p in poses], False)
    # a moving run: the window [t0, min(T_end, L)]; one that never moved: its pre-motion samples
    chosen = [i for i, p in enumerate(poses) if _in_window(p.t_s, end_s) == moving]
    if not poses:
        found_entries = RegionEntries(None, (), 0, 0.0, 0, "no_ground_truth")
    else:
        found_entries = entries(
            [poses[i].t_s or 0.0 for i in chosen],
            [int(seen[i]) for i in chosen],
            [bool(deep[i]) for i in chosen],
            counted,
            end_s=end_s,
            outcome=outcome,
        )
    clocks, why = recovery_rows(reads, run)
    if run.get("navigation_eligible") is False:
        found_recoveries = RegionRecoveries(None, reason="navigation_ineligible")
    elif clocks is None:
        found_recoveries = RegionRecoveries(None, reason=why)
    elif not poses:
        found_recoveries = RegionRecoveries(None, rows=len(clocks), reason="no_ground_truth")
    else:
        onset = next(
            (p.clock_ns - round(p.t_s * 1e9) for p in poses if p.t_s is not None and moving), None
        )
        expected = run.get("recoveries")
        found_recoveries = recoveries(
            clocks,
            onset_clock_ns=onset,
            end_s=end_s,
            sample_clocks=[p.clock_ns for p in poses],
            sample_regions=[int(j) for j in seen],
            counted=counted,
            expected=None if expected is None else int(expected),
        )
    return RunRegions(
        attempt,
        profile,
        label,
        "observed",
        times,
        found_entries,
        found_recoveries,
        0 if chain is None else chain.rows,
        0 if chain is None else chain.skipped,
        0 if chain is None else chain.invalid,
        None if chain is None else snapshot_spacing(chain),
        frozenset(reads.files),
    )


def summary(measures: Sequence[RunRegions]) -> dict[str, Any]:
    """Per profile (T12's counts): runs, runs without measures by reason, runs without T90_j, entries or K_j
    by reason, recovery-row mismatches, and per counted region the runs reached and censored, completed runs not
    reached and runs without a value."""
    found: dict[str, Any] = {}
    for profile in sorted({m.profile_id for m in measures}):
        mine = [m for m in measures if m.profile_id == profile]
        regions_found: dict[str, Counter[str]] = {}
        for m in mine:
            for j, t in m.times.items():
                tally = regions_found.setdefault(str(j), Counter())
                tally["runs"] += 1
                if t.event is True:
                    tally["reached"] += 1
                elif t.event is False:
                    tally["censored"] += 1
                    if m.outcome == "completed":
                        tally["completed_not_reached"] += 1
                else:
                    tally[f"null_{t.reason}"] += 1
        found[profile] = dict(
            runs=len(mine),
            without_measures=dict(Counter(m.reason for m in mine if m.reason != "observed")),
            t90_null=dict(
                Counter(
                    next(iter(m.times.values())).reason
                    for m in mine
                    if m.times
                    and next(iter(m.times.values())).reason
                    in ("snapshots_incomplete", "coverage_ineligible")
                )
            ),
            entries_null=dict(
                Counter(
                    m.entries.reason
                    for m in mine
                    if m.entries is not None and m.entries.counts is None
                )
            ),
            recoveries_null=dict(
                Counter(
                    m.recoveries.reason
                    for m in mine
                    if m.recoveries is not None and m.recoveries.counts is None
                )
            ),
            recovery_rows_mismatch=sum(
                1 for m in mine if m.recoveries is not None and m.recoveries.mismatch
            ),
            regions={j: dict(sorted(t.items())) for j, t in sorted(regions_found.items())},
        )
    return found
