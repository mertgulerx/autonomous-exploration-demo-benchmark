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

"""Time series the Results page draws (RESULTS_STUDY.md decisions 4, 10; RESULTS_OVERHAUL R4b).

``timeline`` is one finished run's run-view series on its shared time axis (``results.timeline``): coverage,
goals, explorer dispatches, preemption gate evaluations, the insight analysis (pool, route length, whether the
route's first frontier differs from greedy) and ground-truth and odometry speed. ``traces`` is the coverage of
every reported final run of one study world (``study.traces``), for the interactive raw-line view. Both read
recorded files, so the engine builds them off its loop (``files.BackgroundReads``). Samples are recorded values,
never interpolated or smoothed; a long series keeps every k-th sample so a reply stays bounded, and says so.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

import pyarrow.parquet as pq

from .contracts import GAP_S, ContractError
from .datasets import iter_rows, materialize_manifest
from .monitoring import PathSettings
from .run_record import DERIVED_VERSION, RUN_RECORD_VERSION, time_origins, track
from .studies import held, with_read_time

MAX_POINTS = 7200  # per series of a run view: two hours at 1 Hz
MAX_TRACE_POINTS = 2000  # per run of a traces reply
TRACE_BUDGET = 150_000  # per traces reply, shared by its runs (about 3 MB)
COVERED_SHARE = 0.5  # a speed interval needs valid pose pairs over at least half of its second
MILESTONES = (
    25,
    50,
    75,
    90,
    95,
)  # the first sample at or above each is kept in a thinned coverage series
# Time columns of the derived tables: movement-relative, or from activation for a run that never moved.
DERIVED = {
    "goals": (
        ("goal_sequence", "dispatch_s", "end_s", "status", "dispatch_kind"),
        ("goal_sequence", "t_activation_s", "lifetime_s", "status", "dispatch_kind"),
    ),
    "explorer_decisions": (
        ("time_s", "kind", "time_mapped"),
        ("t_activation_s", "kind", "time_mapped"),
    ),
    "gate_evaluations": (
        ("time_s", "result", "gate_value", "threshold"),
        ("t_activation_s", "result", "gate_value", "threshold"),
    ),
}


def thinned(rows: Sequence[Any], limit: int) -> tuple[list[Any], int]:
    """Every k-th row (the last always kept) so at most about ``limit`` remain, and k."""
    step = max(1, -(-len(rows) // max(1, limit)))
    kept = list(rows[::step])
    if rows and (len(rows) - 1) % step:
        kept.append(rows[-1])
    return kept, step


def thinned_coverage(rows: Sequence[Sequence[Any]], limit: int) -> tuple[list[Any], int]:
    """``thinned`` for a coverage series of (t, value, …) rows, with RD-6's gap rule applied first: between two
    kept rows whose recorded stretch holds a gap over ``GAP_S`` comes a break row (halfway, every value null),
    so a thinned series marks its gaps itself and its readers apply no time rule of their own. The first valid
    row at or above each milestone (the sample a first-reached time names) is kept. k grows until the kept and
    break rows fit ``limit``. Unthinned series come back as they are (k = 1): readers apply the 2 s rule."""
    if len(rows) <= limit:
        return list(rows), 1
    count = len(rows)
    gaps = [0] * count  # gaps[i]: recorded gaps up to row i
    for index in range(1, count):
        t0, t1 = rows[index - 1][0], rows[index][0]
        gaps[index] = gaps[index - 1] + int(t0 is not None and t1 is not None and t1 - t0 > GAP_S)
    milestones = set()
    for share in MILESTONES:
        found = next(
            (
                index
                for index, row in enumerate(rows)
                if row[0] is not None and row[0] >= 0 and row[1] is not None and row[1] >= share
            ),
            None,
        )
        if found is not None:
            milestones.add(found)
    step = -(-count // max(1, limit))
    while True:
        chosen = sorted(set(range(0, count, step)) | {count - 1} | milestones)
        kept: list[Any] = [rows[chosen[0]]]
        for a, b in zip(chosen, chosen[1:]):
            if gaps[b] > gaps[a]:
                t0, t1 = rows[a][0], rows[b][0]
                middle = None if t0 is None or t1 is None else round((t0 + t1) / 2, 3)
                kept.append((middle, *([None] * (len(rows[b]) - 1))))
            kept.append(rows[b])
        if len(kept) <= limit or step >= count:
            return kept, step
        step += max(1, step // 4)


def _relative_s(clock_ns: int | None, origin_ns: int | None) -> float | None:
    """Seconds from the origin, to the millisecond."""
    return None if clock_ns is None or origin_ns is None else round((clock_ns - origin_ns) / 1e9, 3)


def _derived(directory: Path, name: str, moved: bool) -> list[list[Any]] | None:
    """A derived table's rows on the reply's time axis; None until the run record has written it."""
    path = directory / "derived" / f"{name}.v{DERIVED_VERSION}.parquet"
    if not path.is_file():
        return None
    columns = DERIVED[name][0 if moved else 1]
    rows = [list(row.values()) for row in pq.read_table(path, columns=list(columns)).to_pylist()]
    if name == "goals" and not moved:  # end = dispatch + lifetime, both from activation
        for row in rows:
            row[2] = None if row[1] is None or row[2] is None else row[1] + row[2]
    return rows


def _dispatches(directory: Path, moved: bool) -> list[list[Any]] | None:
    """The explorer's dispatches as (time, kind, mapped): the format-1 decision rows (one per dispatch
    line); from log format 2 those rows are decisions, so the dispatches come from the goals table (each
    paired dispatch, by its dispatch kind, at the goal's dispatch time)."""
    path = directory / "derived" / f"explorer_decisions.v{DERIVED_VERSION}.parquet"
    if not path.is_file():
        return None
    formats = pq.read_table(path, columns=["log_format"]).column("log_format").to_pylist()
    goals_path = path.with_name(f"goals.v{DERIVED_VERSION}.parquet")
    if goals_path.is_file():  # a format-2 run may have dispatched without any decision line
        formats += (
            pq.read_table(goals_path, columns=["log_format"]).column("log_format").to_pylist()
        )
    if not any(isinstance(value, int) and value >= 2 for value in formats):
        return _derived(directory, "explorer_decisions", moved)
    goals = _derived(directory, "goals", moved) or []
    return [[row[1], row[4], True] for row in goals if row[1] is not None and row[4] is not None]


def _first_per_second(points: Iterable[tuple[float, Any]]) -> list[tuple[float, Any]]:
    """The first sample of each simulated second (recorded values, no averaging)."""
    kept: list[tuple[float, Any]] = []
    second = None
    for t, value in points:
        if math.floor(t) != second:
            second = math.floor(t)
            kept.append((t, value))
    return kept


def _ground_truth_speed(
    poses: Sequence[tuple[int, float, float, float, float | None]], origin_ns: int | None
) -> list[tuple[float, float | None]]:
    """Ground-truth speed per simulated second: the distance over the time of the consecutive pose pairs no
    further apart than the path's gap (``PathSettings.gap_seconds``), so breaks and dropouts count as missing,
    not as standing still; null when such pairs cover less than half of the second."""
    gap = PathSettings().gap_seconds
    distance: dict[int, float] = defaultdict(float)
    covered: dict[int, float] = defaultdict(float)
    for (c0, x0, y0, _, l0), (c1, x1, y1, _, l1) in zip(poses, poses[1:]):
        t0, t1 = _relative_s(c0, origin_ns), _relative_s(c1, origin_ns)
        if t0 is None or t1 is None or not 0 < t1 - t0 <= gap:
            continue
        step = l1 - l0 if l0 is not None and l1 is not None else math.hypot(x1 - x0, y1 - y0)
        distance[math.floor(t1)] += step
        covered[math.floor(t1)] += t1 - t0
    return [
        (
            float(second),
            round(distance[second] / covered[second], 4)
            if covered[second] >= COVERED_SHARE
            else None,
        )
        for second in sorted(covered)
    ]


def timeline(directory: Path) -> dict[str, Any]:
    """The run view's series of one finished attempt, times in simulated seconds from the time origin.

    The origin is the first movement (``run_record.time_origins``); a run that never moved uses activation
    (``time_origin``). A derived table is null until the run record has written it.
    """
    manifest = materialize_manifest(directory)
    motion_ns, activation_ns = time_origins(manifest)
    origin_ns = motion_ns if motion_ns is not None else activation_ns
    moved = motion_ns is not None
    end_ns = manifest.get("terminal_trigger_simulation_ns")
    coverage = [
        (
            _relative_s(row["clock_ns"], origin_ns),
            row["coverage_percent"] if row["valid"] and row["status"] == "valid" else None,
            row["status"],
        )
        for row in iter_rows(directory, "coverage")
        if row["clock_ns"] is not None
    ]
    analysis = []
    for row in iter_rows(directory, "decisions"):
        if row["clock_ns"] is None:
            continue
        differs = (
            None
            if row["target"] is None or row["greedy_first"] is None
            else row["target"] != row["greedy_first"]
        )
        route = row["route"] or []
        analysis.append(
            (
                _relative_s(row["clock_ns"], origin_ns),
                row["candidates"],
                row["pool"],
                len(route),
                differs,
            )
        )
    poses, source, _ = track(directory, manifest, end_ns)
    odometry = _first_per_second(
        (t, row["speed"])
        for row in iter_rows(directory, "trajectory")
        if (t := _relative_s(row["clock_ns"], origin_ns)) is not None
    )
    series: dict[str, Any] = dict(
        coverage=coverage,
        goals=_derived(directory, "goals", moved),
        dispatches=_dispatches(directory, moved),
        gates=_derived(directory, "gate_evaluations", moved),
        analysis=analysis,
        speed_ground_truth=_ground_truth_speed(poses, origin_ns)
        if source == "ground_truth"
        else [],
        speed_odometry=odometry,
    )
    steps = {}
    series["coverage"], steps["coverage"] = thinned_coverage(series["coverage"], MAX_POINTS)
    for name in ("analysis", "speed_ground_truth", "speed_odometry"):
        series[name], steps[name] = thinned(series[name], MAX_POINTS)
    return dict(
        attempt_id=directory.name,
        time_origin="movement" if moved else "activation",
        end_s=_relative_s(end_ns, origin_ns),
        columns=dict(
            coverage=["t_s", "coverage_pct", "status"],
            goals=["goal_sequence", "dispatch_s", "end_s", "status", "dispatch_kind"],
            dispatches=["time_s", "kind", "time_mapped"],
            gates=["time_s", "result", "gate_value", "threshold"],
            analysis=["t_s", "candidates", "pool", "route_length", "differs_from_greedy"],
            speed_ground_truth=["t_s", "speed_mps"],
            speed_odometry=["t_s", "speed_mps"],
        ),
        every_kth={name: k for name, k in steps.items() if k > 1},
        analysis_recorded=bool(analysis),
        speed_source=source,
        **series,
    )


TRACE_FIELDS = (
    "attempt_id",
    "campaign_id",
    "robot_id",
    "spawn_id",
    "profile_id",
    "parameter_hash",
    "is_baseline",
    "comparison_key",
    "block_id",
    "repetition",
    "outcome",
    "completion_s",
    "completion_event",
    "coverage_final_pct",
    "coverage_at_limit_pct",
    "time_limit_s",
    "pre_motion_coverage_pct",
    "motion_started",
)


def traces(
    database_path: Path, study_id: str, world_id: str, spawn_id: str | None, robot_id: str | None
) -> dict[str, Any]:
    """Coverage of every reported final run of one study world (``study.traces``), movement-relative.

    Runs left out by report class are counted by class (``not_recorded`` for a reported run without a full
    record or files); only the campaigns an export holds are read (``studies.held``). The runs share a point
    budget, so a reply stays within one message; coverage is rounded to 0.001 percentage points.
    """
    with closing(sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)) as database:
        database.execute("BEGIN")
        condition, replaced = held(database)
        rows = database.execute(
            "SELECT r.record,e.final_attempt_id,e.state,a.directory FROM run_records r "
            "JOIN experiments e ON e.id=r.experiment_id JOIN attempts a ON a.id=r.attempt_id "
            "WHERE r.run_record_version=? AND r.study_id=? AND r.world_id=? "
            "AND (? IS NULL OR r.spawn_id=?) AND (? IS NULL OR r.robot_id=?) "
            f"AND e.final_attempt_id IS r.attempt_id AND {condition} "
            "ORDER BY r.robot_id,r.profile_id,r.block_id",
            (
                RUN_RECORD_VERSION,
                study_id,
                world_id,
                spawn_id,
                spawn_id,
                robot_id,
                robot_id,
                *replaced,
            ),
        ).fetchall()
    if not rows:
        raise ContractError("world_id", "No final runs of this study in this world", "not_found")
    chosen: list[tuple[dict[str, Any], Path]] = []
    excluded: dict[str, int] = {}
    for record_text, final, state, directory in rows:
        record = with_read_time(json.loads(record_text), final, state, {})
        if record["report_class"] != "reported":
            excluded[record["report_class"]] = excluded.get(record["report_class"], 0) + 1
        elif record["record_kind"] != "full" or not directory:
            excluded["not_recorded"] = excluded.get("not_recorded", 0) + 1
        else:
            chosen.append((record, Path(directory)))
    limit = min(MAX_TRACE_POINTS, TRACE_BUDGET // max(1, len(chosen)))
    runs = []
    for record, directory in chosen:
        samples = [
            (round(row["simulation_elapsed_ns"] / 1e9, 3), round(row["coverage_percent"], 3))
            for row in iter_rows(directory, "coverage")
            if row["stage"] != "pre_motion"
            and row["simulation_elapsed_ns"] is not None
            and row["valid"]
            and row["status"] == "valid"
        ]
        samples, step = thinned_coverage(samples, limit)
        runs.append(
            dict({name: record[name] for name in TRACE_FIELDS}, samples=samples, every_kth=step)
        )
    return dict(
        study_id=study_id,
        world_id=world_id,
        spawn_id=spawn_id,
        robot_id=robot_id,
        runs=runs,
        excluded=dict(sorted(excluded.items())),
        columns=["t_s", "coverage_pct"],
    )
