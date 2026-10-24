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

"""A study's runs as the analysis sees them: strata, populations and the per-run values of every estimand
(docs/research/METHODOLOGY.md §3; RESULTS_RESEARCH RD-6 to RD-9, RD-18; docs/MEASUREMENTS.md "Run record" and
"Reported population `reported-outcomes-v2`").

- Final runs only. ``reported`` runs (completed, time limit, no movement) form the population; the other final
  classes are counted beside every result (missing observations: ``missing_startup``, ``missing_platform``;
  exclusions: ``excluded_runtime_failure``, ``excluded_stopped``).
- A stratum is one robot, world and spawn with one comparison key and one time limit L, holding at least one
  reported run; cells without one are listed (``Study.empty_cells``) and counted in T1, which reads the planned
  experiments. The analysis refuses an export whose final runs mix record, schema, metric, eligibility, key or
  timing-rule versions, a cell with two comparison keys or two time limits, two final runs of one block and
  profile, and a profile ID with two parameter hashes.
- Times are movement-relative simulation seconds. Restricted times R = min(T, L): an event when T ≤ L, else
  R = L; ``no_movement`` counts as never reaching (R = L, RD-8). C(h) and A_h hold a completed run's final value
  from its end to L (``carried_forward``, RD-6) and a ``no_movement`` run's pre-motion coverage (RD-8), and
  have no value when valid samples are more than 2 s apart before min(h, T_end), the run has no sample at
  movement onset or is not coverage-eligible. D(h) is odometry distance from the export's 1 s trajectory rows,
  moving runs only, with no value when the last row is more than 2 s old.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pyarrow as pa

from .errors import Refused
from .export import Export

Row = dict[str, Any]
Floats = NDArray[np.float64]
GAP_S = 2.0  # RD-6: C(h), A_h and D(h) have no value across a longer gap between valid samples
REPORTED = "reported"
REPORTED_OUTCOMES = ("completed", "time_limit", "no_movement")  # the outcomes of reported runs
MISSING = ("missing_startup", "missing_platform")
EXCLUDED = ("excluded_runtime_failure", "excluded_stopped")
VERSION_COLUMNS = (
    "run_record_version",
    "dataset_schema",
    "coverage_metric_version",
    "eligibility_version",
    "comparison_key_version",
    "timing_rule",
)
TIME_ENDPOINTS = {
    "R_completion": "completion",
    "R_25": "t25",
    "R_50": "t50",
    "R_75": "t75",
    "R_90": "t90",
    "R_95": "t95",
}
TIME_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class Value:
    """One run's value of an estimand, or why it has none (``value`` None)."""

    value: float | None
    reason: str  # "observed", "carried_forward", "pre_motion" or the reason for no value
    event: bool | None = None  # restricted times: the event was observed by L


@dataclass(frozen=True, slots=True)
class Series:
    """Valid coverage samples of one run (stages ``baseline`` and ``active``), movement-relative."""

    t: Floats
    coverage: Floats


@dataclass(frozen=True, slots=True)
class Stratum:
    robot_id: str
    world_id: str
    spawn_id: str
    comparison_key: str
    limit_s: float
    finals: tuple[Row, ...]  # every final run of the cell

    @property
    def reported(self) -> tuple[Row, ...]:
        return tuple(run for run in self.finals if run["report_class"] == REPORTED)

    def profiles(self) -> list[str]:
        return sorted({run["profile_id"] for run in self.finals})

    def runs_of(self, profile_id: str, population: Iterable[str] = (REPORTED,)) -> list[Row]:
        classes = set(population)
        return [
            run
            for run in self.finals
            if run["profile_id"] == profile_id and run["report_class"] in classes
        ]

    def blocks(self, profile_id: str, baseline_id: str) -> list[tuple[str, Row | None, Row | None]]:
        """(block, profile run, baseline run) for every block holding either profile's final run."""
        by_block: dict[str, dict[str, Row]] = defaultdict(dict)
        for run in self.finals:
            if run["profile_id"] in (profile_id, baseline_id):
                by_block[run["block_id"]][run["profile_id"]] = run
        return [
            (block, runs.get(profile_id), runs.get(baseline_id))
            for block, runs in sorted(by_block.items())
        ]

    def counts(self, profile_id: str) -> dict[str, int]:
        """Final runs of a profile by report class (the counts printed beside every result)."""
        counts: dict[str, int] = defaultdict(int)
        for run in self.finals:
            if run["profile_id"] == profile_id:
                counts[run["report_class"]] += 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True, slots=True)
class Study:
    export: Export
    runs: tuple[Row, ...]  # every attempt
    strata: tuple[Stratum, ...]
    # robot, world, spawn with final runs but none reported
    empty_cells: tuple[tuple[str, str, str], ...]
    profiles: Mapping[str, Row]  # profile_id -> name, parameter hash, baseline flag, robot
    baselines: Mapping[str, str]  # robot_id -> the Default profile's ID
    coverage: Mapping[str, Series] = field(repr=False)
    pre_motion: Mapping[str, float | None] = field(repr=False)
    trajectory: Mapping[str, tuple[Floats, Floats]] = field(repr=False)  # t, odometry distance

    @property
    def finals(self) -> list[Row]:
        return [run for stratum in self.strata for run in stratum.finals]

    def label(self, robot: str, world: str, spawn: str, limit: float | None = None) -> str:
        """The one label of a stratum in every table and figure: the robot when the study has more than
        one, the world, the spawn unless it is the world's default, and L when given."""
        parts = [robot] if len({s.robot_id for s in self.strata}) > 1 else []
        parts.append(world)
        if spawn != f"default-{world}":
            parts.append(spawn)
        text = " · ".join(parts)
        return text if limit is None else f"{text} (L = {limit:g} s)"

    def stratum_label(self, stratum: Stratum, *, limit: bool = False) -> str:
        return self.label(
            stratum.robot_id, stratum.world_id, stratum.spawn_id, stratum.limit_s if limit else None
        )


def _rows(table: pa.Table) -> list[Row]:
    return table.to_pylist()  # type: ignore[no-any-return]


def _groups(runs: Iterable[Row], column: str) -> list[dict[str, Any]]:
    """The runs of a refusal by the value of ``column``: each value with its runs (RO-22)."""
    found: dict[str, list[Row]] = defaultdict(list)
    for run in runs:
        found[str(run[column])].append(run)
    return [
        dict(
            value=value,
            runs=[
                dict(
                    attempt_id=run["attempt_id"],
                    campaign_id=run["campaign_id"],
                    block_id=run["block_id"],
                    profile_id=run["profile_id"],
                )
                for run in sorted(members, key=lambda r: (r["campaign_id"], r["attempt_id"]))
            ],
        )
        for value, members in sorted(found.items())
    ]


def load(export: Export) -> Study:
    """The study's runs, strata and series, or a refusal naming what does not fit."""
    runs = _rows(export.table("runs"))
    finals = [run for run in runs if run["is_final"]]
    if not finals:
        raise Refused("the export holds no final runs")
    for column in VERSION_COLUMNS:
        # a run without the value (a startup failure before recording began) holds no data of that version
        found = sorted({str(run[column]) for run in finals if run[column] is not None})
        if len(found) > 1:
            raise Refused(
                f"final runs mix {column} values {found}; analyse each separately",
                column=column,
                groups=_groups(finals, column),
            )
    hashes: dict[str, set[str]] = defaultdict(set)
    for run in finals:
        hashes[run["profile_id"]].add(run["parameter_hash"])
    for profile, found_hashes in sorted(hashes.items()):
        if len(found_hashes) > 1:
            raise Refused(
                f"profile {profile} ran with {len(found_hashes)} parameter sets: its runs are not one "
                "condition",
                column="parameter_hash",
                groups=_groups([r for r in finals if r["profile_id"] == profile], "parameter_hash"),
            )
    cells: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for run in finals:
        cells[(run["robot_id"], run["world_id"], run["spawn_id"])].append(run)
    strata, empty = [], []
    for (robot, world, spawn), members in sorted(cells.items()):
        name = f"{robot}/{world}/{spawn}"
        keys = sorted({run["comparison_key"] for run in members if run["comparison_key"]})
        found_limits = {run["time_limit_s"] for run in members if run["report_class"] == REPORTED}
        if None in found_limits:
            raise Refused(
                f"{name}: a reported run has no time limit; restricted estimands need L",
                column="time_limit_s",
                stratum=name,
                groups=_groups(members, "time_limit_s"),
            )
        limits = sorted(found_limits)
        if len(keys) > 1:
            raise Refused(
                f"{name} holds {len(keys)} comparison keys "
                f"({', '.join(k[:12] for k in keys)}): its runs are not comparable",
                column="comparison_key",
                stratum=name,
                groups=_groups(members, "comparison_key"),
            )
        if len(limits) > 1 or (limits and (limits[0] is None or limits[0] <= 0)):
            raise Refused(
                f"{name}: one positive time limit L is needed, found {limits}",
                column="time_limit_s",
                stratum=name,
                groups=_groups(members, "time_limit_s"),
            )
        seen: set[tuple[str, str]] = set()
        for run in members:
            pair = (run["block_id"], run["profile_id"])
            if pair in seen:
                raise Refused(
                    f"{name}: two final runs of block {pair[0]} and {pair[1]}",
                    column="block_id",
                    stratum=name,
                    groups=_groups(
                        [r for r in members if (r["block_id"], r["profile_id"]) == pair], "block_id"
                    ),
                )
            seen.add(pair)
        if not limits:
            empty.append((robot, world, spawn))
            continue
        strata.append(
            Stratum(robot, world, spawn, keys[0] if keys else "", float(limits[0]), tuple(members))
        )
    if not strata:  # nothing to estimate or draw: every final run failed or was stopped
        raise Refused(
            "no reported run in any stratum: no final run is reported (their classes in groups)",
            column="report_class",
            groups=_groups(finals, "report_class"),
        )
    profiles: dict[str, Row] = {}
    for row in _rows(
        export.table(
            "profiles", ["profile_id", "profile_name", "robot_id", "parameter_hash", "is_baseline"]
        )
    ):
        profiles.setdefault(row["profile_id"], row)
    baselines = {
        row["robot_id"]: profile_id
        for profile_id, row in sorted(profiles.items())
        if row["is_baseline"]
    }
    reported = {run["attempt_id"] for run in finals if run["report_class"] == REPORTED}
    return Study(
        export=export,
        runs=tuple(runs),
        strata=tuple(strata),
        empty_cells=tuple(empty),
        profiles=profiles,
        baselines=baselines,
        coverage=_coverage(export, reported),
        pre_motion={run["attempt_id"]: run["pre_motion_coverage_pct"] for run in finals},
        trajectory=_trajectory(export, reported),
    )


def _coverage(export: Export, attempts: set[str]) -> dict[str, Series]:
    table = export.table("coverage", ["attempt_id", "stage", "t_s", "coverage_pct", "status"])
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for attempt, stage, t, value, status in zip(
        *(table.column(i).to_pylist() for i in range(5)), strict=True
    ):
        if attempt in attempts and stage in ("baseline", "active") and status == "valid":
            if t is not None and value is not None:
                grouped[attempt].append((t, value))
    series = {}
    for attempt, samples in grouped.items():
        samples.sort()
        series[attempt] = Series(
            np.array([t for t, _ in samples], dtype=np.float64),
            np.array([v for _, v in samples], dtype=np.float64),
        )
    return series


def _trajectory(export: Export, attempts: set[str]) -> dict[str, tuple[Floats, Floats]]:
    table = export.table("trajectory_gt", ["attempt_id", "t_s", "odom_distance_m"])
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for attempt, t, distance in zip(*(table.column(i).to_pylist() for i in range(3)), strict=True):
        if attempt in attempts and t is not None and distance is not None:
            grouped[attempt].append((t, distance))
    return {
        attempt: (
            np.array([t for t, _ in sorted(rows)], dtype=np.float64),
            np.array([d for _, d in sorted(rows)], dtype=np.float64),
        )
        for attempt, rows in grouped.items()
    }


def block_starts(study: Study) -> dict[tuple[str, str, str, str], int]:
    """The first attempt start (wall clock, ns) of every block of every cell, keyed by robot, world, spawn and
    block: the execution order of blocks (F10's columns, F23's example runs). Blocks never started are absent."""
    started: dict[tuple[str, str, str, str], int] = {}
    for attempt in study.runs:
        key = (attempt["robot_id"], attempt["world_id"], attempt["spawn_id"], attempt["block_id"])
        start = attempt["wall_start_ns"]
        if start is not None:
            started[key] = min(started.get(key, start), start)
    return started


# Per-run values ---------------------------------------------------------------------------------------------
# Every function takes the stratum's L, so a run whose own record lacks it is still measured on its stratum.


def restricted_time(run: Row, endpoint: str, limit: float) -> Value:
    """R = min(T, L) of a reported run for ``R_completion`` or ``R_m`` (RD-7, RD-8, RD-10)."""
    if run["outcome"] == "no_movement":
        return Value(limit, "no_movement", False)
    prefix = TIME_ENDPOINTS[endpoint]
    if prefix == "completion" and run["outcome"] == "completed" and not run["motion_started"]:
        return Value(None, "completion_without_motion")
    event = run[f"{prefix}_event"]
    if event:
        time = run[f"{prefix}_s"]
        return Value(min(time, limit), "observed", time <= limit + TIME_EPSILON)
    if event is False:
        below = run["outcome"] == "completed" and prefix != "completion" and run["completion_event"]
        return Value(limit, "completed_below_m" if below else "time_limit", False)
    return Value(None, "coverage_ineligible" if not run["coverage_eligible"] else "not_recorded")


def run_end(run: Row, limit: float) -> tuple[float, bool]:
    """The run's end on the movement clock within [0, L] and whether its values are carried to L (a run that
    completed within L; RD-6)."""
    carried = run["outcome"] == "completed" and bool(run["completion_event"])
    return (min(run["elapsed_s"], limit) if carried else limit), carried


def _first_gap(times: Floats, end: float) -> float | None:
    """The start of the first stretch longer than ``GAP_S`` without a valid sample on [0, end], if any."""
    edges = np.concatenate([[0.0], times[times <= end + TIME_EPSILON], [end]])
    long = np.nonzero(np.diff(edges) > GAP_S + TIME_EPSILON)[0]
    return float(edges[long[0]]) if long.size else None


def _usable(study: Study, run: Row) -> Series | str:
    """The run's coverage series, or why C(h) and A_h have no value for it."""
    if not run["coverage_eligible"]:
        return "coverage_ineligible"
    series = study.coverage.get(run["attempt_id"])
    if series is None or series.t.size == 0 or series.t[0] > TIME_EPSILON:
        return "no_sample_at_onset"
    return series


def coverage_at(study: Study, run: Row, horizon: float, limit: float) -> Value:
    """C(h): the last valid sample at or before h ≤ L, carried forward after completion (RD-6)."""
    if not run["coverage_eligible"]:
        return Value(None, "coverage_ineligible")
    if run["outcome"] == "no_movement":
        held = study.pre_motion.get(run["attempt_id"])
        return (
            Value(held, "pre_motion") if held is not None else Value(None, "no_pre_motion_sample")
        )
    series = _usable(study, run)
    if isinstance(series, str):
        return Value(None, series)
    end, carried = run_end(run, limit)
    until = min(horizon, end)
    if _first_gap(series.t, until) is not None:
        return Value(None, "gap")
    index = int(np.searchsorted(series.t, until + TIME_EPSILON, side="right")) - 1
    return Value(
        float(series.coverage[index]),
        "carried_forward" if carried and horizon > end else "observed",
    )


def coverage_auc(study: Study, run: Row, horizon: float, limit: float) -> Value:
    """A_h = (1/h)∫₀ʰ C(t) dt with left-hold steps over the actual sample times, carried forward after
    completion; the same gap, onset and eligibility rules as C(h)."""
    at = coverage_at(study, run, horizon, limit)
    if at.value is None or at.reason == "pre_motion":
        return at
    series = study.coverage[run["attempt_id"]]
    end, carried = run_end(run, limit)
    until = min(horizon, end)
    keep = series.t <= until + TIME_EPSILON
    t, c = series.t[keep], series.coverage[keep]
    area = float(np.sum(c * np.diff(np.concatenate([t, [until]]))))
    if horizon > until:  # completed before h: the final value holds to h
        area += float(c[-1]) * (horizon - until)
    return Value(area / horizon, "carried_forward" if carried and horizon > end else "observed")


def distance_at(study: Study, run: Row, horizon: float, limit: float) -> Value:
    """D(h): odometry distance at min(h, T_end) for runs that moved (RD-8), from the export's 1 s rows; a
    completed run holds its final distance; no value when the last row is more than ``GAP_S`` old."""
    if run["outcome"] == "no_movement" or not run["motion_started"]:
        return Value(None, "no_movement")
    if not run["motion_eligible"]:
        return Value(None, "motion_ineligible")
    end, carried = run_end(run, limit)
    if carried and horizon >= end and run["odom_distance_m"] is not None:
        return Value(
            float(run["odom_distance_m"]), "carried_forward" if horizon > end else "observed"
        )
    rows = study.trajectory.get(run["attempt_id"])
    if rows is None:
        return Value(None, "no_trajectory")
    until = min(horizon, end)
    index = int(np.searchsorted(rows[0], until + TIME_EPSILON, side="right")) - 1
    if index < 0:
        return Value(None, "no_trajectory")
    if until - float(rows[0][index]) > GAP_S + TIME_EPSILON:
        return Value(None, "gap")
    return Value(float(rows[1][index]), "observed")


def idle_tail(study: Study, run: Row, limit: float, delta_pp: float) -> Value:
    """``stall-v1`` with the plan's δ (RD-18): t_g, the first time the running maximum of valid coverage
    comes within δ of its final value (the peak on [0, min(T_end, L)]), and the idle tail min(T_end, L) − t_g;
    as the run record computes it for δ = 0.1 pp."""
    if run["outcome"] == "no_movement" or not run["motion_started"]:
        return Value(None, "no_movement")
    series = _usable(study, run)
    if isinstance(series, str):
        return Value(None, series)
    end, _ = run_end(run, limit)
    keep = series.t <= end + TIME_EPSILON
    t, c = series.t[keep], series.coverage[keep]
    if t.size == 0:
        return Value(None, "no_sample")
    reached = np.nonzero(np.maximum.accumulate(c) >= c.max() - delta_pp - TIME_EPSILON)[0]
    return Value(end - float(t[reached[0]]), "observed")


def coverage_grid(study: Study) -> pa.Table:
    """``coverage_grid.parquet``: every reported run on a 1 s grid from 0 to L (RESULTS_OVERHAUL §7) with the
    rule each value follows. ``source``: ``sample`` (the last valid sample at or before t; ``sample_age_s`` its
    age), ``carried_forward`` (a completed run's final value), ``pre_motion`` (a run that never moved), or no
    value: ``gap`` (once a stretch without a sample has lasted more than 2 s, as C(h) has no value from then
    on; nothing bridges it), ``coverage_ineligible``, ``no_sample_at_onset``. ``observed`` is true for
    ``sample`` values."""
    columns: dict[str, list[Any]] = defaultdict(list)
    for stratum in study.strata:
        for run in sorted(stratum.reported, key=lambda r: r["attempt_id"]):
            limit = stratum.limit_s
            grid = np.arange(0.0, np.floor(limit + TIME_EPSILON) + 1.0)
            end, carried = run_end(run, limit)
            held = study.pre_motion.get(run["attempt_id"])
            series = _usable(study, run)
            gap = None if isinstance(series, str) else _first_gap(series.t, end)
            for t in grid:
                value: float | None = None
                age: float | None = None
                if not run["coverage_eligible"]:
                    source: str = "coverage_ineligible"
                elif run["outcome"] == "no_movement":
                    source = "pre_motion" if held is not None else "no_pre_motion_sample"
                    value = held
                elif isinstance(series, str):
                    source = series
                elif gap is not None and t > gap + GAP_S + TIME_EPSILON:
                    source = "gap"
                else:
                    until = min(float(t), end)
                    index = int(np.searchsorted(series.t, until + TIME_EPSILON, side="right")) - 1
                    value = float(series.coverage[index])
                    if carried and t > end:
                        source = "carried_forward"
                    else:
                        source, age = "sample", round(until - float(series.t[index]), 6)
                columns["attempt_id"].append(run["attempt_id"])
                columns["t_s"].append(float(t))
                columns["coverage_pct"].append(value)
                columns["source"].append(source)
                columns["carried_forward"].append(source == "carried_forward")
                columns["observed"].append(source == "sample")
                columns["sample_age_s"].append(age)
    schema = pa.schema(
        [
            ("attempt_id", pa.string()),
            ("t_s", pa.float64()),
            ("coverage_pct", pa.float64()),
            ("source", pa.string()),
            ("carried_forward", pa.bool_()),
            ("observed", pa.bool_()),
            ("sample_age_s", pa.float64()),
        ]
    )
    return pa.table({name: columns[name] for name in schema.names}, schema=schema)
