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
"Reported population `reported-outcomes-v4`").

- Final runs only. ``reported`` runs (completed, time limit, stuck, no movement, explorer failure; ``stuck`` from
  run record 6, ``reported-outcomes-v3``; ``explorer_failure`` from run record 7, ``reported-outcomes-v4``;
  earlier records ``reported-outcomes-v2``) form the population; the other final classes are counted beside
  every result (missing observations: ``missing_startup``, ``missing_platform``;
  exclusions: ``excluded_runtime_failure``, ``excluded_stopped``).
- A stratum is one robot, world and spawn with one comparison key and one time limit L, holding at least one
  reported run; cells without one are listed (``Study.empty_cells``) and counted in T1, which reads the planned
  experiments. The analysis refuses an export whose final runs mix record, schema, metric, eligibility, key or
  timing-rule versions, a cell with two comparison keys or two time limits, two final runs of one block and
  profile, and a profile ID with two parameter hashes.
- Times are movement-relative simulation seconds. Restricted times R = min(T, L): an event when T ≤ L, else
  R = L; ``no_movement`` counts as never reaching (R = L, RD-8). C(h) and A_h hold the last valid value of a
  completed or ``stuck`` run from its end to L (``carried_forward``, RD-6) and a ``no_movement`` run's
  pre-motion coverage (RD-8), and
  have no value when valid samples are more than 2 s apart before min(h, T_end), the run has no sample at
  movement onset or is not coverage-eligible. D(h) is odometry distance from the export's 1 s trajectory rows,
  moving runs only, with no value when the last row is more than 2 s old.
- The coverage stop ends a run as ``completed``: ``coverage-stop-v3`` 5 simulated seconds after the first valid
  sample at or above 99.9 % (26 October 2026 on), ``coverage-stop-v2`` at that sample (26 October 2026),
  ``coverage-stop-v1`` at 99.0 % (25 October 2026); run records are rebuilt at the current version, so the record
  version does not say which rule a run ran under (its manifest policy does). ``completion_source`` (run record 3 on) says whether the explorer or that stop ended it; both
  are a completion here (R_completion, success) and T1 counts them apart. T99.9 joins the milestones from run
  record 4; run record 3's T99 columns are not read. Earlier records read None for what they lack.
- The stuck stop (``stuck-stop-v1``, run record 6) ends a run whose robot stayed within 0.75 m for 35 simulated
  seconds, or within 1 m for 60 simulated seconds while coverage gained less than 0.1 pp: at or above 97.0 % coverage as ``completed`` with ``completion_source`` ``stuck_stop`` (a completion
  like the others), below it with the outcome ``stuck``. A stuck run is a time-limit run in every estimand
  (success false; completion and unreached milestones censored at L, milestones reached before the stop
  observed; distance, compute and speed observed to its end) except that C(h) and A_h carry its last valid
  sample from its end to L (RD-6), and D(h) holds its final odometry distance there as a completed run's does.
- An explorer failure (run record 7, ``reported-outcomes-v4``, cross-method readiness C1): a final runtime or
  startup failure whose ``failure_source`` is ``explorer`` (the explorer process exited, its own error result, a
  start protocol's refusal) is reported with the outcome ``explorer_failure`` (its recorded outcome kept in
  ``recorded_outcome``); stack-caused failures keep their earlier classes. It is never successful and its
  unreached endpoints are censored at L (reason ``explorer_failure``). One that ended after movement onset is
  treated like ``stuck``: C(h), A_h and D(h) carried from its end to L; one that ended before is treated like
  ``no_movement`` (RD-8): R = L and its pre-motion coverage held. A milestone its record leaves null (a failure
  outcome without milestone fields) is read from its valid samples by the milestone rule (first valid sample at
  or above m, no interpolation; ``milestone_source`` ``samples``).
- Method identity (C2): every profile belongs to a method, its ``explorer_id`` (the ``profiles`` table from export
  4, else its runs' ``explorer_id``); ours is ``frontier_exploration_ros2``, labelled "RADEF" (the paper's name). An export without
  any explorer ID has one method, ours. ``Study.methods`` lists them in registry order (``explorer_order`` from
  export 4, else ``REGISTRY``); a study with more than one method is a method comparison (``cross_method``), whose
  profiles are labelled "method · profile".
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pyarrow as pa

from .errors import Refused
from .export import Export
from .settings import MILESTONE_OF_ENDPOINT, STOP_MILESTONE, milestone_key

Row = dict[str, Any]
Floats = NDArray[np.float64]
GAP_S = 2.0  # RD-6: C(h), A_h and D(h) have no value across a longer gap between valid samples
REPORTED = "reported"
STUCK = "stuck"  # stuck-stop-v1 below 97.0 % coverage (run record 6, reported-outcomes-v3)
# a reported explorer-caused runtime or startup failure (run record 7, reported-outcomes-v4); ``load`` sets it
# as the run's outcome and keeps the recorded one in RECORDED_OUTCOME
EXPLORER_FAILURE = "explorer_failure"
RECORDED_OUTCOME = "recorded_outcome"
FAILURE_SOURCE = "failure_source"
FAILURE_OUTCOMES = ("runtime_failure", "startup_failure")
# the outcomes of reported runs; ``stuck`` from run record 6 (``STUCK_RECORD_VERSION``), ``explorer_failure``
# from run record 7 (``EXPLORER_FAILURE_RECORD_VERSION``)
REPORTED_OUTCOMES = ("completed", "time_limit", STUCK, "no_movement", EXPLORER_FAILURE)
STUCK_RECORD_VERSION = 6
EXPLORER_FAILURE_RECORD_VERSION = 7
EXPLORER_FAILURE_TEXT = (
    "an explorer failure (failure_source explorer: the explorer process exited, returned its own error or refused "
    "the start protocol; reported-outcomes-v4) is not successful and censored at L; after movement onset it is "
    "treated like a stuck run (coverage and distance carried from its end to L), before it like a run without "
    "movement (pre-motion coverage held, R = L)"
)
# Methods (cross-method readiness C2). The explorer registry (exploration_benchmark_runner/config/profiles.yaml)
# by package, in file order: its labels and positions for exports before version 4, which carry neither; kept
# equal to exploration_benchmark_runner/config/profiles.yaml.
OURS = "frontier_exploration_ros2"
# The paper's name of our method (PAPER_GUIDE §0, maintainer 26 October 2026); output texts spell it out.
OURS_LABEL = "RADEF"
REGISTRY: tuple[tuple[str, str], ...] = (
    (OURS, OURS_LABEL),
    ("explore_lite", "m-explore-ros2"),
    ("roadmap_explorer", "roadmap-explorer"),
    ("rrt_explore", "rrt-explore"),
    ("tdle", "TDLE"),
    ("mrtsp_exploration_ros2", "base MRTSP"),
)
PROFILE_COLUMNS = ("profile_id", "profile_name", "robot_id", "parameter_hash", "is_baseline")
# export 3 on: explorer_id; export 4 on: the method's registry label, position, source and build patch
METHOD_COLUMNS = (
    "explorer_id",
    "explorer_label",
    "explorer_order",
    "explorer_source",
    "explorer_build_patch",
)
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
# I10: the mapping condition of a run (run record 2). A run with a comparison key and no condition (run record
# 1, or recorded before mapping-known-pose-v1) mapped under the earlier stock SLAM condition.
MAPPING_CONDITION = "mapping_condition"
BEFORE_MAPPING_CONDITION = "none (before mapping-known-pose-v1)"
# R_<key> reads the run record's t<key>_s, t<key>_event and t<key>_censor_s (key: settings.milestone_key)
TIME_ENDPOINTS = {
    "R_completion": "completion",
    **{endpoint: f"t{milestone_key(m)}" for endpoint, m in MILESTONE_OF_ENDPOINT.items()},
}
# Run record 3 on: what ended a completed run, ``explorer`` (its completion event), ``coverage_stop`` (COVERAGE_STOP_TEXT)
# or, from run record 6, ``stuck_stop`` (stuck-stop-v1 at or above 97.0 % coverage); null for other outcomes. Run record 4
# replaced T99 by T99.9. Records without a column read None for it (no value, ``not_recorded``), and the
# settings leave milestone 99.9 out before run record 4.
COMPLETION_SOURCE = "completion_source"
# The stop rules as the table and figure notes state them (one wording for every output).
COVERAGE_STOP_TEXT = (
    "the coverage stop 5 simulated seconds after coverage first reached 99.9 % (coverage-stop-v3; v2 stopped at "
    "the 99.9 % sample and v1 at 99.0 %, before 26 October 2026; a study may hold runs of several versions)"
)
STUCK_TEXT = (
    "the robot stayed within 0.75 m for 35 simulated seconds, or within 1 m for 60 simulated seconds while "
    "coverage gained less than 0.1 pp (stuck-stop-v1)"
)
COMPLETION_SOURCES = ("explorer", "coverage_stop", "stuck_stop")
OPTIONAL_COLUMNS = (
    COMPLETION_SOURCE,
    FAILURE_SOURCE,
    "explorer_id",
    "explorer_source",
    "explorer_build_patch",  # run record 7
    *(f"t{milestone_key(STOP_MILESTONE)}{suffix}" for suffix in ("_s", "_event", "_censor_s")),
)
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
class Method:
    """One explorer of a study (C2): its ``explorer_id``, label ("RADEF" for ours), registry position (ours 0),
    and the sources and build patches its profiles name (export 4; else its runs' sources)."""

    id: str
    label: str
    order: int
    sources: tuple[str, ...] = ()
    build_patches: tuple[str, ...] = ()

    @property
    def is_ours(self) -> bool:
        return self.id == OURS


def difference_text(study: "Study") -> str:
    """The sign convention of every paired effect in words: "method − RADEF" in a method comparison (C2), else
    "profile − Default"."""
    return f"method − {OURS_LABEL}" if study.cross_method else "profile − Default"


def reference_text(study: "Study") -> str:
    """What every profile is compared with: "reference (RADEF)" in a method comparison (C2), else "Default"."""
    return f"reference ({OURS_LABEL})" if study.cross_method else "Default"


def is_external(run: Row) -> bool:
    """A run of an external method: its explorer log is not parsed and no insight node ran beside it (A6)."""
    return run["explorer_id"] not in (None, OURS)


def never_moved(run: Row) -> bool:
    """A reported run that never moved: ``no_movement``, or an explorer failure before movement onset (RD-8)."""
    outcome = run["outcome"]
    return outcome == "no_movement" or (outcome == EXPLORER_FAILURE and not run["motion_started"])


def stopped_early(run: Row) -> bool:
    """A reported unsuccessful run whose values are carried from its end to L: ``stuck``, or an explorer failure
    after movement onset."""
    outcome = run["outcome"]
    return outcome == STUCK or (outcome == EXPLORER_FAILURE and bool(run["motion_started"]))


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
    # profile_id -> name, parameter hash, baseline flag, robot, and (C2) ``method`` (its explorer_id),
    # ``method_default`` (the method's Default for the robot) and ``label`` (``profile_label``)
    profiles: Mapping[str, Row]
    baselines: Mapping[str, str]  # robot_id -> the Default profile's ID
    coverage: Mapping[str, Series] = field(repr=False)
    pre_motion: Mapping[str, float | None] = field(repr=False)
    trajectory: Mapping[str, tuple[Floats, Floats]] = field(repr=False)  # t, odometry distance
    has_completion_source: bool = False  # the export's runs carry completion_source (run record 3)
    has_stuck: bool = False  # some run ran under the stuck stop (its stuck_rule, run record 6)
    # run record 7 (reported-outcomes-v4): explorer failures are reported
    has_explorer_failure: bool = False
    methods: Mapping[str, Method] = field(
        default_factory=dict
    )  # explorer_id -> method, registry order

    @property
    def finals(self) -> list[Row]:
        return [run for stratum in self.strata for run in stratum.finals]

    @property
    def cross_method(self) -> bool:
        """A method comparison: the study's profiles belong to more than one explorer (C2)."""
        return len(self.methods) > 1

    def method(self, profile_id: str) -> Method | None:
        row = self.profiles.get(profile_id)
        return None if row is None else self.methods.get(str(row["method"]))

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


def mapping_condition(run: Row) -> str | None:
    """The run's mapping condition; None for a run without a stratum (no data of any condition)."""
    if run[MAPPING_CONDITION] is not None:
        return str(run[MAPPING_CONDITION])
    return BEFORE_MAPPING_CONDITION if run["comparison_key"] else None


def load(export: Export) -> Study:
    """The study's runs, strata and series, or a refusal naming what does not fit."""
    table = export.table("runs")
    runs = _rows(table)
    for run in runs:  # exports of run record 1 have no column: those runs predate the condition
        run.setdefault(MAPPING_CONDITION, None)
        for column in OPTIONAL_COLUMNS:  # before run record 3: no coverage stop; before 4: no T99.9
            run.setdefault(column, None)
        run[RECORDED_OUTCOME] = run["outcome"]
        if run["report_class"] == REPORTED and run["outcome"] in FAILURE_OUTCOMES:
            run["outcome"] = EXPLORER_FAILURE  # reported-outcomes-v4: an explorer-caused failure
    finals = [run for run in runs if run["is_final"]]
    if not finals:
        raise Refused("the export holds no final runs")
    conditions = {c for run in finals if (c := mapping_condition(run)) is not None}
    if len(conditions) > 1:
        # one study-wide condition also keeps every stratum and family to one
        raise Refused(
            f"final runs mix mapping conditions {sorted(conditions)}; analyse each separately",
            column=MAPPING_CONDITION,
            groups=_groups(finals, MAPPING_CONDITION),
        )
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
        # an explorer failure before recording began may lack L: it takes the cell's
        found_limits = {
            run["time_limit_s"]
            for run in members
            if run["report_class"] == REPORTED
            and not (run["outcome"] == EXPLORER_FAILURE and run["time_limit_s"] is None)
        }
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
        if not limits and any(
            run["report_class"] == REPORTED and run["outcome"] == EXPLORER_FAILURE
            for run in members
        ):
            # only explorer failures without L (before recording began): the robot and world's L (C1)
            limits = [_failure_limit(export, cells, robot, world, spawn)]
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
    profiles, methods = _profiles(export, finals)
    baselines = {
        row["robot_id"]: profile_id
        for profile_id, row in sorted(profiles.items())
        if row["is_baseline"]
    }
    reported = {run["attempt_id"] for run in finals if run["report_class"] == REPORTED}
    coverage = _coverage(export, reported)
    for run in finals:
        if run["outcome"] == EXPLORER_FAILURE:
            _failure_milestones(run, coverage.get(run["attempt_id"]))
    version = export.study.get("run_record_version")
    return Study(
        export=export,
        runs=tuple(runs),
        strata=tuple(strata),
        empty_cells=tuple(empty),
        profiles=profiles,
        baselines=baselines,
        coverage=coverage,
        pre_motion={run["attempt_id"]: run["pre_motion_coverage_pct"] for run in finals},
        trajectory=_trajectory(export, reported),
        has_completion_source=COMPLETION_SOURCE in table.column_names,
        has_stuck=has_stuck(export.study, table),
        has_explorer_failure=type(version) is int and version >= EXPLORER_FAILURE_RECORD_VERSION,
        methods=methods,
    )


def _failure_limit(
    export: Export,
    cells: Mapping[tuple[str, str, str], list[Row]],
    robot: str,
    world: str,
    spawn: str,
) -> float:
    """L of a cell whose reported runs are all explorer failures without a time limit (C1): the L of the same
    robot and world's other cells (one L by design), else the world's limit in the runs' campaign
    (``world_time_limits``, when it sets no time-limit rule); a refusal naming the cell when neither gives one
    positive L."""
    name = f"{robot}/{world}/{spawn}"
    found = {
        float(run["time_limit_s"])
        for (r, w, _), runs in cells.items()
        if (r, w) == (robot, world)
        for run in runs
        if run["report_class"] == REPORTED and run["time_limit_s"] is not None
    }
    if not found:
        campaigns = {str(run["campaign_id"]) for run in cells[(robot, world, spawn)]}
        for campaign in export.study.get("campaigns") or ():
            if str(campaign.get("id")) in campaigns and not campaign.get("time_limit_rule"):
                value = (campaign.get("world_time_limits") or {}).get(world)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    found.add(float(value))
    if len(found) != 1 or next(iter(found)) <= 0:
        raise Refused(
            f"{name}: its reported runs are explorer failures without a time limit, and neither another "
            f"cell of {robot} in {world} nor the campaign gives one positive L (found {sorted(found)})",
            column="time_limit_s",
            stratum=name,
            groups=_groups(cells[(robot, world, spawn)], "time_limit_s"),
        )
    return next(iter(found))


def _failure_milestones(run: Row, series: Series | None) -> None:
    """An explorer failure's milestones that its record leaves null (a failure outcome has no milestone fields
    before run record 7 computes them): after movement onset, the first valid sample at or above m within
    [0, min(T_end, L)] is the event (no interpolation, the milestone rule), none is censored at L; before it, or
    without a coverage-eligible series, they stay null. Marks the run's ``milestone_source``."""
    run["milestone_source"] = "record"
    if not run["motion_started"] or not run["coverage_eligible"] or series is None:
        return
    limit = run["time_limit_s"]
    until = min((v for v in (run["elapsed_s"], limit) if v is not None), default=math.inf)
    keep = series.t <= until + TIME_EPSILON
    for m in MILESTONE_OF_ENDPOINT.values():
        key = f"t{milestone_key(m)}"
        if f"{key}_event" not in run or run[f"{key}_event"] is not None:
            continue
        reached = np.nonzero(series.coverage[keep] >= m - TIME_EPSILON)[0]
        if reached.size:
            run[f"{key}_s"], run[f"{key}_event"] = float(series.t[keep][reached[0]]), True
            run[f"{key}_censor_s"] = None
        else:
            run[f"{key}_s"], run[f"{key}_event"], run[f"{key}_censor_s"] = None, False, limit
        run["milestone_source"] = "samples"


def _profiles(export: Export, finals: list[Row]) -> tuple[dict[str, Row], dict[str, Method]]:
    """The study's profiles with their method (C2) and the methods in registry order. A profile's method is the
    ``profiles`` table's ``explorer_id`` (export 3 on), else its final runs' one ``explorer_id``; an export without
    any is one method, ours. The label, position, source and build patch come from export 4's columns (null when
    the profile's runs differ), else from ``REGISTRY`` and the runs' ``explorer_source`` and
    ``explorer_build_patch`` (run record 7)."""
    available = set(export.table("profiles").column_names)
    columns = [*PROFILE_COLUMNS, *(c for c in METHOD_COLUMNS if c in available)]
    rows: dict[str, Row] = {}
    for row in _rows(export.table("profiles", columns)):
        rows.setdefault(row["profile_id"], row)
    from_runs: dict[str, set[str]] = defaultdict(set)
    run_sources: dict[str, set[str]] = defaultdict(set)
    run_patches: dict[str, set[str]] = defaultdict(set)
    for run in finals:
        if run["explorer_id"]:
            from_runs[run["profile_id"]].add(str(run["explorer_id"]))
        if run["explorer_source"]:
            run_sources[run["profile_id"]].add(str(run["explorer_source"]))
        if run["explorer_build_patch"]:
            run_patches[run["profile_id"]].add(str(run["explorer_build_patch"]))
    known = dict(REGISTRY)
    order = {identity: index for index, (identity, _) in enumerate(REGISTRY)}
    found: dict[str, dict[str, Any]] = {}
    for profile_id, row in rows.items():
        explorer = row.get("explorer_id") or (
            next(iter(from_runs[profile_id])) if len(from_runs[profile_id]) == 1 else None
        )
        method = str(explorer) if explorer else OURS
        row["method"] = method
        entry = found.setdefault(
            method, dict(labels=set(), orders=set(), sources=set(), patches=set())
        )
        if row.get("explorer_label"):
            entry["labels"].add(str(row["explorer_label"]))
        if type(row.get("explorer_order")) is int:
            entry["orders"].add(int(row["explorer_order"]))
        source = row.get("explorer_source")
        entry["sources"] |= {str(source)} if source else run_sources[profile_id]
        patch = row.get("explorer_build_patch")
        entry["patches"] |= {str(patch)} if patch else run_patches[profile_id]
    unknown = sorted(m for m in found if m not in order and not found[m]["orders"])
    methods = sorted(
        (
            Method(
                id=method,
                label=OURS_LABEL
                if method == OURS
                else min(entry["labels"], default=known.get(method, method)),
                order=0
                if method == OURS
                else min(entry["orders"])
                if entry["orders"]
                else order.get(method, len(REGISTRY) + unknown.index(method)),
                sources=tuple(sorted(entry["sources"])),
                build_patches=tuple(sorted(entry["patches"])),
            )
            for method, entry in found.items()
        ),
        key=lambda m: (m.order, m.id),
    )
    by_id = {m.id: m for m in methods}
    cross = len(by_id) > 1
    for profile_id, row in rows.items():
        owner = by_id[row["method"]]
        name = str(row["profile_name"] or profile_id)
        short = name.removeprefix(f"{owner.label} ") if not owner.is_ours else name
        row["method_default"] = bool(row["is_baseline"]) or (
            not owner.is_ours and (profile_id.startswith("default-") or short == "Default")
        )
        row["label"] = f"{owner.label} · {short}" if cross else name
    return rows, by_id


def has_stuck(study: Mapping[str, Any], runs: pa.Table) -> bool:
    """Whether any run of the export ran under the stuck stop: run record 6 and later name each run's
    ``stuck_rule`` (records are rebuilt at the current version, so the version alone does not say it)."""
    version = study.get("run_record_version")
    if (
        type(version) is not int
        or version < STUCK_RECORD_VERSION
        or "stuck_rule" not in runs.column_names
    ):
        return False
    return any(isinstance(rule, str) and rule for rule in runs.column("stuck_rule").to_pylist())


def outcome_rule(study: Study) -> str:
    """The reported-population rule of the export's runs, for notes."""
    if study.has_explorer_failure:
        return "reported-outcomes-v4"
    return "reported-outcomes-v3" if study.has_stuck else "reported-outcomes-v2"


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
    """R = min(T, L) of a reported run for ``R_completion`` or ``R_m`` (RD-7, RD-8, RD-10); a ``stuck`` run as
    a time-limit run: its event-free endpoints are censored at L (reason ``stuck``, the export's end reason); an
    explorer failure never completes (censored at L, reason ``explorer_failure``), and one before movement onset
    never reaches anything (R = L, as ``no_movement``)."""
    failure = run["outcome"] == EXPLORER_FAILURE
    if never_moved(run):
        return Value(limit, EXPLORER_FAILURE if failure else "no_movement", False)
    prefix = TIME_ENDPOINTS[endpoint]
    if prefix == "completion" and run["outcome"] == "completed" and not run["motion_started"]:
        return Value(None, "completion_without_motion")
    if failure and prefix == "completion":
        return Value(limit, EXPLORER_FAILURE, False)
    event = run[f"{prefix}_event"]
    if event:
        time = run[f"{prefix}_s"]
        return Value(min(time, limit), "observed", time <= limit + TIME_EPSILON)
    if event is False:
        below = run["outcome"] == "completed" and prefix != "completion" and run["completion_event"]
        reason = (
            "completed_below_m"
            if below
            else run["outcome"]
            if run["outcome"] in (STUCK, EXPLORER_FAILURE)
            else "time_limit"
        )
        return Value(limit, reason, False)
    return Value(None, "coverage_ineligible" if not run["coverage_eligible"] else "not_recorded")


def run_end(run: Row, limit: float) -> tuple[float, bool]:
    """The run's end on the movement clock within [0, L] and whether its values are carried to L: a run that
    completed within L (RD-6), a ``stuck`` run, whose stuck stop ended it before L (stuck-stop-v1), or an
    explorer failure after movement onset (C1)."""
    if stopped_early(run):
        carried = run["elapsed_s"] is not None
    else:
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
    """C(h): the last valid sample at or before h ≤ L, carried forward after a completion, a stuck stop or an
    explorer failure after movement onset (RD-6, C1); a run that never moved holds its pre-motion coverage."""
    if not run["coverage_eligible"]:
        return Value(None, "coverage_ineligible")
    if never_moved(run):
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
    """A_h = (1/h)∫₀ʰ C(t) dt with left-hold steps over the actual sample times, carried forward after a
    completion or a stuck stop; the same gap, onset and eligibility rules as C(h)."""
    at = coverage_at(study, run, horizon, limit)
    if at.value is None or at.reason == "pre_motion":
        return at
    series = study.coverage[run["attempt_id"]]
    end, carried = run_end(run, limit)
    until = min(horizon, end)
    keep = series.t <= until + TIME_EPSILON
    t, c = series.t[keep], series.coverage[keep]
    area = float(np.sum(c * np.diff(np.concatenate([t, [until]]))))
    if horizon > until:  # completed or stuck before h: the final value holds to h
        area += float(c[-1]) * (horizon - until)
    return Value(area / horizon, "carried_forward" if carried and horizon > end else "observed")


def distance_at(study: Study, run: Row, horizon: float, limit: float) -> Value:
    """D(h): odometry distance at min(h, T_end) for runs that moved (RD-8), from the export's 1 s rows; a
    completed or stuck run holds its final distance; no value when the last row is more than ``GAP_S`` old."""
    if never_moved(run) or not run["motion_started"]:
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
    if never_moved(run) or not run["motion_started"]:
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
    age), ``carried_forward`` (a completed or stuck run's, or an explorer failure's after movement, final value),
    ``pre_motion`` (a run that never moved), or no
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
                elif never_moved(run):
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
