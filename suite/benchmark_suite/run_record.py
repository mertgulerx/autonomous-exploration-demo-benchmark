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

"""The run record ``run-record-v2``: its field registry and the facts measured from an attempt's files.

Contract: docs/MEASUREMENTS.md, "Run record run-record-v2". ``measure`` runs in a spawned job (``jobs.py``)
and reads the manifest, the streams and the explorer log; ``run_records.RunRecords`` adds the engine's
facts and stores the record once. Read-time fields (finality, report class) are derived from the store
by ``read_time_fields``. Null never means zero.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any, Literal, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import REPORTED_OUTCOMES, average_speed, content_hash
from .datasets import files, iter_rows, materialize_manifest, recorded_version
from .explorer_log import (
    ClockMap,
    Structured,
    cancels_by_dispatch,
    decision_rows,
    gate_rows,
    goals_by_dispatch,
    match,
    parse,
    read_text,
)
from .goal_facts import PREEMPTED, GoalAccumulator
from .monitoring import PathSettings, base_pose, path_break
from .run_map import GROUND_TRUTH_OFFSET, finite, triple
from .storage import file_hash, load_object, sync_directory
from .strata import COMPARISON_KEY_VERSION, comparison_key

# 2 (I10): adds mapping_condition. Version 1 records stay beside the rebuilt ones (run_records.py).
RUN_RECORD_VERSION = 2
ELIGIBILITY_VERSION = "eligibility-v2"
DERIVED_VERSION = 1
# Goal statuses the explorer itself causes (a cancel request or a replacement goal).
ENDED_BY_EXPLORER = frozenset({"canceled", "preempted"})
MILESTONES = (25, 50, 75, 90, 95)
STALL_DELTA_PP = 0.1  # stall-v1 (RESULTS_RESEARCH RD-18)
MOTION_GAP_NS = 500_000_000  # eligibility-v2: largest interval between valid trajectory rows
RESOURCE_CHAIN_NS = 2_500_000_000  # RD-20: valid resource samples at most 2.5 s apart
# Where every field is defined; the study export's codebook cites it (R4).
DEFINITION = "docs/MEASUREMENTS.md, Results overhaul contracts, Run record run-record-v2"
Kind = Literal["text", "int", "float", "bool", "list"]


@dataclass(frozen=True, slots=True)
class Field:
    """One record field: value kind, unit, what null means, and where it is defined."""

    name: str
    kind: Kind
    unit: str = ""
    null: str = "never"
    definition: str = DEFINITION


def _fields(kind: Kind, unit: str, null: str, *names: str) -> tuple[Field, ...]:
    return tuple(Field(name, kind, unit, null) for name in names)


FIELDS: tuple[Field, ...] = (
    *_fields("int", "", "never", "run_record_version"),
    *_fields("text", "", "never", "record_kind"),
    *_fields("int", "", "stub", "dataset_schema", "coverage_metric_version"),
    # The mapping condition id (comparison_profile.mapping_condition, I10); null: a stub, or a run
    # recorded before mapping-known-pose-v1 (the stock SLAM condition), or no robot configuration.
    *_fields("text", "", "stub, or recorded before I10", "mapping_condition"),
    *_fields("text", "", "stub", "timing_rule"),
    *_fields("text", "", "never", "eligibility_version"),
    *_fields("int", "", "never", "comparison_key_version"),
    *_fields("text", "", "never", "study_id", "evidence_level", "campaign_id", "experiment_id"),
    *_fields("text", "", "never", "attempt_id"),
    *_fields("int", "", "never", "attempt_number"),
    *_fields("text", "", "outcome not manual_discard", "discard_trigger"),
    *_fields("text", "", "never", "robot_id", "world_id", "spawn_id"),
    *_fields("int", "", "never", "repetition"),
    *_fields("text", "", "never", "block_id"),
    *_fields("text", "", "named world", "world_family"),
    *_fields("text", "sha256", "stub", "comparison_key"),
    *_fields("text", "", "never", "profile_id"),
    *_fields("int", "", "revision not recorded", "profile_revision"),
    *_fields("text", "sha256", "stub", "parameter_hash"),
    *_fields("bool", "", "stub or no Default hash recorded", "is_baseline"),
    *_fields("text", "", "never", "explorer_id"),
    *_fields("text", "", "runtime image unlabelled", "explorer_version"),
    *_fields("int", "", "never", "campaign_position", "block_position"),
    *_fields("int", "UTC ns", "attempt never started", "wall_start_ns", "wall_end_ns"),
    *_fields("int", "", "never", "worker_slot"),
    *_fields("int", "", "recorded before suite image 17", "runs_since_engine_start"),
    *_fields("text", "", "never", "outcome"),
    *_fields("text", "", "never", "outcome_detail"),
    *_fields("text", "", "outcome not a failure", "outcome_detail_code"),
    *_fields("float", "sim s", "limit disabled", "time_limit_s"),
    *_fields("float", "sim s", "never moved or stub", "elapsed_s", "restricted_elapsed_s"),
    *_fields("bool", "", "stub", "motion_started"),
    *_fields("float", "sim s", "never moved", "activation_to_movement_s"),
    *_fields("float", "sim s", "moved", "activation_to_movement_censor_s"),
    *_fields("float", "%", "no valid pre-motion sample", "pre_motion_coverage_pct"),
    *_fields("float", "m/s", "as Average speed", "average_speed_mps"),
    *_fields("float", "sim s", "not an observed completion", "completion_s"),
    *_fields("bool", "", "competing outcome", "completion_event"),
    *_fields("float", "sim s", "not censored", "completion_censor_s"),
    *(Field(f"t{m}_s", "float", "sim s", "not an observed event") for m in MILESTONES),
    *(
        Field(f"t{m}_event", "bool", "", "not recorded, not eligible, or competing")
        for m in MILESTONES
    ),
    *(Field(f"t{m}_censor_s", "float", "sim s", "not censored") for m in MILESTONES),
    *_fields("float", "%", "coverage not valid", "coverage_final_pct"),
    *_fields("float", "m²", "coverage not valid", "covered_m2", "reference_free_m2"),
    *_fields("bool", "", "stub", "coverage_valid"),
    *_fields("float", "%", "no limit, invalid or no sample", "coverage_at_limit_pct"),
    *_fields("float", "sim s", "no sample time", "coverage_at_limit_sample_s"),
    *_fields("text", "", "no value", "coverage_at_limit_source"),
    *_fields("float", "sim s", "fewer than two valid samples", "coverage_max_gap_s"),
    *_fields("float", "sim s", "never moved or no valid coverage", "last_gain_s", "idle_tail_s"),
    *_fields("float", "%", "never moved or no valid coverage", "last_gain_coverage_pct"),
    *_fields("float", "m", "no valid odometry", "odom_distance_m"),
    *_fields("float", "m", "diagnostics off or no rows", "gt_path_length_m", "gt_drift_final_m"),
    *_fields(
        "int",
        "",
        "navigation telemetry absent",
        "goals_sent",
        "goals_succeeded",
        "goals_canceled",
        "goals_failed",
        "goals_preempted",
        "goals_open",
        "goals_rejected",
        "recoveries",
    ),
    *_fields("float", "CPU s", "no valid sample", "explorer_cpu_s"),
    *_fields("float", "% of one core", "no valid sample pair", "explorer_cpu_tw_pct"),
    *_fields("float", "MiB", "no valid sample", "explorer_rss_peak_mib"),
    *_fields("bool", "", "stub", "resource_valid"),
    *_fields("float", "CPU s", "no sample in that scope", "cpu_s_instrumentation", "cpu_s_insight"),
    *_fields("float", "ratio", "never moved or no window", "rtf_mean", "rtf_min_5s", "rtf_p05_5s"),
    *_fields("float", "wall s", "never moved or no window", "wall_s_below_095", "wall_s_below_090"),
    *_fields("bool", "", "stub", "low_rtf"),
    *_fields("int", "", "wall times null", "parallel_max"),
    *_fields("float", "0-1", "wall times null", "concurrent_wall_fraction"),
    *_fields("bool", "", "stub", "diagnostics_enabled"),
    *_fields("bool", "", "schema below 9", "insight_started"),
    *_fields("bool", "", "schema below 10 or node not started", "insight_exited_before_stop"),
    *_fields("list", "", "stub", "debug_streams", "quality_flags"),
    *_fields(
        "bool",
        "",
        "stub",
        "timing_eligible",
        "coverage_eligible",
        "motion_eligible",
        "navigation_eligible",
        "event_clock_valid",
    ),
    *_fields("text", "", "stub", "explorer_exit_status"),
    *_fields("int", "", "no exit code", "explorer_exit_code"),
    *_fields("bool", "", "no exit line", "explorer_exited_before_stop"),
    *_fields("int", "", "parser not run (stub)", "goal_match_mismatches", "log_unmapped_lines"),
    *_fields("text", "sha256", "recorded before suite image 17", "provenance_id"),
    # Reserved by RD-51; never measured since decision 59 (no moving-object measures).
    *_fields("int", "", "not measured", "contact_events", "ghost_cells"),
    *_fields("float", "m", "not measured", "moving_object_min_distance_m"),
    *_fields("float", "sim s", "not measured", "moving_object_time_within_0_5_m_s"),
    *_fields("int", "UTC ns", "never", "recorded_ns"),
)
NAMES = tuple(field.name for field in FIELDS)
# Derived when a record is read (``read_time_fields``), never stored; the export adds them as columns.
READ_TIME_FIELDS: tuple[Field, ...] = (
    Field("is_final", "bool"),
    Field("replaced_reason", "text", "", "final attempt, or not replaced"),
    Field("report_class", "text"),
    Field("superseded_by_campaign_id", "text", "", "not superseded; always null in a study export"),
)
# Typed columns of ``run_records`` beside the canonical JSON record (RO-1; store.py).
INDEX_COLUMNS = (
    "record_kind",
    "study_id",
    "evidence_level",
    "campaign_id",
    "experiment_id",
    "attempt_number",
    "outcome",
    "robot_id",
    "world_id",
    "spawn_id",
    "profile_id",
    "repetition",
    "block_id",
    "parameter_hash",
    "comparison_key",
    "time_limit_s",
    "provenance_id",
    "recorded_ns",
)
TYPES: dict[str, tuple[type, ...]] = {
    "text": (str,),
    "int": (int,),
    "float": (float, int),
    "bool": (bool,),
    "list": (list,),
}
# Failures whose detail text maps to outcome_detail_code (RD-26).
FAILURES = frozenset({"startup_failure", "runtime_failure", "storage_failure"})


def validate(record: dict[str, Any]) -> None:
    """Refuse a record whose fields or value types differ from the registry."""
    if tuple(record) != NAMES:
        missing, extra = set(NAMES) - set(record), set(record) - set(NAMES)
        raise ValueError(
            f"Run record fields differ: missing {sorted(missing)}, extra {sorted(extra)}"
        )
    for field in FIELDS:
        value = record[field.name]
        if value is None:
            if field.null == "never":
                raise ValueError(f"Run record field {field.name} cannot be null")
            continue
        if field.kind != "bool" and isinstance(value, bool):
            raise ValueError(f"Run record field {field.name} must be {field.kind}")
        if not isinstance(value, TYPES[field.kind]):
            raise ValueError(f"Run record field {field.name} must be {field.kind}")
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"Run record field {field.name} must be finite or null")


def ordered(record: dict[str, Any]) -> dict[str, Any]:
    """The record in registry order; refuses a missing or unknown field (a misspelt key never becomes
    a silent null). Stored records are canonical JSON with sorted keys."""
    missing, extra = set(NAMES) - set(record), set(record) - set(NAMES)
    if missing or extra:
        raise ValueError(
            f"Run record fields differ: missing {sorted(missing)}, extra {sorted(extra)}"
        )
    return {name: record[name] for name in NAMES}


def detail_code(outcome: str, detail: str) -> str | None:
    """The closed failure cause of an outcome's detail text (RD-26)."""
    if outcome not in FAILURES:
        return None
    text = detail.lower()
    if outcome == "storage_failure":
        return "storage"
    if outcome == "startup_failure":
        return "startup_timeout" if "timed out" in text or "timeout" in text else "startup_error"
    if text.startswith("explorer exited"):
        return "explorer_exited"
    if text.startswith("simulation component exited"):
        return "simulation_exited"
    if text.startswith("required stream stopped"):
        return "stream_stopped"
    return "other"


def report_class(
    is_final: bool,
    outcome: str | None,
    insight_started: bool | None,
    insight_exited_before_stop: bool | None,
    experiment_state: str | None,
) -> str:
    """``reported-outcomes-v2`` (RD-9, RD-17): rows tried in the contract's order."""
    if not is_final:
        return "not_final"
    if insight_started is False or insight_exited_before_stop is True:
        return "missing_platform"
    if outcome == "startup_failure":
        return "missing_startup"
    if outcome == "storage_failure" or (
        outcome == "timing_rejection" and experiment_state == "UNRESOLVED"
    ):
        return "missing_platform"
    if outcome == "runtime_failure":
        return "excluded_runtime_failure"
    if outcome == "manual_stop":
        return "excluded_stopped"
    if outcome in REPORTED_OUTCOMES:
        return "reported"
    return "missing_platform"


def read_time_fields(
    record: dict[str, Any], experiment: dict[str, Any], superseded_by: str | None
) -> dict[str, Any]:
    """Finality, replacement, report class and supersession of a stored record (never stored). A timing
    rejection or an interruption is replaced only when a later attempt of its experiment exists
    (``experiment["last_attempt_number"]``, as the export's accounting counts it); one still waiting has none."""
    is_final = experiment.get("final_attempt_id") == record["attempt_id"]
    outcome = record["outcome"]
    last = experiment.get("last_attempt_number")
    replaced = None
    if not is_final and (
        outcome == "manual_discard"
        or (
            outcome in ("timing_rejection", "engine_interruption")
            and last is not None
            and record["attempt_number"] < last
        )
    ):
        replaced = outcome
    return dict(
        is_final=is_final,
        replaced_reason=replaced,
        report_class=report_class(
            is_final,
            outcome,
            record["insight_started"],
            record["insight_exited_before_stop"],
            experiment.get("state"),
        ),
        superseded_by_campaign_id=superseded_by,
    )


class Censored(NamedTuple):
    """A time-to-event value: observed time, event flag and censor time (MEASUREMENTS)."""

    time_s: float | None
    event: bool | None
    censor_s: float | None


NOT_APPLICABLE = Censored(None, None, None)


def completion(outcome: str, elapsed: float | None, limit: float | None) -> Censored:
    """Completion (RD-7, RD-10): an event only when observed within L; competing outcomes have none."""
    if outcome == "completed" and elapsed is None:
        return Censored(
            None, False, None
        )  # completion without movement onset, flagged by the caller
    if outcome == "completed" and (limit is None or elapsed is not None and elapsed <= limit):
        return Censored(elapsed, True, None)
    if outcome in ("completed", "time_limit"):
        return Censored(None, False, limit)
    return NOT_APPLICABLE


def milestone(
    item: dict[str, Any] | None,
    outcome: str,
    eligible: bool,
    limit: float | None,
    elapsed: float | None,
) -> Censored:
    """A coverage milestone from its recorded entry and ``reached`` flag (RD-7)."""
    if item is None or not eligible or outcome not in ("completed", "time_limit"):
        return NOT_APPLICABLE
    reached = finite(item.get("simulation_elapsed")) if item.get("reached") else None
    if reached is not None and (limit is None or reached <= limit):
        return Censored(reached, True, None)
    return Censored(None, False, limit if limit is not None else elapsed)


def coverage_at_limit(
    outcome: str,
    elapsed: float | None,
    limit: float | None,
    final_pct: float | None,
    pre_motion_pct: float | None,
    sample: tuple[float, float] | None,
) -> tuple[float | None, float | None, str | None]:
    """C(L) with its sample time and source (RO-2); defined only for the reported outcomes."""
    if limit is None:
        return None, None, None
    if outcome == "no_movement":
        return (
            (pre_motion_pct, None, "pre_motion")
            if pre_motion_pct is not None
            else (None, None, None)
        )
    if (
        outcome == "completed"
        and elapsed is not None
        and elapsed <= limit
        and final_pct is not None
    ):
        return final_pct, elapsed, "carried_forward"
    if outcome in ("completed", "time_limit") and sample is not None:
        return sample[1], sample[0], "sample"
    return None, None, None


def time_origins(manifest: dict[str, Any]) -> tuple[int | None, int | None]:
    """The run's time origins in simulation ns: first movement (None unless the run moved; the origin of
    every movement-relative time) and activation."""
    moved = bool((manifest.get("final_metrics") or {}).get("motion_started"))
    origin = finite((manifest.get("first_motion") or {}).get("simulation"))
    activation = finite(manifest.get("simulation_start"))
    return (
        round(origin * 1e9) if origin is not None and moved else None,
        round(activation * 1e9) if activation is not None else None,
    )


def _seconds(nanoseconds: int | None, origin_ns: int | None) -> float | None:
    if nanoseconds is None or origin_ns is None:
        return None
    return (nanoseconds - origin_ns) / 1e9


def _coverage(
    rows: Sequence[dict[str, Any]], limit_s: float | None, end_s: float | None
) -> dict[str, Any]:
    """Coverage facts from the recorded samples: pre-motion value, gaps, C(L) sample, stall-v1."""
    pre = [r for r in rows if r["stage"] == "pre_motion" and r["valid"] and r["status"] == "valid"]
    moving = sorted(
        (r["simulation_elapsed_ns"] / 1e9, float(r["coverage_percent"]))
        for r in rows
        if r["stage"] in ("baseline", "active")
        and r["valid"]
        and r["status"] == "valid"
        and r["simulation_elapsed_ns"] is not None
        and finite(r["coverage_percent"]) is not None
    )
    bounds = [bound for bound in (limit_s, end_s) if bound is not None]
    horizon = min(bounds) if bounds else None
    window = [(t, c) for t, c in moving if horizon is None or t <= horizon]
    gaps = [b[0] - a[0] for a, b in zip(window, window[1:], strict=False)]
    before_limit = [(t, c) for t, c in moving if limit_s is not None and t <= limit_s]
    stall: tuple[float | None, float | None] = (None, None)
    if window:
        peak = max(c for _, c in window)
        running = -math.inf
        for t, c in window:
            running = max(running, c)
            if running >= peak - STALL_DELTA_PP:
                stall = (t, running)
                break
    return dict(
        pre_motion=float(pre[-1]["coverage_percent"]) if pre else None,
        max_gap=max(gaps) if gaps else None,
        at_limit=before_limit[-1] if before_limit else None,
        last_gain=stall,
        horizon=horizon,
    )


def track(
    directory: Path, manifest: dict[str, Any], end_ns: int | None
) -> tuple[list[tuple[int, float, float, float, float | None]], str | None, float | None]:
    """Robot poses (clock, x, y, yaw, cumulative path length) in the world to the stop trigger.

    Ground truth when recorded, else odometry through the frozen alignment (no path length then). The
    ground-truth length sums the steps the displayed path keeps (``monitoring.path_break``); it is null
    when the robot's speed limit is unknown. Returns the poses, their source and the total length.
    """
    settings = PathSettings()
    maximum = finite((manifest.get("robot") or {}).get("max_linear_speed"))
    measured = maximum is not None and maximum > 0
    fallback = triple((manifest.get("ground_truth_alignment") or {}).get("model_to_base"))
    poses: list[tuple[int, float, float, float, float | None]] = []
    length = 0.0
    previous: tuple[int, float, float] | None = None
    for row in iter_rows(directory, "ground_truth"):
        clock = row["clock_ns"]
        if clock is not None and end_ns is not None and clock > end_ns:
            break
        offset = triple([row.get(key) for key in GROUND_TRUTH_OFFSET]) or fallback
        model = triple([row.get(key) for key in ("x", "y", "yaw")]) if row["valid"] else None
        if clock is None or offset is None or model is None:
            previous = None
            continue
        x, y, yaw = base_pose(model, offset)
        if (
            previous is not None
            and maximum is not None
            and measured
            and path_break(previous, clock, x, y, maximum, settings) is None
        ):
            length += math.hypot(x - previous[1], y - previous[2])
        previous = (clock, x, y)
        poses.append((clock, x, y, yaw, length if measured else None))
    if poses:
        return poses, "ground_truth", length if measured else None
    odometry: list[tuple[int, float, float, float, float | None]] = []
    for row in iter_rows(directory, "trajectory"):
        clock = row["clock_ns"]
        pose = triple(
            [
                row.get(key)
                for key in ("odom_aligned_x_m", "odom_aligned_y_m", "odom_aligned_yaw_rad")
            ]
        )
        if row["valid"] and clock is not None and (end_ns is None or clock <= end_ns) and pose:
            odometry.append((clock, pose[0], pose[1], pose[2], None))
    return odometry, ("odometry" if odometry else None), None


def _resources(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """RD-20 explorer resources and the cumulative CPU of the worker's other scopes."""
    explorer: list[tuple[int, float]] = []
    peak: float | None = None
    other: dict[str, float] = {}
    for row in sorted(rows, key=lambda r: r["receipt_monotonic_ns"]):
        cpu = finite(row.get("cpu_seconds"))
        if row.get("scope") == "explorer":
            if cpu is not None:
                explorer.append((row["receipt_monotonic_ns"], cpu))
            rss = finite(row.get("rss_mib"))
            if rss is not None and rss > 0:
                peak = rss if peak is None else max(peak, rss)
        elif cpu is not None and row.get("scope"):
            other[row["scope"]] = cpu
    pairs = [
        (b[0] - a[0], b[1] - a[1])
        for a, b in zip(explorer, explorer[1:], strict=False)
        if 0 < b[0] - a[0] <= RESOURCE_CHAIN_NS and b[1] >= a[1]
    ]
    wall = sum(p[0] for p in pairs)
    return dict(
        explorer_cpu_s=explorer[-1][1] if explorer else None,
        explorer_cpu_tw_pct=100 * sum(p[1] for p in pairs) / (wall / 1e9) if wall else None,
        explorer_rss_peak_mib=peak,
        resource_valid=bool(pairs),
        cpu_s_instrumentation=other.get("instrumentation_and_recording"),
        cpu_s_insight=other.get("insight"),
    )


def _rtf(
    rows: Iterable[dict[str, Any]], manifest: dict[str, Any], origin_ns: int | None
) -> dict[str, Any]:
    """Run-level RTF from the recorder's windows and the onset and stop clocks (RD-20)."""
    windows = sorted(
        (r["receipt_monotonic_ns"], float(r["value"]))
        for r in rows
        if r.get("observation") == "rtf_window"
        and finite(r.get("value")) is not None
        and r["clock_ns"] is not None
        and (origin_ns is None or r["clock_ns"] >= origin_ns)
    )
    values = np.array([value for _, value in windows], dtype=np.float64)

    def below(threshold: float) -> float | None:
        if not windows:
            return None
        return float(
            sum(
                (b[0] - a[0]) / 1e9
                for a, b in zip(windows, windows[1:], strict=False)
                if a[1] < threshold
            )
        )

    first = manifest.get("first_motion") or {}
    mean = None
    if origin_ns is not None and first.get("monotonic_ns") is not None:
        wall = manifest.get("terminal_trigger_monotonic_ns", 0) - first["monotonic_ns"]
        simulation = manifest.get("terminal_trigger_simulation_ns", 0) - origin_ns
        mean = simulation / wall if wall > 0 and simulation >= 0 else None
    return dict(
        rtf_mean=mean,
        rtf_min_5s=float(values.min()) if values.size else None,
        rtf_p05_5s=float(np.percentile(values, 5)) if values.size else None,
        wall_s_below_095=below(0.95) if origin_ns is not None else None,
        wall_s_below_090=below(0.90) if origin_ns is not None else None,
    )


def _motion_gaps(directory: Path, origin_ns: int | None, end_ns: int | None) -> int:
    """Invalid rows and intervals above 0.5 s between valid trajectory rows from onset to the end."""
    gaps, previous = 0, None
    for row in iter_rows(directory, "trajectory"):
        clock = row["clock_ns"]
        if clock is None or origin_ns is None or clock < origin_ns:
            continue
        if end_ns is not None and clock > end_ns:
            break
        if not row["valid"]:
            gaps += 1
            previous = None
            continue
        if previous is not None and (clock <= previous or clock - previous > MOTION_GAP_NS):
            gaps += 1
        previous = clock
    return gaps


def _goals(
    directory: Path, version: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, dict[str, Any]]]:
    """Goal facts (recorded ``goal_facts``, or replayed for schema 9), telemetry counts and goal poses."""
    navigation = sorted(
        iter_rows(directory, "navigation"),
        key=lambda r: (r["receipt_monotonic_ns"], r["event_sequence"] or 0),
    )
    initialized = {r["goal_sequence"]: r for r in navigation if r["event"] == "initialized"}
    rejected = [
        (r["goal_sequence"], r["clock_ns"])
        for r in navigation
        if r["event"] in ("rejected", "preempt_rejected")
    ]
    counts = dict(rejected=rejected, present=bool(navigation))
    if version >= 10:
        facts = list(iter_rows(directory, "goal_facts"))
    else:
        accumulator, ended = GoalAccumulator(), []
        for row in navigation:
            if row["event"] == "initialized":
                accumulator.started(row["goal_sequence"], row["clock_ns"])
            elif row["event"] in ("terminal", PREEMPTED):
                status = row["status"] if row["event"] == "terminal" else PREEMPTED
                goal = accumulator.ended(row["goal_sequence"], status)
                if goal is not None:
                    ended.append(goal.fields(row["clock_ns"]))
            elif row["event"] == "feedback":
                distance = finite(row["distance_remaining"])
                accumulator.feedback(
                    row["native_uuid"] or "",
                    math.nan if distance is None else distance,
                    row["recoveries"] or 0,
                )
        facts = ended + [goal.fields(None) for goal in accumulator.close()]
    return sorted(facts, key=lambda f: f["goal_sequence"]), counts, initialized


def _write_table(
    directory: Path, name: str, rows: list[dict[str, Any]], schema: pa.Schema
) -> dict[str, Any]:
    """Write one derived table atomically and describe it for the artifact index."""
    destination = directory / "derived"
    destination.mkdir(exist_ok=True)
    final = destination / f"{name}.v{DERIVED_VERSION}.parquet"
    staging = final.with_suffix(".staging")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), staging, compression="zstd")
    with staging.open("rb") as handle:
        os.fsync(handle.fileno())
    staging.replace(final)
    sync_directory(destination)
    return dict(
        kind="derived_table",
        table=name,
        path=str(final),
        sha256=file_hash(final),
        bytes=final.stat().st_size,
        rows=len(rows),
    )


GOALS_SCHEMA = pa.schema(
    [
        ("goal_sequence", pa.int64()),
        ("dispatch_s", pa.float64()),
        ("end_s", pa.float64()),
        ("t_activation_s", pa.float64()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("yaw", pa.float64()),
        ("frame", pa.string()),
        ("status", pa.string()),
        ("lifetime_s", pa.float64()),
        ("recoveries", pa.int32()),
        ("min_distance_m", pa.float64()),
        ("displacement_m", pa.float64()),
        ("pose_source", pa.string()),
        ("dispatch_kind", pa.string()),
        ("selection_mode", pa.string()),
        ("frontier_x", pa.float64()),
        ("frontier_y", pa.float64()),
        ("dispatch_x", pa.float64()),
        ("dispatch_y", pa.float64()),
        ("frontier_size", pa.int32()),
        ("cancel_reason", pa.string()),
        ("gate_value", pa.float64()),
        ("log_format", pa.int32()),
    ]
)
DECISIONS_SCHEMA = pa.schema(
    [
        ("time_s", pa.float64()),
        ("t_activation_s", pa.float64()),
        ("time_mapped", pa.bool_()),
        ("utc_ns", pa.int64()),
        ("kind", pa.string()),
        ("selection_mode", pa.string()),
        ("candidates", pa.int32()),
        ("pool_size", pa.int32()),
        ("route_length", pa.int32()),
        ("search_ms", pa.float64()),
        ("ordering_ms", pa.float64()),
        ("decision_map_cache_hits", pa.int64()),
        ("decision_map_cache_misses", pa.int64()),
        ("frontier_cache_hits", pa.int64()),
        ("frontier_cache_misses", pa.int64()),
        ("order_cache_hits", pa.int64()),
        ("order_cache_misses", pa.int64()),
        ("log_format", pa.int32()),
    ]
)
GATES_SCHEMA = pa.schema(
    [
        ("time_s", pa.float64()),
        ("t_activation_s", pa.float64()),
        ("time_mapped", pa.bool_()),
        ("goal_sequence", pa.int64()),
        ("gate_value", pa.float64()),
        ("threshold", pa.float64()),
        ("result", pa.string()),
        ("log_format", pa.int32()),
    ]
)


def _near(
    positions: Sequence[tuple[int, float, float, float, float | None]],
    start_ns: int | None,
    end_ns: int | None,
) -> Iterator[tuple[int, float, float, float, float | None]]:
    for position in positions:
        if start_ns is not None and position[0] < start_ns:
            continue
        if end_ns is not None and position[0] > end_ns:
            break
        yield position


def _goal_rows(
    facts: Sequence[dict[str, Any]],
    initialized: dict[int, dict[str, Any]],
    pairs: dict[int, Any],
    positions: Sequence[tuple[int, float, float, float, float | None]],
    pose_source: str | None,
    manifest: dict[str, Any],
    clocks: tuple[int | None, int | None, int | None],
    log_format: int,
    cancels: dict[int, Structured],
) -> list[dict[str, Any]]:
    """The ``goals`` table (RD-28): measured goal facts with the explorer's dispatch fields and, from log
    format 2, for a goal the explorer ended (canceled or preempted), the reason of its last cancel or
    replace line with that line's gate value."""
    origin_ns, activation_ns, end_ns = clocks
    alignment = triple(
        ((manifest.get("coverage") or {}).get("alignment") or {}).get("world_from_map")
    )
    rows = []
    for fact in facts:
        sequence = fact["goal_sequence"]
        pose = initialized.get(sequence) or {}
        start, end = fact["dispatch_clock_ns"], fact["end_clock_ns"] or end_ns
        goal: tuple[float, float] | None = None
        x, y = finite(pose.get("x")), finite(pose.get("y"))
        if x is not None and y is not None and pose.get("frame") == "map" and alignment:
            ax, ay, angle = alignment
            goal = (
                ax + math.cos(angle) * x - math.sin(angle) * y,
                ay + math.sin(angle) * x + math.cos(angle) * y,
            )
        near = list(_near(positions, start, end))
        dispatch = pairs.get(sequence)
        # the explorer's reason, only for a goal its cancel or replacement ended (not one Nav2 finished)
        ended = (
            cancels.get(dispatch.dispatch_id)
            if dispatch and dispatch.dispatch_id is not None and fact["status"] in ENDED_BY_EXPLORER
            else None
        )
        rows.append(
            dict(
                goal_sequence=sequence,
                dispatch_s=_seconds(start, origin_ns),
                end_s=_seconds(fact["end_clock_ns"], origin_ns),
                t_activation_s=_seconds(start, activation_ns),
                x=x,
                y=y,
                yaw=finite(pose.get("yaw")),
                frame=pose.get("frame"),
                status=fact["status"],
                lifetime_s=_seconds(fact["end_clock_ns"], start),
                recoveries=fact["recoveries"],
                # Only a goal placed in the world frame is compared with world positions.
                min_distance_m=min(math.hypot(p[1] - goal[0], p[2] - goal[1]) for p in near)
                if near and goal is not None
                else None,
                displacement_m=math.hypot(near[-1][1] - near[0][1], near[-1][2] - near[0][2])
                if near
                else None,
                pose_source=pose_source if near else None,
                dispatch_kind=dispatch.kind if dispatch else None,
                selection_mode=dispatch.mode if dispatch else None,
                frontier_x=dispatch.frontier_x if dispatch else None,
                frontier_y=dispatch.frontier_y if dispatch else None,
                dispatch_x=dispatch.dispatch_x if dispatch else None,
                dispatch_y=dispatch.dispatch_y if dispatch else None,
                frontier_size=dispatch.frontier_size if dispatch else None,
                cancel_reason=ended.fields.get("reason") if ended else None,
                gate_value=ended.number("gate_value") if ended else None,
                log_format=log_format,
            )
        )
    return rows


def measure(payload: dict[str, Any]) -> dict[str, Any]:
    """The record fields measured from one attempt's files, and its derived tables (job side).

    ``payload`` names the workspace, the attempt and its outcome as the store holds it, which wins over
    the manifest's (recovery may have changed it).
    """
    started = time.process_time()
    workspace = Path(payload["workspace"])
    directory = workspace / "attempts" / payload["attempt_id"]
    manifest = materialize_manifest(directory)
    version = (
        recorded_version(directory)
        if files(directory, "events")
        else int(manifest.get("schema_version") or 9)
    )
    metrics = manifest.get("final_metrics") or {}
    policy = manifest.get("policy") or {}
    limit = finite(policy.get("simulation_time_limit_seconds")) or None  # 0 disables the limit
    moved = bool(metrics.get("motion_started"))
    origin_ns, activation_ns = time_origins(manifest)
    end_ns = manifest.get("terminal_trigger_simulation_ns")
    elapsed = finite(metrics.get("simulation_elapsed")) if moved else None
    outcome = payload.get("outcome") or manifest.get("outcome") or ""
    coverage_valid = metrics.get("coverage_status") == "valid"
    flags = list(metrics.get("coverage_quality_flags") or [])
    eligibility = manifest.get("eligibility") or {}
    coverage_eligible = (
        int(manifest.get("recording_dropped") or 0) == 0
        and coverage_valid
        and not [flag for flag in flags if flag != "coverage_motion_not_observed"]
    )
    quality_flags = set(metrics.get("quality_flags") or [])
    coverage = _coverage(list(iter_rows(directory, "coverage")), limit, elapsed)
    deadline = finite(policy.get("no_movement_simulation_seconds"))
    waited = finite(metrics.get("pre_motion_simulation_elapsed"))
    finished = completion(outcome, elapsed, limit)
    if outcome == "completed" and elapsed is None:
        quality_flags.add("completion_without_motion")
    profile = (manifest.get("robot_configuration") or {}).get("comparison_profile") or {}
    condition = profile.get("mapping_condition")
    record: dict[str, Any] = dict(
        dataset_schema=version,
        coverage_metric_version=(manifest.get("coverage") or {}).get("metric_version"),
        mapping_condition=condition if isinstance(condition, str) else None,
        timing_rule=manifest.get("timing_rule")
        or ("timing-rule-v2" if version >= 10 else "timing-rule-v1"),
        time_limit_s=limit,
        elapsed_s=elapsed,
        restricted_elapsed_s=min(elapsed, limit)
        if elapsed is not None and limit is not None
        else elapsed,
        motion_started=moved,
        activation_to_movement_s=waited if moved else None,
        activation_to_movement_censor_s=min(deadline, waited)
        if not moved and deadline is not None and waited is not None
        else None,
        pre_motion_coverage_pct=coverage["pre_motion"],
        average_speed_mps=average_speed(metrics.get("distance_m"), elapsed),
        completion_s=finished.time_s,
        completion_event=finished.event,
        completion_censor_s=finished.censor_s,
    )
    milestones = {
        int(item["percent"]): item
        for item in (manifest.get("coverage") or {}).get("milestones", [])
    }
    for percent in MILESTONES:
        value = milestone(milestones.get(percent), outcome, coverage_eligible, limit, elapsed)
        record.update(
            {
                f"t{percent}_s": value.time_s,
                f"t{percent}_event": value.event,
                f"t{percent}_censor_s": value.censor_s,
            }
        )
    final_pct = finite(metrics.get("coverage_percent")) if coverage_valid else None
    at_limit = (
        coverage_at_limit(
            outcome, elapsed, limit, final_pct, coverage["pre_motion"], coverage["at_limit"]
        )
        if coverage_valid
        else (None, None, None)
    )
    last_gain, gain_coverage = coverage["last_gain"] if moved else (None, None)
    record.update(
        coverage_final_pct=final_pct,
        covered_m2=finite(metrics.get("coverage_covered_m2")) if coverage_valid else None,
        reference_free_m2=finite(metrics.get("coverage_target_m2")) if coverage_valid else None,
        coverage_valid=coverage_valid,
        coverage_at_limit_pct=at_limit[0],
        coverage_at_limit_sample_s=at_limit[1],
        coverage_at_limit_source=at_limit[2],
        coverage_max_gap_s=coverage["max_gap"],
        last_gain_s=last_gain,
        idle_tail_s=coverage["horizon"] - last_gain
        if last_gain is not None and coverage["horizon"] is not None
        else None,
        last_gain_coverage_pct=gain_coverage,
        odom_distance_m=finite(metrics.get("distance_m")),
    )
    positions, pose_source, path_length = track(directory, manifest, end_ns)
    drift = [
        value
        for value in (
            finite(r.get("odometry_drift_m")) for r in iter_rows(directory, "ground_truth")
        )
        if value is not None
    ]
    record.update(gt_path_length_m=path_length, gt_drift_final_m=drift[-1] if drift else None)
    facts, telemetry, initialized = _goals(directory, version)
    statuses = [fact["status"] for fact in facts]
    present = telemetry["present"]
    counted = {
        name: statuses.count(name)
        for name in ("succeeded", "canceled", "failed", PREEMPTED, "open")
    }
    if present and sum(counted.values()) != len(initialized):
        quality_flags.add(
            "goal_counts_inconsistent"
        )  # RD-21's identity does not hold (lost events)
    record.update(
        goals_sent=len(initialized) if present else None,
        goals_succeeded=counted["succeeded"] if present else None,
        goals_canceled=counted["canceled"] if present else None,
        goals_failed=counted["failed"] if present else None,
        goals_preempted=counted[PREEMPTED] if present else None,
        goals_open=counted["open"] if present else None,
        goals_rejected=len(telemetry["rejected"]) if present else None,
        recoveries=sum(fact["recoveries"] or 0 for fact in facts) if present else None,
    )
    record.update(_resources(iter_rows(directory, "resources")))
    quality = list(iter_rows(directory, "quality"))
    record.update(_rtf(quality, manifest, origin_ns))
    record["low_rtf"] = "low_rtf" in quality_flags
    run_facts = manifest.get("run_facts") or {}
    insight = manifest.get("insight")
    record.update(
        diagnostics_enabled=bool((manifest.get("configuration") or {}).get("suite_diagnostics")),
        insight_started=bool(insight.get("started")) if isinstance(insight, dict) else None,
        insight_exited_before_stop=run_facts.get("insight_exited_before_stop"),
        # Before schema 10 feedback and commands were always recorded: no switch existed.
        debug_streams=list(manifest.get("debug_streams") or []) if version >= 10 else None,
        timing_eligible=bool(eligibility.get("timing")),
        coverage_eligible=coverage_eligible,
        motion_eligible=bool(eligibility.get("motion"))
        and moved
        and _motion_gaps(directory, origin_ns, end_ns) == 0,
        navigation_eligible=bool(eligibility.get("navigation")),
        quality_flags=sorted(quality_flags),
        event_clock_valid=version >= 10,
    )
    # The explorer log: exit facts, and dispatches paired with Nav2 goals (the parser of suite image 27 reads log formats 1
    # and 2). Rejected goals take part in the pairing, so a rejected dispatch never pairs with the next goal.
    log = parse(read_text(directory))
    clock = ClockMap(
        (
            r["receipt_utc_ns"],
            r["receipt_clock_ns"] if r.get("receipt_clock_ns") is not None else r["clock_ns"],
        )
        for r in quality
        if r.get("observation") == "rtf_window"
        and r["receipt_utc_ns"] is not None
        and (r.get("receipt_clock_ns") is not None or r["clock_ns"] is not None)
    )
    pairs, mismatches, unmapped = match(
        log.dispatches,
        [(sequence, row["clock_ns"]) for sequence, row in initialized.items()]
        + telemetry["rejected"],
        clock,
    )
    record.update(
        explorer_exit_status=log.exit_status,
        explorer_exit_code=log.exit_code
        if log.exit_status in ("exited_nonzero", "signalled")
        else None,
        explorer_exited_before_stop=log.exited_before_stop,
        goal_match_mismatches=mismatches,
        log_unmapped_lines=unmapped,
    )
    # A manifest the engine wrote for a run that never started has no robot, coverage or policy: such a
    # record has no stratum (it is placed by its design factors).
    complete = all(manifest.get(name) for name in ("robot", "coverage", "policy"))
    runtime = manifest.get("runtime") or {}
    runtime_document = (
        load_object(workspace, runtime["object"]) if runtime.get("object") else runtime
    )
    record.update(
        comparison_key=comparison_key(manifest, runtime_document) if complete else None,
        comparison_key_version=COMPARISON_KEY_VERSION,
        parameter_hash=content_hash(manifest.get("effective_parameters", {})),
    )
    derived = [
        _write_table(
            directory,
            "goals",
            _goal_rows(
                facts,
                initialized,
                pairs,
                positions,
                pose_source,
                manifest,
                (origin_ns, activation_ns, end_ns),
                log.version,
                cancels_by_dispatch(log),
            ),
            GOALS_SCHEMA,
        ),
        _write_table(
            directory,
            "explorer_decisions",
            decision_rows(log, clock, origin_ns, activation_ns),
            DECISIONS_SCHEMA,
        ),
        _write_table(
            directory,
            "gate_evaluations",
            gate_rows(log, clock, origin_ns, activation_ns, goals_by_dispatch(pairs)),
            GATES_SCHEMA,
        ),
    ]
    return dict(fields=record, derived=derived, cost=dict(cpu_s=time.process_time() - started))
