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

from collections import Counter
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .analysis import (
    ANALYSIS_VERSION,
    DetectorSettings,
    MotionDetector,
    coverage_plateaus,
    navigation_no_progress,
    tail_durations,
    with_navigation,
)
from .contracts import DATASET_VERSION, average_speed, content_hash
from .datasets import files, iter_rows, materialize_manifest
from .schemas import IDENTITIES, SCHEMAS
from .storage import atomic_json, file_hash, load_object, sync_directory

MILESTONE_PERCENTS = (25, 50, 75, 90, 95)  # coverage milestones (``coverage.MILESTONES``)
SUMMARY_FIELDS = [
    ("repetition", pa.int32()),
    ("block_id", pa.string()),
    ("profile_name", pa.string()),
    ("spawn_name", pa.string()),
    ("outcome", pa.string()),
    ("experiment_state", pa.string()),
    ("is_final", pa.bool_()),
    ("comparison_key", pa.string()),
    ("environment_key", pa.string()),
    ("physical_key", pa.string()),
    ("coverage_key", pa.string()),
    ("parameter_hash", pa.string()),
    ("analysis_version", pa.string()),
    ("timing_eligible", pa.bool_()),
    ("coverage_eligible", pa.bool_()),
    ("motion_eligible", pa.bool_()),
    ("navigation_eligible", pa.bool_()),
    ("simulation_elapsed_seconds", pa.float64()),
    ("completion_seconds", pa.float64()),
    ("coverage_percent", pa.float64()),
    ("covered_m2", pa.float64()),
    ("distance_m", pa.float64()),
    ("average_speed_mps", pa.float64()),
    ("cpu_average_percent", pa.float64()),
    ("rss_average_mib", pa.float64()),
    ("cpu_seconds", pa.float64()),
    ("peak_rss_mib", pa.float64()),
    ("time_weighted_cpu_percent", pa.float64()),
    ("time_weighted_rss_mib", pa.float64()),
    ("preparation_wall_seconds", pa.float64()),
    ("activation_to_movement_seconds", pa.float64()),
    ("rtf", pa.float64()),
    ("moving_seconds", pa.float64()),
    ("stationary_seconds", pa.float64()),
    ("turning_in_place_seconds", pa.float64()),
    ("valid_motion_seconds", pa.float64()),
    ("speed_p25_m_s", pa.float64()),
    ("speed_p50_m_s", pa.float64()),
    ("speed_p75_m_s", pa.float64()),
    ("speed_p95_m_s", pa.float64()),
    ("physical_speed_p50_m_s", pa.float64()),
    ("physical_speed_p95_m_s", pa.float64()),
    ("angular_speed_p50_rad_s", pa.float64()),
    ("angular_speed_p95_rad_s", pa.float64()),
    ("goals", pa.int64()),
    ("preemptions", pa.int64()),
    ("failed_goals", pa.int64()),
    ("rejected_goals", pa.int64()),
    ("recoveries", pa.int64()),
    ("distance_per_covered_m2", pa.float64()),
    ("cpu_seconds_per_covered_m2", pa.float64()),
    ("tail_90_to_95_seconds", pa.float64()),
    ("tail_90_to_completion_seconds", pa.float64()),
    ("drift_p50_m", pa.float64()),
    ("drift_p95_m", pa.float64()),
    ("incidents", pa.int64()),
    ("detail_json", pa.string()),
] + [(f"first_{percent}_simulation_seconds", pa.float64()) for percent in MILESTONE_PERCENTS]

SUMMARY_SCHEMA = pa.schema(
    [pa.field(name, typ, nullable=nullable) for name, typ, nullable in IDENTITIES]
    + [pa.field(name, typ, nullable=True) for name, typ in SUMMARY_FIELDS],
    metadata={
        b"schema_version": str(DATASET_VERSION).encode(),
        b"kind": b"attempt_summary",
        b"time_units": b"seconds",
        b"analysis_version": ANALYSIS_VERSION.encode(),
    },
)

OUTCOME_SCHEMA = pa.schema(
    [
        ("campaign_id", pa.string()),
        ("experiment_id", pa.string()),
        ("robot_id", pa.string()),
        ("world_id", pa.string()),
        ("profile_id", pa.string()),
        ("spawn_id", pa.string()),
        ("block_id", pa.string()),
        ("repetition", pa.int32()),
        ("state", pa.string()),
        ("outcome", pa.string()),
        ("final_attempt_id", pa.string()),
    ],
    metadata={b"schema_version": str(DATASET_VERSION).encode(), b"kind": b"experiment_outcomes"},
)


@contextmanager
def parquet_rows(path: Path, schema: pa.Schema) -> Iterator[Any]:
    writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=True)
    try:
        yield writer
    finally:
        writer.close()
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        sync_directory(path.parent)


def quantiles(directory: Path, kind: str, expression: str) -> list[float] | None:
    paths = files(directory, kind)
    if not paths:
        return None
    with duckdb.connect(config={"threads": 1, "memory_limit": "128MB"}) as connection:
        result = connection.execute(
            f"SELECT quantile_cont({expression},[0.25,0.5,0.75,0.95]) FROM read_parquet(?) WHERE attempt_id=? AND valid AND simulation_elapsed_ns IS NOT NULL AND isfinite({expression})",
            ([str(path) for path in paths], directory.name),
        ).fetchone()
        return result[0] if result else None


def navigation_counts(directory: Path) -> dict[str, int | None]:
    paths = files(directory, "navigation")
    names = ("goals", "preemptions", "failed_goals", "rejected_goals", "recoveries")
    if not paths:
        return dict.fromkeys(names)
    with duckdb.connect(config={"threads": 1, "memory_limit": "128MB"}) as connection:
        counts = connection.execute(
            "WITH observations AS (SELECT * FROM read_parquet(?) WHERE attempt_id=?) SELECT count(DISTINCT CASE WHEN event='initialized' THEN goal_sequence END), count(*) FILTER(WHERE event='preempted'), count(*) FILTER(WHERE event='terminal' AND status='failed'), count(*) FILTER(WHERE event IN ('rejected','preempt_rejected')), (SELECT sum(n) FROM (SELECT max(recoveries) AS n FROM observations WHERE native_uuid IS NOT NULL GROUP BY native_uuid)) FROM observations",
            ([str(path) for path in paths], directory.name),
        ).fetchone()
        assert counts is not None
        return dict(zip(names, counts))


def comparison_identity(manifest: dict[str, Any]) -> tuple[str, str, str]:
    robot = manifest.get("robot", {})
    generated = manifest.get("robot_configuration", {})
    reference = manifest.get("coverage", {}).get("reference", {})
    physical = {
        name: robot.get(name)
        for name in (
            "id",
            "kinematics",
            "footprint",
            "max_linear_speed",
            "max_angular_speed",
            "max_lateral_speed",
            "linear_acceleration",
            "angular_acceleration",
            "angular_deceleration",
            "physics_step",
            "lidar",
            "dependencies",
            "source_revisions",
        )
    }
    physical["sensor_configuration"] = generated.get("sensor_configuration")
    coverage = {
        name: reference.get(name)
        for name in ("id", "metric_version", "grid_sha256", "metadata_sha256", "target_area_m2")
    }
    config = manifest.get("configuration", {})
    key = content_hash(
        dict(
            physical=physical,
            coverage=coverage,
            spawn=config.get("scenario", {}).get("spawn"),
            policy=manifest.get("policy"),
            common=manifest.get("runtime", {}),
            generated_common={
                name: generated.get("sha256", {}).get(name)
                for name in ("navigation.yaml", "slam.yaml", "ekf.yaml")
            },
        )
    )
    return key, content_hash(physical), content_hash(coverage)


def enrich_incidents(directory: Path, destination: Path, settings: DetectorSettings) -> None:
    motion, coverage = files(directory, "trajectory"), files(directory, "coverage")
    source = destination / "incidents.parquet"
    temporary = source.with_suffix(".staging")
    with duckdb.connect(
        config={
            "threads": 1,
            "memory_limit": "256MB",
            "temp_directory": str(destination / "query-staging"),
        }
    ) as connection:
        if motion:
            connection.execute(
                "CREATE TEMP TABLE motion AS SELECT *,row_number() OVER(ORDER BY clock_ns,receipt_monotonic_ns)-1 AS n FROM read_parquet(?) WHERE attempt_id=?",
                ([str(path) for path in motion], directory.name),
            )
        else:
            connection.register(
                "motion",
                pa.Table.from_pylist(
                    [], schema=pa.schema(list(SCHEMAS["trajectory"]) + [pa.field("n", pa.int64())])
                ),
            )
        if coverage:
            connection.execute(
                "CREATE TEMP TABLE coverage AS SELECT * FROM read_parquet(?) WHERE attempt_id=? ORDER BY simulation_elapsed_ns",
                ([str(path) for path in coverage], directory.name),
            )
        else:
            connection.register("coverage", pa.Table.from_pylist([], schema=SCHEMAS["coverage"]))
        query = """SELECT i.* REPLACE(
            CASE WHEN i.trajectory_start_row>=0 AND box.x1 IS NOT NULL THEN [box.x1,box.y1,box.x2,box.y2] ELSE i.bbox END AS bbox,
            CASE WHEN c1.valid AND c2.valid AND c1.status='valid' AND c2.status='valid'
                AND i.start_ns-c1.simulation_elapsed_ns<=? AND i.end_ns-c2.simulation_elapsed_ns<=?
                THEN c2.coverage_percent-c1.coverage_percent ELSE NULL END AS coverage_change_percent)
            FROM read_parquet(?) i
            LEFT JOIN LATERAL (SELECT min(world_x) AS x1,min(world_y) AS y1,max(world_x) AS x2,max(world_y) AS y2 FROM motion WHERE n BETWEEN i.trajectory_start_row AND i.trajectory_end_row AND valid) box ON true
            ASOF LEFT JOIN coverage c1 ON i.start_ns>=c1.simulation_elapsed_ns
            ASOF LEFT JOIN coverage c2 ON i.end_ns>=c2.simulation_elapsed_ns"""
        batches = connection.execute(
            query,
            (
                round(settings.maximum_coverage_gap_seconds * 1e9),
                round(settings.maximum_coverage_gap_seconds * 1e9),
                str(source),
            ),
        ).fetch_record_batch(4096)
        with parquet_rows(temporary, SCHEMAS["incidents"]) as writer:
            for batch in batches:
                rows = batch.to_pylist()
                for row in rows:
                    evidence = json.loads(row["evidence_json"])
                    change = row["coverage_change_percent"]
                    if row["type"] in ("revisit", "closed_loop"):
                        evidence["low_coverage_progress"] = (
                            change < settings.plateau_gain_percent if change is not None else None
                        )
                        evidence["coverage_change_source"] = (
                            "latest valid observed samples within configured gap; no interpolation"
                        )
                    row["evidence_json"] = json.dumps(evidence, allow_nan=False)
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS["incidents"]))
    temporary.replace(source)
    sync_directory(destination)


def summarize(
    directory: Path,
    attempt: dict[str, Any],
    experiment: dict[str, Any],
    settings: DetectorSettings,
    destination: Path,
) -> dict[str, Any]:
    manifest = materialize_manifest(directory)
    config = manifest.get("configuration", {})
    if manifest.get("runtime", {}).get("object"):
        inventory = load_object(directory.parent.parent, manifest["runtime"]["object"])
    else:
        inventory = manifest.get("runtime", {})
    metrics = manifest.get("final_metrics", {})
    phase_times = {
        row["event"]: dict(
            clock_ns=row.get("clock_ns"),
            receipt_utc_ns=row["receipt_utc_ns"],
            receipt_monotonic_ns=row["receipt_monotonic_ns"],
        )
        for row in iter_rows(directory, "events")
        if row["event"] in ("ready", "exploration_started", "terminal")
    }
    ready_time = phase_times.get("ready", {}).get("receipt_utc_ns")
    outcome = attempt["outcome"]
    eligibility = manifest.get("eligibility", {})
    identity = dict(
        campaign_id=attempt["campaign_id"],
        experiment_id=attempt["experiment_id"],
        attempt_id=attempt["id"],
        robot_id=experiment["robot_id"],
        world_id=experiment["world_id"],
        profile_id=experiment["profile_id"],
        spawn_id=experiment["spawn_id"],
    )
    specification = json.loads(experiment["specification"])
    comparison, physical, coverage_key = comparison_identity(manifest)
    robot = manifest.get("robot", {})
    detector = MotionDetector(
        robot.get("max_linear_speed", 0.8), robot.get("footprint", []), settings
    )
    incident_counts: Counter[str] = Counter()
    with parquet_rows(destination / "incidents.parquet", SCHEMAS["incidents"]) as writer:

        def record(rows: list[dict[str, Any]]) -> None:
            if rows:
                incident_counts.update(row["type"] for row in rows)
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS["incidents"]))

        for row in with_navigation(
            iter_rows(directory, "trajectory"), iter_rows(directory, "navigation")
        ):
            record(detector.feed(row))
        record(detector.close_segments())
        for row in coverage_plateaus(iter_rows(directory, "coverage"), settings):
            record([row])
        alignment = (manifest.get("coverage", {}).get("alignment") or {}).get("world_from_map")

        def navigation_positions() -> Iterator[dict[str, Any]]:
            import math

            for row in iter_rows(directory, "navigation"):
                if (
                    alignment
                    and row.get("frame") == "map"
                    and row.get("x") is not None
                    and row.get("y") is not None
                ):
                    x, y, angle = alignment
                    row.update(
                        world_x=x + math.cos(angle) * row["x"] - math.sin(angle) * row["y"],
                        world_y=y + math.sin(angle) * row["x"] + math.cos(angle) * row["y"],
                    )
                yield row

        for row in navigation_no_progress(navigation_positions(), settings):
            record([row])
    enrich_incidents(directory, destination, settings)
    speed = quantiles(directory, "trajectory", "speed")
    physical_speed = quantiles(directory, "ground_truth", "speed")
    angular = quantiles(directory, "trajectory", "abs(wz)")
    drift = quantiles(directory, "ground_truth", "odometry_drift_m")
    milestones = manifest.get("coverage", {}).get("milestones", [])
    sim = metrics.get("simulation_elapsed") if metrics.get("motion_started") else None
    completion = sim if outcome == "completed" else None
    tail = tail_durations(milestones, completion)
    resources = manifest.get("resource_summary", {})
    covered = metrics.get("coverage_covered_m2")
    result = dict(
        identity,
        repetition=experiment["repetition"],
        block_id=experiment["block_id"],
        profile_name=specification["profile"].get("name", experiment["profile_id"]),
        spawn_name=specification["spawn"].get("name", experiment["spawn_id"]),
        outcome=outcome,
        experiment_state=experiment["state"],
        is_final=experiment.get("final_attempt_id") == attempt["id"],
        comparison_key=comparison,
        environment_key=content_hash(
            dict(
                coverage=coverage_key,
                scenario=config.get("scenario"),
                policy=manifest.get("policy"),
                runtime=manifest.get("runtime"),
            )
        ),
        physical_key=physical,
        coverage_key=coverage_key,
        parameter_hash=content_hash(manifest.get("effective_parameters", {})),
        analysis_version=ANALYSIS_VERSION,
        timing_eligible=bool(eligibility.get("timing")),
        coverage_eligible=bool(eligibility.get("coverage")),
        motion_eligible=bool(eligibility.get("motion")) and detector.gaps == 0,
        navigation_eligible=bool(eligibility.get("navigation")),
        simulation_elapsed_seconds=sim,
        completion_seconds=completion,
        coverage_percent=metrics.get("coverage_percent")
        if metrics.get("coverage_status") == "valid"
        else None,
        covered_m2=covered,
        distance_m=metrics.get("distance_m"),
        average_speed_mps=average_speed(metrics.get("distance_m"), sim),
        cpu_average_percent=metrics.get("cpu_average_percent"),
        rss_average_mib=metrics.get("rss_average_mib"),
        cpu_seconds=metrics.get("cpu_seconds"),
        peak_rss_mib=resources.get("peak_explorer_rss_mib"),
        time_weighted_cpu_percent=resources.get("time_weighted_explorer_cpu_percent"),
        time_weighted_rss_mib=resources.get("time_weighted_explorer_rss_mib"),
        preparation_wall_seconds=(ready_time - attempt["reserved_ns"]) / 1e9
        if ready_time is not None
        else None,
        activation_to_movement_seconds=metrics.get("pre_motion_simulation_elapsed"),
        rtf=metrics.get("rtf"),
        moving_seconds=detector.durations["moving"] if detector.valid_samples else None,
        stationary_seconds=detector.durations["stationary"] if detector.valid_samples else None,
        turning_in_place_seconds=detector.durations["turning_in_place"]
        if detector.valid_samples
        else None,
        valid_motion_seconds=sum(detector.durations.values()) if detector.valid_samples else None,
        speed_p25_m_s=speed[0] if speed else None,
        speed_p50_m_s=speed[1] if speed else None,
        speed_p75_m_s=speed[2] if speed else None,
        speed_p95_m_s=speed[3] if speed else None,
        physical_speed_p50_m_s=physical_speed[1] if physical_speed else None,
        physical_speed_p95_m_s=physical_speed[3] if physical_speed else None,
        angular_speed_p50_rad_s=angular[1] if angular else None,
        angular_speed_p95_rad_s=angular[3] if angular else None,
        drift_p50_m=drift[1] if drift else None,
        drift_p95_m=drift[3] if drift else None,
        distance_per_covered_m2=metrics["distance_m"] / covered
        if covered and metrics.get("distance_m") is not None
        else None,
        cpu_seconds_per_covered_m2=metrics["cpu_seconds"] / covered
        if covered and metrics.get("cpu_seconds") is not None
        else None,
        tail_90_to_95_seconds=tail["90_to_95_seconds"],
        tail_90_to_completion_seconds=tail["90_to_completion_seconds"],
        incidents=sum(incident_counts.values()),
        **navigation_counts(directory),
    )
    for percent in MILESTONE_PERCENTS:  # runs recorded before T25 have no 25 % entry: unavailable
        result[f"first_{percent}_simulation_seconds"] = None
    for item in milestones:
        if int(item["percent"]) not in MILESTONE_PERCENTS:
            continue
        result[f"first_{int(item['percent'])}_simulation_seconds"] = (
            item.get("simulation_elapsed") if item.get("reached") else None
        )
    detail = dict(
        schema_version=DATASET_VERSION,
        analysis_version=ANALYSIS_VERSION,
        detectors=asdict(settings),
        incident_counts=incident_counts,
        motion_gaps=detector.gaps,
        spatial_limit_reached=detector.spatial_limit_reached,
        spatial_tolerance_m=detector.radius,
        speed_distribution="valid archived raw-odometry observations; sample quantiles",
        physical_speed_distribution="valid archived Gazebo diagnostic pose differences; sample quantiles; diagnostic only",
        motion_durations="left hold between adjacent valid observations; gaps excluded",
        coverage_policy="observed samples only; no interpolation or extrapolation",
        runtime_inventory_hash=content_hash(inventory),
        lifecycle_timestamps=phase_times,
        profile=load_object(directory.parent.parent, specification["profile"]["object"])
        if specification["profile"].get("object")
        else specification["profile"],
        spawn=specification["spawn"],
        requested_parameters=manifest.get("requested_parameters"),
        effective_parameters=manifest.get("effective_parameters"),
        terminal_detail=manifest.get("detail"),
        quality_flags=metrics.get("quality_flags", []),
        coverage_quality_flags=metrics.get("coverage_quality_flags", []),
        source_manifest_sha256=file_hash(directory / "manifest.json"),
        source_checkpoints_sha256=manifest.get("segments_hash"),
        objects=manifest.get("objects", {}),
        physical_context=robot,
        common_configuration=config.get("runtime", {}),
    )
    result["detail_json"] = json.dumps(detail, allow_nan=False, ensure_ascii=False)
    atomic_json(destination / "summary.json", result)
    return result
