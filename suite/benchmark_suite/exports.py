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

"""Frozen study exports (RESULTS_RESEARCH RD-33, RD-36; RESULTS_OVERHAUL §6, milestone R4).

One spawned job reads the store read-only and the final runs' files, and writes
``results/studies/<study_id>/exports/<utc>-<sha8>/``: tidy tables (Parquet, with a CSV mirror for the small
ones), maps and references, scrubbed raw streams and logs of every final run, a generated codebook,
``study.yaml``, ``provenance.json``, a README, the licence statement and ``SHA256SUMS``. It writes into a
staging directory and renames it when complete; an export is never changed afterwards. The analysis
module reads only an export and the study's file.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
from typing import TYPE_CHECKING, Any
import uuid

import duckdb
from exploration_benchmark_runner.coverage import STUCK_COMPLETION_PERCENT, STUCK_RULE, erode
from exploration_benchmark_runner.worlds import WorldRegistry
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import yaml

from .analysis_block import WorldFacts, plan_parts, validate
from .contracts import (
    NO_MOVEMENT_LIMIT,
    REPORTED_OUTCOMES,
    ContractError,
    canonical,
    content_hash,
    milestone_key,
)
from .datasets import files, iter_rows, materialize_manifest, stream_relation
from .explorer_log import read_text
from .run_record import (
    DERIVED_VERSION,
    DECISIONS_SCHEMA,
    FIELDS,
    GATES_SCHEMA,
    GOALS_SCHEMA,
    MILESTONES,
    READ_TIME_FIELDS,
    RUN_RECORD_VERSION,
    Field,
    read_time_fields,
    time_origins,
    track,
)
from .library import DEFAULT_EXPLORER, load_explorers
from .schemas import SCHEMAS
from .storage import file_hash, load_object, read_object, sync_directory
from .store import Store, replaced_campaigns, study_campaign_rows
from .studies import homogeneity, refused_mixes
from .world_facts import world_facts

if TYPE_CHECKING:
    from .jobs import Job

# 2 (world integration, plans/worlds/I2 §3.10, R-1): the ``worlds`` table, the registry hash and each campaign's
# time-limit rule and inputs in study.yaml, and descriptors.json and regions.png per reference. Additive.
# 3 (stuck-stop-v1, run-record-v6): the accounting count final_stuck and the endpoint end reason stuck.
EXPORT_VERSION = 3
LOGS = ("explorer", "insight", "navigation", "robot", "simulation", "worker")
ARROW = {
    "text": pa.string(),
    "int": pa.int64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
    "list": pa.list_(pa.string()),
}
# t25 … t95 and t99_9 (run-record-v4; t99 under run-record-v3)
MILESTONE_ENDPOINTS = tuple(f"t{milestone_key(m)}" for m in MILESTONES)
# Column descriptions of the tables built here (unit, null meaning, definition); the codebook joins them
# with the Arrow schemas, so every exported column is described.
DESCRIPTIONS: dict[str, dict[str, tuple[str, str, str]]] = {
    "experiments": {
        "experiment_id": ("", "never", "planned profile × block"),
        "campaign_id": ("", "never", "the campaign that planned it"),
        "block_id": ("", "never", "pairing unit (study namespace, RD-30)"),
        "robot_id": ("", "never", ""),
        "world_id": ("", "never", ""),
        "spawn_id": ("", "never", ""),
        "profile_id": ("", "never", ""),
        "repetition": ("", "never", "study-wide repetition number"),
        "state": ("", "never", "done, not_done or unresolved (engine states mapped)"),
        "campaign_unfinished": ("", "never", "the campaign has not finished execution"),
        "final_attempt_id": ("", "no final attempt", ""),
        "n_attempts": ("count", "never", ""),
        "n_replaced": ("count", "never", "attempts a later attempt of the experiment replaced"),
        "n_discarded": ("count", "never", "attempts discarded by a stop or restart"),
    },
    "endpoints": {
        "attempt_id": ("", "never", "a final run"),
        "endpoint": ("", "never", "completion, t25 … t95, t99_9, movement_onset (RD-27)"),
        "time_s": ("sim s", "no observed event", "raw event time; movement onset from activation"),
        "event": ("", "competing or missing outcome", ""),
        "end_reason": (
            "",
            "never",
            "RD-27 end reasons, plus coverage_ineligible, not_recorded, completion_without_motion and "
            "stuck (censored at L as time_limit; export-v3)",
        ),
        "limit_s": (
            "sim s",
            "no limit",
            "the endpoint's censor time (L, or 180 s for movement onset)",
        ),
    },
    "coverage": {
        "attempt_id": ("", "never", ""),
        "stage": ("", "never", "pre_motion, baseline or active"),
        "t_s": ("sim s", "pre_motion rows", "movement-relative sample time"),
        "t_activation_s": ("sim s", "activation unknown", "activation-relative sample time"),
        "coverage_pct": ("%", "not valid", "MEASUREMENTS Coverage"),
        "status": ("", "never", "pending, valid, stale or unavailable"),
        "is_terminal": ("", "never", "the frozen terminal sample"),
    },
    "trajectory_gt": {
        "attempt_id": ("", "never", ""),
        "t_activation_s": ("sim s", "never", "whole seconds from activation to the stop"),
        "t_s": ("sim s", "before movement onset", "movement-relative time of the grid point"),
        "gt_x_m": ("m", "no pose yet", "last ground-truth pose at or before the grid point"),
        "gt_y_m": ("m", "no pose yet", ""),
        "gt_yaw_rad": ("rad", "no pose yet", ""),
        "gt_distance_m": (
            "m",
            "unknown speed limit or no pose",
            "cumulative ground-truth path length",
        ),
        "odom_distance_m": ("m", "no odometry yet", "cumulative raw-odometry distance"),
    },
    # One row per robot, world, spawn and profile (RD-33): the group's identity, then counts
    # (reported-outcomes-v3 report classes, run_record.report_class).
    "accounting": {
        "robot_id": ("", "never", "the group's robot"),
        "world_id": ("", "never", "the group's world"),
        "spawn_id": ("", "never", "the group's spawn"),
        "profile_id": ("", "never", "the group's profile"),
        "comparison_keys": (
            "count",
            "never",
            "distinct comparison keys among the group's reported final runs (more than 1: not comparable)",
        ),
        "planned": ("count", "never", "experiments planned"),
        "done": ("count", "never", "experiments resolved (DONE)"),
        "unresolved": ("count", "never", "experiments left UNRESOLVED"),
        "not_done": ("count", "never", "experiments neither done nor unresolved"),
        "final_completed": ("count", "never", "reported final runs that completed"),
        "final_time_limit": ("count", "never", "reported final runs stopped at the time limit"),
        "final_no_movement": ("count", "never", "reported final runs that never moved"),
        "final_stuck": (
            "count",
            "never",
            f"reported final runs stopped stuck below {STUCK_COMPLETION_PERCENT:g} % coverage "
            f"({STUCK_RULE}; export-v3)",
        ),
        "reported": ("count", "never", "final runs of report class reported"),
        "coverage_ineligible": ("count", "never", "reported final runs not eligible for coverage"),
        "missing_startup": ("count", "never", "final runs of report class missing_startup"),
        "missing_platform": ("count", "never", "final runs of report class missing_platform"),
        "excluded_runtime_failure": (
            "count",
            "never",
            "final runs of report class excluded_runtime_failure",
        ),
        "excluded_stopped": ("count", "never", "final runs of report class excluded_stopped"),
        "replaced_timing_rejection": (
            "count",
            "never",
            "attempts replaced after a timing rejection",
        ),
        "replaced_engine_interruption": (
            "count",
            "never",
            "attempts replaced after an engine interruption",
        ),
        "discarded_campaign_stop": ("count", "never", "attempts discarded by a campaign stop"),
        "discarded_experiment_stop": (
            "count",
            "never",
            "attempts discarded by an experiment stop",
        ),
        "discarded_experiment_restart": (
            "count",
            "never",
            "attempts discarded by an experiment restart",
        ),
    },
    # The registry facts of the worlds of the final runs (contract: plans/worlds/EXECUTION.md "Shared contracts").
    "worlds": {
        "world_id": ("", "never", "world registry ID"),
        "tier": ("", "never", "A, SC, P, F, RP or MO"),
        "family": ("", "not tier F", "F1 … F5"),
        "role": ("", "never", "registry role"),
        "design_point_json": (
            "JSON",
            "never",
            "factor -> level of a tier F world; {} for named worlds",
        ),
        "seed": ("", "not tier F", "world.yaml seed"),
        "generator_version": (
            "",
            "tier A (not a collection world)",
            "worldgen version of the world",
        ),
        "world_yaml_sha256": (
            "sha256",
            "tier A (not a collection world)",
            "the world.yaml the registry pins",
        ),
    },
    "profiles": {
        "profile_id": ("", "never", ""),
        "profile_name": ("", "never", ""),
        "robot_id": ("", "never", ""),
        "explorer_id": (
            "",
            "never",
            "the profile's explorer as the run record's explorer_id (run-record-v5): its runs' value, "
            "else its profiles.yaml explorer's package",
        ),
        "parameter_hash": ("sha256", "never", "content hash of the effective parameters"),
        "is_baseline": ("", "no Default hash recorded", "equal to the robot's Default hash"),
        "parameter": ("", "never", ""),
        "value": ("JSON", "never", "effective value"),
        "default_value": (
            "JSON",
            "the study has no Default profile of the profile's explorer",
            "the value in the profile's explorer's Default for the robot",
        ),
        "differs_from_default": ("", "no Default value", ""),
    },
}
ACCOUNTING_KEYS = ("robot_id", "world_id", "spawn_id", "profile_id")
ACCOUNTING_COUNTS = (
    "planned",
    "done",
    "unresolved",
    "not_done",
    "final_completed",
    "final_time_limit",
    "final_no_movement",
    "final_stuck",
    "reported",
    "coverage_ineligible",
    "missing_startup",
    "missing_platform",
    "excluded_runtime_failure",
    "excluded_stopped",
    "replaced_timing_rejection",
    "replaced_engine_interruption",
    "discarded_campaign_stop",
    "discarded_experiment_stop",
    "discarded_experiment_restart",
)


class Refused(ValueError):
    """A study that cannot be exported as it is; the message says why."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What one consistent read of the store gives an export."""

    campaigns: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    experiments: dict[str, dict[str, Any]]
    records: dict[str, dict[str, Any]]
    documents: dict[str, Any]


def studies_root(workspace: Path) -> Path:
    """``results/studies`` beside ``results/campaigns/<workspace>`` (RD-36; the engine's mount)."""
    return workspace.resolve().parents[1] / "studies"


def _rows(cursor: Iterable[tuple[Any, ...]], names: str) -> list[dict[str, Any]]:
    keys = names.split(",")
    return [dict(zip(keys, row, strict=True)) for row in cursor]


def snapshot(workspace: Path, study_id: str) -> Snapshot:
    """Read the study from the store in one read-only transaction and refuse what cannot be exported.

    A restarted copy replaces its source (and its source's source); a repeated campaign adds blocks.
    """
    database = sqlite3.connect(f"file:{workspace / 'suite.sqlite3'}?mode=ro", uri=True)
    try:
        database.execute("BEGIN")
        campaigns = _rows(
            study_campaign_rows(database, study_id),
            "id,name,state,specification,created_ns,finished_ns",
        )
        replaced = replaced_campaigns(database)
        campaigns = [campaign for campaign in campaigns if campaign["id"] not in replaced]
        if not campaigns:
            raise Refused(f"Study {study_id} has no campaigns")
        for campaign in campaigns:
            campaign["specification"] = json.loads(campaign["specification"])
        ids = [campaign["id"] for campaign in campaigns]
        marks = ",".join("?" * len(ids))
        attempts = _rows(
            database.execute(
                "SELECT id,campaign_id,experiment_id,number,phase,outcome,discarded FROM attempts "
                f"WHERE campaign_id IN ({marks}) ORDER BY campaign_id,experiment_id,number",
                ids,
            ),
            "id,campaign_id,experiment_id,number,phase,outcome,discarded",
        )
        experiments = _rows(
            database.execute(
                "SELECT id,campaign_id,block_id,robot_id,world_id,spawn_id,profile_id,repetition,state,"
                f"final_attempt_id,specification FROM experiments WHERE campaign_id IN ({marks}) "
                "ORDER BY campaign_id,position",
                ids,
            ),
            "id,campaign_id,block_id,robot_id,world_id,spawn_id,profile_id,repetition,state,"
            "final_attempt_id,specification",
        )
        records = {
            row[0]: json.loads(row[1])
            for row in database.execute(
                f"SELECT attempt_id,record FROM run_records WHERE campaign_id IN ({marks}) "
                "AND run_record_version=?",
                (*ids, RUN_RECORD_VERSION),
            )
        }
        documents = {
            row[0]: json.loads(row[1])
            for row in database.execute("SELECT id,document FROM provenance")
        }
    finally:
        database.close()
    if any(attempt["phase"] != "FINISHED" for attempt in attempts):
        raise Refused("Runs of this study are still active; export once they have finished (RO-10)")
    missing = [attempt["id"] for attempt in attempts if attempt["id"] not in records]
    if missing:
        raise Refused(
            f"{len(missing)} run record(s) missing or of another version; export when they exist (a failed "
            "record job is retried at the next engine start; see the run_record.failed event)"
        )
    duplicates = Counter((e["block_id"], e["profile_id"]) for e in experiments)
    if any(count > 1 for count in duplicates.values()):
        raise Refused(
            "Two campaigns hold the same block and profile (a restart whose link was deleted?); "
            "delete one of them first"
        )
    return Snapshot(campaigns, attempts, {e["id"]: e for e in experiments}, records, documents)


def _scrubber(workspace: Path, recorded: Iterable[tuple[str, str]] = ()) -> list[tuple[str, str]]:
    """Text replacements that remove absolute paths from copies (D-27; RD-33).

    Covers this workspace and checkout and the paths attempts recorded (they may have run elsewhere),
    and the home directory that holds them (inside the container HOME is not it; paths are mounted as
    they are). Longest first, so a path is never half replaced.
    """
    resolved = workspace.resolve()
    pairs = {str(resolved): "<workspace>", str(resolved.parents[2]): "<checkout>"}
    for path, name in recorded:
        if path and path.startswith("/"):
            pairs.setdefault(path.rstrip("/"), name)
    for path in list(pairs):
        parts = Path(path).parts
        if len(parts) > 2 and parts[1] == "home":
            pairs.setdefault("/home/" + parts[2], "~")
    return sorted(pairs.items(), key=lambda pair: -len(pair[0]))


def scrub(value: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        value = value.replace(old, new)
    return value


def scrub_table(table: pa.Table, pairs: list[tuple[str, str]]) -> pa.Table:
    """Every string column with absolute paths replaced; schema metadata dropped (deterministic)."""
    columns = list(table.columns)
    for old, new in pairs:
        columns = [
            pc.replace_substring(column, old, new) if pa.types.is_string(column.type) else column
            for column in columns
        ]
    return pa.table(columns, schema=table.schema.remove_metadata())


def _write(
    staging: Path, name: str, table: pa.Table, pairs: list[tuple[str, str]], csv_mirror: bool = True
) -> pa.Schema:
    """A scrubbed table as Parquet, and as CSV (list columns as JSON text) when ``csv_mirror``."""
    table = scrub_table(table, pairs)
    pq.write_table(table, staging / f"{name}.parquet", compression="zstd")
    if csv_mirror:
        columns = [
            pa.array([json.dumps(v) if v is not None else None for v in column.to_pylist()])
            if pa.types.is_list(column.type)
            else column
            for column in table.columns
        ]
        pcsv.write_csv(pa.table(columns, names=table.column_names), staging / f"{name}.csv")
    return table.schema


def _schema_of(
    description: dict[str, tuple[str, str, str]], types: dict[str, pa.DataType]
) -> pa.Schema:
    """A table's Arrow schema: its described columns in order, strings unless ``types`` says otherwise."""
    return pa.schema([(name, types.get(name, pa.string())) for name in description])


def _pgm(path: Path, pixels: "np.ndarray[Any, Any]") -> None:
    """A binary PGM and a PNG of an 8-bit image (top row first)."""
    height, width = pixels.shape
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + pixels.astype(np.uint8).tobytes())
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(buffer, format="PNG", optimize=False)
    path.with_suffix(".png").write_bytes(buffer.getvalue())


def _map(directory: Path, destination: Path, manifest: dict[str, Any]) -> None:
    """The final SLAM map as a Nav2 map (trinary PGM and PNG with YAML), as map_saver writes it, with the
    frozen map-to-world alignment that places it on the reference (F8)."""
    source = directory / "final_map.npz"
    if not source.is_file():
        return
    with np.load(source, allow_pickle=False) as data:
        occupancy = data["occupancy"]
        metadata = json.loads(str(data["metadata"]))
    pixels = np.full(occupancy.shape, 205, dtype=np.uint8)  # unknown
    pixels[(occupancy >= 0) & (occupancy <= 25)] = 254  # free
    pixels[occupancy >= 65] = 0  # occupied
    _pgm(destination.with_suffix(".pgm"), np.flipud(pixels))
    origin = metadata["origin"]
    alignment = ((manifest.get("coverage") or {}).get("alignment") or {}).get("world_from_map")
    description = dict(
        image=destination.with_suffix(".pgm").name,
        mode="trinary",
        resolution=metadata["resolution"],
        origin=[origin[0], origin[1], 0.0],
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.25,
        world_from_map=alignment,
    )
    destination.with_suffix(".yaml").write_text(yaml.safe_dump(description, sort_keys=True))


def _checkout(workspace: Path) -> Path:
    """The checkout beside ``results/campaigns/<workspace>`` (the engine's mount)."""
    return workspace.resolve().parents[2]


def _descriptors(
    registry: WorldRegistry, world: str, metadata_sha256: str, target_dir: Path
) -> str | None:
    """Copy a reference's official descriptors and region raster (I2 §3.9) from the checkout, hash-checked
    against the registry pins. Returns a warning instead, and copies neither file, when the world has no
    descriptor pin or its pinned file describes another reference than the one the attempt used (the
    pin moved on after the runs; the runs stay exportable). A descriptor file that describes another
    reference than the registry's own pin, changed bytes or a raster that does not match refuse."""
    try:
        entry = registry.world(world)
    except ValueError:
        entry = None
    pin = entry.descriptors if entry is not None else None
    if entry is None or pin is None:
        return f"world {world}: no descriptor file in the registry; descriptors.json and regions.png left out"
    data = registry.path(pin.file).read_bytes()
    if hashlib.sha256(data).hexdigest() != pin.sha256:
        raise Refused(f"{pin.file} changed since the registry was loaded")
    try:
        document = json.loads(data)
        described, raster_sha256 = (
            document["grid"]["metadata_sha256"],
            document["region_raster"]["sha256"],
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise Refused(f"{pin.file} is not a descriptor file: {exc!r}") from exc
    if described != entry.reference.metadata_sha256:
        # the registry contradicts itself (a descriptor file pinned to another reference): never a warning
        raise Refused(
            f"{pin.file} describes reference {described[:8]}, not the registry's "
            f"{entry.reference.metadata_sha256[:8]} for {world}"
        )
    if described != metadata_sha256:
        return (
            f"world {world}: the pinned {pin.file} describes another reference than "
            f"{metadata_sha256[:8]}, which runs used; descriptors.json and regions.png left out"
        )
    # The pinned raster (registry naming rule); inside the export it is regions.png, the name the descriptor
    # file's region_raster.file gives.
    raster_bytes = registry.path(pin.regions_file).read_bytes()
    if hashlib.sha256(raster_bytes).hexdigest() != raster_sha256:
        raise Refused(f"{pin.regions_file} does not match {pin.file}")
    (target_dir / "descriptors.json").write_bytes(data)
    (target_dir / "regions.png").write_bytes(raster_bytes)
    return None


def _reference(
    workspace: Path,
    manifest: dict[str, Any],
    destination: Path,
    pairs: list[tuple[str, str]],
    registry: WorldRegistry,
    warnings: list[str],
) -> str | None:
    """The coverage reference an attempt used: masks as PGM/PNG with their YAML (RESULTS_OVERHAUL §6), and
    its official descriptors and region raster (export version 2).

    Its directory is ``<reference id>-<8 hex of its objects' hash>``: a reference regenerated under the same
    ID (a changed world) gets its own directory.
    """
    objects = manifest.get("objects") or {}
    identity = ((manifest.get("coverage") or {}).get("reference") or {}).get("id")
    if not identity or "coverage_metadata" not in objects or "coverage_grid" not in objects:
        return None
    name = (
        f"{identity}-{content_hash([objects['coverage_metadata'], objects['coverage_grid']])[:8]}"
    )
    target_dir = destination / name
    if target_dir.exists():
        return name
    target_dir.mkdir(parents=True)
    metadata_bytes = read_object(workspace, objects["coverage_metadata"])
    metadata = json.loads(metadata_bytes)
    with np.load(
        io.BytesIO(read_object(workspace, objects["coverage_grid"])), allow_pickle=False
    ) as grid:
        collision = grid["collision"].astype(bool)
        inclusion = grid["inclusion"].astype(bool)
    target = inclusion & ~collision
    margin = float(metadata.get("target_margin_m", 0.0)) if metadata["metric_version"] >= 2 else 0.0
    if margin > 0:
        target = erode(target, margin / float(metadata["resolution"]))
    if int(target.sum()) != int(metadata["target_cells"]):
        raise Refused(f"Coverage reference {identity} does not reproduce its target cells")
    for mask_name, mask in (("collision", collision), ("inclusion", inclusion), ("target", target)):
        _pgm(target_dir / f"{mask_name}.pgm", np.flipud(np.where(mask, 255, 0)))
    (target_dir / "metadata.json").write_text(scrub(metadata_bytes.decode(), pairs))
    (target_dir / "reference.yaml").write_text(
        yaml.safe_dump(
            dict(
                id=identity,
                metric_version=metadata["metric_version"],
                resolution=metadata["resolution"],
                origin=metadata["origin"],
                target_cells=metadata["target_cells"],
                target_area_m2=metadata["target_area_m2"],
                target_margin_m=margin,
                masks="255 = in the mask, row 0 = largest y (as a Nav2 map image)",
                metadata_sha256=objects["coverage_metadata"],
                grid_sha256=objects["coverage_grid"],
            ),
            sort_keys=True,
        )
    )
    warning = _descriptors(registry, metadata["world"], objects["coverage_metadata"], target_dir)
    if warning is not None:
        warnings.append(warning)
    return name


def _gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write(text.encode())


def _raw(directory: Path, destination: Path, pairs: list[tuple[str, str]]) -> None:
    """One final run's streams (Parquet), manifest, logs and ROS logs (gzip text), paths replaced."""
    destination.mkdir(parents=True)
    with duckdb.connect(config={"threads": 1, "memory_limit": "256MB"}) as connection:
        for kind in SCHEMAS:
            paths = [str(path) for path in files(directory, kind)]
            if paths:
                table = connection.execute(
                    f"SELECT * FROM {stream_relation(connection, kind, paths)} WHERE attempt_id=? "
                    "ORDER BY clock_ns NULLS FIRST, receipt_monotonic_ns",
                    (paths, directory.name),
                ).fetch_arrow_table()
                pq.write_table(
                    scrub_table(table, pairs), destination / f"{kind}.parquet", compression="zstd"
                )
    # Materialised: the configuration, robot and parameter objects it references are not exported.
    (destination / "manifest.json").write_text(
        scrub(json.dumps(materialize_manifest(directory), indent=2, sort_keys=True), pairs)
    )
    for name in LOGS:
        text = read_text(directory, name)
        if text is not None:
            _gzip_text(destination / "logs" / f"{name}.log.gz", scrub(text, pairs))
    ros = directory / "ros_logs"
    if ros.is_dir():
        for path in sorted(ros.rglob("*.log*")):
            if path.name.endswith(".index.json") or not path.is_file():
                continue
            relative = path.relative_to(ros).as_posix().removesuffix(".zst")
            text = read_text(path.parent, path.name.removesuffix(".zst").removesuffix(".log"))
            if text is not None:
                _gzip_text(destination / "ros_logs" / f"{relative}.gz", scrub(text, pairs))


def _coverage_rows(directory: Path, activation_ns: int | None) -> Iterator[dict[str, Any]]:
    rows = list(iter_rows(directory, "coverage"))
    for index, row in enumerate(rows):
        yield dict(
            attempt_id=directory.name,
            stage=row["stage"],
            t_s=row["simulation_elapsed_ns"] / 1e9
            if row["simulation_elapsed_ns"] is not None
            else None,
            t_activation_s=(row["clock_ns"] - activation_ns) / 1e9
            if row["clock_ns"] is not None and activation_ns is not None
            else None,
            coverage_pct=row["coverage_percent"]
            if row["valid"] and row["status"] == "valid"
            else None,
            status=row["status"],
            is_terminal=index == len(rows) - 1,
        )


def _trajectory_rows(directory: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Ground truth at every whole second from activation to the stop (cumulative distances kept)."""
    origin_ns, activation_ns = time_origins(manifest)
    end_ns = manifest.get("terminal_trigger_simulation_ns")
    if activation_ns is None or end_ns is None:
        return []
    poses, source, _ = track(directory, manifest, end_ns)
    if source != "ground_truth":
        return []
    odometry = [
        (row["clock_ns"], row["distance_m"])
        for row in iter_rows(directory, "trajectory")
        if row["clock_ns"] is not None and row["distance_m"] is not None
    ]
    rows, pose_index, odom_index = [], -1, -1
    for second in range(0, max(0, (end_ns - activation_ns) // 1_000_000_000) + 1):
        grid = activation_ns + second * 1_000_000_000
        while pose_index + 1 < len(poses) and poses[pose_index + 1][0] <= grid:
            pose_index += 1
        while odom_index + 1 < len(odometry) and odometry[odom_index + 1][0] <= grid:
            odom_index += 1
        pose = poses[pose_index] if pose_index >= 0 else None
        rows.append(
            dict(
                attempt_id=directory.name,
                t_activation_s=float(second),
                t_s=(grid - origin_ns) / 1e9
                if origin_ns is not None and grid >= origin_ns
                else None,
                gt_x_m=pose[1] if pose else None,
                gt_y_m=pose[2] if pose else None,
                gt_yaw_rad=pose[3] if pose else None,
                gt_distance_m=pose[4] if pose else None,
                odom_distance_m=odometry[odom_index][1] if odom_index >= 0 else None,
            )
        )
    return rows


def _end_reason(record: dict[str, Any], report: str, endpoint: str) -> str:
    """RD-27 end reason of one endpoint of a final run."""
    if report != "reported":
        return report
    outcome = record["outcome"]
    if endpoint == "movement_onset":
        return "event" if record["motion_started"] else "no_movement"
    if outcome == "no_movement":
        return "no_movement"
    if outcome == "completed" and not record["motion_started"]:
        return "completion_without_motion"
    event = record[f"{endpoint}_event"]
    if event:
        return "event"
    if event is False:
        if outcome == "stuck":
            return "stuck"  # censored at L as a time_limit run (stuck-stop-v1)
        return (
            "completed_below_m"
            if outcome == "completed" and endpoint != "completion" and record["completion_event"]
            else "time_limit"
        )
    return "coverage_ineligible" if not record["coverage_eligible"] else "not_recorded"


def _endpoint_rows(record: dict[str, Any], report: str) -> list[dict[str, Any]]:
    rows = [
        dict(
            attempt_id=record["attempt_id"],
            endpoint=endpoint,
            time_s=record[f"{endpoint}_s"],
            event=record[f"{endpoint}_event"],
            end_reason=_end_reason(record, report, endpoint),
            limit_s=record[f"{endpoint}_censor_s"]
            if record[f"{endpoint}_censor_s"] is not None
            else record["time_limit_s"],
        )
        for endpoint in ("completion", *MILESTONE_ENDPOINTS)
    ]
    moved = record["motion_started"]
    rows.append(
        dict(
            attempt_id=record["attempt_id"],
            endpoint="movement_onset",
            time_s=record["activation_to_movement_s"] if moved else None,
            event=moved,
            end_reason=_end_reason(record, report, "movement_onset"),
            limit_s=record["activation_to_movement_censor_s"]
            if record["activation_to_movement_censor_s"] is not None
            else NO_MOVEMENT_LIMIT,
        )
    )
    return rows


def _schema(fields: Iterable[Field]) -> pa.Schema:
    return pa.schema([(field.name, ARROW[field.kind]) for field in fields])


def _run_rows(
    snap: Snapshot,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any], str]]]:
    """``runs`` rows (every attempt) and the final runs with their report class."""
    rows, finals = [], []
    last = {
        experiment: max(a["number"] for a in own)
        for experiment, own in _attempts_by_experiment(snap).items()
    }
    for attempt in snap.attempts:
        record = snap.records[attempt["id"]]
        # Superseded campaigns are left out of an export, so no exported run is superseded.
        experiment = snap.experiments[attempt["experiment_id"]]
        extra = read_time_fields(
            record, {**experiment, "last_attempt_number": last[attempt["experiment_id"]]}, None
        )
        rows.append({**{field.name: record[field.name] for field in FIELDS}, **extra})
        if extra["is_final"]:
            finals.append((attempt, record, extra["report_class"]))
    full = [record for _, record, _ in finals if record["record_kind"] == "full"]
    mixes = refused_mixes(full)
    if mixes:
        raise Refused(f"{mixes[0]}; export one at a time")
    return rows, finals


def _attempts_by_experiment(snap: Snapshot) -> dict[str, list[dict[str, Any]]]:
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in snap.attempts:
        attempts[attempt["experiment_id"]].append(attempt)
    return attempts


def _experiment_rows(snap: Snapshot) -> list[dict[str, Any]]:
    """The planned experiments: the accounting denominator."""
    attempts = _attempts_by_experiment(snap)
    unfinished = {c["id"] for c in snap.campaigns if c["state"] != "FINISHED"}
    state = {"DONE": "done", "UNRESOLVED": "unresolved"}
    rows = []
    for e in snap.experiments.values():
        own = attempts[e["id"]]
        last = max((a["number"] for a in own), default=0)
        rows.append(
            dict(
                experiment_id=e["id"],
                campaign_id=e["campaign_id"],
                block_id=e["block_id"],
                robot_id=e["robot_id"],
                world_id=e["world_id"],
                spawn_id=e["spawn_id"],
                profile_id=e["profile_id"],
                repetition=e["repetition"],
                state=state.get(e["state"], "not_done"),
                campaign_unfinished=e["campaign_id"] in unfinished,
                final_attempt_id=e["final_attempt_id"],
                n_attempts=len(own),
                n_replaced=sum(1 for a in own if not a["discarded"] and a["number"] < last),
                n_discarded=sum(1 for a in own if a["discarded"]),
            )
        )
    return rows


def _group(experiment: dict[str, Any]) -> tuple[str, ...]:
    return tuple(experiment[key] for key in ACCOUNTING_KEYS)


def _accounting_rows(
    snap: Snapshot,
    experiment_rows: list[dict[str, Any]],
    finals: list[tuple[dict[str, Any], dict[str, Any], str]],
) -> list[dict[str, Any]]:
    """Counts per robot, world, spawn and profile (RD-33; T1 reads them)."""
    counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    keys: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in experiment_rows:
        counts[_group(row)]["planned"] += 1
        counts[_group(row)][row["state"]] += 1
    attempts = _attempts_by_experiment(snap)
    for row in experiment_rows:
        own = attempts[row["experiment_id"]]
        last = max((a["number"] for a in own), default=0)
        for attempt in own:
            record = snap.records[attempt["id"]]
            if attempt["discarded"]:
                counts[_group(row)][
                    f"discarded_{record['discard_trigger'] or 'experiment_stop'}"
                ] += 1
            elif attempt["number"] < last and attempt["outcome"] in (
                "timing_rejection",
                "engine_interruption",
            ):
                counts[_group(row)][f"replaced_{attempt['outcome']}"] += 1
    for attempt, record, report in finals:
        group = _group(snap.experiments[attempt["experiment_id"]])
        counts[group][report] += 1
        if report != "reported":
            continue
        if record["outcome"] in REPORTED_OUTCOMES:
            counts[group][f"final_{record['outcome']}"] += 1
        if record["coverage_eligible"] is False:
            counts[group]["coverage_ineligible"] += 1
        if record["comparison_key"]:
            keys[group].add(record["comparison_key"])
    return [
        dict(
            zip(ACCOUNTING_KEYS, group, strict=True),
            comparison_keys=len(keys[group]),
            **{name: counts[group][name] for name in ACCOUNTING_COUNTS},
        )
        for group in sorted(counts)
    ]


def _per_run(
    workspace: Path,
    staging: Path,
    finals: list[tuple[dict[str, Any], dict[str, Any], str]],
    pairs: list[tuple[str, str]],
    registry: WorldRegistry,
    warnings: list[str],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, list[pa.Table]], dict[str, list[str]]
]:
    """Coverage, ground truth, derived tables, maps, references and raw copies of the final runs."""
    coverage: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    derived: dict[str, list[pa.Table]] = defaultdict(list)
    references: dict[str, list[str]] = defaultdict(list)
    (staging / "maps").mkdir()
    (staging / "references").mkdir()
    for attempt, record, _ in finals:
        if record["record_kind"] != "full":
            continue
        directory = workspace / "attempts" / attempt["id"]
        manifest = materialize_manifest(directory)
        activation = manifest.get("simulation_start")
        activation_ns = round(float(activation) * 1e9) if activation is not None else None
        coverage.extend(_coverage_rows(directory, activation_ns))
        trajectory.extend(_trajectory_rows(directory, manifest))
        for name in ("goals", "explorer_decisions", "gate_evaluations"):
            path = directory / "derived" / f"{name}.v{DERIVED_VERSION}.parquet"
            if path.is_file():
                table = pq.read_table(path)
                derived[name].append(
                    table.add_column(
                        0, "attempt_id", pa.array([attempt["id"]] * table.num_rows, pa.string())
                    )
                )
        _map(directory, staging / "maps" / attempt["id"], manifest)
        reference = _reference(
            workspace, manifest, staging / "references", pairs, registry, warnings
        )
        if reference:
            references[reference].append(attempt["id"])
        _raw(directory, staging / "raw" / attempt["id"], pairs)
    return coverage, trajectory, derived, references


def _profile_rows(
    workspace: Path,
    snap: Snapshot,
    defaults: dict[str, str],
    explorers: dict[str, str],
    packages: dict[str, str],
) -> list[dict[str, Any]]:
    """Effective parameters of every profile revision (keyed by its object) against its explorer's Default
    for the robot: frontier_exploration_ros2's is the profile with the robot's Default hash, an external
    explorer's the profile ``default-<explorer>-<robot>`` (library.Explorer.default_id), when the study ran
    it. ``explorers``: the run records' explorer_id per profile ID; ``packages``: explorer_id (package) per
    profiles.yaml explorer ID (``Library.explorer_ids``), for profiles without a run record."""
    profiles: dict[str, dict[str, Any]] = {}
    for e in snap.experiments.values():
        profile = json.loads(e["specification"])["profile"]
        if profile.get("object") and profile["object"] not in profiles:
            profiles[profile["object"]] = dict(
                load_object(workspace, profile["object"]), id=profile["id"]
            )
    # (robot, profiles.yaml explorer ID) -> that explorer's Default values for the robot
    baseline: dict[tuple[str, str], dict[str, Any]] = {}
    for p in profiles.values():
        explorer = str(p.get("explorer") or DEFAULT_EXPLORER)
        robot = p["robot_id"]
        if explorer == DEFAULT_EXPLORER:
            found = defaults.get(robot) == content_hash(p.get("effective", {}))
        else:
            found = p["id"] == f"default-{explorer}-{robot}"
        if found:
            baseline[(robot, explorer)] = p.get("effective", {})
    rows = []
    for _, profile in sorted(profiles.items()):
        effective = profile.get("effective", {})
        digest = content_hash(effective)
        explorer = str(profile.get("explorer") or DEFAULT_EXPLORER)
        default = baseline.get((profile["robot_id"], explorer))
        robot_default = defaults.get(profile["robot_id"])
        for parameter in sorted(effective):
            known = default is not None and parameter in default
            rows.append(
                dict(
                    profile_id=profile["id"],
                    profile_name=profile.get("name", ""),
                    robot_id=profile["robot_id"],
                    explorer_id=explorers.get(profile["id"], packages.get(explorer, explorer)),
                    parameter_hash=digest,
                    is_baseline=None if robot_default is None else robot_default == digest,
                    parameter=parameter,
                    value=canonical(effective[parameter]),
                    default_value=canonical(default[parameter]) if known and default else None,
                    differs_from_default=default[parameter] != effective[parameter]
                    if known and default
                    else None,
                )
            )
    return rows


def _explorer_packages(workspace: Path) -> dict[str, str]:
    """``Library.explorer_ids`` of the checkout: each profiles.yaml explorer's package (its explorer_id)."""
    try:
        explorers = load_explorers(_checkout(workspace))
    except (OSError, ValueError) as exc:
        raise Refused(f"Explorer profiles of the checkout: {exc}") from exc
    return {identity: explorer.package for identity, explorer in explorers.items()}


def _homogeneity(
    snap: Snapshot, finals: list[tuple[dict[str, Any], dict[str, Any], str]]
) -> dict[str, Any]:
    full = [record for _, record, _ in finals if record["record_kind"] == "full"]
    return homogeneity((c["specification"] for c in snap.campaigns), full, snap.documents)


def _codebook(tables: dict[str, pa.Schema]) -> list[dict[str, str]]:
    """One row per exported column, from the run-record registry, the stream schemas and DESCRIPTIONS."""
    rows = []
    registry = {field.name: field for field in (*FIELDS, *READ_TIME_FIELDS)}
    for table, schema in sorted(tables.items()):
        for column in schema:
            if table == "runs":
                field = registry[column.name]
                unit, null, definition = field.unit, field.null, field.definition
            elif table in DESCRIPTIONS and column.name in DESCRIPTIONS[table]:
                unit, null, definition = DESCRIPTIONS[table][column.name]
            else:
                unit, null, definition = "", "as recorded", f"docs/MEASUREMENTS.md ({table})"
            rows.append(
                dict(
                    table=table,
                    column=column.name,
                    type=str(column.type),
                    unit=unit,
                    null_meaning=null,
                    definition=definition or "docs/MEASUREMENTS.md",
                    contract=f"run-record-v{RUN_RECORD_VERSION}, export-v{EXPORT_VERSION}",
                )
            )
    return rows


README = """# Study export {study_id}

Frozen export of the study `{study_id}` ({evidence_level}) of the frontier_exploration_ros2 benchmark suite,
export format {export_version}. Every file is listed with its SHA-256 in `SHA256SUMS`; verify with
`sha256sum -c SHA256SUMS`.

- `runs`: one row per attempt (the run record `run-record-v{record_version}` plus its finality and report class).
- `experiments`: the planned profile × block experiments, the accounting denominator.
- `endpoints`, `coverage`, `trajectory_gt`, `goals`, `accounting`, `explorer_decisions`, `gate_evaluations`,
  `profiles`, `worlds`: see `codebook.csv`, which describes every column.
- `maps/`, `references/`: final SLAM maps (with the map-to-world alignment) and the coverage references, as
  Nav2 map images with YAML; each reference also holds its official world descriptors (`descriptors.json`) and
  region raster (`regions.png`, 16-bit, row 0 = largest y, 0 outside the free floor) when the registry pins them.
- `raw/`: the recorded streams, manifest and logs of every final run, absolute paths replaced.
- `study.yaml`, `provenance.json`: the study's identity, campaigns, analysis block, code and run environment,
  and the homogeneity check; `plan.md`: the study plan as it was at export, when the study names one.

Carry-forward of completed runs and the 2 s coverage gap rule are applied in the analysis, not in the data.
Regenerate the analysis outputs with the analysis package of the same checkout, from its `analysis/` directory:
`../.venv-analysis/bin/python -m benchmark_analysis --export <this directory>` (outputs go beside the export, under
`analysis/`, or to `--out <new directory>`).
"""
LICENSE = """Data in this export: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. The benchmark code is Apache-2.0
(https://www.apache.org/licenses/LICENSE-2.0). Third-party models used by the worlds are referenced, never
included.
"""


def _plan(
    workspace: Path,
    staging: Path,
    study: dict[str, Any],
    registered: list[str],
    pairs: list[tuple[str, str]],
    worlds: dict[str, WorldFacts],
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Copy the study's plan from the checkout to ``plan.md`` (RO-15); its analysis block, the hash of its
    registered part now, and warnings. The block is passed on only when that part still has the one hash
    every campaign registered, so the analysis never reads parameters the study did not run under."""
    if not study.get("plan"):
        return None, None, []
    checkout = _checkout(workspace)
    path = (checkout / study["plan"]).resolve()
    if not path.is_relative_to(checkout) or not path.is_file():
        return (
            None,
            None,
            ["study plan: not found in the checkout; plan.md and the analysis block are missing"],
        )
    text = path.read_text(encoding="utf-8")
    (staging / "plan.md").write_text(scrub(text, pairs), encoding="utf-8")
    try:
        plan = plan_parts(text)
    except ContractError as exc:
        return None, None, [f"study plan: {exc}; the analysis block is left out"]
    left_out = "the analysis block is left out"
    if registered != [plan.sha256]:
        return (
            None,
            plan.sha256,
            [f"study plan: its registered part changed since submission; {left_out}"],
        )
    confirmatory = study.get("evidence_level") == "confirmatory"
    if plan.block is None:
        warnings = (
            ["study plan: a confirmatory plan without an analysis block"] if confirmatory else []
        )
        return None, plan.sha256, warnings
    try:
        block = validate(plan.block, confirmatory=confirmatory, worlds=worlds)
    except ContractError as exc:
        return None, plan.sha256, [f"study plan: the block no longer validates ({exc}); {left_out}"]
    return block, plan.sha256, []


def _study_files(
    staging: Path,
    study_id: str,
    snap: Snapshot,
    finals: list[tuple[dict[str, Any], dict[str, Any], str]],
    defaults: dict[str, str],
    references: dict[str, list[str]],
    homogeneity: dict[str, Any],
    pairs: list[tuple[str, str]],
    analysis: dict[str, Any] | None,
    plan_sha256_at_export: str | None,
    registry_sha256: str,
    world_warnings: list[str],
) -> None:
    """``study.yaml``, ``provenance.json``, README and licence. ``world_warnings`` (worlds without a registry
    entry or descriptor pin) go under ``provenance.json`` ``world_warnings``, not ``homogeneity``."""
    study = (
        next(
            (
                c["specification"].get("study")
                for c in snap.campaigns
                if c["specification"].get("study")
            ),
            None,
        )
        or {}
    )
    (staging / "study.yaml").write_text(
        scrub(
            yaml.safe_dump(
                dict(
                    study_id=study_id,
                    evidence_level=study.get("evidence_level", "exploratory"),
                    plan=study.get("plan"),
                    plan_sha256=study.get("plan_sha256"),
                    plan_sha256_at_export=plan_sha256_at_export,
                    analysis=analysis,
                    export_version=EXPORT_VERSION,
                    run_record_version=RUN_RECORD_VERSION,
                    campaigns=[
                        dict(
                            id=c["id"],
                            name=c["name"],
                            state=c["state"],
                            first_repetition=(c["specification"].get("study") or {}).get(
                                "first_repetition", 0
                            ),
                            repetitions=c["specification"]["repetitions"],
                            concurrency=c["specification"]["concurrency"],
                            world_time_limits=c["specification"].get("world_time_limits"),
                            # I4: the rule that set the limits and its inputs (null without the rule).
                            time_limit_rule=c["specification"].get("time_limit_rule"),
                            time_limit_inputs=c["specification"].get("time_limit_inputs"),
                            debug_streams=bool(c["specification"].get("debug_streams")),
                        )
                        for c in snap.campaigns
                    ],
                    # The world registry the export read (worlds table, descriptor pins); I1 R-3.
                    world_registry_sha256=registry_sha256,
                    default_parameter_hashes=dict(sorted(defaults.items())),
                    attempts=len(snap.attempts),
                    experiments=len(snap.experiments),
                    final_runs=len(finals),
                ),
                sort_keys=True,
            ),
            pairs,
        )
    )
    used = sorted({record["provenance_id"] for _, record, _ in finals if record["provenance_id"]})
    (staging / "provenance.json").write_text(
        scrub(
            json.dumps(
                dict(
                    documents={identity: snap.documents.get(identity) for identity in used},
                    references=dict(sorted((k, sorted(v)) for k, v in references.items())),
                    homogeneity=homogeneity,
                    world_warnings=world_warnings,
                ),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            pairs,
        )
    )
    (staging / "README.md").write_text(
        README.format(
            study_id=study_id,
            evidence_level=study.get("evidence_level", "exploratory"),
            export_version=EXPORT_VERSION,
            record_version=RUN_RECORD_VERSION,
        )
    )
    (staging / "LICENSE").write_text(LICENSE)


def _texts(path: Path) -> Iterator[str]:
    """The text an exported file holds (string columns of Parquet, decompressed gzip, plain text)."""
    if path.suffix == ".parquet":
        schema = pq.read_schema(path)
        names = [field.name for field in schema if pa.types.is_string(field.type)]
        if names:
            for column in pq.read_table(path, columns=names).columns:
                yield from (value for value in column.to_pylist() if value)
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as handle:
            yield handle.read()
    elif path.suffix not in (".pgm", ".png"):
        yield path.read_text(errors="replace")


def _seal(staging: Path, pairs: list[tuple[str, str]], host_name: str) -> int:
    """Refuse a leak (an absolute home or checkout path, or the host name, D-27), then write
    ``SHA256SUMS``; returns the number of files listed."""
    forbidden = [old for old, _ in pairs] + ["/home/"]
    host = (
        re.compile(rf"(?<![A-Za-z0-9]){re.escape(host_name)}(?![A-Za-z0-9])", re.IGNORECASE)
        if len(host_name) >= 3
        else None
    )
    paths = sorted(path for path in staging.rglob("*") if path.is_file())
    for path in paths:
        for text in _texts(path):
            if any(value in text for value in forbidden):
                raise Refused(f"{path.relative_to(staging)} still holds an absolute path")
            if host is not None and host.search(text):
                raise Refused(f"{path.relative_to(staging)} holds the host name")
    (staging / "SHA256SUMS").write_text(
        "".join(f"{file_hash(path)}  {path.relative_to(staging).as_posix()}\n" for path in paths)
    )
    return len(paths)


def _registry_matches_runs(
    registry: WorldRegistry,
    sources: dict[str, set[str | None]],
    finals: list[tuple[dict[str, Any], dict[str, Any], str]],
    documents: dict[str, Any],
) -> list[str]:
    """Check the checkout's registry, which gives the ``worlds`` table and the descriptor pins, against
    what the runs recorded. ``sources``: per world, the ``world_yaml_sha256`` of each full run's manifest
    (``configuration.scenario.source``; null for tier A). A collection world whose registry world.yaml
    hash differs from a run's is refused, because its row (family, design point) could describe another
    world; tier A has no world.yaml and is not checked. A registry file that differs from the one the
    engine read (provenance ``world_registry_sha256``) only warns."""
    for world_id in sorted(sources):
        try:
            source = registry.world(world_id).source
        except ValueError:
            continue  # not in the registry: warned with the worlds table
        if source is None:
            continue
        recorded = sources[world_id] - {source.world_yaml_sha256}
        if None in recorded:
            raise Refused(
                f"World {world_id}: some runs recorded no world.yaml hash, so the worlds row cannot be "
                "verified against them"
            )
        if recorded:
            found = ", ".join(sorted(str(value)[:8] for value in recorded))
            raise Refused(
                f"World {world_id}: the registry's world.yaml hash {source.world_yaml_sha256[:8]} "
                f"differs from the one its runs recorded ({found}); export with the registry the runs "
                "used"
            )
    used = {record["provenance_id"] for _, record, _ in finals if record["provenance_id"]}
    hashes = {(documents.get(identity) or {}).get("world_registry_sha256") for identity in used}
    # provenance version 1 (before the registry) records no hash: nothing to compare
    other = sorted(str(value)[:8] for value in hashes - {registry.sha256, None})
    if not other:
        return []
    return [
        f"world registry: the export read {registry.sha256[:8]}, the runs' engines read "
        f"{', '.join(other)}; the worlds table and descriptor pins come from the export's"
    ]


def _build(workspace: Path, study_id: str, staging: Path, host_name: str) -> dict[str, Any]:
    snap = snapshot(workspace, study_id)
    run_rows, finals = _run_rows(snap)
    recorded = []
    sources: dict[str, set[str | None]] = defaultdict(set)
    for attempt, record, _ in finals:
        if record["record_kind"] == "full":
            configuration = (
                materialize_manifest(workspace / "attempts" / attempt["id"]).get("configuration")
                or {}
            )
            recorded += [
                (str(configuration.get("project_root") or ""), "<checkout>"),
                (str(configuration.get("suite_workspace") or ""), "<workspace>"),
            ]
            source = (configuration.get("scenario") or {}).get("source") or {}
            sources[record["world_id"]].add(source.get("world_yaml_sha256"))
    pairs = _scrubber(workspace, recorded)
    try:
        registry = WorldRegistry.load(_checkout(workspace))
    except ValueError as exc:
        raise Refused(f"World registry of the checkout: {exc}") from exc
    world_warnings = _registry_matches_runs(registry, sources, finals, snap.documents)
    tables: dict[str, pa.Schema] = {}
    tables["runs"] = _write(
        staging,
        "runs",
        pa.Table.from_pylist(run_rows, schema=_schema((*FIELDS, *READ_TIME_FIELDS))),
        pairs,
    )
    experiment_rows = _experiment_rows(snap)
    types = dict(
        repetition=pa.int64(),
        campaign_unfinished=pa.bool_(),
        n_attempts=pa.int64(),
        n_replaced=pa.int64(),
        n_discarded=pa.int64(),
    )
    tables["experiments"] = _write(
        staging,
        "experiments",
        pa.Table.from_pylist(
            experiment_rows, schema=_schema_of(DESCRIPTIONS["experiments"], types)
        ),
        pairs,
    )
    endpoint_rows = [row for _, record, report in finals for row in _endpoint_rows(record, report)]
    endpoint_types = {"time_s": pa.float64(), "event": pa.bool_(), "limit_s": pa.float64()}
    tables["endpoints"] = _write(
        staging,
        "endpoints",
        pa.Table.from_pylist(
            endpoint_rows, schema=_schema_of(DESCRIPTIONS["endpoints"], endpoint_types)
        ),
        pairs,
    )
    accounting_schema = pa.schema(
        [(name, pa.string()) for name in ACCOUNTING_KEYS]
        + [("comparison_keys", pa.int64())]
        + [(name, pa.int64()) for name in ACCOUNTING_COUNTS]
    )
    tables["accounting"] = _write(
        staging,
        "accounting",
        pa.Table.from_pylist(
            _accounting_rows(snap, experiment_rows, finals), schema=accounting_schema
        ),
        pairs,
    )
    coverage, trajectory, derived, references = _per_run(
        workspace, staging, finals, pairs, registry, world_warnings
    )
    used_worlds = {record["world_id"] for _, record, _ in finals}
    try:
        world_rows = [row for row in registry.worlds_table() if row["world_id"] in used_worlds]
    except (OSError, ValueError) as exc:
        raise Refused(f"World registry of the checkout, worlds table: {exc}") from exc
    world_warnings += [
        f"world {world}: not in the registry; it has no row in the worlds table"
        for world in sorted(used_worlds - {row["world_id"] for row in world_rows})
    ]
    tables["worlds"] = _write(
        staging,
        "worlds",
        pa.Table.from_pylist(
            world_rows, schema=_schema_of(DESCRIPTIONS["worlds"], {"seed": pa.int64()})
        ),
        pairs,
    )
    coverage_types = {
        "t_s": pa.float64(),
        "t_activation_s": pa.float64(),
        "coverage_pct": pa.float64(),
        "is_terminal": pa.bool_(),
    }
    tables["coverage"] = _write(
        staging,
        "coverage",
        pa.Table.from_pylist(coverage, schema=_schema_of(DESCRIPTIONS["coverage"], coverage_types)),
        pairs,
        csv_mirror=False,
    )
    trajectory_types = {
        name: pa.float64() for name in DESCRIPTIONS["trajectory_gt"] if name != "attempt_id"
    }
    tables["trajectory_gt"] = _write(
        staging,
        "trajectory_gt",
        pa.Table.from_pylist(
            trajectory, schema=_schema_of(DESCRIPTIONS["trajectory_gt"], trajectory_types)
        ),
        pairs,
        csv_mirror=False,
    )
    for name, schema in (
        ("goals", GOALS_SCHEMA),
        ("explorer_decisions", DECISIONS_SCHEMA),
        ("gate_evaluations", GATES_SCHEMA),
    ):
        empty = pa.schema([("attempt_id", pa.string()), *schema]).empty_table()
        tables[name] = _write(
            staging, name, pa.concat_tables(derived[name]) if derived[name] else empty, pairs
        )
    defaults: dict[str, str] = {}
    for campaign in snap.campaigns:
        defaults.update(campaign["specification"].get("default_parameter_hashes") or {})
    tables["profiles"] = _write(
        staging,
        "profiles",
        pa.Table.from_pylist(
            _profile_rows(
                workspace,
                snap,
                defaults,
                {str(row["profile_id"]): str(row["explorer_id"]) for row in run_rows},
                _explorer_packages(workspace),
            ),
            schema=_schema_of(
                DESCRIPTIONS["profiles"],
                {"is_baseline": pa.bool_(), "differs_from_default": pa.bool_()},
            ),
        ),
        pairs,
    )
    homogeneity = _homogeneity(snap, finals)
    studies = [c["specification"].get("study") or {} for c in snap.campaigns]
    try:
        facts = world_facts(registry)  # the same facts Library uses at submission
    except ContractError as exc:
        raise Refused(f"World registry of the checkout: {exc}") from exc
    analysis, digest, plan_warnings = _plan(
        workspace,
        staging,
        next((study for study in studies if study), {}),
        sorted({str(study.get("plan_sha256")) for study in studies}),
        pairs,
        facts,
    )
    homogeneity["warnings"] += plan_warnings
    _study_files(
        staging,
        study_id,
        snap,
        finals,
        defaults,
        references,
        homogeneity,
        pairs,
        analysis,
        digest,
        registry.sha256,
        world_warnings,
    )
    with (staging / "codebook.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, ["table", "column", "type", "unit", "null_meaning", "definition", "contract"]
        )
        writer.writeheader()
        writer.writerows(_codebook(tables))
    listed = _seal(staging, pairs, host_name)
    return dict(
        study_id=study_id,
        files=listed + 1,
        bytes=sum(p.stat().st_size for p in staging.rglob("*") if p.is_file()),
        final_runs=len(finals),
        warnings=homogeneity["warnings"] + world_warnings,
    )


def export_study(payload: dict[str, Any]) -> dict[str, Any]:
    """Build one export of a study (job side); returns its receipt."""
    workspace = Path(payload["workspace"]).resolve()
    study_id = payload["study_id"]
    parent = studies_root(workspace) / study_id / "exports"
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / (payload["stamp"] + "." + uuid.uuid4().hex[:8] + ".staging")
    staging.mkdir()
    try:
        receipt = _build(workspace, study_id, staging, payload.get("host_name") or "")
        digest = file_hash(staging / "SHA256SUMS")
        final = parent / f"{payload['stamp']}-{digest[:8]}"
        if final.is_dir() and file_hash(final / "SHA256SUMS") == digest:
            # The same data exported within the same second: that export already holds these files.
            shutil.rmtree(staging)
        else:
            for root, _, _ in os.walk(staging, topdown=False):
                sync_directory(Path(root))
            staging.rename(final)
            sync_directory(parent)
        return dict(
            receipt, export_id=payload.get("export_id"), path=str(final), sums_sha256=digest
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


class Exports:
    """Engine side: one export job at a time; its state lives in events and the export directory."""

    def __init__(self, store: Store, workspace: Path):
        self.store = store
        self.workspace = workspace
        self.job: "Job | None" = None
        self.export_id: str | None = None
        self.study_id: str | None = None
        self.started = 0.0
        # An export interrupted by an engine stop: its staging goes and it is recorded as failed, unless
        # the job had already renamed it (the rename is its last step, so the export is complete).
        root = studies_root(workspace)
        if root.is_dir():
            for staging in root.glob("*/exports/*.staging"):
                shutil.rmtree(staging, ignore_errors=True)
        for row in store.rows(
            "SELECT payload FROM events s WHERE kind='study.export_started' AND NOT EXISTS ("
            "SELECT 1 FROM events t WHERE t.kind IN ('study.exported','study.export_failed') "
            "AND json_extract(t.payload,'$.export_id')=json_extract(s.payload,'$.export_id'))"
        ):
            started = json.loads(row["payload"])
            stamp = started.get("stamp")
            finished = (
                sorted((root / started["study_id"] / "exports").glob(f"{stamp}-*/SHA256SUMS"))
                if stamp
                else []
            )
            if finished:
                sums = finished[-1]
                store.event(
                    "study.exported",
                    dict(
                        started, path=str(sums.parent), sums_sha256=file_hash(sums), recovered=True
                    ),
                )
            else:
                store.event(
                    "study.export_failed",
                    dict(started, detail="The engine stopped during the export"),
                )

    def start(self, study_id: str, busy: bool) -> dict[str, Any]:
        """Start an export; refused while runs or other jobs are active (RO-10) or another export runs."""
        if self.job is not None:
            raise ContractError("study_id", "Another export is running", "conflict")
        if busy:
            raise ContractError(
                "study_id",
                "Runs or other jobs are active; export once they have finished",
                "conflict",
            )
        from .jobs import Job

        export_id = str(uuid.uuid4())
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.job = Job(
            "study_export",
            dict(
                workspace=str(self.workspace),
                study_id=study_id,
                export_id=export_id,
                stamp=stamp,
                host_name=os.environ.get("BENCHMARK_HOST_NAME", ""),
            ),
            uuid.uuid4().hex,
        )
        self.export_id, self.study_id, self.started = export_id, study_id, time.monotonic()
        self.store.event(
            "study.export_started", {"study_id": study_id, "export_id": export_id, "stamp": stamp}
        )
        return dict(study_id=study_id, export_id=export_id, accepted=True)

    def poll(self) -> None:
        if self.job is None:
            return
        result = self.job.poll()
        if result is None:
            return
        self.job.close()
        identity = dict(study_id=self.study_id, export_id=self.export_id)
        self.job, self.study_id, self.export_id = None, None, None
        if result["ok"]:
            receipt = dict(result["result"], wall_s=round(time.monotonic() - self.started, 1))
            self.store.event("study.exported", dict(receipt, **identity))
        else:
            self.store.event("study.export_failed", dict(identity, detail=result["detail"]))

    def status(self, study_id: str, export_id: Any = None) -> dict[str, Any]:
        """An export's state (``export_id``), else the study's running or latest export."""
        if (
            self.job is not None
            and self.study_id == study_id
            and export_id in (None, self.export_id)
        ):
            return dict(study_id=study_id, export_id=self.export_id, state="running")
        row = self.store.one(
            "SELECT kind,payload,utc_ns FROM events WHERE kind IN ('study.exported','study.export_failed') "
            "AND json_extract(payload,'$.study_id')=? "
            "AND (? IS NULL OR json_extract(payload,'$.export_id')=?) ORDER BY sequence DESC LIMIT 1",
            (study_id, export_id, export_id),
        )
        if row is None:
            return dict(study_id=study_id, export_id=export_id, state="none")
        state = "finished" if row["kind"] == "study.exported" else "failed"
        return dict(json.loads(row["payload"]), state=state, utc_ns=row["utc_ns"])

    def close(self) -> None:
        if self.job is not None:
            self.job.close()
            self.job = None
