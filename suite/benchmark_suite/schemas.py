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

import pyarrow as pa

from .contracts import DATASET_VERSION

IDENTITIES = [
    (name, pa.string(), False)
    for name in (
        "campaign_id",
        "experiment_id",
        "attempt_id",
        "robot_id",
        "world_id",
        "profile_id",
        "spawn_id",
    )
]
COMMON = IDENTITIES + [
    ("repetition", pa.int32(), False),
    ("source", pa.string(), False),
    ("source_ns", pa.int64(), True),
    ("clock_ns", pa.int64(), True),
    # Dataset schema 10: the worker's /clock when the row was made (null before the first /clock message).
    ("receipt_clock_ns", pa.int64(), True),
    ("receipt_utc_ns", pa.int64(), False),
    ("receipt_monotonic_ns", pa.int64(), False),
    ("simulation_elapsed_ns", pa.int64(), True),
    ("valid", pa.bool_(), False),
    ("quality_flags", pa.list_(pa.string()), False),
]

FIELDS = {
    "resources": [
        ("cpu_percent", pa.float64()),
        ("rss_mib", pa.float64()),
        ("cpu_average_percent", pa.float64()),
        ("rss_average_mib", pa.float64()),
        ("cpu_seconds", pa.float64()),
        ("distance_m", pa.float64()),
        ("rtf", pa.float64()),
        ("coverage_percent", pa.float64()),
        ("scope", pa.string()),
        ("pid", pa.int64()),
    ],
    "coverage": [
        ("stage", pa.string()),
        ("coverage_percent", pa.float64()),
        ("covered_m2", pa.float64()),
        ("target_m2", pa.float64()),
        ("status", pa.string()),
        ("map_source_ns", pa.int64()),
        ("map_receipt_utc_ns", pa.int64()),
        ("map_receipt_monotonic_ns", pa.int64()),
        ("map_age_ns", pa.int64()),
    ],
    "trajectory": [
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("yaw", pa.float64()),
        ("vx", pa.float64()),
        ("vy", pa.float64()),
        ("wz", pa.float64()),
        ("speed", pa.float64()),
        ("distance_m", pa.float64()),
        ("frame", pa.string()),
        # Odometry through the alignment frozen at readiness, not a ground-truth pose (schema 9: world_*).
        ("odom_aligned_x_m", pa.float64()),
        ("odom_aligned_y_m", pa.float64()),
        ("odom_aligned_yaw_rad", pa.float64()),
        ("command_vx", pa.float64()),
        ("command_vy", pa.float64()),
        ("command_wz", pa.float64()),
    ],
    "commands": [
        ("vx", pa.float64()),
        ("vy", pa.float64()),
        ("wz", pa.float64()),
        ("topic", pa.string()),
    ],
    "ground_truth": [
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("z", pa.float64()),
        ("yaw", pa.float64()),
        ("vx", pa.float64()),
        ("vy", pa.float64()),
        ("wz", pa.float64()),
        ("speed", pa.float64()),
        ("frame", pa.string()),
        ("model_to_base_x", pa.float64()),
        ("model_to_base_y", pa.float64()),
        ("model_to_base_yaw", pa.float64()),
        ("odometry_drift_m", pa.float64()),
    ],
    # Events: the navigator telemetry's goal events, the debug stream's "feedback" and, since schema 11,
    # "recovery" (one row per new recovery count of a goal; goal_facts.py). Readers filter by event.
    "navigation": [
        ("event", pa.string()),
        ("goal_sequence", pa.int64()),
        ("native_uuid", pa.string()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("yaw", pa.float64()),
        ("frame", pa.string()),
        ("status", pa.string()),
        ("native_code", pa.int32()),
        ("distance_remaining", pa.float64()),
        ("recoveries", pa.int32()),
        ("lost_events", pa.int64()),
        ("event_sequence", pa.int64()),
    ],
    "events": [("event", pa.string()), ("payload_json", pa.string())],
    # One row per Nav2 goal sequence from the worker's goal accumulator (schema 10).
    "goal_facts": [
        ("goal_sequence", pa.int64()),
        ("native_uuid", pa.string()),
        ("dispatch_clock_ns", pa.int64()),
        ("end_clock_ns", pa.int64()),
        ("status", pa.string()),
        ("recoveries", pa.int32()),
        ("min_distance_remaining_m", pa.float64()),
        ("feedback_messages", pa.int64()),
    ],
    "quality": [
        ("category", pa.string()),
        ("observation", pa.string()),
        ("value", pa.float64()),
        ("persistent", pa.bool_()),
        ("payload_json", pa.string()),
    ],
    "decisions": [
        ("tick", pa.int64()),
        ("mode", pa.string()),
        ("robot_x", pa.float64()),
        ("robot_y", pa.float64()),
        ("robot_yaw", pa.float64()),
        ("raw_frontiers", pa.int32()),
        ("candidates", pa.int32()),
        ("pool", pa.int32()),
        ("route", pa.list_(pa.uint16())),
        ("route_cost", pa.float64()),
        ("target", pa.uint16()),
        ("greedy_first", pa.uint16()),
        ("analysis_wall_ms", pa.float64()),
        ("skipped_ticks", pa.int64()),
        ("map_source_ns", pa.int64()),
        ("chunk_size_cells", pa.int32()),
        ("chunk_resolution", pa.float64()),
        ("chunk_origin_x", pa.float64()),
        ("chunk_origin_y", pa.float64()),
        ("chunk_map_width", pa.int32()),
        ("chunk_map_height", pa.int32()),
        ("chunk_columns", pa.int32()),
        ("chunk_rows", pa.int32()),
        ("chunk_states", pa.binary()),
        ("chunk_hits", pa.int32()),
        ("chunk_rebuilt", pa.int32()),
        ("chunk_resets", pa.int32()),
    ],
    "frontier_candidates": [
        ("tick", pa.int64()),
        ("candidate", pa.uint16()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("size_cells", pa.int32()),
        ("gain", pa.float64()),
        ("path_cost", pa.float64()),
        ("time_cost", pa.float64()),
        ("start_cost", pa.float64()),
        ("distance_m", pa.float64()),
        ("score_order", pa.uint16()),
        ("pool_rank", pa.uint16()),
        ("route_position", pa.uint16()),
        ("greedy_position", pa.uint16()),
    ],
    "explored_cells": [
        ("kind", pa.string()),
        ("reference_cells", pa.int64()),
        ("free_cells", pa.int64()),
        ("changed_cells", pa.int64()),
        ("mask", pa.binary()),
        ("map_source_ns", pa.int64()),
    ],
    "incidents": [
        ("type", pa.string()),
        ("detector_version", pa.string()),
        ("start_ns", pa.int64()),
        ("end_ns", pa.int64()),
        ("trajectory_start_row", pa.int64()),
        ("trajectory_end_row", pa.int64()),
        ("duration_ns", pa.int64()),
        ("distance_m", pa.float64()),
        ("coverage_change_percent", pa.float64()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("bbox", pa.list_(pa.float64())),
        ("evidence_json", pa.string()),
    ],
}

# Schema-9 names of fields that schema 10 renamed (D-22); readers map them (datasets.stream_relation).
LEGACY_FIELDS = {
    "trajectory": {
        "world_x": "odom_aligned_x_m",
        "world_y": "odom_aligned_y_m",
        "world_yaw": "odom_aligned_yaw_rad",
    }
}
# Fields schema 10 added to every stream, with their SQL type for reading older shards.
ADDED_FIELDS = (("receipt_clock_ns", "BIGINT"),)
# Streams schema 10 added.
ADDED_KINDS = frozenset({"goal_facts"})


def build(version: int) -> dict[str, pa.Schema]:
    """Arrow schemas of the streams as written by dataset schema ``version`` (9, 10 or 11).

    Schema 11 added rows (``navigation`` recovery events), not fields: its layout is schema 10's, and only
    the ``schema_version`` metadata tells them apart.
    """
    added = {name for name, _ in ADDED_FIELDS} if version < 10 else set()
    common = [field for field in COMMON if field[0] not in added]
    legacy = {} if version >= 10 else LEGACY_FIELDS
    return {
        kind: pa.schema(
            [pa.field(name, typ, nullable=nullable) for name, typ, nullable in common]
            + [
                pa.field(
                    {new: old for old, new in legacy.get(kind, {}).items()}.get(name, name),
                    typ,
                    nullable=True,
                )
                for name, typ in fields
            ],
            metadata={
                b"schema_version": str(version).encode(),
                b"time_units": b"nanoseconds",
                b"distance_units": b"metres",
                b"kind": kind.encode(),
            },
        )
        for kind, fields in FIELDS.items()
        if version >= 10 or kind not in ADDED_KINDS
    }


SCHEMAS = build(DATASET_VERSION)
# Every schema a recorded shard may have, by dataset schema version (older versions were written as 9).
# Schema 10 stays listed so its shards keep their version through compaction and in run records.
SCHEMAS_BY_VERSION = {9: build(9), 10: build(10), DATASET_VERSION: SCHEMAS}
