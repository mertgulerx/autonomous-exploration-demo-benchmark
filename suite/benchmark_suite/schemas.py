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
        ("world_x", pa.float64()),
        ("world_y", pa.float64()),
        ("world_yaw", pa.float64()),
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
    "quality": [
        ("category", pa.string()),
        ("observation", pa.string()),
        ("value", pa.float64()),
        ("persistent", pa.bool_()),
        ("payload_json", pa.string()),
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

SCHEMAS = {
    kind: pa.schema(
        [pa.field(name, typ, nullable=nullable) for name, typ, nullable in COMMON]
        + [pa.field(name, typ, nullable=True) for name, typ in fields],
        metadata={
            b"schema_version": str(DATASET_VERSION).encode(),
            b"time_units": b"nanoseconds",
            b"distance_units": b"metres",
            b"kind": kind.encode(),
        },
    )
    for kind, fields in FIELDS.items()
}
