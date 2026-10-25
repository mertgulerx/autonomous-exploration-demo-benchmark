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

from copy import deepcopy
from io import BytesIO
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from exploration_benchmark_runner.coverage import COVERAGE_STOP_PERCENT, COVERAGE_STOP_RULE

from .contracts import DATASET_VERSION
from .schemas import SCHEMAS
from .storage import atomic_bytes, atomic_json, file_hash, object_store, put_object, sync_directory
from .timing import TimingPolicy

TIMING_RULE = "timing-rule-v2"  # the recorder's windows decide rejection (docs/MEASUREMENTS.md)
# Streams recorded only when a campaign turns debug streams on (RESULTS_RESEARCH RD-42).
DEBUG_STREAMS = ("navigation_feedback", "commands")

INSIGHT_KINDS = frozenset({"decisions", "frontier_candidates", "explored_cells"})
QUEUE_SIZE = 8192
INSIGHT_QUEUE_LIMIT = QUEUE_SIZE * 7 // 8


def safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def nanoseconds(value: float | None) -> int | None:
    return round(value * 1e9) if value is not None and math.isfinite(value) else None


class Recorder:
    def __init__(self, directory: str, configuration: dict[str, Any], reference: Any):
        self.directory = Path(directory)
        self.configuration = configuration
        self.identity = configuration["suite_identity"]
        self.objects = Path(configuration["suite_workspace"]) / "objects"
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue(QUEUE_SIZE)
        self.error: Exception | None = None
        self.dropped = 0
        # Explorer insight rows are Monitor and replay data: they yield to measurement rows when the queue
        # runs high and are counted apart, so they never affect eligibility.
        self.dropped_insight = 0
        self.frozen = False
        self.closed = False
        self.coverage_samples = 0
        self.timing = TimingPolicy()
        self.resource_peak_rss: float | None = None
        self.resource_previous: tuple[float, float, float] | None = None
        self.resource_integral_cpu = self.resource_integral_rss = self.resource_duration = 0.0
        self.origin: float | None = None
        # The worker's latest /clock in seconds, or None before the first message (dataset schema 10:
        # every row's receipt clock, and the clock of rows without a source stamp; D-21).
        self.clock_source: Callable[[], float | None] = lambda: None
        self.segments: list[dict[str, Any]] = []
        self.manifest = dict(
            schema_version=DATASET_VERSION,
            source="gazebo",
            phase="STARTING",
            identity=self.identity,
            configuration=configuration,
            seed=1234,
            created_utc_ns=time.time_ns(),
            runtime=configuration.get("runtime", {}),
            robot=configuration["robot"],
            requested_parameters=configuration["requested_parameters"],
            effective_parameters=configuration["effective_parameters"],
            coverage=dict(
                metric_version=reference.metadata["metric_version"],
                reference=reference.descriptor(),
                definition="100 * current SLAM-free reference cells / fixed target cells",
                free_value=0,
                sample_interval_simulation_seconds=1.0,
                stale_delivery_wall_seconds=5.0,
                alignment=None,
                baseline=None,
                milestones=[],
                quality_flags=[],
                samples=0,
                time_origin="first_raw_odometry_movement",
                monotonic=False,
                threshold_policy="first_valid_observed_crossing_no_interpolation",
            ),
            policy=dict(
                simulation_time_limit_seconds=configuration.get("simulation_time_limit"),
                rtf_threshold=0.90,
                window_wall_seconds=5,
                persistence_wall_seconds=10,
                no_movement_simulation_seconds=180,
                coverage_stop_rule=COVERAGE_STOP_RULE,
                coverage_stop_percent=COVERAGE_STOP_PERCENT,
                stationary_warmup_seconds=0,
                trajectory_maximum_hz=20,
                flush_wall_seconds=5,
            ),
            timing_rule=TIMING_RULE,
            debug_streams=list(DEBUG_STREAMS) if configuration.get("debug_streams") else [],
        )
        self.thread = threading.Thread(
            target=self._write, args=(deepcopy(self.manifest),), name="suite-recorder", daemon=False
        )
        self.thread.start()
        for name, data in {
            "coverage_metadata": reference.metadata_bytes,
            "coverage_grid": reference.grid_bytes,
        }.items():
            self.enqueue("object", (name, data))
        self.flush()

    def enqueue(self, kind: str, value: Any) -> bool:
        """Queue one item for the writer; return whether it was queued (rows can be dropped)."""
        if self.closed:
            return False
        if self.error:
            raise OSError("Recording failed: " + str(self.error)) from self.error
        if kind in ("manifest", "map", "object", "finish"):
            try:
                self.queue.put((kind, value), timeout=10)
            except queue.Full as exc:
                raise OSError("Recording checkpoint queue did not drain") from exc
            return True
        insight = kind in INSIGHT_KINDS
        try:
            if insight and self.queue.qsize() >= INSIGHT_QUEUE_LIMIT:
                raise queue.Full
            self.queue.put_nowait((kind, value))
        except queue.Full:
            if insight:
                self.dropped_insight += 1
            else:
                self.dropped += 1
            return False
        return True

    def row(
        self,
        source: str,
        source_ns: int | None = None,
        *,
        valid: bool = True,
        receipt_monotonic_ns: int | None = None,
        receipt_utc_ns: int | None = None,
        clock_ns: int | None = None,
    ) -> dict[str, Any]:
        receipt = receipt_monotonic_ns or time.monotonic_ns()
        motion_sim = self.origin
        receipt_clock = nanoseconds(self.clock_source())
        clock = clock_ns if clock_ns is not None else receipt_clock
        return dict(
            self.identity,
            source=source,
            source_ns=source_ns,
            clock_ns=clock,
            receipt_clock_ns=receipt_clock,
            receipt_monotonic_ns=receipt,
            receipt_utc_ns=receipt_utc_ns or time.time_ns(),
            valid=valid,
            simulation_elapsed_ns=clock - round(motion_sim * 1e9)
            if clock is not None and motion_sim is not None and clock >= round(motion_sim * 1e9)
            else None,
            quality_flags=[],
        )

    def observation(self, kind: str, row: dict[str, Any]) -> bool:
        """Queue a stream row unless frozen; return whether it was queued."""
        return not self.frozen and self.enqueue(kind, safe(row))

    def event(self, event: str, **details: Any) -> None:
        self.enqueue(
            "events",
            dict(
                self.row("benchmark"),
                event=event,
                payload_json=json.dumps(safe(details), allow_nan=False),
            ),
        )

    def sample(self, metrics: dict[str, Any]) -> None:
        row = self.row("process_accounting")
        row.update(
            {
                key: metrics.get(key)
                for key in (
                    "cpu_percent",
                    "rss_mib",
                    "cpu_average_percent",
                    "rss_average_mib",
                    "cpu_seconds",
                    "distance_m",
                    "rtf",
                    "coverage_percent",
                )
            }
        )
        row.update(
            scope="explorer",
            pid=metrics.get("explorer_pid"),
            quality_flags=metrics.get("quality_flags", []),
        )
        if metrics.get("explorer_resources_available", metrics.get("explorer_pid") is not None):
            now = time.monotonic()
            cpu, rss = metrics.get("cpu_percent"), metrics.get("rss_mib")
            if cpu is not None and rss is not None:
                self.resource_peak_rss = max(self.resource_peak_rss or 0.0, rss)
                if self.resource_previous:
                    previous, previous_cpu, previous_rss = self.resource_previous
                    dt = now - previous
                    if 0 < dt <= 2.5:
                        self.resource_integral_cpu += dt * previous_cpu
                        self.resource_integral_rss += dt * previous_rss
                        self.resource_duration += dt
                self.resource_previous = now, cpu, rss
        else:
            self.resource_previous = None
            row.update(cpu_percent=None, rss_mib=None)
        self.enqueue("resources", safe(row))
        clock = self.clock_source()
        if clock is not None:
            self.timing.update(time.monotonic(), clock)
        self.enqueue(
            "quality",
            dict(
                self.row("timing_policy"),
                category="timing",
                observation="rtf_window",
                value=self.timing.rtf,
                persistent=self.timing.rejected,
                payload_json=json.dumps(
                    dict(
                        brief_drops=self.timing.brief_drops,
                        threshold=0.90,
                        rule=TIMING_RULE,
                        clock_reset=self.timing.reset_observed,
                    ),
                    allow_nan=False,
                ),
            ),
        )

    def coverage_sample(self, sample: dict[str, Any]) -> None:
        self.coverage_samples += 1
        row = self.row(
            "slam_map",
            nanoseconds(sample["coverage_map_stamp"]),
            valid=sample["coverage_status"] == "valid",
            receipt_monotonic_ns=nanoseconds(sample["wall_monotonic"]),
            clock_ns=nanoseconds(sample["simulation"]),
        )
        row.update(
            stage=sample["stage"],
            coverage_percent=sample["coverage_percent"],
            covered_m2=sample["coverage_covered_m2"],
            target_m2=sample["coverage_target_m2"],
            status=sample["coverage_status"],
            map_source_ns=nanoseconds(sample["coverage_map_stamp"]),
            map_receipt_utc_ns=nanoseconds(sample["map_received_unix"]),
            map_receipt_monotonic_ns=nanoseconds(sample["map_received_wall"]),
            map_age_ns=nanoseconds(sample["coverage_map_age"]),
            simulation_elapsed_ns=nanoseconds(sample["simulation_elapsed"]),
        )
        self.enqueue("coverage", safe(row))
        for milestone in sample["milestones_crossed"]:
            self.event("coverage_milestone", **milestone)

    def coverage_checkpoint(self, tracker: Any) -> None:
        self.manifest["coverage"].update(
            baseline=tracker.baseline,
            milestones=tracker.milestones,
            quality_flags=sorted(tracker.flags),
            samples=self.coverage_samples,
            final=tracker.snapshot(),
            frozen=tracker.frozen,
        )

    def flush(self) -> None:
        self.enqueue("manifest", safe(deepcopy(self.manifest)))

    def save_visuals(self, map_message: Any, path_message: Any) -> None:
        if map_message is not None:
            self.enqueue("map", map_message)

    def finish(
        self,
        outcome: str,
        detail: str,
        metrics: dict[str, Any],
        map_message: Any,
        path_message: Any,
        native_code: int,
        commands: list[Any],
        run_facts: dict[str, Any] | None = None,
    ) -> None:
        self.frozen = True
        self.manifest.update(
            run_facts=safe(run_facts or {}),
            phase="FINALIZING",
            outcome=outcome,
            detail=detail,
            final_metrics=safe(metrics),
            terminal_utc_ns=self.manifest.get("terminal_trigger_utc_ns", time.time_ns()),
            native_result_code=native_code,
            actual_commands=commands,
            recording_dropped=self.dropped,
            recording_dropped_insight=self.dropped_insight,
            timing_quality=dict(
                rtf=self.timing.rtf,
                rejected=self.timing.rejected,
                brief_drops=self.timing.brief_drops,
                clock_reset=self.timing.reset_observed,
            ),
            resource_summary=dict(
                peak_explorer_rss_mib=self.resource_peak_rss,
                time_weighted_explorer_cpu_percent=self.resource_integral_cpu
                / self.resource_duration
                if self.resource_duration
                else None,
                time_weighted_explorer_rss_mib=self.resource_integral_rss / self.resource_duration
                if self.resource_duration
                else None,
                valid_wall_seconds=self.resource_duration,
                integration="left hold; active explorer; gaps over 2.5s excluded",
            ),
            eligibility=dict(
                timing=self.dropped == 0
                and outcome in ("completed", "time_limit")
                and metrics.get("motion_started", False)
                and not self.timing.rejected
                and not self.timing.reset_observed
                and not any(
                    flag.startswith("stream_gap:") or flag in ("clock_reset", "rtf_unmeasured")
                    for flag in metrics.get("quality_flags", [])
                )
                and outcome != "timing_rejection",
                coverage=self.dropped == 0
                and not metrics.get("coverage_quality_flags")
                and metrics.get("coverage_status") == "valid",
                motion=self.dropped == 0 and metrics.get("motion_started", False),
                navigation=bool(self.manifest.get("navigation"))
                and not self.manifest.get("navigation", {}).get("lost_events"),
            ),
        )
        self.save_visuals(map_message, path_message)
        self.event("terminal", outcome=outcome, detail=detail, native_result_code=native_code)
        self.flush()
        self.enqueue("finish", None)
        self.thread.join(timeout=30)
        if self.thread.is_alive() or self.error:
            raise OSError("Scientific dataset did not commit: " + str(self.error))
        self.closed = True

    def _write(self, manifest: dict[str, Any]) -> None:
        buffers: dict[str, list[dict[str, Any]]] = {kind: [] for kind in SCHEMAS}
        objects: dict[str, str] = {}
        last_flush = time.monotonic()
        map_identity = None

        def save_manifest() -> None:
            snapshot = deepcopy(manifest)
            workspace = Path(self.configuration["suite_workspace"])
            for name in ("configuration", "robot", "requested_parameters", "effective_parameters"):
                if name in snapshot:
                    snapshot[name] = {"object": put_object(workspace, snapshot[name])}
            generated = snapshot.get("robot_configuration", {})
            for name in ("robot_description", "visualization_description"):
                if name in generated:
                    generated[name] = {"object": put_object(workspace, generated[name])}
            atomic_json(self.directory / "manifest.json", snapshot)

        def checkpoint() -> None:
            nonlocal last_flush
            for kind, rows in buffers.items():
                if not rows:
                    continue
                table = pa.Table.from_pylist(rows, schema=SCHEMAS[kind])
                destination = self.directory / "segments" / kind
                destination.mkdir(parents=True, exist_ok=True)
                path = destination / (uuid.uuid4().hex + ".parquet")
                temporary = path.with_suffix(".staging")
                pq.write_table(
                    table, temporary, compression="zstd", use_dictionary=True, version="2.6"
                )
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                checked = pq.ParquetFile(temporary)
                if checked.metadata.num_rows != len(rows) or checked.schema_arrow != SCHEMAS[kind]:
                    raise OSError("Parquet checkpoint validation failed")
                temporary.replace(path)
                sync_directory(destination)
                self.segments.append(
                    dict(
                        kind=kind,
                        path=str(path.relative_to(self.directory)),
                        rows=len(rows),
                        sha256=file_hash(path),
                        bytes=path.stat().st_size,
                    )
                )
                rows.clear()
            atomic_json(
                self.directory / "segments.json",
                {"schema_version": DATASET_VERSION, "segments": self.segments},
            )
            last_flush = time.monotonic()

        try:
            while True:
                try:
                    kind, value = self.queue.get(
                        timeout=max(0.01, 5 - (time.monotonic() - last_flush))
                    )
                except queue.Empty:
                    checkpoint()
                    continue
                if kind in buffers:
                    buffers[kind].append(value)
                    if len(buffers[kind]) >= 2000:
                        checkpoint()
                elif kind == "object":
                    name, data = value
                    digest, _ = object_store(self.objects, data)
                    objects[name] = digest
                elif kind == "manifest":
                    manifest = value
                    manifest["objects"] = dict(objects)
                    manifest["recording_dropped"] = self.dropped
                    manifest["recording_dropped_insight"] = self.dropped_insight
                    save_manifest()
                elif kind == "map":
                    stamp = value.header.stamp
                    identity = (stamp.sec, stamp.nanosec, value.info.width, value.info.height)
                    if identity != map_identity:
                        array = np.asarray(value.data, dtype=np.int8).reshape(
                            value.info.height, value.info.width
                        )
                        origin = value.info.origin
                        metadata = dict(
                            source_ns=stamp.sec * 10**9 + stamp.nanosec,
                            frame=value.header.frame_id,
                            width=value.info.width,
                            height=value.info.height,
                            resolution=value.info.resolution,
                            origin=[origin.position.x, origin.position.y, origin.position.z],
                            orientation=[
                                origin.orientation.x,
                                origin.orientation.y,
                                origin.orientation.z,
                                origin.orientation.w,
                            ],
                        )
                        buffer = BytesIO()
                        np.savez_compressed(buffer, occupancy=array, metadata=json.dumps(metadata))
                        atomic_bytes(self.directory / "final_map.npz", buffer.getvalue())
                        map_identity = identity
                elif kind == "finish":
                    checkpoint()
                    manifest["phase"] = "FINISHED"
                    manifest["artifact_commit_utc_ns"] = time.time_ns()
                    manifest["objects"] = objects
                    manifest["segments_hash"] = file_hash(self.directory / "segments.json")
                    save_manifest()
                    break
                if time.monotonic() - last_flush >= 5:
                    checkpoint()
        except Exception as exc:
            self.error = exc
