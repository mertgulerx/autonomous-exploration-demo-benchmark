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

"""Replay windows of a finished attempt: what the live Monitor showed, rebuilt from the recorded streams.

``window`` reads one simulation-time window (at most 60 s) of the explorer insight streams, the coverage
samples, the run measurements (motion and explorer process accounting, one sample per simulation second) and
the goal history, and returns them in the live layers' shapes (``insight.decisions_layer``,
``insight.chunks_layer``), so a replayed tick shows the values the live view showed. The ground-truth path
comes from the Results run map (``results.map``), whose vertices carry their simulation stamps.
``ReplayWindows`` builds windows off the engine loop: one thread, the latest requests win, results cached.
"""

import base64
from collections import OrderedDict, deque
import json
import math
from pathlib import Path
import threading
from typing import Any
import zlib

from .contracts import canonical
from .datasets import files
from .protocol import MAX_MESSAGE
from .insight import (
    CANDIDATE_VALUES,
    GOAL_CAPACITY,
    GOAL_EVENTS,
    GoalHistory,
    chunks_layer,
    decisions_layer,
    tick_from_rows,
    to_world,
)
from .storage import inside

WINDOW_LIMIT_NS = 60_000_000_000
# Ready windows must fit one protocol message with room for the envelope.
REPLY_LIMIT = MAX_MESSAGE - 64 * 1024
CACHED_WINDOWS = 16
PENDING_WINDOWS = 2
DECISION_COLUMNS = (
    "clock_ns",
    "quality_flags",
    "tick",
    "mode",
    "robot_x",
    "robot_y",
    "robot_yaw",
    "raw_frontiers",
    "route",
    "route_cost",
    "analysis_wall_ms",
    "skipped_ticks",
    "map_source_ns",
    "chunk_size_cells",
    "chunk_resolution",
    "chunk_origin_x",
    "chunk_origin_y",
    "chunk_map_width",
    "chunk_map_height",
    "chunk_columns",
    "chunk_rows",
    "chunk_states",
    "chunk_hits",
    "chunk_rebuilt",
    "chunk_resets",
)


def alignment_of(manifest: dict[str, Any]) -> tuple[float, float, float] | None:
    values = ((manifest.get("coverage") or {}).get("alignment") or {}).get("world_from_map")
    if not isinstance(values, list) or len(values) != 3:
        return None
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return None
    return float(values[0]), float(values[1]), float(values[2])


def seconds_ns(value: Any) -> int | None:
    return round(value * 1e9) if isinstance(value, (int, float)) and math.isfinite(value) else None


def per_second(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The first row of each simulation second (rows ordered by time), so a replay time sees a sample at most
    1 s old: enough for the measurements panel."""
    kept: dict[int, dict[str, Any]] = {}
    for row in rows:
        kept.setdefault(row["clock_ns"] // 1_000_000_000, row)
    return list(kept.values())


def window(workspace: Path, attempt_id: str, start_ns: int, end_ns: int) -> dict[str, Any]:
    """One replay window of a finished attempt (read-only; ``end_ns - start_ns`` at most 60 s)."""
    import duckdb

    if not 0 <= start_ns < end_ns <= start_ns + WINDOW_LIMIT_NS:
        raise ValueError("Replay window must be positive and at most 60 simulation s")
    directory = inside(workspace / "attempts", attempt_id)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    alignment = alignment_of(manifest)
    insight = manifest.get("insight") or {}

    with duckdb.connect(config={"threads": 1, "memory_limit": "256MB"}) as connection:

        def rows(
            kind: str, columns: tuple[str, ...], where: str, *args: Any
        ) -> list[dict[str, Any]]:
            paths = [str(path) for path in files(directory, kind)]
            if not paths:
                return []
            selected = ",".join('"' + name + '"' for name in columns)
            table = connection.execute(
                f"SELECT {selected} FROM read_parquet(?) WHERE attempt_id=? AND {where} "
                "ORDER BY clock_ns, receipt_monotonic_ns",
                (paths, attempt_id, *args),
            ).fetch_arrow_table()
            result: list[dict[str, Any]] = table.to_pylist()
            return result

        # Windows are half-open, [start, end), so consecutive windows never repeat a row.
        between = "clock_ns >= ? AND clock_ns < ?"
        decisions = rows("decisions", DECISION_COLUMNS, between, start_ns, end_ns)
        candidates = rows(
            "frontier_candidates",
            ("clock_ns", "tick", "candidate", "x", "y", *CANDIDATE_VALUES),
            between,
            start_ns,
            end_ns,
        )
        # Cache-hit run means and goal figures need everything recorded before the window.
        summaries = rows(
            "decisions",
            ("clock_ns", "tick", "skipped_ticks", "chunk_hits", "chunk_rebuilt", "chunk_resets"),
            "clock_ns < ?",
            end_ns,
        )
        events = rows(
            "navigation",
            ("clock_ns", "event_sequence", "event", "goal_sequence", "x", "y", "status"),
            "clock_ns < ? AND event IN (" + ",".join("?" * len(GOAL_EVENTS)) + ")",
            end_ns,
            *GOAL_EVENTS,
        )
        coverage = rows(
            "coverage",
            (
                "clock_ns",
                "valid",
                "status",
                "coverage_percent",
                "simulation_elapsed_ns",
                "map_age_ns",
            ),
            between,
            start_ns,
            end_ns,
        )
        # The run measurements the live Monitor showed, as recorded: raw-odometry motion (``trajectory``) and
        # the explorer's process accounting with the RTF and quality flags (``resources``, scope explorer).
        motion = per_second(
            rows(
                "trajectory",
                ("clock_ns", "distance_m", "speed", "wz"),
                between + " AND valid",
                start_ns,
                end_ns,
            )
        )
        resources = per_second(
            rows(
                "resources",
                (
                    "clock_ns",
                    "cpu_percent",
                    "rss_mib",
                    "cpu_average_percent",
                    "rss_average_mib",
                    "cpu_seconds",
                    "rtf",
                    "quality_flags",
                ),
                between + " AND scope='explorer'",
                start_ns,
                end_ns,
            )
        )
        keyframes = rows(
            "explored_cells", ("clock_ns",), "kind='keyframe' AND clock_ns <= ?", start_ns
        )
        explored = rows(
            "explored_cells",
            ("clock_ns", "kind", "reference_cells", "free_cells", "mask"),
            between,
            keyframes[-1]["clock_ns"] if keyframes else 0,
            end_ns,
        )

    events.sort(key=lambda row: (row["clock_ns"], row["event_sequence"] or 0))
    history = GoalHistory()
    pending = deque(events)

    def apply_until(clock_ns: int) -> None:
        while pending and pending[0]["clock_ns"] <= clock_ns:
            event = pending.popleft()
            x, y = event["x"], event["y"]
            position = (
                to_world(alignment, x, y)
                if alignment is not None and x is not None and y is not None
                else None
            )
            history.apply(
                event["event"], event["goal_sequence"], position, event["status"], event["clock_ns"]
            )

    # Rows of one tick share its stamp; the stamp keeps ticks apart if the node's counter restarted.
    by_tick: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in candidates:
        by_tick.setdefault((row["clock_ns"], row["tick"]), []).append(row)
    windowed = {(row["clock_ns"], row["tick"]): row for row in decisions}
    ticks: list[dict[str, Any]] = []
    hit_sum, hit_ticks = 0.0, 0
    for summary in summaries:
        apply_until(summary["clock_ns"])
        total = summary["chunk_hits"] + summary["chunk_rebuilt"] + summary["chunk_resets"]
        ratio = summary["chunk_hits"] / total if total else None
        if ratio is not None:
            hit_sum, hit_ticks = hit_sum + ratio, hit_ticks + 1
        key = (summary["clock_ns"], summary["tick"])
        decision = windowed.get(key)
        if decision is None:
            continue
        # Live, a tick outside the map frame was never placed in the world (frame_not_map).
        placement = None if "frame_not_map" in decision["quality_flags"] else alignment
        tick = tick_from_rows(decision, by_tick.get(key, []), placement)
        target = tick.candidate(tick.target)
        goal = history.current
        recent, run = history.switch_rates(tick.simulation_ns)
        figures = dict(
            skipped_ticks=tick.skipped_ticks,
            cache_hit_last=ratio,
            cache_hit_run=hit_sum / hit_ticks if hit_ticks else None,
            target=tick.target,
            target_goal_distance_m=math.hypot(target.world[0] - goal[0], target.world[1] - goal[1])
            if target is not None and target.world is not None and goal is not None
            else None,
            goal_switches_per_minute_60s=recent,
            goal_switches_per_minute_run=run,
        )
        placed = tick.robot_world is not None
        ticks.append(
            dict(
                simulation_ns=tick.simulation_ns,
                decisions=decisions_layer(tick) if placed else None,
                chunks=chunks_layer(tick) if placed else None,
                figures=figures,
            )
        )
    apply_until(end_ns - 1)

    reference = (manifest.get("coverage") or {}).get("reference") or {}
    available = bool(insight.get("started")) and bool(files(directory, "decisions"))
    return dict(
        attempt_id=attempt_id,
        state="ready",
        start_ns=start_ns,
        end_ns=end_ns,
        extent=dict(
            simulation_start_ns=seconds_ns(manifest.get("simulation_start")),
            first_motion_ns=seconds_ns((manifest.get("first_motion") or {}).get("simulation")),
            end_ns=manifest.get("terminal_trigger_simulation_ns"),
        ),
        analysis=dict(
            available=available,
            reason=None
            if available
            else insight.get("reason")
            or (
                "No explorer analysis for this run"
                if insight
                else "No explorer analysis: the run predates the insight node"
            ),
            world_placed=alignment is not None,
            rows_not_recorded=manifest.get("recording_dropped_insight"),
        ),
        reference=dict(id=reference.get("id"), metadata_sha256=reference.get("metadata_sha256")),
        ticks=ticks,
        goals=list(history.goals.values())[-GOAL_CAPACITY:],
        goals_dropped=history.dropped,
        explored=dict(
            encoding="packbits-zlib-base64",
            order="reference target cells, row-major",
            snapshots=[
                dict(
                    clock_ns=row["clock_ns"],
                    kind=row["kind"],
                    reference_cells=row["reference_cells"],
                    free_cells=row["free_cells"],
                    mask=base64.b64encode(zlib.compress(row["mask"], 6)).decode(),
                )
                for row in explored
            ],
        ),
        coverage=coverage,
        motion=motion,
        resources=resources,
        path="results.map vertices with their simulation stamps",
    )


def trim(cache: OrderedDict[Any, Any]) -> None:
    while len(cache) > CACHED_WINDOWS:
        cache.popitem(last=False)


class ReplayWindows:
    """Replay windows built on one background thread, so recorded reads never block the engine loop.

    ``get`` answers from the cache or queues the window (the two latest requests are kept, so a reader can
    prefetch the next window) and answers ``loading``. A failed read is reported once and not cached.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.lock = threading.Condition()
        self.ready: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()
        self.failures: OrderedDict[tuple[str, int, int], str] = OrderedDict()
        self.wanted: deque[tuple[str, int, int]] = deque(maxlen=PENDING_WINDOWS)
        self.building: tuple[str, int, int] | None = None
        self.closing = False
        self.thread: threading.Thread | None = None

    def get(self, attempt_id: str, start_ns: int, end_ns: int) -> dict[str, Any]:
        key = (attempt_id, start_ns, end_ns)
        with self.lock:
            if key in self.ready:
                self.ready.move_to_end(key)
                return self.ready[key]
            if key in self.failures:
                return dict(attempt_id=attempt_id, state="failed", reason=self.failures.pop(key))
            if key != self.building and key not in self.wanted:
                self.wanted.append(key)
                if self.thread is None or not self.thread.is_alive():
                    self.thread = threading.Thread(
                        target=self._run, name="replay-windows", daemon=True
                    )
                    self.thread.start()
                self.lock.notify()
        return dict(attempt_id=attempt_id, state="loading", start_ns=start_ns, end_ns=end_ns)

    def _run(self) -> None:
        while True:
            with self.lock:
                while not self.wanted and not self.closing:
                    self.lock.wait()
                if self.closing:
                    return
                key = self.building = self.wanted.popleft()
            result: dict[str, Any] | None = None
            failure = "Replay window failed"
            try:
                result = window(self.workspace, *key)
                size = len(canonical(result))
                if size > REPLY_LIMIT:
                    result = None
                    failure = f"Replay window too large ({size:,} bytes); request a shorter window"
            except Exception as exc:
                # The thread boundary: report any failure once and keep serving (a read can fail
                # while compaction moves segments); failures are never cached as windows.
                failure = f"Replay window unreadable: {exc}"
            finally:
                with self.lock:
                    self.building = None
                    if result is not None:
                        self.ready[key] = result
                        trim(self.ready)
                    else:
                        self.failures[key] = failure
                        trim(self.failures)

    def close(self) -> None:
        with self.lock:
            self.closing = True
            self.lock.notify()
        if self.thread is not None:
            self.thread.join(timeout=5)
