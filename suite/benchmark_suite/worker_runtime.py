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

from collections.abc import Callable
from functools import partial
import json
import math
import os
from pathlib import Path
import queue
import time
from typing import Any
import uuid
import zlib

import psutil

from .contracts import ContractError, average_speed
from .goal_facts import PREEMPTED, Goal, GoalAccumulator
from .insight import (
    ExploredSnapshots,
    GoalHistory,
    InsightSummary,
    candidate_rows,
    chunks_layer,
    convert,
    decision_row,
    decisions_layer,
    stamp_ns,
    to_world,
)
from .monitoring import (
    MONITOR_LAYERS,
    PhysicalPath,
    ReferenceCells,
    base_pose,
    explored_pixels,
    png,
)
from .protocol import PendingRequest, Server
from .recording import Recorder, safe

STALE_POSE_WALL_SECONDS = 1.0


class WorkerRuntime:
    def __init__(self, worker: Any):
        self.worker = worker
        self.requests: queue.Queue[PendingRequest] = queue.Queue(64)
        self.server = Server(Path(worker.config["suite_socket"]), self.requests)
        self.state: dict[str, Any] = {
            "phase": "STARTING",
            "outcome": "",
            "detail": "Preparing simulation",
        }
        self.metrics: dict[str, Any] = {}
        self.sequence = 0
        self.history = uuid.uuid4().hex
        self.physical = PhysicalPath(
            worker.config["robot"]["max_linear_speed"],
            reference=ReferenceCells.of(worker.coverage_reference),
        )
        self.physical_pose: tuple[float, float, float, int, float] | None = None
        self.layer_cache: dict[str, tuple[int, dict[str, Any]]] = {}
        self.last_trajectory_ns: int | None = None
        self.last_commands: dict[str, tuple[float, float, float]] = {}
        self.world_from_odom: tuple[float, float, float] | None = None
        self.pose: list[float] | None = None
        self.planar_speed: float | None = None
        self.angular_speed: float | None = None
        self.ground_truth: Any = None
        self.ground_truth_previous: tuple[int, float, float, float] | None = None
        self.model_to_base: tuple[float, float, float] | None = None
        self.goal: list[float] | None = None
        self.goal_sequence: int | None = None
        self.navigation_event_sequence = 0
        self.navigation_lost = 0
        self.resource_previous: dict[str, tuple[float, float]] = {}
        # Explorer insight: read and written under worker.lock, except the explored snapshots, which
        # belong to the supervision thread like the coverage calculator.
        self.insight = InsightSummary()
        self.goal_history = GoalHistory()
        self.explored_snapshots = ExploredSnapshots()
        # Goal facts and the debug-stream switch (RD-42): feedback and command rows are recorded only
        # with debug streams on; the accumulator sees every feedback message either way.
        self.goals = GoalAccumulator()
        # The same truth value the recorder lists in its manifest (recording.DEBUG_STREAMS).
        self.debug_streams = bool(worker.config.get("debug_streams"))
        self.insight_exited = False

    def publish_state(self) -> None:
        worker = self.worker
        self.state = dict(
            phase=worker.phase,
            outcome=worker.outcome,
            detail=worker.detail,
            native_result_code=worker.native_code,
        )
        self.sequence += 1

    def publish_metrics(self, snapshot: dict[str, Any]) -> None:
        worker = self.worker
        result = dict(
            snapshot,
            wall_monotonic=time.monotonic(),
            clock=worker.clock_value,
            source="gazebo",
            speed=self.planar_speed,
            angular_speed=self.angular_speed,
        )
        if not result.get("motion_started"):
            result["simulation_elapsed"] = None
        result["average_speed_mps"] = average_speed(
            result.get("distance_m"), result.get("simulation_elapsed")
        )
        clock_ns = round(worker.clock_value * 1e9)
        with worker.lock:
            figures = self.insight.figures(clock_ns, self.goal)
            recent, run = self.goal_history.switch_rates(clock_ns)
        result["insight"] = dict(
            figures, goal_switches_per_minute_60s=recent, goal_switches_per_minute_run=run
        )
        self.metrics = safe(result)
        self.sequence += 1

    def tick(self) -> None:
        for _ in range(16):
            try:
                pending = self.requests.get_nowait()
            except queue.Empty:
                break
            self._answer(pending)

    def serve(self, seconds: float) -> None:
        """Answer engine requests as they arrive during the worker loop's pause of ``seconds``.

        Runs on the supervision thread like ``tick``; a request still being answered at the end
        of the pause finishes first.
        """
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                pending = self.requests.get(timeout=remaining)
            except queue.Empty:
                return
            self._answer(pending)

    def _answer(self, pending: PendingRequest) -> None:
        if pending.future.done():
            return
        request = pending.request
        try:
            method, params = request.get("method"), request.get("params", {})
            if method == "worker.status":
                result = dict(sequence=self.sequence, state=self.state, metrics=self.metrics)
            elif method == "worker.stop":
                reason = params.get("reason", "engine_interruption")
                if reason not in (
                    "timing_rejection",
                    "manual_discard",
                    "manual_stop",
                    "engine_interruption",
                    "storage_failure",
                    "startup_failure",
                ):
                    raise ContractError("reason", "Unsupported worker stop reason")
                self.worker.terminal.put((reason, str(params.get("detail", "Suite control")), -1))
                result = {"accepted": True}
            elif method == "worker.monitor":
                result = self.monitor(
                    params.get("after_sequence", 0),
                    params.get("layers", []),
                    params.get("revisions", {}),
                )
            else:
                raise ContractError("method", "Unsupported worker operation")
            pending.future.set_result(
                {"ok": True, "request_id": request.get("request_id"), "result": result}
            )
        except (ContractError, ValueError) as exc:
            pending.future.set_result(
                {"ok": False, "error": {"code": "invalid", "message": str(exc)}}
            )

    def odometry(self, message: Any, wall: float) -> None:
        worker, recorder = self.worker, self.worker.recorder
        if not isinstance(recorder, Recorder) or recorder.frozen:
            return
        stamp = message.header.stamp
        source = stamp.sec * 10**9 + stamp.nanosec
        if (
            self.last_trajectory_ns is not None
            and 0 <= source - self.last_trajectory_ns < 50_000_000
        ):
            return
        self.last_trajectory_ns = source
        p, q, v = message.pose.pose.position, message.pose.pose.orientation, message.twist.twist
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        valid = all(
            math.isfinite(value) for value in (p.x, p.y, yaw, v.linear.x, v.linear.y, v.angular.z)
        )
        self.planar_speed = math.hypot(v.linear.x, v.linear.y) if valid else None
        self.angular_speed = v.angular.z if valid else None
        recorder.origin = worker.measurements.movement_simulation_start
        row = recorder.row(
            "raw_odometry",
            source,
            valid=valid,
            receipt_monotonic_ns=round(wall * 1e9),
            clock_ns=source,
        )
        command = self.last_commands.get("/cmd_vel", (None, None, None))
        row.update(
            x=p.x,
            y=p.y,
            yaw=yaw,
            vx=v.linear.x,
            vy=v.linear.y,
            wz=v.angular.z,
            speed=math.hypot(v.linear.x, v.linear.y),
            distance_m=worker.measurements.distance,
            frame=message.header.frame_id,
            command_vx=command[0],
            command_vy=command[1],
            command_wz=command[2],
        )
        if self.world_from_odom and valid:
            x, y, angle = self.world_from_odom
            c, s = math.cos(angle), math.sin(angle)
            row.update(
                odom_aligned_x_m=x + c * p.x - s * p.y,
                odom_aligned_y_m=y + s * p.x + c * p.y,
                odom_aligned_yaw_rad=yaw + angle,
            )
            self.pose = [
                row["odom_aligned_x_m"],
                row["odom_aligned_y_m"],
                row["odom_aligned_yaw_rad"],
            ]
        recorder.observation("trajectory", row)

    def command(self, topic: str, message: Any) -> None:
        recorder = self.worker.recorder
        if not isinstance(recorder, Recorder) or recorder.frozen:
            return
        values = message.linear.x, message.linear.y, message.angular.z
        if self.last_commands.get(topic) == values:
            return
        self.last_commands[topic] = values
        if not self.debug_streams:
            return
        recorder.observation(
            "commands",
            dict(recorder.row("command"), vx=values[0], vy=values[1], wz=values[2], topic=topic),
        )

    def align_odometry(self, alignment: tuple[float, float, float]) -> None:
        from rclpy.time import Time

        transform = self.worker.buffer.lookup_transform("map", "odom", Time()).transform
        q = transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x, y, angle = alignment
        c, s = math.cos(angle), math.sin(angle)
        self.world_from_odom = (
            x + c * transform.translation.x - s * transform.translation.y,
            y + s * transform.translation.x + c * transform.translation.y,
            angle + yaw,
        )
        if self.ground_truth is not None:
            transform = self.worker.buffer.lookup_transform(
                "map", "base_footprint", Time()
            ).transform
            q = transform.rotation
            base_yaw = angle + math.atan2(
                2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
            )
            base_x = x + c * transform.translation.x - s * transform.translation.y
            base_y = y + s * transform.translation.x + c * transform.translation.y
            p, q = self.ground_truth.pose.pose.position, self.ground_truth.pose.pose.orientation
            model_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            c, s = math.cos(model_yaw), math.sin(model_yaw)
            self.model_to_base = (
                c * (base_x - p.x) + s * (base_y - p.y),
                -s * (base_x - p.x) + c * (base_y - p.y),
                base_yaw - model_yaw,
            )
            self.worker.recorder.manifest["ground_truth_alignment"] = dict(
                model_to_base=self.model_to_base,
                source="stationary preparation TF and Gazebo model pose",
                simulation_ns=round(self.worker.clock_value * 1e9),
            )

    def receive_ground_truth(self, message: Any) -> None:
        worker, recorder = self.worker, self.worker.recorder
        with worker.lock:
            if worker.phase in ("STOPPING", "FINISHED"):
                return
            self.ground_truth = message
            p, q, v = message.pose.pose.position, message.pose.pose.orientation, message.twist.twist
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            valid = all(math.isfinite(value) for value in (p.x, p.y, p.z, yaw))
            base = (
                base_pose((p.x, p.y, yaw), self.model_to_base)
                if valid and self.model_to_base
                else None
            )
            self.physical_pose = (*base, stamp, time.monotonic()) if base else None
            if worker.phase != "RUNNING" or not isinstance(recorder, Recorder):
                return
            row = recorder.row("gazebo_model_world_pose", stamp, clock_ns=stamp)
            row.update(
                x=p.x,
                y=p.y,
                z=p.z,
                yaw=yaw,
                vx=v.linear.x,
                vy=v.linear.y,
                wz=v.angular.z,
                speed=math.hypot(v.linear.x, v.linear.y),
                frame=message.header.frame_id,
            )
            row["valid"] = valid
            if not all(math.isfinite(value) for value in (v.linear.x, v.linear.y, v.angular.z)):
                row["quality_flags"] = ["diagnostic_velocity_unavailable"]
            if self.model_to_base:
                row.update(
                    zip(
                        ("model_to_base_x", "model_to_base_y", "model_to_base_yaw"),
                        self.model_to_base,
                    )
                )
                dx, dy, _ = self.model_to_base
                if self.pose:
                    row["odometry_drift_m"] = math.hypot(
                        p.x + math.cos(yaw) * dx - math.sin(yaw) * dy - self.pose[0],
                        p.y + math.sin(yaw) * dx + math.cos(yaw) * dy - self.pose[1],
                    )
                if not recorder.frozen:
                    # The live path takes the samples the ground_truth stream records (a full
                    # recording queue can still drop a row; such drops count in recording_dropped).
                    self.physical.add(stamp, *(base[:2] if base else (math.nan, math.nan)))
            recorder.observation("ground_truth", row)

    def receive_ground_truth_pose(self, message: Any) -> None:
        from nav_msgs.msg import Odometry

        for transform in message.transforms:
            if transform.child_frame_id.split("::")[-1] != self.worker.config["robot"]["model"]:
                continue
            stamp = transform.header.stamp.sec * 10**9 + transform.header.stamp.nanosec
            p, q = transform.transform.translation, transform.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            observation = Odometry()
            observation.header = transform.header
            observation.child_frame_id = "benchmark/model"
            observation.pose.pose.position.x = p.x
            observation.pose.pose.position.y = p.y
            observation.pose.pose.position.z = p.z
            observation.pose.pose.orientation = q
            observation.twist.twist.linear.x = observation.twist.twist.linear.y = math.nan
            observation.twist.twist.angular.z = math.nan
            if (
                self.ground_truth_previous
                and 0 < stamp - self.ground_truth_previous[0] <= 250_000_000
            ):
                previous, x, y, angle = self.ground_truth_previous
                dt = (stamp - previous) / 1e9
                dx, dy = (p.x - x) / dt, (p.y - y) / dt
                observation.twist.twist.linear.x = math.cos(yaw) * dx + math.sin(yaw) * dy
                observation.twist.twist.linear.y = -math.sin(yaw) * dx + math.cos(yaw) * dy
                observation.twist.twist.angular.z = (
                    math.atan2(math.sin(yaw - angle), math.cos(yaw - angle)) / dt
                )
            self.ground_truth_previous = stamp, p.x, p.y, yaw
            self.receive_ground_truth(observation)

    def navigation(self, message: Any) -> None:
        worker, recorder = self.worker, self.worker.recorder
        with worker.lock:
            if worker.phase != "RUNNING" or not isinstance(recorder, Recorder):
                return
            self.navigation_lost = max(
                self.navigation_lost,
                message.lost_events,
                self.navigation_lost
                + max(0, message.event_sequence - self.navigation_event_sequence - 1),
            )
            self.navigation_event_sequence = message.event_sequence
            p, q = message.goal.pose.position, message.goal.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            row = recorder.row(
                "native_navigator_callback",
                message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
                clock_ns=message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
            )
            row.update(
                event=message.event,
                goal_sequence=message.goal_sequence,
                event_sequence=message.event_sequence,
                frame=message.goal.header.frame_id,
                native_code=message.native_result_code if message.event == "terminal" else None,
                status=message.terminal_status or None,
                lost_events=self.navigation_lost,
            )
            if message.goal_valid:
                row.update(x=p.x, y=p.y, yaw=yaw)
            stamp = stamp_ns(message.header.stamp)
            position = None
            if message.event == "initialized":
                self.goal_sequence = message.goal_sequence
                alignment = recorder.manifest["coverage"]["alignment"]["world_from_map"]
                position = to_world(alignment, p.x, p.y)
            elif message.event == "terminal":
                self.goal_sequence = None
            if message.event == "initialized":
                self.goals.started(message.goal_sequence, stamp)
            elif message.event in ("terminal", PREEMPTED):
                status = message.terminal_status if message.event == "terminal" else PREEMPTED
                ended = self.goals.ended(message.goal_sequence, status)
                if ended is not None:
                    self._goal_ended(recorder, ended, stamp)
            self.goal_history.apply(
                message.event, message.goal_sequence, position, message.terminal_status, stamp
            )
            self.goal = list(self.goal_history.current) if self.goal_history.current else None
            recorder.manifest["navigation"] = dict(
                event_sequence=self.navigation_event_sequence,
                lost_events=self.navigation_lost,
                uuid_association="unavailable; native feedback UUID retained separately",
            )
            recorder.observation("navigation", row)

    def receive_insight(self, message: Any) -> None:
        """One frontier insight tick (executor thread): keep it for the Monitor and record it."""
        worker, recorder = self.worker, self.worker.recorder
        with worker.lock:
            if worker.phase != "RUNNING" or not isinstance(recorder, Recorder):
                return
            alignment = (recorder.manifest["coverage"].get("alignment") or {}).get("world_from_map")
            # The alignment places the SLAM map frame; another analysis frame stays unplaced.
            in_map = message.header.frame_id.lstrip("/") == "map"
            tick = convert(message, tuple(alignment) if alignment and in_map else None)
            self.insight.add(tick)
            if recorder.frozen:
                return
            flags = (
                ["frame_not_map"]
                if not in_map
                else []
                if alignment
                else ["world_frame_unavailable"]
            )
            row = recorder.row("frontier_insight", tick.simulation_ns, clock_ns=tick.simulation_ns)
            recorder.observation("decisions", dict(row, **decision_row(tick), quality_flags=flags))
            for fields in candidate_rows(tick):
                recorder.observation(
                    "frontier_candidates", dict(row, **fields, quality_flags=flags)
                )

    def explored_snapshot(
        self, coverage: Any, grid: Any, simulation: float, *, final: bool = False
    ) -> None:
        """Record the explored cells every 5 simulation s and once at the terminal trigger.

        Runs on the supervision thread, like the coverage calculator it shares.
        """
        worker, recorder = self.worker, self.worker.recorder
        clock_ns = round(simulation * 1e9)
        snapshots = self.explored_snapshots
        if (
            worker.phase != ("STOPPING" if final else "RUNNING")
            or not isinstance(recorder, Recorder)
            or recorder.frozen
            or grid is None
            or not (snapshots.last_ns != clock_ns if final else snapshots.due(clock_ns))
        ):
            return
        try:
            free = coverage.calculator.free_cells(grid)
        except ValueError:
            return
        fields = snapshots.take(free, clock_ns)
        source = stamp_ns(grid.header.stamp)
        queued = recorder.observation(
            "explored_cells",
            dict(
                recorder.row("slam_map", source, clock_ns=clock_ns),
                **fields,
                map_source_ns=source,
            ),
        )
        if not queued:
            # A lost row would break the chain of changes: the next snapshot is a keyframe.
            snapshots.restart()

    def insight_manifest(self) -> dict[str, Any]:
        """Insight counters for the attempt manifest at the end of the run."""
        with self.worker.lock:
            latest = self.insight.latest
            return dict(
                ticks_received=self.insight.ticks,
                ticks_lost=self.insight.lost,
                ticks_skipped=latest.skipped_ticks if latest else None,
                last_tick=latest.tick if latest else None,
                goal_updates=self.goal_history.updates,
            )

    def feedback(self, message: Any) -> None:
        worker, recorder = self.worker, self.worker.recorder
        with worker.lock:
            if worker.phase != "RUNNING" or not isinstance(recorder, Recorder):
                return
            feedback = message.feedback
            stamp = feedback.current_pose.header.stamp
            native_uuid = bytes(message.goal_id.uuid).hex()
            goal = self.goals.feedback(
                native_uuid, feedback.distance_remaining, feedback.number_of_recoveries
            )
            if not self.debug_streams:
                return
            row = recorder.row(
                "native_action_feedback",
                stamp.sec * 10**9 + stamp.nanosec,
                clock_ns=stamp.sec * 10**9 + stamp.nanosec,
            )
            row.update(
                event="feedback",
                native_uuid=native_uuid,
                distance_remaining=feedback.distance_remaining,
                recoveries=feedback.number_of_recoveries,
                frame=feedback.current_pose.header.frame_id,
                x=feedback.current_pose.pose.position.x,
                y=feedback.current_pose.pose.position.y,
            )
            # Feedback of no known goal is always kept; a goal's rows are thinned (goal_facts.py).
            if goal is None or goal.keep_feedback(row["clock_ns"], row):
                recorder.observation("navigation", row)

    @staticmethod
    def _goal_ended(recorder: Recorder, goal: Goal, end_clock_ns: int | None) -> None:
        """Record a goal's facts and its held last feedback row; the caller holds worker.lock."""
        if goal.held_row is not None:
            recorder.observation("navigation", goal.held_row)
            goal.held_row = None
        recorder.observation(
            "goal_facts",
            dict(
                recorder.row("goal_accumulator", clock_ns=end_clock_ns),
                **goal.fields(end_clock_ns),
            ),
        )

    def close_goals(self, recorder: Recorder) -> None:
        """Record the goals still open at the stop; called once before ``recorder`` finishes.

        The worker passes its recorder: it may already have dropped its own reference after a storage
        error while the recorder still finishes.
        """
        with self.worker.lock:
            for goal in self.goals.close():
                self._goal_ended(recorder, goal, None)

    def check_insight(self) -> None:
        """Note when the insight node has ended while the run is active (RD-17, ``run_facts``)."""
        roots = self.worker.insight_scope.roots
        if roots and all(root.poll() is not None for root in roots):
            self.insight_exited = True

    def run_facts(self, recorder: Recorder) -> dict[str, Any]:
        """Facts of the whole run for ``recorder``'s manifest (``run_facts``, dataset schema 10)."""
        started = bool(recorder.manifest.get("insight", {}).get("started"))
        return dict(
            self.goals.facts(),
            lost_navigation_events=self.navigation_lost,
            insight_exited_before_stop=self.insight_exited if started else None,
        )

    def sample_resources(self, wall: float) -> None:
        recorder = self.worker.recorder
        if not isinstance(recorder, Recorder) or recorder.frozen:
            return
        totals: dict[str, list[float]] = {}
        # The insight scope is `ros2 run` plus the node it spawns; a full process scan (about 14 ms) runs
        # only while `ros2 run` lives and the node is not yet known.
        insight_scope = self.worker.insight_scope
        self.check_insight()
        insight = set(insight_scope.live())
        if len(insight) < 2 and any(root.poll() is None for root in insight_scope.roots):
            insight = set(insight_scope.discover())
        for process in self.worker.simulation_scope.discover() + list(insight) + [psutil.Process()]:
            try:
                command = " ".join(process.cmdline()).lower()
                if process in insight:
                    scope = "insight"
                elif process.pid == os.getpid():
                    scope = "instrumentation_and_recording"
                elif (
                    "gz sim" in command
                    or "gz-sim" in command
                    or process.name().lower().startswith("gz sim")
                ):
                    scope = "gazebo_including_embedded_drive"
                elif "/lib/nav2_" in command or "opennav_" in command:
                    scope = "navigation"
                elif "slam_toolbox" in command or "robot_localization" in command:
                    scope = "mapping_and_estimation"
                elif "controller_manager" in command or "rosbot_controller" in command:
                    scope = "locomotion"
                else:
                    scope = "robot_and_transport"
                times = process.cpu_times()
                cpu, rss = times.user + times.system, process.memory_info().rss / 1024**2
                values = totals.setdefault(scope, [0.0, 0.0])
                values[0] += cpu
                values[1] += rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        for scope, (cpu, rss) in totals.items():
            previous = self.resource_previous.get(scope)
            percentage = (
                100 * (cpu - previous[1]) / (wall - previous[0])
                if previous and wall > previous[0] and cpu >= previous[1]
                else None
            )
            self.resource_previous[scope] = wall, cpu
            recorder.observation(
                "resources",
                dict(
                    recorder.row("owned_process_accounting"),
                    cpu_percent=percentage,
                    rss_mib=rss,
                    cpu_seconds=cpu,
                    scope=scope,
                ),
            )

    def monitor(self, after: Any, layers: Any, revisions: Any) -> dict[str, Any]:
        if type(after) is not int or after < 0:
            raise ContractError("after_sequence", "Use a non-negative integer")
        if not isinstance(layers, list) or not all(layer in MONITOR_LAYERS for layer in layers):
            raise ContractError("layers", "Unknown monitor layer")
        if not isinstance(revisions, dict) or not all(
            type(value) is int for value in revisions.values()
        ):
            raise ContractError("revisions", "Use integer layer revisions")
        worker = self.worker
        with worker.lock:
            page = self.physical.page(after)
            pose, offset, received = self.physical_pose, self.model_to_base, self.ground_truth
            grid, plan, goal = worker.cache.get("map"), worker.cache.get("plan"), self.goal
            goal_sequence = self.goal_sequence
            tick = self.insight.latest
            # The goals revision is the reader's update cursor (-1 or absent: from the start).
            goals = (
                self.goal_history.page(max(0, revisions.get("goals", 0)))
                if "goals" in layers
                else None
            )
        now = time.monotonic()
        if not worker.config.get("suite_diagnostics", True):
            status, reason = (
                "unavailable",
                "Ground truth is off: the engine runs without diagnostics",
            )
        elif received is None:
            status, reason = "unavailable", "No Gazebo ground-truth pose received yet"
        elif offset is None:
            status, reason = (
                "unavailable",
                (
                    "Waiting for the stationary alignment at readiness"
                    if self.world_from_odom is None
                    else "No ground-truth pose at readiness: the base offset is unknown"
                ),
            )
        elif pose is None:
            status, reason = "unavailable", "The latest ground-truth pose is invalid"
        elif now - pose[4] > STALE_POSE_WALL_SECONDS:
            status, reason = "stale", f"No ground-truth pose for {now - pose[4]:.1f} wall s"
        else:
            status, reason = "available", ""
        alignment = worker.recorder.manifest["coverage"]["alignment"] if worker.recorder else None
        world_from_map = alignment["world_from_map"] if alignment else None
        map_stamp = (
            grid.header.stamp.sec * 10**9 + grid.header.stamp.nanosec if grid is not None else -1
        )
        coverage = worker.coverage
        result: dict[str, Any] = {}
        for name in dict.fromkeys(layers):
            requested = revisions.get(name)
            if name == "explored":
                result[name] = self.layer(
                    name,
                    map_stamp,
                    requested,
                    None
                    if coverage is None or grid is None
                    else partial(self.explored, coverage, grid),
                )
            elif name in ("decisions", "chunks"):
                encode = decisions_layer if name == "decisions" else chunks_layer
                result[name] = self.layer(
                    name,
                    tick.tick if tick else -1,
                    requested,
                    None if tick is None else partial(encode, tick),
                )
                insight = (
                    worker.recorder.manifest.get("insight") if worker.recorder else None
                ) or {}
                if tick is None and insight.get("started") is False:
                    # The node never started: say why instead of waiting for a tick.
                    result[name]["reason"] = str(
                        insight.get("reason") or "Insight node not started"
                    )
            elif name == "goals":
                # Paged by update cursor, not cached; trim_reply adapts it to other readers.
                assert goals is not None
                result[name] = dict(
                    revision=goals["revision"],
                    data=None if goals["revision"] == requested else goals,
                )
            else:
                plan_stamp = (
                    plan.header.stamp.sec * 10**9 + plan.header.stamp.nanosec
                    if plan is not None
                    else -1
                )
                identity = [goal_sequence, goal, plan_stamp, len(plan.poses) if plan else 0]
                result[name] = self.layer(
                    name,
                    zlib.crc32(json.dumps(identity).encode()),
                    requested,
                    None
                    if world_from_map is None
                    else partial(self.navigation_layer, goal, plan, world_from_map),
                )
        reference = worker.coverage_reference
        return dict(
            page,
            history=self.history,
            status=status,
            reason=reason,
            pose=None
            if pose is None
            else dict(
                x=pose[0],
                y=pose[1],
                yaw=pose[2],
                simulation_ns=pose[3],
                age_wall_seconds=now - pose[4],
            ),
            reference=dict(id=reference.identity, metadata_sha256=reference.metadata_hash),
            layers=result,
        )

    def layer(
        self,
        name: str,
        revision: int,
        requested: int | None,
        encode: Callable[[], dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Encode a layer once per revision and send it only to requesters without it."""
        if encode is None:
            self.layer_cache.pop(name, None)
            return dict(revision=-1, data=None)
        if revision == requested:
            return dict(revision=revision, data=None)
        cached = self.layer_cache.get(name)
        if cached is None or cached[0] != revision:
            try:
                cached = revision, encode()
            except ValueError as exc:
                return dict(revision=-1, data=None, reason=str(exc))
            self.layer_cache[name] = cached
        return dict(revision=revision, data=cached[1])

    def explored(self, coverage: Any, grid: Any) -> dict[str, Any]:
        reference = self.worker.coverage_reference
        return dict(
            image_png=png(explored_pixels(reference.target, coverage.calculator.free_cells(grid))),
            origin=list(reference.origin),
            resolution=reference.resolution,
        )

    @staticmethod
    def navigation_layer(
        goal: list[float] | None, plan: Any, world_from_map: list[float]
    ) -> dict[str, Any]:
        points: list[list[float]] = []
        if plan is not None and plan.header.frame_id == "map":
            x, y, yaw = world_from_map
            c, s = math.cos(yaw), math.sin(yaw)
            stride = max(1, len(plan.poses) // 2000)
            points = [
                [
                    x + c * p.pose.position.x - s * p.pose.position.y,
                    y + s * p.pose.position.x + c * p.pose.position.y,
                ]
                for p in plan.poses[::stride]
            ]
        return dict(goal=goal, plan=points)

    def close(self) -> None:
        self.server.close()
