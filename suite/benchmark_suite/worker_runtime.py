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

from collections import deque
from io import BytesIO
import base64
import math
import os
from pathlib import Path
import queue
import time
from typing import Any

import numpy as np
import psutil
from PIL import Image

from .contracts import ContractError
from .protocol import PendingRequest, Server
from .recording import Recorder, safe


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
        self.path: deque[tuple[int, float, float]] = deque(maxlen=10000)
        self.path_sequence = 0
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
        self.inspection_rows: deque[tuple[int, dict[str, Any]]] = deque(maxlen=1000)
        self.inspection_sequence = 0
        self.inspection_after = 0
        self.inspection_detector: Any = None
        self.inspection_incidents: deque[dict[str, Any]] = deque(maxlen=64)

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
            result["simulation_elapsed"] = result["wall_elapsed"] = None
        self.metrics = safe(result)
        self.sequence += 1

    def tick(self) -> None:
        worker = self.worker
        for _ in range(16):
            try:
                pending = self.requests.get_nowait()
            except queue.Empty:
                break
            if pending.future.done():
                continue
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
                        "engine_interruption",
                        "storage_failure",
                        "startup_failure",
                    ):
                        raise ContractError("reason", "Unsupported worker stop reason")
                    worker.terminal.put((reason, str(params.get("detail", "Suite control")), -1))
                    result = {"accepted": True}
                elif method == "worker.inspect":
                    result = self.inspection(
                        params.get("after_sequence", 0), params.get("map_revision", -1)
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
        recorder.origins = (
            worker.measurements.movement_simulation_start,
            worker.measurements.movement_wall_start,
        )
        recorder.clock = worker.clock_value
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
                world_x=x + c * p.x - s * p.y, world_y=y + s * p.x + c * p.y, world_yaw=yaw + angle
            )
            self.pose = [row["world_x"], row["world_y"], row["world_yaw"]]
            self.path_sequence += 1
            self.path.append((self.path_sequence, row["world_x"], row["world_y"]))
        recorder.observation("trajectory", row)
        self.inspection_sequence += 1
        self.inspection_rows.append(
            (self.inspection_sequence, dict(row, active_goal_sequence=self.goal_sequence))
        )

    def command(self, topic: str, message: Any) -> None:
        recorder = self.worker.recorder
        if not isinstance(recorder, Recorder) or recorder.frozen:
            return
        values = message.linear.x, message.linear.y, message.angular.z
        if self.last_commands.get(topic) == values:
            return
        self.last_commands[topic] = values
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
            if worker.phase != "RUNNING" or not isinstance(recorder, Recorder):
                return
            p, q, v = message.pose.pose.position, message.pose.pose.orientation, message.twist.twist
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            row = recorder.row(
                "gazebo_model_world_pose",
                message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
                clock_ns=message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
            )
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
            row["valid"] = all(math.isfinite(value) for value in (p.x, p.y, p.z, yaw))
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
            if message.event == "initialized":
                self.goal_sequence = message.goal_sequence
                alignment = recorder.manifest["coverage"]["alignment"]["world_from_map"]
                x, y, angle = alignment
                self.goal = [
                    x + math.cos(angle) * p.x - math.sin(angle) * p.y,
                    y + math.sin(angle) * p.x + math.cos(angle) * p.y,
                ]
            elif message.event == "terminal":
                self.goal_sequence, self.goal = None, None
            recorder.manifest["navigation"] = dict(
                event_sequence=self.navigation_event_sequence,
                lost_events=self.navigation_lost,
                uuid_association="unavailable; native feedback UUID retained separately",
            )
            recorder.observation("navigation", row)

    def feedback(self, message: Any) -> None:
        worker, recorder = self.worker, self.worker.recorder
        with worker.lock:
            if worker.phase != "RUNNING" or not isinstance(recorder, Recorder):
                return
            feedback = message.feedback
            stamp = feedback.current_pose.header.stamp
            row = recorder.row(
                "native_action_feedback",
                stamp.sec * 10**9 + stamp.nanosec,
                clock_ns=stamp.sec * 10**9 + stamp.nanosec,
            )
            row.update(
                event="feedback",
                native_uuid=bytes(message.goal_id.uuid).hex(),
                distance_remaining=feedback.distance_remaining,
                recoveries=feedback.number_of_recoveries,
                frame=feedback.current_pose.header.frame_id,
                x=feedback.current_pose.pose.position.x,
                y=feedback.current_pose.pose.position.y,
            )
            recorder.observation("navigation", row)

    def sample_resources(self, wall: float) -> None:
        recorder = self.worker.recorder
        if not isinstance(recorder, Recorder) or recorder.frozen:
            return
        totals: dict[str, list[float]] = {}
        for process in self.worker.simulation_scope.discover() + [psutil.Process()]:
            try:
                command = " ".join(process.cmdline()).lower()
                if process.pid == os.getpid():
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

    def inspection(self, after: int, map_revision: int) -> dict[str, Any]:
        worker = self.worker
        with worker.lock:
            grid = worker.cache.get("map")
            points = list(self.path)
            pose = self.pose
            navigation_plan = worker.cache.get("plan")
            motion_rows = list(self.inspection_rows)
        if self.inspection_detector is None:
            from .analysis import DetectorSettings, MotionDetector

            robot = worker.config["robot"]
            self.inspection_detector = MotionDetector(
                robot["max_linear_speed"], robot["footprint"], DetectorSettings()
            )
        for sequence, row in motion_rows:
            if sequence <= self.inspection_after:
                continue
            for incident in self.inspection_detector.feed(row):
                if incident.get("world_x") is not None:
                    self.inspection_incidents.append(
                        {
                            key: incident[key]
                            for key in ("type", "start_ns", "end_ns", "duration_ns", "bbox")
                        }
                    )
            self.inspection_after = sequence
        image = None
        revision = -1
        alignment = worker.recorder.manifest["coverage"]["alignment"] if worker.recorder else None
        if grid is not None and worker.coverage and alignment is not None:
            revision = grid.header.stamp.sec * 10**9 + grid.header.stamp.nanosec
            if revision != map_revision:
                values = np.asarray(grid.data, dtype=np.int16).reshape(
                    grid.info.height, grid.info.width
                )
                pixels = np.where(values < 0, 170, np.rint(245 * (100 - values) / 100)).astype(
                    np.uint8
                )
                buffer = BytesIO()
                Image.fromarray(pixels).save(buffer, format="PNG")
                origin = grid.info.origin
                q = origin.orientation
                angle = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                x, y, yaw = alignment["world_from_map"]
                c, s = math.cos(yaw), math.sin(yaw)
                image = dict(
                    image_png=base64.b64encode(buffer.getvalue()).decode(),
                    resolution=grid.info.resolution,
                    origin=[
                        x + c * origin.position.x - s * origin.position.y,
                        y + s * origin.position.x + c * origin.position.y,
                        yaw + angle,
                    ],
                )
        reset = bool(points and after < points[0][0] - 1)
        plan = []
        if (
            navigation_plan is not None
            and alignment is not None
            and navigation_plan.header.frame_id == "map"
        ):
            x, y, yaw = alignment["world_from_map"]
            c, s = math.cos(yaw), math.sin(yaw)
            stride = max(1, len(navigation_plan.poses) // 2000)
            plan = [
                [
                    x + c * p.pose.position.x - s * p.pose.position.y,
                    y + s * p.pose.position.x + c * p.pose.position.y,
                ]
                for p in navigation_plan.poses[::stride]
            ]
        return dict(
            sequence=self.path_sequence,
            map_revision=revision,
            map=image,
            pose=pose,
            goal=self.goal,
            plan=plan,
            incidents=list(self.inspection_incidents),
            incident_scope="Bounded live preview; saved reports analyze the complete recorded trajectory",
            reset=reset,
            path=[[x, y] for seq, x, y in points if reset or seq > after],
        )

    def close(self) -> None:
        self.server.close()
