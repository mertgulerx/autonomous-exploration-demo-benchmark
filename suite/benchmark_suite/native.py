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

import json
import importlib
from importlib.metadata import version
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

import psutil
import yaml

from exploration_benchmark_runner.configuration import scenario
from exploration_benchmark_runner.coverage import Reference
from exploration_benchmark_runner.launch_spec import WorkerLaunchSpec
from exploration_benchmark_runner.processes import DomainLease, ProcessScope

from .contracts import ContractError, resolve_world_time_limits
from .datasets import finalize_attempt, finalize_partial
from .protocol import Client
from .storage import atomic_json, file_hash, put_object, load_object


class NativeWorker:
    def __init__(self, attempt: dict[str, Any], configuration: dict[str, Any]):
        self.attempt, self.configuration = attempt, configuration
        self.directory = Path(attempt["directory"])
        self.lease = DomainLease([attempt["domain"]], 0, 1)
        self.scope = ProcessScope(attempt["token"])
        self.closed = False
        self.events: queue.Queue[dict[str, Any]] = queue.Queue(128)
        self.lock = threading.Lock()
        self.inspection_request: tuple[int, int] | None = None
        self.latest_inspection: dict[str, Any] = {"sequence": 0, "path": [], "map": None}
        self.stop_requested: tuple[str, str] | None = None
        self.stop_wall: float | None = None
        self.client = Client(Path(configuration["suite_socket"]), timeout=2.0)
        self.sequence = -1
        self.last_state = ""
        self.last_metrics_wall = None
        self.latest_metrics: dict[str, Any] = {}
        self.terminal = False
        self.trigger: dict[str, Any] | None = None
        self.started_wall = time.monotonic()
        try:
            launch_file = self.directory / "launch.json"
            atomic_json(launch_file, configuration)
            from exploration_benchmark_robots.runtime import environment as robot_environment

            environment = robot_environment(
                configuration["robot"],
                dict(
                    os.environ,
                    BENCHMARK_WORKER_OWNER=attempt["token"],
                    ROS_DOMAIN_ID=str(attempt["domain"]),
                    GZ_PARTITION=attempt["partition"],
                ),
            )
            self.process = self.scope.start(
                [
                    sys.executable,
                    "-m",
                    "exploration_benchmark_runner.worker",
                    "--configuration",
                    str(launch_file),
                ],
                environment,
                self.directory / "worker.log",
                configuration["project_root"],
            )
            self.pid = self.process.pid
            self.pid_created = psutil.Process(self.pid).create_time()
            self.thread = threading.Thread(
                target=self._transport, name="suite-worker-" + attempt["id"][:8], daemon=False
            )
            self.thread.start()
        except BaseException:
            try:
                self.scope.stop()
            finally:
                self.client.close()
                self.lease.close()
            raise

    def _emit(self, event: dict[str, Any]) -> None:
        self.events.put(event, timeout=10)

    def _transport(self) -> None:
        manifest: dict[str, Any] = {}
        try:
            while self.process.poll() is None:
                if (
                    not self.stop_requested
                    and self.last_state in ("", "STARTING")
                    and time.monotonic() - self.started_wall
                    > self.configuration.get("startup_timeout", 180) + 10
                ):
                    self.stop("startup_failure", "Parent startup deadline exceeded; see worker.log")
                if self.stop_requested:
                    reason, detail = self.stop_requested
                    if time.monotonic() - (self.stop_wall or time.monotonic()) > 30:
                        self.scope.stop()
                        break
                    try:
                        self.client.call("worker.stop", {"reason": reason, "detail": detail})
                    except (OSError, ContractError, EOFError):
                        pass
                try:
                    result = self.client.call("worker.status")
                    if result["sequence"] != self.sequence:
                        self.sequence = result["sequence"]
                        state, metrics = result["state"], result["metrics"]
                        if state["phase"] != self.last_state:
                            self.last_state = state["phase"]
                            self._emit(dict(kind="state", **state))
                        if metrics and metrics.get("wall_monotonic") != self.last_metrics_wall:
                            self.latest_metrics = metrics
                            self.last_metrics_wall = metrics["wall_monotonic"]
                            self._emit(dict(kind="metrics", **metrics))
                        if state.get("outcome") and not self.terminal:
                            self.terminal = True
                            self.trigger = dict(
                                kind="terminal",
                                outcome=state["outcome"],
                                detail=state["detail"],
                                metrics=metrics,
                            )
                            self._emit(self.trigger)
                    with self.lock:
                        inspection = self.inspection_request
                        self.inspection_request = None
                    if inspection:
                        result = self.client.call(
                            "worker.inspect",
                            {"after_sequence": inspection[0], "map_revision": inspection[1]},
                        )
                        with self.lock:
                            self.latest_inspection = result
                except (OSError, ContractError, EOFError):
                    pass
                time.sleep(0.2)
            self.process.wait(timeout=2)
            self.scope.stop()
            path = self.directory / "manifest.json"
            manifest = json.loads(path.read_text()) if path.exists() else {}
            if manifest.get("phase") == "FINISHED":
                outcome, detail = manifest["outcome"], manifest.get("detail", "")
                if not self.stop_requested or self.stop_requested[0] != "manual_discard":
                    finalize_attempt(self.directory)
            else:
                outcome = (
                    self.trigger["outcome"]
                    if self.trigger
                    else self.stop_requested[0]
                    if self.stop_requested
                    else "runtime_failure"
                    if self.last_state == "RUNNING"
                    else "startup_failure"
                )
                if outcome in ("completed", "time_limit"):
                    outcome = "storage_failure"
                detail = f"Worker exited before artifact commit (exit {self.process.returncode})"
                if outcome != "manual_discard":
                    manifest = finalize_partial(
                        self.directory,
                        outcome,
                        detail,
                        self.trigger.get("metrics", self.latest_metrics)
                        if self.trigger
                        else self.latest_metrics,
                    )
            self._emit(
                dict(
                    kind="terminal",
                    outcome=outcome,
                    detail=detail,
                    metrics=manifest.get(
                        "final_metrics", self.trigger.get("metrics", {}) if self.trigger else {}
                    ),
                )
            )
        except (OSError, ValueError) as exc:
            self._emit(
                dict(
                    kind="terminal",
                    outcome="storage_failure",
                    detail=str(exc),
                    metrics=manifest.get("final_metrics", {}),
                )
            )
        except Exception as exc:
            self._emit(
                dict(
                    kind="terminal",
                    outcome="runtime_failure",
                    detail=str(exc),
                    metrics=manifest.get("final_metrics", {}),
                )
            )
        finally:
            self.client.close()
            try:
                self.scope.stop()
            except RuntimeError as exc:
                self._emit(dict(kind="cleanup_failure", detail=str(exc)))
                return
            self.lease.close()
            self.closed = True

    def poll(self) -> list[dict[str, Any]]:
        events = []
        while not self.events.empty():
            events.append(self.events.get_nowait())
        return events

    def stop(self, reason: str, detail: str) -> None:
        if self.stop_requested is None:
            self.stop_wall = time.monotonic()
            self.stop_requested = reason, detail

    def inspection(self, after: int, map_revision: int) -> dict[str, Any]:
        with self.lock:
            self.inspection_request = after, map_revision
            result = dict(self.latest_inspection)
            first = result.get("sequence", 0) - len(result.get("path", [])) + 1
            result["path"] = result.get("path", [])[max(0, after - first + 1) :]
            if result.get("map_revision") == map_revision:
                result["map"] = None
            return result


class NativeBackend:
    synthetic = False

    def __init__(self, root: Path, *, diagnostics: bool = True):
        self.root = root.resolve()
        self.diagnostics = diagnostics
        self.graphics_checked_at = 0.0
        self.profiles = yaml.safe_load(
            (self.root / "exploration_benchmark_runner/config/profiles.yaml").read_text()
        )["profiles"]
        source_files = {}
        for name in (
            "benchmark_suite",
            "exploration_benchmark_runner",
            "exploration_benchmark_robots",
        ):
            module = importlib.import_module(name)
            if module.__file__ is None:
                raise RuntimeError("Runtime package does not expose its source location: " + name)
            directory = Path(module.__file__).parent
            source_files[name] = {
                str(path.relative_to(directory)): file_hash(path)
                for path in directory.rglob("*.py")
            }
        self.runtime = {
            "image": os.environ.get("BENCHMARK_IMAGE_ID"),
            "python": sys.version,
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "diagnostics_enabled": diagnostics,
            "python_packages": {
                name: version(name)
                for name in (
                    "exploration-benchmark-suite",
                    "numpy",
                    "pyarrow",
                    "duckdb",
                    "matplotlib",
                    "scipy",
                    "zstandard",
                    "psutil",
                    "PyYAML",
                )
            },
            "source_files": source_files,
            "dependencies": subprocess.check_output(
                ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"], text=True
            ),
            "common_configurations": {
                str(path.relative_to(self.root)): file_hash(path)
                for path in (
                    self.root / "simulation/Week-7-8-ROS2-Navigation/bme_ros2_navigation/config"
                ).glob("*.yaml")
            },
        }
        for package in (self.root, self.root / "exploration_packages/frontier_exploration_ros2"):
            result = subprocess.run(
                ["git", "-C", str(package), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.runtime[str(package)] = result.stdout.strip() if result.returncode == 0 else None

    def preflight(self) -> None:
        if time.monotonic() - self.graphics_checked_at < 1.0:
            return
        try:
            result = subprocess.run(
                [sys.executable, "-m", "benchmark_suite.rendering"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired as exc:
            raise ContractError("runtime", "Headless GPU initialization timed out") from exc
        if result.returncode:
            raise ContractError(
                "runtime",
                "The background engine cannot access GPU rendering. "
                "Stop the engine after active runs finish and reopen the suite. "
                + (result.stderr.strip() or result.stdout.strip()),
                "graphics_unavailable",
            )
        self.runtime["graphics"] = json.loads(result.stdout)
        self.graphics_checked_at = time.monotonic()

    def start(
        self, attempt: dict[str, Any], experiment: dict[str, Any], campaign: dict[str, Any]
    ) -> NativeWorker:
        accepted = json.loads(experiment["specification"])
        time_limit = accepted.get("time_limit")
        if time_limit is None:
            specification = json.loads(campaign["specification"])
            time_limit = dict(resolve_world_time_limits(specification, specification["worlds"]))[
                experiment["world_id"]
            ]
        selected = accepted["profile"]
        if "object" in selected:
            selected = load_object(Path(attempt["directory"]).parents[1], selected["object"])
        robot = selected["robot"]
        world = scenario(self.root, experiment["world_id"])
        pose = accepted["spawn"]
        default_z = world["spawn"][2]
        height = (
            default_z
            if robot.get("spawn_z_policy") == "scenario"
            else robot.get("spawn_z_by_world", {}).get(
                experiment["world_id"], robot.get("spawn_z", default_z)
            )
        )
        world["spawn"] = [
            pose["x"],
            pose["y"],
            height,
            pose["yaw"],
        ]
        reference = Reference(
            self.root
            / "exploration_benchmark_runner/config/coverage"
            / (experiment["world_id"] + ".json")
        )
        world["coverage_reference"] = reference.descriptor()
        root = Path(attempt["directory"]).parents[1]
        runtime_hash = put_object(root, self.runtime)
        socket_dir = Path(f"/tmp/benchmark-suite-{os.getuid()}/workers")
        socket_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        config = dict(
            slot=str(attempt["slot"]),
            run_id=attempt["id"],
            domain_id=attempt["domain"],
            observer_domain=0,
            gazebo_partition=attempt["partition"],
            process_token=attempt["token"],
            result_directory=attempt["directory"],
            project_root=str(self.root),
            session_root="/campaign/" + campaign["id"],
            single=True,
            scenario=world,
            spec=dict(
                profile="default",
                label=selected["name"],
                parameter_file="",
                overrides=selected["effective"],
            ),
            profile=self.profiles["default"],
            seed=1234,
            simulation_time_limit=time_limit,
            startup_timeout=180.0,
            robot=robot,
            suite_socket=str(socket_dir / (attempt["token"] + ".sock")),
            suite_workspace=str(root),
            suite_identity={
                "campaign_id": campaign["id"],
                "experiment_id": experiment["id"],
                "attempt_id": attempt["id"],
                "robot_id": experiment["robot_id"],
                "world_id": experiment["world_id"],
                "profile_id": experiment["profile_id"],
                "spawn_id": experiment["spawn_id"],
                "repetition": experiment["repetition"],
            },
            requested_parameters=selected["requested"],
            effective_parameters=selected["effective"],
            runtime={"object": runtime_hash, "image": self.runtime["image"]},
            suite_diagnostics=self.diagnostics,
        )
        return NativeWorker(attempt, WorkerLaunchSpec.from_configuration(config).dictionary())

    def recover(self, attempts: list[dict[str, Any]]) -> None:
        for attempt in attempts:
            scope = ProcessScope(attempt["token"])
            if attempt.get("pid") and attempt.get("pid_created"):
                scope.tracked[attempt["pid"]] = attempt["pid_created"]
            scope.stop()
