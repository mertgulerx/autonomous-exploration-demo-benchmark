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

"""Session supervisor. RViz consumes state and requests work; it owns no processes."""

from dataclasses import dataclass
import json
import csv
import psutil
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import uuid

import rclpy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.node import Node
from exploration_benchmark_interfaces.msg import (
    Profile,
    ProfileCatalog,
    RunState,
    SessionState,
    RunMetrics,
    Scenario,
    ScenarioCatalog,
    ParameterCatalog,
    RobotCatalog as RobotCatalogMessage,
)
from exploration_benchmark_interfaces.srv import (
    StartSingleRun,
    StartComparison,
    StopRun,
    StopComparison,
    ReturnToConfiguration,
    ResolveParameters,
    WorkerCommand,
)
from .configuration import GAZEBO_SEED, Catalog, Specification, scenario, scenarios
from .parameters import parameter_values, ros_parameters
from .processes import ProcessScope, DomainLease
from .artifacts import write_json, freeze_interrupted_coverage
from .coverage import attach_reference
from .worker import LATCHED
from exploration_benchmark_robots.catalog import RobotCatalog


@dataclass
class Handle:
    slot: str
    state: RunState
    scope: ProcessScope
    process: object
    client: object
    subscription: object
    config: dict
    metrics: object = None


class Coordinator(Node):
    def __init__(
        self,
        root,
        world=None,
        session=None,
        pool=None,
        results=None,
        startup_timeout=180,
    ):
        self.session_id = session or "session_" + uuid.uuid4().hex[:12]
        super().__init__(
            "benchmark_coordinator_" + self.session_id, use_global_arguments=False
        )
        self.root = str(Path(root).resolve())
        self.scenario = None
        self.available_scenarios = {
            item["world"]: item for item in scenarios(self.root)
        }
        if world and world not in self.available_scenarios:
            raise ValueError(f"Unknown world: {world}")
        self.preselected_world = world or ""
        self.observer = int(os.environ.get("ROS_DOMAIN_ID", "0"))
        self.pool = pool or list(range(41, 101))
        self.session_root = "/benchmark/" + self.session_id
        self.results = (
            Path(results or Path(self.root) / "results/runs").expanduser().resolve()
        )
        self.startup_timeout = startup_timeout
        self.catalog = Catalog(root=self.root)
        self.robots = RobotCatalog()
        self.robot = None
        self.handles = []
        self.mode, self.busy, self.detail = (
            "IDLE",
            False,
            "Select a world and configurations in RViz",
        )
        self.commands = queue.Queue()
        self.lock = threading.RLock()
        self.closed = False
        self.lease = None
        self.released = False
        self.batch_directory = None
        self.last_capacity_wall = 0.0
        self.capacity_previous = {}
        self.state_pub = self.create_publisher(
            SessionState, self.session_root + "/state", LATCHED
        )
        self.catalog_pub = self.create_publisher(
            ProfileCatalog, self.session_root + "/catalog", LATCHED
        )
        self.robot_pub = self.create_publisher(
            RobotCatalogMessage, self.session_root + "/robots", LATCHED
        )
        self.robot_pub.publish(self.robots.message())
        self.create_subscription(
            RunMetrics, self.session_root + "/metrics", self.metrics_received, 30
        )
        self.reconciled_metrics_pub = self.create_publisher(
            RunMetrics, self.session_root + "/metrics", 20
        )
        self.create_service(
            StartSingleRun,
            self.session_root + "/start_single_run",
            self.start_single_request,
        )
        self.create_service(
            StartComparison,
            self.session_root + "/start_comparison",
            self.start_comparison_request,
        )
        self.create_service(
            StopRun, self.session_root + "/stop_run", self.stop_run_request
        )
        self.create_service(
            StopComparison,
            self.session_root + "/stop_comparison",
            self.stop_comparison_request,
        )
        self.create_service(
            ReturnToConfiguration,
            self.session_root + "/return_to_configuration",
            self.single_mode_request,
        )
        profiles = [
            Profile(
                id=key,
                label=value["label"],
                framework=value["framework"],
                default_parameter_file=self.catalog.parameters(Specification(key)),
            )
            for key, value in self.catalog.profiles.items()
        ]
        self.catalog_pub.publish(
            ProfileCatalog(
                profiles=profiles, default_comparison=self.catalog.default_comparison
            )
        )
        self.callback_executor = SingleThreadedExecutor()
        self.callback_executor.add_node(self)
        self.thread = threading.Thread(target=self.spin_callbacks, daemon=True)
        self.thread.start()
        self.scenario_pub = self.create_publisher(
            ScenarioCatalog, self.session_root + "/scenarios", LATCHED
        )
        self.parameter_pub = self.create_publisher(
            ParameterCatalog, self.session_root + "/parameters", LATCHED
        )
        self.scenario_pub.publish(
            ScenarioCatalog(
                scenarios=[
                    self.scenario_message(name, item)
                    for name, item in self.available_scenarios.items()
                ],
                preselected_world=self.preselected_world,
            )
        )
        parameter_catalog = self.catalog.schema_for_robot(
            self.robots.profile()
        ).message()
        parameter_catalog.robot_id = self.robots.default
        self.parameter_pub.publish(parameter_catalog)
        self.create_service(
            ResolveParameters,
            self.session_root + "/resolve_parameters",
            self.resolve_parameters_request,
        )
        self.publish_state()

    def scenario_message(self, name, item):
        # Catalog publication is idle/read-only. Full source compatibility is
        # checked afresh on every start, before domains or processes are owned.
        try:
            reference = attach_reference(self.root, item, validate=False)[
                "coverage_reference"
            ]
        except ValueError:
            reference = {}
        return Scenario(
            id=name,
            label=name.title(),
            spawn=item["spawn"],
            coverage_reference_id=reference.get("id", ""),
            coverage_metric_version=reference.get("metric_version", 0),
            coverage_target_m2=reference.get("target_area_m2", 0.0),
        )

    def spin_callbacks(self):
        try:
            self.callback_executor.spin()
        except ExternalShutdownException:
            self.closed = True

    def publish_state(self):
        with self.lock:
            self.state_pub.publish(
                SessionState(
                    session_id=self.session_id,
                    mode=self.mode,
                    busy=self.busy,
                    runs=[h.state for h in self.handles],
                    detail=self.detail,
                    world=self.scenario["world"] if self.scenario else "",
                    robot_id=self.robot["id"] if self.robot else "",
                )
            )

    def metrics_received(self, message):
        with self.lock:
            for handle in self.handles:
                if handle.state.run_id == message.run_id:
                    handle.metrics = message

    def publish_reconciled_metrics(self, handle, manifest):
        # A killed observer cannot publish its own frozen sample. Keep the
        # retained cards honest using the coordinator's last available telemetry.
        if handle.metrics:
            handle.metrics.frozen = True
            handle.metrics.coverage_quality_flags = manifest.get(
                "final_metrics", {}
            ).get("coverage_quality_flags", [])
            self.reconciled_metrics_pub.publish(handle.metrics)

    def resolve_parameters_request(self, request, response):
        try:
            schema = self.catalog.schema_for_robot(
                self.robots.profile(request.robot_id or "mogi")
            )
            values = schema.resolve(request.yaml, parameter_values(request.overrides))
            response.parameters = ros_parameters(values)
            response.yaml = schema.yaml(values)
            response.accepted, response.message = True, "Parameters valid"
        except (ValueError, TypeError) as exc:
            response.accepted, response.message = False, str(exc)
        return response

    def submit(self, request, response, single):
        with self.lock:
            lease = None
            try:
                target_mode = "SINGLE" if single else "COMPARISON"
                if self.mode in ("SWITCHING", "STOPPING") or (
                    self.mode == target_mode and self.busy
                ):
                    raise ValueError("A run or mode transition is already active")
                if request.world not in self.available_scenarios:
                    raise ValueError(
                        "Select a supported world: "
                        + ", ".join(self.available_scenarios)
                    )
                selected = attach_reference(
                    self.root, scenario(self.root, request.world)
                )
                robot = self.robots.require(
                    request.robot_id or "mogi",
                    request.world,
                    parameter_values(request.robot_overrides),
                )
                from exploration_benchmark_robots.preflight import validate_spawn
                from .coverage import Reference

                validate_spawn(
                    robot,
                    selected["spawn"],
                    Reference(selected["coverage_reference"]["metadata_file"]),
                )
                specs = self.catalog.validate(
                    (
                        [Specification.from_message(request.spec)]
                        if single
                        else [Specification.from_message(item) for item in request.runs]
                    ),
                    request.simulation_time_limit,
                    comparison=not single,
                    robot=robot,
                )
                lease = (
                    None
                    if single
                    else DomainLease(self.pool, self.observer, len(specs))
                )
                ids = ["run_" + uuid.uuid4().hex[:12] for _ in specs]
                self.busy, self.mode, self.detail = (
                    True,
                    "SWITCHING",
                    "Saving and stopping the previous mode",
                )
                self.commands.put(
                    (
                        "batch",
                        (
                            specs,
                            ids,
                            GAZEBO_SEED,
                            request.simulation_time_limit,
                            selected,
                            single,
                            lease,
                            robot,
                        ),
                    )
                )
                response.accepted, response.message = True, "Accepted"
                if single:
                    response.run_id = ids[0]
                else:
                    response.run_ids = ids
            except (ValueError, OSError, TypeError) as exc:
                if lease:
                    lease.close()
                response.accepted, response.message = False, str(exc)
        return response

    def start_single_request(self, request, response):
        return self.submit(request, response, True)

    def start_comparison_request(self, request, response):
        return self.submit(request, response, False)

    def stop_run_request(self, request, response):
        with self.lock:
            handle = next(
                (h for h in self.handles if h.state.run_id == request.run_id), None
            )
            response.accepted = bool(
                handle and handle.state.phase not in ("FINISHED", "STOPPING")
            )
            response.message = (
                "Accepted" if response.accepted else "Run is absent or already terminal"
            )
            if response.accepted:
                self.commands.put(("stop_run", handle))
        return response

    def stop_comparison_request(self, request, response):
        del request
        with self.lock:
            response.accepted = self.mode == "COMPARISON"
            response.message = (
                "Accepted" if response.accepted else "No comparison is active"
            )
            if response.accepted:
                self.commands.put(("stop_all", None))
        return response

    def single_mode_request(self, request, response):
        del request
        with self.lock:
            response.accepted = self.mode != "SWITCHING"
            response.message = (
                "Accepted"
                if response.accepted
                else "A mode transition is already active"
            )
            if response.accepted:
                self.mode, self.busy = "SWITCHING", True
                self.commands.put(("configuration", None))
        return response

    def prepare_worker(
        self,
        slot,
        run_id,
        domain,
        directory,
        spec,
        seed,
        limit,
        single=False,
    ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        config = dict(
            slot=slot,
            run_id=run_id,
            domain_id=domain,
            observer_domain=self.observer,
            gazebo_partition=self.session_id + "_" + slot,
            process_token=token,
            result_directory=str(directory),
            project_root=self.root,
            session_root=self.session_root,
            scenario=self.scenario,
            single=single,
            spec=spec.dictionary(),
            profile=self.catalog.profiles[spec.profile],
            seed=seed,
            simulation_time_limit=limit,
            startup_timeout=self.startup_timeout,
            robot=self.robot,
        )
        from .launch_spec import WorkerLaunchSpec

        config = WorkerLaunchSpec.from_configuration(config).dictionary()
        write_json(directory / "worker_configuration.json", config)
        state = RunState(
            run_id=run_id,
            robot_id=self.robot["id"],
            spec=spec.message(),
            phase="STARTING",
            domain_id=domain,
            gazebo_partition=config["gazebo_partition"],
            topic_root=self.session_root + "/runs/" + run_id,
            result_directory=str(directory),
            native_result_code=-1,
        )
        # Keep observer services in the observer domain. Only worker children
        # receive the source domain in their launch environment.
        scope = ProcessScope(token)
        client = self.create_client(
            WorkerCommand, self.session_root + "/workers/" + slot + "/command"
        )
        handle = Handle(slot, state, scope, None, client, None, config)
        handle.subscription = self.create_subscription(
            RunState,
            self.session_root + "/workers/" + slot + "/state",
            lambda msg: self.state_received(handle, msg),
            LATCHED,
        )
        with self.lock:
            self.handles.append(handle)
        return handle

    def launch_handle(self, handle):
        directory = Path(handle.config["result_directory"])
        from exploration_benchmark_robots.runtime import environment

        handle.process = handle.scope.start(
            [
                sys.executable,
                "-m",
                "exploration_benchmark_runner.worker",
                "--configuration",
                str(directory / "worker_configuration.json"),
            ],
            environment(handle.config["robot"]),
            directory / "agent.log",
            self.root,
        )

    def state_received(self, handle, state):
        with self.lock:
            handle.state = state
            handle.config.update(
                run_id=state.run_id,
                result_directory=state.result_directory,
                spec=dict(
                    profile=state.spec.profile,
                    label=state.spec.label,
                    parameter_file=state.spec.parameter_file,
                    overrides=parameter_values(state.spec.overrides),
                ),
            )
            if self.mode in ("SINGLE", "COMPARISON") and self.handles:
                self.busy = any(h.state.phase != "FINISHED" for h in self.handles)
        self.publish_state()

    def send(self, handle, command):
        deadline = time.monotonic() + 8
        while (
            not handle.client.service_is_ready()
            and time.monotonic() < deadline
            and not self.closed
        ):
            if handle.process is None or handle.process.poll() is not None:
                break
            time.sleep(0.05)
        if not handle.client.service_is_ready():
            raise RuntimeError(f"Worker {handle.slot} control service is unavailable")
        request = WorkerCommand.Request(command=command)
        future = handle.client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(8):
            raise RuntimeError(f"Worker {handle.slot} control timed out")
        response = future.result()
        if not response.accepted:
            raise RuntimeError(response.message)

    def control(self, handle, command):
        try:
            self.send(handle, command)
        except Exception as exc:
            self.get_logger().error(f"Worker {handle.slot}: {exc}")
            self.finish_crash(handle)

    def dispose(self):
        for handle in list(self.handles):
            if handle.process and handle.process.poll() is None:
                try:
                    self.send(handle, "shutdown")
                    handle.process.wait(timeout=22)
                except Exception as exc:
                    self.get_logger().warn(f"Owned shutdown escalation: {exc}")
            handle.scope.stop()
            if handle.process is None and handle.state.phase != "FINISHED":
                self.finish_pending(
                    handle, "cancelled", "Session ended before process startup"
                )
            # If an agent was killed before it could acknowledge shutdown,
            # reconcile its last checkpoint with an explicit terminal outcome.
            path = Path(handle.config["result_directory"]) / "manifest.json"
            if handle.process is not None and path.is_file():
                manifest = json.loads(path.read_text())
                if manifest.get("phase") != "FINISHED":
                    outcome = (
                        "manual_stop"
                        if handle.state.phase == "RUNNING"
                        else "cancelled"
                    )
                    manifest.update(
                        phase="FINISHED",
                        outcome=outcome,
                        detail="Session exit with last available checkpoint",
                        valid_comparison=False,
                        coverage_valid_comparison=False,
                        finished_unix=time.time(),
                    )
                    if handle.metrics:
                        from rosidl_runtime_py.convert import message_to_ordereddict

                        handle.metrics.frozen = True
                        manifest["final_metrics"] = message_to_ordereddict(
                            handle.metrics
                        )
                    freeze_interrupted_coverage(
                        manifest, manifest.get("final_metrics", {}), path.parent
                    )
                    self.publish_reconciled_metrics(handle, manifest)
                    write_json(path, manifest)
                    handle.state.phase, handle.state.outcome = "FINISHED", outcome
            self.destroy_client(handle.client)
            self.destroy_subscription(handle.subscription)
        self.summarize()
        with self.lock:
            self.handles.clear()
        if self.lease:
            self.lease.close()
            self.lease = None

    def finish_crash(self, handle):
        handle.scope.stop()
        prior_phase = handle.state.phase
        handle.state.phase = "FINISHED"
        handle.state.outcome = (
            "startup_failure"
            if prior_phase in ("STARTING", "READY") and not self.released
            else "runtime_failure"
        )
        handle.state.detail = f"Worker supervision failed (exit={handle.process.returncode if handle.process else None}); inspect agent.log"
        path = Path(handle.config["result_directory"]) / "manifest.json"
        manifest = (
            json.loads(path.read_text())
            if path.is_file()
            else dict(
                schema_version=4,
                configuration=handle.config,
                robot=handle.config["robot"],
            )
        )
        manifest.update(
            phase="FINISHED",
            outcome=handle.state.outcome,
            detail=handle.state.detail,
            valid_comparison=False,
            coverage_valid_comparison=False,
            finished_unix=time.time(),
        )
        if handle.metrics:
            from rosidl_runtime_py.convert import message_to_ordereddict

            handle.metrics.frozen = True
            manifest["final_metrics"] = message_to_ordereddict(handle.metrics)
        freeze_interrupted_coverage(
            manifest, manifest.get("final_metrics", {}), path.parent
        )
        self.publish_reconciled_metrics(handle, manifest)
        write_json(path, manifest)
        with open(path.parent / "events.jsonl", "a") as log:
            log.write(
                json.dumps(
                    dict(
                        event="agent_crash",
                        wall_unix=time.time(),
                        detail=handle.state.detail,
                    )
                )
                + "\n"
            )

    def finish_pending(self, handle, outcome, detail):
        """A validated row cancelled before spawning still gets full diagnostics."""
        from .artifacts import Recorder
        from .adapters import resolve_parameters

        spec = Specification(**handle.config["spec"])
        profile = self.catalog.profiles[spec.profile]
        recorder = Recorder(
            handle.config["result_directory"], handle.config, profile, self.root
        )
        resolve_parameters(spec, handle.config["result_directory"])
        recorder.finish(outcome, detail, {}, None, None, -1, [])
        handle.state.phase, handle.state.outcome, handle.state.detail = (
            "FINISHED",
            outcome,
            detail,
        )

    def record_capacity(self):
        wall = time.monotonic()
        if (
            not self.released
            or not self.batch_directory
            or wall - self.last_capacity_wall < 5
        ):
            return
        count = 0
        rss = cpu = 0.0
        current = {}
        category_cpu = dict(simulation=0.0, locomotion=0.0, sensing=0.0, other=0.0)
        category_rss = dict.fromkeys(category_cpu, 0.0)
        for handle in self.handles:
            for process in handle.scope.discover():
                try:
                    key = (process.pid, process.create_time())
                    times = process.cpu_times()
                    value = times.user + times.system
                    current[key] = value
                    command = process.cmdline()
                    name = next(
                        (Path(arg).name for arg in command if "/lib/" in arg),
                        Path(command[0]).name if command else "",
                    )
                    if name in ("gz", "ruby") or "gz-sim" in name:
                        category = "simulation"
                    elif name in ("ekf_node", "command_bridge"):
                        category = "locomotion"
                    elif name in ("parameter_bridge",):
                        category = "sensing"
                    else:
                        category = "other"
                    if key in self.capacity_previous:
                        delta = max(0, value - self.capacity_previous[key])
                        cpu += delta
                        category_cpu[category] += delta
                    memory = process.memory_info().rss / 1024**2
                    rss += memory
                    category_rss[category] += memory
                    count += 1
                except psutil.NoSuchProcess:
                    continue
        elapsed = wall - self.last_capacity_wall
        cpu_percent = cpu / elapsed * 100 if self.capacity_previous else 0.0
        self.capacity_previous = current
        self.last_capacity_wall = wall
        path = self.batch_directory / "capacity.csv"
        exists = path.exists()
        with open(path, "a") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(
                    [
                        "wall_unix",
                        "running_workers",
                        "owned_processes",
                        "total_cpu_percent",
                        "sum_rss_mib",
                        "host_available_mib",
                    ]
                    + [
                        f"{category}_{unit}"
                        for category in category_cpu
                        for unit in ("cpu_percent", "rss_mib")
                    ]
                )
            writer.writerow(
                [
                    time.time(),
                    sum(h.state.phase == "RUNNING" for h in self.handles),
                    count,
                    cpu_percent,
                    rss,
                    psutil.virtual_memory().available / 1024**2,
                ]
                + [
                    value
                    for category in category_cpu
                    for value in (
                        (
                            category_cpu[category] / elapsed * 100
                            if self.capacity_previous and elapsed > 0
                            else 0.0
                        ),
                        category_rss[category],
                    )
                ]
            )

    def summarize(self):
        if not self.batch_directory:
            return
        valid, excluded = [], []
        coverage_valid, coverage_excluded, reference_keys = [], [], set()
        robot_keys = set()
        for handle in self.handles:
            path = Path(handle.config["result_directory"]) / "manifest.json"
            if path.is_file():
                manifest = json.loads(path.read_text())
                robot_profile = manifest.get("robot_configuration", {}).get(
                    "comparison_profile"
                )
                robot_keys.add(json.dumps(robot_profile, sort_keys=True))
                record = dict(
                    run_id=handle.state.run_id,
                    label=handle.state.spec.label,
                    outcome=handle.state.outcome,
                    metrics=manifest.get("final_metrics", {}),
                )
                (valid if manifest.get("valid_comparison") else excluded).append(record)
                reference = manifest.get("coverage", {}).get("reference", {})
                key = (
                    reference.get("id"),
                    reference.get("metric_version"),
                    reference.get("grid_sha256"),
                    reference.get("metadata_sha256"),
                )
                reference_keys.add(key)
                coverage_record = dict(record, reference=reference)
                (
                    coverage_valid
                    if manifest.get("coverage_valid_comparison")
                    else coverage_excluded
                ).append(coverage_record)
        references_match = len(reference_keys) == 1 and all(
            v is not None for v in next(iter(reference_keys), (None,))
        )
        robots_match = len(robot_keys) == 1 and "null" not in robot_keys
        profiles_match = references_match and robots_match
        if not profiles_match:
            excluded.extend(valid)
            valid = []
        write_json(
            self.batch_directory / "summary.json",
            dict(
                valid_runs=valid,
                excluded_runs=excluded,
                comparison_valid=profiles_match and len(valid) == len(self.handles),
                equivalence_proven=False,
                robot_profiles_match=robots_match,
                coverage_valid_runs=coverage_valid if profiles_match else [],
                coverage_excluded_runs=(
                    coverage_excluded
                    if profiles_match
                    else coverage_excluded + coverage_valid
                ),
                coverage_references_match=references_match,
                coverage_comparison_valid=profiles_match
                and len(coverage_valid) == len(self.handles),
            ),
        )

    def execute(self, command, data):
        if command == "configuration":
            self.dispose()
            self.scenario = None
            self.robot = None
            self.mode, self.busy, self.detail = (
                "IDLE",
                False,
                "Select a world and configurations in RViz",
            )
        elif command == "batch":
            specs, ids, seed, limit, selected, single, lease, robot = data
            self.dispose()
            self.scenario = selected
            self.robot = robot
            self.lease, self.released = lease, False
            self.capacity_previous = {}
            self.last_capacity_wall = 0.0
            self.batch_directory = self.results / (
                ("single_" if single else "comparison_") + uuid.uuid4().hex[:12]
            )
            self.batch_directory.mkdir(parents=True)
            write_json(
                self.batch_directory / "comparison.json",
                dict(
                    session_id=self.session_id,
                    mode="SINGLE" if single else "COMPARISON",
                    seed=seed,
                    simulation_time_limit=limit,
                    scenario=selected,
                    robot=robot,
                    runs=[
                        dict(run_id=run_id, **spec.dictionary())
                        for run_id, spec in zip(ids, specs)
                    ],
                ),
            )
            self.mode, self.detail = (
                "SINGLE" if single else "COMPARISON"
            ), "Preparing fresh headless workers"
            domains = [self.observer] if single else lease.domains
            for spec, run_id, domain in zip(specs, ids, domains):
                self.prepare_worker(
                    run_id,
                    run_id,
                    domain,
                    self.batch_directory / run_id,
                    spec,
                    seed,
                    limit,
                    single=single,
                )
            self.launch_handle(self.handles[0])
        elif command == "stop_run":
            if data.process is None:
                self.finish_pending(
                    data, "manual_stop", "Stopped before process startup"
                )
            else:
                self.control(data, "stop")
        elif command == "stop_all":
            for handle in self.handles:
                if handle.state.phase not in ("FINISHED", "STOPPING"):
                    if handle.process is None:
                        self.finish_pending(
                            handle, "cancelled", "Batch stopped before exploration"
                        )
                    else:
                        self.control(handle, "stop" if self.released else "cancel")

    def monitor(self):
        for handle in list(self.handles):
            if (
                handle.process
                and handle.process.poll() is not None
                and handle.state.phase != "FINISHED"
            ):
                self.finish_crash(handle)
        if self.mode in ("SINGLE", "COMPARISON") and self.handles:
            states = [h.state for h in self.handles]
            if not self.released:
                active = [h for h in self.handles if h.process is not None]
                pending = [h for h in self.handles if h.process is None]
                if pending and active and all(h.state.phase == "READY" for h in active):
                    self.launch_handle(pending[0])
                failed = next((s for s in states if s.phase == "FINISHED"), None)
                if failed:
                    self.detail = "Preparation cancelled: " + failed.detail
                    for handle in self.handles:
                        if handle.state.phase in ("STARTING", "READY"):
                            if handle.process is None:
                                self.finish_pending(
                                    handle,
                                    "cancelled",
                                    "Preparation failed in another worker",
                                )
                            else:
                                self.control(handle, "cancel")
                elif all(s.phase == "READY" for s in states):
                    # Issue one release per worker. Actual clock/start offsets are
                    # recorded individually; this is a barrier, not lockstep.
                    self.released = True
                    self.detail = "Benchmark running"
                    for handle in self.handles:
                        self.control(handle, "start")
            if all(s.phase == "FINISHED" for s in states):
                self.busy = False
                self.detail = "Benchmark finished; final visualizations retained"
                if self.lease:
                    self.lease.close()
                    self.lease = None
                self.summarize()
        self.record_capacity()
        self.publish_state()

    def run(self):
        try:
            while not self.closed and rclpy.ok():
                try:
                    command, data = self.commands.get(timeout=0.2)
                    self.execute(command, data)
                except queue.Empty:
                    pass
                except Exception as exc:
                    self.get_logger().error(str(exc))
                    self.detail = str(exc)
                    # A partially prepared batch may never start exploration.
                    for handle in self.handles:
                        if handle.state.phase not in ("FINISHED", "STOPPING"):
                            try:
                                self.control(handle, "cancel")
                            except Exception:
                                self.finish_crash(handle)
                try:
                    self.monitor()
                except Exception as exc:
                    self.get_logger().error(f"Supervision failure: {exc}")
                    self.detail = str(exc)
                    for handle in self.handles:
                        if handle.state.phase not in ("FINISHED", "STOPPING"):
                            if handle.process is None:
                                self.finish_pending(handle, "cancelled", str(exc))
                            else:
                                self.finish_crash(handle)
        finally:
            self.mode, self.detail = (
                "STOPPING",
                "Saving results and stopping owned processes",
            )
            self.publish_state()
            self.dispose()
            while not self.commands.empty():
                command, data = self.commands.get_nowait()
                if command == "batch" and data[-1]:
                    data[-1].close()
            self.callback_executor.shutdown()
            self.thread.join(timeout=3)
            self.destroy_node()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", default=os.environ.get("BENCHMARK_PROJECT_ROOT", os.getcwd())
    )
    parser.add_argument(
        "--world", help="Preselect a world in RViz; never starts a simulation"
    )
    parser.add_argument("--session", default=os.environ.get("BENCHMARK_SESSION_ID"))
    parser.add_argument(
        "--domain-pool", default=os.environ.get("BENCHMARK_DOMAIN_POOL", "41:100")
    )
    parser.add_argument("--results")
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    if ":" in args.domain_pool:
        lower, upper = map(int, args.domain_pool.split(":"))
        pool = list(range(lower, upper + 1))
    else:
        pool = list(map(int, args.domain_pool.split(",")))
    from exploration_benchmark_robots.runtime import configure_transport

    configure_transport()
    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    coordinator = Coordinator(
        args.project_root,
        world=args.world,
        session=args.session,
        pool=pool,
        results=args.results,
        startup_timeout=args.startup_timeout,
    )
    signal.signal(signal.SIGINT, lambda *_: setattr(coordinator, "closed", True))
    signal.signal(signal.SIGTERM, lambda *_: setattr(coordinator, "closed", True))
    try:
        coordinator.run()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
