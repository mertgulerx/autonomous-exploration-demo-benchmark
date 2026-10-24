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

"""Unattended fresh benchmarks using the same typed requests as RViz."""

import argparse
import os
from pathlib import Path
import signal
import threading
import time

import rclpy
import yaml
from exploration_benchmark_interfaces.srv import StartComparison, StartSingleRun
from .configuration import (
    GAZEBO_SEED,
    Catalog,
    Specification,
    installed_worlds,
    scenario,
    scenarios,
)
from .coordinator import Coordinator
from .coverage import attach_reference
from .parameters import ros_parameters
from exploration_benchmark_robots.catalog import RobotCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", help="Comparison YAML")
    parser.add_argument(
        "--project-root", default=os.environ.get("BENCHMARK_PROJECT_ROOT", os.getcwd())
    )
    parser.add_argument("--results")
    parser.add_argument(
        "--robot", help="Robot catalog ID; defaults to YAML robot_id or mogi"
    )
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    try:
        data = yaml.safe_load(Path(args.configuration).read_text())
        if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
            raise ValueError("Configuration requires a runs list")
        if "world" not in data:
            raise ValueError("Benchmark YAML must select a world")
        worlds = installed_worlds(args.project_root)
        available = {item["world"] for item in scenarios(args.project_root, worlds)}
        if not isinstance(data["world"], str) or data["world"] not in available:
            raise ValueError(
                "Select a supported world: " + ", ".join(sorted(available))
            )
        if "spawn" in data:
            raise ValueError("Spawn poses are fixed by the selected scenario")
        specs = [Specification(**row) for row in data["runs"]]
        if "seed" in data and (
            type(data["seed"]) is not int or data["seed"] != GAZEBO_SEED
        ):
            raise ValueError("Gazebo seed is fixed at 1234; remove seed from the YAML")
        limit = float(data.get("simulation_time_limit", 1800))
        # Relative files are relative to the configuration, not the shell cwd.
        specs = [
            Specification(
                s.profile,
                s.label,
                (
                    str(
                        (
                            Path(args.configuration).expanduser().resolve().parent
                            / Path(s.parameter_file).expanduser()
                        ).resolve()
                    )
                    if s.parameter_file
                    else ""
                ),
                s.overrides,
            )
            for s in specs
        ]
        single = len(specs) == 1
        robot_id = args.robot or data.get("robot_id", "mogi")
        robot_overrides = data.get("robot_overrides", {})
        robot = RobotCatalog(worlds=worlds).require(
            robot_id, data["world"], robot_overrides
        )
        specs = Catalog(root=args.project_root).validate(
            specs, limit, comparison=not single, robot=robot
        )
        accepted_scenario = attach_reference(
            str(Path(args.project_root).resolve()),
            scenario(args.project_root, data["world"], registry=worlds),
        )
        from exploration_benchmark_robots.preflight import validate_spawn
        from .coverage import Reference

        validate_spawn(
            robot,
            accepted_scenario["spawn"],
            Reference(accepted_scenario["coverage_reference"]["metadata_file"]),
        )
    except (ValueError, TypeError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    from exploration_benchmark_robots.runtime import configure_transport

    configure_transport()
    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    coordinator = Coordinator(
        args.project_root,
        data["world"],
        pool=data.get("domain_pool"),
        results=args.results,
        startup_timeout=args.startup_timeout,
    )
    signal.signal(signal.SIGINT, lambda *_: setattr(coordinator, "closed", True))
    signal.signal(signal.SIGTERM, lambda *_: setattr(coordinator, "closed", True))
    options = dict(
        world=data["world"],
        robot_id=robot_id,
        simulation_time_limit=limit,
        robot_overrides=ros_parameters(robot_overrides),
    )
    if single:
        request = StartSingleRun.Request(
            spec=specs[0].message(),
            **options,
        )
        response = coordinator.start_single_request(request, StartSingleRun.Response())
    else:
        request = StartComparison.Request(
            runs=[s.message() for s in specs],
            **options,
        )
        response = coordinator.start_comparison_request(
            request, StartComparison.Response()
        )
    if not response.accepted:
        coordinator.closed = True
        coordinator.run()
        raise SystemExit(response.message)
    thread = threading.Thread(target=coordinator.run)
    thread.start()
    # Interrupted or unexpectedly exited coordinators must not look like a
    # successfully completed unattended experiment.
    exit_code = 1
    previous = None
    try:
        while thread.is_alive() and not coordinator.closed:
            with coordinator.lock:
                states = [
                    (h.state.spec.label, h.state.phase, h.state.outcome, h.state.detail)
                    for h in coordinator.handles
                ]
                finished = (
                    coordinator.mode in ("SINGLE", "COMPARISON")
                    and states
                    and all(s[1] == "FINISHED" for s in states)
                )
            if states != previous:
                for label, phase, outcome, detail in states:
                    print(f"{label}: {phase} {outcome} {detail}", flush=True)
                previous = states
            if finished:
                coordinator.summarize()
                print(f"Results: {coordinator.batch_directory}", flush=True)
                exit_code = int(
                    any(s[2] not in ("completed", "time_limit") for s in states)
                )
                break
            time.sleep(0.2)
    finally:
        coordinator.closed = True
        thread.join()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
