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

"""Viewers of a live run (Monitor → Launch Gazebo, Launch RViz).

The run lives in the engine container: its ROS graph in the attempt's ROS domain, its Gazebo server (headless)
in the attempt's Gazebo partition. A viewer starts in a separate, short-lived container from the running
engine's image that joins the engine's network namespace, so it reaches that run and nothing else. Each
viewer gets the host's X display (the X socket and a wildcard copy of the session's Xauthority cookie), GPU
devices as the engine has them, the checkout and the dataset read-only at the engine's paths and the robot's
vendor underlay, so the robot's meshes resolve as they do for the run.

- **RViz**: the configured ``bme_ros2_navigation/rviz/navigation.rviz`` of this checkout with its view settings
  unchanged except that the manual runner's Exploration Control panel is left out and the robot description is
  read transient-local (published once at the run's start, before the viewer subscribes); beside it the RViz
  package's ``frontier_path_tracker.py`` (with ``frontier_path_tracker.yaml``, from the checkout) publishes the
  traveled path its "Traveled Path Default" display draws. Simulation time; the window title names the run.
- **Gazebo**: the Gazebo Sim GUI client (``gz sim -g``, default GUI configuration) on the run's partition, with
  the run's resource paths (the robot's recorded ``resource_paths`` and the world launch's model directories).

The GUI owns each viewer container (``benchmark-<kind>-<uid>-<attempt>``, removed when it exits): closing the
window ends it, and the Monitor stops a run's viewers when the run ends or the GUI closes. Docker calls run off
the GUI thread.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from typing import Literal

from PySide6 import QtCore
import yaml

from .contracts import ContractError
from .deployment import DEFAULT_WORKSPACE, bind_mount, docker, gpu_arguments, inspect, workspace_key
from .yaml_io import load_mapping

Kind = Literal["gazebo", "rviz"]
TITLES: dict[Kind, str] = {"gazebo": "Gazebo", "rviz": "RViz"}
ROOT = Path(__file__).resolve().parents[2]
# The world registry (exploration_benchmark_runner/worlds.py REGISTRY_FILE).
REGISTRY_FILE = "exploration_benchmark_runner/config/worlds.yaml"
NAVIGATION_VIEW = (
    ROOT / "simulation/Week-7-8-ROS2-Navigation/bme_ros2_navigation/rviz/navigation.rviz"
)
PATH_TRACKER = ROOT / "rviz/src/frontier_path_tracker.py"
TRACKER_PARAMETERS = ROOT / "rviz/src/frontier_path_tracker.yaml"
VIEW_TARGET = "/tmp/benchmark-run-view.rviz"
CONTROL_PANEL = "rviz_autonomous_exploration_benchmark/ExplorationControlPanel"
LABEL = "org.benchmark.suite.viewer"
SETUP = (
    "if [[ -e /dev/nvidiactl && -r /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then "
    "export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json; fi; "
    "source /opt/ros/jazzy/setup.bash && source /opt/benchmark_ws/install/setup.bash && "
    'vendor="/opt/vendor/${BENCHMARK_ROBOT_FAMILY}/install/local_setup.bash"; '
    'if [[ -n "${BENCHMARK_ROBOT_FAMILY}" && -f "${vendor}" ]]; then source "${vendor}"; fi; '
)
# Run-specific values (title, family, paths) come in the environment, never in the script text.
SCRIPTS: dict[Kind, str] = {
    "rviz": SETUP
    + 'python3 "${BENCHMARK_PATH_TRACKER}" --ros-args --params-file "${BENCHMARK_TRACKER_PARAMETERS}" '
    "-p use_sim_time:=true & "
    f'exec rviz2 -d {VIEW_TARGET} -t "${{BENCHMARK_VIEWER_TITLE}}" --ros-args -p use_sim_time:=true',
    # As the world launch does: the bme_ros2_navigation share's parent directory after the recorded paths.
    "gazebo": SETUP + 'export GZ_SIM_RESOURCE_PATH="${BENCHMARK_GZ_RESOURCES}:'
    '$(dirname "$(ros2 pkg prefix --share bme_ros2_navigation)")"; '
    "exec gz sim -g -v1",
}


@dataclass(frozen=True, slots=True)
class RunView:
    """One viewer: its kind and the run it shows (attempt, ROS domain, Gazebo partition, window title, robot
    vendor family and Gazebo resource paths)."""

    kind: Kind
    attempt_id: str
    domain: int
    partition: str
    title: str
    family: str = ""
    resources: tuple[str, ...] = field(default_factory=tuple)


def robot_family(robot_id: str) -> str:
    """The robot's vendor underlay directory under /opt/vendor (as the robot adapters name it); empty when the
    catalog does not list the robot."""
    catalog = load_mapping((ROOT / "exploration_benchmark_robots/config/robots.yaml").read_text())
    robot = (catalog.get("robots") or {}).get(robot_id) or {}
    return str(robot.get("family") or "")


def gazebo_resources(directory: str) -> tuple[str, ...]:
    """The run's Gazebo resource paths: the robot's ``resource_paths`` recorded in the attempt manifest (absent
    before the robot is prepared), then the world launch's model directories in this checkout and every
    registry world's ``resource_paths`` (``worlds.yaml``), as the world launch adds them."""
    recorded: list[str] = []
    manifest = Path(directory) / "manifest.json"
    if directory and manifest.is_file():
        try:
            configuration = json.loads(manifest.read_text()).get("robot_configuration") or {}
        except (OSError, ValueError) as exc:
            raise ContractError("viewer", f"Cannot read the run's manifest: {exc}") from exc
        recorded = [str(path) for path in configuration.get("resource_paths") or []]
    models = [ROOT / "simulation/Week-7-8-ROS2-Navigation/gazebo_models"]
    models += sorted((ROOT / "simulation/worlds").glob("*/models"))
    models += [ROOT / path for path in registry_resource_paths()]
    return tuple(recorded + [str(path) for path in models if path.is_dir()])


def registry_resource_paths() -> list[str]:
    """Every registry world's extra Gazebo resource directories inside the checkout (repository-relative),
    in file order. The GUI has no runner package, so it reads the registry as plain YAML; the engine
    validates it."""
    try:
        registry = load_mapping((ROOT / REGISTRY_FILE).read_text())
    except (OSError, ContractError) as exc:
        raise ContractError("viewer", f"Cannot read the world registry: {exc}") from exc
    paths: list[str] = []
    worlds = registry.get("worlds")
    for world in worlds.values() if isinstance(worlds, dict) else ():
        for path in (world or {}).get("resource_paths") or []:
            inside = isinstance(path, str) and (ROOT / path).resolve().is_relative_to(ROOT)
            if inside and path not in paths:
                paths.append(path)
    return paths


def private_directory(environment: Mapping[str, str]) -> Path:
    """The user's private directory for viewer files (the cookie copy and the generated view)."""
    directory = Path(environment.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()) / (
        f"benchmark-viewers-{os.getuid()}"
    )
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory


def run_view(environment: Mapping[str, str]) -> Path:
    """``navigation.rviz`` without the Exploration Control panel, the robot description read transient-local;
    every other setting as configured. Written to the private directory and returned."""
    try:
        view = yaml.safe_load(NAVIGATION_VIEW.read_text())
        view["Panels"] = [p for p in view.get("Panels") or [] if p.get("Class") != CONTROL_PANEL]
        for display in view["Visualization Manager"]["Displays"]:
            if display.get("Class") == "rviz_default_plugins/RobotModel":
                display["Description Topic"]["Durability Policy"] = "Transient Local"
    except (OSError, yaml.YAMLError, KeyError, TypeError, AttributeError) as exc:
        raise ContractError("viewer", f"Cannot read {NAVIGATION_VIEW.name}: {exc}") from exc
    target = private_directory(environment) / "run_view.rviz"
    target.write_text(yaml.safe_dump(view, sort_keys=False, allow_unicode=True, width=4096))
    return target


def mount(source: Path, target: str) -> list[str]:
    """A read-only bind mount (CSV-quoted like the engine's)."""
    return ["--mount", bind_mount(source, readonly=True).replace(f"dst={source}", f"dst={target}")]


def viewer_name(kind: Kind, attempt_id: str) -> str:
    return f"benchmark-{kind}-{os.getuid()}-{attempt_id[:12]}"


def engine_container() -> tuple[str, str]:
    """The running engine's container name and image ID (this checkout's default workspace)."""
    name = f"benchmark-suite-{os.getuid()}-{workspace_key(DEFAULT_WORKSPACE)}"
    engine = inspect("container", name)
    labels = (engine["Config"].get("Labels") or {}) if engine else {}
    if (
        engine is None
        or labels.get("org.benchmark.suite.workspace") != str(DEFAULT_WORKSPACE)
        or labels.get("org.benchmark.suite.root") != str(ROOT)
    ):
        raise ContractError("viewer", "The engine of this checkout is not running")
    if not engine["State"].get("Running"):
        raise ContractError("viewer", "The engine container is not running")
    return name, str(engine["Image"])


def wildcard_cookie(display: str, cookie: Path, environment: Mapping[str, str]) -> Path | None:
    """A private copy of the display's Xauthority entries valid for any host name (family ``ffff``): the viewer
    has the engine container's host name, which a host-bound entry does not match. None without ``xauth``."""
    if not shutil.which("xauth"):
        return None
    listing = subprocess.run(
        ["xauth", "-f", str(cookie), "nlist", display],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout
    entries = ["ffff" + line[4:] for line in listing.splitlines() if len(line) > 4]
    if not entries:
        return None
    target = private_directory(environment) / "xauthority"
    target.unlink(missing_ok=True)
    target.touch(mode=0o600)
    subprocess.run(
        ["xauth", "-f", str(target), "nmerge", "-"],
        input="\n".join(entries) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return target


def display_arguments(environment: Mapping[str, str]) -> list[str]:
    """The X display: DISPLAY, the X socket directory and an Xauthority cookie, mounted read-only."""
    display = environment.get("DISPLAY", "")
    if not display:
        raise ContractError("viewer", "Viewers need an X display (DISPLAY is not set)")
    args = ["--env", "DISPLAY=" + display, "--env", "QT_X11_NO_MITSHM=1"]
    sockets = Path("/tmp/.X11-unix")
    if sockets.is_dir():
        args += ["--mount", bind_mount(sockets, readonly=True)]
    cookie = Path(environment.get("XAUTHORITY") or Path.home() / ".Xauthority")
    if cookie.is_file():
        source = wildcard_cookie(display, cookie, environment) or cookie
        args += [
            *mount(source, "/tmp/benchmark-xauthority"),
            "--env",
            "XAUTHORITY=/tmp/benchmark-xauthority",
        ]
    return args


def device_arguments() -> list[str]:
    """GPU access as the engine has it: /dev/dri with its groups, NVIDIA through CDI or the runtime."""
    args: list[str] = []
    devices = Path("/dev/dri")
    if devices.exists():
        args += ["--device", str(devices)]
        for group in sorted({path.stat().st_gid for path in devices.iterdir()}):
            args += ["--group-add", str(group)]
    _, gpu = gpu_arguments(json.loads(docker("info", "--format", "{{json .}}", timeout=30)))
    return args + gpu


def viewer_command(
    run: RunView, engine: str, image: str, display: list[str], devices: list[str], view: Path | None
) -> list[str]:
    """``docker run`` arguments of one viewer (without the leading ``docker``); ``view`` is RViz's
    configuration."""
    if run.kind == "rviz":
        if view is None or not (PATH_TRACKER.is_file() and TRACKER_PARAMETERS.is_file()):
            raise ContractError("viewer", f"The path tracker is missing: {PATH_TRACKER}")
        specific = [
            *mount(view, VIEW_TARGET),
            # The checkout is mounted below at its own path: the tracker and its parameters run from it.
            "--env",
            f"BENCHMARK_PATH_TRACKER={PATH_TRACKER}",
            "--env",
            f"BENCHMARK_TRACKER_PARAMETERS={TRACKER_PARAMETERS}",
        ]
    else:
        specific = ["--env", "BENCHMARK_GZ_RESOURCES=" + os.pathsep.join(run.resources)]
    return [
        "run",
        "-d",
        "--rm",
        "--name",
        viewer_name(run.kind, run.attempt_id),
        "--network",
        "container:" + engine,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--security-opt",
        "no-new-privileges",
        "--label",
        f"{LABEL}={run.kind}:{run.attempt_id}",
        "--env",
        f"ROS_DOMAIN_ID={run.domain}",
        "--env",
        "GZ_PARTITION=" + run.partition,
        "--env",
        "BENCHMARK_VIEWER_TITLE=" + run.title,
        "--env",
        "BENCHMARK_ROBOT_FAMILY=" + run.family,
        "--env",
        "HOME=/tmp",
        "--env",
        "ROS_HOME=/tmp/ros",
        *specific,
        # Robot descriptions and worlds may name files by path in the checkout or the dataset.
        "--mount",
        bind_mount(ROOT, readonly=True),
        "--mount",
        bind_mount(DEFAULT_WORKSPACE, readonly=True),
        *display,
        *devices,
        "--entrypoint",
        "/bin/bash",
        image,
        "-c",
        SCRIPTS[run.kind],
    ]


class RunViewers(QtCore.QObject):
    """The viewer containers this GUI started, by kind and attempt. ``launch`` and ``stop`` return at once;
    ``done`` reports a launch (kind, attempt ID and an error message, empty on success)."""

    done = QtCore.Signal(str, str, str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.pending: set[tuple[Kind, str]] = set()
        self.open: set[tuple[Kind, str]] = set()  # viewers this GUI started

    def launch(
        self,
        kind: Kind,
        attempt: Mapping[str, object],
        robot_id: str,
        title: str,
    ) -> None:
        """Start ``kind`` for an active attempt (its ``engine.status`` row)."""
        identity = str(attempt["id"])
        if (kind, identity) in self.pending:
            return
        self.pending.add((kind, identity))
        threading.Thread(
            target=self._launch,
            args=(kind, dict(attempt), robot_id, title, dict(os.environ)),
            name=f"suite-{kind}-{identity[:8]}",
            daemon=True,  # bounded by Docker timeouts; never keeps the GUI from exiting
        ).start()

    def _launch(
        self,
        kind: Kind,
        attempt: dict[str, object],
        robot_id: str,
        title: str,
        environment: dict[str, str],
    ) -> None:
        identity = str(attempt["id"])
        error = ""
        try:
            domain, partition = attempt.get("domain"), attempt.get("partition")
            if not isinstance(domain, int) or isinstance(domain, bool) or not partition:
                raise ContractError("viewer", "This run has no ROS domain or Gazebo partition")
            existing = inspect("container", viewer_name(kind, identity))
            if existing is not None and existing["State"].get("Running"):
                raise ContractError("viewer", f"{TITLES[kind]} is already open for this run")
            engine, image = engine_container()
            run = RunView(
                kind,
                identity,
                domain,
                str(partition),
                title,
                robot_family(robot_id),
                gazebo_resources(str(attempt.get("directory") or "")) if kind == "gazebo" else (),
            )
            command = viewer_command(
                run,
                engine,
                image,
                display_arguments(environment),
                device_arguments(),
                run_view(environment) if kind == "rviz" else None,
            )
            docker(*command, timeout=60)
        except ContractError as exc:
            error = exc.message
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip() or f"exit {exc.returncode}"
            error = f"{TITLES[kind]} did not start: {detail}"
        except (OSError, ValueError) as exc:
            error = f"{TITLES[kind]} did not start: {exc}"
        self.done.emit(kind, identity, error)

    def finished(self, kind: Kind, attempt_id: str, error: str) -> None:
        """Book-keeping for ``done`` (connected by the owner, on the GUI thread)."""
        self.pending.discard((kind, attempt_id))
        if not error:
            self.open.add((kind, attempt_id))

    def stop(self, attempt_id: str) -> None:
        """Stop the viewers of one run (each is removed when it exits); returns without waiting."""
        for kind, identity in tuple(self.open):
            if identity == attempt_id:
                self.open.discard((kind, identity))
                subprocess.Popen(
                    ["docker", "stop", "--time", "3", viewer_name(kind, identity)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

    def attempts(self) -> set[str]:
        """Runs with an open viewer."""
        return {identity for _, identity in self.open}

    def stop_all(self) -> None:
        for identity in self.attempts():
            self.stop(identity)
