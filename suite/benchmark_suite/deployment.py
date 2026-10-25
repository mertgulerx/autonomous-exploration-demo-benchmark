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

import argparse
import csv
from functools import partial
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import signal
from socket import gethostname
import subprocess
import time
from typing import Any, Callable, IO

from .contracts import ContractError
from .protocol import Client
from .provenance import (
    CHECKOUT_ROLE,
    HOST_FACTS,
    REPOSITORY_ROLES,
    RUNTIME_IMAGE_LABEL,
    RUNTIME_LABEL_PREFIX,
    SUITE_LABEL_PREFIX,
    image_labels,
)
from .storage import require_local_storage
from .store import upgrade_blockers

DEFAULT_IMAGE = "autonomous-exploration-benchmark:suite"
VERSION_LABEL = "org.benchmark.suite.version"
# The ID of the runtime image a suite image was built on (provenance; a flattened base has another ID).
BASE_ID_LABEL = "org.benchmark.suite.base-id"
# Campaign states during which the engine is not stopped for an image update.
BUSY_STATES = frozenset({"RUNNING", "RETRYING", "PAUSING", "STOPPING", "REPORTING"})
# Suite CLI commands that only read: they never build or replace the engine.
READ_ONLY_COMMANDS = frozenset({"status", "catalog", "query", "validate", "verify-export"})
BASE_IMAGE = "autonomous-exploration-benchmark:jazzy-harmonic"
# The runtime image's version (docker/RUNTIME_VERSION) and the flattened base its refresh builds on: each
# refresh adds about 20 layers, and Docker refuses images deeper than 127.
RUNTIME_VERSION_LABEL = "org.benchmark.runtime.version"
RUNTIME_FLAT_IMAGE = "autonomous-exploration-benchmark:runtime-flat"
FLATTEN_LAYERS = {"suite": 115, "runtime": 60}
DEFAULT_WORKSPACE = (Path(__file__).resolve().parents[2] / "results/campaigns/default").resolve()


def bind_mount(source: Path, *, readonly: bool = False) -> str:
    output = StringIO()
    csv.writer(output, lineterminator="").writerow(
        [
            "type=bind",
            "src=" + str(source),
            "dst=" + str(source),
            *(("readonly",) if readonly else ()),
        ]
    )
    return output.getvalue()


def workspace_key(workspace: Path) -> str:
    return hashlib.sha256(str(workspace.expanduser().resolve()).encode()).hexdigest()[:20]


def control_socket(workspace: Path) -> Path:
    runtime = (
        Path(
            os.environ.get(
                "BENCHMARK_SUITE_RUNTIME", str(Path.home() / ".cache/benchmark-suite/runtime")
            )
        )
        .expanduser()
        .resolve()
    )
    result = runtime / workspace_key(workspace) / "control.sock"
    if len(os.fsencode(result)) > 100:
        raise ContractError(
            "workspace",
            "Set BENCHMARK_SUITE_RUNTIME to a shorter local directory; Unix socket paths must fit 100 bytes",
        )
    return result


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled and cancelled():
        raise ContractError("runtime", "Workspace opening cancelled", "operation_cancelled")


def stop_command(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
    except ProcessLookupError:
        process.wait(timeout=2)


def docker(
    *args: str,
    capture: bool = True,
    cancelled: Callable[[], bool] | None = None,
    timeout: float = 120,
    input_stream: IO[bytes] | None = None,
) -> str:
    check_cancelled(cancelled)
    command = ["docker", *args]
    process = subprocess.Popen(
        command,
        stdin=input_stream,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            check_cancelled(cancelled)
            if time.monotonic() >= deadline:
                raise ContractError("runtime", f"Docker {args[0]} timed out; inspect the daemon")
            try:
                output, errors = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output, errors)
        return (output or "").strip()
    finally:
        stop_command(process)


def inspect(
    kind: str, name: str, *, cancelled: Callable[[], bool] | None = None
) -> dict[str, Any] | None:
    try:
        data = json.loads(docker(kind, "inspect", name, cancelled=cancelled))
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr or ""
        if "No such" in detail or (kind == "network" and f"network {name} not found" in detail):
            return None
        raise
    return dict(data[0])


def required_version(root: Path) -> int:
    """The suite image version this checkout needs (``suite/IMAGE_VERSION``: one integer, ``#`` comments)."""
    lines = (root / "suite/IMAGE_VERSION").read_text().splitlines()
    values = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if len(values) != 1 or not values[0].isdigit():
        raise ContractError("image", "suite/IMAGE_VERSION must hold one whole number")
    return int(values[0])


def runtime_required_version(root: Path) -> int:
    """The runtime image version this checkout needs (``docker/RUNTIME_VERSION``, the same format)."""
    lines = (root / "docker/RUNTIME_VERSION").read_text().splitlines()
    values = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if len(values) != 1 or not values[0].isdigit():
        raise ContractError("image", "docker/RUNTIME_VERSION must hold one whole number")
    return int(values[0])


def image_version(image: dict[str, Any] | None, label_name: str = VERSION_LABEL) -> int:
    """The version an installed image was built for (0: missing, or built before versioning)."""
    label = ((image or {}).get("Config", {}).get("Labels") or {}).get(label_name, "")
    return int(label) if str(label).isdigit() else 0


def engine_busy(socket: Path) -> bool:
    """Whether the running engine has active runs or a campaign in progress (True when it cannot tell)."""
    try:
        status = Client(socket, timeout=10).call("engine.status")
    except (OSError, ContractError, ValueError):
        return True
    rows = (status.get("campaigns") or {}).get("rows") or []
    return bool(status.get("active")) or any(row.get("state") in BUSY_STATES for row in rows)


def flatten(
    original: dict[str, Any],
    flattened: str,
    purpose: str,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> str:
    """``flattened``: one layer with ``original``'s files, labels, environment, working directory, entrypoint and
    command, rebuilt only when ``original`` changed (its ID in ``org.benchmark.base-source``)."""
    command = partial(docker, cancelled=cancelled)
    saved = inspect("image", flattened, cancelled=cancelled)
    if (
        saved is not None
        and (saved["Config"].get("Labels") or {}).get("org.benchmark.base-source") == original["Id"]
    ):
        return flattened
    print(f"Preparing {purpose}…", flush=True)
    container = command("create", original["Id"])
    try:
        config = original["Config"]
        changes = ["--change", "LABEL org.benchmark.base-source=" + original["Id"]]
        # Keep the original's labels (its repository states, provenance.py).
        for label, value in sorted((config.get("Labels") or {}).items()):
            changes.extend(("--change", f"LABEL {label}={json.dumps(value)}"))
        for entry in config.get("Env") or []:
            name, _, value = entry.partition("=")
            changes.extend(("--change", "ENV " + name + "=" + json.dumps(value)))
        if config.get("WorkingDir"):
            changes.extend(("--change", "WORKDIR " + config["WorkingDir"]))
        for key in ("Entrypoint", "Cmd"):
            if config.get(key):
                changes.extend(("--change", f"{key.upper()} {json.dumps(config[key])}"))
        with subprocess.Popen(
            ["docker", "export", container],
            stdout=subprocess.PIPE,
            start_new_session=True,
        ) as export:
            try:
                imported = command(
                    "import",
                    *changes,
                    "-",
                    flattened,
                    input_stream=export.stdout,
                    timeout=3600,
                )
                if export.stdout:
                    export.stdout.close()
                if export.wait(timeout=5) != 0:
                    raise OSError("Docker could not export " + flattened)
                if not imported:
                    raise OSError("Docker did not return an identity for " + flattened)
            finally:
                stop_command(export)
    finally:
        docker("rm", container, timeout=5)
    return flattened


def refresh_runtime(
    root: Path, *, cancelled: Callable[[], bool] | None = None, version: int = 0
) -> None:
    """Refresh the runtime image in place (``docker/Dockerfile.runtime``: runner, robots, interfaces, RViz,
    explorer, worlds, SLAM Toolbox) on a flattened copy of itself, labelled with ``version`` and its repository
    states. The full build with the vendor SDKs (``./docker.sh``, ``docker/Dockerfile``) stays manual."""
    original = inspect("image", BASE_IMAGE, cancelled=cancelled)
    if original is None:
        raise ContractError(
            "image", f"Build the robotics image with ./docker.sh first ({BASE_IMAGE} is missing)"
        )
    base = BASE_IMAGE
    if len(original["RootFS"]["Layers"]) > FLATTEN_LAYERS["runtime"]:
        base = flatten(
            original, RUNTIME_FLAT_IMAGE, "a flat runtime base for the refresh", cancelled=cancelled
        )
    labels = image_labels(root, RUNTIME_LABEL_PREFIX, REPOSITORY_ROLES)
    docker(
        "build",
        "-f",
        str(root / "docker/Dockerfile.runtime"),
        "--build-arg",
        "BENCHMARK_BASE=" + base,
        "--label",
        f"{RUNTIME_VERSION_LABEL}={version}",
        *(
            argument
            for name, value in labels.items()
            for argument in ("--label", f"{name}={value}")
        ),
        "-t",
        BASE_IMAGE,
        str(root),
        capture=False,
        timeout=7200,
        cancelled=cancelled,
    )


def build_image(
    root: Path, image: str, *, cancelled: Callable[[], bool] | None = None, version: int = 0
) -> None:
    command = partial(docker, cancelled=cancelled)
    lookup = partial(inspect, cancelled=cancelled)
    base = os.environ.get("BENCHMARK_SUITE_BASE", BASE_IMAGE)
    original = lookup("image", base)
    if original is None:
        raise ContractError(
            "image",
            f"Build the existing robotics image with ./docker.sh before building the suite ({base} is missing)",
        )
    if len(original["RootFS"]["Layers"]) > FLATTEN_LAYERS["suite"]:
        base = flatten(
            original,
            "autonomous-exploration-benchmark:suite-base",
            "the robotics base for the suite image",
            cancelled=cancelled,
        )
    # The checkout the suite code comes from (provenance; D-08). The explorer's labels are the base's.
    labels = image_labels(root, SUITE_LABEL_PREFIX, (CHECKOUT_ROLE,))
    command(
        "build",
        "-f",
        str(root / "docker/Dockerfile.suite"),
        "--build-arg",
        "BENCHMARK_BASE=" + base,
        "--label",
        f"{VERSION_LABEL}={version}",
        "--label",
        f"{BASE_ID_LABEL}={original['Id']}",
        *(
            argument
            for name, value in labels.items()
            for argument in ("--label", f"{name}={value}")
        ),
        "-t",
        image,
        str(root),
        capture=False,
        timeout=3600,
    )


def gpu_arguments(info: dict[str, Any]) -> tuple[str, list[str]]:
    if "nvidia" not in info.get("Runtimes", {}) or not shutil.which("nvidia-smi"):
        return "none", []
    if info.get("CDISpecDirs") and shutil.which("nvidia-ctk"):
        try:
            devices = subprocess.run(
                ["nvidia-ctk", "cdi", "list"], capture_output=True, text=True, timeout=10
            )
        except subprocess.TimeoutExpired as exc:
            raise ContractError("runtime", "NVIDIA device discovery timed out") from exc
        if devices.returncode == 0 and "nvidia.com/gpu=all" in devices.stdout.splitlines():
            return "cdi", [
                "--device",
                "nvidia.com/gpu=all",
                "--env",
                "NVIDIA_DRIVER_CAPABILITIES=all",
            ]
    args = ["--runtime", "nvidia", "--gpus", "all", "--env", "NVIDIA_DRIVER_CAPABILITIES=all"]
    for path in sorted(Path("/dev").glob("nvidia*")):
        if path.is_char_device() or path.name == "nvidia-caps" and path.is_dir():
            args.extend(("--device", str(path)))
    return "explicit", args


def ensure_engine(
    root: Path,
    workspace: Path,
    *,
    image: str = DEFAULT_IMAGE,
    rebuild: bool = False,
    diagnostics: bool | None = None,
    cancelled: Callable[[], bool] | None = None,
    update: bool = True,
) -> dict[str, Any]:
    """Start or reconnect the workspace's engine. With ``update``, an image older than ``suite/IMAGE_VERSION`` is
    rebuilt first; a running engine is stopped for that only while no campaign runs, otherwise it keeps its
    image until a later start finds it idle."""
    check_cancelled(cancelled)
    command = partial(docker, cancelled=cancelled)
    lookup = partial(inspect, cancelled=cancelled)
    if not shutil.which("docker"):
        raise ContractError("runtime", "Install Docker and allow this user to access its daemon")
    root, workspace = root.resolve(), workspace.expanduser().resolve()
    require_local_storage(workspace)
    if root.is_relative_to(workspace):
        raise ContractError(
            "workspace",
            "Choose a dataset directory that is not the project root or one of its parents",
        )
    workspace.mkdir(parents=True, exist_ok=True)
    socket = control_socket(workspace)
    socket.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if socket.parent.stat().st_uid != os.getuid() or socket.parent.is_symlink():
        raise ContractError(
            "socket",
            "The runtime directory must be owned by the current user and cannot be a symlink",
        )
    os.chmod(socket.parent, 0o700)
    name = f"benchmark-suite-{os.getuid()}-{workspace_key(workspace)}"
    network_name = name + "-network"
    info = json.loads(command("info", "--format", "{{json .}}"))
    gpu_mode, gpu_args = gpu_arguments(info)
    running = lookup("container", name)
    if running:
        labels = running["Config"].get("Labels") or {}
        if labels.get("org.benchmark.suite.workspace") != str(workspace) or labels.get(
            "org.benchmark.suite.root"
        ) != str(root):
            raise ContractError(
                "runtime", "The container name is already owned by a different installation"
            )
    current_diagnostics = (
        not running
        or (running["Config"].get("Labels") or {}).get("org.benchmark.suite.diagnostics") != "0"
    )
    accepted_diagnostics = current_diagnostics if diagnostics is None else diagnostics
    installed = lookup("image", image)
    version = required_version(root)
    # The runtime image under the suite image is refreshed by its own version (docker/RUNTIME_VERSION); a custom
    # base (BENCHMARK_SUITE_BASE) is left alone.
    runtime = None if os.environ.get("BENCHMARK_SUITE_BASE") else lookup("image", BASE_IMAGE)
    runtime_version = runtime_required_version(root)
    runtime_outdated = (
        runtime is not None and image_version(runtime, RUNTIME_VERSION_LABEL) != runtime_version
    )
    reasons = []
    if installed is not None and image_version(installed) != version:
        reasons.append(
            f"the suite image is version {image_version(installed)}, this checkout needs {version}"
        )
    if runtime_outdated:
        reasons.append(
            f"the runtime image is version {image_version(runtime, RUNTIME_VERSION_LABEL)}, "
            f"this checkout needs {runtime_version}"
        )
    elif (
        installed is not None
        and runtime is not None
        and (installed["Config"].get("Labels") or {}).get(BASE_ID_LABEL) != runtime["Id"]
    ):
        reasons.append("the suite image was built on an older runtime image")
    why = "; ".join(reasons)
    outdated = installed is not None and bool(reasons)
    build = installed is None or rebuild or (outdated and update)
    # A new engine upgrades a version-1 database and refuses while a campaign needs the old image (store.py):
    # keep the installed image until those campaigns are finished or stopped, rather than replace it.
    blockers = upgrade_blockers(workspace) if build else 0
    if blockers:
        if installed is None or rebuild:
            raise ContractError(
                "workspace",
                f"{blockers} campaign(s) in this workspace are active, and this checkout's engine upgrades "
                "the operational database, which it refuses while they are; finish or stop them with the "
                "installed suite image first (start without --rebuild)",
                "conflict",
            )
        build = False
        print(
            f"Update pending ({why or 'rebuild'}); the new engine upgrades the operational database and "
            f"{blockers} campaign(s) are active, so the engine keeps its image until they are finished or "
            "stopped.",
            flush=True,
        )
    if build and running and running["State"]["Running"]:
        if rebuild:
            raise ContractError(
                "runtime",
                "Stop the owned engine before rebuilding; running campaigns keep their accepted runtime",
            )
        if installed is not None and engine_busy(socket):
            build = False
            print(
                f"Update pending ({why}); a campaign is running, so the engine keeps its image and updates "
                "at the next start while it is idle.",
                flush=True,
            )
        else:
            print("Stopping the idle engine to update its image…", flush=True)
            command("stop", "--time", "60", name, timeout=120)
            running = lookup("container", name)
    elif outdated and not update:
        print(
            f"Note: {why}; the images are rebuilt at the next start of the engine or GUI.",
            flush=True,
        )
    if build:
        if runtime_outdated:
            print(f"Refreshing runtime image version {runtime_version}…", flush=True)
            refresh_runtime(root, cancelled=cancelled, version=runtime_version)
        print(f"Building suite image version {version}…", flush=True)
        build_image(root, image, cancelled=cancelled, version=version)
        installed = lookup("image", image)
    assert installed is not None
    if (installed["Config"].get("Labels") or {}).get("org.benchmark.suite") != "1":
        raise ContractError(
            "image", "This image predates suite packaging; run ./suite_launch.sh --rebuild"
        )
    if running and (
        running["Image"] != installed["Id"]
        or running["HostConfig"]["NetworkMode"] != network_name
        or accepted_diagnostics != current_diagnostics
        or (running["Config"].get("Labels") or {}).get("org.benchmark.suite.gpu-mode") != gpu_mode
    ):
        if running["State"]["Running"]:
            raise ContractError(
                "runtime",
                "Stop this workspace's engine explicitly before replacing its image, network, GPU access or diagnostic settings",
            )
        command("rm", name)
        running = None
    if running:
        if not running["State"]["Running"]:
            command("start", name)
    else:
        # Provenance the engine cannot see from inside its container (provenance.document).
        installed_labels = installed["Config"].get("Labels") or {}
        # Suite images carry the runtime image's labels (inherited, or copied into a flattened base).
        runtime_labels = {
            key: value
            for key, value in installed_labels.items()
            if key.startswith(RUNTIME_LABEL_PREFIX) or key == RUNTIME_IMAGE_LABEL
        }
        runtime_image = dict(
            id=installed_labels.get(BASE_ID_LABEL)
            or installed_labels.get("org.benchmark.base-source"),
            labels=runtime_labels or None,
        )
        host = {fact: info.get(fact) for fact in HOST_FACTS}
        studies = root / "results/studies"
        studies.mkdir(parents=True, exist_ok=True)
        network = lookup("network", network_name)
        if network:
            labels = network.get("Labels") or {}
            if labels.get("org.benchmark.suite.workspace") != str(workspace) or labels.get(
                "org.benchmark.suite.root"
            ) != str(root):
                raise ContractError(
                    "runtime", "The workspace network belongs to another installation"
                )
        else:
            command(
                "network",
                "create",
                "--driver",
                "bridge",
                "--label",
                "org.benchmark.suite.workspace=" + str(workspace),
                "--label",
                "org.benchmark.suite.root=" + str(root),
                network_name,
            )
        args = [
            "run",
            "-d",
            "--name",
            name,
            "--network",
            network_name,
            "--init",
            "--restart",
            "unless-stopped",
            "--stop-timeout",
            "60",
            "--shm-size",
            "2g",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--security-opt",
            "no-new-privileges",
            "--label",
            "org.benchmark.suite.workspace=" + str(workspace),
            "--label",
            "org.benchmark.suite.root=" + str(root),
            "--label",
            "org.benchmark.suite.diagnostics=" + ("1" if accepted_diagnostics else "0"),
            "--label",
            "org.benchmark.suite.gpu-mode=" + gpu_mode,
            "--mount",
            bind_mount(root, readonly=True),
            "--mount",
            bind_mount(workspace),
            "--mount",
            bind_mount(socket.parent),
            # Study exports (RESULTS_RESEARCH RD-36), apart from the operational workspace.
            "--mount",
            bind_mount(studies),
            "--workdir",
            str(root),
            "--env",
            "BENCHMARK_IMAGE_ID=" + installed["Id"],
            "--env",
            "BENCHMARK_IMAGE_LABELS=" + json.dumps(installed_labels, sort_keys=True),
            "--env",
            "BENCHMARK_RUNTIME_IMAGE=" + json.dumps(runtime_image, sort_keys=True),
            "--env",
            "BENCHMARK_HOST=" + json.dumps(host, sort_keys=True),
            # Never recorded: study exports refuse files that contain it (exports.py).
            "--env",
            "BENCHMARK_HOST_NAME=" + gethostname(),
        ]
        devices = Path("/dev/dri")
        if devices.exists():
            args.extend(("--device", str(devices)))
            for group in sorted({path.stat().st_gid for path in devices.iterdir()}):
                args.extend(("--group-add", str(group)))
        args.extend(gpu_args)
        args.extend(
            (
                image,
                "--workspace",
                str(workspace),
                "--socket",
                str(socket),
                "engine",
                "--root",
                str(root),
            )
        )
        if not accepted_diagnostics:
            args.append("--without-diagnostics")
        command(*args)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        check_cancelled(cancelled)
        try:
            with_client = Client(socket, timeout=2)
            try:
                status = with_client.call("engine.status")
            finally:
                with_client.close()
            if Path(status["workspace"]).resolve() != workspace:
                raise ContractError("workspace", "The engine socket reports a different workspace")
            return dict(
                container=name,
                socket=str(socket),
                workspace=str(workspace),
                image=installed["Id"],
                diagnostics=accepted_diagnostics,
            )
        except (OSError, EOFError):
            time.sleep(0.1)
    raise ContractError("runtime", f"Engine did not become available; inspect docker logs {name}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="suite_launch.sh")
    parser.add_argument("--image", default=os.environ.get("BENCHMARK_SUITE_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--rebuild", action="store_true")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument(
        "--without-diagnostics",
        dest="diagnostics",
        action="store_false",
        help="Start the engine without optional diagnostic instrumentation",
    )
    diagnostics.add_argument("--with-diagnostics", dest="diagnostics", action="store_true")
    parser.set_defaults(diagnostics=None)
    parser.add_argument(
        "--headless", action="store_true", help="Start/reconnect the engine without opening the GUI"
    )
    parser.add_argument(
        "--stop-engine",
        action="store_true",
        help="Gracefully stop the engine; interrupted runs await Resume",
    )
    parser.add_argument(
        "--cli", nargs=argparse.REMAINDER, help="Run a headless suite command after connecting"
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    workspace = DEFAULT_WORKSPACE
    try:
        if arguments.stop_engine:
            name = f"benchmark-suite-{os.getuid()}-{workspace_key(workspace)}"
            owned = inspect("container", name)
            labels = (owned["Config"].get("Labels") or {}) if owned else {}
            if labels.get("org.benchmark.suite.workspace") == str(workspace) and labels.get(
                "org.benchmark.suite.root"
            ) == str(root):
                docker("stop", "--time", "60", name)
            return 0
        if arguments.cli and arguments.cli[0] == "verify-export":
            # Reads a local export only: no engine is needed.
            from .cli import main as cli

            return cli(["--workspace", str(workspace), *arguments.cli])
        runtime = ensure_engine(
            root,
            workspace,
            image=arguments.image,
            rebuild=arguments.rebuild,
            diagnostics=arguments.diagnostics,
            update=not (arguments.cli and arguments.cli[0] in READ_ONLY_COMMANDS),
        )
        print(
            f"Engine: {runtime['container']}\nWorkspace: {runtime['workspace']}\nSocket: {runtime['socket']}",
            flush=True,
        )
        if arguments.cli is not None:
            from .cli import main as cli

            return cli(
                ["--workspace", str(workspace), "--socket", runtime["socket"], *arguments.cli]
            )
        if arguments.headless:
            return 0
        from .gui import run

        return run(Path(runtime["socket"]))
    except (ContractError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), flush=True)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.strip(), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
