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
import subprocess
import time
from typing import Any, Callable, IO

from .contracts import ContractError
from .protocol import Client
from .storage import require_local_storage

DEFAULT_IMAGE = "autonomous-exploration-benchmark:suite"
BASE_IMAGE = "autonomous-exploration-benchmark:jazzy-harmonic"
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


def build_image(root: Path, image: str, *, cancelled: Callable[[], bool] | None = None) -> None:
    command = partial(docker, cancelled=cancelled)
    lookup = partial(inspect, cancelled=cancelled)
    base = os.environ.get("BENCHMARK_SUITE_BASE", BASE_IMAGE)
    original = lookup("image", base)
    if original is None:
        raise ContractError(
            "image",
            f"Build the existing robotics image with ./docker.sh before building the suite ({base} is missing)",
        )
    if len(original["RootFS"]["Layers"]) > 115:
        flattened = "autonomous-exploration-benchmark:suite-base"
        saved = lookup("image", flattened)
        if (
            saved is None
            or (saved["Config"].get("Labels") or {}).get("org.benchmark.base-source")
            != original["Id"]
        ):
            print("Preparing the robotics base for the suite image…", flush=True)
            container = command("create", base)
            try:
                changes = ["--change", "LABEL org.benchmark.base-source=" + original["Id"]]
                for entry in original["Config"].get("Env", []):
                    name, _, value = entry.partition("=")
                    changes.extend(("--change", "ENV " + name + "=" + json.dumps(value)))
                working = original["Config"].get("WorkingDir")
                if working:
                    changes.extend(("--change", "WORKDIR " + working))
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
                            raise OSError("Docker could not export the robotics base")
                        if not imported:
                            raise OSError("Docker did not return a suite-base image identity")
                    finally:
                        stop_command(export)
            finally:
                docker("rm", container, timeout=5)
        base = flattened
    command(
        "build",
        "-f",
        str(root / "docker/Dockerfile.suite"),
        "--build-arg",
        "BENCHMARK_BASE=" + base,
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
) -> dict[str, Any]:
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
    gpu_mode, gpu_args = gpu_arguments(json.loads(command("info", "--format", "{{json .}}")))
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
    if installed is None or rebuild:
        if running and running["State"]["Running"]:
            raise ContractError(
                "runtime",
                "Stop the owned engine before rebuilding; running campaigns keep their accepted runtime",
            )
        build_image(root, image, cancelled=cancelled)
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
            "--workdir",
            str(root),
            "--env",
            "BENCHMARK_IMAGE_ID=" + installed["Id"],
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
        runtime = ensure_engine(
            root,
            workspace,
            image=arguments.image,
            rebuild=arguments.rebuild,
            diagnostics=arguments.diagnostics,
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
