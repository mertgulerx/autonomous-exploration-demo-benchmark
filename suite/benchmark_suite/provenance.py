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

"""Code identity and run environment of an engine (RESULTS_OVERHAUL R1; D-08).

Provenance stays outside the comparison key: the study's homogeneity check compares it. The module uses the
standard library only, because ``docker.sh`` also runs it on the host to label the runtime image::

    python3 suite/benchmark_suite/provenance.py labels ROOT PREFIX [ROLE...]
"""

from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

# 2: the world registry and robot catalog hashes (I1, 24 October 2026)
PROVENANCE_VERSION = 2
CHECKOUT_ROLE = "checkout"
EXPLORER_ROLE = "exploration_packages/frontier_exploration_ros2"
WORLDS_ROLE = "world_collection"
REPOSITORY_ROLES = (CHECKOUT_ROLE, EXPLORER_ROLE, WORLDS_ROLE)
# The runtime image's own label (docker/Dockerfile), kept with its repository labels.
RUNTIME_IMAGE_LABEL = "org.benchmark.multi_robot"
# Label prefixes of the two images: the runtime image (explorer build) and the suite image built on it.
RUNTIME_LABEL_PREFIX = "org.benchmark.runtime.repository."
SUITE_LABEL_PREFIX = "org.benchmark.suite.repository."
# Facts the launcher copies from ``docker info`` into the engine (never the host name).
HOST_FACTS = (
    "Architecture",
    "CgroupDriver",
    "CgroupVersion",
    "DefaultRuntime",
    "KernelVersion",
    "MemTotal",
    "NCPU",
    "OSType",
    "OSVersion",
    "OperatingSystem",
    "ServerVersion",
)
MIDDLEWARE_VARIABLES = (
    "RMW_IMPLEMENTATION",
    "CYCLONEDDS_URI",
    "FASTDDS_BUILTIN_TRANSPORTS",
    "ROS_AUTOMATIC_DISCOVERY_RANGE",
    "ROS_LOCALHOST_ONLY",
    "ROS_DISTRO",
)
GIT_TIMEOUT_SECONDS = 30
# Checkout files the engine reads at start that decide which worlds and robots exist (I1): the world
# registry and the robot catalog. Hashed outside the comparison key; the study's homogeneity check compares.
REGISTRY_FILES = {
    "world_registry_sha256": "exploration_benchmark_runner/config/worlds.yaml",
    "robot_catalog_sha256": "exploration_benchmark_robots/config/robots.yaml",
}


def _git(path: Path, *arguments: str) -> bytes | None:
    """Output of a read-only git command, or None when it fails (no repository, no git)."""
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(path), *arguments],
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _file_digest(path: Path) -> bytes:
    """SHA-256 of a file read in chunks (untracked files can be large)."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return b"unreadable"
    return digest.digest()


def role_path(root: Path, role: str) -> Path:
    """The directory of a repository role: the checkout itself or a submodule by relative path."""
    return root if role == CHECKOUT_ROLE else root / role


def commit(path: Path) -> str | None:
    """The HEAD commit of a checkout, or None."""
    value = _git(path, "rev-parse", "HEAD")
    return value.decode().strip() if value is not None else None


def repository_state(path: Path) -> dict[str, Any]:
    """Commit, dirty flag and diff hash of one checkout (RO-8).

    Dirty means tracked changes or untracked files that are not ignored; submodule changes count in the
    parent. The diff hash covers the status, the binary diff against HEAD and every untracked file's path
    and content. A dirty submodule shows in the parent only as "modified", so its own state needs its own
    role (``repositories``): equal hashes of every role mean equal working trees on equal commits.
    """
    head = commit(path)
    status = _git(path, "status", "--porcelain=v1", "-z", "--untracked-files=normal")
    diff = _git(path, "diff", "--binary", "HEAD")
    untracked = _git(path, "ls-files", "-z", "--others", "--exclude-standard")
    if head is None or status is None or diff is None or untracked is None:
        return dict(commit=None, dirty=None, diff_sha256=None)
    digest = hashlib.sha256(status + b"\0" + diff)
    for name in sorted(item for item in untracked.split(b"\0") if item):
        digest.update(b"\0" + name + b"\0")
        digest.update(_file_digest(path / os.fsdecode(name)))
    return dict(
        commit=head,
        dirty=bool(status),
        diff_sha256=digest.hexdigest() if status else None,
    )


def repositories(root: Path, roles: Sequence[str] = REPOSITORY_ROLES) -> dict[str, dict[str, Any]]:
    """Repository states keyed by role: the checkout and its submodules by relative path."""
    return {role: repository_state(role_path(root, role)) for role in roles}


def image_labels(root: Path, prefix: str, roles: Sequence[str]) -> dict[str, str]:
    """Image labels naming the repository states an image is built from."""
    labels = {}
    for role, state in repositories(root, roles).items():
        name = prefix + role.replace("/", ".") + "."
        labels[name + "commit"] = state["commit"] or "unknown"
        labels[name + "dirty"] = (
            "unknown" if state["dirty"] is None else str(state["dirty"]).lower()
        )
        labels[name + "diff-sha256"] = state["diff_sha256"] or ""
    return labels


def _proc_field(path: Path, prefix: str) -> str | None:
    try:
        for line in path.read_text().splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def engine_host() -> dict[str, Any]:
    """What the engine sees of its host from inside the container."""
    uname = platform.uname()
    return dict(
        system=uname.system,
        release=uname.release,
        machine=uname.machine,
        cpu_model=_proc_field(Path("/proc/cpuinfo"), "model name"),
        cpu_count=os.cpu_count(),
        cpu_affinity=len(os.sched_getaffinity(0)),
        memory_total=_proc_field(Path("/proc/meminfo"), "MemTotal"),
        cgroup_cpu_max=_read(Path("/sys/fs/cgroup/cpu.max")),
        cgroup_memory_max=_read(Path("/sys/fs/cgroup/memory.max")),
    )


def middleware() -> dict[str, Any]:
    """ROS middleware settings the engine's workers inherit, with the DDS file's hash."""
    result: dict[str, Any] = {name: os.environ.get(name) for name in MIDDLEWARE_VARIABLES}
    uri = os.environ.get("CYCLONEDDS_URI") or ""
    location = uri.removeprefix("file://")
    if location and not location.lstrip().startswith("<"):
        try:
            result["cyclonedds_file_sha256"] = hashlib.sha256(
                Path(location).read_bytes()
            ).hexdigest()
        except OSError:
            result["cyclonedds_file_sha256"] = None
    return result


def _environment_json(name: str) -> Any:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return json.loads(value)
    except ValueError:
        return {"unparsed": value}


def registry_hashes(root: Path | None) -> dict[str, str | None]:
    """SHA-256 of the world registry and the robot catalog the engine reads; None when unreadable."""
    result: dict[str, str | None] = {}
    for name, relative in REGISTRY_FILES.items():
        try:
            result[name] = (
                hashlib.sha256((root / relative).read_bytes()).hexdigest() if root else None
            )
        except OSError:
            result[name] = None
    return result


def document(root: Path | None) -> dict[str, Any]:
    """The engine's provenance document; the launcher passes the image and host facts in.

    ``root`` is the checkout the engine reads; None for the synthetic backend, which has none.
    """
    return dict(
        provenance_version=PROVENANCE_VERSION,
        **registry_hashes(root),
        suite_image=dict(
            id=os.environ.get("BENCHMARK_IMAGE_ID"),
            labels=_environment_json("BENCHMARK_IMAGE_LABELS"),
        ),
        runtime_image=_environment_json("BENCHMARK_RUNTIME_IMAGE"),
        repositories=repositories(root) if root is not None else {},
        host=dict(docker=_environment_json("BENCHMARK_HOST"), engine=engine_host()),
        middleware=middleware(),
    )


def main(arguments: Sequence[str]) -> int:
    if len(arguments) < 3 or arguments[0] != "labels":
        print("Usage: provenance.py labels ROOT PREFIX [ROLE...]", file=sys.stderr)
        return 2
    roles = arguments[3:] or REPOSITORY_ROLES
    for name, value in image_labels(Path(arguments[1]), arguments[2], roles).items():
        print(f"{name}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
