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

"""Select vendor underlays per worker, including incompatible vendor revisions."""

import os
from pathlib import Path
import subprocess


def configure_transport():
    """Configure middleware while respecting an explicit operator choice."""
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")


def sdk_prefix(profile):
    root = (
        Path(os.environ.get("BENCHMARK_ROBOT_DEPENDENCIES", "/opt/vendor"))
        / profile["family"]
        / "install"
    )
    return str(root) if (root / "local_setup.bash").is_file() else ""


def package_lookup(profile, fallback):
    root = sdk_prefix(profile)

    def lookup(package):
        if root:
            for path in (
                Path(root) / "share" / package,
                Path(root) / package / "share" / package,
            ):
                if (path / "package.xml").is_file():
                    return str(path)
        return fallback(package)

    return lookup


def environment(profile, base=None):
    env = dict(os.environ if base is None else base)
    prefix = profile.get("sdk_prefix") or sdk_prefix(profile)
    if prefix:
        result = subprocess.run(
            [
                "/bin/bash",
                "-c",
                'source "$1" >/dev/null; env -0',
                "benchmark_vendor_underlay",
                str(Path(prefix) / "local_setup.bash"),
            ],
            env=env,
            capture_output=True,
            check=True,
            timeout=10,
        )
        env = dict(
            entry.split("=", 1)
            for entry in result.stdout.decode().split("\0")
            if "=" in entry
        )
    return env


def shared_environment(profile, base=None):
    """Keep vendor ABI overrides out of shared SLAM, Nav2 and exploration.

    Native controllers retain their SDK environment. Navigation reads resolved
    absolute parameter paths and only needs the benchmark and ROS underlays.
    """
    env = dict(os.environ if base is None else base)
    prefix = profile.get("sdk_prefix") or sdk_prefix(profile)
    if not prefix:
        return env
    prefix = Path(prefix).resolve()
    for key in (
        "AMENT_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PATH",
    ):
        if key in env:
            env[key] = ":".join(
                item
                for item in env[key].split(":")
                if item and not Path(item).resolve().is_relative_to(prefix)
            )
    return env
