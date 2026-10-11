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

"""Fetch pinned vendor workspaces and record the installed dependency closure."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import yaml


def inventory(packages, lookup):
    """Capture versions and installed resources, including model meshes and config."""
    result = {}
    for package in sorted(packages):
        root = Path(lookup(package))
        manifest = ET.parse(root / "package.xml").getroot()
        files = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(root)
                if (
                    relative.parts[0] in ("cmake", "environment")
                    or relative.name.startswith("local_setup")
                    or relative.name == "package.dsv"
                ):
                    continue  # Installation-prefix metadata is not robot physics.
                files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        result[package] = dict(
            version=manifest.findtext("version"),
            licenses=[e.text for e in manifest.findall("license")],
            resources_sha256=hashlib.sha256(
                json.dumps(files, sort_keys=True).encode()
            ).hexdigest(),
        )
    return result


def fetch(directory, family, resource):
    sources = yaml.safe_load((resource / "config/source_lock.yaml").read_text())[
        "sources"
    ]
    source_root = directory / family / "src"
    source_root.mkdir(parents=True, exist_ok=True)
    for name, specification in sources.items():
        if specification["family"] != family:
            continue
        destination = source_root / name
        revision = specification["revision"]
        if not destination.exists():
            subprocess.run(["git", "init", str(destination)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(destination),
                    "remote",
                    "add",
                    "origin",
                    specification["url"],
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(destination),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    revision,
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
                check=True,
            )
        actual = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise ValueError(
                f"{destination}: revision {actual} differs from lock {revision}"
            )
    return directory / family


def install_binaries(resource, verify=False):
    """Install the recorded binary closure; never silently substitute versions.

    Packages already at the pinned base-image version need no archive download.
    """
    expected = yaml.safe_load((resource / "config/binary_lock.yaml").read_text())[
        "packages"
    ]
    installed = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"], text=True
    )
    current = dict(
        line.split("\t", 1) for line in installed.splitlines() if "\t" in line
    )
    differing = {
        name: version
        for name, version in expected.items()
        if current.get(name) != version
    }
    if verify:
        if differing:
            raise ValueError(
                "Binary lock mismatch: "
                + ", ".join(
                    f"{p}={v} (installed {current.get(p, 'missing')})"
                    for p, v in differing.items()
                )
            )
    elif differing:
        subprocess.run(
            [
                "apt-get",
                "install",
                "-y",
                "--no-install-recommends",
                "--allow-downgrades",
            ]
            + [f"{name}={version}" for name, version in differing.items()],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=["fetch", "snapshot", "install-binaries", "verify-binaries"],
    )
    parser.add_argument("--family", choices=["rosbot", "ugv"], default="rosbot")
    parser.add_argument("--directory", type=Path, default=Path("robot_dependencies"))
    parser.add_argument("--resource-directory", type=Path)
    args = parser.parse_args()
    from ament_index_python.packages import get_package_share_directory

    resource = args.resource_directory or Path(
        get_package_share_directory("exploration_benchmark_robots")
    )
    if args.operation in ("install-binaries", "verify-binaries"):
        install_binaries(resource, verify=args.operation == "verify-binaries")
    elif args.operation == "fetch":
        print(fetch(args.directory.resolve(), args.family, resource))
    else:
        args.directory.mkdir(parents=True, exist_ok=True)
        data = subprocess.check_output(
            ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"], text=True
        )
        binaries = dict(
            line.split("\t", 1) for line in data.splitlines() if "\t" in line
        )
        (args.directory / "binary_versions.json").write_text(
            json.dumps(binaries, indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
