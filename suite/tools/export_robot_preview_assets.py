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
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_robot(root: Path, output: Path, robot_id: str) -> None:
    from exploration_benchmark_robots.adapters import prepare
    from exploration_benchmark_robots.catalog import RobotCatalog
    from exploration_benchmark_robots.runtime import package_lookup
    from exploration_benchmark_runner.configuration import scenario

    catalog = RobotCatalog(root / "exploration_benchmark_robots")
    profile = catalog.require(robot_id, "warehouse")
    lookup = package_lookup(profile, catalog.lookup)
    with TemporaryDirectory(prefix=robot_id + "-", dir=output) as staging:
        generated, _ = prepare(profile, scenario(root, "warehouse"), Path(staging), lookup)
    robot = ET.fromstring(generated["robot_description"])
    appearance = json.loads(
        Path(__file__).with_name("robot_preview_appearance.json").read_text()
    ).get(robot_id, {})
    additional = [
        *appearance.get("mesh_overrides", {}).values(),
        *appearance.get("extra_visuals", []),
    ]
    packages = {}
    meshes = []
    native_meshes = list(robot.iter("mesh"))
    for mesh in [
        *native_meshes,
        *(ET.Element("mesh", filename=entry["uri"]) for entry in additional),
    ]:
        uri = mesh.get("filename", "")
        if uri.startswith("package://"):
            package, _, relative = uri.removeprefix("package://").partition("/")
            package_root = Path(lookup(package))
            source = package_root / relative
        elif uri.startswith("file://") or Path(uri).is_absolute():
            source = Path(unquote(urlparse(uri).path)) if uri.startswith("file://") else Path(uri)
            package_root = next(
                parent for parent in source.parents if (parent / "package.xml").is_file()
            )
            package = ET.parse(package_root / "package.xml").getroot().findtext("name")
        else:
            raise ValueError(f"Unsupported mesh URI: {uri}")
        if not source.is_file():
            raise FileNotFoundError(source)
        key = package + "-" + hashlib.sha256(str(package_root).encode()).hexdigest()[:8]
        destination = output / "packages" / key
        if not destination.exists():
            shutil.copytree(package_root, destination, symlinks=False)
        if key not in packages:
            metadata = ET.parse(package_root / "package.xml").getroot()
            notices = []
            debian = (
                Path("/usr/share/doc") / ("ros-jazzy-" + package.replace("_", "-")) / "copyright"
            )
            candidates = [debian]
            source_root = (package_root / "package.xml").resolve().parent
            if profile.get("sdk_prefix"):
                vendor = Path(profile["sdk_prefix"]).parent / "src"
                for candidate in vendor.rglob("package.xml"):
                    if ET.parse(candidate).getroot().findtext("name") == package:
                        source_root = candidate.parent
                        break
            for ancestor in [package_root, source_root, *source_root.parents[:3]]:
                for filename in ("LICENSE", "LICENSE.txt", "COPYING", "NOTICE"):
                    candidates.append(ancestor / filename)
            if package == "bme_ros2_navigation":
                candidates.append(root / "simulation/Week-7-8-ROS2-Navigation/LICENSE")
            for notice in candidates:
                if notice.is_file():
                    target = output / "notices" / (key + "-" + notice.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and digest(target) != digest(notice):
                        target = target.with_name(
                            target.stem + "-" + digest(notice)[:8] + target.suffix
                        )
                    shutil.copyfile(notice, target)
                    notices.append(str(target.relative_to(output)))
            packages[key] = {
                "name": package,
                "version": metadata.findtext("version"),
                "licenses": [item.text for item in metadata.findall("license")],
                "authors": [item.text for item in metadata.findall("author")],
                "urls": [item.text for item in metadata.findall("url")],
                "notices": sorted(set(notices)),
            }
        copied = destination / source.relative_to(package_root)
        meshes.append(
            {
                "uri": uri,
                "sha256": digest(source),
                "usage": "native_description" if mesh in native_meshes else "preview_visual",
            }
        )
        mesh.set("filename", str(copied.relative_to(output)))
    target = output / (robot_id + ".urdf")
    ET.indent(robot)
    ET.ElementTree(robot).write(target, encoding="utf-8", xml_declaration=True)
    (output / (robot_id + ".json")).write_text(
        json.dumps(
            {
                "robot_id": robot_id,
                "label": profile["label"],
                "description_sha256": hashlib.sha256(
                    generated["robot_description"].encode()
                ).hexdigest(),
                "source_revisions": profile["source_revisions"],
                "packages": packages,
                "meshes": meshes,
            },
            indent=2,
        )
        + "\n"
    )
    print(robot_id, len(meshes), "mesh assets", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export native robot visuals for offline preview rendering."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.robot:
        export_robot(args.root, args.output, args.robot)
        return
    from exploration_benchmark_robots.catalog import RobotCatalog
    from exploration_benchmark_robots.runtime import environment

    catalog = RobotCatalog(args.root / "exploration_benchmark_robots")
    for robot_id in catalog.profiles:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--root",
                str(args.root),
                "--output",
                str(args.output),
                "--robot",
                robot_id,
            ],
            env=environment(catalog.profile(robot_id)),
            check=True,
        )


if __name__ == "__main__":
    main()
