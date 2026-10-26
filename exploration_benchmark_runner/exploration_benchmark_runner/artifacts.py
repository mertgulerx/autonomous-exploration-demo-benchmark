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

"""Crash-readable, incremental resources; no full sensor bag by default."""

import csv
import hashlib
import json
import math
import os
import platform
import shutil
from pathlib import Path
import subprocess
import time
import yaml

from .coverage import (
    COVERAGE_STOP_DELAY_S,
    COVERAGE_STOP_PERCENT,
    COVERAGE_STOP_RULE,
    MEASURED_OUTCOMES,
    STUCK_COMPLETION_PERCENT,
    STUCK_GAIN_PP,
    STUCK_NO_GAIN_RADIUS_M,
    STUCK_NO_GAIN_WINDOW_S,
    STUCK_RADIUS_M,
    STUCK_RULE,
    STUCK_WINDOW_S,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze_interrupted_coverage(manifest, metrics, directory):
    """Reconcile a killed supervisor using its last telemetry and flushed curve."""
    manifest["coverage_valid_comparison"] = False
    coverage = manifest.get("coverage")
    if not coverage:
        return
    flags = set(coverage.get("quality_flags", []))
    flags.update(metrics.get("coverage_quality_flags", []))
    flags.add("coverage_supervisor_interrupted")
    milestones = metrics.get("coverage_milestones", coverage.get("milestones", []))
    for milestone in milestones:
        if not milestone["reached"]:
            milestone["simulation_elapsed"] = None
    metrics["coverage_quality_flags"] = sorted(flags)
    coverage.update(
        frozen=True,
        quality_flags=sorted(flags),
        milestones=milestones,
        final={
            key: value for key, value in metrics.items() if key.startswith("coverage_")
        },
    )
    curve = Path(directory) / "coverage.csv"
    if curve.is_file():
        with curve.open() as file:
            coverage["samples"] = sum(1 for _ in csv.DictReader(file))


def revision(root):
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return {
        "commit": result.stdout.strip(),
        "working_tree_status": status.stdout.splitlines(),
    }


class Recorder:
    def __init__(
        self, directory, configuration, profile, root, coverage_reference=None
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events = open(self.directory / "events.jsonl", "a", buffering=1)
        self.metrics = open(self.directory / "metrics.csv", "w", buffering=1)
        self.writer = csv.DictWriter(
            self.metrics,
            fieldnames=[
                "cpu_percent",
                "rss_mib",
                "cpu_average_percent",
                "rss_average_mib",
                "cpu_seconds",
                "distance_m",
                "simulation_elapsed",
                "motion_started",
                "pre_motion_wall_elapsed",
                "pre_motion_simulation_elapsed",
                "rtf",
                "stream_gaps",
                "quality_flags",
                "frozen",
                "coverage_percent",
                "coverage_covered_m2",
                "coverage_target_m2",
                "coverage_status",
                "coverage_map_stamp",
                "coverage_map_age",
                "coverage_quality_flags",
                "coverage_milestones",
            ],
        )
        self.writer.writeheader()
        self.manifest = dict(
            schema_version=6,  # 6: unmeasured resources are empty (null), not 0
            robot=configuration["robot"],
            configuration=configuration,
            profile=profile,
            source=revision(root),
            created_unix=time.time(),
            phase="STARTING",
            seed_control={
                "gazebo": configuration["seed"],
                "framework": profile.get("random_seed", "unsupported"),
                "ros2_stack": "No seed override applied; baseline SLAM/Nav2 configuration retained",
            },
            # The run's stop rule (coverage-stop-v1 from 25 October 2026, v2 and v3 with its delay from 26 October).
            coverage_stop_policy=dict(
                rule=COVERAGE_STOP_RULE,
                percent=COVERAGE_STOP_PERCENT,
                delay_seconds=COVERAGE_STOP_DELAY_S,
            ),
            # The stuck-robot stop (stuck-stop-v1, 26 October 2026).
            stuck_policy=dict(
                rule=STUCK_RULE,
                window_seconds=STUCK_WINDOW_S,
                radius_m=STUCK_RADIUS_M,
                no_gain_window_seconds=STUCK_NO_GAIN_WINDOW_S,
                no_gain_radius_m=STUCK_NO_GAIN_RADIUS_M,
                gain_threshold_pp=STUCK_GAIN_PP,
                completion_percent=STUCK_COMPLETION_PERCENT,
            ),
            quality_policy={
                "rtf_threshold": 0.95,
                "window_wall_seconds": 5,
                "persistence_wall_seconds": 10,
                "continue_on_degradation": True,
            },
            equivalence_proven=False,
            platform={
                "os": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "renderer": "ogre2",
                "headless": True,
                "ros_distro": os.environ.get("ROS_DISTRO"),
                "middleware": os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
                "discovery_range": os.environ.get(
                    "ROS_AUTOMATIC_DISCOVERY_RANGE", "SUBNET"
                ),
                "fastdds_transports": os.environ.get(
                    "FASTDDS_BUILTIN_TRANSPORTS", "DEFAULT"
                ),
            },
        )
        import xml.etree.ElementTree as ET
        from ament_index_python.packages import get_package_share_directory

        versions = {}
        for package in (
            "bme_ros2_navigation",
            "nav2_bringup",
            "slam_toolbox",
            "ros_gz_sim",
            profile["package"],
        ):
            share = Path(get_package_share_directory(package))
            versions[package] = (
                ET.parse(share / "package.xml").getroot().findtext("version")
            )
        self.manifest["package_versions"] = versions
        installed = subprocess.check_output(
            ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"], text=True
        )
        write_json(
            self.directory / "binary_versions.json",
            dict(
                line.split("\t", 1) for line in installed.splitlines() if "\t" in line
            ),
        )
        # The explorer's source: our submodule's revision, or an external explorer's pinned archive (profiles.yaml).
        source = profile.get("source", {"path": "exploration_packages/frontier_exploration_ros2"})
        if "path" in source:
            framework_path = Path(root) / source["path"]
            self.manifest["framework_source"] = dict(
                path=str(framework_path), **revision(framework_path)
            )
        else:
            self.manifest["framework_source"] = dict(source)
        result = subprocess.run(
            ["gz", "sim", "--versions"], capture_output=True, text=True, timeout=5
        )
        self.manifest["gazebo_versions"] = result.stdout.strip()
        world = configuration["scenario"]["world_file"]
        self.manifest["scenario_sha256"] = digest(world)
        baseline = Path(get_package_share_directory("bme_ros2_navigation"))
        common = self.directory / "common_configuration"
        common.mkdir(exist_ok=True)
        shutil.copyfile(world, common / Path(world).name)
        hashes = {}
        for pattern in ("config/*", "urdf/*"):
            for path in baseline.glob(pattern):
                if path.is_file():
                    relative = path.relative_to(baseline)
                    target = common / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    hashes[str(relative)] = digest(path)
        self.manifest["common_configuration_sha256"] = hashes
        self.coverage_file = open(self.directory / "coverage.csv", "w", buffering=1)
        self.coverage_writer = csv.DictWriter(
            self.coverage_file,
            fieldnames=[
                "stage",
                "simulation",
                "wall_monotonic",
                "simulation_elapsed",
                "map_received_wall",
                "map_received_unix",
                "coverage_percent",
                "coverage_covered_m2",
                "coverage_target_m2",
                "coverage_status",
                "coverage_map_stamp",
                "coverage_map_age",
            ],
        )
        self.coverage_writer.writeheader()
        self.coverage_samples = 0
        descriptor = configuration["scenario"].get("coverage_reference")
        if descriptor:
            from .coverage import Reference

            reference = coverage_reference or Reference(descriptor["metadata_file"])
            if reference.metadata_hash != descriptor["metadata_sha256"]:
                raise ValueError("Coverage reference changed after acceptance")
            destination = self.directory / "coverage_reference"
            reference.save(destination)
            self.manifest["coverage"] = dict(
                metric_version=reference.metric_version,
                reference=descriptor,
                definition="100 * current SLAM-free reference cells / fixed target cells",
                target_margin_m=reference.margin,
                free_value=0,
                sample_interval_simulation_seconds=1.0,
                stale_delivery_wall_seconds=5.0,
                alignment=None,
                baseline=None,
                milestones=[],
                quality_flags=[],
                samples=0,
                time_origin="first_raw_odometry_movement",
                monotonic=False,
                threshold_policy="first_valid_observed_crossing_no_interpolation",
            )
        self.flush()

    def flush(self):
        write_json(self.directory / "manifest.json", self.manifest)

    def event(self, event, **details):
        self.events.write(
            json.dumps(
                safe_json(dict(event=event, wall_unix=time.time(), **details)),
                allow_nan=False,
            )
            + "\n"
        )

    def sample(self, metrics):
        row = {
            k: json.dumps(v) if isinstance(v, list) else v for k, v in metrics.items()
        }
        self.writer.writerow(row)

    def coverage_sample(self, row):
        self.coverage_writer.writerow(
            {key: row[key] for key in self.coverage_writer.fieldnames}
        )
        self.coverage_samples += 1
        for milestone in row["milestones_crossed"]:
            self.event("coverage_milestone", **milestone)

    def coverage_checkpoint(self, tracker):
        if "coverage" not in self.manifest:
            return
        self.manifest["coverage"].update(
            baseline=tracker.baseline,
            milestones=tracker.milestones,
            quality_flags=sorted(tracker.flags),
            samples=self.coverage_samples,
            final=tracker.snapshot(),
            frozen=tracker.frozen,
        )

    def finish(
        self, outcome, detail, metrics, map_message, path_message, native_code, commands
    ):
        coverage = self.manifest.get("coverage")
        if coverage and not coverage.get("frozen", False):
            # Preparation can fail before Measurements/Coverage exist. Retain an
            # explicit unavailable, frozen result rather than a pending record.
            from .coverage import MILESTONES

            metrics.update(
                coverage_percent=float("nan"),
                coverage_covered_m2=None,
                coverage_target_m2=coverage["reference"]["target_area_m2"],
                coverage_status="unavailable",
                coverage_map_stamp=float("nan"),
                coverage_map_age=float("nan"),
                coverage_quality_flags=["coverage_measurements_unavailable"],
                coverage_milestones=[
                    dict(
                        percent=p,
                        reached=False,
                        simulation_elapsed=None,
                    )
                    for p in MILESTONES
                ],
            )
            coverage.update(
                frozen=True,
                milestones=metrics["coverage_milestones"],
                quality_flags=metrics["coverage_quality_flags"],
                final={k: v for k, v in metrics.items() if k.startswith("coverage_")},
            )
        self.manifest.update(
            phase="FINISHED",
            outcome=outcome,
            detail=detail,
            final_metrics=metrics,
            native_result_code=native_code,
            actual_commands=commands,
            finished_unix=time.time(),
            valid_comparison=not metrics.get("quality_flags")
            and outcome in MEASURED_OUTCOMES,
        )
        self.manifest["coverage_valid_comparison"] = bool(
            self.manifest["valid_comparison"]
            and metrics.get("motion_started")
            and metrics.get("coverage_status") == "valid"
            and not metrics.get("coverage_quality_flags")
        )
        self.save_visuals(map_message, path_message)
        self.event(
            "terminal", outcome=outcome, detail=detail, native_result_code=native_code
        )
        self.flush()
        self.metrics.close()
        self.coverage_file.close()
        self.events.close()

    def save_visuals(self, map_message, path_message):
        """Also used periodically so an agent crash leaves readable snapshots."""
        from rosidl_runtime_py.convert import message_to_ordereddict

        if map_message is not None:
            write_json(
                self.directory / "final_map.json", message_to_ordereddict(map_message)
            )
            width, height = map_message.info.width, map_message.info.height
            pixels = bytearray()
            for row in reversed(range(height)):
                for value in map_message.data[row * width : (row + 1) * width]:
                    pixels.append(
                        205 if value < 0 else round(254 * (100 - value) / 100)
                    )
            (self.directory / "final_map.pgm").write_bytes(
                f"P5\n{width} {height}\n255\n".encode() + pixels
            )
            pose = map_message.info.origin
            q = pose.orientation
            yaw = math.atan2(
                2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
            )
            (self.directory / "final_map.yaml").write_text(
                yaml.safe_dump(
                    dict(
                        image="final_map.pgm",
                        mode="trinary",
                        resolution=map_message.info.resolution,
                        origin=[pose.position.x, pose.position.y, yaw],
                        negate=0,
                        occupied_thresh=0.65,
                        free_thresh=0.196,
                    )
                )
            )
        if path_message is not None:
            write_json(
                self.directory / "measured_path.json",
                message_to_ordereddict(path_message),
            )
            with open(self.directory / "measured_path.csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(["simulation_stamp", "x", "y", "z"])
                for pose in path_message.poses:
                    writer.writerow(
                        [
                            pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9,
                            pose.pose.position.x,
                            pose.pose.position.y,
                            pose.pose.position.z,
                        ]
                    )
