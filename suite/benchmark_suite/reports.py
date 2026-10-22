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

from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sqlite3
import shutil
import time
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.transforms import Affine2D
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


from .analysis import ANALYSIS_VERSION, DetectorSettings
from .contracts import (
    DATASET_VERSION,
    REPORT_POLICY_VERSION,
    REPORTED_OUTCOMES,
    STOPPED_OUTCOMES,
    content_hash,
    mapping,
    report_exclusion,
)
from .datasets import files, materialize_manifest, preview
from .datasets import observations as recorded
from .monitoring import (
    IN_COLLISION,
    SEGMENT_START,
    PathSettings,
    explored_pixels,
)
from .report_data import (
    MILESTONE_PERCENTS,
    OUTCOME_SCHEMA,
    SUMMARY_SCHEMA,
    parquet_rows,
    summarize,
)
from .run_map import (
    GROUND_TRUTH_COLUMNS,
    GroundTruthPath,
    explored_cells,
    ground_truth_path,
    load_reference,
)
from .storage import atomic_json, file_hash, sync_directory
from .factor_reports import profile_distributions

COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def display(value: float | None, units: str = "s") -> str:
    return f"{value:.2f} {units}" if value is not None else "unavailable"


def axis_style(axis: Any, title: str, xlabel: str, ylabel: str) -> None:
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)


def series(
    table: pa.Table, key: str, clock: str = "simulation_elapsed_ns", maximum_gap: float = 2.0
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    x = np.asarray(table[clock].to_pylist(), dtype=float) / 1e9
    y = np.asarray(table[key].to_pylist(), dtype=float)
    valid = np.asarray(table["valid"].to_pylist(), dtype=bool)
    if "status" in table.column_names:
        valid &= np.asarray(table["status"].to_pylist()) == "valid"
    y[~valid | ~np.isfinite(x)] = np.nan
    if len(x) > 1:
        breaks = (
            np.diff(table["presentation_segment"].to_numpy()) != 0
            if "presentation_segment" in table.column_names
            else np.diff(x) > maximum_gap
        )
        y[np.r_[False, breaks | (np.diff(x) <= 0)]] = np.nan
    return x, y


def corners(transform: Affine2D, extent: list[float]) -> np.ndarray[Any, Any]:
    width, height = extent[1], extent[3]
    return np.asarray(
        transform.transform(np.array([[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]]))
    )


def final_map(axis: Any, directory: Path, manifest: dict[str, Any]) -> None:
    path = directory / "final_map.npz"
    if not path.exists():
        axis.text(0.5, 0.5, "Final map unavailable", ha="center", transform=axis.transAxes)
        return
    with np.load(path, allow_pickle=False) as archive:
        grid = archive["occupancy"]
        metadata = json.loads(str(archive["metadata"]))
    image = np.where(grid < 0, 0.6, np.where(grid == 0, 1.0, 0.1))
    x, y, _ = metadata["origin"]
    qx, qy, qz, qw = metadata["orientation"]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    world = (manifest.get("coverage", {}).get("alignment") or {}).get("world_from_map")
    if world:
        wx, wy, angle = world
        x, y, yaw = (
            wx + math.cos(angle) * x - math.sin(angle) * y,
            wy + math.sin(angle) * x + math.cos(angle) * y,
            yaw + angle,
        )
    resolution = metadata["resolution"]
    extent = [0, grid.shape[1] * resolution, 0, grid.shape[0] * resolution]
    transform = Affine2D().rotate(yaw).translate(x, y)
    axis.imshow(
        image,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=1,
        transform=transform + axis.transData,
        interpolation="nearest",
        rasterized=True,
    )
    # imshow adds its untransformed extent to the data limits; keep only the placed map.
    axis.ignore_existing_data_limits = True
    axis.update_datalim(corners(transform, extent))
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="datalim")


def ground_truth_layer(
    axis: Any, physical: GroundTruthPath, directory: Path, manifest: dict[str, Any]
) -> None:
    """World reference, explored cells (coverage metric, final map) and the ground-truth path."""
    drawn: list[np.ndarray[Any, Any]] = []
    handles: list[Any] = []
    lines = list(physical.notes)
    reference = physical.reference
    if reference is not None:
        image = np.full(reference.collision.shape, 0.67)
        image[reference.target] = 0.96
        image[reference.collision] = 0.12
        height, width = reference.collision.shape
        extent = [0, width * reference.resolution, 0, height * reference.resolution]
        x, y, yaw = reference.origin
        transform = Affine2D().rotate(yaw).translate(x, y)
        axis.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap="gray",
            vmin=0,
            vmax=1,
            transform=transform + axis.transData,
            interpolation="nearest",
            rasterized=True,
        )
        drawn.append(corners(transform, extent))
        free, note = explored_cells(reference, directory, manifest)
        if free is None:
            lines.append(note or "Explored cells unavailable")
        else:
            overlay = np.zeros((*reference.collision.shape, 4))
            overlay[..., :3] = to_rgb(COLOURS[5])
            overlay[..., 3] = np.where(explored_pixels(reference.target, free) == 255, 0.35, 0.0)
            axis.imshow(
                overlay,
                origin="lower",
                extent=extent,
                transform=transform + axis.transData,
                interpolation="nearest",
                rasterized=True,
            )
            share = 100 * int(free.sum()) / reference.count
            handles.append(
                Patch(
                    color=COLOURS[5],
                    alpha=0.5,
                    label=f"Explored cells, final map ({share:.2f} % of reference cells)",
                )
            )
    path = physical.path
    if path is None:
        axis.text(0.5, 0.5, "Ground-truth path unavailable", ha="center", transform=axis.transAxes)
    else:
        page = path.page(0, path.settings.capacity)
        points = np.asarray([point[:2] for point in page["points"]], dtype=float).reshape(-1, 2)
        flags = np.asarray([point[2] for point in page["points"]], dtype=int)
        speeds = np.asarray(physical.speeds, dtype=float)
        unknown: list[Any] = []
        if len(points) > 1:
            keep = (flags[1:] & SEGMENT_START) == 0
            segments = np.stack((points[:-1], points[1:]), axis=1)[keep]
            values = speeds[1:][keep]
            finite = np.isfinite(values)
            if finite.any():
                layer = LineCollection(
                    segments[finite],
                    array=values[finite],
                    cmap="viridis",
                    linewidth=1.4,
                    rasterized=True,
                )
                axis.add_collection(layer)
                axis.figure.colorbar(
                    layer, ax=axis, label="Gazebo ground-truth speed (m/s)", fraction=0.046
                )
            unknown.extend(segments[~finite])
        if page["tail"] is not None and len(points):
            unknown.append(np.asarray([points[-1], page["tail"]]))
        if unknown:
            axis.add_collection(
                LineCollection(unknown, colors="#8F8D81", linewidth=1.4, rasterized=True)
            )
        if len(points):
            drawn.append(points)
        flagged = points[(flags & IN_COLLISION) != 0]
        if len(flagged):
            handles.append(
                axis.scatter(
                    flagged[:, 0],
                    flagged[:, 1],
                    marker="x",
                    color="#A63A2E",
                    s=18,
                    zorder=3,
                    label="Sample in an obstacle cell",
                )
            )
        counted = [f"{reason} {count}" for reason, count in page["breaks"].items() if count]
        if counted:
            lines.append("Breaks: " + ", ".join(counted))
        if page["collision_samples"]:
            lines.append(f"Samples in obstacle cells: {page['collision_samples']}")
        if page["outside_samples"]:
            lines.append(f"Samples outside the reference region: {page['outside_samples']}")
    if lines:
        axis.text(0.01, 0.99, "\n".join(lines), transform=axis.transAxes, fontsize=7, va="top")
    if handles:
        axis.legend(handles=handles, fontsize=7, loc="lower right")
    if drawn:
        # imshow adds its untransformed extent to the data limits; keep only what is drawn.
        axis.ignore_existing_data_limits = True
        axis.update_datalim(np.vstack(drawn))
        axis.autoscale_view()
    axis.set_aspect("equal", adjustable="datalim")


def path_plotting() -> dict[str, Any]:
    settings = PathSettings()
    return dict(
        source=(
            "Gazebo PosePublisher model pose composed with the recorded model_to_base offset; "
            "ground_truth stream in arrival order"
        ),
        vertex_gate_m=settings.gate_m,
        gap_seconds=settings.gap_seconds,
        jump_floor_m=settings.jump_floor_m,
        jump_speed_factor=settings.jump_speed_factor,
        background=(
            "coverage reference verified by hash; explored cells counted by the coverage metric "
            "(Calculator.free_cells) in the final map"
        ),
    )


def attempt_figures(directory: Path, destination: Path, summary: dict[str, Any]) -> list[Path]:
    manifest = materialize_manifest(directory)
    coverage, motion, truth = (
        preview(directory, kind) for kind in ("coverage", "trajectory", "ground_truth")
    )
    resources = preview(directory, "resources", scope="explorer")
    title = f"{summary['robot_id']} · {summary['world_id']} · {summary['profile_name']} · {summary['spawn_name']} · repetition {summary['repetition'] + 1}"
    pdf = destination / "report.pdf"
    with PdfPages(
        pdf, metadata={"Title": title, "Author": "Mert Güler", "Subject": ANALYSIS_VERSION}
    ) as pages:
        figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
        figure.suptitle(title, fontsize=13)
        x, y = series(coverage, "coverage_percent")
        axes[0, 0].plot(x, y, color=COLOURS[0], marker=".", markersize=2)
        axes[0, 0].set_ylim(0, 100)
        axis_style(
            axes[0, 0],
            "SLAM free-space coverage",
            "Simulation seconds from first movement",
            "Coverage (%)",
        )
        axes[0, 1].axis("off")
        exclusion = report_exclusion(summary["outcome"])
        details = [
            f"Outcome: {summary['outcome']}",
            "Campaign reports: "
            + (
                "not final"
                if not summary["is_final"]
                else "included"
                if exclusion is None
                else f"excluded ({exclusion})"
            ),
            f"Attempt: {summary['attempt_id']}",
            f"Timing eligible: {summary['timing_eligible']}  Coverage eligible: {summary['coverage_eligible']}",
        ]
        for label, key, unit in (
            ("Simulation elapsed", "simulation_elapsed_seconds", "s"),
            ("Distance", "distance_m", "m"),
            ("Average speed", "average_speed_mps", "m/s"),
            ("Average explorer CPU", "cpu_average_percent", "%"),
            ("Average explorer RSS", "rss_average_mib", "MiB"),
        ):
            value = summary[key]
            details.append(
                f"{label}: {value:.3f} {unit}" if value is not None else f"{label}: unavailable"
            )
        details.append("First reached (observed, simulation seconds):")
        for percent in MILESTONE_PERCENTS:
            sim = summary.get(f"first_{percent}_simulation_seconds")
            details.append(
                f"  {percent}%: {sim:.3f}"
                if sim is not None
                else f"  {percent}%: unreached / unavailable"
            )
        details.extend(
            [
                "Timing starts at first planar raw-odometry movement.",
                "Fixed seed repetitions measure execution variability.",
                "Missing samples are not interpolated.",
            ]
        )
        axes[0, 1].text(0, 1, "\n".join(details), va="top", fontsize=9)
        x, y = series(resources, "cpu_percent", maximum_gap=5)
        axes[1, 0].plot(x, y, color=COLOURS[1])
        axis_style(axes[1, 0], "Explorer CPU", "Simulation seconds", "CPU (%)")
        x, y = series(resources, "rss_mib", maximum_gap=5)
        axes[1, 1].plot(x, y, color=COLOURS[2])
        axis_style(axes[1, 1], "Explorer resident memory", "Simulation seconds", "RSS (MiB)")
        pages.savefig(figure, dpi=130)
        figure.savefig(destination / "preview.png", dpi=80)
        plt.close(figure)
        figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
        figure.suptitle(title, fontsize=12)
        x, speed = series(motion, "speed", maximum_gap=0.5)
        axes[0, 0].plot(x, speed, label="Raw odometry", color=COLOURS[0])
        if len(truth):
            tx, physical_speed = series(truth, "speed", maximum_gap=0.5)
            axes[0, 0].plot(
                tx, physical_speed, label="Gazebo diagnostic pose", color=COLOURS[1], alpha=0.7
            )
        axes[0, 0].legend(fontsize=8)
        axis_style(axes[0, 0], "Measured planar speed", "Simulation seconds", "Speed (m/s)")
        x, angular = series(motion, "wz", maximum_gap=0.5)
        axes[0, 1].plot(x, angular, color=COLOURS[2])
        axis_style(
            axes[0, 1], "Measured turning speed", "Simulation seconds", "Angular speed (rad/s)"
        )
        try:
            rows = recorded(directory, "ground_truth", GROUND_TRUTH_COLUMNS).to_pylist()
            physical = ground_truth_path(rows, manifest)
        except duckdb.Error as exc:
            reference, note = load_reference((manifest.get("coverage") or {}).get("reference"))
            unreadable = f"Ground-truth stream unreadable: {exc}"
            physical = GroundTruthPath(reference, None, (), (), (*filter(None, [note]), unreadable))
        ground_truth_layer(axes[1, 0], physical, directory, manifest)
        axis_style(
            axes[1, 0], "World reference and ground-truth path", "World X (m)", "World Y (m)"
        )
        x, rtf = series(resources, "rtf", maximum_gap=5)
        axes[1, 1].plot(x, rtf, color=COLOURS[4])
        axes[1, 1].axhline(0.9, linestyle="--", color="gray", label="Campaign acceptance")
        axes[1, 1].legend(fontsize=8)
        axis_style(axes[1, 1], "Real-time factor", "Simulation seconds", "RTF")
        pages.savefig(figure, dpi=130)
        plt.close(figure)
        figure, axes = plt.subplots(1, 2, figsize=(11.7, 8.3), constrained_layout=True)
        figure.suptitle(title, fontsize=12)
        final_map(axes[0], directory, manifest)
        incidents = pq.read_table(destination / "incidents.parquet").to_pylist()
        counts = Counter(row["type"] for row in incidents)
        for index, kind in enumerate(counts):
            points = [
                (row["x"], row["y"])
                for row in incidents
                if row["type"] == kind and row["trajectory_start_row"] != -1
            ]
            if points:
                xy = np.asarray(points)
                axes[0].scatter(
                    xy[:, 0],
                    xy[:, 1],
                    label=kind.replace("_", " "),
                    color=COLOURS[index % len(COLOURS)],
                    s=30,
                )
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(fontsize=8)
        axis_style(axes[0], "Observed incident locations", "World X (m)", "World Y (m)")
        axes[1].axis("off")
        detail = json.loads(summary["detail_json"])
        labels = [
            "Motion and navigation evidence",
            f"Incident counts: {dict(counts)}",
            "Motion source: raw odometry",
            f"Moving: {display(summary['moving_seconds'])}",
            f"Stationary: {display(summary['stationary_seconds'])}",
            f"Turning in place: {display(summary['turning_in_place_seconds'])}",
            f"Valid motion time: {display(summary['valid_motion_seconds'])}",
            f"Goals: {summary['goals']} · Preemptions: {summary['preemptions']}",
            f"Goal failures: {summary['failed_goals']} · Recoveries: {summary['recoveries']}",
            f"Motion gaps: {detail['motion_gaps']}",
            "Navigation UUID association remains explicit in raw data.",
            "Incidents describe observations, not goal correctness.",
            f"Analysis: {ANALYSIS_VERSION}",
            "Full detector settings and source identities: summary.json",
        ]
        axes[1].text(0, 0.9, "\n".join(labels), va="top", fontsize=10)
        pages.savefig(figure, dpi=130)
        plt.close(figure)
    with pdf.open("rb") as handle:
        os.fsync(handle.fileno())
    return [
        pdf,
        destination / "preview.png",
        destination / "summary.json",
        destination / "incidents.parquet",
    ]


def group_reports(
    destination: Path,
    database: sqlite3.Connection,
    campaign_id: str,
    specification: dict[str, Any],
    paths: dict[str, Path],
) -> list[Path]:
    summary_file = destination / "attempt_summaries.parquet"
    outcome_file = destination / "experiment_outcomes.parquet"
    outputs = []
    with duckdb.connect(
        config={
            "threads": 1,
            "memory_limit": "256MB",
            "temp_directory": str(destination / "query-staging"),
        }
    ) as query:
        summary = query.read_parquet(str(summary_file))
        summary.create_view("summaries")
        query.read_parquet(str(outcome_file)).create_view("outcomes")
        # Figures and statistics use reported final attempts only; exclusions are counted from the
        # experiment outcomes (REPORT_POLICY_VERSION).
        included = ",".join(f"'{outcome}'" for outcome in sorted(REPORTED_OUTCOMES))
        stopped = ",".join(f"'{outcome}'" for outcome in sorted(STOPPED_OUTCOMES))
        query.execute(
            "CREATE VIEW reported AS SELECT * FROM summaries "
            f"WHERE is_final AND outcome IN ({included})"
        )
        groups = query.execute(
            "SELECT robot_id,world_id,spawn_id,comparison_key,count(DISTINCT profile_id) FROM reported GROUP BY ALL ORDER BY robot_id,world_id,spawn_id,comparison_key"
        ).fetchall()
        for group_index, (robot, world, spawn, key, _) in enumerate(groups):
            pdf = destination / f"group-{group_index:05d}.pdf"
            selected = query.execute(
                "SELECT profile_id,profile_name,count(*),count(*) FILTER(WHERE outcome='completed'),count(*) FILTER(WHERE outcome='time_limit'),count(*) FILTER(WHERE outcome='no_movement'), count(*) FILTER(WHERE timing_eligible), median(completion_seconds) FILTER(WHERE timing_eligible), median(distance_m) FILTER(WHERE timing_eligible), median(cpu_average_percent) FILTER(WHERE timing_eligible), median(rss_average_mib) FILTER(WHERE timing_eligible) FROM reported WHERE comparison_key=? GROUP BY profile_id,profile_name ORDER BY profile_name",
                (key,),
            ).fetchall()
            excluded: dict[str, Counter[str]] = {}
            for profile_id, outcome, count in query.execute(
                "SELECT profile_id,outcome,count(*) FROM outcomes WHERE robot_id=? AND world_id=? AND spawn_id=? "
                f"AND final_attempt_id IS NOT NULL AND outcome NOT IN ({included}) GROUP BY ALL",
                (robot, world, spawn),
            ).fetchall():
                excluded.setdefault(profile_id, Counter())[str(report_exclusion(outcome))] += count
            with PdfPages(pdf) as pages:
                for first in range(0, len(selected), 6):
                    profiles = selected[first : first + 6]
                    figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
                    figure.suptitle(
                        f"{robot} · {world} · {spawn} · matching physical and measurement context",
                        fontsize=12,
                    )
                    bar_labels = []
                    for index, profile in enumerate(profiles):
                        identity, label = profile[:2]
                        args = (key, identity)
                        removed = excluded.get(identity, Counter())
                        bar_labels.append(
                            label
                            + (
                                f"\nexcluded: {removed['stopped']} stopped · {removed['failure']} failed"
                                if removed
                                else ""
                            )
                        )
                        attempt_ids = query.execute(
                            "SELECT attempt_id FROM reported WHERE comparison_key=? AND profile_id=? AND coverage_eligible",
                            args,
                        ).fetchall()
                        parquet = sorted(
                            {
                                str(path)
                                for (aid,) in attempt_ids
                                for path in files(paths[aid], "coverage")
                            }
                        )
                        if parquet:
                            # A campaign's attempts may span dataset schemas 9 and 10: read by name.
                            curve = query.execute(
                                "WITH valid AS (SELECT c.*,floor(c.simulation_elapsed_ns/1e9)::BIGINT AS second FROM read_parquet(?,union_by_name=true) c JOIN reported s USING(attempt_id) WHERE s.coverage_eligible AND s.comparison_key=? AND s.profile_id=? AND c.valid AND c.status='valid' AND c.simulation_elapsed_ns IS NOT NULL), observed AS (SELECT * FROM valid QUALIFY row_number() OVER(PARTITION BY attempt_id,second ORDER BY clock_ns DESC,receipt_monotonic_ns DESC)=1) SELECT second,quantile_cont(coverage_percent,0.25) AS q25,median(coverage_percent) AS median,quantile_cont(coverage_percent,0.75) AS q75,count(*) AS support FROM observed GROUP BY second ORDER BY second",
                                (parquet, *args),
                            ).fetchnumpy()
                            if len(curve["second"]):
                                x = curve["second"].astype(float)
                                middle = curve["median"].astype(float)
                                gaps = np.r_[False, np.diff(x) > 1]
                                middle[gaps] = np.nan
                                axes[0, 0].plot(x, middle, label=label, color=COLOURS[index])
                                lower, upper = (
                                    curve["q25"].astype(float),
                                    curve["q75"].astype(float),
                                )
                                lower[gaps], upper[gaps] = np.nan, np.nan
                                axes[0, 0].fill_between(
                                    x, lower, upper, color=COLOURS[index], alpha=0.12
                                )
                                axes[1, 0].step(
                                    x,
                                    curve["support"],
                                    where="post",
                                    label=label,
                                    color=COLOURS[index],
                                )
                        total, completed, limits, no_movement = profile[2:6]
                        axes[0, 1].bar(
                            index,
                            completed,
                            color=COLOURS[2],
                            label="Completed" if index == 0 else None,
                        )
                        axes[0, 1].bar(
                            index,
                            limits,
                            bottom=completed,
                            color=COLOURS[4],
                            label="Time limit" if index == 0 else None,
                        )
                        axes[0, 1].bar(
                            index,
                            no_movement,
                            bottom=completed + limits,
                            color=COLOURS[1],
                            label="No movement" if index == 0 else None,
                        )
                        milestone = query.execute(
                            "SELECT count(*), count(first_90_simulation_seconds), median(first_90_simulation_seconds), quantile_cont(first_90_simulation_seconds,0.25), quantile_cont(first_90_simulation_seconds,0.75) FROM reported WHERE comparison_key=? AND profile_id=? AND coverage_eligible",
                            args,
                        ).fetchone()
                        assert milestone is not None
                        if milestone[2] is not None:
                            axes[1, 1].errorbar(
                                index,
                                milestone[2],
                                yerr=[[milestone[2] - milestone[3]], [milestone[4] - milestone[2]]],
                                fmt="o",
                                color=COLOURS[index],
                            )
                        axes[1, 1].text(
                            index,
                            0.025,
                            f"{milestone[1]}/{milestone[0]}",
                            ha="center",
                            fontsize=8,
                            transform=axes[1, 1].get_xaxis_transform(),
                        )
                    axes[0, 0].set_ylim(0, 100)
                    # Exclusions are counted per robot, world, spawn and profile, not per context.
                    outcome_xlabel = "Profile" + (
                        " · excluded: all contexts at this robot, world and spawn"
                        if any(excluded.get(profile[0]) for profile in profiles)
                        else ""
                    )
                    for axis, title, xlabel, ylabel in (
                        (
                            axes[0, 0],
                            "Observed median coverage and IQR",
                            "Simulation-second observation bin",
                            "Coverage (%)",
                        ),
                        (
                            axes[1, 0],
                            "Observed sample support",
                            "Simulation-second observation bin",
                            "Eligible attempts",
                        ),
                        (axes[0, 1], "Reported final outcomes", outcome_xlabel, "Experiments"),
                        (
                            axes[1, 1],
                            "First reached 90%: attained / eligible",
                            "Profile",
                            "Simulation seconds",
                        ),
                    ):
                        axis_style(axis, title, xlabel, ylabel)
                    for axis in (axes[0, 0], axes[1, 0], axes[0, 1]):
                        if axis.get_legend_handles_labels()[0]:
                            axis.legend(fontsize=7)
                    for axis, names in (
                        (axes[0, 1], bar_labels),
                        (axes[1, 1], [row[1] for row in profiles]),
                    ):
                        axis.set_xticks(
                            range(len(profiles)), names, rotation=20, ha="right", fontsize=8
                        )
                    pages.savefig(figure, dpi=130)
                    plt.close(figure)
                    profile_distributions(pages, query, key, profiles)
                    incident_maps(pages, query, key, profiles, paths, destination)
            outputs.append(pdf)
        aggregation = query.execute(
            "SELECT robot_id,world_id,spawn_id,profile_id,comparison_key,count(*) AS final_attempts,count(*) FILTER(WHERE outcome='completed') AS completed,count(*) FILTER(WHERE timing_eligible) AS timing_eligible, median(completion_seconds) FILTER(WHERE timing_eligible) AS successful_completion_median_seconds,quantile_cont(completion_seconds,0.25) FILTER(WHERE timing_eligible) AS successful_completion_q25_seconds,quantile_cont(completion_seconds,0.75) FILTER(WHERE timing_eligible) AS successful_completion_q75_seconds,median(distance_m) FILTER(WHERE timing_eligible) AS distance_median_m,median(average_speed_mps) FILTER(WHERE timing_eligible) AS average_speed_median_mps,median(cpu_average_percent) FILTER(WHERE timing_eligible) AS cpu_median_percent,median(rss_average_mib) FILTER(WHERE timing_eligible) AS rss_median_mib,median(tail_90_to_95_seconds) FILTER(WHERE coverage_eligible) AS tail_90_95_median_seconds FROM reported GROUP BY ALL"
        ).fetch_arrow_table()
        policy = {
            b"report_policy": REPORT_POLICY_VERSION.encode(),
            b"included_outcomes": ",".join(sorted(REPORTED_OUTCOMES)).encode(),
        }
        pq.write_table(
            aggregation.replace_schema_metadata(
                {
                    b"schema_version": str(DATASET_VERSION).encode(),
                    b"kind": b"group_statistics",
                    b"analysis_version": ANALYSIS_VERSION.encode(),
                    **policy,
                }
            ),
            destination / "group_statistics.parquet",
            compression="zstd",
        )
        pairs = query.execute(
            "WITH candidates AS (SELECT comparison_key,profile_id FROM reported GROUP BY ALL HAVING bool_or(lower(profile_name)='default')), baselines AS (SELECT comparison_key,min(profile_id) AS profile_id FROM candidates GROUP BY comparison_key HAVING count(*)=1), base AS (SELECT s.* FROM reported s JOIN baselines b USING(comparison_key,profile_id) WHERE timing_eligible) SELECT s.experiment_id,s.profile_id,b.profile_id AS baseline_profile_id,s.block_id,s.comparison_key,s.completion_seconds-b.completion_seconds AS paired_successful_completion_difference_seconds,s.distance_m-b.distance_m AS paired_distance_difference_m,s.cpu_average_percent-b.cpu_average_percent AS paired_cpu_difference_percent FROM reported s JOIN base b ON s.block_id=b.block_id AND s.comparison_key=b.comparison_key WHERE s.timing_eligible AND s.profile_id<>b.profile_id"
        ).fetch_arrow_table()
        pq.write_table(
            pairs.replace_schema_metadata(
                {
                    b"schema_version": str(DATASET_VERSION).encode(),
                    b"kind": b"paired_block_differences",
                    **policy,
                }
            ),
            destination / "paired_differences.parquet",
            compression="zstd",
        )
        exclusions = query.execute(
            "SELECT robot_id,world_id,spawn_id,profile_id,outcome,"
            f"CASE WHEN outcome IN ({stopped}) THEN 'stopped' ELSE 'failure' END AS reason,count(*) AS count "
            f"FROM outcomes WHERE final_attempt_id IS NOT NULL AND outcome NOT IN ({included}) "
            "GROUP BY ALL ORDER BY ALL"
        ).fetch_arrow_table()
        pq.write_table(
            exclusions.replace_schema_metadata(
                {
                    b"schema_version": str(DATASET_VERSION).encode(),
                    b"kind": b"report_exclusions",
                    **policy,
                }
            ),
            destination / "report_exclusions.parquet",
            compression="zstd",
        )
        accounting = database.execute(
            "SELECT final_attempt_id IS NOT NULL,COALESCE(outcome,state),count(*) FROM experiments WHERE campaign_id=? GROUP BY 1,2 ORDER BY 1 DESC,2",
            (campaign_id,),
        ).fetchall()
        finals = {row[1]: row[2] for row in accounting if row[0]}
        reported_count = sum(
            n for outcome, n in finals.items() if report_exclusion(outcome) is None
        )
        stopped_count = sum(n for outcome, n in finals.items() if outcome in STOPPED_OUTCOMES)
        failures = {
            outcome: n for outcome, n in finals.items() if report_exclusion(outcome) == "failure"
        }
        labels = [
            row[1] + (" (excluded)" if row[0] and report_exclusion(row[1]) else "")
            for row in accounting
        ]
        values = [row[2] for row in accounting]
        figure, axes = plt.subplots(1, 2, figsize=(11.7, 8.3), constrained_layout=True)
        axes[0].barh(
            labels,
            values,
            color=[
                "#8F8D81" if label.endswith("(excluded)") else COLOURS[i % len(COLOURS)]
                for i, label in enumerate(labels)
            ],
        )
        axis_style(
            axes[0],
            "Campaign experimental accounting",
            "Planned experiments",
            "Outcome / unfinished state",
        )
        axes[1].axis("off")
        attempt_counts = dict(
            database.execute(
                "SELECT outcome,count(*) FROM attempts WHERE campaign_id=? GROUP BY outcome",
                (campaign_id,),
            ).fetchall()
        )
        lines = [
            specification["name"],
            f"Planned experiments: {sum(values):,}",
            f"Attempts by outcome: {attempt_counts}",
            f"Reported: {reported_count:,} final experiments (completed, time limit, no movement);",
            "  time limits and no-movement runs stay in denominators.",
            f"Excluded from figures and statistics: {stopped_count:,} stopped by the researcher,",
            f"  {sum(failures.values()):,} failures" + (f" {failures}." if failures else "."),
            f"Report policy: {REPORT_POLICY_VERSION}",
            "Timing-rejected attempts remain in the dataset.",
            "Successful completion times exclude failures and limits.",
            "Matched context and blocks are required for paired differences.",
            "No curve extrapolation or milestone interpolation.",
            "Spawn strata are kept separate; no unequal-stratum pooling.",
            "Fixed seed 1234 repetitions measure execution variability.",
            "No automated significance claims or winner ranking.",
        ]
        axes[1].text(0, 0.9, "\n".join(lines), va="top", wrap=True, fontsize=10)
        figure.savefig(destination / "campaign-preview.png", dpi=80)
        figure.savefig(destination / "campaign.pdf", dpi=130)
        plt.close(figure)
    outputs.extend(
        destination / name
        for name in (
            "group_statistics.parquet",
            "paired_differences.parquet",
            "campaign.pdf",
            "campaign-preview.png",
        )
    )
    return outputs


def incident_maps(
    pages: Any,
    query: Any,
    key: str,
    profiles: list[tuple[Any, ...]],
    paths: dict[str, Path],
    destination: Path,
) -> None:
    ids = [profile[0] for profile in profiles]
    attempts = query.execute(
        "SELECT attempt_id FROM reported WHERE comparison_key=? AND profile_id IN (SELECT unnest(?)) AND motion_eligible",
        (key, ids),
    ).fetchall()
    incident_files = [destination / "attempts" / aid / "incidents.parquet" for (aid,) in attempts]
    if not incident_files:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.7, 8.3), constrained_layout=True)
    figure.suptitle("Observed spatial incidents · matching context · profiles remain identified")
    reference = paths[attempts[0][0]]
    for axis in axes:
        final_map(axis, reference, materialize_manifest(reference))
    for index, (profile_id, label, *_) in enumerate(profiles):
        for axis, kinds, title in (
            (axes[0], ["revisit", "closed_loop"], "Revisits and closed-loop segments"),
            (
                axes[1],
                ["slow_motion", "stationary", "command_motion_discrepancy"],
                "Slow/stationary and command discrepancy intervals",
            ),
        ):
            observations = query.execute(
                "SELECT floor(i.x/0.5)*0.5 AS x,floor(i.y/0.5)*0.5 AS y,count(*) AS observations FROM read_parquet(?,union_by_name=true) i JOIN reported s USING(attempt_id) WHERE s.motion_eligible AND s.profile_id=? AND i.type IN (SELECT unnest(?)) AND i.x IS NOT NULL AND i.y IS NOT NULL GROUP BY ALL",
                ([str(path) for path in incident_files], profile_id, kinds),
            ).fetchall()
            if observations:
                points = np.asarray(observations)
                axis.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=15 + 15 * np.log1p(points[:, 2]),
                    label=label,
                    color=COLOURS[index],
                    alpha=0.6,
                    rasterized=True,
                )
            axis_style(axis, title, "World X (m)", "World Y (m)")
        if observations and axes[1].get_legend_handles_labels()[0]:
            axes[1].legend(fontsize=7)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7)
    figure.supxlabel(
        "0.5 m location bins; marker size shows incident observations, not occupancy or significance",
        fontsize=9,
    )
    pages.savefig(figure, dpi=130)
    plt.close(figure)


def generate(payload: dict[str, Any]) -> dict[str, Any]:
    destination = (
        Path(payload["workspace"]) / "reports" / payload["campaign_id"] / payload["report_id"]
    )
    if destination.exists():
        raise FileExistsError("Report output already exists: " + str(destination))
    try:
        return _generate(payload)
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise


def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    workspace, campaign_id, report_id = (
        Path(payload["workspace"]),
        payload["campaign_id"],
        payload["report_id"],
    )
    settings = DetectorSettings.accept(payload.get("detectors"))
    destination = workspace / "reports" / campaign_id / report_id
    destination.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    database = sqlite3.connect(f"file:{workspace / 'suite.sqlite3'}?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    artifacts_path = destination / "artifacts.jsonl"
    paths: dict[str, Path] = {}
    started = time.monotonic()
    with database, artifacts_path.open("x", encoding="utf-8") as artifacts:

        def index(path: Path, kind: str, attempt_id: str | None = None) -> None:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            artifacts.write(
                json.dumps(
                    dict(
                        path=str(path.relative_to(workspace)),
                        sha256=file_hash(path),
                        bytes=path.stat().st_size,
                        kind=kind,
                        attempt_id=attempt_id,
                    )
                )
                + "\n"
            )

        specification = json.loads(
            database.execute(
                "SELECT specification FROM campaigns WHERE id=?", (campaign_id,)
            ).fetchone()[0]
        )
        with parquet_rows(destination / "attempt_summaries.parquet", SUMMARY_SCHEMA) as writer:
            summaries = []
            attempts = database.execute(
                "SELECT a.* FROM attempts a WHERE a.campaign_id=? AND a.retained=1 AND a.discarded=0 ORDER BY a.id",
                (campaign_id,),
            )
            for attempt in attempts:
                experiment = dict(
                    database.execute(
                        "SELECT * FROM experiments WHERE id=?", (attempt["experiment_id"],)
                    ).fetchone()
                )
                directory = Path(attempt["directory"])
                paths[attempt["id"]] = directory
                target = destination / "attempts" / attempt["id"]
                target.mkdir(parents=True)
                summary = summarize(directory, dict(attempt), experiment, settings, target)
                summaries.append(summary)
                if len(summaries) >= 128:
                    writer.write_table(pa.Table.from_pylist(summaries, schema=SUMMARY_SCHEMA))
                    summaries.clear()
                for path in attempt_figures(directory, target, summary):
                    index(path, "attempt_" + path.stem, attempt["id"])
                atomic_json(
                    destination / "progress.json",
                    dict(
                        completed=len(paths),
                        current_attempt=attempt["id"],
                        elapsed_wall_seconds=time.monotonic() - started,
                    ),
                )
            if summaries:
                writer.write_table(pa.Table.from_pylist(summaries, schema=SUMMARY_SCHEMA))
        with parquet_rows(destination / "experiment_outcomes.parquet", OUTCOME_SCHEMA) as writer:
            cursor = database.execute(
                "SELECT campaign_id,id AS experiment_id,robot_id,world_id,profile_id,spawn_id,block_id,repetition,state,outcome,final_attempt_id FROM experiments WHERE campaign_id=? ORDER BY position",
                (campaign_id,),
            )
            while rows := cursor.fetchmany(1000):
                writer.write_table(
                    pa.Table.from_pylist([dict(row) for row in rows], schema=OUTCOME_SCHEMA)
                )
        finals = dict(
            database.execute(
                "SELECT outcome,count(*) FROM experiments WHERE campaign_id=? AND final_attempt_id IS NOT NULL GROUP BY outcome",
                (campaign_id,),
            ).fetchall()
        )
        for path in group_reports(destination, database, campaign_id, specification, paths):
            index(path, "campaign_" + path.stem)
        for name in (
            "attempt_summaries.parquet",
            "experiment_outcomes.parquet",
            "report_exclusions.parquet",
        ):
            index(destination / name, "analysis_table")
        artifacts.flush()
        os.fsync(artifacts.fileno())
    database.close()
    manifest = dict(
        schema_version=DATASET_VERSION,
        analysis_version=ANALYSIS_VERSION,
        campaign_id=campaign_id,
        report_id=report_id,
        selection={
            "campaign_id": campaign_id,
            "retained_attempts": len(paths),
            "experiment_specification_hash": content_hash(specification),
            "report_policy": {
                "version": REPORT_POLICY_VERSION,
                "included_outcomes": sorted(REPORTED_OUTCOMES),
                "reported_final_experiments": sum(
                    count for outcome, count in finals.items() if report_exclusion(outcome) is None
                ),
                "reported": {
                    outcome: count
                    for outcome, count in sorted(finals.items())
                    if report_exclusion(outcome) is None
                },
                "excluded": {
                    outcome: count
                    for outcome, count in sorted(finals.items())
                    if report_exclusion(outcome) is not None
                },
            },
        },
        detectors=asdict(settings),
        dataset_identities={"artifacts_index_sha256": file_hash(artifacts_path)},
        elapsed_wall_seconds=time.monotonic() - started,
        created_utc_ns=time.time_ns(),
        complete=True,
        bootstrap=None,
        plotting={
            "preview_maximum_observations": 20000,
            "coverage_bins": "floor of actual simulation seconds; latest valid observation per attempt/bin",
            "curves": "observed only; absent bins not filled",
            "support": "per bin; no extrapolation",
            "path": path_plotting(),
        },
    )
    atomic_json(destination / "manifest.json", manifest)
    sync_directory(destination)
    return mapping(
        dict(
            index=str(artifacts_path.relative_to(workspace)),
            sha256=file_hash(artifacts_path),
            manifest=str((destination / "manifest.json").relative_to(workspace)),
            manifest_sha256=file_hash(destination / "manifest.json"),
        ),
        "report",
    )
