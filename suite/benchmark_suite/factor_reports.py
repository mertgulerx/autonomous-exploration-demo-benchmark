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

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import DATASET_VERSION, canonical, content_hash

METRICS = (
    ("first_90_simulation_seconds", "First reached 90% (s)", "coverage_eligible"),
    ("completion_seconds", "Successful completion (s)", "timing_eligible"),
    ("distance_m", "Distance (m)", "timing_eligible"),
    ("cpu_average_percent", "Explorer CPU (%)", "timing_eligible"),
)


def slices(specification: dict[str, Any]) -> list[dict[str, str | None]]:
    result = []
    for robot, profiles in specification["robots"].items():
        for profile in profiles:
            axes = profile.get("sweep", {}).get("values", {})
            selected = [(axis,) for axis in axes]
            if len(axes) >= 2:
                selected.append(tuple(list(axes)[:2]))
            for selection in selected:
                effective = dict(profile["effective"])
                physical = dict(movement_speed=0.8, turning_speed=1.0, lidar_range=10.0)
                physical.update(profile.get("robot_overrides", {}))
                for axis in selection:
                    (physical if axis.startswith("robot.") else effective).pop(
                        axis[6:] if axis.startswith("robot.") else axis, None
                    )
                axis1, axis2 = selection[0], selection[1] if len(selection) == 2 else None
                result.append(
                    dict(
                        profile_id=profile["id"],
                        robot_id=robot,
                        slice_key=content_hash(
                            dict(
                                robot=robot,
                                parent=profile.get("parent_id"),
                                axes=selection,
                                effective=effective,
                                physical=physical,
                            )
                        ),
                        axis1=axis1,
                        axis2=axis2,
                        value1=canonical(axes[axis1]),
                        value2=canonical(axes[axis2]) if axis2 else None,
                        study="physical factor"
                        if any(k.startswith("robot.") for k in selection)
                        else "algorithm parameter",
                    )
                )
    return result


def generate_sweeps(destination: Path, query: Any, specification: dict[str, Any]) -> list[Path]:
    selections = slices(specification)
    if not selections:
        return []
    schema = pa.schema([(key, pa.string()) for key in selections[0]])
    query.register("factors", pa.Table.from_pylist(selections, schema=schema))
    summaries = query.execute(
        "SELECT f.*,s.environment_key,s.world_id,s.spawn_id,count(*) AS experiments,count(*) FILTER(WHERE s.outcome='completed') AS completed,count(*) FILTER(WHERE s.outcome='time_limit') AS time_limits,count(*) FILTER(WHERE s.outcome NOT IN ('completed','time_limit')) AS failures,"
        + ",".join(
            f"count(s.{key}) FILTER(WHERE s.{eligible}) AS {key}_support,quantile_cont(s.{key},0.25) FILTER(WHERE s.{eligible}) AS {key}_q25,median(s.{key}) FILTER(WHERE s.{eligible}) AS {key}_median,quantile_cont(s.{key},0.75) FILTER(WHERE s.{eligible}) AS {key}_q75"
            for key, _, eligible in METRICS
        )
        + " FROM summaries s JOIN factors f USING(profile_id,robot_id) WHERE s.is_final GROUP BY ALL ORDER BY s.world_id,s.spawn_id,f.slice_key,f.value1,f.value2"
    ).fetch_arrow_table()
    table = destination / "sweep_statistics.parquet"
    pq.write_table(
        summaries.replace_schema_metadata(
            {
                b"schema_version": str(DATASET_VERSION).encode(),
                b"kind": b"sweep_statistics",
                b"selection": b"hold all nonswept explorer and physical factors fixed; separate world/spawn/environment strata",
            }
        ),
        table,
        compression="zstd",
    )
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries.to_pylist():
        groups[row["world_id"], row["spawn_id"], row["environment_key"], row["slice_key"]].append(
            row
        )
    pdf = destination / "parameter-sweeps.pdf"
    with PdfPages(pdf) as pages:
        for (world, spawn, _, _), rows in groups.items():
            if len({(r["value1"], r["value2"]) for r in rows}) < 2:
                continue
            first = rows[0]
            figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
            figure.suptitle(
                f"{first['robot_id']} · {world} · {spawn} · {first['study']} sweep\nOther parameters and physical settings held fixed",
                fontsize=11,
            )
            if first["axis2"]:
                labels1 = sorted({r["value1"] for r in rows}, key=value_order)
                labels2 = sorted({r["value2"] for r in rows}, key=value_order)
                for axis, (key, title, _) in zip(axes.flat, METRICS):
                    values = np.full((len(labels2), len(labels1)), np.nan)
                    for row in rows:
                        if row[key + "_median"] is not None:
                            y, x = labels2.index(row["value2"]), labels1.index(row["value1"])
                            values[y, x] = row[key + "_median"]
                            axis.text(
                                x,
                                y,
                                f"n={row[key + '_support']}",
                                ha="center",
                                fontsize=7,
                                color="white",
                                bbox=dict(facecolor="black", alpha=0.25, edgecolor="none", pad=1),
                            )
                    if np.isfinite(values).any():
                        image = axis.imshow(
                            values,
                            cmap="viridis",
                            origin="lower",
                            aspect="auto",
                        )
                        figure.colorbar(image, ax=axis, label=title)
                    else:
                        axis.text(
                            0.5,
                            0.5,
                            "No eligible observed measurements",
                            ha="center",
                            transform=axis.transAxes,
                        )
                    axis.set(title=title, xlabel=first["axis1"], ylabel=first["axis2"])
                    axis.set_xticks(
                        range(len(labels1)), labels1, rotation=25, ha="right", fontsize=7
                    )
                    axis.set_yticks(range(len(labels2)), labels2, fontsize=7)
                    axis.set_xlim(-0.5, len(labels1) - 0.5)
                    axis.set_ylim(-0.5, len(labels2) - 0.5)
            else:
                rows.sort(key=lambda r: value_order(r["value1"]))
                labels = [r["value1"] for r in rows]
                for axis, (key, title, _) in zip(axes.flat, METRICS):
                    for index, row in enumerate(rows):
                        median = row[key + "_median"]
                        if median is not None:
                            axis.errorbar(
                                index,
                                median,
                                yerr=[[median - row[key + "_q25"]], [row[key + "_q75"] - median]],
                                fmt="o",
                                color="#0072B2",
                            )
                        axis.text(
                            index,
                            0,
                            f"n={row[key + '_support']}/{row['experiments']}",
                            ha="center",
                            fontsize=7,
                            transform=axis.get_xaxis_transform(),
                        )
                    axis.set(title=title + " · median / IQR", xlabel=first["axis1"], ylabel=title)
                    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=7)
                    axis.grid(alpha=0.15)
                    axis.set_xlim(-0.5, len(labels) - 0.5)
            pages.savefig(figure, dpi=130)
            plt.close(figure)
    return [table, pdf] if pdf.exists() else [table]


def value_order(value: str) -> tuple[int, Any]:
    import json

    decoded = json.loads(value)
    return (0, decoded) if type(decoded) in (int, float) else (1, value)


def profile_distributions(
    pages: Any, query: Any, key: str, profiles: list[tuple[Any, ...]]
) -> None:
    ids, labels = [p[0] for p in profiles], [p[1] for p in profiles]
    measures = (
        ("completion_seconds", "Successful-only completion (s)", "timing_eligible"),
        ("rss_average_mib", "Explorer mean RSS (MiB)", "timing_eligible"),
        ("tail_90_to_95_seconds", "Observed 90% → 95% tail (s)", "coverage_eligible"),
        ("stationary_seconds", "Stationary duration (s)", "motion_eligible"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
    figure.suptitle("Repetition distributions · matching robot, world, spawn and physical context")
    for axis, (field, label, eligible) in zip(axes.flat, measures):
        for index, identity in enumerate(ids):
            result = query.execute(
                f"SELECT count(*),quantile_cont({field},[0.25,0.5,0.75]) FROM summaries WHERE is_final AND comparison_key=? AND profile_id=? AND {eligible} AND {field} IS NOT NULL",
                (key, identity),
            ).fetchone()
            assert result is not None
            count, quartiles = result
            if quartiles:
                low, median, high = quartiles
                axis.errorbar(
                    index, median, yerr=[[median - low], [high - median]], fmt="o", color="#0072B2"
                )
            axis.text(
                index,
                0.025,
                f"n={count}",
                ha="center",
                fontsize=8,
                transform=axis.get_xaxis_transform(),
            )
        axis.set(title=label + " · median / IQR", xlabel="Profile", ylabel=label)
        axis.set_xticks(range(len(ids)), labels, rotation=20, ha="right", fontsize=8)
        axis.grid(alpha=0.15)
        axis.set_xlim(-0.5, len(ids) - 0.5)
    pages.savefig(figure, dpi=130)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), constrained_layout=True)
    figure.suptitle("First reached milestones · attained / eligible · observed simulation time")
    for axis, percent in zip(axes.flat, (50, 75, 90, 95)):
        for index, identity in enumerate(ids):
            count, reached, times = query.execute(
                f"SELECT count(*),count(first_{percent}_simulation_seconds),quantile_cont(first_{percent}_simulation_seconds,[0.25,0.5,0.75]) FROM summaries WHERE is_final AND comparison_key=? AND profile_id=? AND coverage_eligible",
                (key, identity),
            ).fetchone()
            if times:
                low, median, high = times
                axis.errorbar(
                    index, median, yerr=[[median - low], [high - median]], fmt="o", color="#009E73"
                )
            axis.text(
                index,
                0.025,
                f"{reached}/{count}",
                ha="center",
                fontsize=8,
                transform=axis.get_xaxis_transform(),
            )
        axis.set(title=f"First reached {percent}%", xlabel="Profile", ylabel="Simulation seconds")
        axis.set_xticks(range(len(ids)), labels, rotation=20, ha="right", fontsize=8)
        axis.grid(alpha=0.15)
        axis.set_xlim(-0.5, len(ids) - 0.5)
    pages.savefig(figure, dpi=130)
    plt.close(figure)
