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

"""Per-profile distribution pages of the campaign group reports."""

from typing import Any

import matplotlib.pyplot as plt


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
                f"SELECT count(*),quantile_cont({field},[0.25,0.5,0.75]) FROM reported WHERE comparison_key=? AND profile_id=? AND {eligible} AND {field} IS NOT NULL",
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
                f"SELECT count(*),count(first_{percent}_simulation_seconds),quantile_cont(first_{percent}_simulation_seconds,[0.25,0.5,0.75]) FROM reported WHERE comparison_key=? AND profile_id=? AND coverage_eligible",
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
