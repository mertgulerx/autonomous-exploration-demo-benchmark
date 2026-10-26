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

"""Figures F18 and F19: how the explorer decides (docs/plans/FIGURE_EXTENSION.md, FX3, FX4; FIGURE_STYLE §3, §6,
§8; METHODOLOGY §2.4).

- F18: decision timing (FX-16). One panel per stratum (a figure per world group and page, ``figures.grouped``,
  plans/worlds/I8 §3.1); every row of ``explorer_decisions`` of the sequential
  reported runs (``concurrent_wall_fraction`` = 0) with both timings and a candidate count as a point: wall time
  ``search_ms + ordering_ms`` on a log axis against the candidate count. Fill shows the order: filled when the
  row's ``order_cache_misses`` rose since the run's previous row (recomputed), open when it did not (from the
  cache); the explorer's built-in second order call is a cache hit by construction and never raises the miss
  counter. A frontier search ran when ``search_ms`` > 0 (0 on a snapshot cache hit), counted in the caption. A
  per-bin median with its IQR (type 2) per profile where a bin holds at least ``min_n`` rows; rows at 0 ms sit at
  the axis floor as triangles and are counted. Without sequential runs, with only explorer log format 1 (no
  timings) or without explorer-log rows the figure says which. Achieved RTF and hardware (T2's sources,
  ``provenance.json``) in the caption.
- F19: decision anatomy (FX-17). The Default run the plan's representative rule picks in the first stratum and its
  tick by SEL-4 (``event_tick``): the first tick at or after ``after_s`` where the route's first target differs from
  the greedy first, otherwise the first one there with at least ``min_candidates`` candidates. (a) On the reference
  map greys, the reference floor explored by the latest ``explored_cells`` snapshot at or before the tick, the
  ground-truth path so far, and the insight re-analysis of that tick: candidates coloured by start cost (cividis,
  dark casing, colour bar over the tick's finite values, non-finite costs hollow grey), the DP route numbered in
  the route colour, the target as a star and the greedy first as an open diamond. (b) The explorer's gate
  evaluations of the goal active at the tick; without one, of the next goal dispatched within one insight tick
  after it (the re-analysed target relates to the next dispatch), else of the latest goal dispatched before it,
  else of the next goal after it, each relation named. The caption gives the disagreement base rate over the
  run's ticks and checks the insight target against the dispatch point of a comparable goal (the active one or
  that next dispatch); with any other goal it says "no comparable goal", never a mismatch.

Values these figures compute (binned medians, base rates, distances) are descriptive display quantities, written to
the sidecar and never cited. Every figure names the export files it reads in ``FigureOut.inputs``.
"""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from matplotlib import patheffects
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np

from . import stats
from .data import Row, Stratum, is_external, run_end
from .figures import (
    GREY,
    TIME_LABEL,
    Extent,
    FigureContext,
    FigureOut,
    MapCode,
    draw_reference,
    finish,
    grid_fits,
    grouped,
    ground_truth,
    legend,
    load_reference,
    PROVENANCE,
    message_figure,
    panel_message,
    panel_title,
    reference_files,
    reference_name,
    representatives,
    stratum_grid,
)
from .streams import (
    Reads,
    ReferenceCells,
    explored_snapshots,
    log_formats,
    reference_cells,
    to_world,
    world_from_map,
)
from .style import EDGE, MAP_UNKNOWN, ROUTE_COLOUR, SEQUENTIAL_CMAP, figure

Alignment = tuple[float, float, float]

# Lower edges of the candidate-count bins of F18's binned median (wider where decisions get rarer).
BIN_EDGES = (0, 1, 2, 3, 5, 7, 10, 13, 17, 25, 33, 49, 65)
POINT_SIZE = 7.0  # F18 points
ZERO_SIZE = 10.0  # F18 triangles for rows at 0 ms
CANDIDATE_SIZE = 16.0  # one size for every candidate: never sized by gain (FX-12)
TARGET_SIZE = 90.0  # the route's first candidate (star)
GREEDY_SIZE = 70.0  # the greedy first (open diamond around the candidate)
GATE_SIZE = 14.0  # gate evaluations in panel (b)
MATCH_M = 0.25  # insight target and goal dispatch point closer than this (5 cells at 0.05 m) count as one point
ROBOT_ARROW_M = 0.6
MAP_SHARE = 0.6  # F19: share of the width for panel (a)
ANATOMY_OVERHEAD = 0.14  # F19: height for titles, the legend and the label strip, per unit of width
ANATOMY_MAX_ASPECT = 1.1  # F19: below the IEEE page height at double width
GATE_ASPECT = 0.5  # F19: height of panel (b) and its legend per unit of width, at most the figure's
INSIGHT_TICK_S = 1.0  # the insight node analyses at 1 Hz simulation time
MESSAGE_CHARS = 60  # line length of a not-drawn message in its one-column figure (8 pt)


def _not_drawn(
    ctx: FigureContext,
    name: str,
    reason: str,
    extra: dict[str, Any],
    inputs: set[str],
    attempts: set[str] | None = None,
) -> FigureOut:
    """A figure that is only its message (``figures.message_figure``), the message also its alt text and
    caption; ``attempts`` the runs it read before it stopped."""
    built = message_figure(
        ctx,
        name,
        f"Not drawn: {reason}.",
        attempts=attempts or (),
        inputs=inputs,
        wrap=MESSAGE_CHARS,
    )
    return replace(built, alt_text=f"{name} not drawn: {reason}.", extra=extra)


# F18 decision timing -------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decision:
    """One ``explorer_decisions`` row: candidate count and wall time (None when not logged), whether the order was
    recomputed (None without the counter) and whether a frontier search ran (None without ``search_ms``)."""

    attempt_id: str
    profile_id: str
    candidates: int | None
    time_ms: float | None
    order_recomputed: bool | None
    searched: bool | None

    @property
    def drawn(self) -> bool:
        return self.time_ms is not None and self.candidates is not None


def order_recomputation(rows: list[Row]) -> list[bool | None]:
    """Per row of one run in log order: True when ``order_cache_misses`` rose since the previous row (cumulative from
    the session start, so the first row compares with 0; a value below the previous one means a new session and
    compares with 0 again). The second order call of every decision is a hit by construction (COST_MODEL.md,
    "Cache") and raises only the hit counter. A miss on a path that writes no row (no-frontier or suppressed
    returns, escape) carries over to the next row."""
    previous = 0
    found: list[bool | None] = []
    for row in rows:
        value = row["order_cache_misses"]
        if value is None:
            found.append(None)
            continue
        found.append(value > (previous if value >= previous else 0))
        previous = value
    return found


def _decisions(reads: Reads, runs: list[Row]) -> list[Decision]:
    """The decision rows of the given runs, each run's rows in log order (``utc_ns``)."""
    wanted = {r["attempt_id"]: r["profile_id"] for r in runs}
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in reads.rows("explorer_decisions"):
        if row["attempt_id"] in wanted:
            grouped[row["attempt_id"]].append(row)
    found: list[Decision] = []
    for attempt in sorted(grouped):
        rows = sorted(grouped[attempt], key=lambda r: (r["utc_ns"] is None, r["utc_ns"] or 0))
        for row, rose in zip(rows, order_recomputation(rows), strict=True):
            search, ordering = row["search_ms"], row["ordering_ms"]
            time = None if search is None or ordering is None else float(search) + float(ordering)
            candidates = None if row["candidates"] is None else int(row["candidates"])
            searched = None if search is None else float(search) > 0
            found.append(Decision(attempt, wanted[attempt], candidates, time, rose, searched))
    return found


@dataclass(frozen=True, slots=True)
class Bin:
    """One candidate-count bin of F18: the smallest and largest count it holds, time quartiles (type 2, rows at
    0 ms included) and its number of rows."""

    first: int
    last: int
    quartiles: tuple[float, float, float]
    n: int


def _bins(timed: list[tuple[int, float]]) -> list[Bin]:
    """The bins of drawn rows given as (candidates, time in ms)."""
    found = []
    for low, high in zip(BIN_EDGES, (*BIN_EDGES[1:], math.inf), strict=True):
        members = [(c, t) for c, t in timed if low <= c < high]
        if members:
            counts = [c for c, _ in members]
            quartiles = stats.quartiles([t for _, t in members])
            found.append(Bin(min(counts), max(counts), quartiles, len(members)))
    return found


def _hardware(ctx: FigureContext, runs: list[Row]) -> str:
    """CPU model, cores and memory of the hosts that ran the given runs, as T2 prints them (``provenance.json``)."""
    documents = ctx.study.export.provenance.get("documents") or {}
    found: set[str] = set()
    for run in runs:
        engine = ((documents.get(run["provenance_id"]) or {}).get("host") or {}).get("engine") or {}
        parts = (
            str(engine.get("cpu_model") or ""),
            f"{engine['cpu_count']} cores" if engine.get("cpu_count") else "",
            f"memory {engine['memory_total']}" if engine.get("memory_total") else "",
        )
        found.add(", ".join(p for p in parts if p))
    return "; ".join(sorted(p for p in found if p)) or "not recorded"


def _rtf(runs: list[Row]) -> tuple[str, dict[str, Any]]:
    means = sorted(float(r["rtf_mean"]) for r in runs if r["rtf_mean"] is not None)
    worst = [float(r["rtf_min_5s"]) for r in runs if r["rtf_min_5s"] is not None]
    low = sum(1 for r in runs if r["low_rtf"])
    if not means:
        return "achieved RTF not recorded", {}
    median = stats.quartiles(means)[1]
    text = (
        f"achieved RTF (run means) median {median:.3f}, range {means[0]:.3f}–{means[-1]:.3f}"
        + (f", worst 5-s window {min(worst):.3f}" if worst else "")
        + f", {low} run(s) flagged low_rtf"
    )
    return text, {
        "rtf_mean_median": median,
        "rtf_mean_min": means[0],
        "rtf_mean_max": means[-1],
        "rtf_min_5s": min(worst) if worst else None,
        "low_rtf_runs": low,
    }


def _timing_panel(
    ax: Any,
    ctx: FigureContext,
    points: list[Decision],
    profiles: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """F18's drawn points and per-bin medians on one panel; returns the panel's descriptive values."""
    drawn: dict[str, Any] = {}
    timed = [p for p in points if p.drawn]
    positive = [p.time_ms for p in timed if p.time_ms is not None and p.time_ms > 0]
    low = min(positive, default=1.0) / 1.8
    high = max(positive, default=10.0) * 1.8
    for profile in profiles:
        style = ctx.styles[profile]
        mine = [p for p in timed if p.profile_id == profile]
        if not mine:
            continue
        for state, face in ((True, style.color), (False, "none"), (None, "none")):
            group = [p for p in mine if p.order_recomputed is state and (p.time_ms or 0) > 0]
            if not group:
                continue
            ax.scatter(
                [float(p.candidates or 0) for p in group] + rng.uniform(-0.3, 0.3, len(group)),
                [p.time_ms for p in group],
                s=POINT_SIZE,
                marker=style.marker,
                facecolors=face,
                edgecolors=style.color if state is not None else GREY,
                linewidths=0.5,
                alpha=0.35 if state else 0.6,  # open marks are thin: less transparent
                rasterized=True,
                zorder=2,
            )
        zeros = [p for p in mine if p.time_ms == 0]
        if zeros:
            ax.scatter(
                [float(p.candidates or 0) for p in zeros],
                [low] * len(zeros),
                s=ZERO_SIZE,
                marker="v",
                color=style.color,
                clip_on=False,
                zorder=4,
            )
        bins = _bins([(p.candidates or 0, p.time_ms or 0.0) for p in mine])
        # each bin on its own span of counts: nothing is drawn between bins (no interpolation)
        for item in (b for b in bins if b.n >= ctx.settings.min_n):
            span = (item.first - 0.45, item.last + 0.45)
            q1, median, q3 = (max(q, low) for q in item.quartiles)
            ax.fill_between(span, q1, q3, color=style.color, alpha=0.15, lw=0, zorder=1)
            ax.plot(span, (median, median), color=style.color, ls=style.linestyle, lw=1.2, zorder=3)
        drawn[style.name] = {
            "decisions": len(mine),
            "runs": len({p.attempt_id for p in mine}),
            "order_recomputed": sum(1 for p in mine if p.order_recomputed is True),
            "order_cached": sum(1 for p in mine if p.order_recomputed is False),
            "order_unknown": sum(1 for p in mine if p.order_recomputed is None),
            "frontier_searched": sum(1 for p in mine if p.searched is True),
            "zero_ms": len(zeros),
            "bins": [
                {
                    "candidates": [b.first, b.last],
                    "q1_ms": b.quartiles[0],
                    "median_ms": b.quartiles[1],
                    "q3_ms": b.quartiles[2],
                    "n": b.n,
                }
                for b in bins
            ],
        }
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    top = max((p.candidates or 0 for p in timed), default=1)
    ax.set_xlim(-0.6, top + 0.8)
    ax.set_xlabel("Frontier candidates")
    ax.set_ylabel("Search + ordering (wall ms)")
    zeros_here = sum(1 for p in timed if p.time_ms == 0)
    if zeros_here:
        ax.text(
            0.99,
            0.02,
            f"0 ms (triangles at the floor): {zeros_here}",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
        )
    return drawn


def f18(ctx: FigureContext) -> FigureOut:
    reads = Reads(ctx.study.export)
    formats = log_formats(reads)
    reported = [r for s in ctx.study.strata for r in s.reported]
    sequential = [r for r in reported if r["concurrent_wall_fraction"] == 0]
    parallel = sum(1 for r in reported if (r["concurrent_wall_fraction"] or 0) > 0)
    unknown = sum(1 for r in reported if r["concurrent_wall_fraction"] is None)
    format_one = [r for r in sequential if formats.get(r["attempt_id"]) == 1]
    usable = [r for r in sequential if formats.get(r["attempt_id"]) == 2]
    counts: dict[str, int] = {
        "reported": len(reported),
        "sequential": len(sequential),
        "parallel": parallel,
        "concurrency_unknown": unknown,
        "sequential_log_format_1": len(format_one),
        "sequential_without_log": len(sequential) - len(format_one) - len(usable),
        # A6: of those, runs of external methods, whose explorer logs are not parsed
        "sequential_external": sum(1 for r in sequential if is_external(r)),
    }
    population = (
        f"{len(reported)} reported runs: {len(sequential)} sequential "
        f"(concurrent_wall_fraction = 0), {parallel} parallel, {unknown} without the fraction"
    )
    points = _decisions(reads, usable)
    drawn = [p for p in points if p.drawn]
    reason = None
    if not sequential:
        reason = "no sequential runs: decision timing is drawn from sequential runs only (METHODOLOGY §2.4)"
    elif not usable and format_one:
        reason = "explorer log format 1: not recorded (no decision timings in the sequential runs)"
    elif not drawn:
        reason = "no explorer-log decision rows with timings and a candidate count in the sequential runs"
    if reason is not None:
        return _not_drawn(
            ctx,
            "F18",
            f"{reason}; {population}; {len(format_one)} sequential runs with log format 1",
            {"row_counts": counts},
            set(reads.files),
        )
    usable_ids = {p.attempt_id for p in drawn}
    strata = [s for s in ctx.study.strata if any(r["attempt_id"] in usable_ids for r in s.reported)]
    fig, cells, width = stratum_grid(ctx, strata)
    flat = [ax for ax, _ in cells]
    rng = stats.random_stream(ctx.settings.seed, stats.JITTER_STREAM)
    panels: dict[str, Any] = {}
    profiles_drawn: list[str] = []
    for index, stratum in enumerate(strata):
        attempts = {r["attempt_id"] for r in stratum.reported} & usable_ids
        mine = [p for p in drawn if p.attempt_id in attempts]
        profiles = ctx.profiles(stratum)
        profiles_drawn += [p for p in profiles if p not in profiles_drawn]
        panel_title(
            flat[index], index, f"{ctx.label(stratum)}, {len(attempts)} runs, {len(mine)} rows"
        )
        panels[ctx.label(stratum)] = _timing_panel(flat[index], ctx, mine, profiles, rng)
    fill = [
        Line2D([], [], marker="o", ls="", color=GREY, label="order recomputed"),
        Line2D([], [], marker="o", ls="", color=GREY, markerfacecolor="none", label="order cached"),
        Patch(facecolor=GREY, alpha=0.3, label="binned median, IQR"),
    ]
    legend(fig, ctx, [p for p in ctx.order if p in profiles_drawn], fill)
    finish(fig, ctx, descriptive=True)
    rtf_text, rtf = _rtf(usable)
    untimed = sum(1 for p in points if p.time_ms is None)
    uncounted = sum(1 for p in points if p.time_ms is not None and p.candidates is None)
    counts |= {
        "rows": len(points),
        "rows_drawn": len(drawn),
        "rows_without_timings": untimed,
        "rows_without_candidate_count": uncounted,
        "rows_at_0_ms": sum(1 for p in drawn if p.time_ms == 0),
        "order_recomputed": sum(1 for p in drawn if p.order_recomputed is True),
        "order_cached": sum(1 for p in drawn if p.order_recomputed is False),
        "order_unknown": sum(1 for p in drawn if p.order_recomputed is None),
        "frontier_searched": sum(1 for p in drawn if p.searched is True),
    }
    return FigureOut(
        "F18",
        fig,
        width,
        usable_ids,
        "Explorer decision time (frontier search plus ordering) against the candidate count, sequential runs.",
        [
            f"Points: the {len(drawn)} explorer_decisions rows with both timings and a candidate count of "
            f"{len(usable_ids)} sequential reported runs with log format 2 ({population}; {len(format_one)} "
            f"sequential runs with explorer log format 1: not recorded; {counts['sequential_without_log']} "
            f"sequential runs without explorer-log rows ({counts['sequential_external']} of them of external "
            "methods, whose explorer logs are not parsed); rows without timings: "
            f"{untimed}, rows without a "
            f"candidate count: {uncounted}, not drawn). y: search_ms + ordering_ms, the explorer's own wall-clock "
            "times (log scale); x: the unsuppressed candidates ordered, jittered by up to ±0.3 for display. "
            "The candidate count is the decision's input size (Kulich et al. 2019, Fig. 10b); the pool size is "
            "min(candidate limit, candidates) in DP mode and would fold every larger decision onto the limit.",
            "Filled: the order was recomputed (order_cache_misses rose since the run's previous row); open: the "
            "order came from the cache. The second order call of each decision is a cache hit by construction "
            "and raises no miss counter; a miss on a path that writes no row (no-frontier or suppressed returns, "
            f"escape) carries over to the next row; {counts['order_unknown']} rows without the counter drawn grey. "
            f"A frontier search ran (search_ms > 0; 0 on a snapshot cache hit) on {counts['frontier_searched']} "
            f"of {len(drawn)} rows. Rows at 0 ms: {counts['rows_at_0_ms']} (triangles at the axis floor).",
            "Bars and bands: per profile, the median and IQR (Hyndman–Fan type 2) of all rows of a candidate bin "
            f"(lower edges {', '.join(str(e) for e in BIN_EDGES)}), drawn over the counts the bin holds where it "
            f"has at least {ctx.settings.min_n} rows, nothing between bins; descriptive, not cited.",
            f"Timing context: {rtf_text}; host: {_hardware(ctx, usable)}. The explorer's timers are wall-paced.",
        ],
        {"row_counts": counts, "panels": panels, "rtf": rtf, "bin_edges": list(BIN_EDGES)},
        inputs={*reads.files, PROVENANCE},
    )


# F19 decision anatomy ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tick:
    """One insight analysis tick (map frame), its time on the movement clock in seconds."""

    tick: int
    clock_ns: int
    t_s: float
    candidates: int | None
    target: int | None
    greedy: int | None
    route: tuple[int, ...]
    robot: Alignment

    @property
    def differs(self) -> bool | None:
        return None if self.target is None or self.greedy is None else self.target != self.greedy


def _ticks(reads: Reads, attempt: str) -> list[Tick]:
    """The valid analysis ticks after first movement, in time order."""
    table = reads.raw(
        attempt,
        "decisions",
        [
            "tick",
            "clock_ns",
            "simulation_elapsed_ns",
            "valid",
            "candidates",
            "target",
            "greedy_first",
            "route",
            "robot_x",
            "robot_y",
            "robot_yaw",
        ],
    )
    if table is None:
        return []
    found = [
        Tick(
            int(r["tick"]),
            int(r["clock_ns"]),
            r["simulation_elapsed_ns"] / 1e9,
            None if r["candidates"] is None else int(r["candidates"]),
            r["target"],
            r["greedy_first"],
            tuple(r["route"] or ()),
            (float(r["robot_x"]), float(r["robot_y"]), float(r["robot_yaw"])),
        )
        for r in table.to_pylist()
        if r["valid"] and r["simulation_elapsed_ns"] is not None and r["clock_ns"] is not None
    ]
    return sorted(found, key=lambda k: (k.t_s, k.tick))


def select_tick(
    ticks: list[Tick], after_s: float, min_candidates: int, end: float
) -> tuple[Tick, str] | None:
    """SEL-4 on ticks within [after_s, end]: the first where the route's first target differs from the greedy
    first, else the first with at least ``min_candidates`` candidates; None when neither exists."""
    window = [k for k in ticks if after_s <= k.t_s <= end]
    chosen = next((k for k in window if k.differs), None)
    if chosen is not None:
        return chosen, (
            f"first tick at or after {after_s:g} s where the DP route's first target differs from the "
            "greedy first"
        )
    chosen = next((k for k in window if (k.candidates or 0) >= min_candidates), None)
    if chosen is not None:
        return chosen, (
            f"no disagreement at or after {after_s:g} s; first tick there with at least "
            f"{min_candidates} candidates"
        )
    return None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True, slots=True)
class GoalChoice:
    """The explorer goal F19 shows in (b) and how it relates to the tick: ``active`` only when dispatched at or
    before the tick and not ended before it; ``comparable`` when the insight target can be checked against it
    (the active goal, or the next goal dispatched within one insight tick after the tick)."""

    goal: Row | None
    active: bool
    comparable: bool
    relation: str


def _goal_at(goals: list[Row], t: float) -> GoalChoice:
    """The goal active at the tick; else the next one dispatched within one insight tick after it (the target the
    insight re-analysed at t relates to that dispatch); else the latest dispatched before it (ended); else the
    next one after it."""
    dispatched = sorted(
        (g for g in goals if g["dispatch_s"] is not None), key=lambda g: g["dispatch_s"]
    )
    before = [g for g in dispatched if g["dispatch_s"] <= t]
    after = [g for g in dispatched if g["dispatch_s"] > t]
    if before and (before[-1]["end_s"] is None or before[-1]["end_s"] >= t):
        goal = before[-1]
        return GoalChoice(goal, True, True, f"goal #{goal['goal_sequence']}, active at the tick")
    if after and after[0]["dispatch_s"] <= t + INSIGHT_TICK_S:
        goal = after[0]
        return GoalChoice(
            goal,
            False,
            True,
            f"no goal active at the tick; next, goal #{goal['goal_sequence']}, dispatched within one "
            "insight tick after it",
        )
    if before:
        goal = before[-1]
        return GoalChoice(
            goal,
            False,
            False,
            f"no goal active at the tick; latest, goal #{goal['goal_sequence']}, ended at "
            f"{goal['end_s']:.1f} s",
        )
    if after:
        goal = after[0]
        return GoalChoice(
            goal,
            False,
            False,
            f"no goal dispatched before the tick; next, goal #{goal['goal_sequence']}, at "
            f"{goal['dispatch_s']:.1f} s",
        )
    return GoalChoice(None, False, False, "no explorer goal in this run")


def _gate_panel(
    ax: Any, rows: list[Row], choice: GoalChoice, tick: Tick, log_format: int | None
) -> str:
    """(b) The explorer's gate evaluations of one goal against time, threshold dashed; returns the caption text."""
    goal = choice.goal
    panel_title(
        ax, 1, "explorer's gate" + ("" if goal is None else f", goal #{goal['goal_sequence']}")
    )
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel("Visible reveal length (m)")
    if log_format != 2 or goal is None:
        text = (
            "explorer log format 1: not recorded"
            if log_format == 1
            else "no explorer log for this run"
            if log_format is None
            else choice.relation
        )
        panel_message(ax, text)
        return text
    of_goal = [r for r in rows if r["goal_sequence"] == goal["goal_sequence"]]
    mine = sorted((r for r in of_goal if r["time_s"] is not None), key=lambda r: r["time_s"])
    markers = {
        "keep": dict(marker="o", facecolors="#000000"),
        "keep_small_cluster": dict(marker="s", facecolors="#000000"),
        "exhausted": dict(marker="X", facecolors="#000000"),
        "unavailable": dict(marker="o", facecolors="none"),
    }
    values = [float(r["gate_value"]) for r in mine if _finite(r["gate_value"])]
    thresholds = [float(r["threshold"]) for r in mine if _finite(r["threshold"])]
    top = max([*values, *thresholds, 1.0]) * 1.15
    for result, spec in markers.items():
        group = [r for r in mine if r["result"] == result]
        if group:
            ax.scatter(
                [r["time_s"] for r in group],
                [float(r["gate_value"]) if _finite(r["gate_value"]) else 0.0 for r in group],
                s=GATE_SIZE,
                edgecolors="#000000",
                linewidths=0.6,
                label=result.replace("_", " "),
                zorder=3,
                clip_on=False,
                **spec,
            )
    marks = [(goal["dispatch_s"], "goal dispatch", "-"), (tick.t_s, "tick", ":")]
    if goal["end_s"] is not None:
        marks.append((goal["end_s"], "goal end", "-."))
    times = [m[0] for m in marks] + [r["time_s"] for r in mine]
    if thresholds:
        # the threshold over the goal's life (one value per goal in practice; each distinct value drawn)
        last = goal["end_s"] if goal["end_s"] is not None else max(times)
        for index, value in enumerate(sorted(set(thresholds))):
            ax.hlines(
                value,
                goal["dispatch_s"],
                last,
                color=GREY,
                lw=0.8,
                ls="--",
                label="threshold" if index == 0 else None,
            )
    for x, label, dash in marks:
        ax.axvline(x, color=EDGE, lw=0.6, ls=dash, label=label)
    span = max(times) - min(times)
    ax.set_xlim(min(times) - 0.05 * span - 0.5, max(times) + 0.05 * span + 0.5)
    ax.set_ylim(0, top)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncols=3,
        handletextpad=0.3,
        columnspacing=0.8,
        frameon=False,
    )
    if not mine:
        ax.text(
            0.5,
            0.5,
            "no gate evaluations while this goal was active",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color=GREY,
        )
    end = f"status {goal['status']}" + (
        f", cancel reason {goal['cancel_reason']}" if goal.get("cancel_reason") else ""
    )
    return (
        f"{choice.relation}: {len(mine)} gate evaluations (gate_evaluations; explorer-log times mapped to "
        f"simulation time, estimates; {len(of_goal) - len(mine)} rows outside the mapped span not drawn); "
        "markers by result, an unavailable evaluation (no value) open at 0; dashed: the threshold over the "
        f"goal's life; goal end: {end}"
    )


@dataclass(frozen=True, slots=True)
class AnatomyInfo:
    """What panel (a) drew, for the caption and the sidecar. ``snapshot`` is ``moving`` (a snapshot after first
    movement), ``pre-motion``, ``none`` (no snapshot at or before the tick) or ``size-mismatch`` (the latest mask
    does not cover the reference's cells); without a usable snapshot all floor is drawn unexplored."""

    candidates: int
    finite_costs: int
    non_finite_costs: int
    cost_range: tuple[float, float] | None
    route_from_candidates: tuple[int, ...]
    snapshot: str
    snapshot_s: float | None
    snapshot_age_s: float | None
    explored_cells: int | None
    mask_cells: int | None
    reference_cells: int


def _explored_codes(
    reads: Reads, attempt: str, tick: Tick, codes: np.ndarray[Any, Any], cells: ReferenceCells
) -> tuple[str, float | None, int | None, int | None]:
    """Mark the reference floor not explored at the tick in the unknown grey (in place); returns the snapshot
    kind, its time, its explored cells and its mask size."""
    snapshots = [
        (ns, m) for ns, m in explored_snapshots(reads, attempt) if (ns or 0) <= tick.t_s * 1e9
    ]
    if not snapshots:
        codes[cells.rows, cells.columns] = MapCode.UNKNOWN
        return "none", None, None, None
    ns, mask = snapshots[-1]
    when = None if ns is None else ns / 1e9
    if mask.size != cells.rows.size:
        codes[cells.rows, cells.columns] = MapCode.UNKNOWN
        return "size-mismatch", when, None, int(mask.size)
    unexplored = ~mask
    codes[cells.rows[unexplored], cells.columns[unexplored]] = MapCode.UNKNOWN
    return ("pre-motion" if ns is None else "moving"), when, int(mask.sum()), int(mask.size)


def _anatomy_map(
    ax: Any,
    ctx: FigureContext,
    reads: Reads,
    attempt: str,
    tick: Tick,
    candidates: list[Row],
    alignment: Alignment,
    reference: tuple[np.ndarray[Any, Any], Extent],
    cells: ReferenceCells,
) -> tuple[Any, AnatomyInfo]:
    """(a) Reference greys with the explored floor, the path so far and the tick's candidates and route."""
    codes, extent = reference
    codes = codes.copy()
    kind, when, explored, mask_size = _explored_codes(reads, attempt, tick, codes, cells)
    draw_reference(ax, codes, extent)
    ax.set_axis_off()
    path = ground_truth(ctx, attempt, tick.t_s)
    if path is not None:
        _, xs, ys, yaw = path
        ax.plot(
            xs,
            ys,
            color="#000000",
            lw=1.0,
            solid_capstyle="round",
            path_effects=[
                patheffects.Stroke(linewidth=2.0, foreground="#FFFFFF"),
                patheffects.Normal(),
            ],
            zorder=3,
        )
        if yaw is not None:
            tip = (xs[0] + ROBOT_ARROW_M * math.cos(yaw), ys[0] + ROBOT_ARROW_M * math.sin(yaw))
            ax.annotate(
                "",
                xy=tip,
                xytext=(xs[0], ys[0]),
                arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.0),
                zorder=4,
            )
    placed = {
        int(r["candidate"]): (*to_world(alignment, float(r["x"]), float(r["y"])), r)
        for r in candidates
    }
    finite = [r for r in candidates if _finite(r["start_cost"])]
    costs = [float(r["start_cost"]) for r in finite]
    low, high = (min(costs), max(costs)) if costs else (0.0, 1.0)
    if high - low < 1e-9:
        low, high = low - 0.5, high + 0.5
    mappable = None
    if finite:
        mappable = ax.scatter(
            [placed[int(r["candidate"])][0] for r in finite],
            [placed[int(r["candidate"])][1] for r in finite],
            c=costs,
            cmap=plt.get_cmap(SEQUENTIAL_CMAP),
            norm=Normalize(low, high),
            s=CANDIDATE_SIZE,
            edgecolors=EDGE,
            linewidths=0.6,
            zorder=4,
        )
    others = [r for r in candidates if not _finite(r["start_cost"])]
    if others:
        ax.scatter(
            [placed[int(r["candidate"])][0] for r in others],
            [placed[int(r["candidate"])][1] for r in others],
            s=CANDIDATE_SIZE,
            facecolors="none",
            edgecolors=GREY,
            linewidths=0.8,
            zorder=4,
        )
    robot = (*to_world(alignment, tick.robot[0], tick.robot[1]), tick.robot[2] + alignment[2])
    route = [placed[c] for c in tick.route if c in placed]
    if route:
        ax.plot(
            [robot[0], *(p[0] for p in route)],
            [robot[1], *(p[1] for p in route)],
            color=ROUTE_COLOUR,
            lw=1.5,
            solid_joinstyle="round",
            path_effects=[patheffects.Stroke(linewidth=2.5, foreground=EDGE), patheffects.Normal()],
            zorder=3.5,
        )
        for x, y, row in route:
            ax.annotate(
                str(row["route_position"]),
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="#FFFFFF")],
                zorder=5,
            )
    tip = (
        robot[0] + ROBOT_ARROW_M * math.cos(robot[2]),
        robot[1] + ROBOT_ARROW_M * math.sin(robot[2]),
    )
    ax.annotate(
        "",
        xy=tip,
        xytext=robot[:2],
        arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.2),
        zorder=4.5,
    )
    if tick.target in placed:
        x, y, _ = placed[tick.target]
        ax.scatter(
            [x],
            [y],
            marker="*",
            s=TARGET_SIZE,
            color=ROUTE_COLOUR,
            edgecolors=EDGE,
            linewidths=0.6,
            zorder=4.6,
        )
    if tick.greedy in placed:
        x, y, _ = placed[tick.greedy]
        ax.scatter(
            [x],
            [y],
            marker="D",
            s=GREEDY_SIZE,
            facecolors="none",
            edgecolors="#000000",
            linewidths=0.9,
            zorder=4.7,
        )
    order = sorted(
        (int(r["route_position"]), int(r["candidate"])) for r in candidates if r["route_position"]
    )
    info = AnatomyInfo(
        candidates=len(candidates),
        finite_costs=len(finite),
        non_finite_costs=len(others),
        cost_range=(low, high) if costs else None,
        route_from_candidates=tuple(c for _, c in order),
        snapshot=kind,
        snapshot_s=when,
        snapshot_age_s=None if kind in ("none", "size-mismatch") else tick.t_s - (when or 0.0),
        explored_cells=explored,
        mask_cells=mask_size,
        reference_cells=int(cells.rows.size),
    )
    return mappable, info


def _anatomy_legend(fig: Figure) -> None:
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=plt.get_cmap(SEQUENTIAL_CMAP)(0.5),
            markeredgecolor=EDGE,
            label="candidate (start cost)",
        ),
        Line2D(
            [], [], marker="o", ls="", color=GREY, markerfacecolor="none", label="non-finite cost"
        ),
        Line2D([], [], color=ROUTE_COLOUR, lw=1.5, label="DP route (numbered)"),
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            ms=9,
            color=ROUTE_COLOUR,
            markeredgecolor=EDGE,
            label="target",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            ms=7,
            color="#000000",
            markerfacecolor="none",
            label="greedy first",
        ),
        Line2D([], [], color="#000000", lw=1.0, label="path so far"),
        Patch(facecolor="#FFFFFF", edgecolor=EDGE, label="explored floor"),
        Patch(facecolor=MAP_UNKNOWN, edgecolor=EDGE, label="not yet explored"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=4)


def _snapshot_text(info: AnatomyInfo) -> str:
    if info.snapshot == "none":
        return "no explored_cells snapshot at or before the tick: all floor drawn unexplored"
    if info.snapshot == "size-mismatch":
        return (
            f"the latest explored_cells snapshot holds {info.mask_cells} cells, not the reference's "
            f"{info.reference_cells}: all floor drawn unexplored"
        )
    where = (
        "the pre-motion snapshot"
        if info.snapshot == "pre-motion"
        else (f"the snapshot at {info.snapshot_s:.1f} s")
    )
    return f"white the reference floor explored by {where} ({info.snapshot_age_s:.1f} s before the tick)"


def _base_rate(ticks: list[Tick], end: float, after: float) -> tuple[dict[str, int], str]:
    """Route-versus-greedy disagreement over the run's ticks from first movement to min(T_end, L)."""
    window = [k for k in ticks if 0 <= k.t_s <= end]
    judged = [k for k in window if k.differs is not None]
    late = [k for k in judged if k.t_s >= after]
    disagree, disagree_late = sum(1 for k in judged if k.differs), sum(1 for k in late if k.differs)
    counts = {
        "ticks": len(window),
        "judged": len(judged),
        "disagree": disagree,
        "judged_at_or_after": len(late),
        "disagree_at_or_after": disagree_late,
    }
    missing = f"{len(window) - len(judged)} ticks without a target or greedy first"
    if not judged:
        return counts, f"no tick had both a target and a greedy first; {missing}"
    return counts, (
        f"the route's first target differed from the greedy first at {disagree} of {len(judged)} ticks with "
        f"both ({100 * disagree / len(judged):.1f} %) between first movement and min(T_end, L), "
        f"{disagree_late} of {len(late)} at or after {after:g} s; {missing}"
    )


def _target_check(choice: GoalChoice, target: Row | None, tick: Tick) -> tuple[dict[str, Any], str]:
    """The insight target against a comparable goal's dispatch point (the point the insight reports), with the
    distance from the target to the goal's frontier centroid beside it; "no comparable goal" otherwise."""
    goal = choice.goal
    if goal is None or target is None:
        reason = choice.relation if goal is None else "the tick has no target"
        return {"goal_sequence": None}, f"not possible: {reason}"
    if not choice.comparable:
        found = {
            "goal_sequence": goal["goal_sequence"],
            "relation": choice.relation,
            "active": False,
            "comparable": False,
            "dispatch_s": goal["dispatch_s"],
            "distance_to_dispatch_point_m": None,
            "target_to_goal_centroid_m": None,
            "match": None,
        }
        return found, (
            f"no comparable goal ({choice.relation}; no goal dispatched within {INSIGHT_TICK_S:g} s, one "
            "insight tick, after it): the target is not compared"
        )
    point, centroid = (
        (goal["dispatch_x"], goal["dispatch_y"]),
        (goal["frontier_x"], goal["frontier_y"]),
    )
    distance = (
        math.hypot(target["x"] - point[0], target["y"] - point[1])
        if all(_finite(v) for v in point)
        else None
    )
    to_centroid = (
        math.hypot(target["x"] - centroid[0], target["y"] - centroid[1])
        if distance is not None and all(_finite(v) for v in centroid)
        else None
    )
    found = {
        "goal_sequence": goal["goal_sequence"],
        "relation": choice.relation,
        "active": choice.active,
        "comparable": True,
        "dispatch_s": goal["dispatch_s"],
        "distance_to_dispatch_point_m": distance,
        "target_to_goal_centroid_m": to_centroid,
        "match": None if distance is None else distance <= MATCH_M,
    }
    if distance is None:
        return found, f"goal #{goal['goal_sequence']} has no dispatch point in the explorer log"
    if distance <= MATCH_M:
        verdict = f"match (tolerance {MATCH_M:g} m)"
    elif choice.active:
        verdict = (
            f"mismatch (tolerance {MATCH_M:g} m): they may differ because the explorer keeps an active goal "
            "between its decisions while the insight node re-analyses every tick"
        )
    else:
        verdict = (
            f"mismatch (tolerance {MATCH_M:g} m); the goal was dispatched after the tick, from the explorer's "
            "own decision then"
        )
    centroid_text = (
        ""
        if to_centroid is None
        else f", and {to_centroid:.2f} m from the goal centroid (goals.frontier_x/y)"
    )
    since = tick.t_s - goal["dispatch_s"]
    when = f"{since:.2f} s before" if since >= 0 else f"{-since:.2f} s after"
    return found, (
        f"{choice.relation} (dispatched {when} the tick): the insight target lies {distance:.2f} m "
        f"from the dispatch point of goal #{goal['goal_sequence']} (goals.dispatch_x/y, the point the insight "
        f"also reports){centroid_text}; {verdict}"
    )


def _tick_candidates(reads: Reads, attempt: str, tick: Tick) -> list[Row]:
    """The raw ``frontier_candidates`` rows of one tick (the same message as the tick's ``decisions`` row)."""
    table = reads.raw(
        attempt,
        "frontier_candidates",
        [
            "tick",
            "clock_ns",
            "candidate",
            "x",
            "y",
            "start_cost",
            "route_position",
            "greedy_position",
        ],
    )
    rows = [] if table is None else table.to_pylist()
    return [r for r in rows if r["tick"] == tick.tick and r["clock_ns"] == tick.clock_ns]


def f19(ctx: FigureContext) -> FigureOut:
    reads = Reads(ctx.study.export)
    after, minimum = ctx.settings.event_after_s, ctx.settings.event_min_candidates
    rule = ctx.settings.representative_rule
    stratum: Stratum | None = ctx.study.strata[0] if ctx.study.strata else None
    baseline = ctx.study.baselines.get(stratum.robot_id) if stratum is not None else None
    picked = (
        [(r, n) for r, n in representatives(ctx, stratum) if r["profile_id"] == baseline]
        if stratum is not None
        else []
    )

    name: str | None = None  # the run's coverage reference, once looked up

    def refused(
        reason: str,
        extra: dict[str, Any],
        attempts: set[str] | None = None,
        reference: bool = False,
    ) -> FigureOut:
        # the reference's files once the run's reference was looked up (reference_name reads provenance.json)
        files = reference_files(ctx, name) if reference else set()
        return _not_drawn(ctx, "F19", reason, extra, reads.files | files, attempts)

    if stratum is None or not picked:
        return refused(
            f"no representative Default run in the first stratum ({rule})", {"rule": rule}
        )
    run, note = picked[0]
    attempt = run["attempt_id"]
    end, _ = run_end(run, stratum.limit_s)
    ticks = _ticks(reads, attempt)
    selection = select_tick(ticks, after, minimum, end)
    base = {
        "rule": rule,
        "attempt_id": attempt,
        "event_tick": {"after_s": after, "min_candidates": minimum},
    }
    if selection is None:
        window = (
            f"run ended at {end:.2f} s, before event_tick.after_s = {after:g} s"
            if end < after
            else f"no insight tick in [{after:g} s, {end:g} s] with a DP–greedy disagreement or at least "
            f"{minimum} candidates"
        )
        return refused(f"run {attempt[:8]} ({note}): {window} (SEL-4)", base, {attempt})
    tick, label = selection
    alignment = world_from_map(reads, attempt)
    if alignment is None:
        return refused(f"run {attempt[:8]} has no map-to-world alignment", base, {attempt})
    name = reference_name(ctx, attempt)
    reference = load_reference(ctx, attempt)
    cells = reference_cells(reads, name) if name is not None else None
    if reference is None or cells is None:
        why = (
            "has no exported coverage reference"
            if name is None
            else f"has a coverage reference ({name}) that is not readable (a rotated origin, or a target "
            "count other than its reference.yaml's)"
        )
        return refused(f"run {attempt[:8]} {why}", base, {attempt}, reference=True)
    # the figure's height follows the map: panel (a) keeps its share of the width at equal axes
    extent = reference[1]
    shape = (extent[3] - extent[2] + 1.0) / (extent[1] - extent[0] + 1.0)
    aspect = min(MAP_SHARE * shape + ANATOMY_OVERHEAD, ANATOMY_MAX_ASPECT)
    fig, axes = figure(ctx.preset, "double", aspect=aspect)
    # the preset figure comes with one axes; the panels go on a grid of their own, so it is removed
    axes[0][0].remove()
    # (b) keeps a fixed height beside a tall map; its legend sits in the cell below it
    gate_share = min(1.0, GATE_ASPECT / aspect)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[MAP_SHARE, 1 - MAP_SHARE],
        height_ratios=[gate_share, 1 - gate_share] if gate_share < 1 else [1, 1e-6],
    )
    ax_map, ax_gate = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1])
    panel_title(ax_map, 0, f"{ctx.label(stratum)}, t = {tick.t_s:.1f} s (insight re-analysis)")
    candidates = _tick_candidates(reads, attempt, tick)
    mappable, info = _anatomy_map(
        ax_map, ctx, reads, attempt, tick, candidates, alignment, reference, cells
    )
    if mappable is not None:
        bar = fig.colorbar(mappable, ax=ax_map, orientation="vertical", fraction=0.04, pad=0.02)
        bar.set_label("Start cost M(0, j)")
    goals = [g for g in reads.rows("goals") if g["attempt_id"] == attempt]
    choice = _goal_at(goals, tick.t_s)
    gates = [g for g in reads.rows("gate_evaluations") if g["attempt_id"] == attempt]
    gate_text = _gate_panel(ax_gate, gates, choice, tick, log_formats(reads).get(attempt))
    _anatomy_legend(fig)
    finish(fig, ctx, descriptive=True)

    rate, rate_text = _base_rate(ticks, end, after)
    target = next((r for r in candidates if int(r["candidate"]) == tick.target), None)
    check, check_text = _target_check(choice, target, tick)
    notes = [
        f"Run {attempt[:8]}, the Default run of {ctx.label(stratum)} picked by {note} (FIGURE_STYLE §8). "
        f"Tick {tick.tick} at t = {tick.t_s:.1f} s chosen by SEL-4 (event_tick after_s = {after:g} s, "
        f"min_candidates = {minimum}, ticks up to min(T_end, L) = {end:.1f} s): {label}.",
        "Candidates, DP route, target and greedy first are the frontier insight node's re-analysis of the "
        "explorer's inputs at that tick (1 Hz simulation time), not the explorer's internal state. Colour: start "
        f"cost M(0, j) on cividis over the tick's {info.finite_costs} finite values; {info.non_finite_costs} "
        "candidates with a non-finite cost hollow grey; one mark size (never by gain). Numbers: position on "
        "the DP route, drawn from the robot pose (arrow); star: the target (route first); open diamond: the "
        "greedy first.",
        f"Map (FP-5 greys): {_snapshot_text(info)}; light grey floor not yet explored and other inside cells; "
        "dark grey obstacles; hatched outside the reference. Black: the ground-truth path from first movement "
        "to the tick (1 s rows), arrow at its start the start pose.",
        f"Disagreement base rate (descriptive): {rate_text}.",
        f"Target check: {check_text}.",
        f"(b) {gate_text}.",
    ]
    extra = {
        **base,
        "representative_note": note,
        "tick": tick.tick,
        "tick_s": tick.t_s,
        "selection": label,
        "base_rate": rate,
        "map": {
            **asdict(info),
            "candidates_in_decision_row": tick.candidates,
            "route": list(tick.route),
            "target": tick.target,
            "greedy_first": tick.greedy,
        },
        "target_check": check,
    }
    inputs = reads.files | reference_files(ctx, name)
    return FigureOut(
        "F19",
        fig,
        "double",
        {attempt},
        "One explorer decision tick of a representative Default run: candidates by start cost, DP route, "
        "target and greedy first on the map, with the gate evaluations of the explorer's goal.",
        notes,
        extra,
        inputs=inputs,
    )


def figures(ctx: FigureContext) -> Iterator[FigureOut]:
    """F18 and F19, each built when the caller asks for it."""
    yield from grouped(ctx, f18, grid_fits)  # once per world group and page (plans/worlds/I8 §3.1)
    yield f19(ctx)
