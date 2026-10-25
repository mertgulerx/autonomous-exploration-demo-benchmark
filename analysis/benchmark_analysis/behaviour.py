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

"""Figures F15 and F16: what one run did over time, and coverage against the path travelled
(docs/plans/FIGURE_EXTENSION.md, FX3; FIGURE_STYLE §3, §4, §8, §9; METHODOLOGY §3).

- F15 (one per stratum, ``F15_<stratum>``, named by ``figures.stratum_variants`` as F17): the behaviour timeline of
  FX-15 for the runs the plan's representative rule picks (SEL-1 to SEL-3), one column per run, one lane per stream
  on one movement-relative time axis in a fixed order: coverage (valid samples as steps, milestones, the outcome
  mark at the run's end); Nav2 goals as bars from dispatch to end, filled by status, the explorer's cancel reason at
  the end of a goal it ended, goals Nav2 rejected as ticks (the shared goal model, ``goals``); the preemption gate's
  results; the explorer's decisions (candidate frontiers and the frontiers in the chosen route); RTF windows with
  lines at 0.90 and 0.95. Every lane counts and draws the same range, [0, the run's end]. The time axis is the
  shared FX-2 axis (``figures.time_axis``, an L beyond it named). Explorer-log times are mapped estimates
  (``time_mapped``). A run whose explorer wrote log format 1 has no gate rows, cancel reasons or decision counts:
  its lanes say so instead of staying empty.
- F16: per stratum, every reported run's coverage (valid samples) as steps against the ground-truth path length
  at each sample (``trajectory_gt.gt_distance_m``, the last row at or before the sample; no interpolation), in the
  profile's style, with an outcome end mark when the trace reaches the run's end; a run that never moved is one
  mark at distance 0 and its pre-motion coverage (FX-7). No band, no carry-forward, no distance-to-coverage table.

Counts and values the figures compute (gate results, RTF extremes, end distances) are descriptive display
quantities in the sidecar, never cited. F16's panel grid is ``figures.stratum_grid``, one figure per world group
and page (``figures.grouped``, plans/worlds/I8 §3.1); F15's labels name the stratum's group when the study has
several (``figures.stratum_variants``).
"""

from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math
from typing import Any

from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from .data import GAP_S, TIME_EPSILON, Row, Stratum, run_end
from .figures import (
    AXIS_NOTE,
    GREY,
    TIME_LABEL,
    TITLE_CHAR_MM,
    FigureContext,
    FigureOut,
    finish,
    grid_fits,
    group_inputs,
    grouped,
    legend,
    panel_message,
    panel_title,
    representatives,
    stratum_grid,
    stratum_variants,
    study_profiles,
    time_axis,
    time_axis_end,
)
from .goals import GoalRecord, bar_handles, bar_span, draw_goal_bars, read_goals
from .settings import milestone_key
from .streams import Reads, log_formats
from .style import MM_PER_INCH, OUTCOMES, PRESETS, RTF_LINES, ProfileStyle, figure

# F15 behaviour timeline ----------------------------------------------------------------------------------------

LANES = ("coverage", "goals", "gate", "decisions", "rtf")  # FX-15: fixed order, top to bottom
LANE_HEIGHTS = (3.0, 1.1, 1.5, 1.5, 1.3)
LANE_LABELS = ("Coverage (%)", "Goals", "Gate", "Frontiers", "RTF")
TIMELINE_MM = 112.0  # the lanes, titles and axis
LEGEND_ROW_MM = 4.5  # one legend row
LEGEND_COLUMNS = 4
GATE_RESULTS = (
    "keep",
    "keep_small_cluster",
    "exhausted",
    "unavailable",
)  # log format 2 (MEASUREMENTS)
GATE_LABELS = {"keep_small_cluster": "keep (small)"}
NOT_RECORDED = "explorer log format 1: not recorded"
NO_LOG = "no explorer-log rows"
# the timeline's own marks, drawn and listed in the legend with one look; lanes report the keys they drew
MARKS: dict[str, dict[str, Any]] = {
    "milestone": dict(marker="o", ms=3, mfc="#FFFFFF", mec="#000000", mew=0.5, ls="none"),
    "completed": dict(
        marker="s",
        ms=4.5,
        mfc=OUTCOMES["completed"]["facecolor"],
        mec="#000000",
        mew=0.6,
        ls="none",
    ),
    "time_limit": dict(marker="^", ms=4.5, mfc="#FFFFFF", mec="#000000", mew=0.6, ls="none"),
    "candidates": dict(color="#000000", lw=0.7, marker=".", ms=2),
    "route": dict(color=GREY, lw=0.7, ls="--"),
}
MARK_LABELS = {
    "milestone": "milestone",
    "completed": "completed (end)",
    "time_limit": "time limit (at L)",
    "candidates": "candidate frontiers",
    "route": "frontiers in the chosen route",
}


@dataclass(frozen=True, slots=True)
class RunStreams:
    """One run's streams for its timeline, on the movement clock (simulated s), every row in [0, ``end_s``]."""

    end_s: float  # the run's end: completion time, or L
    goals: GoalRecord
    gate: tuple[tuple[float, str], ...]  # time, result
    decisions: tuple[tuple[float, int, int | None], ...]  # time, candidates, route length
    rtf: tuple[tuple[float, float], ...] | None  # None when the export holds no quality stream
    outside: dict[
        str, int
    ]  # rows per stream left out: no mapped time, before 0 or after the run's end
    log_format: int | None


def _in_range(rows: list[tuple[float | None, Any]], end: float) -> tuple[list[Any], int]:
    """The (time, value) rows with a time in [0, end] in time order, and how many others there were."""
    kept = sorted(
        ((t, v) for t, v in rows if t is not None and 0.0 <= t <= end + TIME_EPSILON),
        key=lambda row: row[0],
    )
    return kept, len(rows) - len(kept)


def _by_run(reads: Reads, name: str, columns: list[str], wanted: set[str]) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in reads.rows(name, ["attempt_id", *columns]):
        if row["attempt_id"] in wanted:
            grouped[row["attempt_id"]].append(row)
    return grouped


def _read_streams(
    reads: Reads, runs: Sequence[Row], limit: float, formats: dict[str, int]
) -> dict[str, RunStreams]:
    """The streams of every chosen run, each table read once per stratum."""
    wanted = {r["attempt_id"] for r in runs}
    goals = read_goals(reads, runs)
    gate = _by_run(reads, "gate_evaluations", ["time_s", "result"], wanted)
    decisions = _by_run(
        reads, "explorer_decisions", ["time_s", "candidates", "route_length"], wanted
    )
    found = {}
    for run in runs:
        attempt = run["attempt_id"]
        end, _ = run_end(run, limit)
        gate_rows, gate_out = _in_range(
            [(r["time_s"], str(r["result"])) for r in gate.get(attempt, [])], end
        )
        # format-1 rows are dispatch lines without counts (MEASUREMENTS "Goal and explorer-log tables")
        decision_rows, decision_out = _in_range(
            [
                (r["time_s"], (int(r["candidates"]), r["route_length"]))
                for r in decisions.get(attempt, [])
                if r["candidates"] is not None
            ],
            end,
        )
        quality = reads.raw(
            attempt, "quality", ["category", "observation", "value", "simulation_elapsed_ns"]
        )
        rtf, rtf_out = None, 0
        if quality is not None:
            windows = [
                (None if ns is None else ns / 1e9, float(value))
                for category, observation, value, ns in zip(
                    *(quality.column(i).to_pylist() for i in range(4)), strict=True
                )
                if category == "timing" and observation == "rtf_window" and value is not None
            ]
            kept, rtf_out = _in_range(windows, end)
            rtf = tuple(kept)
        found[attempt] = RunStreams(
            end,
            goals[attempt],
            tuple(gate_rows),
            tuple((t, c, r) for t, (c, r) in decision_rows),
            rtf,
            dict(gate=gate_out, decisions=decision_out, rtf=rtf_out),
            formats.get(attempt),
        )
    return found


def _log_note(streams: RunStreams) -> str | None:
    """Why a lane fed by the explorer log is empty, or None when the run wrote format 2."""
    if streams.log_format == 1:
        return NOT_RECORDED
    return NO_LOG if streams.log_format is None else None


def _coverage_lane(
    ax: Any, ctx: FigureContext, run: Row, style: ProfileStyle, end: float, limit: float
) -> tuple[dict[str, Any], set[str]]:
    """Coverage steps broken at gaps over 2 s, the first sample at each declared milestone (at its threshold),
    and the outcome mark at the run's end (a completed run's completion time, L otherwise) when the last sample
    lies within 2 s of it. Returns the run's sidecar fields and the marks drawn."""
    drawn: set[str] = set()
    series = ctx.study.coverage.get(run["attempt_id"])
    last: float | None = None
    keep = series.t <= end + TIME_EPSILON if series is not None else None
    if series is None or keep is None or not keep.any():
        panel_message(ax, "no valid coverage samples", boxed=True)
    else:
        t, c = series.t[keep], series.coverage[keep]
        breaks = np.nonzero(np.diff(t) > GAP_S)[0] + 1
        final = end if end - t[-1] <= GAP_S else t[-1]
        for part_t, part_c in zip(np.split(t, breaks), np.split(c, breaks), strict=True):
            stop = final if part_t[-1] == t[-1] else part_t[-1]
            ax.step(
                np.append(part_t, stop),
                np.append(part_c, part_c[-1]),
                where="post",
                color=style.color,
                lw=1.0,
            )
        if final == end:
            last = float(c[-1])
    for m in ctx.settings.milestones:
        key = milestone_key(m)
        at = run.get(f"t{key}_s")
        if run.get(f"t{key}_event") and at is not None and at <= end + TIME_EPSILON:
            ax.plot([at], [m], zorder=4, **MARKS["milestone"])
            ax.annotate(
                f"{m}",
                (at, m),
                xytext=(3, -1),  # below right: empty under a rising step curve
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=7,
            )
            drawn.add("milestone")
    outcome = run["outcome"]
    if last is not None and outcome in MARKS:
        ax.plot([end], [last], zorder=5, clip_on=False, **MARKS[outcome])
        drawn.add(outcome)
    words = (
        f"completed, T_end = {run['elapsed_s']:.1f} s"
        if outcome == "completed" and end < limit
        else f"{OUTCOMES[outcome]['label'].lower()} at L"
    )
    ax.text(0.99, 0.04, words, transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    return dict(end_s=end, end_coverage_pct=last), drawn


def _goal_lane(ax: Any, streams: RunStreams) -> set[str]:
    """Goal bars (reasons only from log format 2) and what the lane cannot show, named."""
    drawn = draw_goal_bars(ax, streams.goals, streams.end_s, reasons=streams.log_format == 2)
    rejections = streams.goals.rejections
    remarks = []
    if streams.log_format == 1:
        remarks.append("cancel reasons: " + NOT_RECORDED)
    if rejections.counted is None:
        remarks.append("navigation telemetry not recorded")
    if rejections.unplaced:
        remarks.append(f"{rejections.unplaced} rejected goals without a time")
    if remarks:
        ax.text(
            0.99,
            0.97,
            "; ".join(remarks),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color=GREY,
        )
    return drawn


def _gate_lane(ax: Any, streams: RunStreams, rows: Sequence[str]) -> dict[str, int]:
    note = _log_note(streams)
    if note is not None:
        panel_message(ax, note, boxed=True)
        return {}
    if not streams.gate:
        panel_message(ax, "no gate evaluations", boxed=True)
    for index, result in enumerate(rows):
        times = [t for t, r in streams.gate if r == result]
        if times:
            ax.plot(
                times, [index] * len(times), marker="|", ms=4, mew=0.6, color="#333333", ls="none"
            )
    return dict(sorted(Counter(r for _, r in streams.gate).items()))


def _decision_lane(ax: Any, streams: RunStreams) -> set[str]:
    note = _log_note(streams)
    if note is not None:
        panel_message(ax, note, boxed=True)
        return set()
    if not streams.decisions:
        panel_message(ax, "no decisions", boxed=True)
        return set()
    times = [t for t, _, _ in streams.decisions]
    ax.step(times, [c for _, c, _ in streams.decisions], where="post", **MARKS["candidates"])
    routes = [(t, r) for t, _, r in streams.decisions if r is not None]
    if not routes:
        return {"candidates"}
    ax.step([t for t, _ in routes], [r for _, r in routes], where="post", **MARKS["route"])
    return {"candidates", "route"}


def _rtf_lane(ax: Any, streams: RunStreams) -> None:
    if streams.rtf is None:
        panel_message(ax, "quality stream not exported", boxed=True)
    elif streams.rtf:
        ax.plot([t for t, _ in streams.rtf], [v for _, v in streams.rtf], color="#000000", lw=0.6)
    else:
        panel_message(ax, "no RTF windows after first movement", boxed=True)
    for threshold, linestyle, _ in RTF_LINES:
        ax.axhline(threshold, color=GREY, lw=0.6, ls=linestyle)


def _ranges(streams: Sequence[RunStreams]) -> tuple[tuple[float, float], float | None]:
    """Shared y ranges: the RTF range (at least 0.85–1.05) and the frontier axis' top (None without counts)."""
    values = [v for s in streams for _, v in s.rtf or ()]
    low = min([0.85, *(v - 0.01 for v in values)])
    high = max([1.05, *(v + 0.01 for v in values)])
    counts = [max(c, r or 0) for s in streams for _, c, r in s.decisions]
    return (low, high), (max(counts) * 1.15 if counts else None)


def _run_lanes(
    cells: Sequence[Any],
    ctx: FigureContext,
    run: Row,
    streams: RunStreams,
    gate_rows: Sequence[str],
    limit: float,
) -> tuple[dict[str, Any], set[str]]:
    """Draw one run's lanes; returns its sidecar entry (descriptive) and the marks drawn."""
    cover, goals_ax, gate_ax, decision_ax, rtf_ax = cells
    style, attempt = ctx.styles[run["profile_id"]], run["attempt_id"]
    if run["outcome"] == "no_movement":
        for ax in cells:
            panel_message(ax, "never moved: no movement-relative times", boxed=True)
        return dict(pre_motion_coverage_pct=ctx.study.pre_motion.get(attempt)), set()
    end = streams.end_s
    entry, drawn = _coverage_lane(cover, ctx, run, style, end, limit)
    drawn |= _goal_lane(goals_ax, streams)
    rejections = streams.goals.rejections
    entry |= dict(
        log_format=streams.log_format,
        range_s=[0.0, end],
        goals=[
            dict(
                sequence=g.sequence,
                dispatch_s=g.dispatch_s,
                end_s=g.end_s,
                bar_s=None if (span := bar_span(g, end)) is None else list(span),
                status=g.status,
                cancel_reason=g.reason,
            )
            for g in streams.goals.goals
        ],
        rejected_s=list(rejections.times_s),
        rejected_unplaced=rejections.unplaced,
        rejected_counted=rejections.counted,
        gate_results=_gate_lane(gate_ax, streams, gate_rows),
        decisions=len(streams.decisions),
        rtf_windows=None if streams.rtf is None else len(streams.rtf),
        rtf_min=min((v for _, v in streams.rtf or ()), default=None),
        outside_range=streams.outside,
    )
    drawn |= _decision_lane(decision_ax, streams)
    _rtf_lane(rtf_ax, streams)
    return entry, drawn


def _axes_frame(
    cells: Sequence[Any],
    stratum: Stratum,
    gate_rows: Sequence[str],
    ranges: tuple[tuple[float, float], float | None],
) -> None:
    """The shared time axis (``figures.time_axis``, FX-2: L as a line in every lane when on the axis, "L" above
    the coverage lane; named when beyond) and each lane's y axis."""
    cover, goals_ax, gate_ax, decision_ax, rtf_ax = cells
    limit = stratum.limit_s
    if time_axis(cover, stratum, l_text="top") >= limit:  # the column's lanes share the x axis
        for ax in cells[1:]:
            ax.axvline(limit, color=GREY, lw=0.6, ls=":")
    cover.set_ylim(0, 100)
    goals_ax.set_ylim(0, 1)
    goals_ax.set_yticks([])
    gate_ax.set_ylim(len(gate_rows) - 0.3, -0.7)  # first result at the top
    gate_ax.set_yticks(
        range(len(gate_rows)), [GATE_LABELS.get(r, r) for r in gate_rows], fontsize=7
    )
    rtf_range, frontier_top = ranges
    if frontier_top is None:
        decision_ax.set_yticks([])
    else:
        decision_ax.set_ylim(0, frontier_top)
        decision_ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    rtf_ax.set_ylim(*rtf_range)
    rtf_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    rtf_ax.set_xlabel(TIME_LABEL)


def _handles(drawn: set[str]) -> list[Artist]:
    handles = bar_handles(drawn)
    handles += [
        Line2D([], [], label=MARK_LABELS[key], **MARKS[key]) for key in MARKS if key in drawn
    ]
    handles += [
        Line2D([], [], color=GREY, lw=0.6, ls=linestyle, label=f"RTF {value:.2f}")
        for value, linestyle, _ in RTF_LINES
    ]
    return handles


def f15(
    ctx: FigureContext,
    stratum: Stratum,
    variant: tuple[str, str],
    reads: Reads,
    formats: dict[str, int],
) -> FigureOut:
    """One stratum's timeline; ``variant`` is its output name and label (``figures.stratum_variants``)."""
    name, variant_label = variant
    chosen = representatives(ctx, stratum)
    limit, end_axis = stratum.limit_s, time_axis_end(stratum)
    streams = _read_streams(reads, [run for run, _ in chosen], limit, formats)
    gate_rows = [
        *GATE_RESULTS,
        *sorted({r for s in streams.values() for _, r in s.gate} - set(GATE_RESULTS)),
    ]
    # SEL-2 picks a block's two runs and names each one's profile in its note; the other rules pick Default runs
    notes_name_profile = ctx.settings.representative_rule == "SEL-2" and len(chosen) == 2
    columns = max(1, len(chosen))
    record: list[dict[str, Any]] = []
    drawn: set[str] = set()
    layouts: list[tuple[list[Any], Row, str]] = []
    # the legend's rows depend on what is drawn: the figure's height is set once the handles are known
    fig, axes = figure(
        ctx.preset,
        "double",
        aspect=1.0,
        nrows=len(LANES),
        ncols=columns,
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": LANE_HEIGHTS},
    )
    ranges = _ranges(list(streams.values()))
    for column in range(columns):
        cells = [axes[row][column] for row in range(len(LANES))]
        _axes_frame(cells, stratum, gate_rows, ranges)
        if column < len(chosen):
            run, note = chosen[column]
            layouts.append((cells, run, note))
    if not chosen:
        panel_title(axes[0][0], 0, ctx.label(stratum))
        panel_message(
            axes[0][0], "no Default run with a restricted time: no representative run", boxed=True
        )
    for column, (cells, run, note) in enumerate(layouts):
        profile = ctx.styles[run["profile_id"]].name
        title = note if notes_name_profile else f"{note} · {profile}"
        # a long note (a tie at L) breaks into lines of the column's width
        chars = int(PRESETS[ctx.preset].widths_mm["double"] / columns / TITLE_CHAR_MM)
        panel_title(cells[0], column, f"{ctx.label(stratum)} · {title}", wrap=chars)
        entry, marks = _run_lanes(cells, ctx, run, streams[run["attempt_id"]], gate_rows, limit)
        record.append(dict(attempt_id=run["attempt_id"], note=note, outcome=run["outcome"]) | entry)
        drawn |= marks
    for row, label in enumerate(LANE_LABELS):
        axes[row][0].set_ylabel(label, fontsize=8)
    handles = _handles(drawn)
    legend_rows = math.ceil(len(handles) / LEGEND_COLUMNS)
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    fig.set_size_inches(
        width_mm / MM_PER_INCH,
        (TIMELINE_MM + legend_rows * LEGEND_ROW_MM) / MM_PER_INCH,
        forward=False,
    )
    legend(fig, ctx, [], handles, max_columns=LEGEND_COLUMNS)
    finish(fig, ctx)
    rule, endpoint = ctx.settings.representative_rule, ctx.settings.representative_endpoint
    return FigureOut(
        name,
        fig,
        "double",
        {run["attempt_id"] for run, _ in chosen},
        f"Behaviour timeline of the representative runs of {ctx.label(stratum)}: coverage, Nav2 goals, the "
        "preemption gate, the explorer's decisions and the real-time factor on one time axis.",
        [
            f"Runs chosen by {rule} on {endpoint} (FIGURE_STYLE §8). Lanes from the top: coverage (valid samples "
            "as steps, broken at gaps over 2 s; open circles: the first sample at each declared milestone; end "
            "mark: completed (square) at T_end or time limit (open triangle) at L, when the last sample lies "
            "within 2 s of the end); Nav2 goals from dispatch to end, filled by status (an open goal to the run's "
            "end; a goal dispatched before first movement starts at 0), the explorer's cancel reason at the end of "
            "a goal it ended, a goal Nav2 rejected as a tick; results of the visible-reveal preemption gate; the "
            "explorer's decisions: unsuppressed candidate frontiers and the frontiers in the chosen route; RTF "
            "over 5-s wall windows with lines at 0.90 (suite rejection) and 0.95 (low_rtf flag). Every lane "
            "shows [0, the run's end].",
            "Gate and decision times are explorer-log stamps mapped to simulated time through the 1 Hz quality "
            "rows (time_mapped): estimates. Goal and coverage times are recorded on the simulation clock. "
            "Explorer log format 1 records no gate rows, cancel reasons or decision counts; such lanes say so. "
            f"{AXIS_NOTE} Counts in the sidecar are descriptive.",
        ],
        dict(
            stratum=variant_label,
            rule=rule,
            endpoint=endpoint,
            limit_s=limit,
            axis_end_s=end_axis,
            runs=record,
        ),
        inputs=set(reads.files),
    )


# F16 coverage against distance ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DistanceTrace:
    """One run's coverage samples with the ground-truth path length at each (the last row at or before)."""

    distance_m: tuple[float, ...]
    coverage_pct: tuple[float, ...]
    time_s: tuple[float, ...]
    without_distance: int  # samples before the first ground-truth row with a distance
    end_distance_m: float | None  # the last gt_distance_m at or before min(T_end, L)
    reaches_end: bool  # the last sample lies within 2 s of the run's end (the gap rule of F1)


def _ground_truth_rows(
    ctx: FigureContext, attempts: set[str]
) -> dict[str, list[tuple[float | None, float]]]:
    """Per run, the 1 s rows with a ground-truth path length in time order: (t_s, distance), t_s None before
    first movement (ordered by the activation clock, so they come first)."""
    table = ctx.study.export.table(
        "trajectory_gt", ["attempt_id", "t_activation_s", "t_s", "gt_distance_m"]
    )
    grouped: dict[str, list[tuple[float, float | None, float]]] = defaultdict(list)
    for attempt, t_activation, t, distance in zip(
        *(table.column(i).to_pylist() for i in range(4)), strict=True
    ):
        if attempt in attempts and distance is not None and t_activation is not None:
            grouped[attempt].append((t_activation, t, distance))
    return {a: [(t, d) for _, t, d in sorted(rows)] for a, rows in grouped.items()}


def _distance_trace(
    ctx: FigureContext, run: Row, limit: float, rows: Sequence[tuple[float | None, float]]
) -> _DistanceTrace | None:
    """The run's valid coverage samples to min(T_end, L), each paired with the last ground-truth row at or
    before it (a row before first movement counts as before every sample); no interpolation."""
    series = ctx.study.coverage.get(run["attempt_id"])
    if series is None or not series.t.size or not rows:
        return None
    end, _ = run_end(run, limit)
    times = [-math.inf if t is None else t for t, _ in rows]
    distances = [d for _, d in rows]
    pairs = []
    missing = 0
    for t, c in zip(series.t.tolist(), series.coverage.tolist(), strict=True):
        if t > end + TIME_EPSILON:
            break
        index = int(np.searchsorted(times, t + TIME_EPSILON, side="right")) - 1
        if index < 0:
            missing += 1
            continue
        pairs.append((distances[index], c, t))
    last = int(np.searchsorted(times, end + TIME_EPSILON, side="right")) - 1
    return _DistanceTrace(
        tuple(d for d, _, _ in pairs),
        tuple(c for _, c, _ in pairs),
        tuple(t for _, _, t in pairs),
        missing,
        distances[last] if last >= 0 else None,
        bool(pairs) and end - pairs[-1][2] <= GAP_S,
    )


def _distance_steps(ax: Any, trace: _DistanceTrace, style: ProfileStyle) -> float:
    """Steps of coverage against distance, broken where samples are more than 2 s apart; the last stretch runs
    to the run's end distance when the last sample is within 2 s of the end. Returns the trace's last x."""
    t = np.array(trace.time_s)
    d, c = np.array(trace.distance_m), np.array(trace.coverage_pct)
    breaks = np.nonzero(np.diff(t) > GAP_S)[0] + 1
    final = (
        trace.end_distance_m if trace.reaches_end and trace.end_distance_m is not None else d[-1]
    )
    for part_d, part_c, part_t in zip(
        np.split(d, breaks), np.split(c, breaks), np.split(t, breaks), strict=True
    ):
        stop = final if part_t[-1] == t[-1] else part_d[-1]
        ax.step(
            np.append(part_d, stop),
            np.append(part_c, part_c[-1]),
            where="post",
            color=style.color,
            ls=style.linestyle,
            lw=0.7,
            alpha=0.7,
        )
    return float(final)


def _end_look(outcome: str, colour: str, marker: str) -> dict[str, Any]:
    """An F16 end mark: completed the profile's filled marker; time limit an open triangle (the run's position
    at L); no movement a cross."""
    if outcome == "completed":
        return dict(marker=marker, ms=3.5, color=colour, ls="none")
    if outcome == "time_limit":
        return dict(marker="^", ms=4, mfc="#FFFFFF", mec=colour, mew=0.8, ls="none")
    return dict(marker="x", ms=4, mec=colour, mew=0.8, ls="none")


END_LABELS = {
    "completed": "completed",
    "time_limit": "time limit (at L)",
    "no_movement": "no movement",
}


def _distance_panel(
    ax: Any,
    ctx: FigureContext,
    stratum: Stratum,
    truth: dict[str, list[tuple[float | None, float]]],
) -> tuple[list[dict[str, Any]], Counter[str], set[str]]:
    """One stratum's traces and end marks; returns the runs' sidecar rows, the runs not drawn by reason and the
    outcomes marked."""
    rows: list[dict[str, Any]] = []
    unplotted: Counter[str] = Counter()
    marked: set[str] = set()
    furthest = 0.0
    for profile in ctx.profiles(stratum):
        style = ctx.styles[profile]
        for run in stratum.runs_of(profile):
            attempt, outcome = run["attempt_id"], run["outcome"]
            look = _end_look(outcome, style.color, style.marker)
            if outcome == "no_movement":
                held = ctx.study.pre_motion.get(attempt)
                if held is None:
                    unplotted["no_pre_motion_sample"] += 1
                    continue
                ax.plot([0.0], [held], zorder=4, clip_on=False, **look)
                marked.add(outcome)
                rows.append(
                    dict(
                        attempt_id=attempt, outcome=outcome, trace_end_m=0.0, end_coverage_pct=held
                    )
                )
                continue
            trace = _distance_trace(ctx, run, stratum.limit_s, truth.get(attempt, []))
            if trace is None or not trace.distance_m:
                unplotted["no_coverage_or_ground_truth"] += 1
                continue
            x = _distance_steps(ax, trace, style)
            furthest = max(furthest, x)
            end_coverage = trace.coverage_pct[-1] if trace.reaches_end else None
            if end_coverage is not None:
                ax.plot([x], [end_coverage], zorder=4, clip_on=False, **look)
                marked.add(outcome)
            rows.append(
                dict(
                    attempt_id=attempt,
                    outcome=outcome,
                    end_s=run_end(run, stratum.limit_s)[0],
                    end_distance_m=trace.end_distance_m,
                    trace_end_m=x,
                    end_coverage_pct=end_coverage,
                    samples=len(trace.distance_m),
                    samples_without_distance=trace.without_distance,
                )
            )
    ax.set(
        # small margins, so a mark at distance 0 or at 100 % is not drawn on a spine
        xlim=(-0.02 * max(1.0, furthest), max(1.0, furthest) * 1.04),
        ylim=(-2, 102),
        xlabel="Ground-truth path length (m)",
        ylabel="Coverage (%)",
    )
    if unplotted:
        ax.text(
            0.99,
            0.03,
            ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(unplotted.items())),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=GREY,
        )
    return rows, unplotted, marked


def f16(ctx: FigureContext) -> FigureOut:
    strata = ctx.study.strata
    fig, cells, width = stratum_grid(ctx, strata)
    truth = _ground_truth_rows(ctx, {r["attempt_id"] for s in strata for r in s.reported})
    record: list[dict[str, Any]] = []
    attempts: set[str] = set()
    outcomes: set[str] = set()
    for index, (stratum, (ax, _)) in enumerate(zip(strata, cells, strict=True)):
        runs, unplotted, marked = _distance_panel(ax, ctx, stratum, truth)
        outcomes |= marked
        attempts |= {r["attempt_id"] for r in runs}
        panel_title(ax, index, ctx.label(stratum))
        record.append(dict(stratum=ctx.label(stratum), runs=runs, not_drawn=dict(unplotted)))
    marks = [
        Line2D([], [], label=label, **_end_look(outcome, GREY, "o"))
        for outcome, label in END_LABELS.items()
        if outcome in outcomes
    ]
    legend(fig, ctx, study_profiles(ctx), marks)
    finish(fig, ctx)
    return FigureOut(
        "F16",
        fig,
        width,
        attempts,
        f"Coverage against the ground-truth path length travelled for every reported run in {len(strata)} "
        "strata, with each run's outcome at its end.",
        [
            "The x axis is the ground-truth path length (trajectory_gt gt_distance_m: the cumulative planar "
            "length of the 20 Hz ground-truth path, steps across a path break left out), not the raw-odometry "
            "distance of D(h) and ΔD(L). Steps: each run's valid coverage samples to min(T_end, L), each at the "
            "path length of the last 1 s ground-truth row at or before it (no interpolation), broken where "
            "samples are more than 2 s apart; no band and no carry-forward. End marks, only when the last "
            "sample lies within 2 s of the run's end: completed (the profile's marker), time limit (open "
            "triangle, the run's position at L); no movement: a cross at distance 0 and the pre-motion coverage. "
            "A run that stops early has a short trace: its end is not an efficiency. End distances in the "
            "sidecar are descriptive.",
        ],
        dict(strata=record),
    )


def figures(ctx: FigureContext) -> Iterator[FigureOut]:
    """One F15 per stratum, then F16, each built when the caller asks for it."""
    formats = log_formats(
        Reads(ctx.study.export)
    )  # once for the module; each figure hashes its own reads
    for stratum, name, label in stratum_variants(ctx, "F15"):
        built = f15(ctx, stratum, (name, label), Reads(ctx.study.export), formats)
        built.inputs |= group_inputs(ctx, stratum)
        yield built
    yield from grouped(ctx, f16, grid_fits)  # once per world group and page (plans/worlds/I8 §3.1)
