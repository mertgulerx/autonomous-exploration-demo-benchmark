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

"""Figures F9–F12: how the study ran (docs/plans/FIGURE_EXTENSION.md, FX1; FIGURE_STYLE §4–§6, §9).

- F9: the run accounting as a flow per profile (FX-14): planned experiments, attempts, final runs, reported runs
  by outcome, with what leaves the flow at each stage in a side box; the counts are T1's (``tables.accounting``),
  summed over strata. Above 3 profiles or 1 robot the figure is a pointer to T1 (Pocock 2007 rec. 11).
- F10: the trial strip (FX-15): one row per robot, world and spawn, the blocks in execution order (first attempt
  start), the profiles of a block adjacent in study order; a cell's fill is its final run's outcome (FP-4), its
  text R_completion or "> L", glyphs mark low RTF, parallel execution and replaced attempts.
- F11: timing validity: ECDFs of the worst 5-s RTF and the mean RTF of the reported runs per profile, one row per
  execution mode (T8's populations), and each run's wall seconds below 0.90.
- F12: the world atlas: each world's coverage reference at one scale (millimetres per metre the same in every
  panel), its spawn poses, a 5 m bar and the reference's free area, cells and L, ordered by free area.

Every figure here names the export files it reads in ``FigureOut.inputs``.
"""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import math
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from .data import (
    EXCLUDED,
    MISSING,
    REPORTED,
    REPORTED_OUTCOMES,
    Row,
    Stratum,
    block_starts,
    restricted_time,
)
from .figures import (
    GREY,
    REFERENCE_PAD_M,
    Extent,
    FigureContext,
    FigureOut,
    draw_reference,
    finish,
    legend,
    load_reference,
    message_figure,
    panel_message,
    panel_title,
    reference_files,
    reference_meta,
    reference_name,
)
from .style import EDGE, OUTCOMES, PAGE_MAX_MM, PRESETS, RTF_LINES, figure, page_note
from .tables import DISCARD_TRIGGERS, accounting

CHAR_MM = 1.5  # an upper bound on one 7 pt character's width, to wrap side-box lines
LABEL_CHAR_MM = 1.7  # the same for 8 pt row labels


def _ink(outcome: str) -> str:
    """Text and glyph colour on an outcome fill: white only on the dark completed grey."""
    return "#FFFFFF" if outcome == "completed" else "#000000"


def _date(ns: int | None) -> str:
    return "—" if ns is None else datetime.fromtimestamp(ns / 1e9, UTC).strftime("%Y-%m-%d")


def _seconds(values: list[float]) -> str:
    return "/".join(f"{v:g}" for v in values) or "—"


def _limits(ctx: FigureContext) -> str:
    worlds: dict[str, set[float]] = defaultdict(set)
    for stratum in ctx.study.strata:
        worlds[stratum.world_id].add(stratum.limit_s)
    return ", ".join(
        f"{world} {_seconds(sorted(values))} s" for world, values in sorted(worlds.items())
    )


# F9 run accounting flow ---------------------------------------------------------------------------------------

FLOW_MAX_PROFILES = 3  # FX-14: above this (or more than one robot) the flow is T1's table
FLOW_HEIGHT_MM = 92.0
# Geometry in axis units: one column per profile (width 1), the stages from 6.2 at the top down to 0.
FLOW_ROWS = {"planned": 5.0, "attempts": 3.9, "final": 2.8, "reported": 1.7}
FLOW_SIDES = {"not_run": 4.45, "not_final": 3.35, "not_reported": 2.25}
FLOW_TITLES = {"not_run": "Not run", "not_final": "Not final", "not_reported": "Not reported"}
MAIN_X, MAIN_W, SIDE_X, SIDE_W, BOX_H = 0.2, 0.34, 0.68, 0.58, 0.5
OUTCOME_Y, OUTCOME_W, OUTCOME_H, OUTCOME_X0, OUTCOME_STEP = 0.62, 0.3, 0.5, 0.17, 0.33
TEXT_LINE = 0.17  # one 7 pt line
HEADER_MARK_Y, HEADER_NAME_Y = 6.0, 5.72

SideItem = tuple[str, int, bool]  # label, count, whether it adds to the box's total


def _final(counts: dict[str, int]) -> int:
    return sum(counts.get(c, 0) for c in (REPORTED, *MISSING, *EXCLUDED))


def _flow(
    ctx: FigureContext, counts: dict[tuple[str, str, str, str], dict[str, int]]
) -> dict[str, dict[str, int]]:
    """T1's counts summed over strata per profile, with every attempt (``attempts``)."""
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (_robot, _world, _spawn, profile), group in counts.items():
        for name, value in group.items():
            totals[profile][name] += value
    for run in ctx.study.runs:
        totals[run["profile_id"]]["attempts"] += 1
    return {p: dict(c) for p, c in totals.items()}


def _side(counts: dict[str, int]) -> dict[str, list[SideItem]]:
    """What leaves the flow at each stage, with T1's names; only non-zero items. An unresolved experiment (a
    timing rejection without a replacement) has a final run, ``missing_platform``: it is named there, not added."""
    leaving = [
        ("replaced (timing)", counts.get("replaced_timing_rejection", 0)),
        ("replaced (interruption)", counts.get("replaced_engine_interruption", 0)),
        *(
            (
                f"discarded ({name.replace('experiment_restart', 'restart').replace('_', ' ')})",
                counts.get(f"discarded_{name}", 0),
            )
            for name in (*DISCARD_TRIGGERS, "other")
        ),
    ]
    # attempts neither final, replaced nor discarded: one waiting or running in an unfinished campaign
    other = counts.get("attempts", 0) - _final(counts) - sum(v for _, v in leaving)
    planned, done, unresolved = (counts.get(k, 0) for k in ("planned", "done", "unresolved"))
    groups: dict[str, list[SideItem]] = {
        "not_run": [("not done", planned - done - unresolved, True)],  # T1's Not done
        "not_final": [*((label, v, True) for label, v in leaving), ("other", other, True)],
        "not_reported": [
            *(
                (OUTCOMES[c]["label"].lower(), counts.get(c, 0), True)
                for c in (*MISSING, *EXCLUDED)
            ),
            ("of these unresolved", unresolved, False),
        ],
    }
    return {stage: [item for item in items if item[1]] for stage, items in groups.items()}


def _flow_box(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    outcome: str | None = None,
    size: float = 8,
) -> None:
    look = OUTCOMES[outcome] if outcome else dict(facecolor="#FFFFFF", hatch=None)
    ax.add_patch(
        Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            facecolor=look["facecolor"],
            hatch=look["hatch"],
            edgecolor=EDGE,
            lw=0.6,
        )
    )
    colour = _ink(outcome) if outcome else "#000000"
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=colour, linespacing=1.2)


def _arrow(ax: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=EDGE, shrinkA=0, shrinkB=0),
    )


def _wrap(label: str, value: int, chars: int) -> str:
    text = f"{label} {value}"
    if len(text) <= chars or " (" not in label:
        return text
    head, _, tail = label.partition(" (")
    return f"{head}\n({tail} {value}"


def _pointer(ctx: FigureContext, profiles: int, robots: int) -> FigureOut:
    built = message_figure(
        ctx,
        "F9",
        f"{profiles} profiles, {robots} robot{'s' if robots > 1 else ''}: the run accounting is T1\n"
        "(a flow diagram above 3 profiles or 1 robot is clearer as a table)",
        attempts={r["attempt_id"] for r in ctx.study.runs},
        inputs={"experiments.parquet"},
    )
    return replace(
        built,
        alt_text="Pointer to table T1, which holds the run accounting of this study.",
        notes=["FX-14: above 3 profiles or 1 robot the accounting flow is given as table T1."],
    )


def _flow_column(
    ax: Any, ctx: FigureContext, column: int, profile: str, counts: dict[str, int], chars: int
) -> None:
    style = ctx.styles.get(profile)
    main_x, side_x = column + MAIN_X, column + SIDE_X
    if style is not None:
        ax.plot([column + 0.5], [HEADER_MARK_Y], marker=style.marker, color=style.color, ms=4)
    ax.text(
        column + 0.5,
        HEADER_NAME_Y,
        style.name if style else profile,
        ha="center",
        va="center",
        fontsize=8,
        color=style.color if style else "#000000",
    )
    values = {
        "planned": ("Planned", counts.get("planned", 0)),
        "attempts": ("Attempts", counts.get("attempts", 0)),
        "final": ("Final runs", _final(counts)),
        "reported": ("Reported", counts.get(REPORTED, 0)),
    }
    for stage, (label, value) in values.items():
        _flow_box(ax, main_x, FLOW_ROWS[stage], MAIN_W, BOX_H, f"{label}\n{value}")
    stages = list(FLOW_ROWS)
    for upper, lower in zip(stages[:-1], stages[1:], strict=True):
        _arrow(ax, (main_x, FLOW_ROWS[upper] - BOX_H / 2), (main_x, FLOW_ROWS[lower] + BOX_H / 2))
    for stage, items in _side(counts).items():
        lines = [f"{FLOW_TITLES[stage]} {sum(v for _, v, counted in items if counted)}"]
        lines += [_wrap(label, value, chars) for label, value, _ in items]
        text = "\n".join(lines)
        height = TEXT_LINE * (text.count("\n") + 1) + 0.1
        _flow_box(ax, side_x, FLOW_SIDES[stage], SIDE_W, height, text, size=7)
        _arrow(ax, (main_x, FLOW_SIDES[stage]), (side_x - SIDE_W / 2, FLOW_SIDES[stage]))
    for slot, outcome in enumerate(REPORTED_OUTCOMES):
        x = column + OUTCOME_X0 + OUTCOME_STEP * slot
        text = f"{OUTCOMES[outcome]['label']}\n{counts.get('outcome_' + outcome, 0)}"
        _flow_box(ax, x, OUTCOME_Y, OUTCOME_W, OUTCOME_H, text, outcome=outcome, size=7)
        _arrow(
            ax,
            (main_x, FLOW_ROWS["reported"] - BOX_H / 2),
            (x, OUTCOME_Y + OUTCOME_H / 2),
        )


def f9(ctx: FigureContext) -> FigureOut:
    counts = accounting(ctx.study)
    totals = _flow(ctx, counts)
    robots = len({key[0] for key in counts})
    profiles = [p for p in ctx.order if p in totals] + sorted(set(totals) - set(ctx.order))
    if len(profiles) > FLOW_MAX_PROFILES or robots > 1:
        return _pointer(ctx, len(profiles), robots)
    width = "single" if len(profiles) == 1 else "double"
    width_mm = PRESETS[ctx.preset].widths_mm[width]
    fig, axes = figure(ctx.preset, width, aspect=FLOW_HEIGHT_MM / width_mm)
    ax = axes[0][0]
    ax.set_axis_off()
    ax.set(xlim=(0, len(profiles)), ylim=(-0.05, 6.2))
    chars = int(SIDE_W * width_mm / len(profiles) / CHAR_MM)
    for column, profile in enumerate(profiles):
        _flow_column(ax, ctx, column, profile, totals[profile], chars)
    starts = [r["wall_start_ns"] for r in ctx.study.runs if r["wall_start_ns"]]
    ends = [r["wall_end_ns"] for r in ctx.study.runs if r["wall_end_ns"]]
    ax.text(
        len(profiles) / 2,
        -0.03,
        f"Runs {_date(min(starts, default=None))} to {_date(max(ends, default=None))} (UTC) · "
        f"L: {_limits(ctx)}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=GREY,
    )
    finish(fig, ctx)
    strata = len({key[:3] for key in counts})
    return FigureOut(
        "F9",
        fig,
        width,
        {r["attempt_id"] for r in ctx.study.runs},
        f"Run accounting per profile, summed over {strata} robot, world and spawn combinations: planned "
        "experiments, attempts, final runs and reported runs by outcome, with what left the flow at each stage.",
        [
            "Counts as in T1 (per stratum there), summed over strata; the run IDs of each class are in T1's "
            "sources (runs, experiments). Not run: planned experiments without a final run (T1 Not done). Not "
            "final: attempts replaced (a later attempt of the experiment exists), discarded, or other (still "
            "waiting or running in an unfinished campaign). Not reported: final runs outside "
            "reported-outcomes-v2, missing observations and exclusions; an unresolved experiment (a timing "
            "rejection without a replacement) is missing (platform) (FX-14).",
        ],
        {"flow": {p: dict(sorted(totals[p].items())) for p in profiles}},
        inputs={"experiments.parquet"},
    )


# F10 trial strip ----------------------------------------------------------------------------------------------

CELL_MM, CELL_SMALL_MM, CELL_H_MM, BLOCK_GAP_MM = 8.0, 5.0, 5.5, 1.5
LABEL_MIN_MM, LABEL_MAX_MM = 30.0, 60.0
NUMBER_MM, MARKER_MM = 2.8, 3.4  # block numbers above a line of cells, profile markers below
LINE_MM = NUMBER_MM + CELL_H_MM + MARKER_MM
LEGEND_MM = 14.0
GLYPHS = {  # name: marker, legend text
    "low_rtf": ("v", "low RTF"),
    "parallel": ("+", "parallel execution"),
    "replaced": ("x", "replaced attempt"),
}


@dataclass(frozen=True, slots=True)
class StripRow:
    label: str
    stratum: Stratum | None
    blocks: list[str]  # in execution order
    profiles: list[str]  # in study order
    planned: dict[str, dict[str, Row]]  # block -> profile -> experiment


def _strip_rows(ctx: FigureContext) -> list[StripRow]:
    study = ctx.study
    experiments = study.export.table(
        "experiments",
        [
            "block_id",
            "robot_id",
            "world_id",
            "spawn_id",
            "profile_id",
            "final_attempt_id",
            "n_replaced",
        ],
    ).to_pylist()
    started = block_starts(study)
    cells: dict[tuple[str, str, str], dict[str, dict[str, Row]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in experiments:
        cell = (row["robot_id"], row["world_id"], row["spawn_id"])
        cells[cell][row["block_id"]][row["profile_id"]] = row
    stratum_of = {(s.robot_id, s.world_id, s.spawn_id): s for s in study.strata}
    order = {p: i for i, p in enumerate(ctx.order)}
    rows = []
    for cell, by_block in sorted(cells.items()):

        def first_start(block: str, cell: tuple[str, str, str] = cell) -> tuple[bool, int, str]:
            start = started.get((*cell, block))
            return start is None, start or 0, block  # blocks never started last

        profiles = {p for planned in by_block.values() for p in planned}
        rows.append(
            StripRow(
                study.label(*cell),
                stratum_of.get(cell),
                sorted(by_block, key=first_start),
                sorted(profiles, key=lambda p: (order.get(p, len(order)), p)),
                dict(by_block),
            )
        )
    return rows


def _strip_layout(
    rows: list[StripRow], width_mm: float, cell: float, label: float
) -> tuple[int, float]:
    """Blocks per line, the same for every row so that columns align, and the content height in mm."""
    widest = max((len(r.profiles) for r in rows), default=1)
    per_line = max(1, int((width_mm - label + BLOCK_GAP_MM) // (widest * cell + BLOCK_GAP_MM)))
    height = sum(max(1, math.ceil(len(r.blocks) / per_line)) for r in rows) * LINE_MM
    return per_line, height


CellTime = tuple[str, float | None, bool | None]  # the cell's text, R_completion, observed by L


def _cell_time(run: Row | None, stratum: Stratum | None) -> CellTime | None:
    """R_completion of a reported run as the cell's text, with its value and whether it was observed by L."""
    if run is None or stratum is None or run["report_class"] != REPORTED:
        return None
    value = restricted_time(run, "R_completion", stratum.limit_s)
    if value.value is None:
        return "–", None, None
    return (f"{value.value:.0f}" if value.event else "> L"), value.value, value.event


def _strip_cell(
    ax: Any,
    x: float,
    top: float,
    width: float,
    planned: Row,
    run: Row | None,
    text: str | None,
) -> tuple[str, set[str]]:
    """One experiment's cell; returns the fill's outcome key and the glyphs drawn."""
    outcome = "not_done"
    if run is not None:
        outcome = run["outcome"] if run["report_class"] == REPORTED else run["report_class"]
    look = OUTCOMES.get(outcome, OUTCOMES["not_done"])
    ax.add_patch(
        Rectangle(
            (x + 0.25, top),
            width - 0.5,
            CELL_H_MM,
            facecolor=look["facecolor"],
            hatch=look["hatch"],
            edgecolor=EDGE,
            lw=0.4,
        )
    )
    ink = _ink(outcome)
    if text is not None:
        ax.text(
            x + width / 2,
            top + CELL_H_MM / 2 + 0.3,
            text,
            ha="center",
            va="center",
            fontsize=7,
            color=ink,
        )
    present = {
        "low_rtf": (run is not None and bool(run["low_rtf"]), x + 1.2, top + 1.0),
        "parallel": (
            run is not None and (run["concurrent_wall_fraction"] or 0) > 0,
            x + width - 1.2,
            top + 1.0,
        ),
        "replaced": (bool(planned["n_replaced"]), x + 1.2, top + CELL_H_MM - 0.9),
    }
    drawn = set()
    for name, (shown, mx, my) in present.items():
        if shown:
            drawn.add(name)
            ax.plot([mx], [my], marker=GLYPHS[name][0], color=ink, ms=2.8, mew=0.8, ls="none")
    return outcome, drawn


def f10(ctx: FigureContext) -> FigureOut:
    rows = _strip_rows(ctx)
    runs = {r["attempt_id"]: r for r in ctx.study.runs}
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    longest = max((len(r.label) for r in rows), default=0)
    label_mm = min(LABEL_MAX_MM, max(LABEL_MIN_MM, longest * LABEL_CHAR_MM + 2))
    cell_w, with_text = CELL_MM, True
    per_line, height = _strip_layout(rows, width_mm, cell_w, label_mm)
    if height > PAGE_MAX_MM - LEGEND_MM:
        cell_w, with_text = CELL_SMALL_MM, False
        per_line, height = _strip_layout(rows, width_mm, cell_w, label_mm)
    total = height + 4
    fig, axes = figure(ctx.preset, "double", aspect=(total + LEGEND_MM) / width_mm)
    ax = axes[0][0]
    ax.set_axis_off()
    ax.set(xlim=(0, width_mm), ylim=(total, 0))
    used: set[str] = set()
    shown: set[str] = set()
    glyphs: set[str] = set()
    times: dict[str, dict[str, Any]] = {}
    y = 0.0
    for row in rows:
        ax.text(0, y + NUMBER_MM + CELL_H_MM / 2, row.label, ha="left", va="center", fontsize=8)
        for index, block in enumerate(row.blocks):
            line, slot = divmod(index, per_line)
            top = y + line * LINE_MM + NUMBER_MM
            left = label_mm + slot * (len(row.profiles) * cell_w + BLOCK_GAP_MM)
            ax.text(left, top - 0.4, str(index + 1), ha="left", va="bottom", fontsize=7, color=GREY)
            for position, profile in enumerate(row.profiles):
                planned = row.planned[block].get(profile)
                if planned is None:  # the block holds no experiment of this profile
                    continue
                final = planned["final_attempt_id"]
                run = runs.get(final) if final else None
                time = _cell_time(run, row.stratum)
                if run is not None:
                    used.add(run["attempt_id"])
                    if time is not None:
                        times[run["attempt_id"]] = dict(R_completion_s=time[1], event=time[2])
                x = left + position * cell_w
                text = time[0] if time is not None and with_text else None
                outcome, drawn = _strip_cell(ax, x, top, cell_w, planned, run, text)
                shown.add(outcome)
                glyphs |= drawn
                style = ctx.styles.get(profile)
                if style is not None:
                    ax.plot(
                        [x + cell_w / 2],
                        [top + CELL_H_MM + 1.6],
                        marker=style.marker,
                        color=style.color,
                        ms=2.4,
                        ls="none",
                    )
        y += max(1, math.ceil(len(row.blocks) / per_line)) * LINE_MM
    handles: list[Any] = [
        Patch(facecolor=v["facecolor"], hatch=v["hatch"], edgecolor=EDGE, label=v["label"])
        for k, v in OUTCOMES.items()
        if k in shown
    ]
    handles += [
        Line2D([], [], marker=marker, color="#000000", ls="none", ms=4, mew=0.8, label=label)
        for name, (marker, label) in GLYPHS.items()
        if name in glyphs
    ]
    legend(fig, ctx, [p for p in ctx.order if any(p in r.profiles for r in rows)], handles)
    finish(fig, ctx)
    notes = [
        "Columns: blocks in the order their first attempt started (numbered), the profiles of a block adjacent "
        "in study order (marker below each cell). Fill: the final run's outcome or report class (FP-4); an empty "
        "cell has no final run. "
        + (
            'Text: R_completion in simulated seconds, "> L" when not completed by L. '
            if with_text
            else "Times are left out at this size; the sidecar holds each run's R_completion (`times`). "
        )
        + "Glyphs: ▼ low RTF, + the run overlapped another attempt, × the experiment had a replaced attempt."
    ]
    tall = page_note(total + LEGEND_MM)  # the figure's height
    if tall is not None:
        notes.append(tall)
    return FigureOut(
        "F10",
        fig,
        "double",
        used,
        "Every planned experiment as a cell, one row per robot, world and spawn, blocks in execution order, filled "
        "by the final run's outcome.",
        notes,
        {"block_order": {r.label: r.blocks for r in rows}, "times": times},
        inputs={"experiments.parquet"},
    )


# F11 timing validity ------------------------------------------------------------------------------------------

RTF_METRICS = (
    ("rtf_min_5s", "Worst 5-s RTF"),
    ("rtf_mean", "Mean RTF"),
    ("wall_s_below_090", "Wall seconds below RTF 0.90"),
)


def _mode_members(ctx: FigureContext, profiles: list[str], mode: str) -> dict[str, list[Row]]:
    """Reported runs per profile in T8's execution modes (a run without the fraction is in neither)."""
    return {
        p: [
            r
            for s in ctx.study.strata
            for r in s.runs_of(p)
            if r["concurrent_wall_fraction"] is not None
            and (r["concurrent_wall_fraction"] == 0) == (mode == "sequential")
        ]
        for p in profiles
    }


def f11(ctx: FigureContext) -> FigureOut:
    fig, axes = figure(ctx.preset, "double", aspect=0.5, nrows=2, ncols=3)
    profiles = [p for p in ctx.order if any(s.runs_of(p) for s in ctx.study.strata)]
    reported = [r for s in ctx.study.strata for r in s.reported]
    ratios = [float(r[n]) for r in reported for n in ("rtf_min_5s", "rtf_mean") if r[n] is not None]
    walls = [float(r["wall_s_below_090"]) for r in reported if r["wall_s_below_090"] is not None]
    wall_top = max(1.0, 1.1 * max(walls, default=0.0))
    low = min(0.85, min(ratios, default=0.85) - 0.01)
    high = max(1.01, max(ratios, default=1.0) + 0.005)
    used: set[str] = set()
    for row, mode in enumerate(("sequential", "parallel")):
        members = _mode_members(ctx, profiles, mode)
        for column, (name, label) in enumerate(RTF_METRICS):
            ax = axes[row][column]
            values = {
                p: sorted(float(r[name]) for r in members[p] if r[name] is not None)
                for p in profiles
            }
            used |= {r["attempt_id"] for p in profiles for r in members[p] if r[name] is not None}
            counts = " / ".join(str(len(values[p])) for p in profiles)
            panel_title(ax, row * len(RTF_METRICS) + column, f"{mode}, n = {counts}")
            for position, profile in enumerate(profiles):
                style, points = ctx.styles[profile], np.array(values[profile])
                if not points.size:
                    continue
                if column < 2:
                    xs = np.concatenate([[low], points, [high]])
                    ys = np.concatenate([[0.0], np.arange(1, points.size + 1) / points.size, [1.0]])
                    ax.step(xs, ys, where="post", color=style.color, ls=style.linestyle, lw=1.0)
                else:
                    ax.scatter(
                        points,
                        np.full(points.size, -position),
                        s=10,
                        color=style.color,
                        marker=style.marker,
                        alpha=0.8,
                        linewidths=0,
                    )
            if column < 2:
                for x, dash, _ in RTF_LINES:
                    ax.axvline(x, color=GREY, lw=0.6, ls=dash)
                ax.set(xlim=(low, high), ylim=(0, 1.02), xlabel=label, ylabel="Share of runs")
            else:
                ax.set_yticks(
                    [-i for i in range(len(profiles))],
                    [ctx.styles[p].name for p in profiles],
                    fontsize=8,
                )
                ax.set_ylim(-len(profiles) + 0.5, 0.5)
                ax.set_xlabel(label)
                # a little room left of 0, so that runs with no time below 0.90 stay off the spine
                ax.set_xlim(-0.04 * wall_top, wall_top)
            if not any(values.values()):
                panel_message(ax, f"no {mode} runs with a value", boxed=True)
    lines = [
        Line2D([], [], color=GREY, ls=dash, lw=0.6, label=f"{value:.2f} {text}")
        for value, dash, text in RTF_LINES
    ]
    legend(fig, ctx, profiles, lines)
    finish(fig, ctx)
    rejected = sum(1 for r in ctx.study.runs if r["outcome"] == "timing_rejection")
    return FigureOut(
        "F11",
        fig,
        "double",
        used,
        "Distribution of the worst 5-s and mean real-time factor of the reported runs per profile and execution "
        "mode, with each run's wall seconds below 0.90.",
        [
            "Steps: ECDFs over the reported runs of all strata (T8 per stratum); n per panel counts the runs with "
            "a value. Sequential: no other attempt ran at any time (concurrent_wall_fraction = 0); parallel: "
            "otherwise; a run without the fraction is in neither. Dashed: the suite's rejection level 0.90 "
            "(10 wall s, timing-rule-v2); dotted: the 0.95 low_rtf flag. The population is cut at the rule it "
            "shows: the suite rejects an attempt below RTF 0.90 for 10 wall s, so such attempts are not reported "
            f"runs and are absent by design ({rejected} rejected for timing, T1); the low tail is not drawn.",
        ],
        {"rejected_for_timing": rejected},
    )


# F12 world atlas ----------------------------------------------------------------------------------------------

ATLAS_GAP_MM, ATLAS_LABEL_MM, ATLAS_MIN_SLOT_MM, ATLAS_MAX_H_MM = 5.0, 14.0, 36.0, 110.0
ATLAS_FOOT_MM = 8.0  # kept free under the panels for the evidence label
ATLAS_MIN_SCALE = 0.2  # mm per metre: below it the 5 m bar is 1 mm and the atlas no longer reads
SPAWN_ARROW_M = 2.0

Pose = tuple[float, float, float | None]


@dataclass(frozen=True, slots=True)
class AtlasPanel:
    world: str
    reference: str
    area_m2: float | None
    cells: int | None
    codes: np.ndarray[Any, Any]
    extent: Extent
    limits: list[float]
    spawns: dict[str, tuple[str, Pose]]  # spawn -> (attempt, pose)

    @property
    def size_m(self) -> tuple[float, float]:
        """The map with its margins (``draw_reference``)."""
        pad = 2 * REFERENCE_PAD_M
        return self.extent[1] - self.extent[0] + pad, self.extent[3] - self.extent[2] + pad


def _spawn_poses(ctx: FigureContext) -> dict[str, Pose]:
    """Each run's first ground-truth pose after explorer activation: before movement, so its spawn pose. Rows
    before movement have no ``t_s``, so they are found by ``t_activation_s``."""
    table = ctx.study.export.table(
        "trajectory_gt", ["attempt_id", "t_activation_s", "gt_x_m", "gt_y_m", "gt_yaw_rad"]
    )
    first: dict[str, tuple[float, Pose]] = {}
    columns = (table.column(i).to_pylist() for i in range(5))
    for attempt, t, x, y, yaw in zip(*columns, strict=True):
        if t is None or x is None or y is None:
            continue
        if attempt not in first or t < first[attempt][0]:
            first[attempt] = (t, (x, y, yaw))
    return {attempt: pose for attempt, (_, pose) in first.items()}


def _atlas_panels(ctx: FigureContext) -> tuple[list[AtlasPanel], list[str]]:
    """One panel per world: the reference of its earliest final run, spawn poses from each spawn's earliest
    final run with ground truth; ordered by free area, worlds without it last. Also the worlds whose reference
    ``load_reference`` cannot read."""
    study = ctx.study
    poses = _spawn_poses(ctx)
    finals = sorted(
        (r for r in study.runs if r["is_final"]),
        key=lambda r: (r["wall_start_ns"] or 0, r["attempt_id"]),
    )
    chosen: dict[str, tuple[str, str]] = {}  # world -> reference, its first run
    spawns: dict[str, dict[str, tuple[str, Pose]]] = defaultdict(dict)
    for run in finals:
        name = reference_name(ctx, run["attempt_id"])
        if name is None or chosen.setdefault(run["world_id"], (name, run["attempt_id"]))[0] != name:
            continue
        pose = poses.get(run["attempt_id"])
        if pose is not None and run["spawn_id"] not in spawns[run["world_id"]]:
            spawns[run["world_id"]][run["spawn_id"]] = (run["attempt_id"], pose)
    panels, unreadable = [], []
    for world, (name, attempt) in chosen.items():
        loaded = load_reference(ctx, attempt)
        if loaded is None:
            unreadable.append(world)
            continue
        meta = reference_meta(ctx, name)
        area, cells = meta.get("target_area_m2"), meta.get("target_cells")
        panels.append(
            AtlasPanel(
                world,
                name,
                None if area is None else float(area),
                None if cells is None else int(cells),
                *loaded,
                sorted({s.limit_s for s in study.strata if s.world_id == world}),
                dict(spawns[world]),
            )
        )
    ordered = sorted(panels, key=lambda p: (p.area_m2 is None, p.area_m2 or 0.0, p.world))
    return ordered, sorted(unreadable)


def _atlas_rows(
    panels: list[AtlasPanel], width_mm: float
) -> tuple[list[list[int]], list[float], float, float]:
    """Rows of panels packed left to right at one scale (mm per metre), shrunk until the figure fits the page
    or the scale reaches its floor; returns the rows, their heights, the scale and the total height in mm."""
    sizes = [p.size_m for p in panels]
    scale = min(
        ATLAS_MAX_H_MM / max((h for _, h in sizes), default=1.0),
        width_mm / max((w for w, _ in sizes), default=1.0),
    )
    while True:
        rows: list[list[int]] = [[]]
        used = 0.0
        for i, (w, _) in enumerate(sizes):
            slot = max(w * scale, ATLAS_MIN_SLOT_MM)
            if rows[-1] and used + ATLAS_GAP_MM + slot > width_mm:
                rows.append([])
                used = 0.0
            used += (ATLAS_GAP_MM if rows[-1] else 0.0) + slot
            rows[-1].append(i)
        heights = [
            max((sizes[i][1] * scale for i in row), default=0.0) + ATLAS_LABEL_MM for row in rows
        ]
        total = sum(heights) + ATLAS_FOOT_MM
        if total <= PAGE_MAX_MM or scale <= ATLAS_MIN_SCALE:
            return rows, heights, scale, total
        scale = max(ATLAS_MIN_SCALE, scale * PAGE_MAX_MM / total)


def _atlas_panel(ax: Any, index: int, panel: AtlasPanel) -> None:
    ax.set_axis_off()
    draw_reference(ax, panel.codes, panel.extent)
    for spawn, (_, (x, y, yaw)) in sorted(panel.spawns.items()):
        ax.plot([x], [y], marker="o", mfc="#FFFFFF", mec="#000000", mew=0.8, ms=3.5, zorder=4)
        if yaw is not None:
            tip = (x + SPAWN_ARROW_M * math.cos(yaw), y + SPAWN_ARROW_M * math.sin(yaw))
            ax.annotate(
                "",
                xy=tip,
                xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="#000000", lw=0.8),
                zorder=4,
            )
        if len(panel.spawns) > 1:
            ax.text(x, y - 0.6, spawn, fontsize=7, ha="center", va="top")
    area = "—" if panel.area_m2 is None else f"{panel.area_m2:.0f}"
    cells = "—" if panel.cells is None else f"{panel.cells:,}"
    panel_title(
        ax, index, f"{panel.world}\n{area} m², {cells} cells\nL = {_seconds(panel.limits)} s"
    )


def f12(ctx: FigureContext) -> FigureOut:
    panels, unreadable = _atlas_panels(ctx)
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    rows, heights, scale, total = _atlas_rows(panels, width_mm)
    # The panels are placed in millimetres with add_axes, which the constrained layout leaves alone; the empty
    # grid axis keeps the engine from warning (a figure without one gets the engine back from rcParams on save).
    fig, axes = figure(ctx.preset, "double", aspect=max(total, 30.0) / width_mm)
    axes[0][0].set_axis_off()
    if not panels:
        panel_message(axes[0][0], "no coverage references exported")
    top = total
    for row, row_height in zip(rows, heights, strict=True):
        left = 0.0
        top -= row_height
        for i in row:
            panel = panels[i]
            w, h = panel.size_m
            slot = max(w * scale, ATLAS_MIN_SLOT_MM)
            box = (
                (left + (slot - w * scale) / 2) / width_mm,
                (top + row_height - ATLAS_LABEL_MM - h * scale) / total,
                w * scale / width_mm,
                h * scale / total,
            )
            _atlas_panel(fig.add_axes(box), i, panel)
            left += slot + ATLAS_GAP_MM
    finish(fig, ctx)
    mm_per_m = round(scale, 4)
    notes = [
        f"All panels at {scale:.3g} mm per metre (5 m bar). Free floor counted by the coverage metric white "
        "(target cells, metric version in T2), other inside cells light grey, obstacles dark grey, outside the "
        "reference hatched (FP-5). Dot and arrow: the spawn pose, the first ground-truth pose after explorer "
        "activation of the earliest final run of each spawn. Ordered by free area until world descriptors are "
        "integrated."
    ]
    tall = page_note(total)
    if tall is not None:
        notes.append(tall)
    if unreadable:
        notes.append(
            "Coverage reference not readable (a rotated origin), not drawn: "
            f"{', '.join(unreadable)}."
        )
    # reference_name read provenance.json for every final run, whether or not a panel was drawn
    files = reference_files(ctx, None).union(*(reference_files(ctx, p.reference) for p in panels))
    return FigureOut(
        "F12",
        fig,
        "double",
        {attempt for panel in panels for attempt, _ in panel.spawns.values()},
        f"Coverage reference maps of the {len(panels)} worlds at one scale with their spawn poses, ordered by "
        "reference free area.",
        notes,
        {
            "atlas": [
                dict(
                    world=p.world,
                    reference=p.reference,
                    target_area_m2=p.area_m2,
                    target_cells=p.cells,
                    limits_s=p.limits,
                )
                for p in panels
            ],
            "mm_per_m": mm_per_m,
        },
        inputs=files,
    )


def figures(ctx: FigureContext) -> Iterator[FigureOut]:
    """F9–F12, each built when the caller asks for it (it saves and closes one before the next is drawn)."""
    yield f9(ctx)
    yield f10(ctx)
    yield f11(ctx)
    yield f12(ctx)
