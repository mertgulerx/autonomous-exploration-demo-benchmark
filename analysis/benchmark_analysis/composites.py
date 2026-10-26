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

"""Composite figures (docs/plans/FIGURE_EXTENSION.md, FX5; FIGURE_STYLE §4, §9, §10).

- F21: the main-results composite, one row per stratum: the final runs by outcome (F6 (a)), coverage over time
  (F1), attainment of 90 % coverage (F2_R_90) and the per-block ΔR_90 against Default with the ``stats.json``
  estimate and interval (the strip of F13_dR_90, one per profile). Every part is drawn by the function that draws
  it in its standalone figure, so its marks, counts and notes are those of the part; the jitter of the ΔR_90
  strips is drawn from the same stream in the same order as in F13.
- F22_a, teaser per stratum: the representative Default run (the plan's rule, as F8) on its coverage reference,
  beside the coverage over time of Default and the first other profile (F1's panel, those two profiles).
- F22_b, teaser of one run: the representative Default run of the first stratum on its coverage reference with
  its goals placed in the world (the shared goal marks, ``goals.map_style``: filled when reached, open when not,
  a failed goal a filled cross, a close-enough cancel half filled; the explorer's cancel reason beside a goal it
  ended, log format 2), beside its coverage curve over the stratum's Default band (F1's pointwise median and
  IQR), the path and the curve coloured by time on one shared scale.

Both teasers are descriptive: the paired comparison is F3 and F13 (``stats.json``). F21 and F22_a are drawn once
per world group and page (``figures.grouped``, plans/worlds/I8 §3.1), a page no taller than the page height.
"""

from collections import Counter
from collections.abc import Iterator
from functools import partial
from typing import Any

from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import offset_copy
import numpy as np

from .comparisons import difference_strip, paired_notes, paired_record, paired_row
from .data import Row, Stratum, run_end
from .estimates import EFFECTS
from .figures import (
    AXIS_NOTE,
    GREY,
    TIME_LABEL,
    FigureContext,
    FigureOut,
    attainment_notes,
    attainment_panel,
    carried_handle,
    coverage_band,
    coverage_notes,
    coverage_panel,
    coverage_trace,
    finish,
    grouped,
    legend,
    map_panel,
    message_figure,
    path_norm,
    reference_files,
    reference_name,
    outcome_bars,
    outcome_handles,
    outcome_note,
    outcomes,
    paired_strata,
    panel_message,
    panel_title,
    representatives,
    study_profiles,
    time_axis,
    time_axis_end,
    time_bar,
    time_lines,
)
from .goals import (
    ENDED_BY_EXPLORER,
    MAP_KEYS,
    PlacedGoal,
    bar_handles,
    close_enough_text,
    map_handles,
    map_style,
    place,
    read_goals,
    reason_style,
)
from .streams import Reads, log_formats, world_from_map
from .stats import JITTER_STREAM, random_stream
from .style import PAGE_MAX_MM, PRESETS, SEQUENTIAL_CMAP, figure, page_note

# the attainment column and the paired effect of F21 (ΔR_90 is always reported)
MAIN_ENDPOINT = "R_90"
MAIN_EFFECT = next(e for e in EFFECTS if e.endpoint == MAIN_ENDPOINT)
ROW_HEIGHT_MM = 62.0  # one stratum: a panel and the strip or table under it
ROW_RATIOS = (3.0, 1.2)  # the panel over its strip or at-risk table
FRAME_MM = 22.0  # room for the legend and the evidence label
COLUMN_WIDTHS = (0.6, 1.35, 1.35, 0.85)  # outcomes, coverage, attainment, ΔR_90
# the clock named once per axis in the short form, its origin in the notes (FP-10; the full label of F1 and F2
# does not fit four columns)
TIME_LABEL_SHORT = "Simulated time (s)"
# characters per title line in the narrow columns: the at-risk table (8 pt), the ΔR_90 strip and the stratum (9 pt)
TABLE_TITLE_CHARS, STRIP_TITLE_CHARS, LABEL_TITLE_CHARS = 30, 22, 18
TEASER_ROW_MM = 60.0  # F22_a: one stratum, a map beside a coverage panel and its strip
TEASER_WIDTHS = (1.0, 1.3)  # F22_a and F22_b: map, coverage
TEASER_RUN_MM = 88.0  # F22_b: the run's map and its coverage curve
TEASER_RUN_WIDTHS = (1.15, 1.0)  # F22_b: map, coverage
REASON_OFFSET_PT = 4.0  # F22_b: a cancel reason is marked this far up and right of its goal


def _rows_height_mm(row_mm: float, rows: int) -> float:
    """The height of a composite of a row per stratum (F21, F22_a)."""
    return row_mm * rows + FRAME_MM


def _rows_fit(page: FigureContext, *, row_mm: float) -> bool:
    """The page test of F21 and F22_a: a row per stratum, no taller than a page."""
    return _rows_height_mm(row_mm, len(page.study.strata)) <= PAGE_MAX_MM


def _panel_over_strip(fig: Any, cell: Any) -> tuple[Any, Any]:
    """Replace the axes ``cell`` with a panel over its strip or table (``ROW_RATIOS``) in the same grid cell."""
    spec = cell.get_subplotspec()
    cell.remove()
    # a fresh spec of the same grid cell, as the figures were first built (layout round-off stays the same)
    nested = spec.get_gridspec()[spec.rowspan.start, spec.colspan.start].subgridspec(
        2, 1, height_ratios=ROW_RATIOS
    )
    return fig.add_subplot(nested[0]), fig.add_subplot(nested[1])


def f21(ctx: FigureContext) -> FigureOut:
    strata = ctx.study.strata
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    height_mm = _rows_height_mm(ROW_HEIGHT_MM, len(strata))
    # per stratum, its profiles with a Default run to pair with, in F13's row order
    paired = [[(p, b) for s, p, b in paired_strata(ctx) if s is stratum] for stratum in strata]
    slots = max([1, *(len(found) for found in paired)])  # ΔR_90 strips per row: one per profile
    widths = [*COLUMN_WIDTHS[:3], *[COLUMN_WIDTHS[3] / slots] * slots]
    fig, axes = figure(
        ctx.preset,
        "double",
        aspect=height_mm / width_mm,
        nrows=len(strata),
        ncols=len(widths),
        gridspec_kw={"width_ratios": widths},
    )
    # F13's jitter stream, drawn from in F13's row order
    rng = random_stream(ctx.settings.seed, JITTER_STREAM)
    attempts: set[str] = set()
    every_count: list[dict[str, int]] = []
    record: list[dict[str, Any]] = []
    entries: list[
        dict[str, Any]
    ] = []  # the ΔR_90 strips' stats.json entries, for the caption's methods
    index = 0
    for line, stratum in enumerate(strata):
        tall = [axes[line][0], *axes[line][3:]]  # outcomes and the ΔR_90 strips
        top: dict[int, Any] = {}
        bottom: dict[int, Any] = {}
        for column in (1, 2):  # coverage and attainment: a panel over its strip or table
            top[column], bottom[column] = _panel_over_strip(fig, axes[line][column])
        found = outcomes(ctx, stratum)
        attempts |= found.attempt_ids
        every_count += found.counts
        outcome_bars(tall[0], [ctx.styles[p].name for p in found.profiles], found.counts)
        tall[0].xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))  # a narrow column
        # the row's stratum; the axis says "Final runs"
        panel_title(tall[0], index, ctx.label(stratum), wrap=LABEL_TITLE_CHARS)
        attempts |= coverage_panel(top[1], bottom[1], ctx, stratum)
        bottom[1].set_xlabel(TIME_LABEL_SHORT)
        panel_title(top[1], index + 1, "coverage")
        attempts |= attainment_panel(
            top[2], bottom[2], ctx, stratum, MAIN_ENDPOINT, table_wrap=TABLE_TITLE_CHARS
        )
        top[2].set_xlabel(TIME_LABEL_SHORT)
        panel_title(top[2], index + 2, "attainment")  # the y label names the 90 %
        index += 3
        rows = []
        for strip, (profile, baseline) in zip(tall[1:], paired[line], strict=False):
            row = paired_row(ctx, MAIN_EFFECT, stratum, profile, baseline)
            entries.append(row.entry)
            attempts |= set(row.found.attempt_ids)
            lead = f"{ctx.styles[profile].name}, n pairs = {len(row.found.blocks)}"
            wrap = STRIP_TITLE_CHARS // slots
            difference_strip(strip, ctx, MAIN_EFFECT, row, rng, index, lead=(lead,), wrap=wrap)
            index += 1
            rows.append(paired_record(ctx, row))
        for strip in tall[1 + len(paired[line]) :]:
            strip.set_axis_off()
        if not paired[line]:
            panel_message(tall[1], "no profile to\ncompare with Default")
            panel_title(tall[1], index, MAIN_EFFECT.label)
            index += 1
        record.append(
            dict(
                stratum=ctx.label(stratum),
                axis_end_s=time_axis_end(stratum),
                outcomes={p: dict(c) for p, c in zip(found.profiles, found.counts, strict=True)},
                pairs=rows,
            )
        )
    handles = [carried_handle(), *outcome_handles(every_count)]
    legend(fig, ctx, study_profiles(ctx), handles)
    finish(fig, ctx)
    notes = [
        "Rows: strata. Columns: final runs by outcome (as F6 (a)); coverage over time (as F1); attainment of 90 % "
        "coverage (as F2_R_90); per-block ΔR_90 against Default (as the right column of F13_dR_90), one strip per "
        "profile. Each part is drawn by its standalone figure's code; time axes differ per row. Time: simulated "
        "time since first movement.",
        outcome_note(ctx.study),
        *coverage_notes(ctx),
        *attainment_notes(ctx),
        *paired_notes(ctx, MAIN_EFFECT, entries, scatter=False),
    ]
    notes = list(dict.fromkeys(notes))  # the parts share the FX-2 axis note
    if (overflow := page_note(height_mm)) is not None:
        notes.append(overflow)
    return FigureOut(
        "F21",
        fig,
        "double",
        attempts,
        "Main results per stratum: outcomes of the final runs, coverage over time, the share of runs that reached "
        "90 % coverage over time, and the paired difference in restricted time to 90 % coverage against Default.",
        notes,
        {"rows": record, "endpoint": MAIN_ENDPOINT, "effect": MAIN_EFFECT.name},
    )


# F22 teasers --------------------------------------------------------------------------------------------------


def default_representative(ctx: FigureContext, stratum: Stratum) -> tuple[Row, str] | None:
    """The stratum's representative Default run and its note: the first Default run the plan's rule picks (F8)."""
    baseline = ctx.study.baselines.get(stratum.robot_id)
    return next(
        (
            (run, note)
            for run, note in representatives(ctx, stratum)
            if run["profile_id"] == baseline
        ),
        None,
    )


def _teaser_profiles(ctx: FigureContext, stratum: Stratum) -> list[str]:
    """Default and the first other profile of the stratum, in study order (fewer when the stratum lacks one)."""
    baseline = ctx.study.baselines.get(stratum.robot_id)
    present = ctx.profiles(stratum)
    others = [p for p in present if p != baseline]
    return [p for p in present if p == baseline] + others[:1]


def f22a(ctx: FigureContext) -> FigureOut:
    strata = ctx.study.strata
    if not strata:
        return message_figure(ctx, "F22_a", "No strata: no teaser.")
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    height_mm = _rows_height_mm(TEASER_ROW_MM, len(strata))
    fig, axes = figure(
        ctx.preset,
        "double",
        aspect=height_mm / width_mm,
        nrows=len(strata),
        ncols=2,
        gridspec_kw={"width_ratios": TEASER_WIDTHS},
    )
    norm = path_norm(ctx)
    attempts: set[str] = set()
    record: list[dict[str, Any]] = []
    drawn: list[str] = []
    inputs = reference_files(ctx, None)
    for line, stratum in enumerate(strata):
        ax = axes[line][0]
        ax.set_axis_off()
        chosen = default_representative(ctx, stratum)
        if chosen is None:
            panel_message(ax, "no Default run with a\nrestricted time")
            panel_title(ax, 2 * line, ctx.label(stratum))
        else:
            run, note = chosen
            panel_title(ax, 2 * line, f"{ctx.label(stratum)} · {note}", wrap=LABEL_TITLE_CHARS + 12)
            map_panel(ax, ctx, run, stratum, norm)
            attempts.add(run["attempt_id"])
            inputs |= reference_files(ctx, reference_name(ctx, run["attempt_id"]))
        top, bottom = _panel_over_strip(fig, axes[line][1])
        profiles = _teaser_profiles(ctx, stratum)
        attempts |= coverage_panel(top, bottom, ctx, stratum, profiles)
        names = " and ".join(ctx.styles[p].name for p in profiles)
        panel_title(top, 2 * line + 1, f"coverage, {names}")
        drawn += [p for p in profiles if p not in drawn]
        record.append(
            dict(
                stratum=ctx.label(stratum),
                representative=None if chosen is None else chosen[0]["attempt_id"],
                note=None if chosen is None else chosen[1],
                profiles=profiles,
                axis_end_s=time_axis_end(stratum),
            )
        )
    time_bar(fig, [axes[line][0] for line in range(len(strata))], norm)
    legend(fig, ctx, [p for p in ctx.order if p in drawn], [carried_handle()])
    finish(fig, ctx)
    rule, endpoint = ctx.settings.representative_rule, ctx.settings.representative_endpoint
    notes = [
        f"Left: the representative Default run of each stratum ({rule} on {endpoint}, FIGURE_STYLE §8; as F8) on "
        "its coverage reference, the ground-truth path to min(T_end, L) coloured by time; arrow: start pose, "
        "square: end. Right: coverage over time of Default and the first other profile, the first by study order "
        "(not chosen for the figure; as F1). Descriptive: the paired comparison is F3/F13 (stats.json).",
        *coverage_notes(ctx),
    ]
    if (overflow := page_note(height_mm)) is not None:
        notes.append(overflow)
    return FigureOut(
        "F22_a",
        fig,
        "double",
        attempts,
        "Per stratum, the representative Default run's path on the coverage reference beside the coverage over "
        "time of Default and the first other profile.",
        notes,
        dict(rows=record, rule=rule, endpoint=endpoint),
        inputs=inputs,
    )


def _goal_record(placed: PlacedGoal) -> dict[str, Any]:
    """A placed goal as F22_b's sidecar holds it."""
    goal = placed.goal
    return dict(
        sequence=goal.sequence,
        dispatch_s=goal.dispatch_s,
        end_s=goal.end_s,
        status=goal.status,
        cancel_reason=goal.reason,
        world_x=placed.x,
        world_y=placed.y,
    )


def _goal_marks(ax: Any, placed: tuple[PlacedGoal, ...], end_s: float, reasons: bool) -> set[str]:
    """Goal points by map key (``goals.map_style``) and, with ``reasons``, the explorer's cancel reason up and right
    of each goal it ended within [0, ``end_s``] (F15's rule); returns the reason keys drawn (``goals.bar_handles``)."""
    for key in MAP_KEYS:
        mine = [g for g in placed if g.key == key]
        if mine:
            ax.plot([g.x for g in mine], [g.y for g in mine], zorder=4, **map_style(key))
    used: set[str] = set()
    if not reasons:
        return used
    shifted = offset_copy(
        ax.transData, fig=ax.figure, x=REASON_OFFSET_PT, y=REASON_OFFSET_PT, units="points"
    )
    for found in placed:
        goal, ended = found.goal, found.goal.end_s
        if goal.status in ENDED_BY_EXPLORER and ended is not None and 0.0 <= ended <= end_s:
            ax.plot(
                [found.x], [found.y], transform=shifted, zorder=4.5, **reason_style(goal.reason)
            )
            used.add(f"reason:{goal.reason or 'none'}")
    return used


def _time_curve(ax: Any, pieces: list[tuple[Any, Any]], norm: Normalize) -> list[Any]:
    """A run's coverage steps as one line coloured by time on ``norm`` (``time_lines``, as the paths); returns
    the step vertices drawn, one array per piece."""
    segments: list[Any] = []
    times: list[float] = []
    vertices = []
    for steps_t, steps_c in pieces:
        xs = np.repeat(steps_t, 2)[1:]
        ys = np.repeat(steps_c, 2)[:-1]
        points = np.column_stack([xs, ys])
        vertices.append(points)
        segments += list(np.stack([points[:-1], points[1:]], axis=1))
        times += list((xs[:-1] + xs[1:]) / 2)
    ax.add_collection(time_lines(segments, np.array(times), norm))
    return vertices


def _reasons_text(log_format: int | None) -> str | None:
    """Why a run shows no cancel reasons (the plan's rule for streams a run did not record), or None."""
    if log_format is not None and log_format >= 2:
        return None
    return (
        "not recorded" if log_format is None else f"explorer log format {log_format}: not recorded"
    )


def f22b(ctx: FigureContext) -> FigureOut:
    if not ctx.study.strata:
        return message_figure(ctx, "F22_b", "No strata: no teaser run.")
    stratum = ctx.study.strata[0]
    chosen = default_representative(ctx, stratum)
    if chosen is None:
        return message_figure(
            ctx,
            "F22_b",
            f"No Default run with a restricted time in {ctx.label(stratum)}: no teaser run.",
        )
    run, note = chosen
    attempt, limit = str(run["attempt_id"]), stratum.limit_s
    end_run, _ = run_end(run, limit)
    reads = Reads(ctx.study.export)
    log_format = log_formats(reads).get(attempt)
    no_reasons = _reasons_text(log_format)
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    fig, axes = figure(
        ctx.preset,
        "double",
        aspect=TEASER_RUN_MM / width_mm,
        ncols=2,
        gridspec_kw={"width_ratios": TEASER_RUN_WIDTHS},
    )
    ax, curve = axes[0]
    ax.set_axis_off()
    norm = path_norm(ctx)
    panel_title(ax, 0, f"{ctx.label(stratum)} · {note}")
    map_panel(ax, ctx, run, stratum, norm)
    record = read_goals(reads, [run])[attempt]
    alignment = world_from_map(reads, attempt) if record.goals else None
    placed, unplaced = place(record.goals, alignment, end_run)
    rejected = record.rejections.counted
    reasons_drawn = _goal_marks(ax, placed, end_run, no_reasons is None)
    if no_reasons is not None:
        ax.text(
            0.0,
            -0.02,
            f"cancel reasons: {no_reasons}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            color=GREY,
        )
    # (b) the run's coverage over the band of the stratum's Default runs, time-coloured as its path
    baseline = run["profile_id"]
    style = ctx.styles[baseline]
    band_runs = stratum.runs_of(baseline)
    band = coverage_band(curve, ctx, band_runs, style)
    if not band.drawn:  # F1's rule: the band only where at least min_n runs have a value
        curve.text(
            0.98,
            0.04,
            f"no band: {band.runs_with_values} Default runs with values, fewer than {ctx.settings.min_n}",
            transform=curve.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=GREY,
        )
    trace = coverage_trace(ctx, run, stratum)
    vertices = _time_curve(curve, trace.pieces, norm)
    if trace.carried is not None:
        start, value = trace.carried
        curve.plot([start, limit], [value, value], color=GREY, lw=1.0, ls="--", zorder=2.5)
    if trace.held is not None:
        curve.plot([0, limit], [trace.held] * 2, color=GREY, lw=1.0, ls=":", zorder=2.5)
    end = time_axis(curve, stratum, l_text="bottom")
    curve.set(ylim=(0, 100), ylabel="Coverage (%)", xlabel=TIME_LABEL)
    title = "coverage of this run" + (f" over the {style.name} band" if band.drawn else "")
    panel_title(curve, 1, title)
    time_bar(fig, [ax, curve], norm)
    run_colour = plt.get_cmap(SEQUENTIAL_CMAP)(0.3)
    handles: list[Any] = [
        Line2D([], [], color=run_colour, lw=1.0, label="this run (coloured by time)")
    ]
    if band.drawn:
        handles += [
            Patch(
                facecolor=style.color,
                alpha=0.18,
                label=f"{style.name}: pointwise IQR (descriptive)",
            ),
            Line2D(
                [], [], color=style.color, lw=1.2, ls=style.linestyle, label=f"{style.name}: median"
            ),
        ]
    if band.moved_median:
        handles.append(
            Line2D(
                [],
                [],
                color=style.color,
                lw=0.8,
                ls=":",
                label=f"{style.name}: median, runs that moved",
            )
        )
    if trace.held is not None:
        handles.append(
            Line2D([], [], color=GREY, lw=1.0, ls=":", label="never moved: pre-motion coverage")
        )
    handles += [
        *map_handles(
            Counter(g.key for g in placed), close_enough_text(ctx.study.export, [baseline])
        ),
        *bar_handles(reasons_drawn),
    ]
    if trace.carried is not None:
        handles.append(Line2D([], [], color=GREY, lw=1.0, ls="--", label="carried forward to L"))
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    finish(fig, ctx)
    remarks = []
    if no_reasons is not None:
        remarks.append(f"Cancel reasons: {no_reasons}.")
    if unplaced:
        remarks.append(
            f"{sum(unplaced.values())} goals without a world position are not drawn (sidecar)."
        )
    if rejected is None:
        remarks.append("Goals Nav2 rejected: navigation telemetry not recorded.")
    else:
        remarks.append(f"Goals Nav2 rejected: {rejected} (never active, so not on the map).")
    if trace.held is not None:
        remarks.append(
            "This run never moved: its pre-motion coverage is held from 0 to L (dotted, RD-8)."
        )
    rule, endpoint = ctx.settings.representative_rule, ctx.settings.representative_endpoint
    notes = [
        f"The representative Default run of {ctx.label(stratum)} ({rule} on {endpoint}, FIGURE_STYLE §8; as F8). "
        "Left: its coverage reference, its ground-truth path to min(T_end, L) coloured by time (arrow: start pose, "
        "square: end) and its goals dispatched by the run's end, placed in the world with the run's frozen "
        "world_from_map: shape by status; filled: reached (and failed as a filled cross); half filled: ended by "
        "the explorer's close-enough guard (cancel reason completion_distance: the robot came within that "
        "distance); beside a goal the explorer ended, its cancel reason (log format 2; the marks of F15). "
        "Descriptive: one run picked by rule, not a comparison (the paired comparison is F3/F13, stats.json). "
        "Right: the run's coverage steps (as "
        "F1, broken at gaps over 2 s) on the same time colours; dashed: a completed or stuck run's final coverage "
        "carried to L (RD-6); dotted at a constant: a run that never moved, held at its pre-motion coverage (RD-8). Band: "
        "the pointwise median and IQR (type 2) of the stratum's Default runs, descriptive, only where at least "
        f"{ctx.settings.min_n} runs have a value under the gap rule; a dotted median: runs that moved only (RD-8, "
        f"secondary). {AXIS_NOTE}",
        *remarks,
    ]
    return FigureOut(
        "F22_b",
        fig,
        "double",
        {attempt} | {r["attempt_id"] for r in band_runs},
        f"One representative Default run of {ctx.label(stratum)}: its path and goals on the map, and its "
        "coverage over time against the band of the stratum's Default runs, coloured by time.",
        notes,
        dict(
            stratum=ctx.label(stratum),
            attempt_id=attempt,
            note=note,
            rule=rule,
            endpoint=endpoint,
            log_format=log_format,
            band_runs=band.runs_with_values,
            band_drawn=band.drawn,
            goals=[_goal_record(g) for g in placed],
            goals_unplaced=dict(sorted(unplaced.items())),
            goals_rejected=rejected,
            curve_vertices=[v.tolist() for v in vertices],
            axis_end_s=end,
        ),
        inputs=reads.files | reference_files(ctx, reference_name(ctx, attempt)),
    )


def figures(ctx: FigureContext) -> Iterator[FigureOut]:
    """F21, F22_a and F22_b, each built when the caller asks for it."""
    # once per world group and page, a row per stratum, no taller than a page (plans/worlds/I8 §3.1)
    yield from grouped(ctx, f21, partial(_rows_fit, row_mm=ROW_HEIGHT_MM))
    yield from grouped(ctx, f22a, partial(_rows_fit, row_mm=TEASER_ROW_MM))
    yield f22b(ctx)
