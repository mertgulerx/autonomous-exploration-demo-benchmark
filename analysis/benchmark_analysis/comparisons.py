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

"""Figures F13 and F14: each profile against Default, block by block (docs/plans/FIGURE_EXTENSION.md, FX2;
FIGURE_STYLE §4, §5; METHODOLOGY §3).

- F13 (one per paired effect of ``estimates.EFFECTS``): the estimation plot of FX-13. Per stratum and profile,
  the pairs as a scatter (Default on x, the profile on y) with the identity line, beside the per-block
  differences with the ``stats.json`` estimate and interval of the effect's primary estimator. For restricted
  times, a pair with a value censored at L (not reached by L, or no movement) is an open mark, counted; nothing is
  drawn as observed that was not.
- F14: the paired gap at equal coverage. Per block, Δt(c) = t_P(c) − t_D(c) on the 5 pp grid of METHODOLOGY §3
  (t(c) the first valid sample at or above c, no interpolation), drawn only where both runs reached c; a strip
  counts for each c the blocks where both, only Default, only the profile or neither reached it; the pointwise
  median over "both" only where at least ``min_n`` blocks contribute, labelled conditional with its share. No
  band, no estimand: the values are descriptive display quantities, written to the sidecar and never cited.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
import math
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from . import stats
from .data import REPORTED, TIME_EPSILON, Row, Stratum, Value, run_end
from .estimates import EFFECTS, Effect, Pairs, horizon_value, pairs, time_value
from .figures import (
    GREY,
    FigureContext,
    FigureOut,
    estimator_text,
    finish,
    interval_level,
    interval_methods,
    legend,
    message_figure,
    paired_strata,
    panel_message,
    panel_title,
)
from .style import EDGE, PRESETS, SUPPORT_FILLS, figure

TITLE_MM = 14.0  # room above the panels for titles and below for the legend
ROW_HEIGHT_MM = 52.0  # one F13 row: scatter and differences
EQUAL_SPAN = (
    1.0  # F13: the y half-range when every difference is the same (all pairs censored at L, say)
)
GRID_PP = 5.0  # METHODOLOGY §3: t(c) on c = 5, 10, …, 100
GAP_COLUMNS, GAP_ROW_HEIGHT_MM = 3, 62.0  # F14's panel grid
LEFT_OUT_MM = 4.0  # F14: a row's extra height for the "left out" line under its strip
SUPPORT = (  # F14's support classes in stacking order: key, legend text
    ("both", "both reached c"),
    ("default", "only Default"),
    ("profile", "only the profile"),
    ("neither", "neither (incl. no movement)"),
)
NO_PAIRS = "No profile to compare with Default in any stratum."
LEFT_OUT = {  # F14's blocks left out, by reason (``Gaps.left_out``), as the strip's label words them
    "not_both_reported": "not both reported",
    "no_coverage_value": "no coverage value",
}


# F13 paired estimation plot ----------------------------------------------------------------------------------


def _reader(ctx: FigureContext, effect: Effect, stratum: Stratum) -> Callable[[Row], Value]:
    limit = stratum.limit_s
    if effect.endpoint is not None:
        endpoint = effect.endpoint
        return lambda run: time_value(run, endpoint, limit)
    estimand = effect.estimand.removeprefix("delta_")
    return lambda run: horizon_value(ctx.study, run, estimand, limit, limit)


def _censored(
    read: Callable[[Row], Value], stratum: Stratum, profile: str, baseline: str
) -> dict[str, int]:
    """Per block, how many of its two runs hold a value censored at L (``event`` false: a restricted time not
    reached by L); horizon values have no event and are never censored."""
    return {
        block: sum(1 for r in (run, partner) if r is not None and read(r).event is False)
        for block, run, partner in stratum.blocks(profile, baseline)
    }


def _marks(ax: Any, xs: Any, ys: Any, hollow: list[bool], style: Any, size: float) -> None:
    for x, y, empty in zip(xs, ys, hollow, strict=True):
        ax.scatter(
            [x],
            [y],
            s=size,
            marker=style.marker,
            facecolors="none" if empty else style.color,
            edgecolors=style.color,
            linewidths=0.8,
            zorder=3,
        )


def _scatter(
    ax: Any,
    ctx: FigureContext,
    profile: str,
    mine: list[float],
    theirs: list[float],
    hollow: list[bool],
) -> None:
    values = [*mine, *theirs]
    low, high = (min(values), max(values)) if values else (0.0, 1.0)
    pad = 0.06 * (high - low or abs(high) or 1.0)
    bounds = (low - pad, high + pad)
    ax.plot(bounds, bounds, color=GREY, lw=0.6, ls="--", zorder=1)
    _marks(ax, theirs, mine, hollow, ctx.styles[profile], 14)
    ax.set(xlim=bounds, ylim=bounds)
    ax.set_aspect("equal", adjustable="box")


def _differences(
    ax: Any,
    ctx: FigureContext,
    profile: str,
    differences: list[float],
    hollow: list[bool],
    entry: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """The differences (jittered), the estimate and its interval; the achieved level under "estimate" when fewer
    pairs allow less than the requested one."""
    _marks(
        ax, rng.uniform(-0.12, 0.12, len(differences)), differences, hollow, ctx.styles[profile], 12
    )
    ax.axhline(0, color=GREY, lw=0.6)
    estimate, interval = entry.get("estimate"), entry.get("interval") or {}
    if interval.get("lower") is not None and interval.get("upper") is not None:
        ax.plot([0.55, 0.55], [interval["lower"], interval["upper"]], color="#000000", lw=1.2)
    if estimate is not None:
        ax.plot([0.55], [estimate], marker="s", color="#000000", ms=4)
    ax.set_xlim(-0.4, 0.8)
    ax.ticklabel_format(axis="y", useOffset=False)
    # every difference equal: a fixed span instead of matplotlib's offset notation around one value
    if differences and max(differences) - min(differences) < TIME_EPSILON:
        ax.set_ylim(differences[0] - EQUAL_SPAN, differences[0] + EQUAL_SPAN)
    level = interval_level(entry, 1 - ctx.settings.alpha)
    ax.set_xticks([0, 0.55], ["pairs", "estimate" + (f"\n({level})" if level else "")], fontsize=8)


@dataclass(frozen=True, slots=True)
class PairedRow:
    """One stratum and profile against Default for one effect (F13, F21): the reported pairs, which of them hold
    a value censored at L, and the effect's ``stats.json`` entry."""

    stratum: Stratum
    profile: str
    found: Pairs
    hollow: list[bool]  # per pair in ``found.blocks``
    entry: dict[str, Any]


def paired_row(
    ctx: FigureContext, effect: Effect, stratum: Stratum, profile: str, baseline: str
) -> PairedRow:
    read = _reader(ctx, effect, stratum)
    found = pairs(stratum, profile, baseline, "primary", read)
    censored = _censored(read, stratum, profile, baseline)
    entry = ctx.entries.get(
        effect.estimand,
        stratum,
        profile,
        endpoint=effect.endpoint,
        horizon_s=None if effect.endpoint else stratum.limit_s,
        variant=effect.estimator,
    )
    return PairedRow(
        stratum, profile, found, [censored[block] > 0 for block in found.blocks], entry
    )


def difference_strip(
    ax: Any,
    ctx: FigureContext,
    effect: Effect,
    row: PairedRow,
    rng: np.random.Generator,
    index: int,
    *,
    lead: tuple[str, ...] = (),
    wrap: int | None = None,
) -> None:
    """The per-block differences of one row with the estimate and its interval, titled with ``lead``, then the
    censored and missing pairs (F13, F21), broken into lines of about ``wrap`` characters when given. ``rng`` is
    the figure's jitter stream, drawn from in row order."""
    found = row.found
    _differences(ax, ctx, row.profile, found.differences, row.hollow, row.entry, rng)
    ax.set_ylabel(f"{effect.label}, profile − Default")
    parts = [*lead]
    if effect.endpoint is not None:  # censored in both: stats.json's own count
        both = row.entry.get("pairs_censored_in_both", 0)
        parts.append(f"open: censored at L ({sum(row.hollow)}, both {both})")
    parts.append(f"no pair {sum(found.missing.values())}")
    panel_title(ax, index, "; ".join(parts), wrap=wrap)
    if not found.blocks:
        panel_message(ax, "no pairs", boxed=True)


def paired_record(ctx: FigureContext, row: PairedRow) -> dict[str, Any]:
    """A row's sidecar record (F13, F21)."""
    found = row.found
    return dict(
        stratum=ctx.label(row.stratum),
        profile_id=row.profile,
        entry_id=row.entry.get("id"),
        blocks=found.blocks,
        differences=found.differences,
        censored_blocks=[b for b, empty in zip(found.blocks, row.hollow, strict=True) if empty],
        missing_pairs=found.missing,
    )


def paired_notes(
    ctx: FigureContext,
    effect: Effect,
    entries: Iterable[Mapping[str, Any]],
    *,
    scatter: bool = True,
) -> list[str]:
    """How to read the pairs and differences of one effect (F13; F21 without the scatter); ``entries`` are the
    ``stats.json`` entries of the rows drawn, whose interval methods the caption names."""
    strip = (
        "the per-block differences profile − Default (jittered); square and line: the "
        f"{estimator_text(effect)} estimate and its {ctx.level_label} interval from stats.json (F3, T5; a level "
        'in parentheses under "estimate" when fewer pairs allow less), by '
        f"{interval_methods(entries)} (stats.json interval.method). "
        '"No pair": blocks without two reported runs with a value (stats.json missing_pairs).'
    )
    notes = [
        "Left: one mark per block, Default on x, the profile on y, dashed identity line (marks above it: a "
        f"larger value with the profile). Right: {strip}"
        if scatter
        else strip[0].upper() + strip[1:]
    ]
    if effect.endpoint is not None:
        notes.append(
            "Open marks: a pair with a value censored at L (not reached by L, no movement as never reaching; "
            "R = L), counted in the title with the pairs censored in both (stats.json pairs_censored_in_both)."
        )
    return notes


def f13(ctx: FigureContext, effect: Effect) -> FigureOut:
    rows = paired_strata(ctx)
    name = f"F13_{effect.name}"
    if not rows:
        return message_figure(ctx, name, NO_PAIRS)
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    fig, axes = figure(
        ctx.preset,
        "double",
        aspect=(ROW_HEIGHT_MM * len(rows) + TITLE_MM) / width_mm,
        nrows=len(rows),
        ncols=2,
        gridspec_kw={"width_ratios": [1, 1.4]},
    )
    rng = stats.random_stream(ctx.settings.seed, stats.JITTER_STREAM)
    attempts: set[str] = set()
    record = []
    entries = []
    for index, (stratum, profile, baseline) in enumerate(rows):
        row = paired_row(ctx, effect, stratum, profile, baseline)
        entries.append(row.entry)
        found = row.found
        attempts |= set(found.attempt_ids)
        scatter = axes[index][0]
        name_p = ctx.styles[profile].name
        panel_title(
            scatter, 2 * index, f"{ctx.label(stratum)} · {name_p}, n pairs = {len(found.blocks)}"
        )
        _scatter(scatter, ctx, profile, found.profile_values, found.baseline_values, row.hollow)
        scatter.set_xlabel(f"Default {effect.value_label}")
        scatter.set_ylabel(f"{name_p} {effect.value_label}")
        difference_strip(axes[index][1], ctx, effect, row, rng, 2 * index + 1)
        record.append(paired_record(ctx, row))
    finish(fig, ctx)
    return FigureOut(
        name,
        fig,
        "double",
        attempts,
        f"Paired values and differences {effect.label} of each profile against Default per stratum, with the "
        "estimate and its interval.",
        paired_notes(ctx, effect, entries),
        {"rows": record},
    )


# F14 paired gap at equal coverage ----------------------------------------------------------------------------


def reach_times(
    ctx: FigureContext, run: Row, limit: float, grid: list[float]
) -> list[float | None] | None:
    """t(c) for each c of the grid (METHODOLOGY §3): the first valid sample at or above c up to the run's end
    within L; None where not reached. A run that never moved reaches no c, whatever its coverage status (as R_m,
    RD-8); a run that moved has no t(c) (None) when it is not coverage-eligible or has no samples."""
    if run["outcome"] == "no_movement":
        return [None] * len(grid)
    if not run["coverage_eligible"]:
        return None
    series = ctx.study.coverage.get(run["attempt_id"])
    if series is None or not series.t.size:
        return None
    end, _ = run_end(run, limit)
    keep = series.t <= end + TIME_EPSILON
    t, c = series.t[keep], series.coverage[keep]
    found: list[float | None] = []
    for level in grid:
        reached = np.nonzero(c >= level - TIME_EPSILON)[0]
        found.append(float(t[reached[0]]) if reached.size else None)
    return found


@dataclass(frozen=True, slots=True)
class Gaps:
    """One stratum and profile against Default on the grid."""

    deltas: dict[str, list[float | None]]  # block -> Δt(c), None where not both reached c
    support: dict[str, list[int]]  # SUPPORT class -> blocks per c
    left_out: dict[str, int] = field(default_factory=dict)  # blocks not drawn, by reason
    attempt_ids: list[str] = field(default_factory=list)

    def share_both(self) -> list[float | None]:
        n = len(self.deltas)
        return [None if not n else both / n for both in self.support["both"]]


def _gaps(
    ctx: FigureContext, stratum: Stratum, profile: str, baseline: str, grid: list[float]
) -> Gaps:
    """Per block Δt(c) where both reached c, the support counts per c, and the blocks left out."""
    deltas: dict[str, list[float | None]] = {}
    support = {key: [0] * len(grid) for key, _ in SUPPORT}
    left_out: Counter[str] = Counter()
    attempts: list[str] = []
    for block, run, partner in stratum.blocks(profile, baseline):
        if (
            run is None
            or partner is None
            or run["report_class"] != REPORTED
            or partner["report_class"] != REPORTED
        ):
            left_out["not_both_reported"] += 1
            continue
        mine = reach_times(ctx, run, stratum.limit_s, grid)
        theirs = reach_times(ctx, partner, stratum.limit_s, grid)
        if mine is None or theirs is None:
            left_out["no_coverage_value"] += 1
            continue
        attempts += [run["attempt_id"], partner["attempt_id"]]
        row: list[float | None] = []
        for i, (a, b) in enumerate(zip(mine, theirs, strict=True)):
            if a is not None and b is not None:
                key = "both"
            elif a is not None:
                key = "profile"
            else:
                key = "default" if b is not None else "neither"
            support[key][i] += 1
            row.append(a - b if a is not None and b is not None else None)
        deltas[block] = row
    return Gaps(deltas, support, dict(sorted(left_out.items())), sorted(attempts))


def _medians(ctx: FigureContext, gaps: Gaps, size: int) -> list[float | None]:
    """The pointwise median of Δt(c) over the blocks where both reached c, only from ``min_n`` of them."""
    medians: list[float | None] = []
    for i in range(size):
        values = [v for row in gaps.deltas.values() if (v := row[i]) is not None]
        enough = gaps.support["both"][i] >= ctx.settings.min_n
        medians.append(float(np.median(values)) if enough else None)
    return medians


def _gap_panel(
    ax: Any, below: Any, ctx: FigureContext, profile: str, grid: list[float], gaps: Gaps
) -> list[float | None]:
    style = ctx.styles[profile]
    x = np.array(grid)
    for row in gaps.deltas.values():
        y = np.array([np.nan if v is None else v for v in row])
        ax.plot(x, y, color=style.color, lw=0.5, alpha=0.45, marker=".", ms=1.5)
    medians = _medians(ctx, gaps, len(grid))
    y = np.array([np.nan if m is None else m for m in medians])
    ax.plot(
        x, y, color="#000000", lw=1.4, marker="o", ms=2.0
    )  # a marker shows a median at one c alone
    drawn = [i for i, m in enumerate(medians) if m is not None]
    if drawn:  # the median's share of blocks, where it ends (the label FX-20 asks for)
        last = drawn[-1]
        ax.annotate(
            f"both: {gaps.support['both'][last]}/{len(gaps.deltas)}",
            xy=(grid[last], medians[last]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_ylabel("Δt(c), profile − Default (s)")
    ax.tick_params(labelbottom=False)
    bottom = np.zeros(len(grid))
    for key, _ in SUPPORT:
        face, hatch = SUPPORT_FILLS[key]
        counts = np.array(gaps.support[key])
        below.bar(
            x,
            counts,
            bottom=bottom,
            width=GRID_PP * 0.8,
            color=face,
            hatch=hatch,
            edgecolor=EDGE,
            lw=0.3,
        )
        bottom += counts
    below.set_ylabel(f"Blocks (n = {len(gaps.deltas)})")
    left_out = ", ".join(f"{v} {LEFT_OUT[k]}" for k, v in gaps.left_out.items())
    below.set_xlabel("Coverage threshold c (%)" + (f"\nleft out: {left_out}" if left_out else ""))
    if not gaps.deltas:
        panel_message(ax, "no pairs", boxed=True)
    return medians


def f14(ctx: FigureContext) -> FigureOut:
    rows = paired_strata(ctx)
    if not rows:
        return message_figure(ctx, "F14", NO_PAIRS)
    grid = [GRID_PP * k for k in range(1, int(100 / GRID_PP) + 1)]
    columns = min(GAP_COLUMNS, len(rows))
    lines = math.ceil(len(rows) / columns)
    width = "double" if len(rows) > 1 else "single"
    width_mm = PRESETS[ctx.preset].widths_mm[width]
    found = [_gaps(ctx, stratum, profile, baseline, grid) for stratum, profile, baseline in rows]
    row_mm = GAP_ROW_HEIGHT_MM + (LEFT_OUT_MM if any(g.left_out for g in found) else 0.0)
    fig, axes = figure(
        ctx.preset,
        width,
        aspect=(row_mm * lines + TITLE_MM) / width_mm,
        nrows=2 * lines,
        ncols=columns,
        gridspec_kw={"height_ratios": [3, 1] * lines},
    )
    attempts: set[str] = set()
    record = []
    for index, ((stratum, profile, _), gaps) in enumerate(zip(rows, found, strict=True)):
        line, column = divmod(index, columns)
        ax, below = axes[2 * line][column], axes[2 * line + 1][column]
        attempts |= set(gaps.attempt_ids)
        # the axis ends one step after the last c any drawn block reached
        reached = [i for i in range(len(grid)) if gaps.support["neither"][i] < len(gaps.deltas)]
        top = grid[max(reached)] + GRID_PP if reached else 100.0
        medians = _gap_panel(ax, below, ctx, profile, grid, gaps)
        for axis in (ax, below):
            axis.set_xlim(0, min(100 + GRID_PP, top + GRID_PP / 2))
        panel_title(ax, index, f"{ctx.label(stratum)} · {ctx.styles[profile].name}")
        record.append(
            dict(
                stratum=ctx.label(stratum),
                profile_id=profile,
                grid_pp=grid,
                delta_t_s=gaps.deltas,
                support=gaps.support,
                median_over_both_s=medians,
                share_both=gaps.share_both(),
                left_out=gaps.left_out,
            )
        )
    for empty in range(len(rows), lines * columns):
        line, column = divmod(empty, columns)
        axes[2 * line][column].set_axis_off()
        axes[2 * line + 1][column].set_axis_off()
    handles: list[Any] = [
        Patch(
            facecolor=SUPPORT_FILLS[key][0],
            hatch=SUPPORT_FILLS[key][1],
            edgecolor=EDGE,
            label=text,
        )
        for key, text in SUPPORT
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="#000000",
            lw=1.4,
            marker="o",
            ms=2.0,
            label="median over both (conditional)",
        )
    )
    legend(fig, ctx, [p for p in ctx.order if any(p == r[1] for r in rows)], handles)
    finish(fig, ctx, descriptive=True)
    return FigureOut(
        "F14",
        fig,
        width,
        attempts,
        "Per-block difference in the time to reach each coverage threshold, profile minus Default, with the "
        "number of blocks in which both, one or neither run reached it.",
        [
            "Descriptive display quantity (METHODOLOGY §3), not an estimand and not cited: t(c) is the first "
            f"valid coverage sample at or above c on a {GRID_PP:g} pp grid, no interpolation, unreached by L or "
            "no movement as not reached. Thin lines: blocks, drawn only where both runs reached c. Thick: the "
            f"pointwise median over those blocks where at least {ctx.settings.min_n} contribute, conditional on "
            'both reaching c; "both: k/n" at its last point gives that share (every c in the sidecar, '
            "share_both). Bars: blocks by who reached c. Blocks without two reported runs, or with a run that "
            "moved but has no coverage value, are left out, counted under the bars and in the sidecar.",
        ],
        {"rows": record},
    )


def figures(ctx: FigureContext) -> Iterator[FigureOut]:
    """F13 per paired effect, then F14, each built when the caller asks for it."""
    for effect in EFFECTS:
        yield f13(ctx, effect)
    yield f14(ctx)
