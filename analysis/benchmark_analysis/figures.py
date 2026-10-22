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

"""Figures F1–F3, F5, F6 and F8 (RESULTS_OVERHAUL §7; docs/research/FIGURE_STYLE.md §4–§9).

- F1: coverage per run as steps (broken at sample gaps over 2 s), a completed run's carry-forward to L dashed,
  a run that never moved dotted at its pre-motion coverage, a pointwise median with its IQR band only where at
  least ``min_n`` runs have a value under the gap rule (``coverage_grid``), the movement-only median as a
  secondary dotted line when some run never moved (RD-8), a strip of running runs, L marked.
- F2 (one per restricted-time endpoint): attainment 1 − Ŝ(t) as upward steps, Wilson intervals at the plan's
  horizons (from ``stats.json``), the at-risk table whose last column is "k/n not reached by L" (FP-3).
- F3 (one per paired effect of ``estimates.EFFECTS``): forest plot, profile − Default, with "n pairs" and
  "estimate [lo, hi]" columns, and a pooled diamond for a pre-specified pooled hypothesis on the same scale.
- F5: compute, F6: reliability; every run as a point, a box from ``min_n`` runs and a violin from 30 (type 2
  quartiles, Tukey whiskers), runs not reaching by L as triangles at L, success as a dot with its interval.
- F8: the representative runs of each stratum (the plan's rule: SEL-1, SEL-2 or SEL-3, FIGURE_STYLE §8) on the
  coverage reference the run used, coloured by time on one shared scale, with start pose, end and 5 m scale.

Every figure is drawn at its final width, names the clock on its time axes (FP-10), states the plan's interval
level, and is labelled with its evidence level unless the study is confirmatory.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from PIL import Image
import yaml

from . import stats
from .data import (
    GAP_S,
    Row,
    Stratum,
    Study,
    Value,
    idle_tail,
    restricted_time,
    run_end,
)
from .estimates import (
    EFFECTS,
    METRICS,
    PRIMARY_ESTIMATOR,
    Effect,
    Entries,
    time_endpoints,
    time_value,
)
from .settings import Settings
from .style import (
    MAP_FREE,
    MAP_OCCUPIED,
    MAP_OUTSIDE,
    MAP_UNKNOWN,
    MM_PER_INCH,
    OUTCOMES,
    PRESETS,
    ProfileStyle,
    figure,
    profile_styles,
)
from .tables import level_text, with_interval

TIME_LABEL = "Simulated time since first movement (s)"
GREY = "#7F7F7F"
TRACE_WIDTH = 0.5  # per-run traces: thin, above the 0.4 pt floor (FIGURE_STYLE §2)
VIOLIN_MIN = 30  # FIGURE_STYLE §5: violins and densities only from 30 points
PANEL_ASPECT = 0.42  # height of one row of panels per unit of width, multi-panel figures
SINGLE_ASPECT = 0.85  # height per width of a one-stratum figure
STRIP_RATIO = 1.0  # height of a strip or at-risk table under a panel, relative to 4 for the panel
LABEL_STRIP_IN = 0.14  # inches kept free at the bottom for the evidence label
SCALE_BAR_M = 5.0


@dataclass
class FigureOut:
    name: str
    figure: Figure
    width: str
    attempt_ids: set[str]
    alt_text: str
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FigureContext:
    study: Study
    settings: Settings
    entries: Entries
    preset: str
    grid: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str | None]]]
    styles: dict[str, ProfileStyle]
    order: list[str]

    def profiles(self, stratum: Stratum) -> list[str]:
        present = set(stratum.profiles())
        return [p for p in self.order if p in present]

    def label(self, stratum: Stratum) -> str:
        return self.study.stratum_label(stratum)

    @property
    def level_label(self) -> str:
        return level_text(1 - self.settings.alpha)


def context(
    study: Study,
    settings: Settings,
    entries: Entries,
    grid_rows: Iterable[tuple[str, float, float | None, str | None]],
    preset: str,
    order: list[str],
) -> FigureContext:
    collected: dict[str, list[tuple[float, float | None, str | None]]] = defaultdict(list)
    for attempt, t, value, source in grid_rows:
        collected[attempt].append((t, value, source))
    grid = {
        attempt: (
            np.array([t for t, _, _ in rows]),
            np.array([np.nan if v is None else v for _, v, _ in rows]),
            [s for _, _, s in rows],
        )
        for attempt, rows in collected.items()
    }
    return FigureContext(
        study, settings, entries, preset, grid, profile_styles(study.profiles), order
    )


# Shared layout -------------------------------------------------------------------------------------------------


def _grid_size(count: int) -> tuple[int, int]:
    columns = min(3, count)
    return math.ceil(count / columns), columns


def _panels(ctx: FigureContext, *, strip: bool) -> tuple[Figure, list[tuple[Any, Any]], str]:
    """One panel per stratum (and a strip under each when asked), blank cells hidden."""
    strata = ctx.study.strata
    rows, columns = _grid_size(len(strata))
    width = "double" if len(strata) > 1 else "single"
    aspect = PANEL_ASPECT * rows if len(strata) > 1 else SINGLE_ASPECT
    if strip:
        fig, axes = figure(
            ctx.preset,
            width,
            aspect=aspect,
            nrows=2 * rows,
            ncols=columns,
            gridspec_kw={"height_ratios": [4, STRIP_RATIO] * rows},
        )
        cells = [(axes[2 * r][c], axes[2 * r + 1][c]) for r in range(rows) for c in range(columns)]
    else:
        fig, axes = figure(ctx.preset, width, aspect=aspect, nrows=rows, ncols=columns)
        cells = [(axes[r][c], None) for r in range(rows) for c in range(columns)]
    for main, below in cells[len(strata) :]:
        main.set_axis_off()
        if below is not None:
            below.set_axis_off()
    return fig, cells[: len(strata)], width


def _finish(fig: Figure, ctx: FigureContext) -> None:
    """Label a non-confirmatory figure in a strip the layout keeps free (FIGURE_STYLE §11 item 17)."""
    if ctx.settings.confirmatory:
        return
    share = LABEL_STRIP_IN / fig.get_size_inches()[1]
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0, share, 1, 1 - share))  # type: ignore[call-arg]
    text = f"{ctx.settings.evidence_level} · {ctx.study.export.study_id}"
    fig.text(0.995, 0.004, text, ha="right", va="bottom", fontsize=7, color=GREY)


def _title(ax: Any, index: int, text: str) -> None:
    """The panel label in bold, then the panel's subject, at the top left (FIGURE_STYLE §9)."""
    ax.set_title(rf"$\mathbf{{({chr(ord('a') + index)})}}$ {text}", loc="left")


def _legend(
    fig: Figure, ctx: FigureContext, profiles: Sequence[str], extra: Sequence[Any] = ()
) -> None:
    handles = [
        Line2D(
            [],
            [],
            color=ctx.styles[p].color,
            marker=ctx.styles[p].marker,
            linestyle=ctx.styles[p].linestyle,
            label=ctx.styles[p].name,
        )
        for p in profiles
    ]
    columns = min(4, len(handles) + len(extra))
    fig.legend(handles=[*handles, *extra], loc="outside lower center", ncols=columns)


def _study_profiles(ctx: FigureContext) -> list[str]:
    present = {p for s in ctx.study.strata for p in s.profiles()}
    return [p for p in ctx.order if p in present]


# F1 coverage over time ---------------------------------------------------------------------------------------


def _trace(ax: Any, run: Row, ctx: FigureContext, stratum: Stratum, style: ProfileStyle) -> None:
    """One run's recorded steps, broken where valid samples are more than 2 s apart (nothing bridges a gap)."""
    limit = stratum.limit_s
    if run["outcome"] == "no_movement":
        held = ctx.study.pre_motion.get(run["attempt_id"])
        if held is not None:
            ax.plot([0, limit], [held, held], color=style.color, lw=TRACE_WIDTH, ls=":", alpha=0.6)
        return
    series = ctx.study.coverage.get(run["attempt_id"])
    if series is None or not series.t.size:
        return
    end, carried = run_end(run, limit)
    keep = series.t <= end + 1e-9
    t, c = series.t[keep], series.coverage[keep]
    breaks = np.nonzero(np.diff(t) > GAP_S)[0] + 1
    # the last stretch reaches the run's end only when that is within the gap rule; else the trace stops
    final = end if end - t[-1] <= GAP_S else t[-1]
    for part_t, part_c in zip(np.split(t, breaks), np.split(c, breaks), strict=True):
        last = final if part_t[-1] == t[-1] else part_t[-1]
        steps_t, steps_c = np.append(part_t, last), np.append(part_c, part_c[-1])
        ax.step(steps_t, steps_c, where="post", color=style.color, lw=TRACE_WIDTH, alpha=0.45)
    if carried and end < limit and final == end:
        ax.plot(
            [end, limit], [c[-1], c[-1]], color=style.color, lw=TRACE_WIDTH, ls="--", alpha=0.45
        )


def _band(
    ax: Any, ctx: FigureContext, runs: list[Row], style: ProfileStyle
) -> np.ndarray[Any, Any] | None:
    """Pointwise median and IQR (type 2) where at least ``min_n`` runs have a value; the movement-only median
    as a dotted line when a run never moved. Returns the number of runs still running at each grid time."""
    cells = [ctx.grid[r["attempt_id"]] for r in runs if r["attempt_id"] in ctx.grid]
    if not cells:
        return None
    grid_t = cells[0][0]  # one grid per stratum: every run shares its L
    moved = [r for r in runs if r["outcome"] != "no_movement"]
    for selection, primary in ((runs, True), (moved, False)):
        if not primary and len(moved) == len(runs):
            continue
        rows = [ctx.grid[r["attempt_id"]][1] for r in selection if r["attempt_id"] in ctx.grid]
        matrix = np.vstack(rows) if rows else np.full((1, grid_t.size), np.nan)
        support = np.sum(~np.isnan(matrix), axis=0)
        with np.errstate(all="ignore"):
            q = np.nanquantile(matrix, [0.25, 0.5, 0.75], axis=0, method="averaged_inverted_cdf")
        q[:, support < ctx.settings.min_n] = np.nan
        if primary:
            ax.fill_between(grid_t, q[0], q[2], step="post", color=style.color, alpha=0.18, lw=0)
            ax.step(grid_t, q[1], where="post", color=style.color, lw=1.2, ls=style.linestyle)
        else:
            ax.step(grid_t, q[1], where="post", color=style.color, lw=0.8, ls=":")
    return np.array(
        [sum(1 for cell in cells if cell[2][i] == "sample") for i in range(grid_t.size)]
    )


def f1(ctx: FigureContext) -> FigureOut:
    fig, cells, width = _panels(ctx, strip=True)
    attempts: set[str] = set()
    for index, (stratum, (ax, strip)) in enumerate(zip(ctx.study.strata, cells, strict=True)):
        limit, largest = stratum.limit_s, 1
        for profile in ctx.profiles(stratum):
            style, runs = ctx.styles[profile], stratum.runs_of(profile)
            attempts |= {r["attempt_id"] for r in runs}
            for run in runs:
                _trace(ax, run, ctx, stratum, style)
            running = _band(ax, ctx, runs, style)
            if running is not None:
                grid_t = ctx.grid[runs[0]["attempt_id"]][0]
                strip.step(
                    grid_t, running, where="post", color=style.color, lw=1.0, ls=style.linestyle
                )
            largest = max(largest, len(runs))
        ax.axvline(limit, color=GREY, lw=0.6, ls=":")
        ax.text(limit, 2, " L", color=GREY, fontsize=8, ha="left", va="bottom")
        ax.set(xlim=(0, limit * 1.04), ylim=(0, 100), ylabel="Coverage (%)")
        ax.tick_params(labelbottom=False)
        strip.set(
            xlim=(0, limit * 1.04), ylim=(0, largest * 1.15), ylabel="Running", xlabel=TIME_LABEL
        )
        strip.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
        _title(ax, index, ctx.label(stratum))
    carried = Line2D([], [], color=GREY, ls="--", lw=TRACE_WIDTH, label="carried forward to L")
    _legend(fig, ctx, _study_profiles(ctx), [carried])
    _finish(fig, ctx)
    min_n = ctx.settings.min_n
    return FigureOut(
        "F1",
        fig,
        width,
        attempts,
        f"Coverage over time for every reported run in {len(ctx.study.strata)} strata, with the pointwise median "
        f"and interquartile band of each profile where at least {min_n} runs have a value.",
        [
            "Thin steps: runs, broken where valid samples are more than 2 s apart; dashed: a completed run's final "
            "coverage carried to L (RD-6); dotted at a constant: a run that never moved, held at its pre-motion "
            "coverage (RD-8). Band: pointwise median and IQR (type 2), descriptive, only where at least "
            f"{min_n} runs have a value under the gap rule; a dotted median: runs that moved only (RD-8, "
            "secondary). Strip: runs still running.",
        ],
    )


# F2 attainment -------------------------------------------------------------------------------------------------


def f2(ctx: FigureContext, endpoint: str) -> FigureOut:
    fig, cells, width = _panels(ctx, strip=True)
    attempts: set[str] = set()
    label = "completion" if endpoint == "R_completion" else f"{endpoint[2:]} % coverage"
    for index, (stratum, (ax, table)) in enumerate(zip(ctx.study.strata, cells, strict=True)):
        limit = stratum.limit_s
        times = np.linspace(0, limit, 5)
        profiles = ctx.profiles(stratum)
        for position, profile in enumerate(profiles):
            style = ctx.styles[profile]
            entry = ctx.entries.get("restricted_time", stratum, profile, endpoint=endpoint)
            usable = [
                v
                for run in stratum.runs_of(profile)
                if (v := time_value(run, endpoint, limit)).value is not None
            ]
            n = len(usable)
            if not n:
                continue
            attempts |= set(entry.get("attempt_ids", []))
            events = sorted(float(v.value) for v in usable if v.event and v.value is not None)
            xs = np.concatenate([[0.0], events, [limit]])
            ys = np.concatenate([[0.0], np.arange(1, len(events) + 1) / n, [len(events) / n]])
            ax.step(xs, 100 * ys, where="post", color=style.color, ls=style.linestyle, lw=1.0)
            offset = (position - (len(profiles) - 1) / 2) * limit * 0.012
            for point in (entry.get("attainment") or {}).values():
                interval, share = point["interval"], point["k"] / n
                ax.errorbar(
                    [point["horizon_s"] + offset],
                    [100 * share],
                    yerr=[[100 * (share - interval["lower"])], [100 * (interval["upper"] - share)]],
                    fmt=style.marker,
                    color=style.color,
                    ms=3,
                    lw=0.8,
                )
            y = 1 - (position + 1) / (len(profiles) + 1)
            for t in times:
                # runs still without the event at t; at L they are the runs not reached by L (FP-3)
                at_risk = sum(
                    1 for v in usable if v.value is not None and (v.value >= t or not v.event)
                )
                text = f"{at_risk}/{n}" if t == limit else str(at_risk)
                table.text(t, y, text, ha="center", va="center", fontsize=8, color=style.color)
        ax.axvline(limit, color=GREY, lw=0.6, ls=":")
        ax.set(
            xlim=(0, limit * 1.04), ylim=(0, 100), ylabel=f"Reached {label} (%)", xlabel=TIME_LABEL
        )
        table.set_axis_off()
        table.set(xlim=(0, limit * 1.04), ylim=(0, 1))
        table.set_title("At risk (at L: not reached by L)", loc="left", fontsize=8)
        _title(ax, index, ctx.label(stratum))
    _legend(fig, ctx, _study_profiles(ctx))
    _finish(fig, ctx)
    return FigureOut(
        f"F2_{endpoint}",
        fig,
        width,
        attempts,
        f"Share of runs that reached {label} by each time, per profile and stratum, with Wilson intervals at the "
        "plan's horizons and the number of runs still at risk.",
        [
            f"Steps: 1 − Ŝ(t) (all censoring at L, so the share of runs). Points: Wilson {ctx.level_label} intervals at "
            "the horizons. Runs without movement count as never reaching (RD-8); failures and stopped runs are "
            "excluded and counted in T1. At-risk counts at 0, L/4, L/2, 3L/4 and L.",
        ],
    )


# F3 forest plots -----------------------------------------------------------------------------------------------


def _pooled_row(ctx: FigureContext, effect: Effect) -> dict[str, Any] | None:
    """A pre-specified pooled hypothesis drawn on this effect's axis: ΔC(h) and ΔA_h at L share the scale; ΔR
    is pooled as ΔR/L and gets no diamond on a seconds axis (T5H shows it)."""
    wanted = {"delta_C_h": "C_h", "delta_A_h": "A_h"}.get(effect.estimand)
    limits = {s.limit_s for s in ctx.study.strata}
    for e in ctx.entries.where(estimand="hypothesis", world_id="pooled", testable=True):
        same_horizon = e.get("horizon_s") in (None, *limits)
        if e.get("hypothesis_endpoint") == wanted and len(limits) == 1 and same_horizon:
            return e
    return None


def _forest_rows(ctx: FigureContext, effect: Effect) -> list[tuple[str, dict[str, Any], str, str]]:
    """Label, entry, colour and marker of every row, in study order, the pooled row last."""
    rows = []
    for stratum in ctx.study.strata:
        baseline = ctx.study.baselines.get(stratum.robot_id)
        for profile in ctx.profiles(stratum):
            if profile == baseline:
                continue
            e = ctx.entries.get(
                effect.estimand,
                stratum,
                profile,
                endpoint=effect.endpoint,
                horizon_s=None if effect.endpoint else stratum.limit_s,
                variant=effect.estimator,
            )
            if e:
                style = ctx.styles[profile]
                rows.append((f"{ctx.label(stratum)} · {style.name}", e, style.color, "s"))
    pooled = _pooled_row(ctx, effect)
    if pooled is not None:
        name = ctx.study.profiles[pooled["profile_id"]]["profile_name"]
        rows.append((f"pooled ({pooled['hypothesis_id']}) · {name}", pooled, "#000000", "D"))
    return rows


def f3(ctx: FigureContext, effect: Effect) -> FigureOut:
    rows = _forest_rows(ctx, effect)
    width_in = PRESETS[ctx.preset].widths_mm["double"] / MM_PER_INCH
    aspect = (0.22 * max(1, len(rows)) + 1.0) / width_in
    fig, axes = figure(
        ctx.preset, "double", aspect=aspect, ncols=2, gridspec_kw={"width_ratios": [3, 2]}
    )
    ax, text = axes[0][0], axes[0][1]
    level = 1 - ctx.settings.alpha
    attempts: set[str] = set()
    for y, (_, e, colour, marker) in enumerate(reversed(rows)):
        attempts |= set(e.get("attempt_ids", []))
        estimate, interval = e.get("estimate"), e.get("interval") or {}
        if interval.get("lower") is not None and interval.get("upper") is not None:
            ax.plot([interval["lower"], interval["upper"]], [y, y], color=colour, lw=1.2)
        if estimate is not None:
            ax.plot(
                [estimate],
                [y],
                marker=marker,
                color=colour,
                ms=4 if marker == "s" else 6,
                ls="none",
            )
        text.text(0.06, y, str(e.get("n_pairs", 0)), va="center", ha="center", fontsize=8)
        value = with_interval(estimate, interval or None, effect.unit, level)
        text.text(0.2, y, value, va="center", fontsize=8)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_yticks(range(len(rows)), [label for label, *_ in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(f"{effect.label}, profile − Default")
    lower, higher = ("shorter", "longer") if effect.unit == "s" else ("lower", "higher")
    for x, words, align in (
        (0.0, f"← {lower} with the profile", "left"),
        (1.0, f"{higher} with the profile →", "right"),
    ):
        ax.annotate(
            words,
            xy=(x, 0),
            xycoords="axes fraction",
            xytext=(0, -30),
            textcoords="offset points",
            ha=align,
            va="top",
            fontsize=8,
            color=GREY,
        )
    text.set(ylim=ax.get_ylim(), xlim=(0, 1))
    text.set_axis_off()
    top = len(rows) - 0.4
    text.text(0.06, top, "n pairs", fontsize=8, va="bottom", ha="center", fontweight="bold")
    text.text(0.2, top, "estimate [lo, hi]", fontsize=8, va="bottom", fontweight="bold")
    _finish(fig, ctx)
    estimator = (
        "mean (exact sign-flip inversion below 10 pairs, block bootstrap from 10)"
        if effect.estimator == "mean"
        else "Hodges–Lehmann (inverted signed-rank test)"
    )
    return FigureOut(
        f"F3_{effect.name}",
        fig,
        "double",
        attempts,
        f"Paired differences {effect.label} between each profile and Default per stratum, with intervals.",
        [
            f"Squares: {estimator}; lines: {ctx.level_label} intervals (a level in parentheses when fewer pairs allow "
            "less). Rows in study order; a diamond only for a pre-specified pooled hypothesis on this scale: the "
            "equal-weight mean over worlds of each world's mean difference, stratified bootstrap (RD-16)."
        ],
    )


# Distributions (F5, F6) ----------------------------------------------------------------------------------------

# a stratum's label and, per profile, its observed values and its values censored at L
Group = tuple[str, list[tuple[str, list[float], list[float]]]]


def _box(values: Sequence[float]) -> dict[str, Any]:
    """Box statistics with the stated quartiles (type 2) and Tukey whiskers: the furthest values within
    1.5 IQR of the box."""
    q1, median, q3 = stats.quartiles(values)
    spread = 1.5 * (q3 - q1)
    inside = [v for v in values if q1 - spread <= v <= q3 + spread]
    return dict(med=median, q1=q1, q3=q3, whislo=min(inside), whishi=max(inside), fliers=[])


def _member(
    ax: Any, ctx: FigureContext, x: float, member: tuple[str, list[float], list[float]], rng: Any
) -> int:
    """One profile's runs at ``x``; returns how many."""
    profile, observed, censored = member
    style = ctx.styles[profile]
    everything = [*observed, *censored]
    if len(everything) >= VIOLIN_MIN:
        parts = ax.violinplot([everything], positions=[x], showextrema=False, widths=0.3)
        for body in parts["bodies"]:
            body.set_facecolor(style.color)
            body.set_alpha(0.15)
    if len(everything) >= ctx.settings.min_n:
        ax.bxp(
            [_box(everything)],
            positions=[x],
            widths=0.22,
            showfliers=False,
            medianprops=dict(color="#000000", lw=1.2),
            boxprops=dict(lw=0.6),
            whiskerprops=dict(lw=0.6),
            capprops=dict(lw=0.6),
        )
    jitter = rng.uniform(-0.06, 0.06, len(observed))
    ax.scatter(
        x + jitter,
        observed,
        s=8,
        color=style.color,
        marker=style.marker,
        alpha=0.85,
        linewidths=0,
        zorder=3,
    )
    if censored:  # R = L: drawn at L, never as observed values
        jitter = rng.uniform(-0.06, 0.06, len(censored))
        ax.scatter(
            x + jitter,
            censored,
            s=14,
            marker="^",
            facecolors="none",
            edgecolors=style.color,
            linewidths=0.8,
            zorder=3,
        )
    return len(everything)


def _distribution(ax: Any, ctx: FigureContext, groups: Sequence[Group]) -> None:
    """Every run as a point; a box from ``min_n`` runs and a violin from 30, over all values (a run not
    reaching by L counts at R = L); those runs as triangles at L; n per profile under each stratum."""
    rng = stats.random_stream(ctx.settings.seed, stats.JITTER_STREAM)
    labels = []
    for index, (name, members) in enumerate(groups):
        counts = [
            str(_member(ax, ctx, index + (slot - (len(members) - 1) / 2) * 0.34, member, rng))
            for slot, member in enumerate(members)
        ]
        labels.append(f"{name}\nn = {' / '.join(counts)}")
    ax.set_xticks(range(len(groups)), labels, fontsize=8)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    if not any(observed or censored for _, members in groups for _, observed, censored in members):
        ax.text(
            0.5,
            0.5,
            "no runs with a value",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=GREY,
        )
        ax.set_yticks([])


def _groups(ctx: FigureContext, read: Callable[[Row, Stratum], Value]) -> list[Group]:
    """Per stratum and profile: observed values, and values of runs that did not reach by L (``event`` false)."""
    groups: list[Group] = []
    for stratum in ctx.study.strata:
        members: list[tuple[str, list[float], list[float]]] = []
        for profile in ctx.profiles(stratum):
            found = [read(run, stratum) for run in stratum.runs_of(profile)]
            observed = [
                float(v.value) for v in found if v.value is not None and v.event is not False
            ]
            censored = [float(v.value) for v in found if v.value is not None and v.event is False]
            members.append((profile, observed, censored))
        groups.append((ctx.label(stratum), members))
    return groups


def f5(ctx: FigureContext) -> FigureOut:
    fig, axes = figure(ctx.preset, "double", aspect=0.36, ncols=2)
    attempts: set[str] = set()
    for index, (metric, label) in enumerate(
        (("cpu_per_s", "Explorer CPU (% of one core)"), ("rss_peak_mib", "Explorer peak RSS (MiB)"))
    ):
        read = METRICS[metric]
        _distribution(axes[0][index], ctx, _groups(ctx, lambda run, _s: read(run)))  # noqa: B023
        attempts |= {
            r["attempt_id"]
            for s in ctx.study.strata
            for r in s.reported
            if read(r).value is not None
        }
        axes[0][index].set_ylabel(label)
        _title(axes[0][index], index, "sequential runs")
    _legend(fig, ctx, _study_profiles(ctx))
    _finish(fig, ctx)
    return FigureOut(
        "F5",
        fig,
        "double",
        attempts,
        "Explorer CPU use per simulated second and peak memory per profile and stratum, sequential runs only.",
        [
            f"Points: runs; box (median, IQR type 2, whiskers 1.5 IQR) from {ctx.settings.min_n} runs; violin "
            f"from {VIOLIN_MIN}. Explorer process only; runs that overlapped another attempt or lack samples are "
            "left out and counted in stats.json."
        ],
    )


def _outcomes(ax: Any, labels: list[str], counts: list[dict[str, int]]) -> None:
    for y, by_class in enumerate(counts):
        left = 0
        for outcome, look in OUTCOMES.items():
            value = by_class.get(outcome, 0)
            if not value:
                continue
            ax.barh(
                y,
                value,
                left=left,
                color=look["facecolor"],
                hatch=look["hatch"],
                edgecolor="#333333",
                lw=0.4,
            )
            colour = "#FFFFFF" if outcome == "completed" else "#000000"
            ax.text(
                left + value / 2, y, str(value), ha="center", va="center", fontsize=7, color=colour
            )
            left += value
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Final runs")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    used = [
        Patch(facecolor=v["facecolor"], hatch=v["hatch"], edgecolor="#333333", label=v["label"])
        for k, v in OUTCOMES.items()
        if any(c.get(k) for c in counts)
    ]
    ax.legend(handles=used, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=3)
    _title(ax, 0, "final runs by outcome")


def _success(
    ax: Any, ctx: FigureContext, success: list[tuple[int, int, dict[str, Any] | None]]
) -> None:
    for y, (k, n, interval) in enumerate(success):
        if n and interval:
            ax.plot(
                [100 * interval["lower"], 100 * interval["upper"]], [y, y], color="#000000", lw=1.0
            )
            ax.plot([100 * k / n], [y], "o", color="#000000", ms=4)
            ax.text(104, y, f"{k}/{n}", va="center", fontsize=8)
    ax.set_yticks(range(len(success)), [""] * len(success))
    ax.invert_yaxis()
    ax.set(xlim=(0, 120), xlabel=f"Completed within L (%, Wilson {ctx.level_label})")
    ax.set_xticks([0, 25, 50, 75, 100])
    _title(ax, 1, "success (rows as in a)")


def f6(ctx: FigureContext) -> FigureOut:
    fig, axes = figure(ctx.preset, "double", aspect=0.62, nrows=2, ncols=2)
    attempts: set[str] = set()
    labels, counts, success = [], [], []
    for stratum in ctx.study.strata:
        for profile in ctx.profiles(stratum):
            finals = [r for r in stratum.finals if r["profile_id"] == profile]
            attempts |= {r["attempt_id"] for r in finals}
            by_class: dict[str, int] = defaultdict(int)
            for run in finals:
                by_class[
                    run["outcome"] if run["report_class"] == "reported" else run["report_class"]
                ] += 1
            labels.append(f"{ctx.label(stratum)} · {ctx.styles[profile].name}")
            counts.append(by_class)
            entry = ctx.entries.get("success", stratum, profile)
            success.append((entry.get("k", 0), entry.get("n", 0), entry.get("interval")))
    _outcomes(axes[0][0], labels, counts)
    _success(axes[0][1], ctx, success)
    _distribution(
        axes[1][0], ctx, _groups(ctx, lambda run, s: time_value(run, "R_completion", s.limit_s))
    )
    axes[1][0].set_ylabel("R_completion (s)")
    _title(axes[1][0], 2, "R_completion (△: not reached by L, at L)")
    delta = ctx.settings.stall_delta_pp
    _distribution(
        axes[1][1], ctx, _groups(ctx, lambda run, s: idle_tail(ctx.study, run, s.limit_s, delta))
    )
    axes[1][1].set_ylabel("Idle tail (s)")
    _title(axes[1][1], 3, "time after the last coverage gain")
    _legend(fig, ctx, _study_profiles(ctx))
    _finish(fig, ctx)
    return FigureOut(
        "F6",
        fig,
        "double",
        attempts,
        "Outcomes, success with Wilson intervals, restricted completion times and idle tails per profile and "
        "stratum.",
        [
            "Outcome classes follow reported-outcomes-v2 (greys and hatches, FP-4). Triangles: runs that did not "
            "complete by L, drawn at L; boxes over all restricted times. Idle tail: min(T_end, L) − last gain "
            f"(stall-v1, δ = {delta:g} pp)."
        ],
    )


# F8 representative paths -------------------------------------------------------------------------------------


def _block_order(run: Row) -> str:
    return hashlib.sha256(run["block_id"].encode()).hexdigest()


def representatives(ctx: FigureContext, stratum: Stratum) -> list[tuple[Row, str]]:
    """The runs the plan's rule picks in one stratum (FIGURE_STYLE §8), each with its panel note.

    SEL-1: the Default run whose restricted time is closest to the median of the stratum's reported Default
    runs. SEL-2: the block whose paired difference (``stats.json``: the first other profile in parameter-hash
    order − Default, reported pairs) is closest to the estimate; both runs. SEL-3: the SEL-1 run and also the
    most adverse Default run that moved (the longest restricted time), labelled so. Ties: the SHA-256 of the
    block ID."""
    baseline = ctx.study.baselines.get(stratum.robot_id) or ""
    endpoint, limit = ctx.settings.representative_endpoint, stratum.limit_s

    def restricted(run: Row) -> float:
        value = restricted_time(run, endpoint, limit).value
        return float("nan") if value is None else value

    defaults = [r for r in stratum.runs_of(baseline) if not math.isnan(restricted(r))]
    rule = ctx.settings.representative_rule
    others = [p for p in ctx.profiles(stratum) if p != baseline]
    if rule == "SEL-2" and others:
        entry = ctx.entries.get(
            "delta_restricted_time",
            stratum,
            others[0],
            endpoint=endpoint,
            variant=PRIMARY_ESTIMATOR["delta_restricted_time"],
        )
        if entry.get("blocks") and entry.get("estimate") is not None:
            by_block = {
                block: d for block, d in zip(entry["blocks"], entry["differences"], strict=True)
            }
            block = min(
                by_block,
                key=lambda b: (
                    abs(by_block[b] - entry["estimate"]),
                    hashlib.sha256(b.encode()).hexdigest(),
                ),
            )
            _, run, partner = next(
                item for item in stratum.blocks(others[0], baseline) if item[0] == block
            )
            if run is not None and partner is not None:
                return [(partner, "SEL-2, Default"), (run, f"SEL-2, {ctx.styles[others[0]].name}")]
    if not defaults:
        return []
    _, median, _ = stats.quartiles([restricted(r) for r in defaults])
    chosen = min(defaults, key=lambda r: (abs(restricted(r) - median), _block_order(r)))
    picked = [(chosen, "SEL-1" if rule != "SEL-2" else "SEL-1 (no other profile to pair)")]
    moved = [r for r in defaults if r["outcome"] != "no_movement"]
    if rule == "SEL-3" and moved:
        worst = min(moved, key=lambda r: (-restricted(r), _block_order(r)))
        picked.append((worst, "most adverse run (SEL-3)"))
    return picked


def _reference(
    ctx: FigureContext, attempt: str
) -> tuple[np.ndarray[Any, Any], tuple[float, float, float, float]] | None:
    """The coverage reference the run used (``provenance.json`` maps each reference directory to its runs)."""
    directory = ctx.study.export.directory
    references = ctx.study.export.provenance.get("references") or {}
    name = next((ref for ref, runs in references.items() if attempt in runs), None)
    if name is None or not (directory / "references" / name / "reference.yaml").is_file():
        return None
    reference = directory / "references" / name
    meta = yaml.safe_load((reference / "reference.yaml").read_text(encoding="utf-8"))
    masks = {
        key: np.array(Image.open(reference / f"{key}.pgm")) == 255
        for key in ("target", "collision", "inclusion")
    }
    codes = np.full(masks["target"].shape, 3, dtype=np.int8)  # outside the reference
    codes[masks["inclusion"]] = 1  # inside, not free floor: unknown grey
    codes[masks["target"]] = 0
    codes[masks["collision"]] = 2
    x0, y0 = meta["origin"][:2]
    height, width = codes.shape
    extent = (x0, x0 + width * meta["resolution"], y0, y0 + height * meta["resolution"])
    return codes[::-1], extent  # PGM row 0 is the largest y


Path2D = tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], float | None]


def _ground_truth(ctx: FigureContext, attempt: str, end: float) -> Path2D | None:
    table = ctx.study.export.table(
        "trajectory_gt", ["attempt_id", "t_s", "gt_x_m", "gt_y_m", "gt_yaw_rad"]
    )
    rows = sorted(
        (
            r
            for r in table.to_pylist()
            if r["attempt_id"] == attempt
            and r["t_s"] is not None
            and r["gt_x_m"] is not None
            and r["t_s"] <= end + 1e-9
        ),
        key=lambda r: r["t_s"],
    )
    if len(rows) < 2:
        return None
    yaw = rows[0]["gt_yaw_rad"]
    return (
        np.array([r["t_s"] for r in rows]),
        np.array([r["gt_x_m"] for r in rows]),
        np.array([r["gt_y_m"] for r in rows]),
        None if yaw is None else float(yaw),
    )


def _map_panel(
    ax: Any, ctx: FigureContext, run: Row, stratum: Stratum, norm: Normalize
) -> LineCollection | None:
    reference = _reference(ctx, run["attempt_id"])
    end, _ = run_end(run, stratum.limit_s)
    path = _ground_truth(ctx, run["attempt_id"], end)
    if reference is None or path is None:
        ax.text(0.5, 0.5, "no reference or path", transform=ax.transAxes, ha="center", fontsize=8)
        return None
    codes, extent = reference
    palette = ListedColormap([MAP_FREE, MAP_UNKNOWN, MAP_OCCUPIED, MAP_OUTSIDE])
    ax.imshow(
        codes,
        cmap=palette,
        vmin=-0.5,
        vmax=3.5,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        interpolation_stage="rgba",
        zorder=0,
    )
    resolution = (extent[1] - extent[0]) / codes.shape[1]
    centres_x = extent[0] + resolution * (np.arange(codes.shape[1]) + 0.5)
    centres_y = extent[2] + resolution * (np.arange(codes.shape[0]) + 0.5)
    ax.contourf(
        centres_x,
        centres_y,
        codes == 3,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
        zorder=0.5,
    )
    times, xs, ys, yaw = path
    segments = np.stack(
        [np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1
    )
    lines = LineCollection(
        list(segments),
        cmap=plt.get_cmap("cividis"),
        norm=norm,
        linewidths=1.0,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    lines.set_array((times[:-1] + times[1:]) / 2)
    lines.set_path_effects(
        [patheffects.Stroke(linewidth=2.0, foreground="#333333"), patheffects.Normal()]
    )
    ax.add_collection(lines)
    if yaw is not None:
        tip = (xs[0] + 0.6 * math.cos(yaw), ys[0] + 0.6 * math.sin(yaw))
        arrow = dict(arrowstyle="-|>", color="#000000", lw=1.0)
        ax.annotate("", xy=tip, xytext=(xs[0], ys[0]), arrowprops=arrow, zorder=4)
    ax.plot(
        [xs[-1]],
        [ys[-1]],
        marker="s",
        color="#000000",
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.5,
        ms=4,
        zorder=4,
    )
    ax.set_xlim(extent[0] - 0.5, extent[1] + 0.5)
    ax.set_ylim(extent[2] - 0.5, extent[3] + 0.5)
    ax.set_aspect("equal", adjustable="box")
    bar = AnchoredSizeBar(
        ax.transData,
        SCALE_BAR_M,
        f"{SCALE_BAR_M:g} m",
        loc="lower right",
        frameon=False,
        size_vertical=0.15,
        label_top=True,
        sep=2,
        fontproperties={"size": 8},
    )
    ax.add_artist(bar)  # the same 5 m in every panel (FIGURE_STYLE §6)
    return lines


def f8(ctx: FigureContext) -> FigureOut:
    panels = [(s, run, note) for s in ctx.study.strata for run, note in representatives(ctx, s)]
    if ctx.settings.representative_rule in ("SEL-2", "SEL-3") and len(panels) > 1:
        # one row per stratum: its two chosen runs side by side
        rows, columns = math.ceil(len(panels) / 2), 2
    else:
        rows, columns = _grid_size(max(1, len(panels)))
    width = "double" if len(panels) > 1 else "single"
    fig, axes = figure(ctx.preset, width, aspect=0.45 * rows, nrows=rows, ncols=columns)
    flat = [axes[r][c] for r in range(rows) for c in range(columns)]
    norm = Normalize(0, max(s.limit_s for s in ctx.study.strata))
    chosen: dict[str, str] = {}
    mappable = None
    for ax in flat:
        ax.set_axis_off()
    for index, (stratum, run, note) in enumerate(panels):
        _title(flat[index], index, f"{ctx.label(stratum)} · {note}")
        mappable = _map_panel(flat[index], ctx, run, stratum, norm) or mappable
        chosen[f"{ctx.label(stratum)} · {note}"] = run["attempt_id"]
    if mappable is not None:
        bar = fig.colorbar(
            mappable, ax=flat, orientation="horizontal", fraction=0.05, pad=0.02, aspect=40
        )
        bar.set_label(TIME_LABEL)
    _finish(fig, ctx)
    rule, endpoint = ctx.settings.representative_rule, ctx.settings.representative_endpoint
    return FigureOut(
        "F8",
        fig,
        width,
        set(chosen.values()),
        "Ground-truth path of the representative runs of each stratum on the coverage reference, coloured by time.",
        [
            f"Runs chosen by {rule} on {endpoint} (FIGURE_STYLE §8): "
            + ", ".join(f"{k}: {v}" for k, v in chosen.items())
            + ". Map: free floor white, other inside cells light grey, obstacles dark grey, outside the reference "
            "hatched. Arrow: start pose; square: end. Path to min(T_end, L) from the export's 1 s ground-truth rows."
        ],
        {"representative": chosen, "rule": rule, "endpoint": endpoint},
    )


def figures(ctx: FigureContext) -> list[FigureOut]:
    built = [f1(ctx)]
    built += [f2(ctx, e) for e in time_endpoints(ctx.settings)]
    built += [f3(ctx, effect) for effect in EFFECTS]
    return built + [f5(ctx), f6(ctx), f8(ctx)]
