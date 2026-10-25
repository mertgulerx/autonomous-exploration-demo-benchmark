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
  secondary dotted line when some run never moved (RD-8), a strip of running runs, L marked; the time axis ends
  at the stratum's latest run end rounded up to a tick, or at L when a run ends there (FX-2, ``time_axis_end``).
- F2 (one per restricted-time endpoint): attainment 1 − Ŝ(t) as upward steps, Wilson intervals at the plan's
  horizons (from ``stats.json``), the at-risk table whose last column is "k/n not reached by L" (FP-3); the time
  axis as F1's.
- F3 (one per paired effect of ``estimates.EFFECTS``): forest plot, profile − Default, with "n pairs" and
  "estimate [lo, hi]" columns, and a pooled diamond for a pre-specified pooled hypothesis on the same scale.
- F5: compute, F6: reliability; every run as a point, a box from ``min_n`` runs and a violin from 30 (type 2
  quartiles, Tukey whiskers), runs not reaching by L as triangles at L, success as a dot with its interval; F5 (c)
  the per-pair CPU ratio of sequential pairs with the geometric-mean ratio (FIGURE_EXTENSION FX2).
- F8: the representative runs of each stratum (the plan's rule: SEL-1, SEL-2 or SEL-3, FIGURE_STYLE §8) on the
  coverage reference the run used, coloured by time on one shared scale, with start pose, end and 5 m scale.

Every figure is drawn at its final width, names the clock on its time axes (FP-10), states the plan's interval
level, and is labelled with its evidence level unless the study is confirmatory.

The module is also the figures' shared toolkit (FIGURE_EXTENSION), used by every other figure module:
- context and output: ``FigureContext``, ``context``, ``FigureOut``, ``finish`` (evidence label), ``legend``;
- layout and text: ``stratum_grid`` (a panel per stratum, a strip under each on request), ``panel_label`` and
  ``panel_title`` (a … z, aa, …; ``wrap`` for narrow columns), ``wrap_text``, ``panel_message`` and
  ``message_figure`` (nothing to draw, said so), ``stratum_variants`` (per-stratum output names and labels),
  ``study_profiles``, ``paired_strata``, ``estimator_text`` and ``interval_methods``;
- time axes (FX-2): ``time_axis_end`` and ``time_axis`` (the L line or the named L, the x limits), ``name_limit``,
  ``AXIS_NOTE``;
- parts drawn into given axes, shared by the standalone figures and the composites F21 and F22:
  ``coverage_panel``, ``coverage_trace``, ``coverage_band``, ``attainment_panel``, ``outcomes`` and
  ``outcome_bars``, ``map_panel``;
- maps and paths: ``load_reference``, ``draw_reference`` (``MapCode``, ``REFERENCE_PAD_M``), ``ground_truth``,
  ``representatives`` (SEL-1 to SEL-3), ``time_lines``, ``time_bar``, ``path_norm``, ``reference_files``;
- world groups (plans/worlds/I8 §3.1): ``grouped`` draws an all-strata figure once per world group and page
  (``group_pages``: at most ``PAGE_PANELS`` panels, never splitting a world, and the figure no taller than a page
  where its layout allows, tested on the page's context), ``grid_fits`` the page test of ``stratum_grid``
  figures, ``page_context``, ``group_inputs``; ``FigureContext.full_study`` for labels and study-wide scales.

Group variants: a study with several world groups writes ``<name>_<group>.pdf`` (``F22_a_SC``); a group drawn
on several pages adds ``_p1``, ``_p2`` …; a study with one group (or an export without the ``worlds`` table,
export version 1) on one page keeps the ungrouped name (``F1.pdf``), so its outputs stay comparable with
analyses of ungrouped studies. A group variant's sidecar names its group in ``extra["group"]`` (``grouped``).
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import partial
import hashlib
import math
import re
import textwrap
from typing import Any, Literal

from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from PIL import Image
import yaml

from . import stats
from .data import (
    GAP_S,
    TIME_EPSILON,
    Row,
    Stratum,
    Study,
    Value,
    idle_tail,
    TIME_ENDPOINTS,
    restricted_time,
    run_end,
)
from .estimates import (
    EFFECTS,
    METRICS,
    PRIMARY_ESTIMATOR,
    Effect,
    Entries,
    pairs,
    time_endpoints,
    time_value,
)
from .export import Export
from .settings import MILESTONE_OF_ENDPOINT, Settings, milestone_label
from .style import (
    EDGE,
    MAP_FREE,
    MAP_OCCUPIED,
    MAP_OUTSIDE,
    MAP_UNKNOWN,
    MM_PER_INCH,
    OUTCOMES,
    PAGE_MAX_MM,
    PRESETS,
    SEQUENTIAL_CMAP,
    ProfileStyle,
    figure,
    profile_styles,
)
from .tables import level_text, with_interval
from .worlds import WorldGroup, world_groups

TIME_LABEL = "Simulated time since first movement (s)"
GREY = "#7F7F7F"
TRACE_WIDTH = 0.5  # per-run traces: thin, above the 0.4 pt floor (FIGURE_STYLE §2)
VIOLIN_MIN = 30  # FIGURE_STYLE §5: violins and densities only from 30 points
PANEL_ASPECT = 0.42  # height of one row of panels per unit of width, multi-panel figures
SINGLE_ASPECT = 0.85  # height per width of a one-stratum figure
STRIP_RATIO = 1.0  # height of a strip or at-risk table under a panel, relative to 4 for the panel
LABEL_STRIP_IN = 0.14  # inches kept free at the bottom for the evidence label
SCALE_BAR_M = 5.0
TITLE_CHAR_MM = 1.8  # about one 9 pt title character's width, to wrap a panel title to its column
# I8 §3.1: at most this many panels per page (strata, rows of a composite, atlas worlds)
PAGE_PANELS = 10


@dataclass
class FigureOut:
    name: str
    figure: Figure
    width: str
    attempt_ids: set[str]
    alt_text: str
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    # export files the figure read or depends on beyond the study tables (streams.Reads), hashed in the sidecar
    inputs: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class FigureContext:
    study: Study
    settings: Settings
    entries: Entries
    preset: str
    grid: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str | None]]]
    styles: dict[str, ProfileStyle]
    order: list[str]
    groups: tuple[WorldGroup, ...] = ()  # ``tables.world_groups``
    # the whole study when ``study`` holds only one group's page (``grouped``): labels stay the study's
    whole: Study | None = None

    def profiles(self, stratum: Stratum) -> list[str]:
        present = set(stratum.profiles())
        return [p for p in self.order if p in present]

    @property
    def full_study(self) -> Study:
        """The whole study, also on a group's page (``page_context``): labels and study-wide scales use it."""
        return self.whole or self.study

    def label(self, stratum: Stratum) -> str:
        return self.full_study.stratum_label(stratum)

    def group_of(self, stratum: Stratum) -> WorldGroup | None:
        """The stratum's world group when the export lists worlds and the study has several groups (found by
        identity: ``Stratum`` holds dict rows and is not hashable, and pages keep the study's objects)."""
        if len(self.groups) < 2:
            return None
        return next((g for g in self.groups if any(s is stratum for s in g.strata)), None)

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
        study,
        settings,
        entries,
        preset,
        grid,
        profile_styles(study.profiles),
        order,
        world_groups(study),
    )


# Shared layout -------------------------------------------------------------------------------------------------


def _grid_size(count: int) -> tuple[int, int]:
    columns = min(3, count)
    return math.ceil(count / columns), columns


def _grid_shape(count: int) -> tuple[int, int, str, float]:
    """``stratum_grid``'s rows, columns, width and aspect for ``count`` panels."""
    rows, columns = _grid_size(count)
    width = "double" if count > 1 else "single"
    return rows, columns, width, PANEL_ASPECT * rows if count > 1 else SINGLE_ASPECT


def stratum_grid(
    ctx: FigureContext, strata: Sequence[Stratum], *, strip: bool = False
) -> tuple[Figure, list[tuple[Any, Any]], str]:
    """One panel per stratum of ``strata`` in rows of up to three (and a strip under each when asked), blank
    cells hidden: the figure, a (panel, strip or None) pair per stratum, and the width ("single" for one)."""
    rows, columns, width, aspect = _grid_shape(len(strata))
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


def finish(fig: Figure, ctx: FigureContext, *, descriptive: bool = False) -> None:
    """Label a non-confirmatory figure in a strip the layout keeps free (FIGURE_STYLE §11 item 17), and with
    ``descriptive`` (a figure of display quantities, METHODOLOGY §3) tag it "descriptive" there in every study,
    confirmatory ones too; a figure legend "outside lower ..." is raised above the strip. Call it once the
    figure's legends are made."""
    if ctx.settings.confirmatory and not descriptive:
        return
    share = LABEL_STRIP_IN / fig.get_size_inches()[1]
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0, share, 1, 1 - share))  # type: ignore[call-arg]
    for placed in fig.legends:
        # constrained layout keeps the margin of an outside legend inside the rect, but the legend is anchored
        # to the figure's edge: its pad grows by the strip (in its font size), so it sits on that margin.
        # ``_outside_loc`` is what the layout reads too; a legend with its own anchor is left alone.
        if (
            getattr(placed, "_outside_loc", None) == "lower"
            and placed.get_bbox_to_anchor() is fig.bbox
        ):
            placed.borderaxespad += LABEL_STRIP_IN * 72 / placed.prop.get_size_in_points()
    parts = [] if ctx.settings.confirmatory else [ctx.settings.evidence_level]
    parts += ["descriptive"] if descriptive else []
    text = " · ".join([*parts, ctx.study.export.study_id])
    fig.text(0.995, 0.004, text, ha="right", va="bottom", fontsize=7, color=GREY)


def panel_label(index: int) -> str:
    """The letters of panel ``index``: a … z, then aa, ab, … (bijective base 26; letters only, safe in mathtext)."""
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(ord("a") + rest) + letters
    return letters


def wrap_text(text: str, width: int, *, first_taken: int = 0) -> str:
    """``text`` broken into lines of about ``width`` characters (never fewer than 12) at spaces, its first line
    ``first_taken`` shorter (room for a label before it); words, hyphenated ones too, are never split."""
    lines = textwrap.wrap(
        text,
        max(12, width),
        initial_indent=" " * first_taken,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join([lines[0].lstrip(), *lines[1:]]) if lines else text


def panel_title(ax: Any, index: int, text: str, *, wrap: int | None = None) -> None:
    """The panel label in bold, then the panel's subject, at the top left (FIGURE_STYLE §9); with ``wrap``, the
    subject broken into lines of about that many characters for a narrow column."""
    label = panel_label(index)
    if wrap is not None:
        text = wrap_text(text, wrap, first_taken=len(label) + 3)
    ax.set_title(rf"$\mathbf{{({label})}}$ {text}", loc="left")


def panel_message(ax: Any, text: str, *, boxed: bool = False, backdrop: bool = False) -> None:
    """A grey message in the middle of a panel that has nothing to draw: the frame kept with ``boxed``, else the
    axes hidden; with ``backdrop``, on a white box above what the panel draws (a map), the axes left as they
    are."""
    if not boxed and not backdrop:
        ax.set_axis_off()
    box = (
        dict(bbox=dict(facecolor="#FFFFFF", edgecolor="none", pad=1.0), zorder=5)
        if backdrop
        else {}
    )
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color=GREY,
        **box,
    )


def legend(
    fig: Figure,
    ctx: FigureContext,
    profiles: Sequence[str],
    extra: Sequence[Any] = (),
    *,
    max_columns: int = 4,
) -> None:
    """One legend under the figure: the profiles' encodings, then ``extra``, in at most ``max_columns`` columns."""
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
    columns = min(max_columns, len(handles) + len(extra))
    fig.legend(handles=[*handles, *extra], loc="outside lower center", ncols=columns)


def study_profiles(ctx: FigureContext) -> list[str]:
    """Every profile with a final run in some stratum, in study order."""
    present = {p for s in ctx.study.strata for p in s.profiles()}
    return [p for p in ctx.order if p in present]


def stratum_variants(ctx: FigureContext, prefix: str) -> list[tuple[Stratum, str, str]]:
    """One output per stratum, in study order, never dropping one: (stratum, output name, label). The name is
    ``<prefix>_<slug>`` of the stratum's label; where two strata share it, of the label with L; where L does not
    separate them, with the comparison key's first 8 characters; where names still collide, a numeric suffix.
    The label is the text the name was made from (the plain label unless it had to be told apart), led by the
    stratum's world group when the study has several (``SC · SC3``, I8 §3.1; the name stays without it): views
    take it from the sidecar's ``extra["stratum"]``."""
    strata = ctx.study.strata
    texts = [ctx.label(s) for s in strata]
    texts = [
        ctx.study.stratum_label(s, limit=True) if texts.count(text) > 1 else text
        for s, text in zip(strata, texts, strict=True)
    ]
    texts = [
        f"{text} {s.comparison_key[:8]}" if texts.count(text) > 1 else text
        for s, text in zip(strata, texts, strict=True)
    ]
    names = [f"{prefix}_{_slug(text)}" for text in texts]
    seen: Counter[str] = Counter()
    found = []
    for stratum, name, text in zip(strata, names, texts, strict=True):
        seen[name] += 1
        if names.count(name) > 1:  # slugs that still collide: numbered in study order
            name, text = f"{name}-{seen[name]}", f"{text} ({seen[name]})"
        group = ctx.group_of(stratum)
        found.append((stratum, name, text if group is None else f"{group.key} · {text}"))
    return found


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text.replace(" · ", "-")).strip("-") or "stratum"


def message_figure(
    ctx: FigureContext,
    name: str,
    text: str,
    *,
    attempts: Iterable[str] = (),
    inputs: Iterable[str] = (),
    wrap: int | None = None,
) -> FigureOut:
    """A one-column figure that only says why there is nothing to draw (its alt text and note the same); with
    ``wrap``, the message is broken into lines of about that many characters."""
    fig, axes = figure(ctx.preset, "single", aspect=0.25)
    panel_message(axes[0][0], text if wrap is None else wrap_text(text, wrap))
    finish(fig, ctx)
    return FigureOut(name, fig, "single", set(attempts), text, [text], inputs=set(inputs))


# World groups (plans/worlds/I8 §3.1) ---------------------------------------------------------------------------

# a page: its strata and worlds in order
Page = tuple[tuple[Stratum, ...], tuple[str, ...]]
# a figure's page test on the page's context (``page_context``): no taller than a page in its own layout
Fits = Callable[[FigureContext], bool]
# a page's panels from its strata and worlds
Panels = Callable[[Sequence[Stratum], Sequence[str]], int]


def _any_height(ctx: FigureContext) -> bool:
    return True


def _one_per_stratum(strata: Sequence[Stratum], worlds: Sequence[str]) -> int:
    return len(strata)


def grid_fits(ctx: FigureContext) -> bool:
    """A ``stratum_grid`` figure of the page's strata (F1, F2, F16, F18) is no taller than a page."""
    _, _, width, aspect = _grid_shape(len(ctx.study.strata))
    return PRESETS[ctx.preset].widths_mm[width] * aspect <= PAGE_MAX_MM


def _pack(units: Sequence[Page], ok: Callable[[Page], bool], count_panels: Panels) -> list[Page]:
    """Consecutive worlds in the fewest pages of balanced size that pass ``ok``; one page per world when no
    count does."""
    total = sum(count_panels(*unit) for unit in units)
    for count in range(max(1, math.ceil(total / PAGE_PANELS)), len(units) + 1):
        # each page takes worlds up to its share of what is left: 10 panels on 4 pages are 3, 3, 2, 2
        pages: list[Page] = []
        left, current = total, units[0]
        for strata, worlds in units[1:]:
            share = math.ceil(left / (count - len(pages)))
            joined = (current[0] + strata, current[1] + worlds)
            if count_panels(*joined) > share and len(pages) < count - 1:
                pages.append(current)
                left -= count_panels(*current)
                current = (strata, worlds)
            else:
                current = joined
        pages.append(current)
        if all(ok(page) for page in pages):
            return pages
    return list(units)


def group_pages(
    ctx: FigureContext, group: WorldGroup, fits: Fits = _any_height, panels: Panels | None = None
) -> list[Page]:
    """A group's pages: its worlds (with their strata) in order, never split, in the fewest pages of balanced
    size that each hold at most ``PAGE_PANELS`` panels and pass ``fits`` on the page's context; a world that
    fails alone gets a page of its own and the worlds between such worlds are packed as usual. ``panels``
    counts a page's panels from its strata and worlds (default: one per stratum)."""
    count_panels = panels or _one_per_stratum
    units: list[Page] = [
        (tuple(s for s in group.strata if s.world_id == w), (w,)) for w in group.worlds
    ]

    def ok(page: Page) -> bool:
        return count_panels(*page) <= PAGE_PANELS and fits(page_context(ctx, *page))

    pages: list[Page] = []
    run: list[Page] = []
    for unit in units:
        if ok(unit):
            run.append(unit)
            continue
        pages += _pack(run, ok, count_panels) if run else []
        pages.append(unit)
        run = []
    return pages + (_pack(run, ok, count_panels) if run else [])


def page_context(
    ctx: FigureContext, strata: Sequence[Stratum], worlds: Sequence[str]
) -> FigureContext:
    """``ctx`` with the study narrowed to ``strata`` and the runs of ``worlds``, ``whole`` the full study (labels
    and study-wide scales, ``FigureContext.full_study``); ``ctx`` itself when the page holds every stratum in
    study order and every world with a final run. The page keeps the study's ``Stratum`` objects, which the
    identity lookups (``FigureContext.group_of``, F8's cached picks) rely on."""
    study, chosen = ctx.study, set(worlds)
    every = {s.world_id for s in study.strata} | {cell[1] for cell in study.empty_cells}
    if tuple(strata) == study.strata and chosen >= every:
        return ctx
    narrowed = replace(
        study,
        runs=tuple(r for r in study.runs if r["world_id"] in chosen),
        strata=tuple(strata),
        empty_cells=tuple(c for c in study.empty_cells if c[1] in chosen),
    )
    return replace(ctx, study=narrowed, whole=ctx.full_study)


def grouped(
    ctx: FigureContext,
    build: Callable[[FigureContext], FigureOut],
    fits: Fits = _any_height,
    panels: Panels | None = None,
    *,
    worlds_only: bool = False,
) -> Iterator[FigureOut]:
    """An all-strata figure once per world group and page (``group_pages``), each built when the caller asks for
    it: ``<name>_<group>`` with several groups, ``_p<k>`` on several pages. A variant's sidecar gets
    ``extra["group"]`` (key, title, ``index`` the group's place in the group order, page, pages, ``variant`` the
    ungrouped name) when the export lists worlds or the group has several pages, and the ``worlds`` table among
    its inputs when the export lists worlds. A group without a stratum (its worlds' final runs all missing or
    excluded) is drawn only by a figure of worlds (``worlds_only``: F12). A context without groups is one group
    of the whole study."""
    study = ctx.study
    groups = ctx.groups or world_groups(study)
    named = len(groups) > 1
    for index, group in enumerate(groups):
        if not group.strata and not worlds_only:
            continue
        pages = group_pages(ctx, group, fits, panels)
        for number, (strata, worlds) in enumerate(pages, 1):
            built = build(page_context(ctx, strata, worlds))
            variant = built.name
            built.name += (f"_{group.key}" if named else "") + (
                f"_p{number}" if len(pages) > 1 else ""
            )
            if group.key is not None or len(pages) > 1:
                built.extra["group"] = dict(
                    key=group.key,
                    title=group.title or None,
                    index=index,
                    page=number,
                    pages=len(pages),
                    variant=variant,
                )
            if group.key is not None:
                built.inputs.add("worlds.parquet")
            yield built


def group_inputs(ctx: FigureContext, stratum: Stratum) -> set[str]:
    """The export files a per-stratum figure (F15, F17) reads for its label: the ``worlds`` table when the label
    names the stratum's group (``stratum_variants``)."""
    return set() if ctx.group_of(stratum) is None else {"worlds.parquet"}


def paired_strata(
    ctx: FigureContext, strata: Sequence[Stratum] | None = None
) -> list[tuple[Stratum, str, str]]:
    """Every stratum (of ``strata``, default the study's) and profile with a Default run to pair with, in study
    order: (stratum, profile, Default)."""
    found = []
    for stratum in ctx.study.strata if strata is None else strata:
        baseline = ctx.study.baselines.get(stratum.robot_id)
        if baseline is None or baseline not in stratum.profiles():
            continue
        found += [(stratum, p, baseline) for p in ctx.profiles(stratum) if p != baseline]
    return found


def estimator_text(effect: Effect) -> str:
    """The effect's primary estimator in words, for captions (F3, F13, F21)."""
    return "mean" if effect.estimator == "mean" else "Hodges–Lehmann"


def interval_methods(entries: Iterable[Mapping[str, Any]]) -> str:
    """How the drawn entries' intervals were computed, from each ``stats.json`` entry's ``interval.method`` (B and
    the seed where it resamples), for captions (F3, F13, F21); with several methods, each with the pair counts of
    its rows."""
    pairs_by_method: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        method = (entry.get("interval") or {}).get("method")
        if method:
            pairs_by_method[str(method)].append(int(entry.get("n_pairs") or 0))
    if not pairs_by_method:
        return "no interval in stats.json"
    if len(pairs_by_method) == 1:
        return next(iter(pairs_by_method))
    spans = []
    for method, counts in sorted(pairs_by_method.items(), key=lambda item: min(item[1])):
        low, high = min(counts), max(counts)
        spans.append(f"{method} ({low} pairs)" if low == high else f"{method} ({low}–{high} pairs)")
    return "; ".join(spans)


def interval_level(entry: Mapping[str, Any], requested: float) -> str | None:
    """An entry's achieved interval level as text when it is below the requested one (fewer pairs allow no more,
    METHODOLOGY §4.3; "no interval" at level 0, as ``tables.with_interval``), else None: printed in parentheses
    by F5 (c), F13 and F21 (F3 prints ``with_interval``)."""
    level = (entry.get("interval") or {}).get("level")
    if level is None or level >= requested - TIME_EPSILON:
        return None
    return "no interval" if level <= 0 else level_text(float(level))


def time_axis_end(stratum: Stratum) -> float:
    """Where a stratum's time axes end (FX-2, display only): at the latest end of its reported runs (a completed
    run's end, L for a run at the limit or one that never moved), or of an observed event time should one be
    stamped later, rounded up to a tick, and at L when a run ends at L. Estimates stay at L and the plan's
    horizons."""
    limit = stratum.limit_s
    ends = [
        limit if run["elapsed_s"] is None else run_end(run, limit)[0] for run in stratum.reported
    ]
    events = [
        float(v.value)
        for run in stratum.reported
        for endpoint in TIME_ENDPOINTS
        if (v := restricted_time(run, endpoint, limit)).event and v.value is not None
    ]
    latest = max([*ends, *events], default=limit)
    if latest <= 0 or latest >= limit - TIME_EPSILON:
        return limit
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, latest)
    return min(limit, min(float(t) for t in ticks if t >= latest - TIME_EPSILON))


def name_limit(ax: Any, limit: float) -> None:
    """Name an L that lies beyond the panel's time axis (FX-2), at the panel's top right, clear of the data."""
    ax.set_title(f"L = {limit:g} s", loc="right", fontsize=8, color=GREY)


def time_axis(
    ax: Any, stratum: Stratum, *, l_text: Literal["bottom", "top"] | None = None
) -> float:
    """A panel's time axis by FX-2: the dotted L line when the axis reaches L, with an "L" when ``l_text`` says
    where (``bottom``: beside the line near y = 2; ``top``: above the panel, over the line), else L named at
    the top right; the x limits to 1.04 × the axis end. Returns the axis end."""
    limit, end = stratum.limit_s, time_axis_end(stratum)
    if end < limit:
        name_limit(ax, limit)
    else:
        ax.axvline(limit, color=GREY, lw=0.6, ls=":")
        if l_text == "bottom":
            ax.text(limit, 2, " L", color=GREY, fontsize=8, ha="left", va="bottom")
        elif l_text == "top":
            ax.text(
                limit,
                1.0,
                "L",
                transform=ax.get_xaxis_transform(),
                color=GREY,
                fontsize=8,
                ha="center",
                va="bottom",
            )
    ax.set_xlim(0, end * 1.04)
    return end


AXIS_NOTE = (
    "Time axis: to the stratum's latest run end (a completed run's end; L for a run at the limit or one that "
    "never moved) rounded up to a tick, so to L when a run ends at L; an L beyond the axis is named in the panel "
    "(FX-2, display only: estimates stay at L and the plan's horizons)."
)


# F1 coverage over time ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trace:
    """One run's coverage as F1 draws it (F1, F22)."""

    # (times, values) of each step piece (``where="post"``), broken where valid samples are more than 2 s apart
    pieces: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]
    # a completed run's (end, final value), carried to L (RD-6); None otherwise
    carried: tuple[float, float] | None
    # a run that never moved: its pre-motion coverage, held from 0 to L (RD-8); None otherwise
    held: float | None


def coverage_trace(ctx: FigureContext, run: Row, stratum: Stratum) -> Trace:
    """One run's recorded steps, broken where valid samples are more than 2 s apart (nothing bridges a gap)."""
    limit = stratum.limit_s
    if run["outcome"] == "no_movement":
        return Trace([], None, ctx.study.pre_motion.get(run["attempt_id"]))
    series = ctx.study.coverage.get(run["attempt_id"])
    if series is None or not series.t.size:
        return Trace([], None, None)
    end, carried = run_end(run, limit)
    keep = series.t <= end + 1e-9
    t, c = series.t[keep], series.coverage[keep]
    breaks = np.nonzero(np.diff(t) > GAP_S)[0] + 1
    # the last stretch reaches the run's end only when that is within the gap rule; else the trace stops
    final = end if end - t[-1] <= GAP_S else t[-1]
    pieces = []
    for part_t, part_c in zip(np.split(t, breaks), np.split(c, breaks), strict=True):
        last = final if part_t[-1] == t[-1] else part_t[-1]
        pieces.append((np.append(part_t, last), np.append(part_c, part_c[-1])))
    forward = (end, float(c[-1])) if carried and end < limit and final == end else None
    return Trace(pieces, forward, None)


def _trace(ax: Any, run: Row, ctx: FigureContext, stratum: Stratum, style: ProfileStyle) -> None:
    limit = stratum.limit_s
    trace = coverage_trace(ctx, run, stratum)
    if trace.held is not None:
        held = trace.held
        ax.plot([0, limit], [held, held], color=style.color, lw=TRACE_WIDTH, ls=":", alpha=0.6)
    for steps_t, steps_c in trace.pieces:
        ax.step(steps_t, steps_c, where="post", color=style.color, lw=TRACE_WIDTH, alpha=0.45)
    if trace.carried is not None:
        end, value = trace.carried
        ax.plot(
            [end, limit], [value, value], color=style.color, lw=TRACE_WIDTH, ls="--", alpha=0.45
        )


@dataclass(frozen=True, slots=True)
class Band:
    """What ``coverage_band`` drew for one profile."""

    # runs still running at each grid time (the strip under F1's panels); None without runs on the grid
    running: np.ndarray[Any, Any] | None
    runs_with_values: int  # runs on the coverage grid
    drawn: bool  # some grid time had at least min_n runs with a value: the band and median show somewhere
    moved_median: bool  # the dotted median of the runs that moved was drawn (a run never moved)


def coverage_band(ax: Any, ctx: FigureContext, runs: list[Row], style: ProfileStyle) -> Band:
    """Pointwise median and IQR (type 2) where at least ``min_n`` runs have a value; the movement-only median
    as a dotted line when a run never moved. Reports the runs still running at each grid time and what showed."""
    cells = [ctx.grid[r["attempt_id"]] for r in runs if r["attempt_id"] in ctx.grid]
    if not cells:
        return Band(None, 0, False, False)
    grid_t = cells[0][0]  # one grid per stratum: every run shares its L
    moved = [r for r in runs if r["outcome"] != "no_movement"]
    drawn = moved_median = False
    for selection, primary in ((runs, True), (moved, False)):
        if not primary and len(moved) == len(runs):
            continue
        rows = [ctx.grid[r["attempt_id"]][1] for r in selection if r["attempt_id"] in ctx.grid]
        matrix = np.vstack(rows) if rows else np.full((1, grid_t.size), np.nan)
        support = np.sum(~np.isnan(matrix), axis=0)
        with np.errstate(all="ignore"):
            q = np.nanquantile(matrix, [0.25, 0.5, 0.75], axis=0, method="averaged_inverted_cdf")
        q[:, support < ctx.settings.min_n] = np.nan
        shows = bool(np.any(support >= ctx.settings.min_n))
        if primary:
            ax.fill_between(grid_t, q[0], q[2], step="post", color=style.color, alpha=0.18, lw=0)
            ax.step(grid_t, q[1], where="post", color=style.color, lw=1.2, ls=style.linestyle)
            drawn = shows
        else:
            ax.step(grid_t, q[1], where="post", color=style.color, lw=0.8, ls=":")
            moved_median = shows
    running = np.array(
        [sum(1 for cell in cells if cell[2][i] == "sample") for i in range(grid_t.size)]
    )
    return Band(running, len(cells), drawn, moved_median)


def coverage_panel(
    ax: Any,
    strip: Any,
    ctx: FigureContext,
    stratum: Stratum,
    profiles: Sequence[str] | None = None,
) -> set[str]:
    """One stratum's coverage panel (F1, F21, F22): every reported run of ``profiles`` (all of the stratum's by
    default), each profile's band, L, and the strip of running runs below. Returns the attempts drawn."""
    attempts: set[str] = set()
    largest = 1
    for profile in ctx.profiles(stratum) if profiles is None else profiles:
        style, runs = ctx.styles[profile], stratum.runs_of(profile)
        attempts |= {r["attempt_id"] for r in runs}
        for run in runs:
            _trace(ax, run, ctx, stratum, style)
        running = coverage_band(ax, ctx, runs, style).running
        if running is not None:
            grid_t = ctx.grid[runs[0]["attempt_id"]][0]
            strip.step(grid_t, running, where="post", color=style.color, lw=1.0, ls=style.linestyle)
        largest = max(largest, len(runs))
    end = time_axis(ax, stratum, l_text="bottom")
    ax.set(ylim=(0, 100), ylabel="Coverage (%)")
    ax.tick_params(labelbottom=False)
    strip.set(xlim=(0, end * 1.04), ylim=(0, largest * 1.15), ylabel="Running", xlabel=TIME_LABEL)
    strip.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    return attempts


def coverage_notes(ctx: FigureContext) -> list[str]:
    """How to read a coverage panel (F1, F21)."""
    return [
        "Thin steps: runs, broken where valid samples are more than 2 s apart; dashed: a completed run's final "
        "coverage carried to L (RD-6); dotted at a constant: a run that never moved, held at its pre-motion "
        "coverage (RD-8). Band: pointwise median and IQR (type 2), descriptive, only where at least "
        f"{ctx.settings.min_n} runs have a value under the gap rule; a dotted median: runs that moved only (RD-8, "
        "secondary). Strip: runs still running.",
        AXIS_NOTE,
    ]


def carried_handle() -> Line2D:
    """The legend entry of a completed run's coverage carried to L (F1, F21)."""
    return Line2D([], [], color=GREY, ls="--", lw=TRACE_WIDTH, label="carried forward to L")


def f1(ctx: FigureContext) -> FigureOut:
    fig, cells, width = stratum_grid(ctx, ctx.study.strata, strip=True)
    attempts: set[str] = set()
    for index, (stratum, (ax, strip)) in enumerate(zip(ctx.study.strata, cells, strict=True)):
        attempts |= coverage_panel(ax, strip, ctx, stratum)
        panel_title(ax, index, ctx.label(stratum))
    legend(fig, ctx, study_profiles(ctx), [carried_handle()])
    finish(fig, ctx)
    return FigureOut(
        "F1",
        fig,
        width,
        attempts,
        f"Coverage over time for every reported run in {len(ctx.study.strata)} strata, with the pointwise median "
        f"and interquartile band of each profile where at least {ctx.settings.min_n} runs have a value.",
        coverage_notes(ctx),
        {"axis_end_s": {ctx.label(s): time_axis_end(s) for s in ctx.study.strata}},
    )


# F2 attainment -------------------------------------------------------------------------------------------------


def endpoint_label(endpoint: str) -> str:
    """The event a restricted-time endpoint waits for, in words: "completion", "90 % coverage" or "99.9 %
    coverage"."""
    if endpoint == "R_completion":
        return "completion"
    return f"{milestone_label(MILESTONE_OF_ENDPOINT[endpoint])} coverage"


def attainment_panel(
    ax: Any,
    table: Any,
    ctx: FigureContext,
    stratum: Stratum,
    endpoint: str,
    *,
    table_wrap: int | None = None,
) -> set[str]:
    """One stratum's attainment panel (F2, F21): 1 − Ŝ(t) per profile, the Wilson intervals at the plan's
    horizons from ``stats.json``, and the at-risk table below. Returns the attempts of the entries drawn.

    On an axis that ends before L (FX-2) the last table column and the latest horizon beyond the axis are drawn at
    the axis end with their values at L: no run is still running there, so the share is the same."""
    attempts: set[str] = set()
    limit, end = stratum.limit_s, time_axis_end(stratum)
    columns = [(float(x), float(x)) for x in np.linspace(0, end, 5)]
    columns[-1] = (end, limit)  # (where, at which time): the last column counts at L
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
        offset = (position - (len(profiles) - 1) / 2) * end * 0.012
        for point in _attainment_points(entry, end):
            interval, share = point["interval"], point["k"] / n
            ax.errorbar(
                [min(point["horizon_s"], end) + offset],
                [100 * share],
                yerr=[  # a rounding error must not make a bar length negative
                    [100 * max(0.0, share - interval["lower"])],
                    [100 * max(0.0, interval["upper"] - share)],
                ],
                fmt=style.marker,
                color=style.color,
                ms=3,
                lw=0.8,
            )
        y = 1 - (position + 1) / (len(profiles) + 1)
        for x, t in columns:
            # runs still without the event at t; at L they are the runs not reached by L (FP-3)
            at_risk = sum(
                1 for v in usable if v.value is not None and (v.value >= t or not v.event)
            )
            text = f"{at_risk}/{n}" if t == limit else str(at_risk)
            table.text(x, y, text, ha="center", va="center", fontsize=8, color=style.color)
    time_axis(ax, stratum)
    ax.set(
        ylim=(0, 100),
        ylabel=f"Reached {endpoint_label(endpoint)} (%)",
        xlabel=TIME_LABEL,
    )
    table.set_axis_off()
    table.set(xlim=(0, end * 1.04), ylim=(0, 1))
    where = "at L" if end == limit else "last column"
    title = f"At risk ({where}: not reached by L)"
    # a narrow column (F21) breaks the table's title into lines
    table.set_title(
        title if table_wrap is None else wrap_text(title, table_wrap), loc="left", fontsize=8
    )
    return attempts


def _attainment_points(entry: dict[str, Any], end: float) -> list[dict[str, Any]]:
    """The entry's attainment points within the axis, and of those beyond it only the latest (L), which is drawn
    at the axis end; earlier ones beyond it hold the same share."""
    points = sorted((entry.get("attainment") or {}).values(), key=lambda p: p["horizon_s"])
    inside = [p for p in points if p["horizon_s"] <= end + TIME_EPSILON]
    return inside + points[len(inside) :][-1:]


def attainment_notes(ctx: FigureContext) -> list[str]:
    """How to read an attainment panel (F2, F21)."""
    return [
        f"Steps: 1 − Ŝ(t) (all censoring at L, so the share of runs). Points: Wilson {ctx.level_label} intervals at "
        "the horizons. Runs without movement count as never reaching (RD-8); failures and stopped runs are "
        "excluded and counted in T1. At-risk counts at 0, 1/4, 1/2 and 3/4 of the axis end and, in the last "
        "column, at L (k/n not reached by L). The latest horizon beyond the axis end (L) is drawn there: no run "
        "is still running after the axis end, so the share and its interval are those at L; earlier horizons "
        "beyond it have the same share and are not drawn.",
        AXIS_NOTE,
    ]


def f2(ctx: FigureContext, endpoint: str) -> FigureOut:
    fig, cells, width = stratum_grid(ctx, ctx.study.strata, strip=True)
    attempts: set[str] = set()
    for index, (stratum, (ax, table)) in enumerate(zip(ctx.study.strata, cells, strict=True)):
        attempts |= attainment_panel(ax, table, ctx, stratum, endpoint)
        panel_title(ax, index, ctx.label(stratum))
    legend(fig, ctx, study_profiles(ctx))
    finish(fig, ctx)
    return FigureOut(
        f"F2_{endpoint}",
        fig,
        width,
        attempts,
        f"Share of runs that reached {endpoint_label(endpoint)} by each time, per profile and stratum, with Wilson "
        "intervals at the plan's horizons and the number of runs still at risk.",
        attainment_notes(ctx),
        {"axis_end_s": {ctx.label(s): time_axis_end(s) for s in ctx.study.strata}},
    )


# F3 forest plots -----------------------------------------------------------------------------------------------


def _pooled_row(ctx: FigureContext, effect: Effect) -> dict[str, Any] | None:
    """A pre-specified pooled hypothesis drawn on this effect's axis: ΔC(h) and ΔA_h at L share the scale; ΔR
    is pooled as ΔR/L and gets no diamond on a seconds axis (T5H shows it). The pooled estimate
    (``estimates._pooled``) is the equal-weight average over the profile's robot's named worlds; a generated
    family's worlds are left out of it (``family_worlds_left_out``; a study of family worlds only has no testable
    pooled hypothesis). It is drawn only on a page that holds every stratum, so only for a study with one world
    group, never on one group's page of a study with several groups and never pooled across groups
    (plans/worlds/I8 §3.1)."""
    if len(ctx.study.strata) != len(ctx.full_study.strata):
        return None
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


F3_ROW_IN, F3_FRAME_IN = 0.22, 1.0  # one forest row; axes, titles and the evidence label


def _f3_height_in(rows: int) -> float:
    return F3_ROW_IN * max(1, rows) + F3_FRAME_IN


def _f3_fits(ctx: FigureContext, effect: Effect) -> bool:
    return _f3_height_in(len(_forest_rows(ctx, effect))) * MM_PER_INCH <= PAGE_MAX_MM


def f3(ctx: FigureContext, effect: Effect) -> FigureOut:
    rows = _forest_rows(ctx, effect)
    width_in = PRESETS[ctx.preset].widths_mm["double"] / MM_PER_INCH
    aspect = _f3_height_in(len(rows)) / width_in
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
    finish(fig, ctx)
    estimator = estimator_text(effect)
    methods = interval_methods(e for label, e, _, _ in rows if not label.startswith("pooled"))
    return FigureOut(
        f"F3_{effect.name}",
        fig,
        "double",
        attempts,
        f"Paired differences {effect.label} between each profile and Default per stratum, with intervals.",
        [
            f"Squares: the {estimator} estimate; lines: its {ctx.level_label} interval (a level in parentheses when "
            f"fewer pairs allow less), by {methods} (stats.json interval.method). Rows in study order; a diamond "
            "only for a pre-specified pooled hypothesis on this scale: the equal-weight mean over worlds of each "
            "world's mean difference, stratified bootstrap (RD-16), drawn only where the figure holds every stratum of "
            "the study (never on one world group's page)."
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


DISTRIBUTION_FLAT = 5  # strata whose labels sit side by side under a distribution panel


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
    # more strata than horizontal labels fit side by side (a world group of up to 10): labels upright
    upright = len(groups) > DISTRIBUTION_FLAT
    ax.set_xticks(range(len(groups)), labels, fontsize=8, rotation=90 if upright else 0)
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


def _cpu_ratios(ax: Any, ctx: FigureContext) -> tuple[set[str], int, list[dict[str, Any]]]:
    """(c) Per pair, the profile's CPU per simulated second over its block's Default run's, sequential pairs only
    (both runs alone on the host), on a log axis with the ``stats.json`` geometric-mean ratio and interval.
    Returns the attempts, the number of pairs drawn and the ``stats.json`` entries drawn."""
    read = METRICS["cpu_per_s"]
    # each panel restarts the jitter stream, so jitter is the same whatever other panels draw
    rng = stats.random_stream(ctx.settings.seed, stats.JITTER_STREAM)
    several = len({profile for _, profile, _ in paired_strata(ctx)}) > 1
    labels: list[str] = []
    attempts: set[str] = set()
    drawn = 0
    entries: list[dict[str, Any]] = []
    for stratum, profile, baseline in paired_strata(ctx):
        style, found = ctx.styles[profile], pairs(stratum, profile, baseline, "primary", read)
        ratios = [
            a / b
            for a, b in zip(found.profile_values, found.baseline_values, strict=True)
            if a > 0 and b > 0
        ]
        drawn += len(ratios)
        x = len(labels)
        ax.scatter(
            x + rng.uniform(-0.08, 0.08, len(ratios)),
            ratios,
            s=8,
            color=style.color,
            marker=style.marker,
            alpha=0.85,
            linewidths=0,
            zorder=3,
        )
        entry = ctx.entries.get("delta_cpu_per_s", stratum, profile, variant="ratio")
        entries.append(entry)
        interval = entry.get("interval") or {}
        if interval.get("lower") is not None and interval.get("upper") is not None:
            ax.plot([x + 0.25] * 2, [interval["lower"], interval["upper"]], color="#000000", lw=1.0)
        if entry.get("estimate") is not None:
            ax.plot([x + 0.25], [entry["estimate"]], marker="s", color="#000000", ms=3.5)
        dropped = len(found.differences) - len(ratios)  # a CPU value of 0: no ratio
        name = f"\n{style.name}" if several else ""
        level = interval_level(entry, 1 - ctx.settings.alpha)
        labels.append(
            f"{ctx.label(stratum)}{name}\nn = {len(ratios)}"
            + (f" (+{dropped} at 0)" if dropped else "")
            + (f"\n({level})" if level else "")
        )
        attempts |= set(found.attempt_ids)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=2, subs=(1.0, 1.5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1, color=GREY, lw=0.6)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlim(-0.6, max(len(labels), 1) - 0.4)
    ax.set_ylabel("CPU ratio, profile / Default")
    if not attempts:
        ax.set_ylim(0.5, 2)
        ax.text(
            0.5,
            0.75,
            "no sequential pairs",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=GREY,
        )
    panel_title(ax, 2, "paired ratio, sequential pairs")
    return attempts, drawn, entries


def f5(ctx: FigureContext) -> FigureOut:
    fig, axes = figure(
        ctx.preset, "double", aspect=0.4, ncols=3, gridspec_kw={"width_ratios": [1, 1, 1.1]}
    )
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
        panel_title(axes[0][index], index, "sequential runs")
    paired_attempts, cpu_pairs, ratio_entries = _cpu_ratios(axes[0][2], ctx)
    if any((e.get("interval") or {}).get("method") for e in ratio_entries):
        ratio_text = (
            f"square and line: the geometric-mean ratio and its {ctx.level_label} interval (a level in "
            f"parentheses under the column when fewer pairs allow less; by {interval_methods(ratio_entries)}; "
            "stats.json delta_cpu_per_s, ratio; T6), which has none when a value is 0."
        )
    else:
        ratio_text = "no interval in stats.json (no sequential pairs)."
    attempts |= paired_attempts
    legend(fig, ctx, study_profiles(ctx))
    finish(fig, ctx)
    return FigureOut(
        "F5",
        fig,
        "double",
        attempts,
        "Explorer CPU use per simulated second and peak memory per profile and stratum, and the paired CPU "
        "ratio against Default, sequential runs only.",
        [
            f"Points: runs; box (median, IQR type 2, whiskers 1.5 IQR) from {ctx.settings.min_n} runs; violin "
            f"from {VIOLIN_MIN}. Explorer process only; runs that overlapped another attempt or lack samples are "
            "left out and counted in stats.json. (c) one point per block whose two runs both ran alone with "
            "positive CPU values (a pair at 0 is counted under its column), log axis, line at 1; "
            + ratio_text,
        ],
        {"cpu_ratio_pairs": cpu_pairs},
    )


@dataclass(frozen=True, slots=True)
class Outcomes:
    """One stratum's final runs per profile, in study order (F6, F21)."""

    profiles: list[str]
    # per profile: reported runs by outcome, the other final runs by report class
    counts: list[dict[str, int]]
    # per profile: k, n and the Wilson interval (stats.json)
    success: list[tuple[int, int, dict[str, Any] | None]]
    attempt_ids: set[str]


def outcomes(ctx: FigureContext, stratum: Stratum) -> Outcomes:
    """Per profile of the stratum: its final runs by outcome (reported runs) or report class, and success."""
    profiles = ctx.profiles(stratum)
    counts: list[dict[str, int]] = []
    success: list[tuple[int, int, dict[str, Any] | None]] = []
    attempts: set[str] = set()
    for profile in profiles:
        finals = [r for r in stratum.finals if r["profile_id"] == profile]
        attempts.update(r["attempt_id"] for r in finals)
        by_class: dict[str, int] = defaultdict(int)
        for run in finals:
            by_class[
                run["outcome"] if run["report_class"] == "reported" else run["report_class"]
            ] += 1
        counts.append(by_class)
        entry = ctx.entries.get("success", stratum, profile)
        success.append((entry.get("k", 0), entry.get("n", 0), entry.get("interval")))
    return Outcomes(profiles, counts, success, attempts)


def outcome_bars(ax: Any, labels: list[str], counts: list[dict[str, int]]) -> None:
    """Horizontal stacked counts of final runs by outcome, one bar per label, each part labelled (F6, F21)."""
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
                edgecolor=EDGE,
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


def outcome_handles(counts: Iterable[dict[str, int]]) -> list[Patch]:
    """Legend entries of the outcome classes present in ``counts``, in the fixed class order (FP-4)."""
    present = list(counts)
    return [
        Patch(facecolor=v["facecolor"], hatch=v["hatch"], edgecolor=EDGE, label=v["label"])
        for k, v in OUTCOMES.items()
        if any(c.get(k) for c in present)
    ]


OUTCOME_NOTE = "Outcome classes follow reported-outcomes-v2 (greys and hatches, FP-4)."


def _outcomes(ax: Any, labels: list[str], counts: list[dict[str, int]]) -> None:
    outcome_bars(ax, labels, counts)
    used = outcome_handles(counts)
    ax.legend(handles=used, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=3)
    panel_title(ax, 0, "final runs by outcome")


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
    panel_title(ax, 1, "success (rows as in a)")


F6_ASPECT = 0.62
# (a) and (b): height per bar, and for titles, axis and legend
F6_BAR_MM, F6_BARS_EXTRA_MM = 4.5, 20.0


def f6(ctx: FigureContext) -> FigureOut:
    rows = sum(len(ctx.profiles(s)) for s in ctx.study.strata)  # the bars of (a) and (b)
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    half_mm = F6_ASPECT * width_mm / 2
    top_mm = F6_BAR_MM * rows + F6_BARS_EXTRA_MM
    if top_mm <= half_mm:
        fig, axes = figure(ctx.preset, "double", aspect=F6_ASPECT, nrows=2, ncols=2)
    else:  # many bars (a world group): the top row grows, the bottom row keeps its height
        fig, axes = figure(
            ctx.preset,
            "double",
            aspect=(top_mm + half_mm) / width_mm,
            nrows=2,
            ncols=2,
            gridspec_kw={"height_ratios": [top_mm, half_mm]},
        )
    attempts: set[str] = set()
    labels: list[str] = []
    counts: list[dict[str, int]] = []
    success: list[tuple[int, int, dict[str, Any] | None]] = []
    for stratum in ctx.study.strata:
        found = outcomes(ctx, stratum)
        attempts |= found.attempt_ids
        labels += [f"{ctx.label(stratum)} · {ctx.styles[p].name}" for p in found.profiles]
        counts += found.counts
        success += found.success
    _outcomes(axes[0][0], labels, counts)
    _success(axes[0][1], ctx, success)
    _distribution(
        axes[1][0], ctx, _groups(ctx, lambda run, s: time_value(run, "R_completion", s.limit_s))
    )
    axes[1][0].set_ylabel("R_completion (s)")
    panel_title(axes[1][0], 2, "R_completion (△: not reached by L, at L)")
    delta = ctx.settings.stall_delta_pp
    _distribution(
        axes[1][1], ctx, _groups(ctx, lambda run, s: idle_tail(ctx.study, run, s.limit_s, delta))
    )
    axes[1][1].set_ylabel("Idle tail (s)")
    panel_title(axes[1][1], 3, "time after the last coverage gain")
    legend(fig, ctx, study_profiles(ctx))
    finish(fig, ctx)
    return FigureOut(
        "F6",
        fig,
        "double",
        attempts,
        "Outcomes, success with Wilson intervals, restricted completion times and idle tails per profile and "
        "stratum.",
        [
            f"{OUTCOME_NOTE} Triangles: runs that did not complete by L, drawn at L; boxes over all restricted "
            f"times. Idle tail: min(T_end, L) − last gain (stall-v1, δ = {delta:g} pp)."
        ],
    )


# F8 representative paths -------------------------------------------------------------------------------------


def _hash_order(block_id: str) -> str:
    """The tie-break of the representative rules (SEL-1 to SEL-3, F8): the SHA-256 of the block ID."""
    return hashlib.sha256(block_id.encode()).hexdigest()


def representatives(ctx: FigureContext, stratum: Stratum) -> list[tuple[Row, str]]:
    """The runs the plan's rule picks in one stratum (FIGURE_STYLE §8), each with its panel note.

    SEL-1: the Default run whose restricted time is closest to the median of the stratum's reported Default
    runs. SEL-2: the block whose paired difference (``stats.json``: the first other profile in parameter-hash
    order − Default, reported pairs) is closest to the estimate; both runs. SEL-3: the SEL-1 run and also the
    most adverse Default run that moved (the longest restricted time): when the SEL-1 run moved, only one
    strictly longer than it (else the SEL-1 run alone, noted: it is itself the most adverse, or tied with it at
    L); when the SEL-1 run never moved, the longest run that moved, labelled so ("no run moved" when none did).
    Ties: the SHA-256 of the block ID; a pick tied with other Default runs at L (not reached by L) is labelled
    "one of k Default runs tied at L (hash tie-break)", k counting the runs it was picked from (when the SEL-1 run
    is also SEL-3's, the runs that moved among them are named too). A SEL-2 study without a usable pair falls back
    to SEL-1, noted."""
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
                    _hash_order(b),
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
    chosen = min(defaults, key=lambda r: (abs(restricted(r) - median), _hash_order(r["block_id"])))

    def tied_at_limit(run: Row, among: Sequence[Row]) -> str:
        """ ", one of k Default runs tied at L (hash tie-break)" when ``run`` is at L with others of ``among``."""
        if restricted(run) < limit - TIME_EPSILON:
            return ""
        k = sum(1 for r in among if restricted(r) >= limit - TIME_EPSILON)
        return f", one of {k} Default runs tied at L (hash tie-break)" if k > 1 else ""

    chosen_tie = tied_at_limit(chosen, defaults)
    note = f"SEL-1{chosen_tie}"
    if rule == "SEL-2":  # no pair to show: no other profile, or none with an estimate and both runs
        with_whom = f"with {ctx.styles[others[0]].name}" if others else "(no other profile)"
        note = f"SEL-1{chosen_tie} (SEL-2: no usable pair {with_whom})"
    # SEL-3 (FIGURE_STYLE §8, the most adverse Default run that moved): when the SEL-1 run moved, a second run
    # only when one that moved is strictly longer (never the same run twice, never a shorter one); when the SEL-1
    # run never moved, the run that moved with the longest restricted time
    chosen_moved = chosen["outcome"] != "no_movement"
    moved = [
        r
        for r in defaults
        if r["outcome"] != "no_movement"
        and r["attempt_id"] != chosen["attempt_id"]
        and (not chosen_moved or restricted(r) > restricted(chosen) + TIME_EPSILON)
    ]
    if rule == "SEL-3" and not moved:
        reason = "is also the most adverse run that moved" if chosen_moved else "no run moved"
        note = f"SEL-1{chosen_tie} (SEL-3: {reason})"
        if chosen_moved and chosen_tie:
            # tied at L and the most adverse: one label for both rules, with SEL-3's k over the runs that moved
            at_limit = [r for r in defaults if restricted(r) >= limit - TIME_EPSILON]
            k, k_moved = len(at_limit), sum(r["outcome"] != "no_movement" for r in at_limit)
            of_moved = "" if k_moved == k else f", {k_moved} of them moved"
            note = f"SEL-1 and SEL-3, one of {k} Default runs tied at L{of_moved} (hash tie-break)"
    picked = [(chosen, note)]
    if rule == "SEL-3" and moved:
        worst = min(moved, key=lambda r: (-restricted(r), _hash_order(r["block_id"])))
        label = "most adverse run (SEL-3)" if chosen_moved else "longest run that moved (SEL-3)"
        picked.append((worst, label + tied_at_limit(worst, moved)))
    return picked


Extent = tuple[float, float, float, float]
# map panels show the reference's extent plus this margin on every side (FIGURE_STYLE §6 item 5)
REFERENCE_PAD_M = 0.5


class MapCode(IntEnum):
    """The cell codes of a reference drawn as a map (FP-5 greys, in this order)."""

    FREE = 0  # free floor of the reference (its target cells)
    UNKNOWN = 1  # inside the reference, not free floor
    OCCUPIED = 2  # obstacle (collision) cells
    OUTSIDE = 3  # outside the reference, hatched


PROVENANCE = "provenance.json"  # maps each run to its coverage reference (``reference_name``)


def export_reference_name(export: Export, attempt: str) -> str | None:
    """The coverage reference directory the run used (``provenance.json`` maps each to its runs), if exported."""
    references = export.provenance.get("references") or {}
    name = next((ref for ref, runs in references.items() if attempt in runs), None)
    if name is None or f"references/{name}/reference.yaml" not in export.files:
        return None
    return str(name)


def reference_name(ctx: FigureContext, attempt: str) -> str | None:
    """``export_reference_name`` of the figure's export. A figure that calls it lists ``reference_files`` in its
    inputs."""
    return export_reference_name(ctx.study.export, attempt)


def reference_meta(ctx: FigureContext, name: str) -> dict[str, Any]:
    """A reference's ``reference.yaml`` (origin, resolution, target cells and area, metric version)."""
    path = ctx.study.export.directory / "references" / name / "reference.yaml"
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def reference_files(ctx: FigureContext, name: str | None) -> set[str]:
    """The export files a figure reads to draw reference ``name`` (``reference_name``, ``reference_meta``,
    ``load_reference``): its directory's files and ``provenance.json``; only the latter for None (a run without
    an exported reference, or a figure that looked one up and drew none). For ``FigureOut.inputs``."""
    files = set(ctx.study.export.files)
    found = {PROVENANCE} & files
    if name is not None:
        found |= {f for f in files if f.startswith(f"references/{name}/")}
    return found


def load_reference(ctx: FigureContext, attempt: str) -> tuple[np.ndarray[Any, Any], Extent] | None:
    """The coverage reference the run used, as map codes (row 0 the smallest y) and its extent in metres; None
    when it is not exported or its origin is rotated (the extent assumes yaw 0, as ``streams.reference_cells``
    does and every reference so far has): panels then say the reference is not readable."""
    name = reference_name(ctx, attempt)
    if name is None:
        return None
    reference = ctx.study.export.directory / "references" / name
    meta = reference_meta(ctx, name)
    origin = meta["origin"]
    if len(origin) > 2 and float(origin[2]) != 0.0:
        return None
    masks = {
        key: np.array(Image.open(reference / f"{key}.pgm")) == 255
        for key in ("target", "collision", "inclusion")
    }
    codes = np.full(masks["target"].shape, MapCode.OUTSIDE, dtype=np.int8)
    codes[masks["inclusion"]] = MapCode.UNKNOWN
    codes[masks["target"]] = MapCode.FREE
    codes[masks["collision"]] = MapCode.OCCUPIED
    x0, y0 = meta["origin"][:2]
    height, width = codes.shape
    extent = (x0, x0 + width * meta["resolution"], y0, y0 + height * meta["resolution"])
    return codes[::-1], extent  # PGM row 0 is the largest y


Path2D = tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], float | None]


def ground_truth(ctx: FigureContext, attempt: str, end: float) -> Path2D | None:
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


def draw_reference(
    ax: Any, codes: np.ndarray[Any, Any], extent: Extent, *, bar_zorder: float = 5.0
) -> AnchoredSizeBar:
    """A reference map in the map greys with the outside hatched (FP-5), equal axes, and a 5 m bar, the same in
    every panel (FIGURE_STYLE §6): its label on a white box, the bar at ``bar_zorder`` (text and scale bar 5;
    raise it above marks drawn higher). Returns the bar."""
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
        codes == MapCode.OUTSIDE,
        levels=[0.5, 1.5],
        colors="none",
        hatches=["////"],
        zorder=0.5,
    )
    ax.set_xlim(extent[0] - REFERENCE_PAD_M, extent[1] + REFERENCE_PAD_M)
    ax.set_ylim(extent[2] - REFERENCE_PAD_M, extent[3] + REFERENCE_PAD_M)
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
    bar.set_zorder(bar_zorder)
    # the label's Text (a TextArea's only child) on a white box, readable over walls and marks
    bar.txt_label.get_children()[0].set_bbox(dict(facecolor="#FFFFFF", edgecolor="none", pad=1.0))
    ax.add_artist(bar)
    return bar


def path_norm(ctx: FigureContext) -> Normalize:
    """The time scale of every path and curve coloured by time: 0 to the whole study's longest L, the same on
    every group page (F8, F22)."""
    return Normalize(0, max(s.limit_s for s in ctx.full_study.strata))


def time_lines(segments: Sequence[Any], times: Any, norm: Normalize) -> LineCollection:
    """Line segments coloured by ``times`` on ``norm`` (``SEQUENTIAL_CMAP``), 1.0 pt with round ends and the 0.5 pt dark casing
    of paths (FIGURE_STYLE §6 item 3), above the map layers; the caller adds it to its axes."""
    lines = LineCollection(
        list(segments),
        cmap=plt.get_cmap(SEQUENTIAL_CMAP),
        norm=norm,
        linewidths=1.0,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    lines.set_array(times)
    lines.set_path_effects(
        [patheffects.Stroke(linewidth=2.0, foreground="#333333"), patheffects.Normal()]
    )
    return lines


def time_bar(fig: Figure, axes: Sequence[Any], norm: Normalize, mappable: Any = None) -> None:
    """One horizontal colour bar under ``axes`` for the time colours (``mappable``, or ``SEQUENTIAL_CMAP`` on ``norm``),
    labelled with the clock (FIGURE_STYLE §6 item 6)."""
    found = (
        mappable
        if mappable is not None
        else ScalarMappable(norm=norm, cmap=plt.get_cmap(SEQUENTIAL_CMAP))
    )
    bar = fig.colorbar(found, ax=axes, orientation="horizontal", fraction=0.05, pad=0.02, aspect=40)
    bar.set_label(TIME_LABEL)


def map_panel(
    ax: Any, ctx: FigureContext, run: Row, stratum: Stratum, norm: Normalize
) -> LineCollection | None:
    """One run on the coverage reference it used (F8, F22): the ground-truth path to min(T_end, L) coloured by time
    on ``norm`` with a dark casing, the start pose as an arrow and the end as a square. Returns the path, or None
    (and says so in the panel) without a reference or a path."""
    reference = load_reference(ctx, run["attempt_id"])
    end, _ = run_end(run, stratum.limit_s)
    path = ground_truth(ctx, run["attempt_id"], end)
    if reference is None or path is None:
        text = "coverage reference not readable" if path is not None else "no reference or path"
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", fontsize=8)
        return None
    draw_reference(ax, *reference)
    times, xs, ys, yaw = path
    segments = np.stack(
        [np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1
    )
    lines = time_lines(list(segments), (times[:-1] + times[1:]) / 2, norm)
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
    return lines


Picks = list[tuple[Stratum, list[tuple[Row, str]]]]
F8_ROW_ASPECT = 0.45  # height of one row of F8's maps per unit of width


def _f8_layout(ctx: FigureContext, picks: Picks) -> tuple[int, int, list[int], str]:
    """F8's rows, columns, the cell of each pick in order, and the width."""
    panels = sum(len(chosen) for _, chosen in picks)
    if ctx.settings.representative_rule in ("SEL-2", "SEL-3") and panels > 1:
        # one row per stratum, a column per pick (SEL-2 and SEL-3 pick one or two runs), unused cells hidden
        rows, columns = len(picks), max(len(chosen) for _, chosen in picks)
        cells = [
            line * columns + slot
            for line, (_, chosen) in enumerate(picks)
            for slot in range(len(chosen))
        ]
    else:
        rows, columns = _grid_size(max(1, panels))
        cells = list(range(panels))
    return rows, columns, cells, "double" if panels > 1 else "single"


def _f8_height_mm(ctx: FigureContext, picks: Picks) -> float:
    rows, _, _, width = _f8_layout(ctx, picks)
    return PRESETS[ctx.preset].widths_mm[width] * F8_ROW_ASPECT * rows


def f8(ctx: FigureContext) -> FigureOut:
    picks = [(s, representatives(ctx, s)) for s in ctx.study.strata]
    panels = [(s, run, note) for s, chosen in picks for run, note in chosen]
    rows, columns, cells, width = _f8_layout(ctx, picks)
    fig, axes = figure(ctx.preset, width, aspect=F8_ROW_ASPECT * rows, nrows=rows, ncols=columns)
    flat = [axes[r][c] for r in range(rows) for c in range(columns)]
    norm = path_norm(ctx)
    chosen_runs: dict[str, str] = {}
    inputs = reference_files(ctx, None)
    mappable = None
    for ax in flat:
        ax.set_axis_off()
    # a long note (a tie at L) breaks into lines of the column's width
    chars = int(PRESETS[ctx.preset].widths_mm[width] / columns / TITLE_CHAR_MM)
    for index, ((stratum, run, note), cell) in enumerate(zip(panels, cells, strict=True)):
        panel_title(flat[cell], index, f"{ctx.label(stratum)} · {note}", wrap=chars)
        mappable = map_panel(flat[cell], ctx, run, stratum, norm) or mappable
        chosen_runs[f"{ctx.label(stratum)} · {note}"] = run["attempt_id"]
        inputs |= reference_files(ctx, reference_name(ctx, run["attempt_id"]))
    if mappable is not None:
        time_bar(fig, flat, norm, mappable)
    finish(fig, ctx)
    rule, endpoint = ctx.settings.representative_rule, ctx.settings.representative_endpoint
    return FigureOut(
        "F8",
        fig,
        width,
        set(chosen_runs.values()),
        "Ground-truth path of the representative runs of each stratum on the coverage reference, coloured by time.",
        [
            f"Runs chosen by {rule} on {endpoint} (FIGURE_STYLE §8): "
            + ", ".join(f"{k}: {v}" for k, v in chosen_runs.items())
            + ". Map: free floor white, other inside cells light grey, obstacles dark grey, outside the reference "
            "hatched. Arrow: start pose; square: end. Path to min(T_end, L) from the export's 1 s ground-truth rows."
        ],
        {"representative": chosen_runs, "rule": rule, "endpoint": endpoint},
        inputs=inputs,
    )


def figures(ctx: FigureContext) -> Iterator[FigureOut]:
    """F1–F8, each built when the caller asks for it, so the caller saves and closes one before the next; F1, F2,
    F3, F5, F6 and F8 once per world group and page (``grouped``; F8's panels are its picks)."""
    yield from grouped(ctx, f1, grid_fits)
    for endpoint in time_endpoints(ctx.settings):
        yield from grouped(ctx, partial(f2, endpoint=endpoint), grid_fits)
    for effect in EFFECTS:
        yield from grouped(ctx, partial(f3, effect=effect), partial(_f3_fits, effect=effect))
    yield from grouped(ctx, f5)
    yield from grouped(ctx, f6)
    # the picks of every stratum, by identity (pages keep the study's Stratum objects, which are not hashable)
    picks = {id(s): representatives(ctx, s) for s in ctx.study.strata}

    def f8_fits(page: FigureContext) -> bool:
        chosen = [(s, picks[id(s)]) for s in page.study.strata]
        return _f8_height_mm(page, chosen) <= PAGE_MAX_MM

    def f8_panels(strata: Sequence[Stratum], worlds: Sequence[str]) -> int:
        return sum(len(picks[id(s)]) for s in strata)

    yield from grouped(ctx, f8, f8_fits, f8_panels)
