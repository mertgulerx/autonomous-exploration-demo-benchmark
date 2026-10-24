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

"""Figures of the family estimands and the moving-object density contrast (``family-sample-v1``;
plans/worlds/I6, I8 §3.3 and §3.7; FIGURE_STYLE §4, §5, §9).

- F24 (one per family, ``F24_<family>``): one panel per outcome (rho = R/L, success, C(L), A_L); each world's
  reported runs as points, its mean as a bar mark with its n, and the family average with its t interval where
  the family has one ("approximate; design points treated as exchangeable"), its estimate alone where it is
  descriptive, and "not estimable" written where the world means give no interval (never a zero width). Worlds
  are labelled by their design levels only; worlds without a value are named under the panel.
- F24b: the family profile contrasts (profile − Default, the mean over worlds of each world's mean block
  difference) as a forest, one row per family and profile, never pooled across families.
- F25 (one per family with factorial effects or pairs, ``F25_<family>``): the factorial effects on rho as a dot
  plot with Lenth's ME and SME as reference lines (labelled approximate) where stats.json gives them; effects
  without them and the declared pairs as points without lines, labelled descriptive.
- F28 (one per density contrast, ``F28_<contrast>``): an estimation plot per outcome (rho, C(L)): each spawn's
  runs in the base and the moving world with their means (repetitions unpaired), and the difference moving −
  base with its stratified Welch interval (primary, labelled approximate, with its count of cells without
  variation and the ceiling limitation of METHODOLOGY §4.9) beside the ``exclusions_as_failures`` sensitivity.

They read only ``stats.json``'s family entries (primary population, at L, unless said otherwise) and compute
nothing statistical; a study without family entries gets none of them. The sidecars name the drawn entries'
IDs (``entries``), so every value is a ``stats.json#<id>``.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np

from .families import CONTRACT
from .figures import (
    GREY,
    FigureContext,
    FigureOut,
    finish,
    legend,
    panel_message,
    panel_title,
    wrap_text,
)
from .style import MM_PER_INCH, PRESETS, figure
from .tables import (
    DASH,
    FAMILY_OUTCOMES,
    NOT_ESTIMABLE,
    WELCH_CEILING,
    declared_family,
    design_keys,
    design_label,
    effect_label,
    effect_name,
    effect_sort,
    family_ids,
    family_rank,
    family_title,
    family_value,
    welch_note,
    world_order,
)
from .worlds import GROUP_ORDER

APPROXIMATE = "approximate; design points treated as exchangeable"  # families.APPROXIMATE, as I8 §3.3 words it
TITLES = {
    "rho": "ρ = R/L",
    "success": "Success: completed within L (%)",
    "C_L": "C(L): coverage at L (%)",
    "A_L": "A_L: mean coverage to L (%)",
}
SPANS = {"rho": 1.0, "success": 100.0, "C_L": 100.0, "A_L": 100.0}  # the axis range of each outcome
TICK_CHARACTERS = 12  # a design level wraps at about this many characters under its world
SMALL_PT = 7  # n labels and in-panel notes
F24_PANEL_MM, F24_FRAME_MM, TICK_LINE_MM = 34.0, 22.0, 3.0
ROW_IN, FRAME_IN = (
    0.24,
    1.1,
)  # a forest or dot-plot row; axes, titles, legend and the evidence label


def _entries(
    ctx: FigureContext,
    estimand: str,
    kind: str,
    unit: str,
    population: str = "primary",
    **fields: Any,
) -> list[dict[str, Any]]:
    """The unit's ``family-sample-v1`` entries of ``estimand`` at L in ``population``."""
    return list(
        ctx.entries.where(
            contract=CONTRACT,
            estimand=estimand,
            population=population,
            horizon_fraction=None,
            **{kind: unit},
            **fields,
        )
    )


def _profiles(ctx: FigureContext, entries: Iterable[Mapping[str, Any]]) -> list[str]:
    present = {e["profile_id"] for e in entries}
    return [p for p in ctx.order if p in present]


def _attempts(entries: Iterable[Mapping[str, Any]]) -> set[str]:
    return {a for e in entries for a in e.get("attempt_ids", [])}


def _interval_drawn(entry: Mapping[str, Any]) -> bool:
    interval = entry.get("interval") or {}
    return interval.get("lower") is not None and interval.get("upper") is not None


def _not_estimable(entry: Mapping[str, Any]) -> str | None:
    """The interval's "not estimable: …" note, if it has one."""
    note = str((entry.get("interval") or {}).get("note") or "")
    return note if note.startswith(NOT_ESTIMABLE) else None


def _group(family: str, variant: str) -> dict[str, Any]:
    """The sidecar's ``group`` of a per-family figure, so the Study tab labels it with the family's title and
    orders it by the group order (``outputs.views_document``)."""
    index = GROUP_ORDER.index(family) if family in GROUP_ORDER else len(GROUP_ORDER)
    return dict(
        key=family, title=family_title(family), index=index, page=1, pages=1, variant=variant
    )


def _spread(count: int, room: float) -> np.ndarray[Any, Any]:
    """Horizontal offsets of a world's run points: evenly spaced within ``room``, fixed (no random jitter)."""
    if count < 2:
        return np.zeros(count)
    return np.linspace(-room, room, count)


# F24 family world means ----------------------------------------------------------------------------------------


def _panel_note(averages: Sequence[Mapping[str, Any]], outcome: str) -> str:
    """What the family-average column of a panel shows, for its title."""
    if outcome == "success" or all(e.get("label") for e in averages):
        return "family average descriptive, no interval"
    found = [n for e in averages if (n := _not_estimable(e))]
    if found and not any(_interval_drawn(e) for e in averages):
        return f"family average: interval {found[0]}"
    return f"family average with t interval, {APPROXIMATE}"


def f24(ctx: FigureContext, family: str, inputs: Iterable[str]) -> FigureOut:
    means = _entries(ctx, "family_world_mean", "world_family", family)
    averages = _entries(ctx, "family_average", "world_family", family)
    profiles = _profiles(ctx, means)
    worlds = world_order(ctx.groups, family, (str(e["world_id"]) for e in means))
    points = {str(e["world_id"]): e.get("design_point") or {} for e in means}
    keys = design_keys(declared_family(ctx.settings, family), points.values())
    ticks = [
        "\n".join(wrap_text(design_label(points[w], [k]), TICK_CHARACTERS) for k in keys)
        for w in worlds
    ]
    lines = max((t.count("\n") + 1 for t in ticks), default=1)
    outcomes = list(FAMILY_OUTCOMES)
    height_mm = F24_PANEL_MM * len(outcomes) + F24_FRAME_MM + TICK_LINE_MM * lines
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    fig, axes = figure(
        ctx.preset, "double", aspect=height_mm / width_mm, nrows=len(outcomes), sharex=True
    )
    by_key = {(str(e["world_id"]), e["profile_id"], e["endpoint"]): e for e in means}
    average_of = {(e["profile_id"], e["endpoint"]): e for e in averages}
    count = len(worlds)
    family_x = count + 0.3
    slot = 0.7 / max(1, len(profiles))
    drawn: list[str] = []
    for index, outcome in enumerate(outcomes):
        ax = axes[index][0]
        _, _, scale = FAMILY_OUTCOMES[outcome]
        span = SPANS[outcome]
        n_row = blended_transform_factory(ax.transData, ax.transAxes)
        missing: list[str] = []
        for number, profile in enumerate(profiles):
            style = ctx.styles[profile]
            offset = (number - (len(profiles) - 1) / 2) * slot
            for x, world in enumerate(worlds):
                e = by_key.get((world, profile, outcome), {})
                if not e:
                    continue
                drawn.append(e["id"])
                values = [scale * float(v["value"]) for v in e.get("values", [])]
                xs = x + offset + _spread(len(values), min(slot * 0.3, 0.03 * len(values)))
                ax.plot(
                    xs,
                    values,
                    ls="none",
                    marker=style.marker,
                    ms=3,
                    mfc="none",
                    mec=style.color,
                    mew=0.6,
                )
                if e.get("estimate") is not None:
                    level = scale * float(e["estimate"])
                    ax.plot(
                        [x + offset - slot * 0.4, x + offset + slot * 0.4],
                        [level, level],
                        color=style.color,
                        lw=1.6,
                        solid_capstyle="butt",
                    )
                else:
                    missing.append(design_label(points[world], keys, " / "))
                ax.text(
                    x + offset,
                    0.02,
                    str(e.get("n", 0)),
                    transform=n_row,
                    ha="center",
                    va="bottom",
                    fontsize=SMALL_PT,
                    color=GREY,
                )
            average = average_of.get((profile, outcome), {})
            if not average:
                continue
            drawn.append(average["id"])
            column = family_x + offset
            estimate = average.get("estimate")
            with_interval = _interval_drawn(average)
            if with_interval:
                interval = average["interval"]
                ax.plot(
                    [column, column],
                    [scale * interval["lower"], scale * interval["upper"]],
                    color=style.color,
                    lw=1.2,
                )
            if estimate is not None:
                ax.plot(
                    [column],
                    [scale * float(estimate)],
                    ls="none",
                    marker=style.marker,
                    ms=5,
                    color=style.color,
                    mfc=style.color if with_interval else "white",
                )
            if not with_interval and _not_estimable(average):
                ax.text(
                    column + slot * 0.25,
                    0.5 * span,
                    NOT_ESTIMABLE,
                    rotation=90,
                    ha="left",
                    va="center",
                    fontsize=SMALL_PT,
                    color=GREY,
                )
            ax.text(
                column,
                0.02,
                f"{average.get('n_worlds', 0)} w",
                transform=n_row,
                ha="center",
                va="bottom",
                fontsize=SMALL_PT,
                color=GREY,
            )
        ax.axvline(count - 0.35, color=GREY, lw=0.5)
        ax.set_ylim(-0.17 * span, 1.08 * span)
        ax.set_yticks(np.linspace(0, span, 5))
        ax.set_ylabel(TITLES[outcome].split(":")[0])
        mine = [average_of[(p, outcome)] for p in profiles if (p, outcome) in average_of]
        panel_title(ax, index, f"{TITLES[outcome]} · {_panel_note(mine, outcome)}")
        label = "World, by its levels of " + " / ".join(keys) if index == len(outcomes) - 1 else ""
        if missing:
            names = "; ".join(dict.fromkeys(missing))
            label = f"{label}\nWithout a value: {names}" if label else f"Without a value: {names}"
        ax.set_xlabel(label)
    bottom = axes[-1][0]
    bottom.set_xticks([*range(count), family_x], [*ticks, "Family\naverage"], fontsize=SMALL_PT)
    bottom.set_xlim(-0.6, family_x + 0.6)
    extra = [
        Line2D([], [], ls="none", marker="o", mfc="none", mec=GREY, ms=3, label="run"),
        Line2D([], [], color=GREY, lw=1.6, label="world mean"),
        Line2D([], [], color=GREY, marker="o", ms=5, label="family average, t interval"),
        Line2D(
            [], [], ls="none", marker="o", ms=5, mfc="white", mec=GREY, label="without interval"
        ),
    ]
    legend(fig, ctx, profiles, extra, max_columns=4)
    finish(fig, ctx)
    title = family_title(family)
    world_ids = {w: design_label(points[w], keys, " / ") for w in worlds}
    return FigureOut(
        f"F24_{family}",
        fig,
        "double",
        _attempts(means),
        f"{title}: each world's reported runs, world means and the family average with its interval, per outcome "
        f"(rho, success, C(L), A_L) and profile.",
        [
            "Points: the reported runs of each world (primary population, at L); bars: the world mean over its "
            "spawns (each spawn weighted equally) with its n under it; worlds labelled by their design levels "
            f"({' / '.join(keys)}), never pooled with another family.",
            f"Family average: the equal-weight mean of the world means (w: worlds with a value) with a "
            f"{ctx.level_label} t interval on the world means (n − 1 df), {APPROXIMATE} (plans/worlds/I6 §3.4); "
            "success and families without an interval descriptive (open marker); an interval the world means "
            f"cannot give is written {NOT_ESTIMABLE}, never drawn.",
            "Worlds without a value are named under their panel; the family average is conditional on the "
            "remaining worlds (stats.json family_average note).",
        ],
        dict(
            group=_group(family, "F24"),
            family=family,
            worlds=world_ids,
            entries=sorted(set(drawn)),
        ),
        inputs=set(inputs),
    )


# F24b family profile contrasts ---------------------------------------------------------------------------------


def f24b(ctx: FigureContext, inputs: Iterable[str]) -> FigureOut | None:
    found = [
        e
        for family in family_ids(ctx.entries)
        for e in _entries(ctx, "family_profile_contrast", "world_family", family)
    ]
    if not found:
        return None
    rows = sorted(
        {(str(e["world_family"]), e["profile_id"]) for e in found},
        key=lambda r: (family_rank(r[0]), ctx.order.index(r[1]) if r[1] in ctx.order else 0),
    )
    lookup = {(str(e["world_family"]), e["profile_id"], e["endpoint"]): e for e in found}
    outcomes = list(FAMILY_OUTCOMES)
    width_in = PRESETS[ctx.preset].widths_mm["double"] / MM_PER_INCH
    height_in = 2 * ROW_IN * len(rows) + 2 * FRAME_IN
    fig, axes = figure(
        ctx.preset, "double", aspect=height_in / width_in, nrows=2, ncols=2, sharey=True
    )
    labels = [f"{family_title(f)} · {ctx.styles[p].name}" for f, p in rows]
    for index, outcome in enumerate(outcomes):
        ax = axes[index // 2][index % 2]
        name, _, scale = FAMILY_OUTCOMES[outcome]
        for y, (family, profile) in enumerate(reversed(rows)):
            e = lookup.get((family, profile, outcome), {})
            if not e:
                continue
            style = ctx.styles[profile]
            estimate = e.get("estimate")
            if _interval_drawn(e):
                ax.plot(
                    [scale * e["interval"]["lower"], scale * e["interval"]["upper"]],
                    [y, y],
                    color=style.color,
                    lw=1.2,
                )
            if estimate is not None:
                ax.plot(
                    [scale * float(estimate)],
                    [y],
                    ls="none",
                    marker=style.marker,
                    ms=4,
                    color=style.color,
                    mfc=style.color if _interval_drawn(e) else "white",
                )
            words = NOT_ESTIMABLE if _not_estimable(e) else None
            words = words or ("no value" if estimate is None else None)
            if words:
                ax.annotate(
                    words,
                    xy=(0.0 if estimate is None else scale * float(estimate), y),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=SMALL_PT,
                    color=GREY,
                )
        ax.axvline(0, color=GREY, lw=0.6)
        ax.set_yticks(range(len(rows)), list(reversed(labels)))
        ax.set_ylim(-0.6, len(rows) - 0.4)
        unit = "" if outcome == "rho" else " (pp)"
        ax.set_xlabel(f"Δ{name.split(' (')[0]}, profile − Default{unit}")
        note = "descriptive, no interval" if outcome == "success" else f"t interval, {APPROXIMATE}"
        panel_title(ax, index, f"{name.split(' (')[0]} · {note}", wrap=48)
    profiles = list(dict.fromkeys(p for _, p in rows))
    extra = [
        Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=GREY, label="without interval")
    ]
    legend(fig, ctx, profiles, extra)
    finish(fig, ctx)
    return FigureOut(
        "F24b",
        fig,
        "double",
        _attempts(found),
        "Family profile contrasts, profile − Default, per family and outcome, with their intervals.",
        [
            "Rows: one family and profile each, never pooled across families. Markers: the family profile "
            "contrast, the mean over the family's worlds of each world's mean block difference (profile − "
            f"Default; primary population, at L); lines: its {ctx.level_label} t interval on the world "
            f"differences (n − 1 df), {APPROXIMATE} (plans/worlds/I6 §3.3, §3.4).",
            "Success and families without an interval: the estimate alone (open marker), descriptive; an "
            f"interval the world differences cannot give is written {NOT_ESTIMABLE}. rho in its units, success, "
            "C(L) and A_L in percentage points; worlds and pairs per world in T10 and stats.json.",
        ],
        dict(entries=sorted(e["id"] for e in found)),
        inputs=set(inputs),
    )


# F25 factorial effects -----------------------------------------------------------------------------------------


def f25(ctx: FigureContext, family: str, inputs: Iterable[str]) -> FigureOut | None:
    effects = _entries(ctx, "family_factorial_effect", "world_family", family, endpoint="rho")
    pairs = _entries(ctx, "family_pair", "world_family", family, endpoint="rho")
    if not effects and not pairs:
        return None
    means = _entries(ctx, "family_world_mean", "world_family", family, endpoint="rho")
    points = {str(e["world_id"]): e.get("design_point") or {} for e in means}
    keys = design_keys(declared_family(ctx.settings, family), points.values())
    profiles = _profiles(ctx, effects + pairs)
    names = list(dict.fromkeys(effect_name(e) for e in sorted(effects, key=effect_sort)))
    pair_names = list(dict.fromkeys(str(e["variant"]) for e in pairs))

    def pair_label(variant: str) -> str:
        _, first, second = variant.split(":", 2)
        return (
            f"pair: {design_label(points.get(second, {}), keys, ' / ')} − "
            f"{design_label(points.get(first, {}), keys, ' / ')}"
        )

    rows = [*names, *(pair_label(v) for v in pair_names)]
    lookup = {(effect_name(e), e["profile_id"]): e for e in effects}
    lookup |= {(pair_label(str(e["variant"])), e["profile_id"]): e for e in pairs}
    width_in = PRESETS[ctx.preset].widths_mm["double"] / MM_PER_INCH
    fig, axes = figure(
        ctx.preset,
        "double",
        aspect=(ROW_IN * len(rows) + FRAME_IN) / width_in,
        ncols=2,
        gridspec_kw={"width_ratios": [3, 2]},
    )
    ax, text = axes[0][0], axes[0][1]
    slot = 0.5 / max(1, len(profiles))
    lines_drawn, not_estimable = False, False
    for number, profile in enumerate(profiles):
        style = ctx.styles[profile]
        offset = ((len(profiles) - 1) / 2 - number) * slot
        lenth = next(
            (e["lenth"] for e in effects if e["profile_id"] == profile and e.get("lenth")), None
        )
        if lenth and lenth.get("me") is not None and lenth.get("sme") is not None:
            lines_drawn = True
            for value, dash in ((lenth["me"], "--"), (lenth["sme"], ":")):
                for sign in (-1, 1):
                    ax.axvline(sign * value, color=style.color, ls=dash, lw=0.8)
        elif lenth:
            not_estimable = True
        for y, row in enumerate(reversed(rows)):
            e = lookup.get((row, profile))
            if not e or e.get("estimate") is None:
                continue
            approximate = str(e.get("label") or "").startswith("approximate")
            ax.plot(
                [float(e["estimate"])],
                [y + offset],
                ls="none",
                marker=style.marker,
                ms=4,
                color=style.color,
                mfc=style.color if approximate else "white",
            )
    first_profile = profiles[0]
    for y, row in enumerate(reversed(rows)):
        e = lookup.get((row, first_profile)) or next(
            (lookup[(row, p)] for p in profiles if (row, p) in lookup), {}
        )
        words = effect_label(e.get("label")) if e.get("estimate") is not None else "no value"
        text.text(0.04, y, words, va="center", fontsize=8)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_yticks(range(len(rows)), [wrap_text(r, 40) for r in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Effect on ρ = R/L (high − low level; pair: second − first)")
    ax.set_title(f"{family_title(family)}: factorial effects and pairs on ρ", loc="left")
    text.set(ylim=ax.get_ylim(), xlim=(0, 1))
    text.set_axis_off()
    text.text(0.04, len(rows) - 0.4, "label", fontsize=8, va="bottom", fontweight="bold")
    extra = [
        Line2D([], [], color=GREY, ls="--", lw=0.8, label="± ME (Lenth, approximate)"),
        Line2D([], [], color=GREY, ls=":", lw=0.8, label="± SME (Lenth, approximate)"),
        Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=GREY, label="descriptive"),
    ]
    legend(fig, ctx, profiles, extra if lines_drawn else extra[2:])
    finish(fig, ctx)
    notes = [
        "Effects: the mean of the world means of rho at a factor's high level minus at its low level (an "
        "interaction: by the product of the factors' ±1 codes), primary population, at L; never pooled with another "
        "family.",
        "Dashed and dotted lines: ± Lenth's margin of error (ME) and simultaneous margin of error (SME) from the "
        f"pseudo standard error of the effects (m/3 df), {ctx.level_label}, approximate: no pure error with one seed "
        "per setting (plans/worlds/I6 §3.4.1, Lenth 1989); drawn only from 7 effects with every cell."
        if lines_drawn
        else "No Lenth margins: fewer than 7 effects or a missing cell (plans/worlds/I6 §3.4).",
        "Open markers: descriptive (no Lenth margins; a missing cell gives the unbalanced level difference of the "
        "worlds present, aliased with interactions); pairs: second − first world mean, descriptive. Worlds "
        f"labelled by their design levels ({' / '.join(keys)}). Holm-adjusted p of pre-registered factor "
        "hypotheses only in T11.",
    ]
    if not_estimable:
        notes.append(f"Lenth's margins {NOT_ESTIMABLE}: the pseudo standard error is 0.")
    found = effects + pairs
    return FigureOut(
        f"F25_{family}",
        fig,
        "double",
        _attempts(found),
        f"{family_title(family)}: factorial effects and declared pairs on rho, with Lenth's margins where given.",
        notes,
        dict(group=_group(family, "F25"), family=family, entries=sorted(e["id"] for e in found)),
        inputs=set(inputs),
    )


# F28 density contrast ------------------------------------------------------------------------------------------

POPULATIONS = (("primary", "primary"), ("exclusions_as_failures", "excl. as\nfailures"))
F28_ROW_MM, F28_FRAME_MM = 58.0, 26.0


def _contrast_text(entry: Mapping[str, Any], unit: str, level: float) -> str:
    """An entry's estimate with its Welch interval and the permutation interval, for the caption."""
    welch = family_value(entry, unit, level)
    exact = entry.get("permutation_interval")
    note = str((exact or {}).get("note") or "")
    if exact is None:
        permutation = DASH
    elif exact.get("lower") is None or exact.get("upper") is None:
        permutation = note or NOT_ESTIMABLE
    else:
        permutation = family_value(
            dict(estimate=entry.get("estimate"), interval=exact), unit, level
        )
    return f"{welch} (Welch, {welch_note(entry) or 'approximate'}); permutation (sensitivity) {permutation}"


def f28(ctx: FigureContext, contrast: str, inputs: Iterable[str]) -> FigureOut:
    found = [
        e
        for population, _ in POPULATIONS
        for e in _entries(ctx, "density_contrast", "density_contrast", contrast, population)
    ]
    profiles = _profiles(ctx, found)
    lookup = {(e["profile_id"], e["endpoint"], e["population"]): e for e in found}
    base, moving = str(found[0]["base"]), str(found[0]["moving"])
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    height_mm = F28_ROW_MM * len(profiles) + F28_FRAME_MM
    fig, axes = figure(
        ctx.preset,
        "double",
        aspect=height_mm / width_mm,
        nrows=len(profiles),
        ncols=4,
        gridspec_kw={"width_ratios": [3, 2, 3, 2]},
    )
    level = 1 - ctx.settings.alpha
    notes_values: list[str] = []
    panel = 0
    for row, profile in enumerate(profiles):
        style = ctx.styles[profile]
        for column, outcome in enumerate(("rho", "C_L")):
            name, unit, scale = FAMILY_OUTCOMES[outcome]
            short = name.split(" (")[0]
            values_ax, difference_ax = axes[row][2 * column], axes[row][2 * column + 1]
            primary = lookup.get((profile, outcome, "primary"), {})
            cells: Mapping[str, Any] = primary.get("cells") or {}
            missing: Mapping[str, Any] = primary.get("spawns_missing") or {}
            spawns = sorted({*cells, *missing})
            for x, spawn in enumerate(spawns):
                cell = cells.get(spawn)
                if cell is None:
                    continue
                centres = []
                for side, shift, filled in (("base", -0.18, False), ("moving", 0.18, True)):
                    values = [scale * float(v) for v in cell[side]["values"]]
                    xs = x + shift + _spread(len(values), 0.06)
                    values_ax.plot(
                        xs,
                        values,
                        ls="none",
                        marker="o" if side == "base" else "^",
                        ms=3.5,
                        mec=style.color,
                        mfc=style.color if filled else "none",
                        mew=0.6,
                    )
                    centre = scale * float(cell[side]["mean"])
                    values_ax.plot(
                        [x + shift - 0.1, x + shift + 0.1],
                        [centre, centre],
                        color=style.color,
                        lw=1.6,
                        solid_capstyle="butt",
                    )
                    centres.append((x + shift, centre))
                values_ax.plot(*zip(*centres, strict=True), color=style.color, lw=0.5)
            span = SPANS[outcome]
            values_ax.set_ylim(-0.05 * span, 1.05 * span)
            values_ax.set_xticks(range(len(spawns)), spawns)
            values_ax.set_xlim(-0.6, max(len(spawns), 1) - 0.4)
            values_ax.set_ylabel(name.replace("ρ", "ρ = R/L"))
            label = f"Spawn ({base} ○, {moving} ▲)"
            if missing:
                left = "; ".join(f"{s} ({r})" for s, r in sorted(missing.items()))
                label += "\n" + wrap_text(f"left out: {left}", 44)
            values_ax.set_xlabel(label)
            if not cells:
                panel_message(values_ax, "no spawn with 2 values\nin both worlds", boxed=True)
            who = f" · {style.name}" if len(profiles) > 1 else ""
            panel_title(values_ax, panel, f"{short}: {moving} against {base}{who}")
            panel += 1
            for x, (population, tick) in enumerate(POPULATIONS):
                e = lookup.get((profile, outcome, population), {})
                if not e:
                    continue
                estimate = e.get("estimate")
                if _interval_drawn(e):
                    difference_ax.plot(
                        [x, x],
                        [scale * e["interval"]["lower"], scale * e["interval"]["upper"]],
                        color=style.color,
                        lw=1.2,
                    )
                if estimate is not None:
                    difference_ax.plot(
                        [x],
                        [scale * float(estimate)],
                        ls="none",
                        marker="D" if population == "primary" else "s",
                        ms=4.5,
                        color=style.color,
                        mfc=style.color if population == "primary" else "white",
                    )
                words = NOT_ESTIMABLE if _not_estimable(e) else None
                words = words or ("no value" if estimate is None else None)
                if words:
                    difference_ax.text(
                        x + 0.12,
                        0.5,
                        words,
                        transform=blended_transform_factory(
                            difference_ax.transData, difference_ax.transAxes
                        ),
                        rotation=90,
                        ha="left",
                        va="center",
                        fontsize=SMALL_PT,
                        color=GREY,
                    )
                notes_values.append(
                    f"{style.name}, {short}, {population.replace('_', ' ')}: "
                    f"{_contrast_text(e, unit, level)}"
                )
            difference_ax.axhline(0, color=GREY, lw=0.6)
            difference_ax.set_xticks(
                range(len(POPULATIONS)), [t for _, t in POPULATIONS], fontsize=SMALL_PT
            )
            difference_ax.set_xlim(-0.6, len(POPULATIONS) - 0.4)
            difference_ax.set_ylabel(f"Δ{short}, {moving} − {base}")
            panel_title(difference_ax, panel, "Difference")
            panel += 1
    extra = [
        Line2D([], [], ls="none", marker="o", mfc="none", mec=GREY, ms=3.5, label=f"{base} run"),
        Line2D([], [], ls="none", marker="^", color=GREY, ms=3.5, label=f"{moving} run"),
        Line2D(
            [], [], color=GREY, marker="D", ms=4.5, label="primary, Welch interval (approximate)"
        ),
        Line2D(
            [],
            [],
            color=GREY,
            marker="s",
            mfc="white",
            ms=4.5,
            label="exclusions as failures",
        ),
    ]
    legend(fig, ctx, profiles, extra, max_columns=3)
    finish(fig, ctx)
    first = lookup.get((profiles[0], "rho", "primary"), {})
    limits = first.get("limit_s") or {}
    notes = [
        f"Left: each spawn's reported runs in {base} (base) and {moving} (moving objects), with their means "
        "(bars); repetitions are unpaired, the thin line only joins the two means. Right: the density contrast, "
        "the mean over spawns of the moving world's spawn mean minus the base world's (plans/worlds/I6 §3.9), "
        f"with its stratified Welch {ctx.level_label} interval (primary population; approximate), "
        "beside the exclusions-as-failures sensitivity (runtime failures and stopped runs as rho = 1; C(L) has no "
        "value for them). Each value gives its count of cells (one world in one spawn) without variation, which "
        f"add nothing to the variance. {WELCH_CEILING}",
        "A spawn where either world has fewer than 2 values is left out and named; an interval the cells cannot "
        f"give is written {NOT_ESTIMABLE}, never drawn. Outcomes only: no moving-object measures (decision 59).",
        *notes_values,
    ]
    if first.get("limits_differ"):
        notes.append(f"The two worlds' time limits differ ({limits}): rho is not one ΔR/L.")
    return FigureOut(
        f"F28_{contrast}",
        fig,
        "double",
        _attempts(found),
        f"Density contrast {contrast}: runs per spawn in {base} and {moving}, and the difference {moving} − {base} "
        "on rho and C(L) with its interval and the exclusions-as-failures sensitivity.",
        notes,
        dict(
            label=f"{contrast}: {moving} − {base}",
            density_contrast=contrast,
            entries=sorted(e["id"] for e in found),
        ),
        inputs=set(inputs),
    )


def figures(ctx: FigureContext, inputs: Iterable[str] = ()) -> Iterator[FigureOut]:
    """F24 per family, F24b, F25 per family with effects or pairs, F28 per density contrast, each built when the
    caller asks for it; nothing for a study without family entries. ``inputs``: the export files the family
    estimates read (``families.FamilyResult.inputs``), hashed in every sidecar."""
    read = sorted(set(inputs))
    families = family_ids(ctx.entries)
    for family in families:
        yield f24(ctx, family, read)
    contrasts_figure = f24b(ctx, read)
    if contrasts_figure is not None:
        yield contrasts_figure
    for family in families:
        built = f25(ctx, family, read)
        if built is not None:
            yield built
    for contrast in family_ids(ctx.entries, "density_contrast"):
        yield f28(ctx, contrast, read)
