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

"""Cross-world summaries of a method comparison (contract ``method-summaries-v1``; cross-method readiness C5,
decision B10): ``stats.json`` entries, tables T13 and T15 and figures F29, F30 and F31, written only when the study
compares methods (``data.Study.cross_method``). All are descriptive: no p-values, no ranking, neutral colours for
everything that is not a method's own curve.

Per robot and world group (``worlds.world_groups``; one group for an export without the ``worlds`` table), on the
primary measures of a method comparison (C4): T95 as R_95 = min(T95, L) and C(L).

- ``data_profile`` (per profile): the share of reported runs that reached 95 % coverage by t, over x = t/L, each
  world weighted 1/W (W: the group's worlds where the profile has runs with a value) and within a world each
  spawn weighted equally: a fixed-target runtime distribution (data profile; Moré and Wild 2009, COCO). Runs
  that never reach 95 % stay in every denominator, so the curve ends at the attainment by L; it does not depend on
  the other methods.
- ``performance_profile`` (per profile): the Dolan–Moré profile rho(tau), the share of problems on which the
  profile's ratio r = its Kaplan–Meier median T95 / the best profile's is at most tau, on a log2 axis. Problems are
  the strata of the robot and group in which every profile has runs with a value; a median not reached by L is a
  failure (r infinite); a problem no profile solves counts as a failure for all. Read only against the best
  method; it depends on the method set (Dolan and Moré 2002; Gould and Scott 2016). With an even number of runs
  and exactly half reaching 95 %, the median follows the midpoint rule (RD-10) toward L, so it depends on L.
- ``win_tie_loss`` (per compared profile, against our Default of the robot): every block in which both runs are
  reported is scored for the compared profile. Level 1: the run that reached 95 % first within L wins (an earlier
  time; one that reached against one that did not). Level 2, when neither reached: the higher C(L) wins. Else a
  tie (equal times, or equal C(L)). A pair without the value its level needs is left out and counted. Win
  difference, "method − RADEF": (wins − losses) / pairs per world, averaged over worlds with equal weight; the
  counts are totals (Pocock et al. 2012, 2024).
- ``probability_of_improvement`` (per compared profile): P(method better) on R_95 over the complete blocks (both
  runs reported with a value), per spawn the share of (method run, our run) pairs of its blocks in which the
  method's restricted time is shorter, ties counted half (all censoring is at L, so two runs not reaching 95 %
  tie), averaged over a world's spawns and then over its worlds with equal weight (Agarwal et al. 2021).

Both intervals are percentile intervals of the within-world block bootstrap: complete blocks resampled with
replacement within each world (its spawns and repetitions pooled), B and the seed as the plan's; following
RD-16's pooled-row rule, an interval only when every contributing world has at least 10 complete blocks
(``interval_note`` says why there is none).
"""

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from . import stats
from .data import TIME_EPSILON, Row, Stratum, Study, Value
from .estimates import Entries, horizon_value, time_value
from .figures import (
    GREY,
    FigureContext,
    FigureOut,
    finish,
    grouped,
    legend,
    message_figure,
    panel_message,
    panel_title,
)
from .settings import Settings
from .style import EDGE, PRESETS, figure, page_note
from .tables import DASH, Sources, Table, level_text, number, profile_order, with_interval
from .worlds import world_groups

CONTRACT = "method-summaries-v1"
ENDPOINT = "R_95"
LABEL = "descriptive (no p-value, no ranking)"
DM_CAPTION = "read only against the best method; depends on the method set"
KM_NOTE = (
    "A Kaplan–Meier median uses the midpoint rule (RD-10): with an even number of runs and exactly half reaching "
    "95 %, it lies halfway to L, so it depends on L."
)
FRACTIONS = (0.25, 0.5, 0.75, 1.0)  # T15's points of the data profile, as shares of L
TAUS = (1.0, 2.0, 4.0)  # T15's points of the Dolan–Moré profile
# F31's outcome greys (never a method hue, FP-4): win dark, tie light, loss mid grey with a hatch
WTL_LOOKS = {
    "wins": ("#4D4D4D", None, "method wins"),
    "ties": ("#E6E6E6", None, "tie"),
    "losses": ("#B3B3B3", "////", "RADEF wins"),
}
PANEL_MM, FRAME_MM = 52.0, 30.0  # F29 and F30: one row of panels, and room for titles and legend
WTL_ROW_MM, WTL_FRAME_MM = 7.0, 34.0  # F31: one row and the frame


@dataclass(frozen=True, slots=True)
class Unit:
    """One robot and world group: its strata in display order."""

    robot: str
    group: str | None  # the world group's key; None for an export without the worlds table
    title: str
    strata: tuple[Stratum, ...]

    @property
    def worlds(self) -> list[str]:
        return list(dict.fromkeys(s.world_id for s in self.strata))


def units(study: Study) -> list[Unit]:
    """Every robot × world group with strata, groups in display order, robots in study order within a group."""
    found = []
    for group in world_groups(study):
        for robot in dict.fromkeys(s.robot_id for s in group.strata):
            found.append(
                Unit(
                    robot,
                    group.key,
                    group.title or "All strata",
                    tuple(s for s in group.strata if s.robot_id == robot),
                )
            )
    return found


def entry_id(estimand: str, unit: Unit, profile: str) -> str:
    """The key of a ``method-summaries-v1`` entry: ``estimates.entry_id``'s fields with ``group:<key>`` in the
    world field."""
    return "|".join(
        (
            estimand,
            ENDPOINT,
            unit.robot,
            f"group:{unit.group or 'all'}",
            "",
            profile,
            "",
            "primary",
            "",
        )
    )


def _entry(
    estimand: str, unit: Unit, profile: str, attempt_ids: Sequence[str], **values: Any
) -> dict[str, Any]:
    return dict(
        id=entry_id(estimand, unit, profile),
        contract=CONTRACT,
        estimand=estimand,
        endpoint=ENDPOINT,
        robot_id=unit.robot,
        world_id=None,
        spawn_id=None,
        world_group=unit.group,
        world_group_title=unit.title,
        profile_id=profile,
        horizon_s=None,
        population="primary",
        role="descriptive",
        label=LABEL,
        attempt_ids=sorted(set(attempt_ids)),
        **values,
    )


def _usable(stratum: Stratum, profile: str) -> tuple[list[tuple[str, Value]], dict[str, int]]:
    """The profile's reported runs in the stratum with their R_95 (attempt, value), and why the others have
    none."""
    found: list[tuple[str, Value]] = []
    reasons: dict[str, int] = defaultdict(int)
    for run in stratum.runs_of(profile):
        value = time_value(run, ENDPOINT, stratum.limit_s)
        if value.value is None:
            reasons[value.reason] += 1
        else:
            found.append((str(run["attempt_id"]), value))
    return found, dict(reasons)


def _merge(total: dict[str, int], more: Mapping[str, int]) -> None:
    for key, count in more.items():
        total[key] = total.get(key, 0) + count


# (a) data profile ----------------------------------------------------------------------------------------------


def data_profile(unit: Unit, profile: str) -> dict[str, Any]:
    # world -> spawn -> (event shares x = R/L of the runs that reached, runs with a value)
    worlds: dict[str, list[tuple[list[float], int]]] = defaultdict(list)
    reasons: dict[str, int] = {}
    attempts: list[str] = []
    runs_per_world: dict[str, int] = defaultdict(int)
    for stratum in unit.strata:
        found, why = _usable(stratum, profile)
        _merge(reasons, why)
        if not found:
            continue
        events = [
            float(v.value) / stratum.limit_s for _, v in found if v.event and v.value is not None
        ]
        worlds[stratum.world_id].append((events, len(found)))
        runs_per_world[stratum.world_id] += len(found)
        attempts += [a for a, _ in found]
    without = [w for w in unit.worlds if w not in worlds and _ran(unit, w, profile)]

    def share(x: float) -> float | None:
        if not worlds:
            return None
        per_world = [
            float(np.mean([sum(e <= x + TIME_EPSILON for e in events) / n for events, n in spawns]))
            for spawns in worlds.values()
        ]
        return float(np.mean(per_world))

    xs = sorted({e for spawns in worlds.values() for events, _ in spawns for e in events})
    points = [[0.0, share(0.0)], *([x, share(x)] for x in xs)] if worlds else []
    return _entry(
        "data_profile",
        unit,
        profile,
        attempts,
        points=points,
        attained_by_limit=share(1.0),
        at_fractions={repr(f): share(f) for f in FRACTIONS},
        n_worlds=len(worlds),
        worlds=sorted(worlds),
        runs_per_world=dict(sorted(runs_per_world.items())),
        n_runs=sum(runs_per_world.values()),
        no_value=dict(sorted(reasons.items())),
        worlds_without_value=without,
        weights="each world 1/W, each spawn equally within its world",
    )


def _ran(unit: Unit, world: str, profile: str) -> bool:
    """Whether the profile has a final run in the world (reported or not)."""
    return any(
        r["profile_id"] == profile for s in unit.strata if s.world_id == world for r in s.finals
    )


# (b) Dolan–Moré performance profile ----------------------------------------------------------------------------


def performance_profiles(unit: Unit, profiles: Sequence[str]) -> list[dict[str, Any]]:
    """One entry per profile on the problems where every profile has runs with a value."""
    medians: dict[str, dict[str, float | None]] = defaultdict(dict)  # problem -> profile -> median
    attempts: dict[str, list[str]] = defaultdict(list)
    left_out: dict[str, list[str]] = {}
    for stratum in unit.strata:
        name = f"{stratum.world_id}/{stratum.spawn_id}"
        found = {p: _usable(stratum, p)[0] for p in profiles}
        missing = [p for p, values in found.items() if not values]
        if missing:
            left_out[name] = missing
            continue
        for profile, values in found.items():
            times = [float(v.value) for _, v in values if v.value is not None]
            curve = stats.kaplan_meier(times, [bool(v.event) for _, v in values])
            medians[name][profile] = stats.km_quantile(curve, 0.5)
            attempts[profile] += [a for a, _ in values]
    ratios: dict[str, list[float]] = defaultdict(list)
    for problem, by_profile in medians.items():
        solved = [m for m in by_profile.values() if m is not None]
        best = min(solved) if solved else None
        for profile in profiles:
            m = by_profile[profile]
            if m is None or best is None:
                ratios[profile].append(math.inf)
            elif best <= TIME_EPSILON:
                ratios[profile].append(1.0 if m <= TIME_EPSILON else math.inf)
            else:
                ratios[profile].append(m / best)
    count = len(medians)
    entries = []
    for profile in profiles:
        mine = ratios[profile]
        finite = sorted({r for r in mine if math.isfinite(r)})

        def rho(tau: float, values: list[float] = mine) -> float | None:
            return sum(r <= tau + TIME_EPSILON for r in values) / count if count else None

        entries.append(
            _entry(
                "performance_profile",
                unit,
                profile,
                attempts[profile],
                problems=count,
                points=[[math.log2(r), rho(r)] for r in finite],
                rho_at=dict((repr(t), rho(t)) for t in TAUS),
                solved=None if not count else sum(math.isfinite(r) for r in mine) / count,
                medians={p: medians[p][profile] for p in sorted(medians)},
                ratios=dict(sorted((p, _finite(r)) for p, r in zip(medians, mine, strict=True))),
                problems_left_out={p: sorted(v) for p, v in sorted(left_out.items())},
                note=DM_CAPTION,
                measure="Kaplan–Meier median of T95 per problem (stratum); not reached by L: a failure",
            )
        )
    return entries


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


# (c) paired win/tie/loss ---------------------------------------------------------------------------------------


def _score(study: Study, stratum: Stratum, run: Row, partner: Row) -> tuple[int, int] | str:
    """(score for the compared run: 1 win, 0 tie, −1 loss; the deciding level) or why the pair has no score."""
    limit = stratum.limit_s
    a, b = time_value(run, ENDPOINT, limit), time_value(partner, ENDPOINT, limit)
    if a.value is None or b.value is None:
        return "no_T95_value"
    if a.event or b.event:
        if a.event and b.event:
            difference = float(a.value) - float(b.value)
            return (0 if abs(difference) <= TIME_EPSILON else (1 if difference < 0 else -1)), 1
        return (1 if a.event else -1), 1
    c, d = (horizon_value(study, r, "C_h", limit, limit) for r in (run, partner))
    if c.value is None or d.value is None:
        return "no_C_L_value"
    difference = float(c.value) - float(d.value)
    return (0 if abs(difference) <= TIME_EPSILON else (1 if difference > 0 else -1)), 2


def _complete_blocks(
    stratum: Stratum, profile: str, baseline: str
) -> Iterator[tuple[Row, Row] | str]:
    """Each block of the stratum as its (compared run, our run) when both are reported, else why not."""
    for _, run, partner in stratum.blocks(profile, baseline):
        if run is None or partner is None:
            yield "not_both_run"
        elif run["report_class"] != "reported" or partner["report_class"] != "reported":
            yield "not_both_reported"
        else:
            yield run, partner


def _too_few(blocks: Mapping[str, int]) -> str | None:
    """RD-16's pooled-row rule for a bootstrap interval: every contributing world needs at least
    ``BOOTSTRAP_MIN_CLUSTERS`` complete blocks; the reason there is none, else None."""
    if not blocks:
        return "no complete block: no interval"
    short = {w: n for w, n in sorted(blocks.items()) if n < stats.BOOTSTRAP_MIN_CLUSTERS}
    if short:
        return (
            f"fewer than {stats.BOOTSTRAP_MIN_CLUSTERS} blocks in a world "
            f"({', '.join(f'{w}: {n}' for w, n in short.items())}): no interval"
        )
    return None


def win_tie_loss(
    study: Study, settings: Settings, unit: Unit, profile: str, baseline: str
) -> dict[str, Any]:
    """Win/tie/loss of the compared profile against ours; the win difference (wins − losses)/pairs per world,
    averaged over worlds with equal weight, with the within-world block bootstrap (C5 (c))."""
    counts = {f"{k}_level_{level}": 0 for k in ("wins", "losses", "ties") for level in (1, 2)}
    left: dict[str, int] = defaultdict(int)
    scores: dict[str, list[float]] = defaultdict(list)  # world -> block scores (its spawns pooled)
    per_world: dict[str, dict[str, int]] = {}
    attempts: list[str] = []
    for stratum in unit.strata:
        for block in _complete_blocks(stratum, profile, baseline):
            if isinstance(block, str):
                left[block] += 1
                continue
            run, partner = block
            found = _score(study, stratum, run, partner)
            if isinstance(found, str):
                left[found] += 1
                continue
            score, level = found
            name = {1: "wins", 0: "ties", -1: "losses"}[score]
            counts[f"{name}_level_{level}"] += 1
            scores[stratum.world_id].append(float(score))
            attempts += [run["attempt_id"], partner["attempt_id"]]
            world = per_world.setdefault(stratum.world_id, dict(wins=0, ties=0, losses=0))
            world[name] += 1
    wins = counts["wins_level_1"] + counts["wins_level_2"]
    losses = counts["losses_level_1"] + counts["losses_level_2"]
    ties = counts["ties_level_1"] + counts["ties_level_2"]
    arrays = {w: np.asarray(v, dtype=np.float64) for w, v in sorted(scores.items())}
    estimate = float(np.mean([v.mean() for v in arrays.values()])) if arrays else None
    note = _too_few({w: v.size for w, v in arrays.items()})
    interval = None
    if note is None:
        rng = stats.random_stream(settings.seed, stats.WIN_BOOTSTRAP_STREAM)
        replicates = np.zeros(settings.resamples)
        for values in arrays.values():
            weights = rng.multinomial(
                values.size, np.full(values.size, 1 / values.size), settings.resamples
            )
            replicates += (weights @ values) / values.size
        interval = _percentile(
            replicates / len(arrays),
            settings,
            "within-world block bootstrap",
            stats.WIN_BOOTSTRAP_STREAM,
        )
    return _entry(
        "win_tie_loss",
        unit,
        profile,
        attempts,
        comparison=f"{profile} − {baseline}",
        reference=baseline,
        n_pairs=wins + losses + ties,
        wins=wins,
        ties=ties,
        losses=losses,
        **counts,
        pairs_left_out=dict(sorted(left.items())),
        per_world=dict(sorted(per_world.items())),
        world_differences={w: float(v.mean()) for w, v in arrays.items()},
        n_worlds=len(arrays),
        estimate=estimate,
        interval=interval,
        interval_note=note,
        weights="(wins − losses)/pairs per world, worlds weighted equally; counts are totals",
        levels="1: reached 95 % coverage first within L; 2 (neither reached): higher C(L); else a tie",
    )


def _percentile(
    replicates: np.ndarray[Any, Any], settings: Settings, what: str, stream: int
) -> dict[str, Any]:
    lower, upper = np.quantile(replicates, [settings.alpha / 2, 1 - settings.alpha / 2])
    return dict(
        lower=float(lower),
        upper=float(upper),
        level=1 - settings.alpha,
        method=f"{what}, percentile, B = {settings.resamples}, seed {settings.seed} (stream {stream})",
        note=None,
        lower_beyond_limit=False,
        upper_beyond_limit=False,
    )


# (d) probability of improvement --------------------------------------------------------------------------------


def probability_of_improvement(
    settings: Settings, unit: Unit, profile: str, baseline: str
) -> dict[str, Any]:
    """P(method better) on R_95 over the complete blocks: per spawn, the share of (method run, our run) pairs of
    its blocks in which the method's R_95 is shorter, ties half; averaged over a world's spawns and then over
    worlds with equal weight; within-world block bootstrap (C5 (d))."""
    # world -> (spawn index per block, method R_95 per block, our R_95 per block)
    worlds: dict[str, tuple[list[int], list[float], list[float]]] = {}
    left: dict[str, int] = defaultdict(int)
    attempts: list[str] = []
    per_stratum: dict[str, dict[str, Any]] = {}
    for stratum in unit.strata:
        spawns, theirs, ours = worlds.setdefault(stratum.world_id, ([], [], []))
        index = len({*spawns})
        mine = 0
        for block in _complete_blocks(stratum, profile, baseline):
            if isinstance(block, str):
                left[block] += 1
                continue
            run, partner = block
            a = time_value(run, ENDPOINT, stratum.limit_s)
            b = time_value(partner, ENDPOINT, stratum.limit_s)
            if a.value is None or b.value is None:
                left["no_T95_value"] += 1
                continue
            spawns.append(index)
            theirs.append(float(a.value))
            ours.append(float(b.value))
            attempts += [run["attempt_id"], partner["attempt_id"]]
            mine += 1
        if mine:
            per_stratum[f"{stratum.world_id}/{stratum.spawn_id}"] = dict(blocks=mine)
    worlds = {w: found for w, found in worlds.items() if found[0]}

    def world_values(weights: Mapping[str, np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """Per replicate (rows of block weights), the equal-weight mean over worlds of each world's mean over
        its spawns of P(method better)."""
        per_world = []
        for world, (spawns, theirs, ours) in worlds.items():
            index = np.asarray(spawns)
            y, x = np.asarray(theirs), np.asarray(ours)
            gap = x[None, :] - y[:, None]  # ours − method: positive when the method is shorter
            matrix = np.where(np.abs(gap) <= TIME_EPSILON, 0.5, (gap > 0).astype(np.float64))
            w = weights[world]
            shares = []
            for spawn in sorted(set(spawns)):
                keep = index == spawn
                ws = w[:, keep]
                total = ws.sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    shares.append(
                        np.einsum("bi,ij,bj->b", ws, matrix[np.ix_(keep, keep)], ws) / total**2
                    )
            per_world.append(np.nanmean(np.vstack(shares), axis=0))
        result: np.ndarray[Any, Any] = np.mean(np.vstack(per_world), axis=0)
        return result

    estimate = None
    interval = None
    note = _too_few({w: len(found[0]) for w, found in worlds.items()})
    if worlds:
        ones = {w: np.ones((1, len(found[0]))) for w, found in worlds.items()}
        estimate = float(world_values(ones)[0])
    if worlds and note is None:
        rng = stats.random_stream(settings.seed, stats.IMPROVEMENT_BOOTSTRAP_STREAM)
        weights = {
            w: rng.multinomial(
                len(found[0]), np.full(len(found[0]), 1 / len(found[0])), settings.resamples
            ).astype(np.float64)
            for w, found in worlds.items()
        }
        interval = _percentile(
            world_values(weights),
            settings,
            "within-world block bootstrap",
            stats.IMPROVEMENT_BOOTSTRAP_STREAM,
        )
    return _entry(
        "probability_of_improvement",
        unit,
        profile,
        attempts,
        comparison=f"{profile} − {baseline}",
        reference=baseline,
        estimate=estimate,
        interval=interval,
        interval_note=note,
        direction="P(method better): the method's R_95 shorter than RADEF's, ties half",
        n_worlds=len(worlds),
        blocks_per_world={w: len(found[0]) for w, found in sorted(worlds.items())},
        blocks_left_out=dict(sorted(left.items())),
        per_stratum=dict(sorted(per_stratum.items())),
        weights="each world equally, each spawn equally within its world; complete blocks only",
    )


# All entries ---------------------------------------------------------------------------------------------------


def summary_entries(study: Study, settings: Settings) -> list[dict[str, Any]]:
    """Every ``method-summaries-v1`` entry; none unless the study compares methods."""
    if not study.cross_method:
        return []
    order = profile_order(study)
    entries: list[dict[str, Any]] = []
    for unit in units(study):
        present = {p for s in unit.strata for p in s.profiles()}
        profiles = [p for p in order if p in present]
        entries += [data_profile(unit, p) for p in profiles]
        entries += performance_profiles(unit, profiles)
        baseline = study.baselines.get(unit.robot)
        if baseline is None or baseline not in present:
            continue
        for profile in profiles:
            if profile == baseline:
                continue
            entries.append(win_tie_loss(study, settings, unit, profile, baseline))
            entries.append(probability_of_improvement(settings, unit, profile, baseline))
    return [_clean(e) for e in entries]


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


# Tables T13 and T15 --------------------------------------------------------------------------------------------


def _found(entries: Entries, estimand: str, unit: Unit, profile: str) -> dict[str, Any]:
    return entries.by_id(entry_id(estimand, unit, profile))


def _grouped(table: Table, all_units: Sequence[Unit]) -> Table:
    """With more than one world group: a first column ``Group`` (heading rows in LaTeX and the sheet)."""
    titles = {u.group or "all": u.title for u in all_units}
    if len(titles) < 2:
        table.rows = [row[1:] for row in table.rows]
        table.columns = table.columns[1:]
        return table
    table.group_titles = titles
    table.extra["groups"] = dict(titles)
    table.notes.append(
        "Groups (never pooled): " + "; ".join(f"{k}, {t}" for k, t in titles.items()) + "."
    )
    return table


def _with_or_without(entry: Mapping[str, Any], level: float) -> str:
    """An estimate with its interval, or "(no interval)" when the bootstrap's minimum was not met."""
    text = with_interval(entry.get("estimate"), entry.get("interval"), "ratio", level)
    return (
        text if entry.get("interval") or entry.get("estimate") is None else f"{text} (no interval)"
    )


def t13(src: Sources) -> Table | None:
    """C5 (c), (d): per robot, world group and compared method, the paired win/tie/loss split with the win
    difference and P(method better), with their intervals."""
    if not src.study.cross_method:
        return None
    every = units(src.study)
    table = Table(
        "T13",
        "Paired wins, ties and losses and the probability of improvement, each method against RADEF",
        [
            "Group",
            "Robot",
            "Compared method",
            "Pairs",
            "Wins (L1 + L2)",
            "Ties",
            "Losses (L1 + L2)",
            "Left out",
            "Win difference [CI]",
            "P(method better) [CI]",
            "Worlds",
        ],
    )
    for unit in every:
        for profile in src.order:
            wtl = _found(src.entries, "win_tie_loss", unit, profile)
            poi = _found(src.entries, "probability_of_improvement", unit, profile)
            if not wtl:
                continue
            left = "; ".join(f"{k} {v}" for k, v in (wtl.get("pairs_left_out") or {}).items())
            table.rows.append(
                [
                    unit.group or "all",
                    unit.robot,
                    src.name(profile),
                    str(wtl.get("n_pairs", 0)),
                    f"{wtl.get('wins', 0)} ({wtl.get('wins_level_1', 0)} + {wtl.get('wins_level_2', 0)})",
                    str(wtl.get("ties", 0)),
                    f"{wtl.get('losses', 0)} ({wtl.get('losses_level_1', 0)} + "
                    f"{wtl.get('losses_level_2', 0)})",
                    left or DASH,
                    _with_or_without(wtl, src.level),
                    _with_or_without(poi, src.level),
                    str(poi.get("n_worlds", 0)),
                ]
            )
            table.attempt_ids |= set(wtl.get("attempt_ids", [])) | set(poi.get("attempt_ids", []))
    table.notes = [
        "Descriptive (method-summaries-v1, C5): no p-values, no ranking. Pairs: blocks in which both runs are "
        "reported, scored for the compared method against RADEF (RADEF's Default of the robot): level 1 (L1), the run "
        "that reached 95 % coverage first within L wins; level 2 (L2), when neither reached it, the higher C(L) "
        "wins; else a tie. Left out: pairs without the value their level needs, or without two reported runs.",
        "Win difference, method − RADEF: (wins − losses) / pairs per world, averaged over worlds with equal weight "
        "(−1: RADEF won every pair, 1: the method won every pair); the counts are totals over the worlds. "
        "P(method better): per spawn, the share of (method run, RADEF run) pairs of its complete blocks in which "
        "the method's R_95 is shorter, ties half, averaged over spawns and then over worlds with equal weight; "
        "0.5 is no difference.",
        f"Both: {level_text(src.level)} percentile interval from the within-world block bootstrap (complete blocks "
        "resampled within each world, its spawns and repetitions pooled; stats.json interval.method), only when "
        f"every contributing world has at least {stats.BOOTSTRAP_MIN_CLUSTERS} complete blocks (RD-16), else "
        '"no interval" (the reason in stats.json interval_note).',
    ]
    return _grouped(table, every)


def t15(src: Sources) -> Table | None:
    """C5 (a), (b): per robot, world group and profile, the data profile at four shares of L and the Dolan–Moré
    profile at tau = 1, 2 and 4."""
    if not src.study.cross_method:
        return None
    every = units(src.study)
    table = Table(
        "T15",
        "Data profiles and Dolan–Moré profiles of T95 per robot and world group",
        [
            "Group",
            "Robot",
            "Profile",
            "Worlds",
            "Runs",
            *(f"Reached by {f:g} L (%)" for f in FRACTIONS),
            "Problems",
            *(f"ρ({t:g}) (%)" for t in TAUS),
            "Solved (%)",
        ],
    )

    def percent(value: Any) -> str:
        return DASH if value is None else number(100 * float(value), 0)

    for unit in every:
        for profile in src.order:
            data = _found(src.entries, "data_profile", unit, profile)
            if not data:
                continue
            perf = _found(src.entries, "performance_profile", unit, profile)
            at = data.get("at_fractions") or {}
            rho = perf.get("rho_at") or {}
            table.rows.append(
                [
                    unit.group or "all",
                    unit.robot,
                    src.name(profile),
                    str(data.get("n_worlds", 0)),
                    str(data.get("n_runs", 0)),
                    *(percent(at.get(repr(f))) for f in FRACTIONS),
                    str(perf.get("problems", 0)),
                    *(percent(rho.get(repr(t))) for t in TAUS),
                    percent(perf.get("solved")),
                ]
            )
            table.attempt_ids |= set(data.get("attempt_ids", []))
    table.notes = [
        "Descriptive (method-summaries-v1, C5). Data profile: the share of reported runs that reached 95 % "
        "coverage by t = f · L, each world weighted 1/W and each spawn equally within its world; runs that never "
        "reach stay in the denominator (Reached by 1 L is the attainment by L). It depends only on the method.",
        "Dolan–Moré profile: rho(tau), the share of problems (strata where every profile has runs) on which the "
        "profile's Kaplan–Meier median T95 is at most tau times the best profile's; a median not reached by L is "
        f"a failure; Solved = rho at infinity. Dolan–Moré profiles: {DM_CAPTION}. No ranking is implied. "
        + KM_NOTE,
    ]
    return _grouped(table, every)


def tables(src: Sources) -> list[Table]:
    return [t for t in (t13(src), t15(src)) if t is not None]


# Figures F29, F30 and F31 --------------------------------------------------------------------------------------


def _page_units(ctx: FigureContext) -> list[Unit]:
    """The units of a group's page (``figures.grouped``): its strata's robot × group units of the whole study."""
    attempts = {id(s) for s in ctx.study.strata}
    return [u for u in units(ctx.full_study) if any(id(s) in attempts for s in u.strata)]


def _panels(ctx: FigureContext, count: int) -> tuple[Any, list[Any], str, float]:
    columns = min(3, max(1, count))
    rows = math.ceil(max(1, count) / columns)
    width = "double" if count > 1 else "single"
    height_mm = PANEL_MM * rows + FRAME_MM
    fig, axes = figure(
        ctx.preset,
        width,
        aspect=height_mm / PRESETS[ctx.preset].widths_mm[width],
        nrows=rows,
        ncols=columns,
    )
    flat = [axes[r][c] for r in range(rows) for c in range(columns)]
    for ax in flat[count:]:
        ax.set_axis_off()
    return fig, flat[:count], width, height_mm


def _title(ctx: FigureContext, unit: Unit) -> str:
    several = len({s.robot_id for s in ctx.full_study.strata}) > 1
    return f"{unit.robot} · {unit.title}" if several else unit.title


def _counts_note(ctx: FigureContext, estimand: str, chosen: Sequence[Unit]) -> str:
    """Per unit and profile: worlds and runs (data profile) or problems (Dolan–Moré), and what is left out."""
    found = []
    for unit in chosen:
        parts = []
        for profile in ctx.order:
            e = _found(ctx.entries, estimand, unit, profile)
            if not e:
                continue
            name = ctx.styles[profile].name
            if estimand == "data_profile":
                extra = "; ".join(f"{k} {v}" for k, v in (e.get("no_value") or {}).items())
                without = ", ".join(e.get("worlds_without_value") or [])
                text = f"{name}: W = {e.get('n_worlds', 0)}, n = {e.get('n_runs', 0)}"
                text += f" (no value: {extra})" if extra else ""
                text += f" (worlds without a T95 value: {without})" if without else ""
            else:
                text = f"{name}: {e.get('problems', 0)} problems"
            parts.append(text)
        if estimand == "performance_profile":
            first = next(
                (
                    _found(ctx.entries, estimand, unit, p)
                    for p in ctx.order
                    if _found(ctx.entries, estimand, unit, p)
                ),
                {},
            )
            left = first.get("problems_left_out") or {}
            if left:
                parts.append(
                    "left out (a profile without runs): "
                    + ", ".join(f"{k} ({', '.join(v)})" for k, v in left.items())
                )
        if parts:
            found.append(f"{_title(ctx, unit)}: " + "; ".join(parts))
    return "n per panel and profile: " + " | ".join(found) + "."


def f29(ctx: FigureContext) -> FigureOut:
    chosen = _page_units(ctx)
    if not chosen:
        return message_figure(ctx, "F29", "No method comparison to summarise.")
    fig, axes, width, height = _panels(ctx, len(chosen))
    attempts: set[str] = set()
    drawn: list[str] = []
    for index, (ax, unit) in enumerate(zip(axes, chosen, strict=True)):
        for profile in ctx.order:
            e = _found(ctx.entries, "data_profile", unit, profile)
            points = e.get("points") or []
            if not points:
                continue
            style = ctx.styles[profile]
            xs = [float(x) for x, _ in points] + [1.0]
            ys = [100 * float(y) for _, y in points] + [100 * float(points[-1][1])]
            ax.step(xs, ys, where="post", color=style.color, ls=style.linestyle, lw=1.0)
            ax.plot([1.0], [ys[-1]], marker=style.marker, color=style.color, ms=3, ls="none")
            attempts |= set(e.get("attempt_ids", []))
            drawn.append(profile)
        ax.set(xlim=(0, 1.04), ylim=(0, 100), xlabel="t / L", ylabel="Reached 95 % coverage (%)")
        panel_title(ax, index, _title(ctx, unit), wrap=40)
        if not any(_found(ctx.entries, "data_profile", unit, p).get("points") for p in ctx.order):
            panel_message(ax, "no runs with a value", boxed=True)
    legend(fig, ctx, list(dict.fromkeys(drawn)))
    finish(fig, ctx, descriptive=True)
    notes = [
        "Data profile (method-summaries-v1, C5 (a)): the share of reported runs that reached 95 % coverage by t, "
        "over t/L (simulated time since first movement over the stratum's limit), each world weighted 1/W and each "
        "spawn equally within its world; runs that never reach 95 % (time limit, stuck, no movement, explorer "
        "failure) stay in every denominator, so a curve ends at its attainment by L (marker at t = L). It "
        "depends only on the method, not on the others drawn. Descriptive: no p-values, no ranking.",
        _counts_note(ctx, "data_profile", chosen),
    ]
    if (overflow := page_note(height)) is not None:
        notes.append(overflow)
    return FigureOut(
        "F29",
        fig,
        width,
        attempts,
        "Share of runs that reached 95 % coverage by each share of the time limit, per method, robot and world "
        "group, worlds weighted equally.",
        notes,
        {"entries": sorted(entry_id("data_profile", u, p) for u in chosen for p in drawn)},
    )


def f30(ctx: FigureContext) -> FigureOut:
    chosen = _page_units(ctx)
    if not chosen:
        return message_figure(ctx, "F30", "No method comparison to summarise.")
    fig, axes, width, height = _panels(ctx, len(chosen))
    attempts: set[str] = set()
    drawn: list[str] = []
    for index, (ax, unit) in enumerate(zip(axes, chosen, strict=True)):
        top = 1.0
        for profile in ctx.order:
            e = _found(ctx.entries, "performance_profile", unit, profile)
            if not e or not e.get("problems"):
                continue
            points = [(float(x), 100 * float(y)) for x, y in e.get("points") or []]
            top = max([top, *(x for x, _ in points)])
            style = ctx.styles[profile]
            start = 100 * float((e.get("rho_at") or {}).get(repr(1.0)) or 0.0)
            xs = [0.0, *(x for x, _ in points if x > TIME_EPSILON)]
            ys = [start, *(y for x, y in points if x > TIME_EPSILON)]
            ax.step(
                [*xs, top * 1.1 + 0.2],
                [*ys, ys[-1]],
                where="post",
                color=style.color,
                ls=style.linestyle,
                lw=1.0,
            )
            attempts |= set(e.get("attempt_ids", []))
            drawn.append(profile)
        ax.set(
            xlim=(0, top * 1.1 + 0.2),
            ylim=(0, 100),
            xlabel=r"$\log_2 \tau$ ($\tau$: ratio to the best median T95)",
            ylabel="Problems within τ of the best (%)",
        )
        panel_title(ax, index, _title(ctx, unit), wrap=40)
        if not drawn:
            panel_message(ax, "no problem with runs of every profile", boxed=True)
    legend(fig, ctx, list(dict.fromkeys(drawn)))
    finish(fig, ctx, descriptive=True)
    notes = [
        f"Dolan–Moré profile: {DM_CAPTION}. rho(tau) is the share of problems (strata where every profile has "
        "reported runs with a value) on which the profile's Kaplan–Meier median T95 is at most tau times the best "
        "profile's; a median not reached by L is a failure and never enters, so a curve's right end is its share "
        "of problems solved. rho at log2 tau = 0 is the share where it was the best (ties shared). Steps of "
        f"1/problems; no ranking is implied (method-summaries-v1, C5 (b)). {KM_NOTE}",
        _counts_note(ctx, "performance_profile", chosen),
    ]
    if (overflow := page_note(height)) is not None:
        notes.append(overflow)
    return FigureOut(
        "F30",
        fig,
        width,
        attempts,
        "Dolan–Moré performance profile of the median time to 95 % coverage per method, robot and world group; "
        f"{DM_CAPTION}.",
        notes,
        {"entries": sorted(entry_id("performance_profile", u, p) for u in chosen for p in drawn)},
    )


def f31(ctx: FigureContext) -> FigureOut:
    chosen = _page_units(ctx)
    several = len({s.robot_id for s in ctx.full_study.strata}) > 1
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for unit in chosen:
        for profile in ctx.order:
            wtl = _found(ctx.entries, "win_tie_loss", unit, profile)
            if wtl:
                poi = _found(ctx.entries, "probability_of_improvement", unit, profile)
                name = ctx.styles[profile].name
                rows.append((f"{unit.robot} · {name}" if several else name, wtl, poi))
    if not rows:
        return message_figure(ctx, "F31", "No compared method paired with RADEF.")
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    height = WTL_ROW_MM * len(rows) + WTL_FRAME_MM
    fig, axes = figure(
        ctx.preset,
        "double",
        aspect=height / width_mm,
        ncols=3,
        gridspec_kw={"width_ratios": [1.6, 1, 1]},
    )
    bars, wins_ax, poi_ax = axes[0]
    attempts: set[str] = set()
    level = 1 - ctx.settings.alpha
    for y, (_, wtl, poi) in enumerate(rows):
        attempts |= set(wtl.get("attempt_ids", [])) | set(poi.get("attempt_ids", []))
        n = int(wtl.get("n_pairs") or 0)
        left = 0.0
        for key, (face, hatch, _) in WTL_LOOKS.items():
            share = int(wtl.get(key) or 0) / n if n else 0.0
            if share:
                bars.barh(y, share, left=left, color=face, hatch=hatch, edgecolor=EDGE, lw=0.4)
                colour = "#FFFFFF" if key == "wins" else "#000000"
                bars.text(
                    left + share / 2,
                    y,
                    str(wtl.get(key)),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=colour,
                )
            left += share
        bars.text(1.02, y, f"n = {n}", va="center", fontsize=7)
        for ax, entry, centre in ((wins_ax, wtl, 0.0), (poi_ax, poi, 0.5)):
            interval = entry.get("interval") or {}
            if interval.get("lower") is not None:
                ax.plot([interval["lower"], interval["upper"]], [y, y], color="#000000", lw=1.0)
            if entry.get("estimate") is not None:
                ax.plot([entry["estimate"]], [y], "s", color="#000000", ms=3.5)
            elif entry:
                ax.text(centre, y, "no value", ha="center", va="center", fontsize=7, color=GREY)
    labels = [label for label, _, _ in rows]
    bars.set_yticks(range(len(rows)), labels, fontsize=8)
    bars.set(xlim=(0, 1.18), xlabel="Share of pairs")
    bars.invert_yaxis()
    panel_title(bars, 0, "paired wins, ties, losses")
    for index, (ax, lim, centre, xlabel, title) in enumerate(
        (
            (wins_ax, (-1.05, 1.05), 0.0, "Win difference, method − RADEF", "win difference"),
            (poi_ax, (-0.02, 1.02), 0.5, "P(method better) on R_95", "P(method better)"),
        ),
        1,
    ):
        ax.axvline(centre, color=GREY, lw=0.6)
        ax.set(xlim=lim, ylim=bars.get_ylim(), xlabel=xlabel)
        ax.set_yticks(range(len(rows)), [""] * len(rows))
        panel_title(ax, index, title)
    handles: list[Any] = [
        Patch(facecolor=face, hatch=hatch, edgecolor=EDGE, label=text)
        for face, hatch, text in WTL_LOOKS.values()
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="#000000",
            marker="s",
            ms=3.5,
            label=f"estimate, {level_text(level)} interval",
        )
    )
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    finish(fig, ctx, descriptive=True)
    notes = [
        "Descriptive (method-summaries-v1, C5 (c), (d)): no p-values, no ranking; greys only. (a) Every block "
        "with both runs reported, scored for the compared method against RADEF: level 1, the run that reached 95 % "
        "coverage first within L wins; level 2, when neither reached it, the higher C(L) wins; else a tie; counts "
        "in the bars (totals over the worlds), n pairs at the right (pairs left out in T13 and stats.json). (b) "
        "Win difference, method − RADEF: (wins − losses)/pairs per world, averaged over worlds with equal weight. "
        "(c) P(method better): per spawn, the share of (method run, RADEF run) pairs of the complete blocks in which "
        "the method's R_95 is shorter, ties half, averaged over spawns and then over worlds with equal weight; 0.5 "
        "is no difference. (b), (c): percentile intervals of the within-world block bootstrap (complete blocks "
        "resampled within each world, spawns and repetitions pooled) only when every contributing world has at "
        f"least {stats.BOOTSTRAP_MIN_CLUSTERS} complete blocks (RD-16), else the estimate alone.",
    ]
    if (overflow := page_note(height)) is not None:
        notes.append(overflow)
    return FigureOut(
        "F31",
        fig,
        "double",
        attempts,
        "Paired wins, ties and losses of each method against RADEF, the win difference and the probability that "
        "the method is faster to 95 % coverage than RADEF, per robot and world group.",
        notes,
        {
            "rows": [
                dict(label=label, win_tie_loss=w.get("id"), improvement=p.get("id"))
                for label, w, p in rows
            ]
        },
    )


def figures(ctx: FigureContext) -> Iterator[FigureOut]:
    """F29, F30 and F31 once per world group (one page each: a panel or rows per robot), only for a method
    comparison."""
    if not ctx.study.cross_method:
        return

    def robots(strata: Sequence[Stratum], worlds: Sequence[str]) -> int:
        return len({s.robot_id for s in strata})

    for build in (f29, f30, f31):
        yield from grouped(ctx, build, panels=robots)
