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

"""Generated families as random samples of their declared design, and the moving-object density contrast
(contract ``family-sample-v1``; plans/worlds/I6 §3.1 to §3.9, METHODOLOGY revision 2 §4.8 and §4.9).

A family (analysis block version 3, ``families``) is the set of worlds the export's ``worlds`` table assigns to
it; the world is the replication unit. Outcomes are dimensionless: rho = R/L (``R_completion``), success,
C(L) and A_L, and with ``shorter_horizons`` rho_h, C(h) and A_h at the horizons every world declares as the same
share of its L (horizons that drop out are named). Per profile and population:

- the world mean (spawns weighted equally within a world) with every run's value and the world's final-run
  accounting;
- the family average, the equal-weight mean of world means, with a t interval on the world means (n_F − 1
  degrees of freedom, labelled approximate) when the family has an interval; success has none;
- the profile contrast against Default, the mean over worlds of each world's mean block difference;
- the factorial effects of the declared factors (contrasts of the cell means), with Lenth's pseudo standard
  error on rho when there are at least 7 effects; with a missing cell only the unbalanced level differences of
  the worlds present (aliased with interactions), descriptive;
- the declared pairs (second minus first), descriptive;
- the optional variance split (I6 §3.5, O-5 (b)), descriptive.

A density contrast (``density_contrasts``) is the mean over the shared spawns of the moving world's spawn mean
minus the base world's, on rho and C(L), with the stratified Welch interval (primary) and the exact stratified
permutation interval (sensitivity, O-10 (b)); repetitions are never paired. ``exclusions_as_failures`` is always
added for it, and ``sequential_only`` for any unit whose runs mix execution modes.

Cells come from every final run, not only from the strata of ``data.load``: a world and spawn whose final runs
are all excluded still counts its runs as failures in ``exclusions_as_failures`` (L from the world's other
cells or the runs' own time limit, else no value, ``no_limit``). A world, spawn or outcome without values is
left out of an estimate and named, never refused. Hypotheses on families and contrasts (O-4 (a)) get the test
matched to their interval; Holm covers the plan's whole list (``merge_hypotheses``).
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import itertools
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as distributions

from . import stats
from .data import EXCLUDED, REPORTED, TIME_EPSILON, Row, Stratum, Study, Value
from .errors import Refused
from .estimates import POPULATIONS, horizon_value, interval_document, pairs, runs_for, time_value
from .settings import DensityContrast, Factor, Family, Hypothesis, Level, Settings
from .streams import Reads
from .worlds import World, worlds

CONTRACT = "family-sample-v1"
OUTCOMES = ("rho", "success", "C_L", "A_L")
CONTRAST_OUTCOMES = ("rho", "C_L")
HORIZON_OUTCOMES = ("rho", "C_L", "A_L")  # at h < L: rho_h, C(h), A_h (I6 §3.2)
INTERVAL_OUTCOMES = ("rho", "C_L", "A_L")  # success is descriptive (I6 §3.4, O-9 (a))
HYPOTHESIS_OUTCOMES = {"R_completion": "rho", "C_h": "C_L", "A_h": "A_L"}
# Lenth's method from a 2 × 2 × 2 on: with 3 effects it has 1 degree of freedom (I6 §3.4)
LENTH_MIN_EFFECTS = 7
PERMUTATION_MAX = (
    1_000_000  # relabellings enumerated exactly; above, the permutation interval is not given
)
FRACTION_DIGITS = 9
APPROXIMATE = "approximate; design points treated as exchangeable"
NO_MANIFEST = "no manifest in the export"
# the registry hash at the top level of a provenance document (suite provenance.REGISTRY_FILES; I1 §3.5, R-3)
REGISTRY_HASH_KEY = "world_registry_sha256"
LENTH_NULL = (
    "Lenth's ratio |c|/PSE against a seeded Monte Carlo null: m independent standard normal effects (global "
    "null, independent equal-variance normal effects); a calibration of this project, not of Lenth (1989)"
)
ESTIMANDS = (
    "family_world_mean",
    "family_average",
    "family_profile_contrast",
    "family_factorial_effect",
    "family_pair",
    "family_variance_split",
    "density_contrast",
)
Floats = NDArray[np.float64]


# Entries -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitRef:
    """What an entry names of its family or contrast."""

    kind: str  # "world_family" or "density_contrast"
    id: str
    robot: str | None
    mandated: Mapping[str, str] = field(
        default_factory=dict
    )  # population -> why it is always reported


def family_entry_id(
    estimand: str,
    unit: str,
    profile: str,
    *,
    robot: str | None,
    endpoint: str = "",
    member: str = "",
    fraction: float | None = None,
    population: str = "primary",
    variant: str = "",
) -> str:
    """The key of a ``family-sample-v1`` entry: ``estimates.entry_id``'s fields with the family or contrast ID in
    the world field, the world (world means) in the spawn field and the horizon as a share of L."""
    return "|".join(
        (
            estimand,
            endpoint,
            robot or "",
            unit,
            member,
            profile,
            "" if fraction is None else f"{fraction!r}L",
            population,
            variant,
        )
    )


def _entry(
    estimand: str,
    unit: UnitRef,
    profile: str,
    *,
    attempt_ids: Iterable[str],
    endpoint: str = "",
    member: str = "",
    fraction: float | None = None,
    population: str = "primary",
    variant: str = "",
    role: str = "descriptive",
    **values: Any,
) -> dict[str, Any]:
    return dict(
        id=family_entry_id(
            estimand,
            unit.id,
            profile,
            robot=unit.robot,
            endpoint=endpoint,
            member=member,
            fraction=fraction,
            population=population,
            variant=variant,
        ),
        contract=CONTRACT,
        estimand=estimand,
        endpoint=endpoint or None,
        robot_id=unit.robot,
        world_id=member or unit.id,
        spawn_id=None,
        **{unit.kind: unit.id},
        profile_id=profile,
        horizon_s=None,
        horizon_fraction=fraction,
        population=population,
        mandatory=unit.mandated.get(population),
        variant=variant or None,
        role=role,
        attempt_ids=sorted(set(attempt_ids)),
        **values,
    )


# Context: cells, worlds and units ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Context:
    study: Study
    settings: Settings
    reads: Reads
    table: Mapping[str, World]
    # world -> one stratum per (robot, spawn) holding every final run: data.load's, or one built here for a cell
    # without a reported run (L from the world's other cells or the runs' time limit, else NaN)
    cells: Mapping[str, tuple[Stratum, ...]]
    finals: Mapping[tuple[str, str], tuple[Row, ...]]  # (world, profile) -> final runs
    manifests: dict[str, Any] = field(default_factory=dict)

    def populations(self, mandated: Mapping[str, str]) -> list[str]:
        wanted = set(self.settings.sensitivity) | set(mandated)
        return ["primary", *(p for p in POPULATIONS[1:] if p in wanted)]


def _cells(study: Study) -> dict[str, tuple[Stratum, ...]]:
    known = {(s.robot_id, s.world_id, s.spawn_id): s for s in study.strata}
    orphans: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for run in study.runs:
        key = (run["robot_id"], run["world_id"], run["spawn_id"])
        if run["is_final"] and key not in known:
            orphans[key].append(run)
    built = dict(known)
    for (robot, world, spawn), runs in orphans.items():
        others = {s.limit_s for s in study.strata if s.robot_id == robot and s.world_id == world}
        own = {r["time_limit_s"] for r in runs}
        if len(others) == 1:
            limit = others.pop()
        elif len(own) == 1 and None not in own and float(next(iter(own))) > 0:
            limit = float(next(iter(own)))
        else:
            limit = math.nan
        runs.sort(key=lambda r: r["attempt_id"])
        built[(robot, world, spawn)] = Stratum(robot, world, spawn, "", limit, tuple(runs))
    grouped: dict[str, list[Stratum]] = defaultdict(list)
    for key in sorted(built):
        grouped[key[1]].append(built[key])
    return {world: tuple(group) for world, group in grouped.items()}


def _context(study: Study, settings: Settings, reads: Reads, table: Mapping[str, World]) -> Context:
    finals: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for run in study.runs:
        if run["is_final"]:
            finals[(run["world_id"], run["profile_id"])].append(run)
    return Context(
        study,
        settings,
        reads,
        table,
        MappingProxyType(_cells(study)),
        MappingProxyType({key: tuple(runs) for key, runs in finals.items()}),
    )


@dataclass(frozen=True, slots=True)
class Unit:
    """A declared family or density contrast with its worlds and cells."""

    ref: UnitRef
    members: tuple[
        str, ...
    ]  # a family's worlds from the worlds table; a contrast's base and moving world
    cells: Mapping[str, tuple[Stratum, ...]]  # world -> its cells, by robot and spawn
    profiles: tuple[str, ...]
    baseline: str | None

    def runs(self, classes: Iterable[str]) -> list[Row]:
        wanted = set(classes)
        return [
            run
            for group in self.cells.values()
            for cell in group
            for run in cell.finals
            if run["report_class"] in wanted
        ]


def _unit(context: Context, kind: str, identity: str, members: Sequence[str]) -> Unit:
    cells = {world: context.cells.get(world, ()) for world in sorted(members)}
    finals = [run for group in cells.values() for cell in group for run in cell.finals]
    reported = [run for run in finals if run["report_class"] == REPORTED]
    robots = sorted({run["robot_id"] for run in reported or finals})
    robot = robots[0] if robots else None
    modes = {run["concurrent_wall_fraction"] == 0 for run in reported}
    mandated = {"sequential_only": "mixed execution modes (METHODOLOGY §2.4, revision 2)"}
    if kind == "density_contrast":
        mandated["exclusions_as_failures"] = "always reported for a density contrast (I6 §3.9)"
    if len(modes) < 2:
        mandated.pop("sequential_only")
    return Unit(
        UnitRef(kind, identity, robot, MappingProxyType(mandated)),
        tuple(members),
        cells,
        tuple(sorted({run["profile_id"] for run in finals})),
        context.study.baselines.get(robot) if robot else None,
    )


def _canonical(value: Any) -> str:
    """A level as compared with the worlds table: integral floats as integers, lists and tuples alike,
    recursively (YAML 3.0 equals 3). Mirrored in ``suite/benchmark_suite/analysis_block.py``: change both."""

    def plain(item: Any) -> Any:
        if isinstance(item, bool):
            return item
        if isinstance(item, float) and item.is_integer():
            return int(item)
        if isinstance(item, list | tuple):
            return [plain(v) for v in item]
        if isinstance(item, dict):
            return {str(k): plain(v) for k, v in item.items()}
        return item

    return json.dumps(plain(value), sort_keys=True)


def _same(a: Any, b: Level) -> bool:
    return _canonical(a) == _canonical(b)


def _members(family: Family, table: Mapping[str, World]) -> list[str]:
    """The family's worlds by the worlds table; a family absent from it, holding a world not of tier F or
    pairing a world outside it is refused (I6 §3.8)."""
    members = sorted(w.world_id for w in table.values() if w.family == family.id)
    if not members:
        raise Refused(
            f"family {family.id}: the export's worlds table holds no world of that family",
            column="family",
            stratum=family.id,
        )
    wrong = sorted(w for w in members if table[w].tier != "F")
    if wrong:
        raise Refused(
            f"family {family.id}: worlds {wrong} are not of tier F",
            column="tier",
            stratum=family.id,
        )
    for first, second in family.pairs:
        outside = [w for w in (first, second) if w not in members]
        if outside:
            raise Refused(
                f"family {family.id}: pair {first}, {second} names worlds outside the family: "
                f"{outside}",
                column="pairs",
                stratum=family.id,
            )
    return members


def _factorial(
    family: Family, table: Mapping[str, World], members: Sequence[str]
) -> dict[str, tuple[int, ...]]:
    """The factorial's worlds (those whose parameters equal ``fixed``) with their coded levels (−1 low, +1 high),
    or a refusal when a key or level does not match a world's parameters or the cells are not one world each."""
    if not family.factors:
        return {}
    coded: dict[str, tuple[int, ...]] = {}
    for world in members:
        point = table[world].design_point
        for key in [*family.fixed, *(f.key for f in family.factors)]:
            if key not in point:
                raise Refused(
                    f"family {family.id}: world {world} has no parameter {key}",
                    column="design_point_json",
                    stratum=family.id,
                )
        if not all(_same(point[key], value) for key, value in family.fixed.items()):
            continue
        coded[world] = tuple(
            _code(family, world, factor, point[factor.key]) for factor in family.factors
        )
    expected = set(itertools.product((-1, 1), repeat=len(family.factors)))
    found: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for world, cell in coded.items():
        found[cell].append(world)
    crowded = sorted(w for w in found.values() if len(w) > 1)
    if crowded or set(found) != expected:
        raise Refused(
            f"family {family.id}: the declared factors need one world per cell of the {len(expected)}-cell "
            f"factorial; empty cells {sorted(expected - set(found))}, crowded cells {crowded} (declare "
            "fixed parameters)",
            column="design_point_json",
            stratum=family.id,
        )
    return coded


def _code(family: Family, world: str, factor: Factor, level: Any) -> int:
    if _same(level, factor.low):
        return -1
    if _same(level, factor.high):
        return 1
    raise Refused(
        f"family {family.id}: world {world} has {factor.key} = {level!r}, neither declared level "
        f"{factor.low!r} nor {factor.high!r}",
        column="design_point_json",
        stratum=family.id,
    )


# Homogeneity (I6 §3.8) -----------------------------------------------------------------------------------------


def _manifest(context: Context, attempt: str) -> Any:
    if attempt not in context.manifests:
        data = context.reads.file(f"raw/{attempt}/manifest.json")
        context.manifests[attempt] = None if data is None else json.loads(data)
    return context.manifests[attempt]


def _slam(context: Context, run: Row) -> Any:
    manifest = _manifest(context, run["attempt_id"])
    if manifest is None:
        return NO_MANIFEST
    profile = (manifest.get("robot_configuration") or {}).get("comparison_profile") or {}
    return profile.get("slam_sha256")


def _time_limit_rule(context: Context, run: Row) -> Any:
    for campaign in context.study.export.study.get("campaigns") or ():
        if campaign.get("id") == run["campaign_id"]:
            return campaign.get("time_limit_rule")
    return None


def _registry(context: Context, run: Row) -> Any:
    if "provenance.json" in context.study.export.files:
        context.reads.files.add("provenance.json")
    documents = context.study.export.provenance.get("documents") or {}
    return (documents.get(run["provenance_id"]) or {}).get(REGISTRY_HASH_KEY)


def _worldgen(context: Context, run: Row) -> Any:
    world = context.table.get(run["world_id"])
    return None if world is None else world.generator_version


CHECKS: tuple[tuple[str, Callable[[Context, Row], Any]], ...] = (
    ("robot_id", lambda context, run: run["robot_id"]),
    # before provenance_id: the registry hash lives in the provenance document, so a registry mix also mixes
    # provenance_id, and the more specific refusal is named
    ("registry_sha256", _registry),
    ("provenance_id", lambda context, run: run["provenance_id"]),
    ("slam_sha256", _slam),
    ("time_limit_rule", _time_limit_rule),
)


def _homogeneous(context: Context, unit: Unit) -> None:
    """Refuse a unit whose runs (every population it reports: reported runs, and excluded runs when
    ``exclusions_as_failures`` is used) mix robot, registry, code identity, SLAM configuration, time-limit rule
    or (a family) generator version, naming each value's runs (RO-22). Only uniformity is checked, not the
    values themselves."""
    classes = [REPORTED]
    if "exclusions_as_failures" in context.populations(unit.ref.mandated):
        classes += EXCLUDED
    runs = sorted(unit.runs(classes), key=lambda r: (r["campaign_id"], r["attempt_id"]))
    checks = list(CHECKS)
    if unit.ref.kind == "world_family":
        checks.append(("worldgen_version", _worldgen))
    for column, read in checks:
        grouped: dict[str, list[Row]] = defaultdict(list)
        for run in runs:
            grouped[json.dumps(read(context, run))].append(run)
        if len(grouped) > 1:
            raise Refused(
                f"{unit.ref.id}: runs mix {column} values {sorted(grouped)}; one {unit.ref.kind} needs one",
                column=column,
                stratum=unit.ref.id,
                groups=[
                    dict(
                        value=json.loads(value),
                        runs=[
                            dict(
                                attempt_id=r["attempt_id"],
                                campaign_id=r["campaign_id"],
                                block_id=r["block_id"],
                                profile_id=r["profile_id"],
                                world_id=r["world_id"],
                            )
                            for r in members
                        ],
                    )
                    for value, members in sorted(grouped.items())
                ],
            )


# Per-run and per-world values ----------------------------------------------------------------------------------


def run_value(
    study: Study, run: Row, outcome: str, limit: float, horizon: float | None = None
) -> Value:
    """One run's outcome at h (L when None): rho_h = min(R, h)/h, success (completed with T_c ≤ L, as
    ``data.restricted_time`` marks the event for rho; an excluded run
    counts as not successful; no value for a completion without motion, as rho), C(h), A_h (excluded runs
    have none, ``estimates.horizon_value``); no value without a time limit (``no_limit``)."""
    if not math.isfinite(limit):
        return Value(None, "no_limit")
    h = limit if horizon is None else horizon
    if outcome == "rho":
        time = time_value(run, "R_completion", limit)
        if time.value is None:
            return time
        return Value(min(float(time.value), h) / h, time.reason, time.event)
    if outcome == "success":
        if run["report_class"] in EXCLUDED:
            return Value(0.0, "excluded_as_failure")
        if run["outcome"] == "completed" and not run["motion_started"]:
            return Value(None, "completion_without_motion")
        reached = (
            run["outcome"] == "completed"
            and bool(run["completion_event"])
            and run["completion_s"] <= limit + TIME_EPSILON
        )
        return Value(1.0 if reached else 0.0, "observed")
    estimand = {"C_L": "C_h", "A_L": "A_h"}[outcome]
    return horizon_value(study, run, estimand, h, limit)


@dataclass(frozen=True, slots=True)
class WorldValue:
    """A world's mean of an outcome over its spawns (each spawn's mean, weighted equally), or None."""

    world: str
    mean: float | None
    n: int  # runs with a value
    spawn_means: dict[str, float]
    spawns_without_value: list[str]
    values: list[dict[str, Any]]  # every run's value
    no_value: dict[str, int]
    attempt_ids: list[str]
    final_runs: dict[str, int]
    limit_s: list[float | None]


def _horizon(context: Context, cell: Stratum, fraction: float | None) -> float | None:
    """The world's declared horizon at a share of its L (None: at L). ``_fractions`` keeps only shares every
    cell with a finite L declares, so the lookup finds one; a cell without L has no values anyway."""
    if fraction is None or not math.isfinite(cell.limit_s):
        return None
    for h in context.settings.horizons_s.get(cell.world_id, ()):
        if round(h / cell.limit_s, FRACTION_DIGITS) == fraction:
            return h
    raise RuntimeError(f"{cell.world_id} declares no horizon at {fraction} L")


def world_value(
    context: Context,
    world: str,
    profile: str,
    outcome: str,
    population: str,
    fraction: float | None = None,
) -> WorldValue:
    spawn_means: dict[str, float] = {}
    without: list[str] = []
    values: list[dict[str, Any]] = []
    reasons: dict[str, int] = defaultdict(int)
    cells = context.cells.get(world, ())
    for cell in cells:
        horizon = _horizon(context, cell, fraction)
        found = []
        for run in runs_for(cell, profile, population):
            value = run_value(context.study, run, outcome, cell.limit_s, horizon)
            if value.value is None:
                reasons[value.reason] += 1
                continue
            found.append(float(value.value))
            values.append(
                dict(
                    attempt_id=run["attempt_id"],
                    spawn_id=cell.spawn_id,
                    block_id=run["block_id"],
                    value=float(value.value),
                    reason=value.reason,
                )
            )
        if found:
            spawn_means[cell.spawn_id] = float(np.mean(found))
        elif any(run["profile_id"] == profile for run in cell.finals):
            without.append(cell.spawn_id)
    finals: dict[str, int] = defaultdict(int)
    for run in context.finals.get((world, profile), ()):
        finals[run["report_class"]] += 1
    return WorldValue(
        world,
        float(np.mean(list(spawn_means.values()))) if spawn_means else None,
        len(values),
        dict(sorted(spawn_means.items())),
        sorted(without),
        values,
        dict(sorted(reasons.items())),
        sorted(v["attempt_id"] for v in values),
        dict(sorted(finals.items())),
        sorted({c.limit_s if math.isfinite(c.limit_s) else None for c in cells}, key=str),
    )


@dataclass(frozen=True, slots=True)
class WorldDifference:
    """A world's mean over spawns of each spawn's mean block difference (profile − Default), or None."""

    world: str
    mean: float | None
    n_pairs: int
    spawn_means: dict[str, float]
    missing_pairs: dict[str, int]
    attempt_ids: list[str]


def world_difference(
    context: Context,
    world: str,
    profile: str,
    baseline: str,
    outcome: str,
    population: str,
    fraction: float | None = None,
) -> WorldDifference:
    spawn_means: dict[str, float] = {}
    missing: dict[str, int] = defaultdict(int)
    attempts: list[str] = []
    n = 0
    for cell in context.cells.get(world, ()):
        horizon = _horizon(context, cell, fraction)

        def read(run: Row, limit: float = cell.limit_s, h: float | None = horizon) -> Value:
            return run_value(context.study, run, outcome, limit, h)

        found = pairs(cell, profile, baseline, population, read)
        for key, count in found.missing.items():
            missing[key] += count
        if found.differences:
            spawn_means[cell.spawn_id] = float(np.mean(found.differences))
            n += len(found.differences)
            attempts += found.attempt_ids
    return WorldDifference(
        world,
        float(np.mean(list(spawn_means.values()))) if spawn_means else None,
        n,
        dict(sorted(spawn_means.items())),
        dict(sorted(missing.items())),
        sorted(attempts),
    )


# Intervals, tests and Lenth's method ---------------------------------------------------------------------------


def world_t_interval(means: Sequence[float], alpha: float) -> stats.Interval:
    """The t interval on world means (n − 1 degrees of freedom), labelled approximate and not truncated to the
    outcome's range; a zero SD or fewer than 2 worlds is "not estimable", never a zero-width interval."""
    method = f"t on world means, {APPROXIMATE}"
    if len(means) < 2:
        return stats.Interval(None, None, 1 - alpha, method, "not estimable: fewer than 2 worlds")
    if float(np.std(means, ddof=1)) <= stats.TOLERANCE * max(1.0, float(np.max(np.abs(means)))):
        return stats.Interval(
            None, None, 1 - alpha, method, "not estimable: the world means do not vary"
        )
    interval = stats.t_interval(means, alpha)
    return stats.Interval(interval.lower, interval.upper, interval.level, method, "approximate")


def one_sample_t_p(values: Sequence[float]) -> float | None:
    """Two-sided one-sample t test of mean 0 on world differences; None when not estimable."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return None
    sd = float(x.std(ddof=1))
    if sd <= stats.TOLERANCE * max(1.0, float(np.max(np.abs(x)))):
        return None
    t = float(x.mean()) / (sd / math.sqrt(x.size))
    return float(2 * distributions.t.sf(abs(t), x.size - 1))


@dataclass(frozen=True, slots=True)
class Lenth:
    """Lenth's pseudo standard error of m effects with its margins (I6 §3.4.1); ``pse`` None: not estimable."""

    s0: float
    pse: float | None
    df: float
    me: float | None
    sme: float | None
    gamma: float


def lenth_pse(effects: Sequence[float]) -> tuple[float, float | None]:
    """s0 = 1.5 · median |c| and PSE = 1.5 · median{|c| : |c| < 2.5 s0}, with contrasts that are zero up to
    rounding counted as zero; PSE None when it is 0 or undefined."""
    c = np.abs(np.asarray(effects, dtype=np.float64))
    tolerance = stats.TOLERANCE * max(1.0, float(c.max(initial=0.0)))
    c = np.where(c <= tolerance, 0.0, c)
    s0 = 1.5 * float(np.median(c))
    kept = c[c < 2.5 * s0]
    if kept.size == 0:
        return s0, None
    pse = 1.5 * float(np.median(kept))
    return s0, pse if pse > tolerance else None


def lenth(effects: Sequence[float], alpha: float) -> Lenth:
    """Lenth's PSE with ME = t_{1−α/2,d} · PSE and SME = t_{γ,d} · PSE, d = m/3, γ = (1 + (1 − α)^{1/m})/2: Lenth's
    0.975 and 0.95 written for the plan's α (the project's generalisation)."""
    m = len(effects)
    df = m / 3
    gamma = (1 + (1 - alpha) ** (1 / m)) / 2
    s0, pse = lenth_pse(effects)
    if pse is None:
        return Lenth(s0, None, df, None, None, gamma)
    return Lenth(
        s0,
        pse,
        df,
        float(distributions.t.ppf(1 - alpha / 2, df)) * pse,
        float(distributions.t.ppf(gamma, df)) * pse,
        gamma,
    )


def lenth_p(ratio: float, m: int, *, resamples: int, seed: int) -> float:
    """P(|z_1|/PSE(z) ≥ ratio) for m independent standard normal effects z (``LENTH_NULL``), by seeded Monte
    Carlo, (1 + #extreme)/(B + 1); the ratio does not depend on scale and the effects are exchangeable, so the
    first stands for any."""
    rng = stats.random_stream(seed, stats.LENTH_STREAM)
    extreme, remaining = 0, resamples
    while remaining:
        batch = min(remaining, 10_000)
        z = np.abs(rng.standard_normal((batch, m)))
        s0 = 1.5 * np.median(z, axis=1)
        kept = np.where(z < 2.5 * s0[:, None], z, np.nan)
        pse = 1.5 * np.nanmedian(kept, axis=1)
        extreme += int(np.sum(z[:, 0] / pse >= ratio - stats.TOLERANCE))
        remaining -= batch
    return (1 + extreme) / (resamples + 1)


def effect_names(family: Family) -> list[tuple[str, tuple[int, ...]]]:
    """Every factorial effect: its name (factor keys joined by ×) and the factor indices, main effects first."""
    keys = [f.key for f in family.factors]
    return [
        ("×".join(keys[i] for i in subset), subset)
        for size in range(1, len(keys) + 1)
        for subset in itertools.combinations(range(len(keys)), size)
    ]


def factorial_effects(
    cells: Mapping[str, tuple[int, ...]], means: Mapping[str, float], family: Family
) -> dict[str, float | None]:
    """Each effect as the contrast 2 · mean(y · Π x) of the cell means when every cell has a value; otherwise
    only the unbalanced level differences of the main factors (mean at high minus mean at low over the worlds
    present; aliased with interactions), interactions None."""
    complete = all(world in means for world in cells)
    found: dict[str, float | None] = {}
    for name, subset in effect_names(family):
        if complete:
            products = [
                means[w] * math.prod(levels[i] for i in subset) for w, levels in cells.items()
            ]
            found[name] = 2 * float(np.mean(products))
        elif len(subset) == 1:
            index = subset[0]
            high = [means[w] for w, levels in cells.items() if w in means and levels[index] > 0]
            low = [means[w] for w, levels in cells.items() if w in means and levels[index] < 0]
            found[name] = float(np.mean(high)) - float(np.mean(low)) if high and low else None
        else:
            found[name] = None
    return found


@dataclass(frozen=True, slots=True)
class Contrast:
    """A density contrast over usable spawns: estimate, stratified Welch interval and its p-value."""

    estimate: float | None
    interval: stats.Interval
    p_value: float | None


def welch(cells: Sequence[tuple[Sequence[float], Sequence[float]]], alpha: float) -> Contrast:
    """The mean over strata of (moving − base) with the stratified Welch interval, Satterthwaite degrees of
    freedom, and its two-sided p-value (I6 §3.9). Cells hold (base, moving) values, each at least 2 (the
    caller leaves out other spawns). A cell without variation adds nothing to the variance, so the interval
    can be too narrow when runs pile up at rho = 1; "not estimable" when no cell varies."""
    method = "stratified Welch, Satterthwaite df, approximate"
    k = len(cells)
    if k == 0:
        return Contrast(
            None, stats.Interval(None, None, 1 - alpha, method, "not estimable: no spawn"), None
        )
    estimate = float(np.mean([np.mean(m) - np.mean(b) for b, m in cells]))
    terms = [(float(np.var(x, ddof=1)) / len(x) / k**2, len(x)) for pair in cells for x in pair]
    variance = sum(term for term, _ in terms)
    if variance <= 0:
        note = "not estimable: no cell varies"
        return Contrast(estimate, stats.Interval(None, None, 1 - alpha, method, note), None)
    df = variance**2 / sum(term**2 / (n - 1) for term, n in terms if term > 0)
    half = float(distributions.t.ppf(1 - alpha / 2, df)) * math.sqrt(variance)
    p = float(2 * distributions.t.sf(abs(estimate) / math.sqrt(variance), df))
    zero = sum(1 for term, _ in terms if term == 0)
    # tables.welch_note reads the count back from this text: change both together
    note = f"df = {df:.3f}" + (f"; {zero} cells without variation" if zero else "")
    interval = stats.Interval(estimate - half, estimate + half, 1 - alpha, method, note)
    return Contrast(estimate, interval, p)


def _relabellings(base: Sequence[float], moving: Sequence[float]) -> tuple[Floats, Floats]:
    """Every split of one spawn's runs into the two worlds' group sizes: the difference of means a and its slope
    b in a shift θ of the moving world's values (the difference at θ is a − θ·b)."""
    pool = np.asarray([*base, *moving], dtype=np.float64)
    n_b, n_m = len(base), len(moving)
    chosen = np.array(list(itertools.combinations(range(n_b + n_m), n_m)), dtype=np.int64)
    mask = np.zeros((len(chosen), n_b + n_m), dtype=bool)
    mask[np.arange(len(chosen))[:, None], chosen] = True
    total = mask @ pool
    a = total / n_m - (pool.sum() - total) / n_b
    j = mask[:, n_b:].sum(axis=1)  # the moving world's own runs kept in the moving group
    return a, j / n_m - (n_m - j) / n_b


def permutation(
    cells: Sequence[tuple[Sequence[float], Sequence[float]]], alpha: float
) -> tuple[stats.Interval, float | None]:
    """The exact stratified permutation test of the density contrast (I6 O-10 (b)): world labels relabelled
    within each spawn, every relabelling enumerated (C(6,3)^3 = 8,000 with 3 + 3 runs in 3 spawns), the
    statistic the estimate itself; its p-value and the interval of shifts θ it does not reject. Not computed
    above ``PERMUTATION_MAX`` relabellings."""
    method = "inverted exact stratified permutation test (sensitivity)"
    count = math.prod(math.comb(len(b) + len(m), len(m)) for b, m in cells)
    if not cells or count > PERMUTATION_MAX:
        note = "not computed: " + ("no spawn" if not cells else f"{count} relabellings")
        return stats.Interval(None, None, 1 - alpha, method, note), None
    a_total: Floats = np.zeros(1)
    b_total: Floats = np.zeros(1)
    for base, moving in cells:
        a, b = _relabellings(base, moving)
        a_total = (a_total[:, None] + a[None, :]).ravel()
        b_total = (b_total[:, None] + b[None, :]).ravel()
    a_total, b_total = a_total / len(cells), b_total / len(cells)
    observed = float(np.mean([np.mean(m) - np.mean(b) for b, m in cells]))
    scale = max(1.0, float(np.abs(a_total).max()))

    def p_at(thetas: Floats) -> Floats:
        out = np.empty(thetas.size)
        for start in range(0, thetas.size, 256):
            t = thetas[start : start + 256, None]
            extreme = np.abs(a_total[None, :] - t * b_total[None, :]) >= (
                np.abs(observed - t) - stats.TOLERANCE * scale
            )
            out[start : start + 256] = extreme.mean(axis=1)
        return out

    p_value = float(p_at(np.zeros(1))[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        roots = np.concatenate(
            [(a_total - observed) / (b_total - 1), (a_total + observed) / (b_total + 1)]
        )
    points = np.unique(np.round(roots[np.isfinite(roots)], 12))
    span = float(points.max() - points.min()) if points.size else 1.0
    probes = np.concatenate(
        [points, (points[:-1] + points[1:]) / 2, [points.min() - span - 1, points.max() + span + 1]]
    )
    accepted = probes[p_at(probes) > alpha + stats.TOLERANCE]
    if accepted.size == 0:
        return stats.Interval(None, None, 1 - alpha, method, "no shift accepted"), p_value
    if accepted.min() < points.min() or accepted.max() > points.max():
        note = f"{count} relabellings cannot reach level {1 - alpha:g}: unbounded"
        return stats.Interval(None, None, 1 - alpha, method, note), p_value
    lower = float(max((p for p in points if p <= accepted.min()), default=accepted.min()))
    upper = float(min((p for p in points if p >= accepted.max()), default=accepted.max()))
    return stats.Interval(lower, upper, 1 - alpha, method, f"{count} relabellings"), p_value


# Family estimates ----------------------------------------------------------------------------------------------


def _fractions(context: Context, unit: Unit) -> tuple[list[float], dict[str, list[float]]]:
    """The shares of L at which every cell of the unit with a time limit declares a horizon below L
    (shorter_horizons), and per world the declared horizons that drop out because another world lacks them."""
    if "shorter_horizons" not in context.settings.sensitivity:
        return [], {}
    shares: dict[str, set[float]] = {}
    declared: dict[str, dict[float, float]] = {}
    for world, group in unit.cells.items():
        for cell in group:
            if not math.isfinite(cell.limit_s):
                continue
            mine = {
                round(h / cell.limit_s, FRACTION_DIGITS): h
                for h in context.settings.horizons_s.get(world, ())
                if 0 < h < cell.limit_s
            }
            declared.setdefault(world, {}).update(mine)
            shares[f"{world}/{cell.robot_id}/{cell.spawn_id}"] = set(mine)
    common = set.intersection(*shares.values()) if shares else set()
    dropped = {
        world: sorted(h for share, h in found.items() if share not in common)
        for world, found in sorted(declared.items())
        if set(found) - common
    }
    return sorted(common), dropped


def _missing_note(missing: Iterable[str], kind: str) -> str | None:
    names = list(missing)
    if not names:
        return None
    return f"conditional on the remaining {kind}; without a value: {', '.join(names)}"


@dataclass(frozen=True, slots=True)
class Declared:
    """A family with its worlds, factorial cells and unit."""

    family: Family
    unit: Unit
    cells: Mapping[str, tuple[int, ...]]
    dropped_horizons: Mapping[str, list[float]]


def _family_profile(
    context: Context,
    declared: Declared,
    profile: str,
    outcome: str,
    population: str,
    fraction: float | None,
) -> list[dict[str, Any]]:
    family, unit = declared.family, declared.unit
    found = {
        world: world_value(context, world, profile, outcome, population, fraction)
        for world in unit.members
    }
    means = {w: v.mean for w, v in found.items() if v.mean is not None}
    missing = [w for w in unit.members if w not in means]
    entries = [
        _entry(
            "family_world_mean",
            unit.ref,
            profile,
            attempt_ids=value.attempt_ids,
            member=world,
            endpoint=outcome,
            fraction=fraction,
            population=population,
            design_point=context.table[world].design_point,
            estimate=value.mean,
            n=value.n,
            spawn_means=value.spawn_means,
            spawns_without_value=value.spawns_without_value,
            values=value.values,
            no_value=value.no_value,
            final_runs=value.final_runs,
            limit_s=value.limit_s,
        )
        for world, value in found.items()
    ]
    world_means = [means[w] for w in sorted(means)]
    with_interval = family.interval and outcome in INTERVAL_OUTCOMES
    interval = world_t_interval(world_means, context.settings.alpha) if with_interval else None
    entries.append(
        _entry(
            "family_average",
            unit.ref,
            profile,
            attempt_ids=(a for v in found.values() for a in v.attempt_ids),
            endpoint=outcome,
            fraction=fraction,
            population=population,
            estimate=float(np.mean(world_means)) if world_means else None,
            n_worlds=len(world_means),
            worlds_used=sorted(means),
            worlds_missing=missing,
            world_means=dict(sorted(means.items())),
            n_per_world={w: v.n for w, v in sorted(found.items())},
            interval=interval_document(interval),
            note=_missing_note(missing, "worlds"),
            label=None if with_interval else "descriptive, no interval",
            horizons_dropped=dict(declared.dropped_horizons),
        )
    )
    if fraction is None and outcome != "success" and declared.cells:
        entries += _effects(context, declared, profile, outcome, population, found)
    if fraction is None and outcome != "success":
        entries += [
            _pair(unit, profile, outcome, population, found, first, second)
            for first, second in family.pairs
        ]
    return entries


def _pair(
    unit: Unit,
    profile: str,
    outcome: str,
    population: str,
    found: Mapping[str, WorldValue],
    first: str,
    second: str,
) -> dict[str, Any]:
    a, b = found[first].mean, found[second].mean
    return _entry(
        "family_pair",
        unit.ref,
        profile,
        attempt_ids=[*found[first].attempt_ids, *found[second].attempt_ids],
        variant=f"pair:{first}:{second}",
        endpoint=outcome,
        population=population,
        comparison=f"{second} − {first}",
        estimate=None if a is None or b is None else b - a,
        worlds=[first, second],
        world_means={first: a, second: b},
        worlds_missing=[w for w in (first, second) if found[w].mean is None],
        n_per_world={w: found[w].n for w in (first, second)},
        label="descriptive",
    )


def _effects(
    context: Context,
    declared: Declared,
    profile: str,
    outcome: str,
    population: str,
    found: Mapping[str, WorldValue],
) -> list[dict[str, Any]]:
    family, unit, cells = declared.family, declared.unit, declared.cells
    means = {w: v.mean for w, v in found.items() if w in cells and v.mean is not None}
    effects = factorial_effects(cells, means, family)
    missing = sorted(w for w in cells if w not in means)
    method = None
    if not missing and outcome == "rho" and len(effects) >= LENTH_MIN_EFFECTS:
        method = lenth([v for v in effects.values() if v is not None], context.settings.alpha)
    if missing:
        label = (
            "descriptive only: unbalanced level difference over the worlds present; aliased with "
            "interactions; not the factorial main effect (missing: " + ", ".join(missing) + ")"
        )
    elif method is None:
        label = "descriptive"
    elif method.pse is None:
        label = "not estimable: PSE is 0"
    else:
        label = "approximate (Lenth's PSE; no pure error with one seed per setting)"
    attempts = [a for w in cells for a in found[w].attempt_ids]
    entries = []
    for name, subset in effect_names(family):
        estimate = effects[name]
        extra: dict[str, Any] = {}
        if method is not None:
            extra = dict(
                lenth=dict(
                    s0=method.s0,
                    pse=method.pse,
                    df=method.df,
                    me=method.me,
                    sme=method.sme,
                    gamma=method.gamma,
                    level=1 - context.settings.alpha,
                    note=None if method.pse is not None else "not estimable: PSE is 0",
                    source="plans/worlds/I6 §3.4.1",
                ),
                ratio=None
                if method.pse is None or estimate is None
                else abs(estimate) / method.pse,
            )
        chosen = [family.factors[i] for i in subset]
        entries.append(
            _entry(
                "family_factorial_effect",
                unit.ref,
                profile,
                attempt_ids=attempts if estimate is not None else (),
                endpoint=outcome,
                population=population,
                variant=f"factor:{name}",
                factors=[f.key for f in chosen],
                levels={f.key: [f.low, f.high] for f in chosen},
                main_effect=len(subset) == 1,
                estimate=estimate,
                cells={
                    w: dict(levels=list(c), mean=means.get(w)) for w, c in sorted(cells.items())
                },
                worlds_missing=missing,
                label=label,
                **extra,
            )
        )
    return entries


def _profile_contrast(
    context: Context,
    declared: Declared,
    profile: str,
    outcome: str,
    population: str,
    fraction: float | None,
) -> dict[str, Any]:
    family, unit = declared.family, declared.unit
    found = {
        world: world_difference(
            context, world, profile, unit.baseline or "", outcome, population, fraction
        )
        for world in unit.members
    }
    means = {w: d.mean for w, d in found.items() if d.mean is not None}
    missing = [w for w in unit.members if w not in means]
    values = [means[w] for w in sorted(means)]
    with_interval = family.interval and outcome in INTERVAL_OUTCOMES
    interval = world_t_interval(values, context.settings.alpha) if with_interval else None
    return _entry(
        "family_profile_contrast",
        unit.ref,
        profile,
        attempt_ids=(a for d in found.values() for a in d.attempt_ids),
        endpoint=outcome,
        fraction=fraction,
        population=population,
        comparison=f"{profile} − Default",
        estimate=float(np.mean(values)) if values else None,
        n_worlds=len(values),
        worlds_used=sorted(means),
        worlds_missing=missing,
        world_differences=dict(sorted(means.items())),
        n_pairs_per_world={w: d.n_pairs for w, d in sorted(found.items())},
        missing_pairs={w: d.missing_pairs for w, d in sorted(found.items()) if d.missing_pairs},
        interval=interval_document(interval),
        note=_missing_note(missing, "worlds"),
        label=None if with_interval else "descriptive, no interval",
    )


def variance_split(
    cells: Mapping[str, tuple[int, ...]], runs: Mapping[str, Mapping[str, Sequence[float]]]
) -> dict[str, Any] | None:
    """ANOVA of Y_wk = mu + Σ beta_j x_wj + u_w + e with the main effects fitted on the world means and the
    interactions pooled into u_w (I6 §3.5, O-5 (b)). ``runs``: world -> spawn -> values. sigma_e² is the pooled
    within-spawn variance; sigma_w² = s_r² − sigma_e² · mean(1/n_w), with s_r² the residual variance of the world
    means (unbiased for a complete orthogonal 2^k, whose leverages are equal); None when a world has no value or
    no degrees of freedom remain."""
    worlds = sorted(cells)
    if any(not any(runs.get(w, {}).values()) for w in worlds):
        return None
    k = len(next(iter(cells.values())))
    y = np.array([np.mean([np.mean(v) for v in runs[w].values() if len(v)]) for w in worlds])
    x = np.column_stack(
        [np.ones(len(worlds)), np.array([cells[w] for w in worlds], dtype=np.float64)]
    )
    df_world = len(worlds) - 1 - k
    within = [np.asarray(v, dtype=np.float64) for w in worlds for v in runs[w].values() if len(v)]
    df_within = sum(v.size - 1 for v in within)
    if df_world <= 0 or df_within <= 0:
        return None
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    s_r2 = float(residuals @ residuals) / df_world
    sigma_e2 = float(sum(float(((v - v.mean()) ** 2).sum()) for v in within)) / df_within
    n_w = np.array([sum(len(v) for v in runs[w].values()) for w in worlds], dtype=np.float64)
    raw = s_r2 - sigma_e2 * float(np.mean(1 / n_w))
    sigma_w2 = max(raw, 0.0)
    total = sigma_w2 + sigma_e2
    pooled = {w: np.concatenate([np.asarray(v) for v in runs[w].values()]) for w in worlds}
    return dict(
        sigma_w2=sigma_w2,
        sigma_w2_raw=raw,
        sigma_e2=sigma_e2,
        df_world=df_world,
        df_within=df_within,
        share_between=sigma_w2 / total if total > 0 else None,
        coefficients=[float(b) for b in beta],
        world_residuals=dict(zip(worlds, [float(r) for r in residuals], strict=True)),
        world_sd={w: float(np.std(v, ddof=1)) if v.size > 1 else None for w, v in pooled.items()},
        note=f"main effects only; interactions pooled into the world variance (assumption); sigma_w² is an "
        f"upper-biased {df_world}-df guide to the seed variance"
        + ("; sigma_w² truncated at 0" if raw < 0 else ""),
    )


def _variance_entry(
    context: Context, declared: Declared, profile: str, outcome: str
) -> dict[str, Any]:
    family, unit, cells = declared.family, declared.unit, declared.cells
    runs: dict[str, dict[str, list[float]]] = {}
    attempts: list[str] = []
    for world in cells:
        value = world_value(context, world, profile, outcome, "primary")
        per_spawn: dict[str, list[float]] = defaultdict(list)
        for item in value.values:
            per_spawn[item["spawn_id"]].append(item["value"])
        runs[world] = dict(per_spawn)
        attempts += value.attempt_ids
    split = variance_split(cells, runs)
    if split is not None:
        names = ["intercept", *(f.key for f in family.factors)]
        split["coefficients"] = dict(zip(names, split["coefficients"], strict=True))
    missing = sorted(w for w in cells if not any(runs[w].values()))
    reason = "worlds without a value: " + ", ".join(missing) if missing else "no degrees of freedom"
    return _entry(
        "family_variance_split",
        unit.ref,
        profile,
        attempt_ids=attempts,
        endpoint=outcome,
        estimate=split,
        worlds_missing=missing,
        note=None if split is not None else "not estimable: " + reason,
        label="descriptive; primary population; pre-registered model (I6 §3.5)",
    )


def _declared(context: Context, family: Family) -> Declared:
    members = _members(family, context.table)
    cells = _factorial(family, context.table, members)
    unit = _unit(context, "world_family", family.id, members)
    _homogeneous(context, unit)
    fractions = _fractions(context, unit)
    return Declared(family, unit, cells, fractions[1])


def family_entries(context: Context, declared: Declared) -> list[dict[str, Any]]:
    family, unit = declared.family, declared.unit
    fractions, _ = _fractions(context, unit)
    entries: list[dict[str, Any]] = []
    for profile in unit.profiles:
        contrast = unit.baseline is not None and profile != unit.baseline
        for population in context.populations(unit.ref.mandated):
            for outcome in OUTCOMES:
                entries += _family_profile(context, declared, profile, outcome, population, None)
                if contrast:
                    entries.append(
                        _profile_contrast(context, declared, profile, outcome, population, None)
                    )
        for fraction in fractions:
            for outcome in HORIZON_OUTCOMES:
                entries += _family_profile(
                    context, declared, profile, outcome, "shorter_horizons", fraction
                )
                if contrast:
                    entries.append(
                        _profile_contrast(
                            context, declared, profile, outcome, "shorter_horizons", fraction
                        )
                    )
        if family.variance_split and declared.cells:
            for outcome in INTERVAL_OUTCOMES:
                entries.append(_variance_entry(context, declared, profile, outcome))
    return entries


# Density contrasts ---------------------------------------------------------------------------------------------


def _planned_spawns(context: Context, world: str) -> set[str]:
    rows = context.reads.rows("experiments", ["world_id", "spawn_id"])
    return {row["spawn_id"] for row in rows if row["world_id"] == world}


def _contrast_unit(context: Context, contrast: DensityContrast) -> tuple[Unit, list[str]]:
    """The contrast's unit and spawns: the planned spawn IDs, which must agree when both worlds were planned; a
    world without planned spawns is missing, not refused."""
    unknown = [w for w in (contrast.base, contrast.moving) if w not in context.table]
    if unknown:
        raise Refused(
            f"density contrast {contrast.id}: worlds {unknown} are not in the export's worlds table",
            column="world_id",
            stratum=contrast.id,
        )
    base, moving = (_planned_spawns(context, w) for w in (contrast.base, contrast.moving))
    if base and moving and base != moving:
        raise Refused(
            f"density contrast {contrast.id}: {contrast.base} and {contrast.moving} do not share their "
            f"spawn IDs ({sorted(base)} against {sorted(moving)})",
            column="spawn_id",
            stratum=contrast.id,
        )
    unit = _unit(context, "density_contrast", contrast.id, [contrast.base, contrast.moving])
    return unit, sorted(base | moving)


@dataclass(frozen=True, slots=True)
class ContrastCells:
    cells: dict[
        str, tuple[list[float], list[float]]
    ]  # usable spawn -> (base values, moving values)
    spawns_missing: dict[str, str]  # spawn -> why it is left out
    attempts: list[str]


def contrast_cells(
    context: Context,
    contrast: DensityContrast,
    spawns: Sequence[str],
    profile: str,
    outcome: str,
    population: str,
) -> ContrastCells:
    """Per spawn the two worlds' values; a spawn where either world has fewer than 2 values is left out of the
    estimate and both intervals and named (the missing-spawn rule, I6 §3.9)."""
    by_spawn: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for world in (contrast.base, contrast.moving):
        value = world_value(context, world, profile, outcome, population)
        for spawn in spawns:
            by_spawn[spawn][world] = [v for v in value.values if v["spawn_id"] == spawn]
    cells: dict[str, tuple[list[float], list[float]]] = {}
    missing: dict[str, str] = {}
    attempts: list[str] = []
    for spawn in spawns:
        short = [
            f"{world}: {len(by_spawn[spawn][world])} values"
            for world in (contrast.base, contrast.moving)
            if len(by_spawn[spawn][world]) < 2
        ]
        if short:
            missing[spawn] = "fewer than 2 values (" + "; ".join(short) + ")"
            continue
        base, moving = by_spawn[spawn][contrast.base], by_spawn[spawn][contrast.moving]
        cells[spawn] = ([v["value"] for v in base], [v["value"] for v in moving])
        attempts += [v["attempt_id"] for v in base + moving]
    return ContrastCells(cells, missing, sorted(attempts))


def _contrast_accounting(
    unit: Unit, profile: str
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, list[dict[str, Any]]]]]:
    """Final runs by report class per world and spawn, and the excluded runs with their outcome detail."""
    counts: dict[str, dict[str, dict[str, int]]] = {}
    excluded: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for world, group in unit.cells.items():
        for cell in group:
            for run in cell.finals:
                if run["profile_id"] != profile:
                    continue
                spawn = counts.setdefault(world, {}).setdefault(cell.spawn_id, {})
                spawn[run["report_class"]] = spawn.get(run["report_class"], 0) + 1
                if run["report_class"] in EXCLUDED:
                    excluded.setdefault(world, {}).setdefault(cell.spawn_id, []).append(
                        dict(
                            attempt_id=run["attempt_id"],
                            report_class=run["report_class"],
                            outcome=run["outcome"],
                            outcome_detail=run["outcome_detail"],
                        )
                    )
    return counts, excluded


def contrast_entries(context: Context, contrast: DensityContrast) -> list[dict[str, Any]]:
    unit, spawns = _contrast_unit(context, contrast)
    _homogeneous(context, unit)
    limits = {
        world: sorted({c.limit_s for c in group if math.isfinite(c.limit_s)})
        for world, group in unit.cells.items()
    }
    distinct = {limit for values in limits.values() for limit in values}
    entries = []
    for profile in unit.profiles:
        counts, excluded = _contrast_accounting(unit, profile)
        for population in context.populations(unit.ref.mandated):
            for outcome in CONTRAST_OUTCOMES:
                found = contrast_cells(context, contrast, spawns, profile, outcome, population)
                cells = list(found.cells.values())
                result = welch(cells, context.settings.alpha)
                exact, _ = permutation(cells, context.settings.alpha)
                notes = [_missing_note(found.spawns_missing, "spawns")]
                if not cells:
                    notes = ["not estimable: no spawn with 2 values in both worlds", notes[0]]
                if len(distinct) > 1:
                    notes.append(
                        f"the two worlds' L differ ({sorted(distinct)}): rho is not one ΔR/L"
                    )
                entries.append(
                    _entry(
                        "density_contrast",
                        unit.ref,
                        profile,
                        attempt_ids=found.attempts,
                        endpoint=outcome,
                        population=population,
                        comparison=f"{contrast.moving} − {contrast.base}",
                        base=contrast.base,
                        moving=contrast.moving,
                        estimate=result.estimate,
                        spawns_used=sorted(found.cells),
                        spawns_missing=found.spawns_missing,
                        cells={
                            s: dict(
                                base=dict(n=len(b), mean=float(np.mean(b)), values=b),
                                moving=dict(n=len(m), mean=float(np.mean(m)), values=m),
                            )
                            for s, (b, m) in found.cells.items()
                        },
                        limit_s=limits,
                        limits_differ=len(distinct) > 1,
                        interval=interval_document(result.interval),
                        permutation_interval=interval_document(exact),
                        note="; ".join(n for n in notes if n) or None,
                        repetitions=(
                            "unpaired: repetitions within a world and spawn are exchangeable, so "
                            "repetition k of one world is never matched with repetition k of the other"
                        ),
                        final_runs=counts,
                        excluded_runs=excluded,
                    )
                )
    return entries


# Hypotheses (O-4 (a)) ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Test:
    estimate: float | None = None
    interval: stats.Interval | None = None
    p_value: float | None = None
    method: str | None = None
    note: str | None = None
    attempts: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


def _contrast_test(context: Context, hypothesis: Hypothesis, contrast: DensityContrast) -> Test:
    method = "stratified Welch t test, Satterthwaite df (I6 §3.9)"
    outcome = HYPOTHESIS_OUTCOMES.get(hypothesis.endpoint)
    if outcome not in CONTRAST_OUTCOMES or hypothesis.factor is not None:
        return Test(method=method, note="a contrast hypothesis tests R_completion or C_h (at L)")
    _, spawns = _contrast_unit(context, contrast)
    found = contrast_cells(context, contrast, spawns, hypothesis.profile, outcome, "primary")
    cells = list(found.cells.values())
    result = welch(cells, context.settings.alpha)
    _, exact_p = permutation(cells, context.settings.alpha)
    return Test(
        result.estimate,
        result.interval,
        result.p_value,
        method,
        _missing_note(found.spawns_missing, "spawns"),
        tuple(found.attempts),
        dict(permutation_p_value=exact_p, permutation_label="sensitivity (I6 O-10 (b))"),
    )


def _factor_test(context: Context, hypothesis: Hypothesis, declared: Declared) -> Test:
    settings = context.settings
    method = f"{LENTH_NULL}; B = {settings.resamples}, seed {settings.seed}"
    method += f" (stream {stats.LENTH_STREAM})"
    names = dict(effect_names(declared.family))
    if hypothesis.factor not in names or HYPOTHESIS_OUTCOMES.get(hypothesis.endpoint) != "rho":
        return Test(
            method=method, note="a factor hypothesis tests a declared factor on R_completion"
        )
    means: dict[str, float] = {}
    attempts: list[str] = []
    for world in declared.cells:
        value = world_value(context, world, hypothesis.profile, "rho", "primary")
        attempts += value.attempt_ids
        if value.mean is not None:
            means[world] = value.mean
    effects = factorial_effects(declared.cells, means, declared.family)
    estimate = effects[hypothesis.factor]
    missing = sorted(w for w in declared.cells if w not in means)
    if missing:
        return Test(estimate, None, None, method, "not testable: no value in " + ", ".join(missing))
    if len(effects) < LENTH_MIN_EFFECTS:
        note = f"not testable: {len(effects)} effects; Lenth's method needs {LENTH_MIN_EFFECTS}"
        return Test(estimate, None, None, method, note)
    _, pse = lenth_pse([v for v in effects.values() if v is not None])
    if pse is None or estimate is None:
        return Test(estimate, None, None, method, "not testable: PSE is 0")
    p = lenth_p(abs(estimate) / pse, len(effects), resamples=settings.resamples, seed=settings.seed)
    return Test(estimate, None, p, method, None, tuple(attempts))


def _profile_test(context: Context, hypothesis: Hypothesis, declared: Declared) -> Test:
    method = "one-sample t test on world mean differences (I6 §3.4)"
    outcome = HYPOTHESIS_OUTCOMES.get(hypothesis.endpoint)
    unit = declared.unit
    if outcome is None or not unit.baseline or hypothesis.profile == unit.baseline:
        note = "a family hypothesis tests a profile against Default on R_completion, C_h or A_h"
        return Test(method=method, note=note)
    if not declared.family.interval:
        return Test(method=method, note="not testable: the family has no interval")
    entry = _profile_contrast(context, declared, hypothesis.profile, outcome, "primary", None)
    values = list(entry["world_differences"].values())
    p = one_sample_t_p(values)
    note = entry["note"] if p is not None else "not testable: fewer than 2 worlds or no variation"
    interval = world_t_interval(values, context.settings.alpha)
    return Test(entry["estimate"], interval, p, method, note, tuple(entry["attempt_ids"]))


def _hypothesis(
    context: Context, hypothesis: Hypothesis, declared: Mapping[str, Declared]
) -> dict[str, Any] | None:
    """The test of a hypothesis on a declared family or density contrast; None for any other hypothesis. The
    units' refusals (membership, factorial, homogeneity) ran before, in ``family_entries`` and
    ``contrast_entries``."""
    contrasts = {c.id: c for c in context.settings.density_contrasts}
    if hypothesis.world in contrasts:
        contrast = contrasts[hypothesis.world]
        unit, _ = _contrast_unit(context, contrast)
        test = _contrast_test(context, hypothesis, contrast)
    elif hypothesis.world in declared:
        unit = declared[hypothesis.world].unit
        test = (
            _factor_test(context, hypothesis, declared[hypothesis.world])
            if hypothesis.factor is not None
            else _profile_test(context, hypothesis, declared[hypothesis.world])
        )
    else:
        return None
    return _entry(
        "hypothesis",
        UnitRef(unit.ref.kind, unit.ref.id, unit.ref.robot),
        hypothesis.profile,
        attempt_ids=test.attempts,
        endpoint=hypothesis.endpoint,
        variant=hypothesis.id,
        role="confirmatory",
        hypothesis_id=hypothesis.id,
        hypothesis_endpoint=hypothesis.endpoint,
        factor=hypothesis.factor,
        comparison=None if hypothesis.factor else f"{hypothesis.profile} − Default",
        estimate=test.estimate,
        interval=interval_document(test.interval),
        p_value=test.p_value,
        p_method=test.method,
        testable=test.p_value is not None,
        note=test.note,
        **test.extra,
    )


def merge_hypotheses(entries: Sequence[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    """The study's entries with Holm over the plan's whole hypothesis list (RD-15): the stratum-level
    placeholders ``estimates.hypotheses`` writes for family and contrast hypotheses (untestable there) are
    replaced by this module's entries, and every hypothesis entry gets its adjusted p (an untestable one keeps
    its place with p = 1). The Holm list is the field ``family``; the unit an entry belongs to is
    ``world_family`` or ``density_contrast``."""
    units = {f.id for f in settings.families} | {c.id for c in settings.density_contrasts}
    ours = {h.id for h in settings.hypotheses if h.world in units}
    kept = [
        dict(e)
        for e in entries
        if not (
            e["estimand"] == "hypothesis"
            and e.get("hypothesis_id") in ours
            and e.get("contract") != CONTRACT
        )
    ]
    order = [h.id for h in settings.hypotheses]
    by_id = {e["hypothesis_id"]: e for e in kept if e["estimand"] == "hypothesis"}
    listed = [by_id[h] for h in order if h in by_id]
    if listed:
        adjusted = stats.holm(
            [e["p_value"] if e.get("p_value") is not None else 1.0 for e in listed]
        )
        for entry, value in zip(listed, adjusted, strict=True):
            entry["p_holm"] = value
            entry["family"] = order
    return kept


# All entries ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FamilyResult:
    entries: list[dict[str, Any]]  # stats.json entries, each with ``contract: family-sample-v1``
    # the export files read (worlds table, experiments, provenance.json, manifests), for the sidecar
    inputs: frozenset[str]


def family_estimates(study: Study, settings: Settings, reads: Reads | None = None) -> FamilyResult:
    """Every ``family-sample-v1`` entry of a study whose analysis block declares families or density contrasts
    (none otherwise), with the hypothesis entries of a confirmatory study; refusals per I6 §3.8."""
    if not settings.families and not settings.density_contrasts:
        return FamilyResult([], frozenset())
    reads = reads if reads is not None else Reads(study.export)
    table = worlds(reads)
    if table is None:
        raise Refused(
            "the analysis block declares families or density contrasts, but the export has no worlds "
            "table (export version 1)",
            column="worlds",
        )
    units = {f.id for f in settings.families} | {c.id for c in settings.density_contrasts}
    clashes = sorted(units & set(table))
    if clashes:
        raise Refused(
            f"family or contrast IDs {clashes} are also world IDs: hypotheses could not tell them apart",
            column="world_id",
        )
    context = _context(study, settings, reads, table)
    entries: list[dict[str, Any]] = []
    declared: dict[str, Declared] = {}
    for family in settings.families:
        declared[family.id] = _declared(context, family)
        entries += family_entries(context, declared[family.id])
    for contrast in settings.density_contrasts:
        entries += contrast_entries(context, contrast)
    if settings.confirmatory:
        for hypothesis in settings.hypotheses:
            found = _hypothesis(context, hypothesis, declared)
            if found is not None:
                entries.append(found)
    seen = [entry["id"] for entry in entries]
    if len(set(seen)) != len(seen):
        duplicate = next(key for key in seen if seen.count(key) > 1)
        raise RuntimeError(f"two family entries share the ID {duplicate}")
    return FamilyResult([_clean(e) for e in entries], frozenset(reads.files))


def _clean(value: Any) -> Any:
    """JSON-safe: non-finite numbers become null; tuples become lists; mapping proxies become dicts."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.floating | np.integer):
        return _clean(value.item())
    return value
