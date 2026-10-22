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

"""Every number of a study's analysis, one ``stats.json`` entry each (format ``stats-v1``; RESULTS_OVERHAUL
§7; METHODOLOGY §3-§4; RESULTS_RESEARCH §5.2).

Per stratum and profile: success with its Wilson interval; attainment and the Kaplan–Meier median of each
restricted time with the order-statistic interval; RMST(L) with a t interval below 10 runs and a percentile
bootstrap from 10 (RD-5, RD-11); C(h), A_h, D(h), compute, RTF and reliability as median [Q1, Q3]. Per stratum
and profile against Default ("profile − Default", blocks paired), one entry per estimator: the effects of
``EFFECTS`` with their primary estimator (RD-13: the mean ΔR with the sign-flip test for times,
Hodges–Lehmann with the signed-rank test for C(h), A_h, D(h)), Hodges–Lehmann as the secondary estimate of
times, compute and RTF, the geometric-mean ratio for compute, the discordant split of success. p-values exist
only for a confirmatory study's hypotheses, Holm-adjusted over the plan's whole list (RD-15). The sensitivity
populations repeat the paired effects. Every entry lists the attempts behind it.

An entry's ID (``entry_id``) joins estimand, endpoint, robot, world, spawn, profile, horizon, population and
variant (the estimator of a paired entry, or a hypothesis ID) with ``|``; IDs are unique.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from . import stats
from .data import (
    EXCLUDED,
    REPORTED,
    TIME_ENDPOINTS,
    Row,
    Stratum,
    Study,
    Value,
    coverage_at,
    coverage_auc,
    distance_at,
    idle_tail,
    restricted_time,
)
from .settings import Hypothesis, Settings

STATS_FORMAT = "stats-v1"
HORIZON_ESTIMANDS: dict[str, Callable[[Study, Row, float, float], Value]] = {
    "C_h": coverage_at,
    "A_h": coverage_auc,
    "D_h": distance_at,
}
POPULATIONS = ("primary", "no_low_rtf", "exclusions_as_failures", "sequential_only")
UNITS = {"C_h": "pp", "A_h": "pp", "D_h": "m", "cpu": "%", "success": "pp"}  # times: "s"


@dataclass(frozen=True, slots=True)
class Effect:
    """A paired effect shown in T5, S and F3, with its primary estimator (RD-13)."""

    name: str
    estimand: str
    endpoint: str | None  # restricted-time endpoint, or None for an effect at h = L
    label: str
    unit: str
    estimator: str  # "mean" or "hodges_lehmann"


EFFECTS = (
    Effect(
        "dR_completion", "delta_restricted_time", "R_completion", "ΔR_completion (s)", "s", "mean"
    ),
    Effect("dR_90", "delta_restricted_time", "R_90", "ΔR_90 (s)", "s", "mean"),
    Effect("dC_L", "delta_C_h", None, "ΔC(L) (pp)", "pp", "hodges_lehmann"),
    Effect("dA_L", "delta_A_h", None, "ΔA_L (pp)", "pp", "hodges_lehmann"),
    Effect("dD_L", "delta_D_h", None, "ΔD(L) (m)", "m", "hodges_lehmann"),
)
PRIMARY_TIME = tuple(e.endpoint for e in EFFECTS if e.endpoint)  # the sensitivity endpoints
# The primary estimator of each paired estimand (RD-13), from EFFECTS; restricted times also get the other one as
# the secondary estimate, and every effect at a plan horizon uses its estimand's estimator.
PRIMARY_ESTIMATOR = {e.estimand: e.estimator for e in EFFECTS}
if len(PRIMARY_ESTIMATOR) != len({(e.estimand, e.estimator) for e in EFFECTS}):
    raise RuntimeError("EFFECTS gives one estimand two primary estimators")
OTHER_ESTIMATOR = {"mean": "hodges_lehmann", "hodges_lehmann": "mean"}


def time_endpoints(settings: Settings) -> list[str]:
    """The restricted-time endpoints a study reports: completion, the plan's milestones and every milestone an
    effect of EFFECTS needs (ΔR_90 is always shown), in milestone order."""
    needed = {int(e.endpoint[2:]) for e in EFFECTS if e.endpoint and e.endpoint != "R_completion"}
    return ["R_completion", *(f"R_{m}" for m in sorted(set(settings.milestones) | needed))]


@dataclass(frozen=True, slots=True)
class EstimateContext:
    study: Study
    settings: Settings

    def horizons(self, stratum: Stratum) -> list[float]:
        """The plan's horizons of this world below L, then L itself."""
        declared = self.settings.horizons_s.get(stratum.world_id, ())
        return sorted({h for h in declared if 0 < h < stratum.limit_s} | {stratum.limit_s})

    def milestones(self) -> list[str]:
        return time_endpoints(self.settings)


def interval_document(interval: stats.Interval | None) -> dict[str, Any] | None:
    if interval is None:
        return None
    return dict(
        lower=interval.lower,
        upper=interval.upper,
        level=interval.level,
        method=interval.method,
        note=interval.note,
        lower_beyond_limit=interval.lower_beyond_limit,
        upper_beyond_limit=interval.upper_beyond_limit,
    )


def entry_id(
    estimand: str,
    stratum: Stratum | None,
    profile: str,
    endpoint: str | None = None,
    horizon_s: float | None = None,
    population: str = "primary",
    variant: str = "",
) -> str:
    """The stable key of a ``stats.json`` entry."""
    return "|".join(
        (
            estimand,
            endpoint or "",
            stratum.robot_id if stratum else "pooled",
            stratum.world_id if stratum else "pooled",
            stratum.spawn_id if stratum else "pooled",
            profile,
            "" if horizon_s is None else repr(float(horizon_s)),
            population,
            variant,
        )
    )


class Entries:
    """``stats.json`` entries by ID, the one lookup tables and figures use."""

    def __init__(self, entries: Iterable[dict[str, Any]]):
        self._by_id = {entry["id"]: entry for entry in entries}

    def get(
        self,
        estimand: str,
        stratum: Stratum | None,
        profile: str,
        *,
        endpoint: str | None = None,
        horizon_s: float | None = None,
        population: str = "primary",
        variant: str = "",
    ) -> dict[str, Any]:
        """The entry, or an empty mapping when the study has none (for example no Default in a stratum)."""
        key = entry_id(estimand, stratum, profile, endpoint, horizon_s, population, variant)
        return self._by_id.get(key, {})

    def where(self, **fields: Any) -> Iterator[dict[str, Any]]:
        """Entries whose fields equal all the given values, in ID order."""
        for key in sorted(self._by_id):
            entry = self._by_id[key]
            if all(entry.get(name) == value for name, value in fields.items()):
                yield entry


def _entry(
    estimand: str,
    stratum: Stratum | None,
    profile: str,
    *,
    attempt_ids: Iterable[str],
    population: str = "primary",
    endpoint: str | None = None,
    horizon_s: float | None = None,
    comparison: str | None = None,
    role: str = "descriptive",
    **values: Any,
) -> dict[str, Any]:
    variant = str(values.get("hypothesis_id") or values.get("estimator") or "")
    return dict(
        id=entry_id(estimand, stratum, profile, endpoint, horizon_s, population, variant),
        estimand=estimand,
        endpoint=endpoint,
        robot_id=stratum.robot_id if stratum else None,
        world_id=stratum.world_id if stratum else "pooled",
        spawn_id=stratum.spawn_id if stratum else None,
        comparison_key=stratum.comparison_key if stratum else None,
        limit_s=stratum.limit_s if stratum else None,
        profile_id=profile,
        comparison=comparison,
        horizon_s=horizon_s,
        population=population,
        role=role,
        attempt_ids=sorted(set(attempt_ids)),
        **values,
    )


# Per-run values in a population --------------------------------------------------------------------------------


def runs_for(stratum: Stratum, profile: str, population: str) -> list[Row]:
    """The runs of one profile in a population (S (a)–(c); METHODOLOGY §3 and RD-15)."""
    classes = (REPORTED, *EXCLUDED) if population == "exclusions_as_failures" else (REPORTED,)
    runs = stratum.runs_of(profile, classes)
    if population == "no_low_rtf":
        runs = [run for run in runs if run["low_rtf"] is not True]
    elif population == "sequential_only":
        runs = [run for run in runs if run["concurrent_wall_fraction"] == 0]
    return runs


def time_value(run: Row, endpoint: str, limit: float) -> Value:
    """R of a run; an excluded run counts as never reaching (R = L) in the exclusions-as-failures population."""
    if run["report_class"] in EXCLUDED:
        return Value(limit, "excluded_as_failure", False)
    return restricted_time(run, endpoint, limit)


def horizon_value(study: Study, run: Row, estimand: str, horizon: float, limit: float) -> Value:
    if run["report_class"] in EXCLUDED:
        return Value(None, "excluded_no_value")  # RD-15: no C(h), A_h or D(h) for failed runs
    return HORIZON_ESTIMANDS[estimand](study, run, horizon, limit)


def cpu_per_s(field: str) -> Callable[[Row], Value]:
    """CPU s of one ``resources`` scope per simulated s from activation to the end, as % of one core
    (sequential runs with valid explorer samples; RESULTS_OVERHAUL §7 F5/T6). The scope's CPU counts from its
    process start, so time before activation is in the numerator but not the denominator."""

    def read(run: Row) -> Value:
        if not run["resource_valid"] or run[field] is None:
            return Value(None, "resource_samples_missing")
        if run["concurrent_wall_fraction"] != 0:
            return Value(None, "parallel")
        if run["activation_to_movement_s"] is None or run["elapsed_s"] is None:
            return Value(None, "no_duration")
        span = run["activation_to_movement_s"] + run["elapsed_s"]
        if span <= 0:
            return Value(None, "no_duration")
        return Value(100 * run[field] / span, "observed")

    return read


def field_value(name: str, *, moving_only: bool = False) -> Callable[[Row], Value]:
    def read(run: Row) -> Value:
        if moving_only and run["outcome"] == "no_movement":
            return Value(None, "no_movement")  # RD-8: efficiency and paths of runs that moved
        value = run[name]
        return Value(None, "not_recorded") if value is None else Value(float(value), "observed")

    return read


def _sequential(read: Callable[[Row], Value]) -> Callable[[Row], Value]:
    def only(run: Row) -> Value:
        return read(run) if run["concurrent_wall_fraction"] == 0 else Value(None, "parallel")

    return only


METRICS: dict[str, Callable[[Row], Value]] = {
    "cpu_per_s": cpu_per_s("explorer_cpu_s"),
    "instrumentation_cpu_per_s": cpu_per_s("cpu_s_instrumentation"),
    "insight_cpu_per_s": cpu_per_s("cpu_s_insight"),
    "rss_peak_mib": _sequential(field_value("explorer_rss_peak_mib")),
    "gt_path_length_m": field_value("gt_path_length_m", moving_only=True),
    "rtf_mean": field_value("rtf_mean"),
    "rtf_min_5s": field_value("rtf_min_5s"),
    "wall_s_below_095": field_value("wall_s_below_095"),
    "recoveries": field_value("recoveries"),
}


# Descriptive entries -------------------------------------------------------------------------------------------


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return dict(median=None, q1=None, q3=None, min=None, max=None, mean=None, sd=None)
    q1, median, q3 = stats.quartiles(values)
    return dict(
        median=median,
        q1=q1,
        q3=q3,
        min=min(values),
        max=max(values),
        mean=float(np.mean(values)),
        sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
    )


def _reasons(values: Iterable[tuple[Row, Value]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for _, value in values:
        if value.value is None:
            counts[value.reason] += 1
    return dict(sorted(counts.items()))


def _with_values(
    stratum: Stratum, profile: str, name: str, found: list[tuple[Row, Value]], **extra: Any
) -> dict[str, Any]:
    values = [float(v.value) for _, v in found if v.value is not None]
    return _entry(
        name,
        stratum,
        profile,
        attempt_ids=(r["attempt_id"] for r, v in found if v.value is not None),
        n=len(values),
        no_value=_reasons(found),
        final_runs=stratum.counts(profile),
        **extra,
        **_summary(values),
    )


def _rmst_interval(times: list[float], settings: Settings) -> stats.Interval:
    """RMST(L) is the mean of R: a t interval below 10 runs, a percentile bootstrap over runs from 10."""
    if len(times) < stats.BOOTSTRAP_MIN_CLUSTERS:
        return stats.t_interval(times, settings.alpha)
    return stats.bootstrap_interval(
        [[t] for t in times],
        lambda x: float(x.mean()),
        resamples=settings.resamples,
        seed=settings.seed,
        alpha=settings.alpha,
    )


def _restricted(
    context: EstimateContext, stratum: Stratum, profile: str, endpoint: str
) -> dict[str, Any]:
    settings = context.settings
    runs = runs_for(stratum, profile, "primary")
    found = [(run, time_value(run, endpoint, stratum.limit_s)) for run in runs]
    usable = [(run, value) for run, value in found if value.value is not None]
    times = [float(value.value) for _, value in usable if value.value is not None]
    events = [bool(value.event) for _, value in usable]
    document: dict[str, Any] = dict(
        n=len(usable),
        n_events=sum(events),
        not_reached_by_limit=len(usable) - sum(events),
        no_movement=sum(1 for _, v in usable if v.reason == "no_movement"),
        completed_below_m=sum(1 for _, v in usable if v.reason == "completed_below_m"),
        no_value=_reasons(found),
        final_runs=stratum.counts(profile),
    )
    if usable:
        curve = stats.kaplan_meier(times, events)
        attainment = {}
        for h in context.horizons(stratum):
            k = sum(1 for t, e in zip(times, events, strict=True) if e and t <= h)
            attainment[repr(float(h))] = dict(
                horizon_s=h,
                k=k,
                interval=interval_document(stats.wilson(k, len(times), settings.alpha)),
            )
        document.update(
            km_median=stats.km_quantile(curve, 0.5),
            km_median_interval=interval_document(
                stats.median_interval(times, events, settings.alpha)
            ),
            km_quartiles=[stats.km_quantile(curve, p) for p in (0.25, 0.75)],
            rmst=stats.rmst(curve, stratum.limit_s),
            rmst_interval=interval_document(_rmst_interval(times, settings)),
            attainment=attainment,
        )
    return _entry(
        "restricted_time",
        stratum,
        profile,
        attempt_ids=(run["attempt_id"] for run, _ in usable),
        endpoint=endpoint,
        censoring="right-censored at L; no movement never reaches (RD-8, RD-10)",
        **document,
    )


def descriptive(context: EstimateContext, stratum: Stratum, profile: str) -> list[dict[str, Any]]:
    settings, study = context.settings, context.study
    runs = runs_for(stratum, profile, "primary")
    completed = sum(1 for run in runs if run["outcome"] == "completed" and run["completion_event"])
    entries = [
        _entry(
            "success",
            stratum,
            profile,
            attempt_ids=(run["attempt_id"] for run in runs),
            n=len(runs),
            k=completed,
            estimate=completed / len(runs) if runs else None,
            interval=interval_document(
                stats.wilson(completed, len(runs), settings.alpha) if runs else None
            ),
            final_runs=stratum.counts(profile),
        )
    ]
    entries += [
        _restricted(context, stratum, profile, endpoint) for endpoint in context.milestones()
    ]
    for estimand in HORIZON_ESTIMANDS:
        for h in context.horizons(stratum):
            found = [(run, horizon_value(study, run, estimand, h, stratum.limit_s)) for run in runs]
            entries.append(
                _with_values(
                    stratum,
                    profile,
                    estimand,
                    found,
                    horizon_s=h,
                    carried_forward=sum(1 for _, v in found if v.reason == "carried_forward"),
                    pre_motion=sum(1 for _, v in found if v.reason == "pre_motion"),
                )
            )
    for name, read in METRICS.items():
        entries.append(_with_values(stratum, profile, name, [(run, read(run)) for run in runs]))
    tails = [(run, idle_tail(study, run, stratum.limit_s, settings.stall_delta_pp)) for run in runs]
    entries.append(
        _with_values(stratum, profile, "idle_tail_s", tails, delta_pp=settings.stall_delta_pp)
    )
    entries.append(_reliability(context, stratum, profile, runs, tails))
    entries += _timing(stratum, profile, runs)
    return entries


TIMING = ("rtf_mean", "rtf_min_5s", "wall_s_below_095")


def _timing(stratum: Stratum, profile: str, runs: list[Row]) -> list[dict[str, Any]]:
    """T8 per execution mode: sequential runs (no other attempt running) and parallel ones."""
    entries = []
    for mode, members in (
        ("sequential", [r for r in runs if r["concurrent_wall_fraction"] == 0]),
        ("parallel", [r for r in runs if (r["concurrent_wall_fraction"] or 0) > 0]),
    ):
        summaries: dict[str, Any] = {
            name: _summary([float(r[name]) for r in members if r[name] is not None])
            for name in TIMING
        }
        entries.append(
            _entry(
                "timing",
                stratum,
                profile,
                attempt_ids=(r["attempt_id"] for r in members),
                population=mode,
                n=len(members),
                low_rtf=sum(1 for r in members if r["low_rtf"]),
                metrics=summaries,
            )
        )
    return entries


def _reliability(
    context: EstimateContext,
    stratum: Stratum,
    profile: str,
    runs: list[Row],
    tails: list[tuple[Row, Value]],
) -> dict[str, Any]:
    """T7: goal outcomes per goal sent (runs without navigation telemetry left out and counted), low-RTF runs
    and, with the plan's window W, time-limit runs whose idle tail reaches W (RD-18)."""
    alpha = context.settings.alpha
    totals: dict[str, int] = defaultdict(int)
    with_goals = [run for run in runs if run["goals_sent"] is not None]
    for run in with_goals:
        for name in ("sent", "succeeded", "canceled", "failed", "preempted", "open", "rejected"):
            totals[name] += run[f"goals_{name}"] or 0
    window = context.settings.stall_window_s
    stalled = None
    if window is not None:
        k = sum(
            1
            for run, tail in tails
            if run["outcome"] == "time_limit" and tail.value is not None and tail.value >= window
        )
        stalled = dict(
            window_s=window,
            delta_pp=context.settings.stall_delta_pp,
            k=k,
            n=len(runs),
            interval=interval_document(stats.wilson(k, len(runs), alpha) if runs else None),
        )
    return _entry(
        "reliability",
        stratum,
        profile,
        attempt_ids=(run["attempt_id"] for run in runs),
        n=len(runs),
        goals=dict(sorted(totals.items())),
        no_navigation_telemetry=len(runs) - len(with_goals),
        low_rtf=sum(1 for run in runs if run["low_rtf"]),
        stalled=stalled,
        final_runs=stratum.counts(profile),
    )


# Paired entries ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pairs:
    differences: list[float]
    profile_values: list[float]
    baseline_values: list[float]
    blocks: list[str]
    attempt_ids: list[str]
    missing: dict[str, int]


def pairs(
    stratum: Stratum,
    profile: str,
    baseline: str,
    population: str,
    read: Callable[[Row], Value],
) -> Pairs:
    """Δ = profile − Default per block with both values; the reason each other block has no pair."""
    allowed = {
        run["attempt_id"] for p in (profile, baseline) for run in runs_for(stratum, p, population)
    }
    differences, mine, theirs, blocks, attempts = [], [], [], [], []
    missing: dict[str, int] = defaultdict(int)
    for block, run, partner in stratum.blocks(profile, baseline):
        if run is None or run["attempt_id"] not in allowed:
            missing["profile_run_" + (run["report_class"] if run else "absent")] += 1
            continue
        if partner is None or partner["attempt_id"] not in allowed:
            missing["default_run_" + (partner["report_class"] if partner else "absent")] += 1
            continue
        a, b = read(run), read(partner)
        if a.value is None or b.value is None:
            missing["value_" + (a.reason if a.value is None else b.reason)] += 1
            continue
        differences.append(float(a.value) - float(b.value))
        mine.append(float(a.value))
        theirs.append(float(b.value))
        blocks.append(block)
        attempts += [run["attempt_id"], partner["attempt_id"]]
    return Pairs(differences, mine, theirs, blocks, sorted(attempts), dict(sorted(missing.items())))


def _paired_document(
    found: Pairs, method: str, settings: Settings, *, test: bool
) -> dict[str, Any]:
    """Estimate, interval and (only when ``test``) the matching p-value."""
    document: dict[str, Any] = dict(
        n_pairs=len(found.differences),
        missing_pairs=found.missing,
        both_equal=sum(1 for d in found.differences if d == 0),
        differences=found.differences,
        blocks=found.blocks,
        estimator=method,
    )
    if not found.differences:
        document.update(estimate=None, interval=None)
        if test:
            document["p_value"] = None
        return document
    seed = settings.seed
    if method == "mean":
        estimate, interval, p = stats.paired_mean(
            found.differences, alpha=settings.alpha, resamples=settings.resamples, seed=seed
        )
    elif method == "hodges_lehmann":
        estimate, interval, p = stats.paired_hodges_lehmann(found.differences, settings.alpha)
    else:  # geometric-mean ratio
        if any(v <= 0 for v in found.profile_values + found.baseline_values):
            document.update(estimate=None, interval=None, note="values not all positive")
            return document
        estimate, interval, p = stats.geometric_ratio(
            found.profile_values,
            found.baseline_values,
            alpha=settings.alpha,
            resamples=settings.resamples,
            seed=seed,
        )
    document.update(estimate=estimate, interval=interval_document(interval), seed=seed)
    if test:
        document["p_value"] = p
    return document


def _paired_entry(
    estimand: str,
    stratum: Stratum,
    profile: str,
    found: Pairs,
    method: str,
    settings: Settings,
    *,
    role: str,
    population: str = "primary",
    endpoint: str | None = None,
    horizon_s: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return _entry(
        estimand,
        stratum,
        profile,
        attempt_ids=found.attempt_ids,
        population=population,
        endpoint=endpoint,
        horizon_s=horizon_s,
        comparison=f"{profile} − Default",
        role=role,
        **extra,
        **_paired_document(found, method, settings, test=False),
    )


def paired(
    context: EstimateContext, stratum: Stratum, profile: str, population: str
) -> list[dict[str, Any]]:
    settings, study = context.settings, context.study
    baseline = study.baselines.get(stratum.robot_id)
    if baseline is None or profile == baseline:
        return []
    limit = stratum.limit_s
    entries = []
    endpoints = context.milestones() if population == "primary" else list(PRIMARY_TIME)
    for endpoint in endpoints:
        found = pairs(
            stratum, profile, baseline, population, lambda r: time_value(r, endpoint, limit)
        )
        censored = sum(
            1
            for a, b in zip(found.profile_values, found.baseline_values, strict=True)
            if a >= limit and b >= limit
        )
        primary = PRIMARY_ESTIMATOR["delta_restricted_time"]
        estimators = ((primary, "primary"), (OTHER_ESTIMATOR[primary], "secondary"))
        for method, role in estimators[: 2 if population == "primary" else 1]:
            entries.append(
                _paired_entry(
                    "delta_restricted_time",
                    stratum,
                    profile,
                    found,
                    method,
                    settings,
                    role=role,
                    population=population,
                    endpoint=endpoint,
                    pairs_censored_in_both=censored,
                )
            )
    horizons = context.horizons(stratum) if population == "primary" else [limit]
    for estimand in HORIZON_ESTIMANDS:
        for h in horizons:
            found = pairs(
                stratum,
                profile,
                baseline,
                population,
                lambda r: horizon_value(study, r, estimand, h, limit),
            )
            entries.append(
                _paired_entry(
                    f"delta_{estimand}",
                    stratum,
                    profile,
                    found,
                    PRIMARY_ESTIMATOR[f"delta_{estimand}"],
                    settings,
                    role="primary",
                    population=population,
                    horizon_s=h,
                )
            )
    if population == "primary":
        for name, methods in (
            ("cpu_per_s", ("hodges_lehmann", "ratio")),
            ("rss_peak_mib", ("hodges_lehmann",)),
            ("rtf_mean", ("hodges_lehmann",)),
        ):
            found = pairs(stratum, profile, baseline, population, METRICS[name])
            for method in methods:
                entries.append(
                    _paired_entry(
                        f"delta_{name}",
                        stratum,
                        profile,
                        found,
                        method,
                        settings,
                        role="primary" if method == "hodges_lehmann" else "secondary",
                    )
                )
        entries.append(_success_split(context, stratum, profile, baseline))
    return entries


def _success_counts(stratum: Stratum, profile: str, baseline: str) -> dict[str, Any]:
    """The 2 × 2 table of completion within L over blocks where both runs are reported."""
    counts = dict(n10=0, n01=0, both=0, neither=0)
    attempts: list[str] = []
    for _, run, partner in stratum.blocks(profile, baseline):
        if run is None or partner is None:
            continue
        if run["report_class"] != REPORTED or partner["report_class"] != REPORTED:
            continue
        a = run["outcome"] == "completed" and bool(run["completion_event"])
        b = partner["outcome"] == "completed" and bool(partner["completion_event"])
        key = "both" if a and b else "n10" if a else "n01" if b else "neither"
        counts[key] += 1
        attempts += [run["attempt_id"], partner["attempt_id"]]
    return dict(counts, n_pairs=sum(counts.values()), attempt_ids=attempts)


def _success_split(
    context: EstimateContext, stratum: Stratum, profile: str, baseline: str
) -> dict[str, Any]:
    """Paired success (RD-52, RD-53): the discordant split with its exact interval, and the difference in
    rates with Tango's interval (descriptive)."""
    alpha = context.settings.alpha
    counts = _success_counts(stratum, profile, baseline)
    attempts = counts.pop("attempt_ids")
    n, n10, n01 = counts["n_pairs"], counts["n10"], counts["n01"]
    return _entry(
        "success_split",
        stratum,
        profile,
        attempt_ids=attempts,
        comparison=f"{profile} − Default",
        role="primary",
        **counts,
        split_interval=interval_document(
            stats.clopper_pearson(n10, n10 + n01, alpha) if n10 + n01 else None
        ),
        difference=(n10 - n01) / n if n else None,
        difference_interval=interval_document(stats.tango(n10, n01, n, alpha) if n else None),
        difference_label="descriptive (RD-53)",
    )


# Hypotheses ----------------------------------------------------------------------------------------------------


def _hypothesis_strata(context: EstimateContext, hypothesis: Hypothesis) -> list[Stratum]:
    """The strata a hypothesis addresses: its profile's robot, in its world (every spawn of that world)."""
    robot = context.study.profiles.get(hypothesis.profile, {}).get("robot_id")
    return [
        s
        for s in context.study.strata
        if s.robot_id == robot and (hypothesis.world == "pooled" or s.world_id == hypothesis.world)
    ]


def _hypothesis_read(
    context: EstimateContext, hypothesis: Hypothesis, stratum: Stratum
) -> tuple[Callable[[Row], Value], str] | str:
    """How a hypothesis's endpoint is read from a run and its estimator, or why it cannot be tested."""
    endpoint, limit = hypothesis.endpoint, stratum.limit_s
    if endpoint in TIME_ENDPOINTS:
        return (lambda r: time_value(r, endpoint, limit)), PRIMARY_ESTIMATOR[
            "delta_restricted_time"
        ]
    if endpoint in HORIZON_ESTIMANDS:
        horizon = float(hypothesis.horizon_s or limit)
        if horizon > limit:
            return f"horizon {horizon:g} s beyond L = {limit:g} s"
        estimator = PRIMARY_ESTIMATOR[f"delta_{endpoint}"]
        return (lambda r: horizon_value(context.study, r, endpoint, horizon, limit)), estimator
    if endpoint == "cpu":
        return METRICS["cpu_per_s"], "hodges_lehmann"
    return f"endpoint {endpoint} cannot be tested as a paired difference"


def _merged(found: Iterable[Pairs]) -> Pairs:
    """The pairs of several spawns of one world as one sample of blocks."""
    parts = list(found)
    missing: dict[str, int] = defaultdict(int)
    for part in parts:
        for key, value in part.missing.items():
            missing[key] += value
    return Pairs(
        [d for p in parts for d in p.differences],
        [v for p in parts for v in p.profile_values],
        [v for p in parts for v in p.baseline_values],
        [b for p in parts for b in p.blocks],
        sorted(a for p in parts for a in p.attempt_ids),
        dict(sorted(missing.items())),
    )


def _located(entry: dict[str, Any], hypothesis: Hypothesis, robot: str | None) -> dict[str, Any]:
    """A hypothesis entry not tied to one stratum still names its world and robot."""
    if entry["spawn_id"] is None:
        entry.update(world_id=hypothesis.world, robot_id=robot)
    return entry


def _untestable(
    hypothesis: Hypothesis, base: dict[str, Any], note: str, attempts: Iterable[str] = ()
) -> dict[str, Any]:
    return _entry(
        "hypothesis",
        None,
        hypothesis.profile,
        attempt_ids=attempts,
        endpoint=hypothesis.endpoint,
        horizon_s=hypothesis.horizon_s,
        **base,
        estimate=None,
        p_value=None,
        testable=False,
        note=note,
    )


def _hypothesis_entry(context: EstimateContext, hypothesis: Hypothesis) -> dict[str, Any]:
    """The pre-registered test of one hypothesis: the paired estimate over the world's blocks (all spawns)
    with its p-value, or the pooled equal-weight row (RD-16)."""
    settings = context.settings
    base: dict[str, Any] = dict(
        hypothesis_id=hypothesis.id, hypothesis_endpoint=hypothesis.endpoint, role="confirmatory"
    )
    strata = _hypothesis_strata(context, hypothesis)
    if not strata:
        return _untestable(
            hypothesis, base, f"no stratum of world {hypothesis.world} for the profile's robot"
        )
    if hypothesis.world == "pooled":
        return _pooled(context, hypothesis, strata, base)
    baseline = context.study.baselines.get(strata[0].robot_id, "")
    note = f"{len(strata)} spawns pooled as one sample of blocks" if len(strata) > 1 else None
    if hypothesis.endpoint == "success":
        totals: dict[str, Any] = defaultdict(int)
        attempts: list[str] = []
        for stratum in strata:
            counts = _success_counts(stratum, hypothesis.profile, baseline)
            attempts += counts.pop("attempt_ids")
            for key, value in counts.items():
                totals[key] += value
        discordant = (totals["n10"], totals["n01"])
        return _entry(
            "hypothesis",
            strata[0] if len(strata) == 1 else None,
            hypothesis.profile,
            attempt_ids=attempts,
            endpoint="success",
            comparison=f"{hypothesis.profile} − Default",
            **base,
            **totals,
            testable=totals["n_pairs"] > 0,
            note=note if totals["n_pairs"] else "no block with both runs reported",
            split_interval=interval_document(
                stats.clopper_pearson(discordant[0], sum(discordant), settings.alpha)
                if sum(discordant)
                else None
            ),
            floor=dict(
                nonzero_needed=stats.exact_floor(len(settings.hypotheses), settings.alpha),
                nonzero=sum(discordant),
            ),
            p_value=stats.mcnemar_exact(*discordant) if totals["n_pairs"] else None,
            p_method="exact conditional McNemar on the discordant blocks (RD-52)",
        )
    reads = [_hypothesis_read(context, hypothesis, stratum) for stratum in strata]
    refusals = [r for r in reads if isinstance(r, str)]
    if refusals:
        return _untestable(hypothesis, base, refusals[0])
    readers = [r for r in reads if not isinstance(r, str)]
    method = readers[0][1]
    found = _merged(
        pairs(stratum, hypothesis.profile, baseline, "primary", read)
        for stratum, (read, _) in zip(strata, readers, strict=True)
    )
    document = _paired_document(found, method, settings, test=True)
    document["floor"] = dict(
        nonzero_needed=stats.exact_floor(len(settings.hypotheses), settings.alpha),
        nonzero=sum(1 for d in found.differences if d != 0),
    )
    return _entry(
        "hypothesis",
        strata[0] if len(strata) == 1 else None,
        hypothesis.profile,
        attempt_ids=found.attempt_ids,
        endpoint=hypothesis.endpoint,
        horizon_s=hypothesis.horizon_s,
        comparison=f"{hypothesis.profile} − Default",
        testable=document.get("p_value") is not None,
        note=note,
        **{**document, **base},
    )


def _pooled(
    context: EstimateContext, hypothesis: Hypothesis, strata: list[Stratum], base: dict[str, Any]
) -> dict[str, Any]:
    """RD-16: the equal-weight mean over worlds of each world's mean difference on a scale-free endpoint (ΔR/L,
    ΔC(h), ΔA_h; a world's spawns weighted equally within it), only when every world has at least 10 blocks;
    blocks resampled within each spawn, and the matching test flips the sign of every block."""
    settings = context.settings
    if hypothesis.endpoint not in (*TIME_ENDPOINTS, "C_h", "A_h"):
        return _untestable(
            hypothesis, base, "pooling needs a scale-free endpoint (ΔR/L, ΔC(h), ΔA_h; RD-16)"
        )
    baseline = context.study.baselines.get(strata[0].robot_id, "")
    # world -> per spawn, scaled differences
    worlds: dict[str, list[list[float]]] = defaultdict(list)
    attempts: list[str] = []
    for stratum in strata:
        read = _hypothesis_read(context, hypothesis, stratum)
        if isinstance(read, str):
            return _untestable(hypothesis, base, read)
        found = pairs(stratum, hypothesis.profile, baseline, "primary", read[0])
        scale = stratum.limit_s if hypothesis.endpoint in TIME_ENDPOINTS else 1.0
        worlds[stratum.world_id].append([d / scale for d in found.differences])
        attempts += found.attempt_ids
    blocks = {world: sum(len(d) for d in spawns) for world, spawns in worlds.items()}
    if min(blocks.values()) < stats.BOOTSTRAP_MIN_CLUSTERS or any(
        not d for s in worlds.values() for d in s
    ):
        return _untestable(
            hypothesis,
            base,
            f"a pooled row needs at least {stats.BOOTSTRAP_MIN_CLUSTERS} blocks in every world and pairs in "
            f"every spawn (blocks per world: {dict(sorted(blocks.items()))})",
            attempts,
        )
    spawns = [(world, np.asarray(d)) for world, parts in sorted(worlds.items()) for d in parts]

    def statistic(samples: list[tuple[str, np.ndarray[Any, Any]]]) -> float:
        per_world: dict[str, list[float]] = defaultdict(list)
        for world, values in samples:
            per_world[world].append(float(values.mean()))
        return float(np.mean([np.mean(v) for v in per_world.values()]))

    estimate = statistic(spawns)
    rng = stats.random_stream(settings.seed, stats.POOLED_BOOTSTRAP_STREAM)
    replicates = np.array(
        [
            statistic([(w, rng.choice(d, size=d.size, replace=True)) for w, d in spawns])
            for _ in range(settings.resamples)
        ]
    )
    lower, upper = np.quantile(replicates, [settings.alpha / 2, 1 - settings.alpha / 2])
    flips = stats.random_stream(settings.seed, stats.POOLED_SIGN_FLIP_STREAM)
    extreme = sum(
        abs(statistic([(w, d * flips.choice((-1.0, 1.0), size=d.size)) for w, d in spawns]))
        >= abs(estimate) - stats.TOLERANCE
        for _ in range(settings.resamples)
    )
    return _entry(
        "hypothesis",
        None,
        hypothesis.profile,
        attempt_ids=attempts,
        endpoint=hypothesis.endpoint,
        horizon_s=hypothesis.horizon_s,
        **base,
        estimate=estimate,
        scale="ΔR/L" if hypothesis.endpoint in TIME_ENDPOINTS else "pp",
        interval=dict(
            lower=float(lower),
            upper=float(upper),
            level=1 - settings.alpha,
            method=f"stratified block bootstrap, percentile, B = {settings.resamples}",
            note="a limit is the most extreme resample"
            if lower <= replicates.min() or upper >= replicates.max()
            else None,
            lower_beyond_limit=False,
            upper_beyond_limit=False,
        ),
        p_value=(1 + extreme) / (settings.resamples + 1),
        p_method=f"stratified sign-flip, Monte Carlo, B = {settings.resamples}, seed {settings.seed}",
        n_pairs=sum(blocks.values()),
        blocks_per_world=dict(sorted(blocks.items())),
        testable=True,
        seed=settings.seed,
    )


def hypotheses(context: EstimateContext) -> list[dict[str, Any]]:
    """Confirmatory studies only (RD-4, RD-24): every listed hypothesis, Holm over the whole list (RD-15). A
    hypothesis that cannot be tested keeps its place in the family with p = 1, so the others are never
    adjusted over a smaller family than the plan declared."""
    if not context.settings.confirmatory:
        return []
    profiles = context.study.profiles
    entries = [
        _located(_hypothesis_entry(context, h), h, profiles.get(h.profile, {}).get("robot_id"))
        for h in context.settings.hypotheses
    ]
    family = [e["p_value"] if e.get("p_value") is not None else 1.0 for e in entries]
    for entry, adjusted in zip(entries, stats.holm(family) if family else [], strict=True):
        entry["p_holm"] = adjusted
        entry["family"] = [h.id for h in context.settings.hypotheses]
    return entries


# All entries ---------------------------------------------------------------------------------------------------


def estimates(study: Study, settings: Settings) -> list[dict[str, Any]]:
    context = EstimateContext(study, settings)
    entries: list[dict[str, Any]] = []
    sensitivity = [p for p in POPULATIONS[1:] if p in settings.sensitivity]
    for stratum in study.strata:
        for profile in stratum.profiles():
            entries += descriptive(context, stratum, profile)
            entries += paired(context, stratum, profile, "primary")
            for population in sensitivity:
                entries += paired(context, stratum, profile, population)
            if "shorter_horizons" in settings.sensitivity:
                entries += _shorter(context, stratum, profile)
    entries += hypotheses(context)
    seen = [entry["id"] for entry in entries]
    if len(set(seen)) != len(seen):
        duplicate = next(key for key in seen if seen.count(key) > 1)
        raise RuntimeError(f"two stats.json entries share the ID {duplicate}")
    return [_clean(entry) for entry in entries]


def _shorter(context: EstimateContext, stratum: Stratum, profile: str) -> list[dict[str, Any]]:
    """S (d): restricted means and attainment at the plan's horizons below L (RMST(h) = mean of min(T, h))."""
    alpha = context.settings.alpha
    runs = runs_for(stratum, profile, "primary")
    out = []
    for h in context.horizons(stratum)[:-1]:
        for endpoint in PRIMARY_TIME:
            found = [(run, time_value(run, endpoint, stratum.limit_s)) for run in runs]
            usable = [(run, v) for run, v in found if v.value is not None]
            times = [min(float(v.value), h) for _, v in usable if v.value is not None]
            reached = sum(1 for _, v in usable if v.event and v.value is not None and v.value <= h)
            out.append(
                _entry(
                    "restricted_time_at_horizon",
                    stratum,
                    profile,
                    attempt_ids=(run["attempt_id"] for run, _ in usable),
                    population="shorter_horizons",
                    endpoint=endpoint,
                    horizon_s=h,
                    n=len(times),
                    rmst=float(np.mean(times)) if times else None,
                    rmst_interval=interval_document(
                        stats.t_interval(times, alpha) if times else None
                    ),
                    k=reached,
                    attainment_interval=interval_document(
                        stats.wilson(reached, len(times), alpha) if times else None
                    ),
                )
            )
    return out


def _clean(value: Any) -> Any:
    """JSON-safe: non-finite numbers become null (never written as NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value
