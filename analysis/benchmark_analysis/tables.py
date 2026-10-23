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

"""Tables T1–T8, T5H and S as CSV and booktabs LaTeX (RESULTS_OVERHAUL §7; FIGURE_STYLE §9; RD-2, RD-24).

Rows in a fixed order: strata as the study lists them, Default first, then the other profiles in the order of
their parameter hashes; n in every row; no bold. Values without an interval keep the measurement's resolution
(coverage 0.1 pp, times 0.1 s, distance 0.01 m, CPU 0.1 % of a core); an estimate with an interval is rounded so
the interval width has at most two significant digits, never finer than the resolution (FP-2). Median [Q1, Q3]
uses the Hyndman–Fan type 2 quantiles; mean (SD) only for never-censored metrics. p-values appear only in the
hypothesis table of a confirmatory study, Holm-adjusted, rounded up to one significant digit, floored at
"< 0.001". Limits that fall on a run censored at L read "> L"; an interval the data cannot give reads "—".
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass, field
import io
import math
from pathlib import Path
from typing import Any

from .data import EXCLUDED, MISSING, REPORTED, Stratum, Study
from .estimates import EFFECTS, PRIMARY_TIME, UNITS, Entries, EstimateContext, time_endpoints
from .settings import Settings

DASH = "—"
RESOLUTION = {"s": 1, "pp": 1, "%": 1, "m": 2, "MiB": 1, "ratio": 3, "count": 0}
DISCARD_TRIGGERS = ("campaign_stop", "experiment_stop", "experiment_restart")


@dataclass
class Table:
    name: str
    title: str
    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    attempt_ids: set[str] = field(default_factory=set)
    # its sheet may leave out count columns that are 0 in every row (T1)
    sparse_counts: bool = False

    def csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.columns)
        writer.writerows(self.rows)
        return buffer.getvalue()

    def tex(self) -> str:
        align = "l" * 2 + "r" * (len(self.columns) - 2)
        lines = [
            f"% {self.name}: {self.title}",
            f"\\begin{{tabular}}{{{align}}}",
            "\\toprule",
            " & ".join(_tex(c) for c in self.columns) + " \\\\",
            "\\midrule",
        ]
        previous = None
        for row in self.rows:
            if previous is not None and row[0] != previous:
                lines.append("\\addlinespace")
            previous = row[0]
            lines.append(" & ".join(_tex(cell) for cell in row) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines += [f"% {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


def _tex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("−", "$-$")
        .replace("–", "--")
        .replace(">", "$>$")
        .replace("<", "$<$")
        .replace("·", "$\\cdot$")
    )


# Number formatting (FP-2) ------------------------------------------------------------------------------------


def number(value: float | None, decimals: int) -> str:
    if value is None or not math.isfinite(value):
        return DASH
    text = f"{round(value, decimals):.{max(decimals, 0)}f}"
    if text.startswith("-") and float(text) != 0:
        return "−" + text[1:]
    return text.removeprefix("-")


def interval_decimals(lower: float | None, upper: float | None, unit: str) -> int:
    finest = RESOLUTION[unit]
    if lower is None or upper is None or upper <= lower:
        return finest
    return min(finest, 1 - math.floor(math.log10(upper - lower)))


def level_text(level: float) -> str:
    return f"{100 * level:.4g} %"


def with_interval(
    estimate: float | None, interval: Mapping[str, Any] | None, unit: str, requested_level: float
) -> str:
    """ "estimate [lower, upper]", with the level when below the requested one; "—" for a limit the data
    cannot give."""
    if estimate is None:
        return DASH
    if not interval or interval.get("level", 0) <= 0:
        return number(estimate, RESOLUTION[unit]) + (f" [{DASH}]" if interval else "")
    lower, upper = interval.get("lower"), interval.get("upper")
    decimals = interval_decimals(lower, upper, unit)
    low = "> L" if interval.get("lower_beyond_limit") else number(lower, decimals)
    high = "> L" if interval.get("upper_beyond_limit") else number(upper, decimals)
    text = f"{number(estimate, decimals)} [{low}, {high}]"
    if interval["level"] < requested_level - 1e-9:
        text += f" ({level_text(interval['level'])})"
    return text


def median_iqr(summary: Mapping[str, Any], unit: str) -> str:
    if summary.get("median") is None:
        return DASH
    d = RESOLUTION[unit]
    return (
        f"{number(summary['median'], d)} [{number(summary['q1'], d)}, {number(summary['q3'], d)}]"
    )


def median_range(summary: Mapping[str, Any], unit: str) -> str:
    if summary.get("median") is None:
        return DASH
    d = RESOLUTION[unit]
    return (
        f"{number(summary['median'], d)} [{number(summary['min'], d)}, {number(summary['max'], d)}]"
    )


def mean_sd(summary: Mapping[str, Any], unit: str) -> str:
    if summary.get("mean") is None:
        return DASH
    d = RESOLUTION[unit]
    return f"{number(summary['mean'], d)} ({number(summary['sd'], d)})"


def proportion(k: int, n: int, interval: Mapping[str, Any] | None) -> str:
    """ "k/n (p %, lower–upper %)" with a Wilson interval (FIGURE_STYLE §5: never 100 % without it)."""
    if not n or not interval:
        return f"{k}/{n}"
    low, high = 100 * interval["lower"], 100 * interval["upper"]
    return f"{k}/{n} ({100 * k / n:.0f} %, {low:.0f}–{high:.0f} %)"


def p_value(p: float | None) -> str:
    """Rounded up to one significant digit, floored at "< 0.001" (FIGURE_STYLE §9)."""
    if p is None:
        return DASH
    if p < 0.001:
        return "< 0.001"
    exponent = math.floor(math.log10(p))
    mantissa = math.ceil(p / 10**exponent - 1e-9)
    if mantissa == 10:
        mantissa, exponent = 1, exponent + 1
    return number(mantissa * 10**exponent, max(0, -exponent))


# Context -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Sources:
    study: Study
    settings: Settings
    entries: Entries
    order: list[str]  # profile IDs: Default first, then by parameter hash

    @property
    def level(self) -> float:
        return 1 - self.settings.alpha

    def profiles(self, stratum: Stratum) -> list[str]:
        present = set(stratum.profiles())
        return [p for p in self.order if p in present]

    def name(self, profile: str) -> str:
        return str(self.study.profiles.get(profile, {}).get("profile_name") or profile)

    def baseline(self, stratum: Stratum) -> str | None:
        return self.study.baselines.get(stratum.robot_id)

    def label(self, stratum: Stratum) -> str:
        return self.study.stratum_label(stratum, limit=True)


def profile_order(study: Study) -> list[str]:
    """Default first, then the other profiles by parameter hash (the colour order, FP-7)."""
    rows = sorted(study.profiles.items())
    order = [pid for pid, row in rows if row["is_baseline"]]
    return order + [
        pid
        for _, pid in sorted(
            (row["parameter_hash"], pid) for pid, row in rows if not row["is_baseline"]
        )
    ]


def sources(study: Study, settings: Settings, entries: Iterable[dict[str, Any]]) -> Sources:
    return Sources(study, settings, Entries(entries), profile_order(study))


# T1 accounting -----------------------------------------------------------------------------------------------


T1_COLUMNS = [
    "Stratum",
    "Profile",
    "Planned",
    "Done",
    "Completed",
    "Time limit",
    "No movement",
    "Missing (startup)",
    "Missing (platform)",
    "Runtime failure",
    "Stopped",
    "Replaced (timing)",
    "Replaced (interruption)",
    "Discarded (campaign stop)",
    "Discarded (experiment stop)",
    "Discarded (restart)",
    "Discarded (other)",
    "Unresolved",
    "Not done",
    "Completion (Wilson)",
]


def t1(src: Sources) -> Table:
    """Accounting per robot, world, spawn and profile of every planned experiment, counted from
    ``experiments`` and ``runs`` (not the export's own accounting table, so that the two can be compared)."""
    study = src.study
    experiments = study.export.table(
        "experiments", ["robot_id", "world_id", "spawn_id", "profile_id", "state"]
    ).to_pylist()
    counts: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in experiments:
        group = counts[(row["robot_id"], row["world_id"], row["spawn_id"], row["profile_id"])]
        group["planned"] += 1
        group[row["state"]] += 1
    for run in study.runs:
        group = counts[(run["robot_id"], run["world_id"], run["spawn_id"], run["profile_id"])]
        if run["outcome"] == "manual_discard":
            # a discard without a trigger is an experiment stop, as the engine reads it (exports.py)
            trigger = run["discard_trigger"] or "experiment_stop"
            group["discarded_" + (trigger if trigger in DISCARD_TRIGGERS else "other")] += 1
        elif run["replaced_reason"] in ("timing_rejection", "engine_interruption"):
            # replaced: a later attempt of the experiment exists (one still waiting is in Not done)
            group["replaced_" + run["replaced_reason"]] += 1
        elif run["is_final"]:
            group[run["report_class"]] += 1
            if run["report_class"] == REPORTED:
                group["outcome_" + run["outcome"]] += 1
    table = Table(
        "T1", "Outcome accounting per stratum and profile", list(T1_COLUMNS), sparse_counts=True
    )
    stratum_of = {(s.robot_id, s.world_id, s.spawn_id): s for s in study.strata}
    order = {profile: index for index, profile in enumerate(src.order)}
    for key in sorted(counts, key=lambda k: (k[:3], order.get(k[3], len(order)), k[3])):
        group, stratum = counts[key], stratum_of.get((key[0], key[1], key[2]))
        label = src.label(stratum) if stratum else study.label(key[0], key[1], key[2])
        success = src.entries.get("success", stratum, key[3]) if stratum else {}
        reported = group[REPORTED]
        table.rows.append(
            [
                label,
                src.name(key[3]),
                str(group["planned"]),
                str(group["done"]),
                *(str(group["outcome_" + o]) for o in ("completed", "time_limit", "no_movement")),
                *(str(group[c]) for c in (*MISSING, *EXCLUDED)),
                str(group["replaced_timing_rejection"]),
                str(group["replaced_engine_interruption"]),
                *(str(group["discarded_" + t]) for t in (*DISCARD_TRIGGERS, "other")),
                str(group["unresolved"]),
                str(group["planned"] - group["done"] - group["unresolved"]),
                proportion(success.get("k", 0), reported, success.get("interval")),
            ]
        )
        table.attempt_ids |= set(success.get("attempt_ids", []))
    table.attempt_ids |= {r["attempt_id"] for r in study.runs}
    table.notes = [
        "Completed counts final runs that completed within L; completion = completed / reported runs "
        f"(completed, time limit, no movement), Wilson {level_text(src.level)} interval.",
        "Missing observations and exclusions are final runs outside the reported population "
        "(reported-outcomes-v2); replaced and discarded attempts are not final. Every planned experiment has "
        "a row, also where no run was reported.",
    ]
    return table


# T2 setup and provenance -------------------------------------------------------------------------------------


def t2(src: Sources) -> Table:
    study = src.study
    provenance = study.export.provenance
    documents = provenance.get("documents") or {}
    used = sorted({r["provenance_id"] for r in study.finals if r["provenance_id"]})
    docs = [documents.get(i) or {} for i in used]
    table = Table("T2", "Setup and provenance", ["Item", "Value"])

    def add(item: str, values: Iterable[str]) -> None:
        distinct = sorted({v for v in values if v})
        table.rows.append([item, "; ".join(distinct) if distinct else DASH])

    add("Study", [f"{study.export.study_id} ({src.settings.evidence_level})"])
    add("Plan SHA-256", [str(study.export.study.get("plan_sha256") or "")[:12]])
    add("Robots", [s.robot_id for s in study.strata])
    for stratum in study.strata:
        blocks = len({r["block_id"] for r in stratum.finals})
        add(study.stratum_label(stratum), [f"L = {stratum.limit_s:g} s, {blocks} blocks"])
    for profile in src.order:
        row = study.profiles[profile]
        default = " (Default)" if row["is_baseline"] else ""
        add(
            f"Profile {row['profile_name']}",
            [f"parameter hash {row['parameter_hash'][:12]}{default}"],
        )
    add(
        "Runtime image",
        [
            str((d.get("runtime_image") or {}).get("id", "")).removeprefix("sha256:")[:12]
            for d in docs
        ],
    )
    add(
        "Suite image",
        [
            str((d.get("suite_image") or {}).get("id", "")).removeprefix("sha256:")[:12]
            for d in docs
        ],
    )
    repositories: dict[str, list[str]] = defaultdict(list)
    for document in docs:
        for role, state in (document.get("repositories") or {}).items():
            state = state or {}
            dirty = " (dirty)" if state.get("dirty") else ""
            repositories[role].append(f"{str(state.get('commit') or '')[:12]}{dirty}")
    for role, values in sorted(repositories.items()):
        add(f"Commit {role}", values)
    engines = [(d.get("host") or {}).get("engine") or {} for d in docs]
    dockers = [(d.get("host") or {}).get("docker") or {} for d in docs]
    add("CPU", [str(e.get("cpu_model") or "") for e in engines])
    add("CPU cores", [str(e.get("cpu_count") or "") for e in engines])
    add("Memory", [str(e.get("memory_total") or "") for e in engines])
    add(
        "Operating system",
        [
            f"{k.get('OperatingSystem', '')}, kernel {e.get('release', '')}"
            for k, e in zip(dockers, engines, strict=True)
        ],
    )
    middleware = [d.get("middleware") or {} for d in docs]
    add(
        "ROS 2 and RMW",
        [f"{m.get('ROS_DISTRO', '')}, {m.get('RMW_IMPLEMENTATION', '')}" for m in middleware],
    )
    add("Timing rule", [str(r["timing_rule"]) for r in study.finals])
    add(
        "Timing rules",
        ["attempts below RTF 0.90 for 10 wall s rejected; low_rtf below 0.95 kept and flagged"],
    )
    add(
        "Versions",
        [
            f"run record {study.export.study.get('run_record_version')}, dataset schema {r['dataset_schema']}, "
            f"coverage metric {r['coverage_metric_version']}, comparison key {r['comparison_key_version']}"
            for r in study.finals
        ],
    )
    table.attempt_ids = {r["attempt_id"] for r in study.finals}
    table.notes = ["Hashes and commits shortened to 12 characters; full values in provenance.json."]
    return table


# T3 censored times -------------------------------------------------------------------------------------------


def t3(src: Sources) -> Table:
    table = Table(
        "T3",
        "Restricted times: Kaplan–Meier median, RMST(L) and attainment by L",
        [
            "Stratum",
            "Profile",
            "Endpoint",
            "n",
            "Not reached by L",
            "KM median (s)",
            "RMST(L) (s)",
            "Attained by L",
        ],
    )
    for stratum in src.study.strata:
        for profile in src.profiles(stratum):
            for endpoint in time_endpoints(src.settings):
                e = src.entries.get("restricted_time", stratum, profile, endpoint=endpoint)
                if not e:
                    continue
                n = e.get("n", 0)
                median = e.get("km_median")
                median_text = (
                    with_interval(median, e.get("km_median_interval"), "s", src.level)
                    if median is not None
                    else "not reached"
                )
                attained = (e.get("attainment") or {}).get(repr(float(stratum.limit_s))) or {}
                table.rows.append(
                    [
                        src.label(stratum),
                        src.name(profile),
                        endpoint,
                        str(n),
                        f"{e.get('not_reached_by_limit', 0)}/{n}",
                        median_text,
                        with_interval(e.get("rmst"), e.get("rmst_interval"), "s", src.level),
                        proportion(attained.get("k", 0), n, attained.get("interval")),
                    ]
                )
                table.attempt_ids |= set(e.get("attempt_ids", []))
    table.notes = [
        "Times are simulated seconds since first movement, restricted to L; runs without movement never reach "
        "(RD-8). KM median by the midpoint rule with the order-statistic interval (Bin(n, 1/2)); a limit on a "
        "run censored at L reads > L. RMST(L) = mean of min(T, L): t interval below 10 runs, percentile "
        "bootstrap over runs from 10 (RD-5, RD-11).",
    ]
    return table


# T4 Default per stratum ----------------------------------------------------------------------------------------


def t4(src: Sources) -> Table:
    table = Table(
        "T4",
        "Default per stratum",
        ["Stratum", "Quantity", "n", "Median [Q1, Q3]", "Mean (SD) or estimate [CI]"],
    )
    for stratum in src.study.strata:
        baseline = src.baseline(stratum)
        if baseline is None or baseline not in stratum.profiles():
            continue
        label = src.label(stratum)
        horizons = EstimateContext(src.study, src.settings).horizons(stratum)
        for estimand, name in (("C_h", "C(h) (%)"), ("A_h", "A_h (%)"), ("D_h", "D(h) (m)")):
            unit = UNITS[estimand]
            for h in horizons:
                e = src.entries.get(estimand, stratum, baseline, horizon_s=h)
                table.rows.append(
                    [
                        label,
                        f"{name}, h = {h:g} s",
                        str(e.get("n", 0)),
                        median_iqr(e, unit),
                        mean_sd(e, unit),
                    ]
                )
                table.attempt_ids |= set(e.get("attempt_ids", []))
        for endpoint in PRIMARY_TIME:
            e = src.entries.get("restricted_time", stratum, baseline, endpoint=endpoint)
            rmst = with_interval(e.get("rmst"), e.get("rmst_interval"), "s", src.level)
            table.rows.append([label, f"RMST(L) of {endpoint} (s)", str(e.get("n", 0)), DASH, rmst])
        for estimand, unit, name in (
            ("gt_path_length_m", "m", "Ground-truth path length (m, moving runs)"),
            ("cpu_per_s", "%", "Explorer CPU (% of one core, sequential runs)"),
            ("rss_peak_mib", "MiB", "Explorer peak RSS (MiB, sequential runs)"),
        ):
            e = src.entries.get(estimand, stratum, baseline)
            table.rows.append(
                [label, name, str(e.get("n", 0)), median_iqr(e, unit), mean_sd(e, unit)]
            )
            table.attempt_ids |= set(e.get("attempt_ids", []))
    table.notes = [
        "C(h) and A_h hold a completed run's final coverage to L (carry-forward) and a run without movement at "
        "its pre-motion coverage; no value across a sample gap above 2 s. D(h): odometry, moving runs only.",
        "Quartiles: Hyndman–Fan type 2. Mean (SD) only for quantities never censored.",
    ]
    return table


# T5 paired effects ---------------------------------------------------------------------------------------------


def t5(src: Sources) -> Table:
    table = Table(
        "T5",
        "Paired differences, profile − Default",
        [
            "Stratum",
            "Profile",
            "Effect",
            "n pairs",
            "Censored in both",
            "Estimate [CI]",
            "Estimator",
            "Missing pairs",
        ],
    )
    for stratum in src.study.strata:
        baseline = src.baseline(stratum)
        for profile in src.profiles(stratum):
            if profile == baseline:
                continue
            for effect in EFFECTS:
                horizon = None if effect.endpoint else stratum.limit_s
                e = src.entries.get(
                    effect.estimand,
                    stratum,
                    profile,
                    endpoint=effect.endpoint,
                    horizon_s=horizon,
                    variant=effect.estimator,
                )
                if not e:
                    continue
                missing = (
                    "; ".join(f"{k} {v}" for k, v in (e.get("missing_pairs") or {}).items()) or DASH
                )
                table.rows.append(
                    [
                        src.label(stratum),
                        src.name(profile),
                        effect.label,
                        str(e.get("n_pairs", 0)),
                        str(e["pairs_censored_in_both"]) if "pairs_censored_in_both" in e else DASH,
                        with_interval(e.get("estimate"), e.get("interval"), effect.unit, src.level),
                        "mean (sign-flip)"
                        if effect.estimator == "mean"
                        else "Hodges–Lehmann (signed rank)",
                        missing,
                    ]
                )
                table.attempt_ids |= set(e.get("attempt_ids", []))
            split = src.entries.get("success_split", stratum, profile)
            if split:
                difference = split.get("difference")
                table.rows.append(
                    [
                        src.label(stratum),
                        src.name(profile),
                        "Completion n10 : n01",
                        str(split.get("n_pairs", 0)),
                        DASH,
                        f"{split.get('n10', 0)} : {split.get('n01', 0)}; difference "
                        + with_interval(
                            None if difference is None else 100 * difference,
                            _scaled(split.get("difference_interval"), 100),
                            "pp",
                            src.level,
                        ),
                        "Tango score (descriptive)",
                        DASH,
                    ]
                )
                table.attempt_ids |= set(split.get("attempt_ids", []))
    table.notes = [
        "Δ = profile − Default per block. Mean ΔR: exact sign-flip inversion below 10 pairs, block bootstrap "
        "(percentile) from 10; Hodges–Lehmann: inverted signed-rank test (midranks, Pratt). Pairs censored in "
        "both arms have Δ = 0. A level in parentheses is the highest the pairs allow. No p-values (RD-4).",
    ]
    return table


def _scaled(interval: Mapping[str, Any] | None, factor: float) -> dict[str, Any] | None:
    if not interval:
        return None
    return {
        **interval,
        "lower": None if interval.get("lower") is None else factor * interval["lower"],
        "upper": None if interval.get("upper") is None else factor * interval["upper"],
    }


def t5_hypotheses(src: Sources) -> Table | None:
    """Confirmatory studies only: the pre-registered tests with Holm over the plan's family (RD-24)."""
    if not src.settings.confirmatory:
        return None
    table = Table(
        "T5H",
        "Pre-registered hypotheses (Holm over the plan's family)",
        [
            "Hypothesis",
            "Stratum",
            "Profile",
            "Endpoint",
            "n pairs",
            "Estimate [CI, unadjusted]",
            "p",
            "p (Holm)",
            "Nonzero differences (floor)",
        ],
    )
    for hypothesis in src.settings.hypotheses:  # in the plan's order
        e = next(src.entries.where(estimand="hypothesis", hypothesis_id=hypothesis.id), {})
        endpoint = str(e.get("hypothesis_endpoint"))
        if not e.get("testable"):
            value = f"not testable: {e.get('note')} (p = 1 in the family)"
        elif endpoint == "success":
            value = f"{e.get('n10', 0)} : {e.get('n01', 0)} (n10 : n01)"
        elif e.get("scale") == "ΔR/L":
            value = (
                with_interval(e.get("estimate"), e.get("interval"), "ratio", src.level) + " (ΔR/L)"
            )
        else:
            unit = UNITS.get(endpoint, "s")
            value = with_interval(e.get("estimate"), e.get("interval"), unit, src.level)
        floor = e.get("floor") or {}
        where = "pooled" if e.get("world_id") == "pooled" else str(e.get("world_id"))
        if e.get("spawn_id"):
            where = src.study.label(e["robot_id"], e["world_id"], e["spawn_id"])
        table.rows.append(
            [
                e.get("hypothesis_id", ""),
                where,
                src.name(e.get("profile_id", "")),
                endpoint,
                str(e.get("n_pairs", DASH)),
                value,
                p_value(e.get("p_value")),
                p_value(e.get("p_holm")),
                f"{floor['nonzero']} ({floor['nonzero_needed']})" if floor else DASH,
            ]
        )
        table.attempt_ids |= set(e.get("attempt_ids", []))
    table.notes = [
        f"Intervals are unadjusted {level_text(src.level)} intervals (RD-15). Holm runs over every listed "
        "hypothesis; one that cannot be tested counts with p = 1. Floor: the fewest nonzero differences with which "
        "an exact test can reject within this family, 1 + log2(m/alpha) (RESULTS_RESEARCH §5.3).",
    ]
    return table


# T6 compute ----------------------------------------------------------------------------------------------------


def t6(src: Sources) -> Table:
    table = Table(
        "T6",
        "Compute cost (sequential runs with valid explorer samples)",
        [
            "Stratum",
            "Profile",
            "n",
            "Explorer CPU (% core)",
            "Peak RSS (MiB)",
            "Instrumentation CPU (% core)",
            "Insight CPU (% core)",
            "ΔCPU (HL)",
            "CPU ratio",
        ],
    )
    for stratum in src.study.strata:
        for profile in src.profiles(stratum):
            cpu = src.entries.get("cpu_per_s", stratum, profile)
            delta = src.entries.get("delta_cpu_per_s", stratum, profile, variant="hodges_lehmann")
            ratio = src.entries.get("delta_cpu_per_s", stratum, profile, variant="ratio")
            table.rows.append(
                [
                    src.label(stratum),
                    src.name(profile),
                    str(cpu.get("n", 0)),
                    median_iqr(cpu, "%"),
                    median_iqr(src.entries.get("rss_peak_mib", stratum, profile), "MiB"),
                    median_iqr(src.entries.get("instrumentation_cpu_per_s", stratum, profile), "%"),
                    median_iqr(src.entries.get("insight_cpu_per_s", stratum, profile), "%"),
                    with_interval(delta.get("estimate"), delta.get("interval"), "%", src.level)
                    if delta
                    else DASH,
                    with_interval(ratio.get("estimate"), ratio.get("interval"), "ratio", src.level)
                    if ratio
                    else DASH,
                ]
            )
            table.attempt_ids |= set(cpu.get("attempt_ids", []))
    table.notes = [
        "CPU seconds of each scope per simulated second from activation to the end, as % of one core; runs that "
        "overlapped another attempt or lack valid explorer samples are left out and counted in stats.json.",
    ]
    return table


# T7 reliability ------------------------------------------------------------------------------------------------


def t7(src: Sources) -> Table:
    table = Table(
        "T7",
        "Reliability",
        [
            "Stratum",
            "Profile",
            "n",
            "Final runs by class",
            "Goals sent",
            "Succeeded / sent",
            "Canceled",
            "Failed",
            "Recoveries",
            "Idle tail (s)",
            "Stalled",
        ],
    )
    for stratum in src.study.strata:
        for profile in src.profiles(stratum):
            e = src.entries.get("reliability", stratum, profile)
            goals = e.get("goals") or {}
            sent = goals.get("sent", 0)
            stalled = e.get("stalled")
            table.rows.append(
                [
                    src.label(stratum),
                    src.name(profile),
                    str(e.get("n", 0)),
                    "; ".join(f"{k} {v}" for k, v in (e.get("final_runs") or {}).items()),
                    str(sent)
                    + (
                        f" ({e['no_navigation_telemetry']} runs without telemetry)"
                        if e.get("no_navigation_telemetry")
                        else ""
                    ),
                    f"{goals.get('succeeded', 0)}/{sent}",
                    str(goals.get("canceled", 0)),
                    str(goals.get("failed", 0)),
                    median_iqr(src.entries.get("recoveries", stratum, profile), "count"),
                    median_iqr(src.entries.get("idle_tail_s", stratum, profile), "s"),
                    proportion(stalled["k"], stalled["n"], stalled.get("interval"))
                    if stalled
                    else "no window (described only)",
                ]
            )
            table.attempt_ids |= set(e.get("attempt_ids", []))
    table.notes = [
        "Goal counts summed over reported runs with navigation telemetry; idle tail = min(T_end, L) − last gain "
        f"(stall-v1, δ = {src.settings.stall_delta_pp:g} pp from the plan). Stalls are counted only with the "
        "plan's window W: time-limit runs whose idle tail reaches W, Wilson interval.",
    ]
    return table


# T8 timing validity --------------------------------------------------------------------------------------------


def t8(src: Sources) -> Table:
    table = Table(
        "T8",
        "Timing validity per execution mode",
        [
            "Stratum",
            "Profile",
            "Mode",
            "n",
            "RTF mean",
            "Worst 5-s RTF",
            "Wall s below 0.95",
            "low_rtf runs",
            "ΔRTF mean (HL)",
        ],
    )
    for stratum in src.study.strata:
        for profile in src.profiles(stratum):
            delta = src.entries.get("delta_rtf_mean", stratum, profile, variant="hodges_lehmann")
            first = True
            for mode in ("sequential", "parallel"):
                e = src.entries.get("timing", stratum, profile, population=mode)
                if not e or not e.get("n"):
                    continue
                metrics = e.get("metrics") or {}
                table.rows.append(
                    [
                        src.label(stratum),
                        src.name(profile),
                        mode,
                        str(e.get("n", 0)),
                        median_range(metrics.get("rtf_mean") or {}, "ratio"),
                        median_range(metrics.get("rtf_min_5s") or {}, "ratio"),
                        median_range(metrics.get("wall_s_below_095") or {}, "s"),
                        str(e.get("low_rtf", 0)),
                        with_interval(
                            delta.get("estimate"), delta.get("interval"), "ratio", src.level
                        )
                        if delta and first
                        else DASH,
                    ]
                )
                first = False
                table.attempt_ids |= set(e.get("attempt_ids", []))
    table.notes = ["Median [min, max]. ΔRTF over all paired reported runs, shown once per profile."]
    return table


# S sensitivity -------------------------------------------------------------------------------------------------


def sensitivity(src: Sources) -> Table:
    chosen = ("no_low_rtf", "exclusions_as_failures", "sequential_only")
    populations = ["primary", *(p for p in chosen if p in src.settings.sensitivity)]
    table = Table(
        "S",
        "Sensitivity of the paired estimates",
        ["Stratum", "Profile", "Effect", *(p.replace("_", " ") for p in populations)],
    )
    for stratum in src.study.strata:
        baseline = src.baseline(stratum)
        for profile in src.profiles(stratum):
            if profile == baseline:
                continue
            for effect in EFFECTS:
                horizon = None if effect.endpoint else stratum.limit_s
                cells = []
                for population in populations:
                    if (
                        population == "exclusions_as_failures"
                        and effect.estimand != "delta_restricted_time"
                    ):
                        cells.append("no value (RD-15)")
                        continue
                    e = src.entries.get(
                        effect.estimand,
                        stratum,
                        profile,
                        endpoint=effect.endpoint,
                        horizon_s=horizon,
                        population=population,
                        variant=effect.estimator,
                    )
                    estimate = with_interval(
                        e.get("estimate"), e.get("interval"), effect.unit, src.level
                    )
                    cells.append(f"{estimate} (n = {e.get('n_pairs', 0)})" if e else DASH)
                    table.attempt_ids |= set(e.get("attempt_ids", []))
                table.rows.append([src.label(stratum), src.name(profile), effect.label, *cells])
    if "shorter_horizons" in src.settings.sensitivity:
        for e in src.entries.where(estimand="restricted_time_at_horizon"):
            stratum = next(
                s
                for s in src.study.strata
                if (s.robot_id, s.world_id, s.spawn_id)
                == (e["robot_id"], e["world_id"], e["spawn_id"])
            )
            rmst = with_interval(e.get("rmst"), e.get("rmst_interval"), "s", src.level)
            table.rows.append(
                [
                    src.label(stratum),
                    src.name(e["profile_id"]),
                    f"RMST({e['horizon_s']:g} s) of {e['endpoint']}",
                    f"{rmst} (n = {e.get('n', 0)})",
                    *([DASH] * (len(populations) - 1)),
                ]
            )
            table.attempt_ids |= set(e.get("attempt_ids", []))
    table.notes = [
        "Populations: no low_rtf runs; runtime failures and stopped runs as never reaching (R = L), which gives "
        "C(h), A_h and D(h) no value; sequential runs only. Same estimators as T5. RMST(h) rows: the plan's "
        "horizons below L.",
    ]
    return table


def tables(src: Sources) -> list[Table]:
    built = [t1(src), t2(src), t3(src), t4(src), t5(src)]
    hypotheses = t5_hypotheses(src)
    if hypotheses is not None:
        built.append(hypotheses)
    return built + [t6(src), t7(src), t8(src), sensitivity(src)]


def write(table: Table, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{table.name}.csv", directory / f"{table.name}.tex"]
    paths[0].write_text(table.csv(), encoding="utf-8")
    paths[1].write_text(table.tex(), encoding="utf-8")
    return paths
