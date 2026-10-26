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

"""Tables T1–T11, T5H and S as CSV and booktabs LaTeX (RESULTS_OVERHAUL §7; FIGURE_STYLE §9; RD-2, RD-24).

Rows in a fixed order: strata as the study lists them, Default first, then the other profiles in the order of
their parameter hashes; n in every row; no bold. Values without an interval keep the measurement's resolution
(coverage 0.1 pp, times 0.1 s, distance 0.01 m, CPU 0.1 % of a core); an estimate with an interval is rounded so
the interval width has at most two significant digits, never finer than the resolution (FP-2). Median [Q1, Q3]
uses the Hyndman–Fan type 2 quantiles; mean (SD) only for never-censored metrics. p-values appear only in the
hypothesis table of a confirmatory study, Holm-adjusted, rounded up to one significant digit, floored at
"< 0.001". Limits that fall on a run censored at L read "> L"; an interval the data cannot give reads "—".

World groups (plans/worlds/I8 §3.1): when the export holds the ``worlds`` table, T3–T7 and S order their rows
by ``worlds.world_groups`` (group order, then the registry order within a group, then as the study lists them),
also with one group; with more than one group they gain a first column ``Group`` (the group's key, its title in
a note and in the sidecar's ``groups``), which the LaTeX and the sheet turn into a heading row per group. T1, T2,
T8 and T9 keep the study order.

Families (``family-sample-v1``, plans/worlds/I6, I8 §3.3): T10 the family estimates (world means, family
average, profile contrasts, accounting) and T11 the factorial effects and descriptive pairs, both headed by
family and written only when ``stats.json`` holds family entries; they read those entries and compute nothing.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass, field
import io
import math
from pathlib import Path
import re
from typing import Any

from .data import (
    COMPLETION_SOURCE,
    COMPLETION_SOURCES,
    COVERAGE_STOP_TEXT,
    EXCLUDED,
    MISSING,
    REPORTED,
    STUCK,
    STUCK_TEXT,
    Stratum,
    Study,
    mapping_condition,
    outcome_rule,
)
from .estimates import EFFECTS, PRIMARY_TIME, UNITS, Entries, EstimateContext, time_endpoints
from .families import CONTRACT as FAMILY_CONTRACT
from .settings import Family, Settings
from .worlds import GROUP_ORDER, GROUP_TITLES, WorldGroup, world_groups

DASH = "—"
RESOLUTION = {"s": 1, "pp": 1, "%": 1, "m": 2, "MiB": 1, "ratio": 3, "count": 0}
DISCARD_TRIGGERS = ("campaign_stop", "experiment_stop", "experiment_restart")
# reported runs a coding file does not name (T9, F23); no category ID holds a space
NOT_CODED = "not coded"
REPORTED_RUNS = "reported runs"  # T9's denominator row
# how the coding was done, in the captions of F23 and T9: the coding file's ``blinded`` (the coder's statement)
CODED_BLIND = (
    "Coded blind to profile and outcome from the numbered sheet (the coding file's blinded: true)."
)
CODED_UNBLINDED = "Coder not blinded to profile and outcome (the coding file's blinded: false)."
CODER_NOTE = "One coder: no reliability estimate."
CATEGORY_NOTE = (
    "Category names are the plan's codes, not measurements: a name such as stalled is not the stall-v1 "
    "measurement (RD-18: stalls are described, never classified)."
)
WRAP_TEX = "0.45\\linewidth"  # the LaTeX width of a wrapping column


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
    # a column whose cells wrap: on the sheet at sheets.WRAP_PT, in LaTeX as a p{} column (T9's definitions)
    wrap_column: int | None = None
    # further fields of the table's sidecar (T9: the coding file's record)
    extra: dict[str, Any] = field(default_factory=dict)
    # a grouped table (``with_groups``, T10, T11): column 0 holds the group's key, these its titles (headings)
    group_titles: dict[str, str] = field(default_factory=dict)
    # export files the table read beyond its INPUTS (T10, T11: the family estimates' inputs), hashed in its sidecar
    inputs: set[str] = field(default_factory=set)

    def csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.columns)
        writer.writerows(self.rows)
        return buffer.getvalue()

    def tex(self) -> str:
        # a grouped table's group column becomes a heading row per group
        skip = 1 if self.group_titles else 0
        columns = self.columns[skip:]
        align = "l" * 2 + "r" * (len(columns) - 2)
        if self.wrap_column is not None:
            align = "".join(
                f"p{{{WRAP_TEX}}}" if index == self.wrap_column - skip else letter
                for index, letter in enumerate(align)
            )
        lines = [
            f"% {self.name}: {self.title}",
            f"\\begin{{tabular}}{{{align}}}",
            "\\toprule",
            " & ".join(_tex(c) for c in columns) + " \\\\",
            "\\midrule",
        ]
        previous, group = None, None
        for row in self.rows:
            if skip and row[0] != group:
                if group is not None:
                    lines.append("\\midrule")
                group = row[0]
                title = _tex(self.group_titles.get(group, group))
                lines.append(f"\\multicolumn{{{len(columns)}}}{{l}}{{\\emph{{{title}}}}} \\\\")
                previous = None
            if previous is not None and row[skip] != previous:
                lines.append("\\addlinespace")
            previous = row[skip]
            lines.append(" & ".join(_tex(cell) for cell in row[skip:]) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        lines += [f"% {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


# one character to its LaTeX text, in one pass so that no replacement is escaped again: the specials of free
# text (T9's definitions) and the package's own signs
TEX = str.maketrans(
    {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "_": "\\_",
        "#": "\\#",
        "$": "\\$",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
        "−": "$-$",
        "–": "--",
        ">": "$>$",
        "<": "$<$",
        "·": "$\\cdot$",
    }
)


def _tex(text: str) -> str:
    return text.translate(TEX)


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
    # the coding file's attempt ID -> category and its record for the sidecar (T9), when one was given
    codes: Mapping[str, str] | None = None
    coding_record: Mapping[str, Any] | None = None
    groups: tuple[WorldGroup, ...] = ()  # ``worlds.world_groups``
    # the export files the family estimates read (``families.FamilyResult.inputs``): T10's and T11's inputs
    family_inputs: frozenset[str] = frozenset()

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

    def with_groups(self, table: Table) -> Table:
        """T3–T7 and S when the export lists worlds: the rows sorted stably by group and by the stratum's place
        in it (a stratum's rows keep their order; every row starts with its stratum's label, ``Sources.label``,
        and a row whose label is not a stratum's goes last); with more than one group also a first column
        ``Group`` (the key), its titles in a note and the sidecar's ``groups``. Unchanged without the table."""
        if not self.groups or self.groups[0].key is None:
            return table
        last = (len(self.groups), 0, "")
        place = {
            self.label(stratum): (index, position, group.key or "")
            for index, group in enumerate(self.groups)
            for position, stratum in enumerate(group.strata)
        }
        rows = sorted(table.rows, key=lambda row: place.get(row[0], last)[:2])
        if len(self.groups) < 2:
            table.rows = rows
            return table
        titles = {group.key or "": group.title for group in self.groups}
        table.columns = ["Group", *table.columns]
        table.rows = [[place.get(row[0], last)[2], *row] for row in rows]
        if table.wrap_column is not None:
            table.wrap_column += 1
        table.group_titles = titles
        table.notes.append(
            "Groups (never pooled): " + "; ".join(f"{k}, {t}" for k, t in titles.items()) + "."
        )
        table.extra["groups"] = titles
        return table


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


def sources(
    study: Study,
    settings: Settings,
    entries: Iterable[dict[str, Any]],
    codes: Mapping[str, str] | None = None,
    coding_record: Mapping[str, Any] | None = None,
    family_inputs: Iterable[str] = (),
) -> Sources:
    return Sources(
        study,
        settings,
        Entries(entries),
        profile_order(study),
        codes,
        coding_record,
        world_groups(study),
        frozenset(family_inputs),
    )


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
# T1's split of Completed by completion_source, after Completed, when the export carries it (run record 3;
# stuck_stop from run record 6)
SOURCE_COLUMNS = {
    "explorer": "Completed (explorer)",
    "coverage_stop": "Completed (coverage stop)",
    "stuck_stop": "Completed (stuck stop)",
}
# T1's Stuck column, after Time limit, when the export's run record knows the stuck stop (run record 6)
STUCK_COLUMN = "Stuck"
# the reported outcomes in words, by whether the export knows the stuck stop
REPORTED_TEXT = {
    False: "completed, time limit, no movement",
    True: "completed, time limit, stuck, no movement",
}


def accounting(study: Study) -> dict[tuple[str, str, str, str], dict[str, int]]:
    """Counts per robot, world, spawn and profile of every planned experiment, from ``experiments`` and ``runs``
    (not the export's own accounting table, so that the two can be compared): ``planned``, the experiment states
    (``done``, ``unresolved``), final runs by report class and reported runs by ``outcome_<outcome>``, completed
    reported runs by ``source_<completion_source>`` (run record 3), attempts replaced (``replaced_<reason>``)
    and discarded (``discarded_<trigger>`` or ``discarded_other``). T1 and F9."""
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
                if run["outcome"] == "completed" and run[COMPLETION_SOURCE] in COMPLETION_SOURCES:
                    group["source_" + run[COMPLETION_SOURCE]] += 1
    return counts


def t1(src: Sources) -> Table:
    """Accounting per robot, world, spawn and profile of every planned experiment (``accounting``)."""
    study = src.study
    counts = accounting(study)
    sources = (
        [s for s in COMPLETION_SOURCES if study.has_stuck or s != "stuck_stop"]
        if study.has_completion_source
        else []
    )
    others = (
        ("time_limit", STUCK, "no_movement") if study.has_stuck else ("time_limit", "no_movement")
    )
    columns = list(T1_COLUMNS)
    at = columns.index("Completed") + 1
    columns[at:at] = [SOURCE_COLUMNS[source] for source in sources]
    if study.has_stuck:
        at = columns.index("Time limit") + 1
        columns[at:at] = [STUCK_COLUMN]
    table = Table("T1", "Outcome accounting per stratum and profile", columns, sparse_counts=True)
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
                str(group["outcome_completed"]),
                *(str(group["source_" + source]) for source in sources),
                *(str(group["outcome_" + o]) for o in others),
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
        f"({REPORTED_TEXT[study.has_stuck]}), Wilson {level_text(src.level)} interval.",
        "Missing observations and exclusions are final runs outside the reported population "
        f"({outcome_rule(study)}); replaced and discarded attempts are not final. Every planned experiment has "
        "a row, also where no run was reported.",
    ]
    if sources:
        stuck_stop = (
            f" Completed (stuck stop): {STUCK_TEXT} at or above 97.0 % coverage."
            if "stuck_stop" in sources
            else ""
        )
        table.notes.append(
            "Completed (explorer): the explorer's completion event ended the run; Completed (coverage stop): "
            f"{COVERAGE_STOP_TEXT}.{stuck_stop} "
            f"{'All' if stuck_stop else 'Both'} count as completed; a completed run without a recorded source "
            "is in no source column."
        )
    if study.has_stuck:
        table.notes.append(
            f"Stuck: {STUCK_TEXT} below 97.0 % coverage; a reported run, not completed, analysed as a "
            "time-limit run except that its coverage is carried from its end to L."
        )
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
    add("Mapping condition", [mapping_condition(r) or "not recorded" for r in study.finals])
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
    if src.study.has_stuck:
        table.notes.append(
            f"R_completion counts every completion: the explorer's completion event, {COVERAGE_STOP_TEXT} and "
            f"the stuck stop at or above 97.0 % coverage ({STUCK_TEXT}); T1 counts them apart. A stuck run "
            "(below 97.0 %) is censored at L like a time-limit run; a milestone it reached before the stop is "
            "observed."
        )
    elif src.study.has_completion_source:
        table.notes.append(
            f"R_completion counts both completions: the explorer's completion event and {COVERAGE_STOP_TEXT}; "
            "T1 counts them apart."
        )
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
        "C(h) and A_h hold a completed or stuck run's final coverage to L (carry-forward) and a run without "
        "movement at its pre-motion coverage; no value across a sample gap above 2 s. D(h): odometry, moving "
        "runs only, a completed or stuck run's final distance held to L.",
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
        elif e.get("contract") == FAMILY_CONTRACT:
            value = _family_hypothesis_value(e, src.level)
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
    if any(
        e.get("contract") == FAMILY_CONTRACT
        for e in src.entries.where(estimand="hypothesis", testable=True)
    ):
        table.notes.append(
            "Hypotheses on a family or density contrast (family-sample-v1, the unit in the Stratum column) are "
            "on rho = R/L, C(L) or A_L at L: a factor's effect by Lenth's ratio against a Monte Carlo null, a "
            "profile contrast by a one-sample t test on world mean differences, a density contrast by the "
            "stratified Welch test (stats.json p_method)."
        )
    if any(
        welch_note(e)
        for e in src.entries.where(estimand="hypothesis", testable=True, contract=FAMILY_CONTRACT)
    ):
        table.notes.append(
            "A density contrast's stratified Welch interval is approximate; the count of cells "
            "without variation is given beside it, as such a cell adds nothing to the variance. "
            + WELCH_CEILING
        )
    return table


def _family_hypothesis_value(e: Mapping[str, Any], level: float) -> str:
    """A testable family or contrast hypothesis's estimate (T5H): the outcome at L it is on, with its interval
    where it has one (a factor effect has none)."""
    outcome = {"R_completion": "rho", "C_h": "C_L", "A_h": "A_L"}.get(
        str(e.get("hypothesis_endpoint"))
    )
    name, unit, scale = FAMILY_OUTCOMES.get(outcome or "", ("", "ratio", 1.0))
    if e.get("factor"):
        name = f"effect of {e['factor']} on {name}"
    welch = welch_note(e)
    return f"{family_value(e, unit, level, scale)} ({name}{'; ' + welch if welch else ''})"


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
        "plan's window W: time-limit and stuck runs whose idle tail reaches W, Wilson interval.",
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


# T9 behaviour codebook ---------------------------------------------------------------------------------------


def t9(src: Sources) -> Table | None:
    """The behaviour codebook (FIGURE_EXTENSION FX-19): each category of the plan's taxonomy in declared order
    with its definition and the reported runs coded with it per profile over all strata and in all (per stratum:
    F23), then the reported runs the coding file does not name and the reported runs (the denominator); None
    without a taxonomy or a coding file."""
    taxonomy, codes = src.settings.taxonomy, src.codes
    if taxonomy is None or codes is None:
        return None
    present = {p for stratum in src.study.strata for p in stratum.profiles()}
    reported = {
        profile: [r for s in src.study.strata for r in s.reported if r["profile_id"] == profile]
        for profile in src.order
        if profile in present
    }
    table = Table(
        "T9",
        "Behaviour codebook and coded runs",
        ["Category", "Definition", *(src.name(p) for p in reported), "All"],
        wrap_column=1,
        extra=dict(coding=dict(src.coding_record or {})),
    )
    for identity, definition in [
        *taxonomy,
        (NOT_CODED, "Reported runs the coding file does not name."),
    ]:
        counts = [
            sum(codes.get(r["attempt_id"], NOT_CODED) == identity for r in runs)
            for runs in reported.values()
        ]
        table.rows.append([identity, definition, *map(str, counts), str(sum(counts))])
    sizes = [len(runs) for runs in reported.values()]
    table.rows.append(
        [REPORTED_RUNS, "The denominator of each column.", *map(str, sizes), str(sum(sizes))]
    )
    table.attempt_ids = {r["attempt_id"] for runs in reported.values() for r in runs}
    table.notes = [
        f"Reported runs ({REPORTED_TEXT[src.study.has_stuck]}) per profile over all strata, coded by the researcher "
        "against the categories and definitions declared in the plan's taxonomy (hashed with the plan); per "
        "stratum in F23. The last row is each column's denominator. Coding never changes a recorded outcome; "
        f"descriptive counts, not in stats.json. {blinding_text(src.coding_record)} {CODER_NOTE} "
        f"{CATEGORY_NOTE}",
    ]
    return table


def blinding_text(coding_record: Mapping[str, Any] | None) -> str:
    """How the coding was done as the coding file states it (``blinded``, ``taxonomy.read_coding``)."""
    return CODED_BLIND if (coding_record or {}).get("blinded") is True else CODED_UNBLINDED


# Families (family-sample-v1; plans/worlds/I6, I8 §3.3) ----------------------------------------------------------

# outcome -> its display name, the unit it is rounded in and the factor to that unit (success is a share)
FAMILY_OUTCOMES: dict[str, tuple[str, str, float]] = {
    "rho": ("ρ", "ratio", 1.0),
    "success": ("Success (%)", "pp", 100.0),
    "C_L": ("C(L) (%)", "pp", 1.0),
    "A_L": ("A_L (%)", "pp", 1.0),
}
NOT_ESTIMABLE = "not estimable"
DESCRIPTIVE_MARK = "†"  # T10: a family average or contrast without an interval (descriptive)
FAMILY_INTERVAL_NOTE = (
    "Family average: the equal-weight mean of the world means (spawns weighted equally within a world), with a t "
    "interval on the world means (n − 1 df) where the family has one; approximate; design points treated as "
    "exchangeable (plans/worlds/I6 §3.4). Success and families without an interval are descriptive; an interval "
    "the world means cannot give reads not estimable."
)


def family_ids(entries: Entries, kind: str = "world_family") -> list[str]:
    """The families (``kind`` "world_family") or density contrasts ("density_contrast") with ``family-sample-v1``
    entries, families in the world-group order (F1 … F5, others by name), contrasts by name."""
    found = {str(e[kind]) for e in entries.where(contract=FAMILY_CONTRACT) if e.get(kind)}
    if kind != "world_family":
        return sorted(found)
    return sorted(found, key=family_rank)


def family_rank(identity: str) -> tuple[int, str]:
    """A family's place in ``worlds.GROUP_ORDER`` (others after it, by name)."""
    return (GROUP_ORDER.index(identity) if identity in GROUP_ORDER else len(GROUP_ORDER), identity)


def family_title(identity: str) -> str:
    return GROUP_TITLES.get(identity, identity)


def declared_family(settings: Settings, identity: str) -> Family | None:
    return next((f for f in settings.families if f.id == identity), None)


def level_text_of(value: Any) -> str:
    """One design-point level as text: numbers without a trailing .0, lists joined by ×."""
    if isinstance(value, list | tuple):
        return "×".join(level_text_of(v) for v in value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return f"{value:g}"
    return str(value)


def design_keys(family: Family | None, design_points: Iterable[Mapping[str, Any]]) -> list[str]:
    """The design-point keys that label a family's worlds: the declared factors and fixed parameters, or every
    key of the worlds' design points for a family declared without factors."""
    if family is not None and family.factors:
        return [f.key for f in family.factors] + sorted(family.fixed)
    return sorted({key for point in design_points for key in point})


def design_label(point: Mapping[str, Any], keys: Iterable[str], separator: str = ", ") -> str:
    """A world's levels of ``keys`` (a key it lacks reads "—"), without the world's ID (I8 §3.3)."""
    return separator.join(level_text_of(point[k]) if k in point else DASH for k in keys)


def family_value(
    entry: Mapping[str, Any], unit: str, level: float, scale: float = 1.0, *, n: str | None = None
) -> str:
    """An entry's estimate with its interval: "not estimable" when the interval says so (never a zero width),
    the estimate alone without an interval; ``n`` appended as "(n = …)"."""
    estimate = entry.get("estimate")
    interval = entry.get("interval")
    value = None if estimate is None else scale * float(estimate)
    note = str((interval or {}).get("note") or "")
    if value is None:
        text = DASH
    elif interval and note.startswith(NOT_ESTIMABLE):
        text = f"{number(value, RESOLUTION[unit])} [{NOT_ESTIMABLE}]"
    else:
        text = with_interval(value, _scaled(interval, scale), unit, level)
    return text if n is None else f"{text} ({n})"


WELCH_CEILING = (
    "Near a ceiling mass of 0.8 (most runs at rho = 1) the stratified Welch interval does not keep nominal "
    "coverage (METHODOLOGY §4.9)."
)


def welch_note(entry: Mapping[str, Any]) -> str | None:
    """For an entry with a stratified Welch interval (a density contrast or its hypothesis): "approximate" and
    the count of cells without variation, read from the interval's note (``families.welch``); None for any other
    interval or without one."""
    interval = entry.get("interval") or {}
    if "Welch" not in str(interval.get("method") or ""):
        return None
    note = str(interval.get("note") or "")
    if note.startswith(f"{NOT_ESTIMABLE}: no cell varies"):
        return "approximate; no cell varies"
    if note.startswith(NOT_ESTIMABLE):
        return "approximate"
    found = re.search(r"(\d+) cells without variation", note)
    return f"approximate; {found.group(1) if found else 0} cells without variation"


def _family_entries(
    src: Sources, estimand: str, family: str, **fields: Any
) -> list[dict[str, Any]]:
    """The family's primary-population entries at L of ``estimand``."""
    return list(
        src.entries.where(
            contract=FAMILY_CONTRACT,
            estimand=estimand,
            world_family=family,
            population="primary",
            horizon_fraction=None,
            **fields,
        )
    )


def world_order(groups: Iterable[WorldGroup], family: str, members: Iterable[str]) -> list[str]:
    """A family's worlds in its world group's order (the ``worlds`` table's), then by ID."""
    group = next((g for g in groups if g.key == family), None)
    rank = {w: i for i, w in enumerate(group.worlds)} if group else {}
    return sorted(set(members), key=lambda w: (rank.get(w, len(rank)), w))


def _counts_text(counts: Mapping[str, int]) -> str:
    return "; ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(counts.items()) if v) or DASH


def t10(src: Sources) -> Table | None:
    """Family estimates (I8 §3.3): per family and profile, each world's mean of rho, success, C(L) and A_L with its
    n and final-run accounting, the family average with its interval and n worlds, and for each profile other
    than Default the family profile contrast; worlds without a value named in the notes. None without family
    entries."""
    families = family_ids(src.entries)
    if not families:
        return None
    outcomes = list(FAMILY_OUTCOMES)
    table = Table(
        "T10",
        "Family estimates (primary population, at L)",
        [
            "Family",
            "Profile",
            "World",
            "Design point",
            *(FAMILY_OUTCOMES[o][0].replace("ρ", "ρ = R/L") for o in outcomes),
            "Final runs",
        ],
        inputs=set(src.family_inputs),
    )
    missing_notes: list[str] = []
    for family in families:
        declared = declared_family(src.settings, family)
        means = _family_entries(src, "family_world_mean", family)
        profiles = [p for p in src.order if any(e["profile_id"] == p for e in means)]
        keys = design_keys(declared, (e.get("design_point") or {} for e in means))
        for profile in profiles:
            mine = [e for e in means if e["profile_id"] == profile]
            by_world: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for e in mine:
                by_world[str(e["world_id"])][str(e["endpoint"])] = e
            accounting: dict[str, int] = defaultdict(int)
            for world in world_order(src.groups, family, by_world):
                cells = by_world[world]
                first = cells.get("rho") or next(iter(cells.values()))
                for report_class, count in (first.get("final_runs") or {}).items():
                    accounting[report_class] += int(count)
                row = [
                    family,
                    src.name(profile),
                    world,
                    design_label(first.get("design_point") or {}, keys),
                ]
                for outcome in outcomes:
                    e = cells.get(outcome, {})
                    name, unit, scale = FAMILY_OUTCOMES[outcome]
                    if e.get("estimate") is None:
                        reasons = _counts_text(e.get("no_value") or {})
                        row.append(DASH if reasons == DASH else f"{DASH} ({reasons})")
                    else:
                        row.append(family_value(e, unit, src.level, scale, n=str(e.get("n", 0))))
                    table.attempt_ids |= set(e.get("attempt_ids", []))
                row.append(_counts_text(first.get("final_runs") or {}))
                table.rows.append(row)
            averages = {
                str(e["endpoint"]): e
                for e in _family_entries(src, "family_average", family, profile_id=profile)
            }
            row = [family, src.name(profile), "Family average", DASH]
            for outcome in outcomes:
                e = averages.get(outcome, {})
                _, unit, scale = FAMILY_OUTCOMES[outcome]
                text = family_value(e, unit, src.level, scale, n=str(e.get("n_worlds", 0)))
                if e and e.get("label"):
                    text += DESCRIPTIVE_MARK
                row.append(text)
                if e.get("worlds_missing"):
                    missing_notes.append(
                        f"{family}, {src.name(profile)}, {FAMILY_OUTCOMES[outcome][0]}: {e.get('note')}."
                    )
                for world, horizons in sorted((e.get("horizons_dropped") or {}).items()):
                    if horizons and outcome == "rho":
                        missing_notes.append(
                            f"{family}: horizons of {world} not shared by every world, left out of the "
                            f"shorter-horizon entries: {', '.join(f'{h:g} s' for h in horizons)}."
                        )
            row.append(_counts_text(accounting))
            table.rows.append(row)
            contrasts = {
                str(e["endpoint"]): e
                for e in _family_entries(src, "family_profile_contrast", family, profile_id=profile)
            }
            if contrasts:
                row = [family, src.name(profile), "Profile − Default", DASH]
                for outcome in outcomes:
                    e = contrasts.get(outcome, {})
                    _, unit, scale = FAMILY_OUTCOMES[outcome]
                    text = family_value(e, unit, src.level, scale, n=str(e.get("n_worlds", 0)))
                    row.append(text + (DESCRIPTIVE_MARK if e and e.get("label") else ""))
                    table.attempt_ids |= set(e.get("attempt_ids", []))
                row.append(DASH)
                table.rows.append(row)
    table.group_titles = {family: family_title(family) for family in families}
    table.extra["groups"] = dict(table.group_titles)
    table.notes = [
        "World rows: the world mean over its spawns (each spawn's mean weighted equally) of the reported runs, (n) = "
        "runs with a value; a dash names why a world has none. Final runs by report class "
        f"({outcome_rule(src.study)}); the average row sums them over the family's worlds.",
        FAMILY_INTERVAL_NOTE
        + f" Level {level_text(src.level)}; (n) = worlds with a value; {DESCRIPTIVE_MARK} descriptive, no "
        "interval.",
        "Profile − Default: the mean over worlds of each world's mean block difference (rho, C(L), A_L in their "
        "units, success in percentage points), with the t interval on the world differences where the family has "
        "one. Families are never pooled; sensitivity populations and shorter horizons are in stats.json "
        "(family-sample-v1).",
        *missing_notes,
    ]
    return table


EFFECT_LABELS = (
    ("approximate", "approximate (Lenth)"),
    ("descriptive only", "descriptive (cell missing)"),
    (NOT_ESTIMABLE, "not estimable (PSE 0)"),
    ("descriptive", "descriptive"),
)


def effect_label(label: str | None) -> str:
    """A factorial effect's or pair's label in a table cell or on a figure: the start of stats.json's label."""
    text = str(label or "")
    return next((short for start, short in EFFECT_LABELS if text.startswith(start)), text or DASH)


def effect_name(entry: Mapping[str, Any]) -> str:
    return str(entry.get("variant") or "").removeprefix("factor:").replace("×", " × ")


def effect_sort(entry: Mapping[str, Any]) -> tuple[int, list[str]]:
    """Main effects first, then two- and three-factor interactions, each by factor key."""
    factors = list(entry.get("factors") or [])
    return (len(factors), factors)


def family_hypothesis(
    src: Sources, family: str, profile: str, factor: str
) -> dict[str, Any] | None:
    """The pre-registered factor hypothesis on rho of ``family`` and ``profile`` (confirmatory studies), if any."""
    if not src.settings.confirmatory:
        return None
    return next(
        src.entries.where(
            contract=FAMILY_CONTRACT,
            estimand="hypothesis",
            world_family=family,
            profile_id=profile,
            factor=factor,
            hypothesis_endpoint="R_completion",
        ),
        None,
    )


def t11(src: Sources) -> Table | None:
    """Factorial effects and descriptive pairs (I8 §3.3): per family, profile and outcome (rho, C(L), A_L) each
    effect with Lenth's PSE, ME and SME where stats.json gives them, its label, and the Holm-adjusted p only for a
    pre-registered factor hypothesis of a confirmatory study (RD-4); then the declared pairs. None without them."""
    families = family_ids(src.entries)
    table = Table(
        "T11",
        "Factorial effects and descriptive pairs (primary population, at L)",
        [
            "Family",
            "Profile",
            "Outcome",
            "Effect",
            "Estimate",
            "|c|/PSE",
            "PSE",
            "ME",
            "SME",
            "Label",
            "p (Holm)",
        ],
        inputs=set(src.family_inputs),
    )
    labels: dict[str, str] = {}
    for family in families:
        effects = _family_entries(src, "family_factorial_effect", family)
        pairs_found = _family_entries(src, "family_pair", family)
        profiles = [
            p for p in src.order if any(e["profile_id"] == p for e in effects + pairs_found)
        ]
        for profile in profiles:
            for outcome in ("rho", "C_L", "A_L"):
                name, unit, _ = FAMILY_OUTCOMES[outcome]
                chosen = sorted(
                    (e for e in effects if e["profile_id"] == profile and e["endpoint"] == outcome),
                    key=effect_sort,
                )
                for e in chosen:
                    lenth = e.get("lenth") or {}
                    d = RESOLUTION[unit]
                    hypothesis = (
                        family_hypothesis(
                            src, family, profile, str(e["variant"]).removeprefix("factor:")
                        )
                        if outcome == "rho"
                        else None
                    )
                    p_text = DASH
                    if hypothesis is not None:
                        p_text = (
                            f"{p_value(hypothesis.get('p_holm'))} ({hypothesis['hypothesis_id']})"
                            if hypothesis.get("testable")
                            else f"not testable ({hypothesis['hypothesis_id']})"
                        )
                    ratio = e.get("ratio")
                    table.rows.append(
                        [
                            family,
                            src.name(profile),
                            name,
                            effect_name(e),
                            number(e.get("estimate"), d),
                            DASH if ratio is None else number(ratio, 2),
                            number(lenth.get("pse"), d + 1) if lenth else DASH,
                            number(lenth.get("me"), d + 1) if lenth else DASH,
                            number(lenth.get("sme"), d + 1) if lenth else DASH,
                            effect_label(e.get("label")),
                            p_text,
                        ]
                    )
                    labels[str(e.get("label"))] = effect_label(e.get("label"))
                    table.attempt_ids |= set(e.get("attempt_ids", []))
                for e in pairs_found:
                    if e["profile_id"] != profile or e["endpoint"] != outcome:
                        continue
                    first, second = e.get("worlds") or ["", ""]
                    table.rows.append(
                        [
                            family,
                            src.name(profile),
                            name,
                            f"pair {second} − {first}",
                            number(e.get("estimate"), RESOLUTION[unit]),
                            DASH,
                            DASH,
                            DASH,
                            DASH,
                            effect_label(e.get("label")),
                            DASH,
                        ]
                    )
                    table.attempt_ids |= set(e.get("attempt_ids", []))
    if not table.rows:
        return None
    table.group_titles = {
        f: family_title(f) for f in families if any(r[0] == f for r in table.rows)
    }
    table.extra["groups"] = dict(table.group_titles)
    table.notes = [
        "Effect c = mean of the world means at the factor's high level − at its low level (an interaction: of the "
        "product of the factors' ±1 codes), on rho = R/L and on C(L) and A_L in percentage points. Lenth's pseudo "
        "standard error (PSE) on rho from 7 effects on: ME = t(1 − α/2, m/3)·PSE, SME = t(γ, m/3)·PSE, "
        "γ = (1 + (1 − α)^(1/m))/2, labelled approximate (no pure error with one seed per setting; plans/worlds/I6 "
        f"§3.4.1); α from the plan ({level_text(src.level)}). Effects without Lenth and pairs are descriptive.",
        "Pairs: second − first world mean, descriptive. p (Holm) only for a pre-registered factor hypothesis of a "
        "confirmatory study, adjusted over the plan's whole list (Lenth's ratio against a Monte Carlo null; "
        "stats.json p_method); a dash elsewhere (RD-4).",
        *(f"{family}: {full}." for full, family in sorted(labels.items()) if full != family),
    ]
    return table


def tables(src: Sources) -> list[Table]:
    grouped = src.with_groups
    built = [t1(src), t2(src), grouped(t3(src)), grouped(t4(src)), grouped(t5(src))]
    hypotheses = t5_hypotheses(src)
    if hypotheses is not None:
        built.append(hypotheses)
    built += [grouped(t6(src)), grouped(t7(src)), t8(src), grouped(sensitivity(src))]
    codebook = t9(src)
    built += [] if codebook is None else [codebook]
    return built + [t for t in (t10(src), t11(src)) if t is not None]


def write(table: Table, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{table.name}.csv", directory / f"{table.name}.tex"]
    paths[0].write_text(table.csv(), encoding="utf-8")
    paths[1].write_text(table.tex(), encoding="utf-8")
    return paths
