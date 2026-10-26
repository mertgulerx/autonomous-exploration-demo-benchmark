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

"""Writes every output with its sidecar (RESULTS_OVERHAUL §7 and A15; FIGURE_STYLE §1, §11).

The output directory holds ``stats.json``, ``coverage_grid.parquet``, ``tables/<T>.csv|.tex|.pdf`` (the PDF a
typeset sheet, RO-19), ``figures/<F>.pdf`` with ``contact_sheets/<F>.png``, and one ``<output>.sidecar.json`` beside each: the code and
environment, the export's ``SHA256SUMS`` hash and the hashes of the files read, the settings and their sources,
the seed, the evidence level, the confirmatory checks, the attempts behind the output, its ``counts`` (RO-21: n
as it uses them, runs and for paired outputs pairs, and what it leaves out) and, for figures, the ``check_pdf``
result and the alt text. Nothing written holds a time, a path or a host name, so two runs on one
export give byte-identical files.

``stats.json`` holds ``estimates.estimates``, for a method comparison the ``method-summaries-v1`` entries of
``summaries`` (C5), every entry of a profile with its ``method`` (explorer_id, C2), and, when the analysis block declares families or density
contrasts, the ``family-sample-v1`` entries of ``families.family_estimates`` (plans/worlds/I6 §3.8), with Holm
over the plan's whole hypothesis list (``families.merge_hypotheses``); the export files the family estimates read
(the ``worlds`` table, experiments, provenance and the runs' manifests) are hashed in the sidecars of
``stats.json``, T10, T11 and the family figures (F24, F24b, F25, F28).
"""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
import io
from itertools import chain
import json
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from . import (
    behaviour,
    checks,
    comparisons,
    composites,
    conduct,
    decisions,
    families,
    families_figures,
    figures,
    heatmaps,
    regionmaps,
    schematic,
    sheets,
    style,
    summaries,
    tables,
    taxonomy,
)
from .data import Study, coverage_grid, load
from .environment import environment
from .estimates import EFFECTS, STATS_FORMAT, Entries, estimates
from .export import Export
from .settings import MILESTONES, Settings, milestone_endpoint, milestone_label

STUDY_FILES = ("study.yaml", "runs.parquet", "profiles.parquet")
STATS_INPUTS = (*STUDY_FILES, "coverage.parquet", "trajectory_gt.parquet")
INPUTS = {
    "stats": STATS_INPUTS,
    "coverage_grid": (*STUDY_FILES, "coverage.parquet"),
    "T1": (*STUDY_FILES, "experiments.parquet"),
    "T2": (*STUDY_FILES, "provenance.json"),
    "T9": STUDY_FILES,
    "figure": (*STUDY_FILES, "coverage.parquet", "trajectory_gt.parquet"),
    # ordered and headed by world group (plans/worlds/I8 §3.1): the worlds table too, when exported
    **{
        name: (*STATS_INPUTS, "worlds.parquet")
        for name in ("T3", "T4", "T5", "T6", "T7", "S", "T12", "T13", "T15")
    },
}


@dataclass(frozen=True, slots=True)
class View:
    """One row of the Study tab's list (FIGURE_EXTENSION FX-4), written into analysis.json: the files a view shows,
    or a family of files chosen by variant (a file-name prefix) with what goes under the chosen one and the
    variants' labels. ``kind`` is table, chart or data; ``runs`` is False for a view with no runs behind it (a
    schematic), which the tab shows without an n line. ``grouped`` marks a figure drawn once per world group and
    page (``figures.grouped``, plans/worlds/I8 §3.1): when its plain file is not written, its group variants are
    listed instead. The GUI's built-in list (``gui_study.KINDS``) serves only analyses before 1.3.0; a view added
    here needs no GUI change."""

    code: str
    title: str
    kind: str
    icon: str
    files: tuple[str, ...] = ()
    variants: str = ""
    then: tuple[str, ...] = ()
    variant_labels: tuple[tuple[str, str], ...] = ()
    runs: bool = True
    grouped: bool = False

    def document(self) -> dict[str, Any]:
        return dict(
            code=self.code,
            title=self.title,
            kind=self.kind,
            icon=self.icon,
            files=list(self.files),
            variants=self.variants,
            then=list(self.then),
            variant_labels=dict(self.variant_labels),
            runs=self.runs,
        )


# the variants of F3 and F13, one per paired effect, in file-name order, labelled without the unit
EFFECT_LABELS = tuple(sorted((e.name, e.label.rsplit(" (", 1)[0]) for e in EFFECTS))
VIEWS: tuple[View, ...] = (
    View("T1", "Accounting", "table", "table", ("tables/T1.pdf",)),
    View("F9", "Accounting flow", "chart", "chart", ("figures/F9.pdf",)),
    View("F10", "Trial strip", "chart", "chart", ("figures/F10.pdf",), grouped=True),
    View("T2", "Setup and provenance", "table", "table", ("tables/T2.pdf",)),
    View("F12", "World atlas", "chart", "chart", ("figures/F12.pdf",), grouped=True),
    View("F1", "Coverage over time", "chart", "chart", ("figures/F1.pdf",), grouped=True),
    View(
        "F2/T3",
        "Attainment",
        "chart",
        "chart",
        variants="figures/F2_",
        then=("tables/T3.pdf",),
        variant_labels=(
            *((milestone_endpoint(m), milestone_label(m)) for m in MILESTONES),
            ("R_completion", "Completion"),
        ),
        grouped=True,
    ),
    View(
        "F3",
        "Paired effects",
        "chart",
        "chart",
        variants="figures/F3_",
        variant_labels=EFFECT_LABELS,
        grouped=True,
    ),
    View(
        "F13",
        "Paired estimation",
        "chart",
        "chart",
        variants="figures/F13_",
        variant_labels=EFFECT_LABELS,
        grouped=True,
    ),
    View(
        "F14", "Paired gap at equal coverage", "chart", "chart", ("figures/F14.pdf",), grouped=True
    ),
    View("F16", "Coverage against distance", "chart", "chart", ("figures/F16.pdf",), grouped=True),
    View("T4", tables.T4_TITLE, "table", "table", ("tables/T4.pdf",)),
    View("T5", "Paired differences", "table", "table", ("tables/T5.pdf", "tables/T5H.pdf")),
    View("S", "Sensitivity", "table", "table", ("tables/S.pdf",)),
    View("T6", "Compute cost", "table", "table", ("tables/T6.pdf",)),
    View("F5", "Compute", "chart", "chart", ("figures/F5.pdf",), grouped=True),
    View("T7", "Reliability", "table", "table", ("tables/T7.pdf",)),
    View("F6", "Outcomes", "chart", "chart", ("figures/F6.pdf",), grouped=True),
    # one file per stratum: the labels come from the figures' sidecars when the analysis is written
    View("F15", "Behaviour timeline", "chart", "chart", variants="figures/F15_"),
    View("F17", "Map heat maps", "chart", "chart", variants="figures/F17_"),
    # region-measures-v1 (plans/worlds/I8 §3.6): one F27 per stratum, labelled from its sidecar, then T12
    View(
        "F27/T12", "Region map", "chart", "chart", variants="figures/F27_", then=("tables/T12.pdf",)
    ),
    # C4: completions by source and the explorer completions below 95 %
    View("T14", "Completions by source", "table", "table", ("tables/T14.pdf",)),
    View("T8", "Timing validity", "table", "table", ("tables/T8.pdf",)),
    View("F11", "RTF distribution", "chart", "chart", ("figures/F11.pdf",)),
    View("F18", "Decision timing", "chart", "chart", ("figures/F18.pdf",), grouped=True),
    View("F19", "Decision anatomy", "chart", "chart", ("figures/F19.pdf",)),
    View("F20", "Algorithm steps (schematic)", "chart", "chart", ("figures/F20.pdf",), runs=False),
    View("F8", "Representative runs", "chart", "chart", ("figures/F8.pdf",), grouped=True),
    View("F21", "Main results", "chart", "chart", ("figures/F21.pdf",), grouped=True),
    View(
        "F22",
        "Teasers",
        "chart",
        "chart",
        variants="figures/F22_",
        variant_labels=(("a", "Per world"), ("b", "One run")),
        grouped=True,
    ),
    View("F23/T9", "Behaviour taxonomy", "chart", "chart", ("figures/F23.pdf", "tables/T9.pdf")),
    # family-sample-v1 (plans/worlds/I8 §3.3, §3.7): one F24 and F25 per family, labelled with its group's title
    View(
        "F24/T10",
        "Family world means",
        "chart",
        "chart",
        variants="figures/F24_",
        then=("tables/T10.pdf",),
        grouped=True,
    ),
    View("F24b", "Family profile contrasts", "chart", "chart", ("figures/F24b.pdf",)),
    View(
        "F25/T11",
        "Factorial effects",
        "chart",
        "chart",
        variants="figures/F25_",
        then=("tables/T11.pdf",),
        grouped=True,
    ),
    # one per density contrast: the labels come from the figures' sidecars (``label``)
    View("F28", "Density contrast", "chart", "chart", variants="figures/F28_"),
    # method-summaries-v1 (cross-method readiness C5): a method comparison's cross-world summaries
    View(
        "F29/T15",
        "Data profile",
        "chart",
        "chart",
        ("figures/F29.pdf",),
        then=("tables/T15.pdf",),
        grouped=True,
    ),
    View("F30", "Dolan–Moré profile", "chart", "chart", ("figures/F30.pdf",), grouped=True),
    View(
        "F31/T13",
        "Wins and improvement",
        "chart",
        "chart",
        ("figures/F31.pdf",),
        then=("tables/T13.pdf",),
        grouped=True,
    ),
    # the coding sheet's pages (FIGURE_EXTENSION FX-19); their number depends on the study
    View("F23 sheet", "Coding sheet", "chart", "chart", variants="figures/F23_sheet_"),
    View("stats", "Statistics", "data", "braces", ("stats.json",)),
)
# figures with no runs behind them (a schematic): their sidecars hash no study inputs
NO_RUNS = frozenset(file for view in VIEWS if not view.runs for file in view.files)


def _sidecar_field(directory: Path, pdf: str, key: str) -> Any:
    """One key of an output's sidecar (``figures/F15_x.pdf`` -> ``figures/F15_x.sidecar.json``), or None."""
    sidecar = directory / f"{pdf.removesuffix('.pdf')}.sidecar.json"
    return json.loads(sidecar.read_text(encoding="utf-8")).get(key) if sidecar.is_file() else None


def _group_label(group: dict[str, Any]) -> str:
    """A group variant's label: its group's title (the strata's when the export lists no worlds), with the page
    when the group has several."""
    title = group.get("title") or "All strata"
    pages = group.get("pages") or 1
    return str(title) if pages == 1 else f"{title}, page {group.get('page')} of {pages}"


def views_document(directory: Path) -> list[dict[str, Any]]:
    """``analysis.json``'s ``views``: VIEWS with only the files written in ``directory`` (a fixed label of a
    variant not written dropped, a view with no file left out), where a view of per-stratum variants without
    fixed labels takes each variant's label from its figure's sidecar (``stratum``), so the Study tab names the
    stratum.

    A grouped view (``View.grouped``, ``figures.grouped``) whose file was written per world group or page
    (``F1_SC.pdf``, ``F22_a_SC_p2.pdf``) lists those files as variants (``F1_SC``; a view of fixed files gets the
    prefix ``figures/F1_``), each labelled with its sidecar's group (after the fixed label of the variant it
    belongs to, the sidecar's ``group.variant``: "Per world · Single challenge"), ``variant_order``, the
    variants in fixed-label, group (``group.index``) and page order (the Study tab sorts by name;
    plans/worlds/I8 §3.1), and ``variant_groups``, per group variant its base variant (``base``, "" for a view
    of fixed files), the base's fixed label (``base_label``) and its group's ``key``, ``index``, ``page``,
    ``pages`` and ``title`` (the Study tab's variant and group head). Views without group variants carry
    neither ``variant_order`` nor ``variant_groups``."""
    found = []
    for view in VIEWS:
        document = view.document()
        document["files"] = [f for f in view.files if (directory / f).is_file()]
        document["then"] = [f for f in view.then if (directory / f).is_file()]
        variants = view.variants
        if view.grouped and view.files and not document["files"]:
            variants = view.files[0].removesuffix(".pdf") + "_"
            document["variants"] = variants
        written = (
            sorted(
                p.relative_to(directory).as_posix().removeprefix(variants).removesuffix(".pdf")
                for p in directory.glob(f"{variants}*.pdf")
            )
            if variants
            else []
        )
        fixed = dict(view.variant_labels)
        groups = {
            variant: group
            for variant in (written if view.grouped else [])
            if isinstance(
                group := _sidecar_field(directory, f"{variants}{variant}.pdf", "group"), dict
            )
        }
        if groups:
            labels: dict[str, str] = {}
            places: dict[str, tuple[int, int, int]] = {}
            heads: dict[str, dict[str, Any]] = {}
            stem = variants.rsplit("/", 1)[-1]
            for variant in written:
                group = groups.get(variant)
                # the fixed variant a group variant belongs to, from its ungrouped name: "a" of F22_a_SC; ""
                # for a view of fixed files (F1 of F1_SC)
                base = variant
                if group is not None:
                    ungrouped = str(group["variant"])
                    base = ungrouped.removeprefix(stem) if ungrouped.startswith(stem) else ""
                if group is not None:
                    heads[variant] = dict(
                        base=base,
                        base_label=fixed.get(base, ""),
                        key=str(group.get("key") or ""),
                        index=int(group["index"]),
                        page=int(group["page"]),
                        pages=int(group["pages"]),
                        title=str(group.get("title") or "All strata"),
                    )
                parts = [fixed[base]] if base in fixed else []
                parts += [_group_label(group)] if group is not None else []
                labels[variant] = " · ".join(parts) or variant
                rank = list(fixed).index(base) if base in fixed else 0
                order = (int(group["index"]), int(group["page"])) if group else (-1, 0)
                places[variant] = (rank, *order)
            document["variant_labels"] = labels
            document["variant_order"] = sorted(written, key=lambda v: places[v])
            # the Study tab's variant and group head (I9): each group variant's base and group
            document["variant_groups"] = heads
        elif variants and fixed:
            document["variant_labels"] = {k: fixed[k] for k in fixed if k in written}
        elif variants:
            labels = {}
            for variant in written:
                path = f"{variants}{variant}.pdf"
                stratum = _sidecar_field(directory, path, "stratum")
                stratum = (
                    stratum if stratum is not None else _sidecar_field(directory, path, "label")
                )
                if isinstance(stratum, str):
                    labels[variant] = stratum
            document["variant_labels"] = labels
        if document["files"] or written:
            found.append(document)
    return found


def json_text(value: Any) -> str:
    """How every JSON output is written: sorted keys, UTF-8, no NaN."""
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"


def confirmatory_checks(study: Study, settings: Settings) -> list[dict[str, Any]]:
    """The checks RESULTS_OVERHAUL §7 records in every sidecar; they warn, the researcher decides the label."""
    meta = study.export.study
    found: list[dict[str, Any]] = []
    registered = settings.plan_registered_utc
    if registered is None:
        found.append(
            dict(check="attempts_before_registration", passed=None, detail="no plan_registered_utc")
        )
    else:
        moment = datetime.strptime(registered, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        limit = int(moment.timestamp() * 1e9)
        early = sorted(
            r["attempt_id"] for r in study.runs if r["wall_start_ns"] and r["wall_start_ns"] < limit
        )
        found.append(
            dict(
                check="attempts_before_registration",
                passed=not early,
                detail=f"{len(early)} attempts started before {registered}",
                attempt_ids=early,
            )
        )
    before, after = meta.get("plan_sha256"), meta.get("plan_sha256_at_export")
    found.append(
        dict(
            check="plan_hash_unchanged",
            passed=None if before is None or after is None else before == after,
            detail="the registered part's hash at submission and at export"
            if before and after
            else "no plan or no hash at export",
        )
    )
    provenance = study.export.provenance
    documents = provenance.get("documents") or {}
    dirty = sorted(
        {
            f"{identity[:12]}:{role}"
            for identity in {r["provenance_id"] for r in study.finals if r["provenance_id"]}
            for role, state in ((documents.get(identity) or {}).get("repositories") or {}).items()
            if (state or {}).get("dirty")
        }
    )
    found.append(
        dict(
            check="clean_checkouts",
            passed=not dirty,
            detail="repositories dirty in the provenance of final runs: "
            + (", ".join(dirty) or "none"),
        )
    )
    return found


# Outputs whose n is pairs of runs (block × profile against Default), besides their runs.
PAIRED = ("T5", "S", "F3_", "F13_", "F14", "F24b", "T13", "F31")


def _pairs(study: Study, used: set[str]) -> int:
    """Profile runs paired with their block's Default run, both among ``used``."""
    count = 0
    for stratum in study.strata:
        baseline = study.baselines.get(stratum.robot_id)
        blocks: dict[str, set[str]] = {}
        for run in stratum.finals:
            if run["attempt_id"] in used:
                blocks.setdefault(run["block_id"], set()).add(run["profile_id"])
        count += sum(
            len(profiles - {baseline}) for profiles in blocks.values() if baseline in profiles
        )
    return count


def scope_counts(study: Study, attempt_ids: Iterable[str], *, pairs: bool) -> dict[str, Any]:
    """RO-21: the output's n and what it leaves out, for the n line under its view.

    ``runs``: final runs it uses; ``pairs`` (paired outputs only); ``left_out``: the study's other final runs by
    report class (``reported`` for reported runs this output does not use, such as parallel runs in the compute
    outputs), including the final runs of cells without a reported run; ``replaced`` (a later attempt exists) and
    ``discarded``: attempts that are not final; ``strata`` and ``profiles``."""
    used = set(attempt_ids)
    finals = [run for run in study.runs if run["is_final"]]  # every cell's, not only the strata's
    left = Counter(str(run["report_class"]) for run in finals if run["attempt_id"] not in used)
    earlier = Counter(
        "discarded" if run["outcome"] == "manual_discard" else "replaced"
        for run in study.runs
        if not run["is_final"]
        and (run["outcome"] == "manual_discard" or run["replaced_reason"] is not None)
    )
    counts: dict[str, Any] = dict(
        runs=sum(run["attempt_id"] in used for run in finals),
        final_runs=len(finals),
        left_out=dict(sorted(left.items())),
        replaced=earlier["replaced"],
        discarded=earlier["discarded"],
        strata=len(study.strata),
        profiles=len({run["profile_id"] for run in finals}),
    )
    if pairs:
        counts["pairs"] = _pairs(study, used)
    return counts


def _sidecar(
    base: dict[str, Any],
    study: Study,
    inputs: Iterable[str],
    attempt_ids: Iterable[str],
    *,
    with_counts: bool = True,
    **extra: Any,
) -> str:
    """One sidecar; ``with_counts`` False for an output with no runs behind it (``NO_RUNS``): no ``counts``. An
    ``extra`` key (an output's own fields, a figure's ``extra``) that would overwrite a base or sidecar key is
    refused, so no output's values can replace the sidecar's own."""
    reserved = {*base, "inputs", "attempt_ids", "counts"}
    if clash := sorted(reserved & set(extra)):
        raise ValueError(
            f"sidecar of {extra.get('output')}: extra keys {clash} would overwrite its own"
        )
    export = study.export
    names = sorted({name for name in inputs if name in export.files})
    used = sorted(set(attempt_ids))
    output = str(extra.get("output", ""))
    counts = (
        {"counts": scope_counts(study, used, pairs=output.startswith(PAIRED))}
        if with_counts
        else {}
    )
    return json_text(
        {**base, "inputs": export.input_hashes(*names), "attempt_ids": used, **counts, **extra}
    )


# axes positions after one constrained-layout pass, rounded to this figure fraction (about 0.05 pt at double width)
LAYOUT_ROUNDING = 1e-4


def freeze_layout(figure: Figure) -> None:
    """Fix a constrained-layout figure's layout before it is saved: one layout pass (a PDF save to memory), every
    axes position rounded
    to ``LAYOUT_ROUNDING`` of the figure, and the layout engine removed, so the saved PDF and the contact sheet's
    redraw use the same positions. The layout solver's result carries float noise of up to about 0.01 pt that
    follows the process's memory layout (two runs of one export differed; FIGURE_STYLE §1 asks for
    byte-identical reruns); rounding removes it unless a value lies on a rounding boundary. Figures laid out by
    hand (no constrained layout) are left as they are."""
    if not isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine):
        return
    # the layout as the PDF backend computes it on saving (its renderer and dpi), not a screen renderer's
    figure.savefig(io.BytesIO(), format="pdf", metadata={"CreationDate": None})
    for axes in figure.axes:
        box = axes.get_position()
        x0, y0, width, height = (
            round(v / LAYOUT_ROUNDING) * LAYOUT_ROUNDING
            for v in (box.x0, box.y0, box.width, box.height)
        )
        axes.set_position((x0, y0, width, height))
    figure.set_layout_engine("none")


def write_all(
    export: Export,
    settings: Settings,
    directory: Path,
    *,
    preset: str,
    sans: bool,
    log: Callable[[str], None],
    coding: taxonomy.Coding | None = None,
    study: Study | None = None,
    family: families.FamilyResult | None = None,
) -> list[str]:
    """Every output into ``directory``; returns the warnings to print. ``coding`` is the researcher's coding file
    (``--coding``: ``taxonomy.read_coding``, already checked against the runs by ``taxonomy.check_runs``);
    ``study`` the export's loaded study and ``family`` its family estimates when the caller has them
    (``data.load`` and ``families.family_estimates`` otherwise; a refusal of either is the caller's)."""
    study = study or load(export)
    family = family if family is not None else families.family_estimates(study, settings)
    entries = families.merge_hypotheses(estimates(study, settings) + family.entries, settings)
    entries += summaries.summary_entries(study, settings)
    for entry in entries:  # C2: every entry of a profile names its method (explorer_id)
        row = study.profiles.get(str(entry.get("profile_id")))
        if row is not None:
            entry["method"] = row["method"]
    family_inputs = sorted(family.inputs)
    checks_found = confirmatory_checks(study, settings)
    warnings = [
        f"confirmatory check {c['check']} failed: {c['detail']}"
        for c in checks_found
        if c["passed"] is False
    ]
    base = dict(
        study_id=export.study_id,
        evidence_level=settings.evidence_level,
        export_sha256sums=export.sums_sha256,
        plan_sha256=export.study.get("plan_sha256"),
        environment=environment(),
        settings=settings.document(),
        seed=settings.seed,
        confirmatory_checks=checks_found,
        preset=dict(name=preset, sans_serif=sans),
    )
    (directory / "stats.json").write_text(
        json_text(dict(format=STATS_FORMAT, study_id=export.study_id, entries=entries)),
        encoding="utf-8",
    )
    # the runs any entry uses (the sensitivity populations add excluded ones)
    used = sorted({a for entry in entries for a in entry.get("attempt_ids", [])})
    (directory / "stats.sidecar.json").write_text(
        _sidecar(
            base,
            study,
            [*INPUTS["stats"], *family_inputs],
            used,
            output="stats.json",
            entries=len(entries),
        ),
        encoding="utf-8",
    )
    log(f"stats.json: {len(entries)} entries")
    grid = coverage_grid(study)
    pq.write_table(
        grid, directory / "coverage_grid.parquet", compression="zstd", store_schema=False
    )
    (directory / "coverage_grid.sidecar.json").write_text(
        _sidecar(
            base,
            study,
            INPUTS["coverage_grid"],
            grid.column("attempt_id").unique().to_pylist(),
            output="coverage_grid.parquet",
            rows=grid.num_rows,
        ),
        encoding="utf-8",
    )
    source = tables.sources(
        study,
        settings,
        entries,
        None if coding is None else coding.categories(),
        None if coding is None else coding.document(),
        family_inputs,
    )
    table_directory = directory / "tables"
    for table in chain(tables.tables(source), regionmaps.tables(source), summaries.tables(source)):
        tables.write(table, table_directory)
        sheet_path = table_directory / f"{table.name}.pdf"
        with style.context(preset, sans):
            width = sheets.write(table, sheet_path, preset)
        result = checks.check_pdf(sheet_path, width)
        if result.get("problems"):
            warnings.append(f"table {table.name}: " + "; ".join(result["problems"]))
        (table_directory / f"{table.name}.sidecar.json").write_text(
            _sidecar(
                base,
                study,
                [*INPUTS.get(table.name, INPUTS["stats"]), *sorted(table.inputs)],
                table.attempt_ids,
                output=table.name,
                files=[f"{table.name}.csv", f"{table.name}.tex", f"{table.name}.pdf"],
                title=table.title,
                notes=table.notes,
                check_pdf=result,
                **table.extra,
            ),
            encoding="utf-8",
        )
        log(f"table {table.name}")
    figure_directory, sheet_directory = directory / "figures", directory / "contact_sheets"
    figure_directory.mkdir()
    sheet_directory.mkdir()
    grid_rows = zip(
        *(
            grid.column(name).to_pylist()
            for name in ("attempt_id", "t_s", "coverage_pct", "source")
        ),
        strict=True,
    )
    missing_poppler = False
    with style.context(preset, sans):
        context = figures.context(
            study, settings, Entries(entries), grid_rows, preset, source.order
        )
        # one module per group of figures (FIGURE_EXTENSION), in figure-number order; the family figures (F24,
        # F25, F28) after them, as they also take the family estimates' inputs; the taxonomy last, as it also
        # takes the coding file
        modules = (
            figures,
            conduct,
            comparisons,
            behaviour,
            heatmaps,
            regionmaps,
            decisions,
            schematic,
            composites,
            summaries,
        )
        # every module yields its figures one at a time: each is saved and closed before the next is drawn
        built_all = chain(
            *(module.figures(context) for module in modules),
            families_figures.figures(context, family_inputs),
            taxonomy.figures(context, coding),
        )
        for built in built_all:
            path = figure_directory / f"{built.name}.pdf"
            freeze_layout(built.figure)
            built.figure.savefig(path, metadata={"CreationDate": None})
            result = checks.check_pdf(path, style.width_pt(preset, built.width))
            views = checks.contact_sheet(built.figure, sheet_directory / f"{built.name}.png")
            plt.close(built.figure)
            if not result.get("available", True):
                missing_poppler = True
            elif result.get("problems"):
                warnings.append(f"{built.name}: " + "; ".join(result["problems"]))
            # a figure with no runs behind it reads no study files and has no counts
            with_runs = f"figures/{built.name}.pdf" not in NO_RUNS
            study_inputs = INPUTS["figure"] if with_runs else ()
            inputs = [*study_inputs, *sorted(built.inputs)]
            (figure_directory / f"{built.name}.sidecar.json").write_text(
                _sidecar(
                    base,
                    study,
                    inputs,
                    built.attempt_ids,
                    with_counts=with_runs,
                    output=f"{built.name}.pdf",
                    alt_text=built.alt_text,
                    # where an encoding repeats (style.profile_styles): said on every figure with runs
                    caption_notes=[*built.notes, *(context.style_notes if with_runs else ())],
                    check_pdf=result,
                    contact_sheet=dict(file=f"contact_sheets/{built.name}.png", views=views),
                    **built.extra,
                ),
                encoding="utf-8",
            )
            log(
                f"figure {built.name}"
                + ("" if result.get("ok", True) else " (check_pdf: see sidecar)")
            )
    if missing_poppler:
        warnings.append(
            "check_pdf did not run: poppler-utils (pdfinfo, pdffonts, pdfimages) not found"
        )
    if study.empty_cells:
        cells = ", ".join("/".join(cell) for cell in study.empty_cells)
        warnings.append(f"no reported run in {cells}: only T1 counts them")
    return warnings
