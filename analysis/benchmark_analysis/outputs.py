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

The output directory holds ``stats.json``, ``coverage_grid.parquet``, ``tables/<T>.csv|.tex|.pdf`` (the PDF a typeset sheet, RO-19),
``figures/<F>.pdf`` with ``contact_sheets/<F>.png``, and one ``<output>.sidecar.json`` beside each: the code and
environment, the export's ``SHA256SUMS`` hash and the hashes of the files read, the settings and their sources,
the seed, the evidence level, the confirmatory checks, the attempts behind the output, its ``counts`` (RO-21: n
as it uses them, runs and for paired outputs pairs, and what it leaves out) and, for figures, the ``check_pdf``
result and the alt text. Nothing written holds a time, a path or a host name, so two runs on one
export give byte-identical files.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from . import checks, figures, sheets, style, tables
from .data import Study, coverage_grid, load
from .environment import environment
from .estimates import STATS_FORMAT, Entries, estimates
from .export import Export
from .settings import Settings

STUDY_FILES = ("study.yaml", "runs.parquet", "profiles.parquet")
INPUTS = {
    "stats": (*STUDY_FILES, "coverage.parquet", "trajectory_gt.parquet"),
    "coverage_grid": (*STUDY_FILES, "coverage.parquet"),
    "T1": (*STUDY_FILES, "experiments.parquet"),
    "T2": (*STUDY_FILES, "provenance.json"),
    "figure": (*STUDY_FILES, "coverage.parquet", "trajectory_gt.parquet"),
}


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
PAIRED = ("T5", "S", "F3_")


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
    **extra: Any,
) -> str:
    export = study.export
    names = sorted({name for name in inputs if name in export.files})
    used = sorted(set(attempt_ids))
    output = str(extra.get("output", ""))
    return json_text(
        {
            **base,
            "inputs": export.input_hashes(*names),
            "attempt_ids": used,
            "counts": scope_counts(study, used, pairs=output.startswith(PAIRED)),
            **extra,
        }
    )


def write_all(
    export: Export,
    settings: Settings,
    directory: Path,
    *,
    preset: str,
    sans: bool,
    log: Callable[[str], None],
) -> list[str]:
    """Every output into ``directory``; returns the warnings to print."""
    study = load(export)
    entries = estimates(study, settings)
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
        _sidecar(base, study, INPUTS["stats"], used, output="stats.json", entries=len(entries)),
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
    source = tables.sources(study, settings, entries)
    table_directory = directory / "tables"
    for table in tables.tables(source):
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
                INPUTS.get(table.name, INPUTS["stats"]),
                table.attempt_ids,
                output=table.name,
                files=[f"{table.name}.csv", f"{table.name}.tex", f"{table.name}.pdf"],
                title=table.title,
                notes=table.notes,
                check_pdf=result,
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
        for built in figures.figures(context):
            path = figure_directory / f"{built.name}.pdf"
            built.figure.savefig(path, metadata={"CreationDate": None})
            result = checks.check_pdf(path, style.width_pt(preset, built.width))
            views = checks.contact_sheet(built.figure, sheet_directory / f"{built.name}.png")
            plt.close(built.figure)
            if not result.get("available", True):
                missing_poppler = True
            elif result.get("problems"):
                warnings.append(f"{built.name}: " + "; ".join(result["problems"]))
            inputs = list(INPUTS["figure"])
            if built.name == "F8":
                inputs += [name for name in export.files if name.startswith("references/")]
            (figure_directory / f"{built.name}.sidecar.json").write_text(
                _sidecar(
                    base,
                    study,
                    inputs,
                    built.attempt_ids,
                    output=f"{built.name}.pdf",
                    alt_text=built.alt_text,
                    caption_notes=built.notes,
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
