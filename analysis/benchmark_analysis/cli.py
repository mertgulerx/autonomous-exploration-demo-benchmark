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

"""``python -m benchmark_analysis --export DIR [--out DIR] [--coding FILE]`` (RESULTS_OVERHAUL A15, §7).

The outputs go to ``--out`` or, by default, beside the export: ``<study>/analysis/<version>-<hash8>`` where the
hash is the export's ``SHA256SUMS`` hash (writing into the export would break it). The directory is built under a
temporary name and renamed when complete; an existing one is never overwritten, and a stop (SIGTERM) removes
both (once the outputs are complete, a stop is ignored and the run finishes). A relative ``--out`` is relative to
the working directory. A refusal of a study export writes ``<study>/analysis/refused-<version>-<hash8>.json`` beside where the outputs would have gone (RO-22: the
reason, the column whose values differ, the stratum and each value's runs); a later run of that version on
that export that finishes removes it. The family estimates (``families.family_estimates``, plans/worlds/I6 §3.8)
run with the study's load, before any directory is made: their refusals (families declared on an export without
world metadata, a mismatched factor, runs of one family that mix robot, code, SLAM configuration, time-limit rule,
registry or generator version) are written the same way, with the family or contrast as the stratum.

``--coding`` names the researcher's coding of runs against the plan's taxonomy (FIGURE_EXTENSION FX-19,
``taxonomy.read_coding``); it is never part of the export: its SHA-256, coder, procedure and number of codes are
recorded at the top of ``analysis.json`` (``coding``, null without one) and in the sidecars of F23 and T9, never
its path. A coded analysis has its own default name, ``<version>-<hash8>-coding-<sha8>`` with the first 8 hex
digits of the coding file's SHA-256, so it sits beside the uncoded analysis whose coding sheet was coded. A
coding file that cannot be used is refused as a call before any directory is made (no refusal file): the study
declares no taxonomy, the file is malformed, or it names a category the codebook lacks or a run that is not a
reported final run.

Protocol for a caller such as the GUI: progress lines on stdout, ``warning: …`` lines on stdout, ``done: <dir>``
as the last stdout line, ``refused: …`` on stderr. Exit codes: 0 done, 2 refused, 1 failed, 143 stopped.
"""

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import shutil
import signal
import sys

from . import __version__
from .data import load
from .environment import environment
from .errors import Refused
from .export import Export, file_sha256, open_export
from .families import family_estimates
from .outputs import json_text, views_document, write_all
from .settings import settings
from .style import PRESETS
from .taxonomy import Coding, check_runs, read_coding


def log(message: str) -> None:
    print(message, flush=True)


def default_out(export: Export, coding: Coding | None = None) -> Path:
    """``results/studies/<study>/analysis/<version>-<hash8>`` beside ``.../exports/<export>``, with
    ``-coding-<sha8>`` for an analysis with a coding file."""
    if export.directory.parent.name != "exports":
        raise Refused(
            "the export is not in a study's exports/ directory: name an output with --out",
            kept=False,
        )
    name = f"{__version__}-{export.sums_sha256[:8]}"
    if coding is not None:
        name += f"-coding-{coding.sha256[:8]}"
    return export.directory.parent.parent / "analysis" / name


def refusal_path(export: Export) -> Path | None:
    """Where a refusal of ``export`` is kept: beside its default output (None outside a study's exports/)."""
    try:
        target = default_out(export)
    except Refused:
        return None
    return target.with_name(f"refused-{target.name}.json")


def run(
    export_directory: Path,
    out: Path | None,
    *,
    preset: str = "ieee-journal",
    sans: bool = False,
    coding: Path | None = None,
) -> Path:
    log(f"benchmark_analysis {__version__}: checking {export_directory.name}")
    export = open_export(export_directory)
    try:
        return _analyse(export, out, preset=preset, sans=sans, coding_file=coding)
    except Refused as exc:
        kept = refusal_path(export) if out is None and exc.kept else None
        if kept is not None:
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_text(
                json_text(
                    dict(
                        format="refusal-v1",
                        study_id=export.study_id,
                        export_sha256sums=export.sums_sha256,
                        module_version=__version__,
                        **exc.document(),
                    )
                ),
                encoding="utf-8",
            )
        raise


def _analyse(
    export: Export, out: Path | None, *, preset: str, sans: bool, coding_file: Path | None
) -> Path:
    config = settings(export.study)
    coding = None if coding_file is None else read_coding(coding_file, config)
    target = (out or default_out(export, coding)).resolve()
    if target.is_relative_to(export.directory):
        raise Refused(
            "the output directory cannot be inside the export (it would break SHA256SUMS)",
            kept=False,
        )
    study = load(export)
    # the family estimates' refusals (plans/worlds/I6 §3.8) are the study's, as data.load's: no directory is made
    family = family_estimates(study, config)
    if coding is not None:
        check_runs(coding, study)  # a refused coding file leaves no directory behind
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f"{target.name}.partial-{os.getpid()}")
    claimed = False  # only a directory this run claimed is removed on a stop or an error
    try:
        try:
            target.mkdir()  # claims the name; the finished outputs replace this empty directory
        except FileExistsError as exc:
            raise Refused(
                f"{target.name} already exists; outputs are never overwritten", kept=False
            ) from exc
        claimed = True
        staging.mkdir()
        log(f"study {export.study_id} ({config.evidence_level}), {len(export.files)} files checked")
        if coding is not None:
            log(f"coding file {coding.sha256[:8]}: {len(coding.codes)} runs coded")
        for warning in write_all(
            export,
            config,
            staging,
            preset=preset,
            sans=sans,
            log=log,
            coding=coding,
            study=study,
            family=family,
        ):
            log(f"warning: {warning}")
        listing = {
            path.relative_to(staging).as_posix(): file_sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        (staging / "analysis.json").write_text(
            json_text(
                dict(
                    study_id=export.study_id,
                    export_sha256sums=export.sums_sha256,
                    environment=environment(),
                    settings=config.document(),
                    coding=None if coding is None else coding.document(),
                    outputs=listing,
                    views=views_document(staging),
                )
            ),
            encoding="utf-8",
        )
        # From here the outputs are complete: a stop request no longer interrupts the rename that keeps them.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        staging.rename(target)  # replaces the empty claimed directory
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if claimed and target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        raise
    stale = refusal_path(export)
    if out is None and stale is not None:
        try:
            stale.unlink(missing_ok=True)  # this version now analyses the export
        except OSError as exc:  # the outputs are complete; the old refusal only stays visible
            log(f"warning: the earlier refusal {stale.name} could not be removed: {exc}")
    log(f"done: {target}")
    return target


def _stop(signum: int, frame: object) -> None:
    signal.signal(signum, signal.SIG_IGN)  # a second request must not interrupt the cleanup
    raise SystemExit(128 + signum)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark_analysis", description="Statistics, tables and figures of a study export"
    )
    parser.add_argument(
        "--export", type=Path, required=True, help="a finished study export directory"
    )
    parser.add_argument(
        "--out", type=Path, help="a new output directory (default: beside the export)"
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="ieee-journal",
        help="journal figure preset (FP-6)",
    )
    parser.add_argument("--sans", action="store_true", help="sans-serif figures (FP-9)")
    parser.add_argument(
        "--coding",
        type=Path,
        help="the researcher's coding of runs against the plan's taxonomy (YAML; F23, T9)",
    )
    arguments = parser.parse_args(argv)
    # A stop request (the GUI's terminate) unwinds like an error, so the claimed directory and the staging are
    # removed and the next run is not blocked.
    signal.signal(signal.SIGTERM, _stop)
    try:
        run(
            arguments.export,
            arguments.out,
            preset=arguments.preset,
            sans=arguments.sans,
            coding=arguments.coding,
        )
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0
