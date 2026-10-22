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

"""``python -m benchmark_analysis --export DIR [--out DIR]`` (RESULTS_OVERHAUL A15, §7).

The outputs go to ``--out`` or, by default, beside the export: ``<study>/analysis/<version>-<hash8>`` where the
hash is the export's ``SHA256SUMS`` hash (writing into the export would break it). The directory is built under a
temporary name and renamed when complete; an existing one is never overwritten, and a stop (SIGTERM) removes
both. A relative ``--out`` is relative to the working directory.

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
from .environment import environment
from .errors import Refused
from .export import Export, file_sha256, open_export
from .outputs import json_text, write_all
from .settings import settings
from .style import PRESETS


def log(message: str) -> None:
    print(message, flush=True)


def default_out(export: Export) -> Path:
    """``results/studies/<study>/analysis/<version>-<hash8>`` beside ``.../exports/<export>``."""
    if export.directory.parent.name != "exports":
        raise Refused(
            "the export is not in a study's exports/ directory: name an output with --out"
        )
    return export.directory.parent.parent / "analysis" / f"{__version__}-{export.sums_sha256[:8]}"


def run(
    export_directory: Path, out: Path | None, *, preset: str = "ieee-journal", sans: bool = False
) -> Path:
    log(f"benchmark_analysis {__version__}: checking {export_directory.name}")
    export = open_export(export_directory)
    config = settings(export.study)
    target = (out or default_out(export)).resolve()
    if target.is_relative_to(export.directory):
        raise Refused(
            "the output directory cannot be inside the export (it would break SHA256SUMS)"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir()  # claims the name; the finished outputs replace this empty directory
    except FileExistsError as exc:
        raise Refused(f"{target.name} already exists; outputs are never overwritten") from exc
    staging = target.with_name(f"{target.name}.partial-{os.getpid()}")
    try:
        staging.mkdir()
        log(f"study {export.study_id} ({config.evidence_level}), {len(export.files)} files checked")
        for warning in write_all(export, config, staging, preset=preset, sans=sans, log=log):
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
                    outputs=listing,
                )
            ),
            encoding="utf-8",
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        target.rmdir()
        raise
    log(f"done: {target}")
    return target


def _stop(signum: int, frame: object) -> None:
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
    arguments = parser.parse_args(argv)
    # A stop request (the GUI's terminate) unwinds like an error, so the claimed directory and the staging are
    # removed and the next run is not blocked.
    signal.signal(signal.SIGTERM, _stop)
    try:
        run(arguments.export, arguments.out, preset=arguments.preset, sans=arguments.sans)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0
