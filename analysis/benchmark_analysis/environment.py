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

"""What produced an output: package version, source and lock hashes, Python and library versions and the
checkout commit (RESULTS_OVERHAUL A15). Nothing here holds a path, a host name or a time, so two runs on the
same export give byte-identical sidecars."""

from functools import cache
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import subprocess
from typing import Any

from . import __version__

PACKAGE = Path(__file__).resolve().parent
LOCK = PACKAGE.parent / "requirements.lock"


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PACKAGE), *arguments],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


@cache
def environment() -> dict[str, Any]:
    sources = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".mplstyle"):
            sources.update(path.relative_to(PACKAGE).as_posix().encode() + b"\0")
            sources.update(path.read_bytes() + b"\0")
    packages: dict[str, str | None] = {}
    if LOCK.is_file():
        for line in LOCK.read_text(encoding="utf-8").splitlines():
            name = line.partition("==")[0].strip()
            if name:
                try:
                    packages[name] = metadata.version(name)
                except metadata.PackageNotFoundError:
                    packages[name] = None
    status = _git("status", "--porcelain", "--", ".")
    return dict(
        module="benchmark_analysis",
        module_version=__version__,
        module_sha256=sources.hexdigest(),
        lock_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest() if LOCK.is_file() else None,
        python=platform.python_version(),
        packages=packages,
        commit=_git("rev-parse", "HEAD"),
        module_dirty=None if status is None else bool(status),
    )
