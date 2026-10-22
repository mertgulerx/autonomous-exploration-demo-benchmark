#!/usr/bin/env bash
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

set -euo pipefail
suite_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
suite_env="${suite_root}/.venv-suite-gui"
suite_lock="${suite_root}/suite/requirements-gui.lock"
if [[ ! -x "${suite_env}/bin/python" ]]; then
    python3.12 -m venv "${suite_env}"
fi
suite_hash="$(sha256sum "${suite_lock}")"
if [[ ! -f "${suite_env}/.dependencies" || "$(cat "${suite_env}/.dependencies")" != "${suite_hash}" ]]; then
    "${suite_env}/bin/python" -m pip install --disable-pip-version-check -r "${suite_lock}"
    printf '%s\n' "${suite_hash}" > "${suite_env}/.dependencies"
fi
# The analysis package's own environment (RESULTS_OVERHAUL R5); the suite runs without it, so a failed build only
# warns and is retried at the next start.
analysis_env="${suite_root}/.venv-analysis"
analysis_lock="${suite_root}/analysis/requirements.lock"
analysis_hash="$(sha256sum "${analysis_lock}" 2>/dev/null || true)"
if [[ ! -f "${analysis_lock}" ]]; then
    printf 'suite_launch.sh: %s is missing; the suite starts without the analysis environment\n' "${analysis_lock}" >&2
elif [[ ! -f "${analysis_env}/.dependencies" || "$(cat "${analysis_env}/.dependencies")" != "${analysis_hash}" ]]; then
    if { [[ -x "${analysis_env}/bin/python" ]] || python3.12 -m venv "${analysis_env}"; } \
        && "${analysis_env}/bin/python" -m pip install --disable-pip-version-check -r "${analysis_lock}" >&2; then
        printf '%s\n' "${analysis_hash}" > "${analysis_env}/.dependencies"
    else
        printf 'suite_launch.sh: the analysis environment %s could not be built; the suite starts without it\n' \
            "${analysis_env}" >&2
    fi
fi
cd -- "${suite_root}"
exec env -u QT_PLUGIN_PATH -u QT_QPA_PLATFORM_PLUGIN_PATH -u LD_LIBRARY_PATH \
    PYTHONPATH="${suite_root}/suite" "${suite_env}/bin/python" -m benchmark_suite.deployment "$@"
