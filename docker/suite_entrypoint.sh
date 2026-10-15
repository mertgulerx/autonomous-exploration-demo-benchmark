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

set -eo pipefail
if [[ -e /dev/nvidiactl && -r /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
fi
source /opt/ros/jazzy/setup.bash
source /opt/benchmark_ws/install/setup.bash
if [[ -n "${BENCHMARK_TELEMETRY_INSTALL:-}" ]]; then
  source "${BENCHMARK_TELEMETRY_INSTALL}/setup.bash"
fi
if [[ -n "${BENCHMARK_SUITE_PROJECT:-}" ]]; then
  export PYTHONPATH="${BENCHMARK_SUITE_PROJECT}/suite:${BENCHMARK_SUITE_PROJECT}/exploration_benchmark_runner:${BENCHMARK_SUITE_PROJECT}/exploration_benchmark_robots:${PYTHONPATH:-}"
else
  export PYTHONPATH="/opt/benchmark-suite/plugins/runner:/opt/benchmark-suite/plugins/robots:${PYTHONPATH:-}"
fi
exec /opt/benchmark-suite-env/bin/python -m benchmark_suite.cli "$@"
