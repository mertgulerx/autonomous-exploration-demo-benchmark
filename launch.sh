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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
WORKSPACE_SETUP="${PROJECT_ROOT}/install/setup.bash"

# Native SDKs are installed in the runtime image. Use that installed runtime on
# hosts without the vendor underlays; inside Docker this branch cannot recurse.
if [[ ! -f /.dockerenv && "${BENCHMARK_USE_HOST:-0}" != "1" && ! -d /opt/vendor/rosbot/install ]]; then
  runtime_image="${BENCHMARK_DOCKER_IMAGE:-autonomous-exploration-benchmark:jazzy-harmonic}"
  if command -v docker >/dev/null 2>&1 && [[ "$(docker image inspect --format '{{index .Config.Labels "org.benchmark.multi_robot"}}' "${runtime_image}" 2>/dev/null || true)" == "1" ]]; then
    exec env BENCHMARK_DOCKER_SKIP_BUILD=1 "${PROJECT_ROOT}/docker.sh" "$@"
  fi
fi

print_workspace_build_help() {
  echo "Build the workspace from this checkout:" >&2
  printf '  cd %q\n' "${PROJECT_ROOT}" >&2
  echo "  source /opt/ros/jazzy/setup.bash" >&2
  echo "  git submodule update --init --recursive" >&2
  echo "  colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3" >&2
  echo "Then rerun ./launch.sh. The launcher sources the local overlay automatically." >&2
}

if [[ ! -f "${WORKSPACE_SETUP}" ]]; then
  echo "Workspace overlay is missing: ${WORKSPACE_SETUP}" >&2
  print_workspace_build_help
  exit 1
fi

# ROS setup scripts may read unset vars, so keep nounset disabled while sourcing.
source /opt/ros/jazzy/setup.bash
source "${WORKSPACE_SETUP}"
set -u

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [world_name]" >&2
  exit 1
fi

# Optional RViz preselection; no scenario or simulation starts here.
WORLD="${1:-}"

collect_available_worlds() {
  # The worlds the runner offers: the checkout registry's entries except development
  # worlds, in file order. The coordinator validates its installed copy; after a
  # registry edit, rebuild exploration_benchmark_runner so both agree.
  python3 -c '
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    worlds = (yaml.safe_load(handle) or {}).get("worlds") or {}
for name, entry in worlds.items():
    if (entry or {}).get("role") != "development":
        print(name)
' "${PROJECT_ROOT}/exploration_benchmark_runner/config/worlds.yaml" ||
    echo "Cannot read the world registry: ${PROJECT_ROOT}/exploration_benchmark_runner/config/worlds.yaml" >&2
}

# Precompute world list once so validation and error messages stay consistent.
mapfile -t AVAILABLE_WORLDS < <(collect_available_worlds)

WORLD_VALID=false
[[ -z "${WORLD}" ]] && WORLD_VALID=true
for name in "${AVAILABLE_WORLDS[@]}"; do
  if [[ "${name}" == "${WORLD}" ]]; then
    WORLD_VALID=true
    break
  fi
done

if [[ "${WORLD_VALID}" == false ]]; then
  # Fail early with explicit choices to avoid launching into an unintended world.
  echo "Unknown world: ${WORLD}" >&2
  if [[ ${#AVAILABLE_WORLDS[@]} -gt 0 ]]; then
    echo "Available worlds: ${AVAILABLE_WORLDS[*]}" >&2
  fi
  exit 1
fi

# Verify critical overlay packages before launching anything.
for package in bme_ros2_navigation rviz_autonomous_exploration_benchmark exploration_benchmark_runner exploration_benchmark_interfaces exploration_benchmark_robots; do
  if ! ros2 pkg prefix "${package}" >/dev/null 2>&1; then
    echo "Workspace overlay is incomplete: '${package}' was not found after sourcing ${WORKSPACE_SETUP}." >&2
    print_workspace_build_help
    exit 1
  fi
done

# Scope control topics to this invocation; RViz inherits the session identifier.
export BENCHMARK_PROJECT_ROOT="${PROJECT_ROOT}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
export BENCHMARK_SESSION_ID="session_$(cat /proc/sys/kernel/random/uuid | tr -d '-')"
COORDINATOR_PID=""
RVIZ_PID=""
cleanup() {
  trap - INT TERM EXIT
  if [[ -n "${COORDINATOR_PID}" ]]; then
    kill -INT "${COORDINATOR_PID}" 2>/dev/null || true
    # The runner owns and saves its workers, including descendants in new groups.
    wait "${COORDINATOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RVIZ_PID}" ]]; then
    kill -INT "${RVIZ_PID}" 2>/dev/null || true
    wait "${RVIZ_PID}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT
# Run the installed entry point directly: a background `ros2 run` wrapper can
# inherit an ignored SIGINT and fail to forward the launcher shutdown request.
COORDINATOR_EXECUTABLE="$(ros2 pkg prefix exploration_benchmark_runner)/lib/exploration_benchmark_runner/coordinator"
COORDINATOR_ARGS=(--project-root "${PROJECT_ROOT}")
[[ -z "${WORLD}" ]] || COORDINATOR_ARGS+=(--world "${WORLD}")
"${COORDINATOR_EXECUTABLE}" "${COORDINATOR_ARGS[@]}" &
COORDINATOR_PID=$!
RVIZ_CONFIG="$(ros2 pkg prefix --share bme_ros2_navigation)/rviz/navigation.rviz"
rviz2 -d "${RVIZ_CONFIG}" --ros-args -p use_sim_time:=true &
RVIZ_PID=$!
# Closing either application ends this session. Cleanup never uses executable names.
wait -n "${COORDINATOR_PID}" "${RVIZ_PID}" || true
