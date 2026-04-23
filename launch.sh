#!/usr/bin/env bash
set -eo pipefail

# Resolve repository root from script location so the script works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# ROS setup scripts may read unset vars, so keep nounset disabled while sourcing.
source /opt/ros/jazzy/setup.bash
if [[ -f "${PROJECT_ROOT}/install/setup.bash" ]]; then
  source "${PROJECT_ROOT}/install/setup.bash"
fi
set -u

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [world_name]" >&2
  exit 1
fi

# Default benchmark world. Users can override by passing a single world name argument.
WORLD="${1:-bookstore}"

collect_available_worlds() {
  # A world is considered runnable when folder and world file name match:
  # simulation/worlds/<name>/<name>.world or <name>.sdf
  local -a worlds=()
  local name dir

  while IFS= read -r -d '' dir; do
    name="$(basename "${dir}")"
    if [[ -f "${dir}/${name}.world" || -f "${dir}/${name}.sdf" ]]; then
      worlds+=("${name}")
    fi
  done < <(find "${PROJECT_ROOT}/simulation/worlds" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null || true)

  printf '%s\n' "${worlds[@]}" | sort -u
}

# Precompute world list once so validation and error messages stay consistent.
mapfile -t AVAILABLE_WORLDS < <(collect_available_worlds)

WORLD_VALID=false
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

# Baseline spawn tuned for the benchmark maps.
SPAWN_X="2.5"
SPAWN_Y="1.5"
SPAWN_Z="0.5"
SPAWN_YAW="-1.5707"

# Map-specific spawn overrides reduce initial collisions and improve reproducibility.
if [[ "${WORLD}" == "corridor" || "${WORLD}" == "bookstore" ]]; then
  SPAWN_X="0.0"
  SPAWN_Y="0.0"
fi
if [[ "${WORLD}" == "bookstore" ]]; then
  SPAWN_X="1.23"
  SPAWN_Y="6.35"
  SPAWN_Z="0.11"
  SPAWN_YAW="-3.10"
fi

# Verify critical overlay packages before launching anything.
if ! ros2 pkg prefix bme_ros2_navigation >/dev/null 2>&1; then
  echo "Workspace overlay is not available. Expected package 'bme_ros2_navigation' was not found." >&2
  echo "Rebuild the image/container so /opt/benchmark_ws/install/setup.bash contains the workspace packages." >&2
  exit 1
fi

if ! ros2 pkg prefix rviz_autonomous_exploration_benchmark >/dev/null 2>&1; then
  echo "Workspace overlay is not available. Expected package 'rviz_autonomous_exploration_benchmark' was not found." >&2
  echo "Rebuild the image/container so /opt/benchmark_ws/install/setup.bash contains the workspace packages." >&2
  exit 1
fi

# Nav2 can leave lifecycle-managed processes around between runs. We detect and
# terminate known nodes up front to avoid namespace conflicts and flaky bringup.
declare -a NAV2_PATTERNS=(
  "nav2_lifecycle_manager/lifecycle_manager"
  "nav2_controller/controller_server"
  "nav2_planner/planner_server"
  "nav2_bt_navigator/bt_navigator"
  "nav2_waypoint_follower/waypoint_follower"
  "nav2_behaviors/behavior_server"
  "nav2_map_server/map_server"
  "nav2_amcl/amcl"
  "nav2_velocity_smoother/velocity_smoother"
  "nav2_collision_monitor/collision_monitor"
  "nav2_smoother/smoother_server"
)

cleanup_existing_nav2() {
  # Fast pre-check avoids unnecessary pkill loops when nothing is running.
  local pattern
  local found=false

  for pattern in "${NAV2_PATTERNS[@]}"; do
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
      found=true
      break
    fi
  done

  if [[ "${found}" == false ]]; then
    return
  fi

  # Two-phase shutdown: try TERM first, then escalate to KILL.
  echo "Existing Nav2 processes detected. Cleaning up..."
  for pattern in "${NAV2_PATTERNS[@]}"; do
    pkill -TERM -f "${pattern}" 2>/dev/null || true
  done
  sleep 1
  for pattern in "${NAV2_PATTERNS[@]}"; do
    pkill -KILL -f "${pattern}" 2>/dev/null || true
  done
}

cleanup() {
  # Prevent recursive trap calls while we are already in cleanup.
  trap - EXIT INT TERM
  # Best-effort shutdown of launched child processes.
  [[ -n "${SPAWN_PID:-}" ]] && kill "${SPAWN_PID}" 2>/dev/null || true
  [[ -n "${NAV_PID:-}" ]] && kill "${NAV_PID}" 2>/dev/null || true
  [[ -n "${TRACKER_PID:-}" ]] && kill "${TRACKER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Ensure a clean process baseline before starting the new run.
cleanup_existing_nav2

# 1) Start world + robot spawning.
ros2 launch bme_ros2_navigation spawn_robot.launch.py \
  world:="${WORLD}" \
  x:="${SPAWN_X}" \
  y:="${SPAWN_Y}" \
  z:="${SPAWN_Z}" \
  yaw:="${SPAWN_YAW}" &
SPAWN_PID=$!

# Give simulation/spawn a short head start before Nav2 + tracker.
echo "spawn_robot.launch.py started (pid=${SPAWN_PID}, world=${WORLD}, x=${SPAWN_X}, y=${SPAWN_Y}, z=${SPAWN_Z}, yaw=${SPAWN_YAW}). Waiting 5 seconds..."
sleep 5

# 2) Start Nav2 + SLAM stack.
ros2 launch bme_ros2_navigation navigation_with_slam.launch.py &
NAV_PID=$!
echo "navigation_with_slam.launch.py started (pid=${NAV_PID})."

# Prefer installed tracker params; fall back to source file for pre-install/dev runs.
TRACKER_PARAMS_FILE="${PROJECT_ROOT}/install/rviz_autonomous_exploration_benchmark/share/rviz_autonomous_exploration_benchmark/config/frontier_path_tracker.yaml"
if [[ ! -f "${TRACKER_PARAMS_FILE}" ]]; then
  TRACKER_PARAMS_FILE="${PROJECT_ROOT}/rviz/src/frontier_path_tracker.yaml"
fi

# 3) Start path tracker for traveled-path telemetry and reset integration.
ros2 run rviz_autonomous_exploration_benchmark frontier_path_tracker.py \
  --ros-args \
  --params-file "${TRACKER_PARAMS_FILE}" \
  -p use_sim_time:=true &
TRACKER_PID=$!
echo "frontier_path_tracker.py started (pid=${TRACKER_PID})."

# Block while children run; Ctrl+C triggers trap-driven cleanup.
echo "All processes are running. Press Ctrl+C to stop all."
wait
