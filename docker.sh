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

DOCKER_IMAGE="${BENCHMARK_DOCKER_IMAGE:-autonomous-exploration-benchmark:jazzy-harmonic}"
DOCKERFILE_PATH="${PROJECT_ROOT}/docker/Dockerfile"
WORLD="${1:-}"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [world_name]" >&2
  exit 1
fi

nvidia_runtime_available() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
}

build_docker_image() {
  if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "Dockerfile not found: ${DOCKERFILE_PATH}" >&2
    exit 1
  fi

  if [[ "${BENCHMARK_DOCKER_SKIP_BUILD:-0}" == "1" ]]; then
    if ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
      echo "Docker image '${DOCKER_IMAGE}' does not exist and BENCHMARK_DOCKER_SKIP_BUILD=1 is set." >&2
      exit 1
    fi
    return
  fi

  # Starting an installed runtime should not rebuild all native robot SDKs.
  if [[ "${BENCHMARK_DOCKER_REBUILD:-0}" != "1" ]] && docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
    return
  fi

  # Label the image with the repository states it is built from (commits, dirty flags, diff hashes), so
  # the suite records which explorer build ran (suite/benchmark_suite/provenance.py).
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to label the image with its source commits." >&2
    exit 1
  fi
  local labels=()
  local label
  while IFS= read -r label; do
    labels+=(--label "${label}")
  done < <(python3 "${PROJECT_ROOT}/suite/benchmark_suite/provenance.py" labels "${PROJECT_ROOT}" \
    org.benchmark.runtime.repository.)
  if [[ ${#labels[@]} -eq 0 ]]; then
    echo "Could not read the repository states for the image labels." >&2
    exit 1
  fi
  # A full build holds everything a refresh would; label it with the runtime version so ./suite_launch.sh does not
  # refresh it again (docker/RUNTIME_VERSION).
  labels+=(--label "org.benchmark.runtime.version=$(grep -v '^#' "${PROJECT_ROOT}/docker/RUNTIME_VERSION" | tr -d '[:space:]')")

  echo "Building Docker image '${DOCKER_IMAGE}'..."
  if docker --version 2>/dev/null | grep -qi podman; then
    docker build --format docker --pull "${labels[@]}" -t "${DOCKER_IMAGE}" -f "${DOCKERFILE_PATH}" "${PROJECT_ROOT}"
  else
    docker build --pull "${labels[@]}" -t "${DOCKER_IMAGE}" -f "${DOCKERFILE_PATH}" "${PROJECT_ROOT}"
  fi
}

prepare_x11_access() {
  if [[ -n "${DISPLAY:-}" && $(uname -s) == "Linux" ]] && command -v xhost >/dev/null 2>&1; then
    xhost +si:localuser:root >/dev/null 2>&1 || true
  fi
}

run_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed or not available in PATH." >&2
    exit 1
  fi

  build_docker_image

  local host_os
  host_os="$(uname -s)"

  if [[ -z "${DISPLAY:-}" && "${host_os}" != "Linux" ]]; then
    export DISPLAY="${BENCHMARK_DOCKER_DISPLAY:-host.docker.internal:0}"
  fi

  prepare_x11_access

  local -a docker_args
  docker_args=(
    run
    --rm
    -it
    --shm-size=1g
    -e QT_X11_NO_MITSHM=1
    -e LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-0}"
  )

  if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
    docker_args+=(-e ROS_DOMAIN_ID)
  fi

  if [[ -n "${DISPLAY:-}" ]]; then
    docker_args+=(-e DISPLAY)
  fi

  if [[ -d /tmp/.X11-unix ]]; then
    docker_args+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  fi

  if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    docker_args+=(-e XAUTHORITY -v "${XAUTHORITY}:${XAUTHORITY}:ro")
  fi

  if [[ -n "${WAYLAND_DISPLAY:-}" && -n "${XDG_RUNTIME_DIR:-}" && -d "${XDG_RUNTIME_DIR}" ]]; then
    docker_args+=(
      -e WAYLAND_DISPLAY
      -e XDG_RUNTIME_DIR=/tmp/xdg-runtime-dir
      -e QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland;xcb}"
      -v "${XDG_RUNTIME_DIR}:/tmp/xdg-runtime-dir:rw"
    )
  else
    docker_args+=(-e QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}")
  fi

  if [[ -e /dev/dri ]]; then
    docker_args+=(--device /dev/dri)
  fi

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia_runtime_available; then
    docker_args+=(
      --gpus all
      -e NVIDIA_VISIBLE_DEVICES=all
      -e NVIDIA_DRIVER_CAPABILITIES=all
    )
  fi

  if [[ -d "${PROJECT_ROOT}/logs" ]]; then
    docker_args+=(-v "${PROJECT_ROOT}/logs:/opt/benchmark_ws/logs")
  fi

  if [[ -d "${PROJECT_ROOT}/results" ]]; then
    docker_args+=(-v "${PROJECT_ROOT}/results:/opt/benchmark_ws/results")
  fi

  echo "Starting benchmark configuration in Docker (image=${DOCKER_IMAGE}, preselected world=${WORLD:-none})..."
  local -a launch_args=(./launch.sh)
  [[ -z "${WORLD}" ]] || launch_args+=("${WORLD}")
  docker "${docker_args[@]}" "${DOCKER_IMAGE}" "${launch_args[@]}"
}

run_docker
