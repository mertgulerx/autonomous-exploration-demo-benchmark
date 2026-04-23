#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

DOCKER_IMAGE="${BENCHMARK_DOCKER_IMAGE:-autonomous-exploration-benchmark:jazzy-harmonic}"
DOCKERFILE_PATH="${PROJECT_ROOT}/docker/Dockerfile"
WORLD="${1:-bookstore}"

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

  echo "Building Docker image '${DOCKER_IMAGE}'..."
  if docker --version 2>/dev/null | grep -qi podman; then
    docker build --format docker --pull -t "${DOCKER_IMAGE}" -f "${DOCKERFILE_PATH}" "${PROJECT_ROOT}"
  else
    docker build --pull -t "${DOCKER_IMAGE}" -f "${DOCKERFILE_PATH}" "${PROJECT_ROOT}"
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

  echo "Starting benchmark in Docker (image=${DOCKER_IMAGE}, world=${WORLD})..."
  docker "${docker_args[@]}" "${DOCKER_IMAGE}" ./launch.sh "${WORLD}"
}

run_docker
