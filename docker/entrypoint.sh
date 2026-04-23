#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
if [[ -f /opt/benchmark_ws/install/setup.bash ]]; then
  source /opt/benchmark_ws/install/setup.bash
fi

exec "$@"
