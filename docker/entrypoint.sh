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

set -e

source /opt/ros/jazzy/setup.bash
# Some NVIDIA runtimes mount the EGL library without its GLVND discovery file.
# Provide the standard vendor descriptor only when that library is present.
if [[ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] && ldconfig -p | grep -q libEGL_nvidia; then
  mkdir -p /usr/share/glvnd/egl_vendor.d
  printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi
if [[ -f /opt/benchmark_ws/install/setup.bash ]]; then
  source /opt/benchmark_ws/install/setup.bash
fi

exec "$@"
