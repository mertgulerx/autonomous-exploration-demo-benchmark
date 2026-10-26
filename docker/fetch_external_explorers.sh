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

# Place the external explorers of the comparison (plans/EXTERNAL_EXPLORERS, EXTERNAL_EXPLORERS_RRT_TDLE) at their
# pinned commits into a colcon source directory: explore_lite and explore_lite_msgs from m-explore-ros2,
# roadmap_explorer and roadmap_explorer_msgs from roadmap-explorer, rrt_explore from rrt-explore and tdle from
# TDLE. Each source archive is checked by its SHA-256; the other packages and files of the repositories
# (multirobot_map_merge, roadmap_explorer_rviz_plugins, TDLE's bcr_bot demo workspaces) are left out.
# Two build-only edits: roadmap_explorer builds its gtests whenever ament_cmake_gtest is found, ignoring
# BUILD_TESTING, so they are gated on BUILD_TESTING as well (the image build skips them); tdle installs a
# launch/ directory its repository does not have, so an empty one is created. One patch to a method:
# rrt_explore gets docker/patches/rrt_explore-41054af-bounds.patch (decision X5: grid reads outside the grid
# read as unknown), checked by its SHA-256 as profiles.yaml records it.
# Usage: fetch_external_explorers.sh SOURCE_DIR
set -euo pipefail

EXPLORE_LITE_COMMIT="326cf8a0b487c34246bb8f3326afbcd69576dc60"
EXPLORE_LITE_URL="https://codeload.github.com/robo-friends/m-explore-ros2/tar.gz/${EXPLORE_LITE_COMMIT}"
EXPLORE_LITE_SHA256="9f10c7590d68e45b0ba16dba10fd193ddb50b08d09a10165c27cff7bea40415e"
ROADMAP_COMMIT="203ff2602e1a1e41eec7ef663eacc21266e270c6"
ROADMAP_URL="https://codeload.github.com/suchetanrs/roadmap-explorer/tar.gz/${ROADMAP_COMMIT}"
ROADMAP_SHA256="0a6b7a4e07e777a8da03dbd1ee690833b9a66a5fb2a342e91b79bcf919052201"
RRT_EXPLORE_COMMIT="41054af8fb68d074ca4927773ecc96afcce8bb90"
RRT_EXPLORE_URL="https://codeload.github.com/acachathuranga/rrt-explore/tar.gz/${RRT_EXPLORE_COMMIT}"
RRT_EXPLORE_SHA256="9b584eb9e8a63cc426560be6974c9f7ed5039b7a942f0487cc42af5eba268128"
RRT_EXPLORE_PATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches/rrt_explore-41054af-bounds.patch"
RRT_EXPLORE_PATCH_SHA256="fef19df8c7da2feb7d3c4f3c860ceb50692a57e4efce3ada59175f10902d16e1"
TDLE_COMMIT="b5ed3cf8573cb4a2387b5434d3ec4bb782ba286e"
TDLE_URL="https://codeload.github.com/SeanZsya/tdle/tar.gz/${TDLE_COMMIT}"
TDLE_SHA256="197f15e4d0e738af5657da715b56aad5718a680db67586fe094d8b1e5a196b7f"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SOURCE_DIR" >&2
  exit 2
fi
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

# fetch NAME URL SHA256 PACKAGE_DIR...: download and check one archive, then keep only the listed package
# directories (and the repository licence) in SOURCE_DIR/NAME. A package directory "." keeps the whole
# repository (a repository that is one package); one with slashes keeps that directory under its last name.
fetch() {
  local name="$1" url="$2" sha256="$3"
  shift 3
  local archive="${work}/${name}.tar.gz" staged="${work}/${name}"
  curl -fsSL --retry 3 --connect-timeout 30 --max-time 600 -o "${archive}" "${url}"
  echo "${sha256}  ${archive}" | sha256sum -c --quiet -
  mkdir -p "${work}/${name}-src" "${staged}"
  tar -xzf "${archive}" -C "${work}/${name}-src" --strip-components=1
  local package
  for package in "$@"; do
    if [[ "${package}" == "." ]]; then
      rmdir "${staged}"
      mv "${work}/${name}-src" "${staged}"
      return
    fi
    mv "${work}/${name}-src/${package}" "${staged}/$(basename "${package}")"
  done
  if [[ -f "${work}/${name}-src/LICENSE" ]]; then
    mv "${work}/${name}-src/LICENSE" "${staged}/LICENSE"
  fi
}

fetch m-explore-ros2 "${EXPLORE_LITE_URL}" "${EXPLORE_LITE_SHA256}" explore explore_lite_msgs
fetch roadmap-explorer "${ROADMAP_URL}" "${ROADMAP_SHA256}" roadmap_explorer roadmap_explorer_msgs
fetch rrt-explore "${RRT_EXPLORE_URL}" "${RRT_EXPLORE_SHA256}" .
fetch tdle "${TDLE_URL}" "${TDLE_SHA256}" tdle_ws/src/tdle

roadmap_cmake="${work}/roadmap-explorer/roadmap_explorer/CMakeLists.txt"
grep -qx 'if(ament_cmake_gtest_FOUND)' "${roadmap_cmake}"
sed -i 's/^if(ament_cmake_gtest_FOUND)$/if(BUILD_TESTING AND ament_cmake_gtest_FOUND)/' "${roadmap_cmake}"
echo "${RRT_EXPLORE_PATCH_SHA256}  ${RRT_EXPLORE_PATCH}" | sha256sum -c --quiet -
patch -d "${work}/rrt-explore" -p1 --forward --quiet < "${RRT_EXPLORE_PATCH}"
tdle_package="${work}/tdle/tdle"
grep -qx 'install(DIRECTORY launch/' "${tdle_package}/CMakeLists.txt"
[[ ! -e "${tdle_package}/launch" ]]
mkdir "${tdle_package}/launch"

mkdir -p "$1"
for name in m-explore-ros2 roadmap-explorer rrt-explore tdle; do
  rm -rf "${1:?}/${name}"
  mv "${work}/${name}" "$1/${name}"
done
echo "explore_lite ${EXPLORE_LITE_COMMIT:0:7} in $1/m-explore-ros2," \
  "roadmap_explorer ${ROADMAP_COMMIT:0:7} in $1/roadmap-explorer," \
  "rrt_explore ${RRT_EXPLORE_COMMIT:0:7} with $(basename "${RRT_EXPLORE_PATCH}") in $1/rrt-explore, tdle ${TDLE_COMMIT:0:7} in $1/tdle"
