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
# roadmap_explorer and roadmap_explorer_msgs from roadmap-explorer, rrt_explore from rrt-explore, tdle from
# TDLE and mrtsp_exploration_ros2 (base MRTSP, plans/EXTERNAL_EXPLORERS_MRTSP). Each source archive is checked by its SHA-256; the other packages and files of the repositories
# (multirobot_map_merge, roadmap_explorer_rviz_plugins, TDLE's bcr_bot demo workspaces) are left out.
# Two build-only edits: roadmap_explorer builds its gtests whenever ament_cmake_gtest is found, ignoring
# BUILD_TESTING, so they are gated on BUILD_TESTING as well (the image build skips them); tdle installs a
# launch/ directory its repository does not have, so an empty one is created. Three patches to methods, each
# checked by its SHA-256 as profiles.yaml records it: rrt_explore gets
# docker/patches/rrt_explore-41054af-bounds.patch (decision X5: grid reads outside the grid read as unknown),
# tdle docker/patches/tdle-b5ed3cf-bounds.patch (decision B5, the same rule for its gridValue) and
# mrtsp_exploration_ros2 docker/patches/mrtsp_exploration_ros2-d9bc180-benchmark.patch (decision M1: its
# completion is published on /exploration_complete; occ_threshold defaults to 60, maintainer 26 October 2026). mrtsp_exploration_ros2's tests are left out.
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
MRTSP_COMMIT="d9bc18085f49f6aa4396b4009d22ad5255096bf9"
MRTSP_URL="https://codeload.github.com/mertgulerx/mrtsp_exploration_ros2/tar.gz/${MRTSP_COMMIT}"
MRTSP_SHA256="971375f44998d796c92ee02519baf35d8aef9ed45089ba164beef4c6c1e4f5d5"
MRTSP_PATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches/mrtsp_exploration_ros2-d9bc180-benchmark.patch"
MRTSP_PATCH_SHA256="c998dc09178c49d0d63dbf7fd376086610440edd66b5249f3200d3317ecc9fac"
TDLE_COMMIT="b5ed3cf8573cb4a2387b5434d3ec4bb782ba286e"
TDLE_URL="https://codeload.github.com/SeanZsya/tdle/tar.gz/${TDLE_COMMIT}"
TDLE_SHA256="197f15e4d0e738af5657da715b56aad5718a680db67586fe094d8b1e5a196b7f"
TDLE_PATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches/tdle-b5ed3cf-bounds.patch"
TDLE_PATCH_SHA256="46bc72e212c9dc270aa5607d178c5b6eff3dc1574e9f5066618a638eea014c93"

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
fetch mrtsp_exploration_ros2 "${MRTSP_URL}" "${MRTSP_SHA256}" .

roadmap_cmake="${work}/roadmap-explorer/roadmap_explorer/CMakeLists.txt"
grep -qx 'if(ament_cmake_gtest_FOUND)' "${roadmap_cmake}"
sed -i 's/^if(ament_cmake_gtest_FOUND)$/if(BUILD_TESTING AND ament_cmake_gtest_FOUND)/' "${roadmap_cmake}"
echo "${RRT_EXPLORE_PATCH_SHA256}  ${RRT_EXPLORE_PATCH}" | sha256sum -c --quiet -
patch -d "${work}/rrt-explore" -p1 --forward --quiet < "${RRT_EXPLORE_PATCH}"
echo "${MRTSP_PATCH_SHA256}  ${MRTSP_PATCH}" | sha256sum -c --quiet -
patch -d "${work}/mrtsp_exploration_ros2" -p1 --forward --quiet < "${MRTSP_PATCH}"
rm -rf "${work}/mrtsp_exploration_ros2/test"
tdle_package="${work}/tdle/tdle"
grep -qx 'install(DIRECTORY launch/' "${tdle_package}/CMakeLists.txt"
[[ ! -e "${tdle_package}/launch" ]]
mkdir "${tdle_package}/launch"
echo "${TDLE_PATCH_SHA256}  ${TDLE_PATCH}" | sha256sum -c --quiet -
patch -d "${tdle_package}" -p1 --forward --quiet < "${TDLE_PATCH}"

mkdir -p "$1"
for name in m-explore-ros2 roadmap-explorer rrt-explore tdle mrtsp_exploration_ros2; do
  rm -rf "${1:?}/${name}"
  mv "${work}/${name}" "$1/${name}"
done
echo "explore_lite ${EXPLORE_LITE_COMMIT:0:7} in $1/m-explore-ros2," \
  "roadmap_explorer ${ROADMAP_COMMIT:0:7} in $1/roadmap-explorer," \
  "rrt_explore ${RRT_EXPLORE_COMMIT:0:7} with $(basename "${RRT_EXPLORE_PATCH}") in $1/rrt-explore, tdle ${TDLE_COMMIT:0:7} with $(basename "${TDLE_PATCH}") in $1/tdle," \
  "mrtsp_exploration_ros2 ${MRTSP_COMMIT:0:7} with $(basename "${MRTSP_PATCH}") in $1/mrtsp_exploration_ros2"
