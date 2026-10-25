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

# Place SLAM Toolbox 2.8.5 (the release the binary lock pins) with the benchmark's grid-extent
# patch into a colcon source directory, so a workspace build overlays the apt package
# (mapping-known-pose-v3, plans/worlds/I10). The source archive is checked by its SHA-256.
# Usage: fetch_slam_toolbox.sh SOURCE_DIR
set -euo pipefail

VERSION="2.8.5"
ARCHIVE_URL="https://github.com/SteveMacenski/slam_toolbox/archive/refs/tags/${VERSION}.tar.gz"
ARCHIVE_SHA256="657de32eb37b9e39da80882f356438df6aa8d8898a17acdb62f7a4d0fa8d3f70"
PATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches/slam_toolbox-${VERSION}-grid-extent.patch"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SOURCE_DIR" >&2
  exit 2
fi
destination="$1/slam_toolbox"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

curl -fsSL --retry 3 --connect-timeout 30 --max-time 600 -o "${work}/slam_toolbox.tar.gz" "${ARCHIVE_URL}"
echo "${ARCHIVE_SHA256}  ${work}/slam_toolbox.tar.gz" | sha256sum -c --quiet -
tar -xzf "${work}/slam_toolbox.tar.gz" -C "${work}"
patch -d "${work}/slam_toolbox-${VERSION}" -p1 --forward --quiet < "${PATCH}"
rm -rf "${destination}"
mkdir -p "$1"
mv "${work}/slam_toolbox-${VERSION}" "${destination}"
echo "SLAM Toolbox ${VERSION} with $(basename "${PATCH}") in ${destination}"
