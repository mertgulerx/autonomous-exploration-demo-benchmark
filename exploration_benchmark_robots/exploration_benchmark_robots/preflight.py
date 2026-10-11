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

"""Conservative planar spawn screening using existing static collision grids.

It never changes scenario coordinates or coverage references.
"""

import math
import numpy as np


def validate_spawn(profile, spawn, reference):
    footprint = np.asarray(profile["footprint"], dtype=float)
    x, y, _, yaw = spawn
    c, s = math.cos(yaw), math.sin(yaw)
    points = footprint @ np.array([[c, s], [-s, c]]) + [x, y]
    c, s = math.cos(reference.origin[2]), math.sin(reference.origin[2])
    grid = (
        (points - reference.origin[:2])
        @ np.array([[c, -s], [s, c]])
        / reference.resolution
    )
    low = np.floor(grid.min(axis=0)).astype(int)
    high = np.floor(grid.max(axis=0)).astype(int)
    height, width = reference.collision.shape
    if min(low) < 0 or high[0] >= width or high[1] >= height:
        raise ValueError(
            f"{profile['label']}: fixed spawn footprint exceeds verified world geometry"
        )
    # Reject any collision cell in the rotated footprint's bounding region.
    # The small region is deliberately conservative around cell boundaries.
    if reference.collision[low[1] : high[1] + 1, low[0] : high[0] + 1].any():
        raise ValueError(
            f"{profile['label']}: fixed spawn footprint overlaps static collision geometry; choose another supported world"
        )
