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
            f"{profile['label']}: spawn footprint exceeds world geometry"
        )
    yy, xx = np.mgrid[low[1] : high[1] + 1, low[0] : high[0] + 1]
    centres = np.stack((xx + .5, yy + .5), axis=-1)
    overlaps = np.ones(xx.shape, dtype=bool)
    edges = np.roll(grid, -1, axis=0) - grid
    axes = np.concatenate((np.eye(2), np.stack((-edges[:, 1], edges[:, 0]), axis=1)))
    for axis in axes:
        projection = grid @ axis
        middle = centres @ axis
        radius = .5 * np.abs(axis).sum()
        overlaps &= (middle + radius > projection.min()) & (middle - radius < projection.max())
    if (reference.collision[low[1] : high[1] + 1, low[0] : high[0] + 1] & overlaps).any():
        raise ValueError(
            f"{profile['label']}: spawn footprint overlaps static collision geometry; choose another pose"
        )
    if (~reference.inclusion[low[1] : high[1] + 1, low[0] : high[0] + 1] & overlaps).any():
        raise ValueError(f"{profile['label']}: spawn footprint leaves the included world region")
