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

"""World metadata of a study export: the ``worlds`` table and each reference's region raster and descriptors.

Export version 2 carries them (plans/worlds/I2 §3.10; contract in plans/worlds/EXECUTION.md "Shared
contracts"). An older export holds none of them: every reader here returns ``None``, and the outputs that need
them are left out and counted. A world's display group is its family when it has one, else its tier (I8 §3.1);
``world_groups`` groups a study's strata for every grouped table and figure.
"""

from collections import defaultdict
from dataclasses import dataclass
import io
import json
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
import yaml

from .data import Stratum, Study
from .streams import Reads

# The display order of the groups (I8 §3.1, the world selection's section order, I9 §1).
GROUP_ORDER = ("A", "SC", "P", "F1", "F2", "F3", "F4", "F5", "RP", "MO")
GROUP_TITLES = {
    "A": "Original worlds",
    "SC": "Single challenge",
    "P": "Mixed worlds",
    "F1": "F1 Mazes",
    "F2": "F2 Offices",
    "F3": "F3 Warehouse aisles",
    "F4": "F4 Forests",
    "F5": "F5 Urban blocks",
    "RP": "Real places",
    "MO": "Moving objects",
}


@dataclass(frozen=True, slots=True)
class World:
    """One row of the export's ``worlds`` table."""

    world_id: str
    tier: str
    family: str | None  # set for tier F only
    role: str
    design_point: dict[str, Any]  # factor -> level; empty for named worlds
    seed: int | None
    generator_version: str | None
    world_yaml_sha256: str | None

    @property
    def group(self) -> str:
        return self.family if self.family is not None else self.tier


@dataclass(frozen=True, slots=True)
class Regions:
    """A reference's region raster: ``labels`` on the reference grid with row 0 the smallest y (as
    ``streams.ReferenceCells``), 0 outside the free floor, regions numbered 1 … n; ``counted`` the labels of
    regions of at least 4.0 m² (T1 §4.2)."""

    labels: npt.NDArray[np.uint16]
    counted: frozenset[int]
    origin: tuple[float, float]
    resolution: float


def worlds(reads: Reads) -> dict[str, World] | None:
    """The export's worlds by id, or ``None`` for an export without a ``worlds`` table."""
    rows = reads.table("worlds")
    if rows is None:
        return None
    found: dict[str, World] = {}
    for row in rows.to_pylist():
        found[str(row["world_id"])] = World(
            world_id=str(row["world_id"]),
            tier=str(row["tier"]),
            family=None if row["family"] is None else str(row["family"]),
            role=str(row["role"]),
            design_point=json.loads(row["design_point_json"]) if row["design_point_json"] else {},
            seed=None if row["seed"] is None else int(row["seed"]),
            generator_version=row["generator_version"],
            world_yaml_sha256=row["world_yaml_sha256"],
        )
    return found


def regions(reads: Reads, reference: str) -> Regions | None:
    """The region raster of ``references/<reference>/``, or ``None`` when the export holds none."""
    meta_bytes, raster, descriptors = (
        reads.file(f"references/{reference}/reference.yaml"),
        reads.file(f"references/{reference}/regions.png"),
        reads.file(f"references/{reference}/descriptors.json"),
    )
    if meta_bytes is None or raster is None or descriptors is None:
        return None
    meta = yaml.safe_load(meta_bytes)
    labels = np.array(Image.open(io.BytesIO(raster)), dtype=np.uint16)[::-1]
    counted = frozenset(int(label) for label in json.loads(descriptors)["regions_counted"])
    origin = (float(meta["origin"][0]), float(meta["origin"][1]))
    return Regions(labels, counted, origin, float(meta["resolution"]))


def descriptors(reads: Reads, reference: str) -> dict[str, Any] | None:
    """The official descriptors of ``references/<reference>/`` (``world-descriptors-v1``), or ``None``."""
    data = reads.file(f"references/{reference}/descriptors.json")
    return None if data is None else dict(json.loads(data))


# World groups (plans/worlds/I8 §3.1) ------------------------------------------------------------------------

# the group of a stratum whose world the ``worlds`` table does not list (shown last, never pooled with another)
UNLISTED = "unlisted"
UNLISTED_TITLE = "Worlds not in the worlds table"


@dataclass(frozen=True, slots=True)
class WorldGroup:
    """One display group: its key (``worlds.World.group``; None for an export without a ``worlds`` table, whose
    one group holds every stratum in study order), its title, its strata in display order and its worlds in that
    order, including worlds with final runs but no reported one (``Study.empty_cells``)."""

    key: str | None
    title: str
    strata: tuple[Stratum, ...]
    worlds: tuple[str, ...]


def world_groups(study: Study) -> tuple[WorldGroup, ...]:
    """The study's strata by world group in ``GROUP_ORDER`` (other keys after it, by name; ``UNLISTED`` last),
    each group's strata by the ``worlds`` table's row order (the registry order), then in study order; one group
    of every stratum in study order for an export without the table (export version 1)."""
    listed = worlds(Reads(study.export))
    every = list(
        dict.fromkeys([*(s.world_id for s in study.strata), *(c[1] for c in study.empty_cells)])
    )
    if listed is None:
        return (WorldGroup(None, "", study.strata, tuple(every)),)
    rank = {world: index for index, world in enumerate(listed)}
    members: dict[str, list[str]] = defaultdict(list)
    for world in sorted(every, key=lambda w: (rank.get(w, len(rank)), w)):
        found = listed.get(world)
        members[found.group if found is not None else UNLISTED].append(world)
    keys = [
        *(key for key in GROUP_ORDER if key in members),
        *sorted(key for key in members if key not in GROUP_ORDER and key != UNLISTED),
        *([UNLISTED] if UNLISTED in members else []),
    ]
    return tuple(
        WorldGroup(
            key,
            UNLISTED_TITLE if key == UNLISTED else GROUP_TITLES.get(key, key),
            tuple(s for world in members[key] for s in study.strata if s.world_id == world),
            tuple(members[key]),
        )
        for key in keys
    )
