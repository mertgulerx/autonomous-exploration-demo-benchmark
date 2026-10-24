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

"""Registry facts for ``analysis_block.validate(worlds=...)``: the one builder used at submission
(``Library.world_facts``) and at export (``exports``), so a plan block is judged the same way at both."""

import json
from typing import Any

from .analysis_block import WorldFacts
from .contracts import ContractError


def world_facts(registry: Any) -> dict[str, WorldFacts]:
    """The registry's runnable worlds (``WorldRegistry.runnable()``, the worlds a campaign can name) by ID.

    - ``tier`` and ``family`` (``World.family``: the family of tier F, else None);
    - ``parameters``: the ``world.yaml`` parameters of a generated world (``worlds_table``'s design point,
      ``{}`` for named worlds); a world.yaml changed since the registry was loaded is refused;
    - ``spawns``: the spawn IDs a campaign can run. A collection world runs exactly its predefined registry
      spawns; a tier-A world's set is its registry default spawn ID followed by its registry spawns (custom
      spawns are campaign-specific and not registry facts).
    """
    try:
        rows = {row["world_id"]: row for row in registry.worlds_table()}
    except (OSError, ValueError) as exc:
        raise ContractError("worlds", str(exc), "dependency_or_configuration") from exc
    facts: dict[str, WorldFacts] = {}
    for world in registry.runnable():
        predefined = tuple(str(spawn.id) for spawn in world.spawns)
        facts[str(world.id)] = WorldFacts(
            str(world.tier),
            world.family,
            json.loads(rows[world.id]["design_point_json"]),
            predefined
            if world.spawn_policy == "predefined"
            else (world.default_spawn.id, *predefined),
        )
    return facts
