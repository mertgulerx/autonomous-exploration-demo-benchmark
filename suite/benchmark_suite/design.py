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

from dataclasses import dataclass
from collections.abc import Iterator
import hashlib
import uuid

from .contracts import CampaignSpec, Profile, Spawn, canonical

ORDERING_VERSION = "hashed-blocks-rotating-profiles-v1"


@dataclass(frozen=True, slots=True)
class Experiment:
    id: str
    campaign_id: str
    position: int
    block_id: str
    robot_id: str
    world_id: str
    profile: Profile
    spawn: Spawn
    repetition: int
    time_limit: float

    def dictionary(self) -> dict[str, object]:
        return {
            "id": self.id,
            "campaign_id": self.campaign_id,
            "position": self.position,
            "block_id": self.block_id,
            "robot_id": self.robot_id,
            "world_id": self.world_id,
            "profile": self.profile.dictionary(),
            "spawn": self.spawn.dictionary(),
            "repetition": self.repetition,
            "time_limit": self.time_limit,
        }


def experiments(spec: CampaignSpec, campaign_id: str) -> Iterator[Experiment]:
    namespace = uuid.UUID(campaign_id)
    excluded = set(spec.exclusions)
    limits = dict(spec.world_time_limits)
    blocks = []
    for robot, profiles in spec.robots:
        for world, spawns in spec.worlds:
            for spawn in spawns:
                if (robot, world, spawn.id) in excluded:
                    continue
                for repetition in range(spec.repetitions):
                    identity = canonical([robot, world, spawn.id, repetition])
                    key = hashlib.sha256((ORDERING_VERSION + identity).encode()).digest()
                    blocks.append((key, robot, world, spawn, repetition, profiles, identity))
    blocks.sort(key=lambda item: item[0])
    position = 0
    for _, robot, world, spawn, repetition, profiles, identity in blocks:
        block_id = str(uuid.uuid5(namespace, identity))
        shift = repetition % len(profiles)
        rotated = profiles[shift:] + profiles[:shift]
        for profile in rotated:
            yield Experiment(
                str(uuid.uuid5(namespace, block_id + ":" + profile.id)),
                campaign_id,
                position,
                block_id,
                robot,
                world,
                profile,
                spawn,
                repetition,
                limits[world],
            )
            position += 1
