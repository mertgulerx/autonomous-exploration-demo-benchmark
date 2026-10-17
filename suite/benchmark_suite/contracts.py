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
from collections.abc import Iterable
from enum import StrEnum
import hashlib
import json
import math
import re
from typing import Any, Mapping

PROTOCOL_VERSION = 1
DATASET_VERSION = 5
GAZEBO_SEED = 1234
DEFAULT_LIMIT = 1800.0
NO_MOVEMENT_LIMIT = 180.0
HIDDEN_GROUPS = frozenset({"Suppression", "QoS", "Visualization", "Benchmark Integration"})
# Most campaigns or profiles one deletion request may name; the GUI checks the same bounds.
CAMPAIGN_DELETION_LIMIT = 100
PROFILE_DELETION_LIMIT = 250
# Campaign reports draw figures and statistics from these final outcomes only; every report states how many
# runs it excluded and why (docs/MEASUREMENTS.md, "Reported population").
REPORT_POLICY_VERSION = "reported-outcomes-v1"
REPORTED_OUTCOMES = frozenset({"completed", "time_limit", "no_movement"})
STOPPED_OUTCOMES = frozenset({"manual_stop"})


def report_exclusion(outcome: str | None) -> str | None:
    """None when campaign reports include the outcome, otherwise "stopped" or "failure".

    reports.py applies the same rule in SQL (the ``reported`` view and the ``report_exclusions`` reasons).
    """
    if outcome in REPORTED_OUTCOMES:
        return None
    return "stopped" if outcome in STOPPED_OUTCOMES else "failure"


class CampaignState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    PAUSING = "PAUSING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    BLOCKED = "BLOCKED"
    REPORTING = "REPORTING"
    FINISHED = "FINISHED"


class ContractError(ValueError):
    def __init__(self, field: str, message: str, code: str = "invalid_configuration"):
        super().__init__(f"{field}: {message}")
        self.field, self.message, self.code = field, message, code

    def dictionary(self) -> dict[str, str]:
        return {"code": self.code, "field": self.field, "message": self.message}


def canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("configuration", "Use finite, JSON-compatible typed values") from exc


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ContractError(field, "Expected an object with string keys")
    return value


def known_keys(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise ContractError(field, "Unknown fields: " + ", ".join(sorted(unexpected)))


def text(value: Any, field: str, *, maximum: int = 160) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ContractError(field, f"Expected nonempty text of at most {maximum} characters")
    if maximum <= 160 and any(c in value for c in ("\n", "\r", "\x00")):
        raise ContractError(field, "Use a single line of text")
    return value.strip()


def number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError(field, "Expected a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ContractError(field, f"Minimum is {minimum:g}")
    return result


_SPAWN_NUMBER = re.compile(r"spawn\s+([0-9]+)", re.IGNORECASE | re.ASCII)


def next_spawn_name(names: Iterable[str]) -> str:
    """Default name for a new spawn of one world: one more than its highest "Spawn N".

    The world's built-in Default pose counts as Spawn 1, so the first new spawn is "Spawn 2". Every recorded
    name of the world counts (old revisions and archived spawns too); other names are ignored.
    """
    numbers = [
        int(match.group(1))
        for name in names
        if (match := _SPAWN_NUMBER.fullmatch(name.strip())) is not None
    ]
    return f"Spawn {max([1, *numbers]) + 1}"


def integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ContractError(field, f"Expected an integer between {minimum} and {maximum}")
    return value


def resolve_world_time_limits(
    value: Mapping[str, Any], worlds: Iterable[str]
) -> tuple[tuple[str, float], ...]:
    identities = set(worlds)
    if "world_time_limits" not in value:
        limit = number(value.get("time_limit", DEFAULT_LIMIT), "time_limit", minimum=0)
        return tuple((world, limit) for world in sorted(identities))
    if "time_limit" in value:
        raise ContractError("world_time_limits", "Use per-world limits without a global time_limit")
    limits = mapping(value["world_time_limits"], "world_time_limits")
    unknown, missing = set(limits) - identities, identities - set(limits)
    if unknown:
        raise ContractError(
            "world_time_limits", "Worlds are not selected: " + ", ".join(sorted(unknown))
        )
    if missing:
        raise ContractError("world_time_limits", "Set a limit for: " + ", ".join(sorted(missing)))
    return tuple(
        (world, number(limits[world], "world_time_limits." + world, minimum=0))
        for world in sorted(identities)
    )


@dataclass(frozen=True, slots=True)
class Profile:
    id: str
    name: str
    robot_id: str
    snapshot_json: str

    @classmethod
    def accepted(cls, robot_id: str, value: dict[str, Any]) -> "Profile":
        name = text(value.get("name", "Default"), "profile.name")
        identity = text(value.get("id", content_hash(value)), "profile.id")
        mapping(value.get("effective", {}), "profile.effective")
        mapping(value.get("requested", {}), "profile.requested")
        return cls(identity, name, robot_id, canonical(value))

    def dictionary(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.snapshot_json)
        return result


@dataclass(frozen=True, slots=True)
class Spawn:
    id: str
    name: str
    world_id: str
    x: float
    y: float
    yaw: float

    @classmethod
    def accepted(cls, world_id: str, value: dict[str, Any]) -> "Spawn":
        known_keys(value, {"id", "name", "x", "y", "yaw", "revision", "hash"}, "spawn")
        return cls(
            text(value.get("id", content_hash(value)), "spawn.id"),
            text(value.get("name", "Default"), "spawn.name"),
            world_id,
            number(value.get("x"), "spawn.x"),
            number(value.get("y"), "spawn.y"),
            number(value.get("yaw"), "spawn.yaw"),
        )

    def dictionary(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "x": self.x, "y": self.y, "yaw": self.yaw}


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    name: str
    description: str
    robots: tuple[tuple[str, tuple[Profile, ...]], ...]
    worlds: tuple[tuple[str, tuple[Spawn, ...]], ...]
    repetitions: int
    concurrency: int
    world_time_limits: tuple[tuple[str, float], ...]
    exclusions: tuple[tuple[str, str, str], ...]
    source_campaign_id: str | None = None

    @classmethod
    def accepted(cls, value: dict[str, Any]) -> "CampaignSpec":
        known_keys(
            value,
            {
                "name",
                "description",
                "robots",
                "worlds",
                "repetitions",
                "concurrency",
                "time_limit",
                "world_time_limits",
                "exclusions",
                "source_campaign_id",
            },
            "campaign",
        )
        robots, worlds = [], []
        for identity, entries in mapping(value.get("robots"), "robots").items():
            if not isinstance(entries, list) or not entries:
                raise ContractError("robots." + identity, "Select at least one profile")
            profiles = tuple(Profile.accepted(identity, mapping(p, "profile")) for p in entries)
            if len({p.id for p in profiles}) != len(profiles):
                raise ContractError("robots." + identity, "Duplicate profile revisions")
            robots.append((text(identity, "robot_id"), profiles))
        for identity, entries in mapping(value.get("worlds"), "worlds").items():
            if not isinstance(entries, list) or not entries:
                raise ContractError("worlds." + identity, "Select at least one spawn")
            spawns = tuple(Spawn.accepted(identity, mapping(p, "spawn")) for p in entries)
            if len({p.id for p in spawns}) != len(spawns):
                raise ContractError("worlds." + identity, "Duplicate spawn revisions")
            worlds.append((text(identity, "world_id"), spawns))
        if not robots or not worlds:
            raise ContractError("campaign", "Select at least one robot and one world")
        exclusions = value.get("exclusions", [])
        if not isinstance(exclusions, list):
            raise ContractError("exclusions", "Expected robot/world/spawn triples")
        valid = {(r, w, s.id) for r, _ in robots for w, spawns in worlds for s in spawns}
        excluded = []
        for entry in exclusions:
            if (
                not isinstance(entry, list)
                or len(entry) != 3
                or any(not isinstance(identity, str) for identity in entry)
                or tuple(entry) not in valid
            ):
                raise ContractError(
                    "exclusions", "Each exclusion must match a selected combination"
                )
            excluded.append(tuple(entry))
        if len(set(excluded)) != len(excluded):
            raise ContractError("exclusions", "Duplicate exclusion")
        description = value.get("description", "")
        if not isinstance(description, str) or len(description) > 10000:
            raise ContractError("description", "Use at most 10000 characters")
        spec = cls(
            text(value.get("name"), "name"),
            description,
            tuple(sorted(robots)),
            tuple(sorted(worlds)),
            integer(value.get("repetitions", 1), "repetitions", 1, 2**31 - 1),
            integer(value.get("concurrency", 1), "concurrency", 1, 4),
            resolve_world_time_limits(value, (world for world, _ in worlds)),
            tuple(sorted(excluded)),
            text(value["source_campaign_id"], "source_campaign_id")
            if value.get("source_campaign_id") is not None
            else None,
        )
        if not spec.count:
            raise ContractError("exclusions", "All selected combinations are excluded")
        return spec

    @property
    def count(self) -> int:
        excluded = set(self.exclusions)
        return sum(
            len(profiles) * self.repetitions
            for robot, profiles in self.robots
            for world, spawns in self.worlds
            for spawn in spawns
            if (robot, world, spawn.id) not in excluded
        )

    def dictionary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "robots": {r: [p.dictionary() for p in ps] for r, ps in self.robots},
            "worlds": {w: [s.dictionary() for s in ss] for w, ss in self.worlds},
            "repetitions": self.repetitions,
            "concurrency": self.concurrency,
            "world_time_limits": dict(self.world_time_limits),
            "exclusions": [list(e) for e in self.exclusions],
            **({"source_campaign_id": self.source_campaign_id} if self.source_campaign_id else {}),
        }
