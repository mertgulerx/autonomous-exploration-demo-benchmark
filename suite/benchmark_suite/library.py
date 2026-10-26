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

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any
import uuid

import yaml

from exploration_benchmark_runner.parameters import ParameterSchema, UniqueKeyLoader
from exploration_benchmark_runner.configuration import robot_parameter_schema, scenario
from exploration_benchmark_runner.coverage import Reference
from exploration_benchmark_runner.time_limits import (
    RULE_METRIC_VERSION,
    TIME_LIMIT_RULE,
    kinematic_time_limit,
    target_area_m2,
)
from exploration_benchmark_runner.worlds import WorldRegistry
from exploration_benchmark_robots.catalog import RobotCatalog, fill_robot_profiles
from exploration_benchmark_robots.preflight import validate_spawn

from .analysis_block import WorldFacts, plan_parts, validate
from .contracts import (
    CampaignSpec,
    ContractError,
    DEFAULT_LIMIT,
    PROFILE_DELETION_LIMIT,
    RuleValue,
    Study,
    canonical,
    content_hash,
    integer,
    known_keys,
    mapping,
    number,
    text,
)
from .monitoring import png, reference_pixels
from .storage import file_hash
from .store import TUNING_CAMPAIGN, Store
from .world_facts import world_facts

LOG = logging.getLogger(__name__)

PROFILE_FIELDS = {
    "id",
    "name",
    "robot_id",
    "explorer",
    "requested",
    "overrides",
    "yaml_text",
    "description",
    "effective",
    "robot",
    "baseline_hash",
    "metadata_hash",
    "hash",
    "family_id",
    "created_ns",
}
# Retired keys accepted from recorded snapshots and earlier YAML, then ignored (tags, 16 October 2026;
# library revision numbers, 18 October 2026; robot settings, 25 October 2026, whose non-empty
# ``robot_overrides`` ``resolve_profile`` refuses before the drop).
LEGACY_PROFILE_FIELDS = frozenset({"tags", "revision", "robot_overrides"})
# Full coverage references (grids and target cells) kept in memory; the catalog needs only their
# metadata, and a large world's reference holds millions of cells (I1 §3.6).
REFERENCE_CACHE = 4
# Spawn names that stand for a world's Default spawn (tier A) or its whole predefined set (collection
# worlds, I9 §3.4), besides "default-<world>" and the default spawn's own ID.
DEFAULT_SPAWN_NAMES = ("Default", "default")
# The explorer launch profiles (docs/plans/EXTERNAL_EXPLORERS.md): a suite profile's ``explorer`` is one of
# their IDs. A profile without the key runs DEFAULT_EXPLORER (frontier_exploration_ros2); resolved profiles of
# it leave the key out, so profiles saved before external explorers keep their content, hash and ID.
PROFILES_FILE = "exploration_benchmark_runner/config/profiles.yaml"
DEFAULT_EXPLORER = "default"


def world_code(registry: Any, world: Any) -> str | None:
    """A collection world's display code ("SC2a", "F1-01", "P1", "MO2"): the ``id`` of its world.yaml,
    whose bytes must still match the registry's pin. None for tier A (no world.yaml; the label names it)."""
    if world.source is None:
        return None
    path = registry.path(world.source.world_yaml)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != world.source.world_yaml_sha256:
        raise ValueError(f"{world.source.world_yaml}: changed since the registry was loaded")
    document = yaml.load(data, Loader=UniqueKeyLoader)
    code = document.get("id") if isinstance(document, dict) else None
    if not isinstance(code, str) or not code:
        raise ValueError(f"{world.source.world_yaml}: expected a text id")
    return code


def registry_spawn(spawn: Any) -> dict[str, Any]:
    """A registry spawn as campaigns and the catalog hold it."""
    return {"id": spawn.id, "name": spawn.name, "x": spawn.x, "y": spawn.y, "yaw": spawn.yaw}


def check_registry_spawn(
    field: str, world_id: str, item: dict[str, Any], expected: dict[str, Any], rule: str
) -> None:
    """Refuse a campaign spawn that names a registry spawn's ID but not its pose (``x``, ``y``, ``yaw``
    numerically equal: an int 0 equals 0.0) or, when it gives one, its name. ``rule`` ends the message."""
    edited = [
        key
        for key in ("x", "y", "yaw")
        if type(item.get(key)) not in (int, float) or item[key] != expected[key]
    ]
    if item.get("name", expected["name"]) != expected["name"]:
        edited.append("name")
    if edited:
        raise ContractError(
            field,
            f"{world_id}: spawn {expected['id']} differs from its registry {', '.join(edited)}; {rule}",
        )


def incompatible(robot_id: str, world_id: str) -> ContractError:
    return ContractError(
        "compatibility",
        f"{robot_id} cannot use {world_id}; explicitly exclude this combination",
    )


@dataclass(frozen=True, slots=True)
class Explorer:
    """One explorer launch profile of ``profiles.yaml`` with its parameter schema: the Default file (relative to
    the checkout), the metadata (beside profiles.yaml) and their hashes, which stored profiles must match."""

    id: str
    label: str
    framework: str
    package: str
    insight: bool
    robot_defaults: bool
    source: dict[str, Any]
    entry: dict[str, Any]
    schema: ParameterSchema
    baseline_hash: str
    metadata_hash: str
    groups: tuple[str, ...]

    @classmethod
    def load(cls, root: Path, identity: str, entry: dict[str, Any]) -> "Explorer":
        baseline = root / text(entry.get("parameters"), f"profiles.{identity}.parameters")
        metadata = (root / PROFILES_FILE).parent / text(
            entry.get("metadata"), f"profiles.{identity}.metadata"
        )
        node = text(entry.get("node"), f"profiles.{identity}.node")
        try:
            schema = ParameterSchema(baseline, metadata, node)
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"Explorer {identity}: invalid parameter schema: {exc}") from exc
        groups: list[str] = []
        for information in schema.metadata.values():
            group = information.get("group", "") if isinstance(information, dict) else ""
            if isinstance(group, str) and group and group not in groups:
                groups.append(group)
        source = entry.get("source")
        return cls(
            id=identity,
            label=text(entry.get("label", identity), f"profiles.{identity}.label"),
            framework=text(entry.get("framework", identity), f"profiles.{identity}.framework"),
            package=text(entry.get("package"), f"profiles.{identity}.package"),
            insight=entry.get("insight") is True,
            robot_defaults=entry.get("robot_defaults") is True,
            source=dict(source) if isinstance(source, dict) else {},
            entry=deepcopy(entry),
            schema=schema,
            baseline_hash=file_hash(baseline),
            metadata_hash=file_hash(metadata),
            groups=tuple(groups),
        )

    def default_id(self, robot_id: str) -> str:
        """The ID of the robot's Default profile of this explorer: "default-<robot>" for frontier_exploration_ros2
        (unchanged), "default-<explorer>-<robot>" for the others."""
        if self.id == DEFAULT_EXPLORER:
            return "default-" + robot_id
        return f"default-{self.id}-{robot_id}"

    def default_name(self) -> str:
        return "Default" if self.id == DEFAULT_EXPLORER else f"{self.label} Default"


def load_explorers(root: Path) -> dict[str, Explorer]:
    """Every explorer launch profile of the checkout's profiles.yaml, in file order (DEFAULT_EXPLORER first)."""
    document = mapping(
        yaml.load((root / PROFILES_FILE).read_text(), Loader=UniqueKeyLoader), "profiles"
    )
    entries = mapping(document.get("profiles"), "profiles")
    if DEFAULT_EXPLORER not in entries:
        raise ValueError(f"{PROFILES_FILE} has no {DEFAULT_EXPLORER} explorer profile")
    explorers = {
        identity: Explorer.load(root, identity, mapping(entry, "profiles." + identity))
        for identity, entry in entries.items()
    }
    return {DEFAULT_EXPLORER: explorers.pop(DEFAULT_EXPLORER), **explorers}


def load_yaml(source: str) -> dict[str, Any]:
    try:
        return mapping(yaml.load(source, Loader=UniqueKeyLoader), "yaml")
    except yaml.YAMLError as exc:
        raise ContractError("yaml", str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ReferenceHeader:
    """A coverage reference's metadata without its grid: what the catalog shows. ``descriptor`` equals
    ``Reference.descriptor`` (``Reference`` checks that its target count equals ``target_cells``)."""

    file: Path
    metadata: dict[str, Any]
    metadata_hash: str

    @classmethod
    def load(cls, file: Path) -> "ReferenceHeader":
        try:
            data = file.read_bytes()
            metadata = json.loads(data)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid coverage reference {file}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid coverage reference {file}: expected a mapping")
        return cls(file, metadata, hashlib.sha256(data).hexdigest())

    @property
    def origin(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.metadata["origin"])

    @property
    def resolution(self) -> float:
        return float(self.metadata["resolution"])

    def descriptor(self) -> dict[str, Any]:
        return dict(
            id=self.metadata["id"],
            metric_version=self.metadata["metric_version"],
            metadata_file=str(self.file.resolve()),
            metadata_sha256=self.metadata_hash,
            grid_sha256=self.metadata["grid_sha256"],
            target_area_m2=self.metadata["target_cells"] * self.resolution**2,
        )


class Library:
    def __init__(self, root: Path, *, native: bool = True):
        self.root, self.native = root.resolve(), native
        self.store: Store | None = None
        # One parameter schema per explorer (profiles.yaml); frontier_exploration_ros2's stays the library's
        # ``schema`` and hash pair, so its stored profiles keep validating unchanged.
        self.explorers = load_explorers(self.root)
        ours = self.explorers[DEFAULT_EXPLORER]
        self.schema = ours.schema
        self.baseline_hash, self.metadata_hash = ours.baseline_hash, ours.metadata_hash
        # The world registry (worlds.yaml) of this checkout, read once at engine start.
        self.registry = WorldRegistry.load(self.root)
        resource = self.root / "exploration_benchmark_robots"
        self.robot_catalog = RobotCatalog(resource, worlds=self.registry) if native else None
        # Robot profiles with world fit from the registry, as RobotCatalog fills them.
        catalog = yaml.safe_load((resource / "config/robots.yaml").read_text())
        self.robots: dict[str, dict[str, Any]] = (
            self.robot_catalog.profiles
            if self.robot_catalog
            else fill_robot_profiles(catalog, self.registry)
        )
        # The catalog's default robot: the robot of requests that name none.
        self.default_robot = text(catalog.get("default_robot"), "default_robot")
        if self.default_robot not in self.robots:
            raise ContractError(
                "default_robot", "The catalog's default robot is not a catalog robot"
            )
        self.accepted_robots: dict[str, dict[str, Any]] = {}
        self.worlds = {world.id: world for world in self.registry.runnable()}
        self.headers = {
            identity: ReferenceHeader.load(world.reference.metadata_file)
            for identity, world in self.worlds.items()
        }
        # The registry checked each pin a moment ago; this guards the read above against an edit between.
        for identity, header in self.headers.items():
            if header.metadata_hash != self.worlds[identity].reference.metadata_sha256:
                raise ValueError(f"Coverage reference of {identity} changed while loading")
        self.loaded: OrderedDict[str, Reference] = OrderedDict()
        self.scenarios = {
            identity: scenario(self.root, identity, registry=self.registry)
            for identity in self.worlds
        }
        # spawns.map replies per world; the references never change after loading.
        self.maps: dict[str, dict[str, Any]] = {}
        # Registry facts for analysis-block validation, read once on first use (world.yaml parameters).
        self.facts: dict[str, WorldFacts] | None = None
        self.codes = {
            identity: world_code(self.registry, world) for identity, world in self.worlds.items()
        }

    def reference(self, world: str) -> Reference:
        """The world's full coverage reference, loaded on demand (bounded LRU, REFERENCE_CACHE)."""
        loaded = self.loaded.pop(world, None)
        if loaded is None:
            header = self.headers[world]
            try:
                loaded = Reference(header.file)
            except ValueError as exc:
                raise ContractError("reference", str(exc), "reference_mismatch") from exc
            if loaded.metadata_hash != header.metadata_hash:
                raise ContractError(
                    "reference",
                    f"Coverage reference of {world} changed since engine start; restart the engine",
                    "reference_mismatch",
                )
        self.loaded[world] = loaded
        while len(self.loaded) > REFERENCE_CACHE:
            self.loaded.popitem(last=False)
        return loaded

    def rule_value(self, world_id: str, robot_id: str) -> RuleValue | None:
        """``kinematic-time-limit-v1`` for one world and robot: A from the world's pinned reference
        (the rule module's ``target_area_m2``, as the tool computes it), R and v from the robot's Default
        catalog profile (``lidar.range_max``, ``max_linear_speed``). None without a reference of the
        rule's metric version (``RULE_METRIC_VERSION``)."""
        header = self.headers[world_id]
        if self.worlds[world_id].reference.metric_version != RULE_METRIC_VERSION:
            return None
        profile = self.robots[robot_id]
        cells, resolution = header.metadata["target_cells"], header.metadata["resolution"]
        range_m, speed = profile["lidar"]["range_max"], profile["max_linear_speed"]
        try:
            limit = kinematic_time_limit(cells, resolution, range_m, speed)
            area = float(target_area_m2(cells, resolution))
        except ValueError as exc:
            raise ContractError(
                f"time_limit_rule.{world_id}.{robot_id}", str(exc), "dependency_or_configuration"
            ) from exc
        return RuleValue(
            seconds=int(limit.seconds),
            uncapped_seconds=int(limit.uncapped_seconds),
            capped=bool(limit.capped),
            area_m2=area,
            range_m=float(range_m),
            speed_mps=float(speed),
        )

    def rule_time_limits(self, world_id: str) -> dict[str, RuleValue]:
        """The catalog's ``rule_time_limits`` of a world: each robot the registry fits to it, in registry
        order. A robot whose value cannot be computed is left out and logged, not the whole catalog."""
        result: dict[str, RuleValue] = {}
        for robot_id in self.worlds[world_id].robot_ids:
            try:
                value = self.rule_value(world_id, robot_id)
            except (ContractError, KeyError, TypeError) as exc:
                LOG.warning("No rule time limit for %s in %s: %s", robot_id, world_id, exc)
                continue
            if value is not None:
                result[robot_id] = value
        return result

    def world_facts(self) -> dict[str, WorldFacts]:
        """Registry facts for analysis-block validation (``world_facts.world_facts``), built once."""
        if self.facts is None:
            self.facts = world_facts(self.registry)
        return self.facts

    def explorer_ids(self) -> dict[str, str]:
        """Each explorer's run-record ``explorer_id`` (its package) by its profiles.yaml ID."""
        return {identity: explorer.package for identity, explorer in self.explorers.items()}

    def world_families(self) -> dict[str, str | None]:
        """Each registry world's run-record ``world_family`` (the family of tier F, else null)."""
        return {world.id: world.family for world in self.registry.worlds}

    def bind(self, store: Store) -> None:
        self.store = store

    def robot(self, identity: str) -> dict[str, Any]:
        if identity not in self.robots:
            raise ContractError("robot_id", "Unknown robot: " + identity)
        try:
            if identity not in self.accepted_robots:
                fit = self.robots[identity]["worlds"]
                if not fit:
                    raise ValueError(f"{identity} fits no world of the registry")
                self.accepted_robots[identity] = (
                    self.robot_catalog.require(identity, fit[0])
                    if self.robot_catalog
                    else deepcopy(self.robots[identity])
                )
            return mapping(deepcopy(self.accepted_robots[identity]), "robot")
        except ValueError as exc:
            raise ContractError("robot_id", str(exc), "dependency_or_configuration") from exc

    def resolve_profile(self, robot_id: str, value: dict[str, Any]) -> dict[str, Any]:
        if mapping(value.get("robot_overrides", {}), "robot_overrides"):
            raise ContractError(
                "robot_overrides",
                "Robot settings were removed; robots run with their catalog limits",
            )
        value = {k: v for k, v in value.items() if k not in LEGACY_PROFILE_FIELDS}
        known_keys(value, PROFILE_FIELDS, "profile")
        if value.get("robot_id", robot_id) != robot_id:
            raise ContractError("robot_id", "Profile belongs to another robot")
        explorer = self.explorer(value.get("explorer", DEFAULT_EXPLORER))
        if (
            value.get("baseline_hash", explorer.baseline_hash) != explorer.baseline_hash
            or value.get("metadata_hash", explorer.metadata_hash) != explorer.metadata_hash
        ):
            raise ContractError(
                "profile",
                "Stored profile uses different defaults or metadata; import its explorer YAML to save it as a new profile",
            )
        requested = mapping(value.get("requested", value.get("overrides", {})), "overrides")
        robot = self.robot(robot_id)
        schema = robot_parameter_schema(explorer.schema, robot, explorer.entry)
        imported = value.get("yaml_text", "")
        if not isinstance(imported, str):
            raise ContractError("yaml_text", "Expected YAML text")
        if len(imported.encode()) > 128 * 1024:
            raise ContractError("yaml_text", "Explorer YAML exceeds 128 KiB")
        try:
            if imported:
                requested = {**schema.import_yaml(imported), **requested}
            effective = schema.resolve(overrides=requested)
        except ValueError as exc:
            raise ContractError("parameters", str(exc)) from exc
        result = {
            "name": text(value.get("name", explorer.default_name()), "profile.name"),
            "robot_id": robot_id,
            "requested": deepcopy(requested),
            "effective": effective,
            "robot": robot,
            "baseline_hash": explorer.baseline_hash,
            "metadata_hash": explorer.metadata_hash,
            "yaml_text": imported,
            "description": value.get("description", ""),
        }
        if explorer.id != DEFAULT_EXPLORER:
            # in the content and hash; absent for frontier_exploration_ros2 (DEFAULT_EXPLORER)
            result["explorer"] = explorer.id
        if not isinstance(result["description"], str):
            raise ContractError("description", "Description must be text")
        if len(result["description"]) > 4096:
            raise ContractError("description", "Use a description up to 4096 characters")
        if "family_id" in value:
            result["family_id"] = text(value["family_id"], "family_id")
        if "created_ns" in value:
            result["created_ns"] = integer(value["created_ns"], "created_ns", 1, 2**63 - 1)
        result["hash"] = content_hash(result)
        result["id"] = text(
            value.get("id", explorer.default_id(robot_id) if not requested else result["hash"]),
            "profile.id",
        )
        return result

    def explorer(self, identity: Any) -> Explorer:
        """The explorer a profile names (``explorer``, a profiles.yaml ID)."""
        if not isinstance(identity, str) or identity not in self.explorers:
            raise ContractError(
                "explorer",
                f"Unknown explorer {identity!r}; choose {', '.join(self.explorers)}",
            )
        return self.explorers[identity]

    def default_reference(self, robot_id: str, item: str) -> dict[str, Any] | None:
        """A campaign's shorthand for a robot's Default profile of an explorer, as the profile to resolve:
        "Default", "default" or "default-<robot>" (frontier_exploration_ros2) and "default-<explorer>-<robot>".
        None for any other text (a saved profile's ID)."""
        for explorer in self.explorers.values():
            names = {explorer.default_id(robot_id)}
            if explorer.id == DEFAULT_EXPLORER:
                names |= {"Default", "default"}
            if item in names:
                reference = {"name": explorer.default_name(), "id": explorer.default_id(robot_id)}
                if explorer.id != DEFAULT_EXPLORER:
                    reference["explorer"] = explorer.id
                return reference
        return None

    def explorer_catalog(self) -> list[dict[str, Any]]:
        """The catalog's ``explorers``: per explorer its identity, capabilities, source, parameter metadata and
        groups, hashes, and the Default values per catalog robot (robot-derived where ``robot_defaults``)."""
        result = []
        for explorer in self.explorers.values():
            defaults = {
                identity: robot_parameter_schema(explorer.schema, profile, explorer.entry).defaults
                for identity, profile in self.robots.items()
            }
            result.append(
                {
                    "id": explorer.id,
                    "label": explorer.label,
                    "framework": explorer.framework,
                    "package": explorer.package,
                    "node": explorer.schema.node,
                    "insight": explorer.insight,
                    "robot_defaults": explorer.robot_defaults,
                    "source": deepcopy(explorer.source),
                    "parameters": deepcopy(explorer.schema.metadata),
                    "groups": list(explorer.groups),
                    "defaults": defaults,
                    "baseline_hash": explorer.baseline_hash,
                    "metadata_hash": explorer.metadata_hash,
                }
            )
        return result

    def catalog(self) -> dict[str, Any]:
        robots = []
        for identity, profile in self.robots.items():
            available, reason = (
                self.robot_catalog.availability(identity)
                if self.robot_catalog
                else (True, "Synthetic execution")
            )
            defaults = robot_parameter_schema(self.schema, profile).defaults
            robots.append(
                {
                    "id": identity,
                    "label": profile["label"],
                    "available": available,
                    "reason": reason,
                    "lidar_range": profile["lidar"]["range_max"],
                    "movement_speed": profile["max_linear_speed"],
                    "turning_speed": profile["max_angular_speed"],
                    "footprint": profile["footprint"],
                    "kinematics": profile["kinematics"],
                    "physics_step": profile["physics_step"],
                    "worlds": profile["worlds"],
                    "defaults": defaults,
                }
            )
        worlds = []
        for identity, world in self.worlds.items():
            header = self.headers[identity]
            x, y, _, yaw = self.scenarios[identity]["spawn"]
            worlds.append(
                {
                    "id": identity,
                    "label": world.label,
                    # display code of a collection world ("F1-01"); null for tier A
                    "code": self.codes[identity],
                    # Registry facts the world selection groups by (I9): tier A keeps custom spawns,
                    # collection worlds run all their predefined spawns.
                    "tier": world.tier,
                    "family": world.family,
                    "role": world.role,
                    "spawn_policy": world.spawn_policy,
                    "default_time_limit": DEFAULT_LIMIT,
                    # time_limit_rule's value per robot that fits the world (I4 §3.3); {} without a
                    # metric-version-2 reference.
                    "rule_time_limits": self.rule_time_limits(identity),
                    # Registry spawns in registry order: all of them run on a collection world (I9
                    # §3.1); fixed choices beside Default and custom spawns on tier A (decision 60).
                    "spawns": [registry_spawn(s) for s in world.spawns],
                    "reference": header.descriptor(),
                    "default_spawn": {
                        "id": world.default_spawn.id,
                        "name": world.default_spawn.name,
                        "x": x,
                        "y": y,
                        "yaw": yaw,
                    },
                    "origin": header.origin,
                    "resolution": header.resolution,
                    "width": header.metadata["width"],
                    "height": header.metadata["height"],
                }
            )
        return {
            "robots": robots,
            # The catalog's default robot (robots.yaml ``default_robot``), preselected by the GUI.
            "default_robot": self.default_robot,
            "worlds": worlds,
            # frontier_exploration_ros2's metadata and hashes (as before); every explorer's in ``explorers``
            "parameters": self.schema.metadata,
            "baseline_hash": self.baseline_hash,
            "metadata_hash": self.metadata_hash,
            "explorers": self.explorer_catalog(),
            "default_explorer": DEFAULT_EXPLORER,
            "source": "native" if self.native else "synthetic",
            "defaults": {"concurrency": 1},
        }

    def _saved(self, identity: str, kind: str) -> dict[str, Any]:
        """A saved profile or spawn by its ID. Rows saved before revisions were removed (18 October 2026)
        carry a ``revision`` number in their snapshot; it is dropped so new campaigns never record it."""
        if self.store is None:
            raise RuntimeError("Library is not attached to the engine")
        row = self.store.one("SELECT * FROM library WHERE id=? AND kind=?", (identity, kind))
        if not row:
            raise ContractError(
                "id",
                f"Saved {kind} {identity} does not exist; it may have been edited or deleted",
                "not_found",
            )
        snapshot = mapping(json.loads(row["snapshot"]), "snapshot")
        snapshot.pop("revision", None)
        return snapshot

    def resolve_campaign(self, raw: dict[str, Any], *, tuning: bool = False) -> CampaignSpec:
        """A submitted campaign as its accepted specification. ``tuning``: a tuning run, which picks one
        predefined spawn of a collection world instead of running all of them (I9 §8.1)."""
        allowed = {
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
            "study",
            "debug_streams",
            # an accepted campaign (Validate, then Add to queue; exported YAML) carries it: recomputed below
            "default_parameter_hashes",
            "time_limit_rule",
            # recomputed below with the rule; without it CampaignSpec refuses the key
            "time_limit_inputs",
        }
        known_keys(raw, allowed, "campaign")
        accepted = deepcopy(raw)
        accepted["robots"], accepted["worlds"] = {}, {}
        for robot_id, profiles in mapping(raw.get("robots"), "robots").items():
            if not isinstance(profiles, list) or not profiles:
                raise ContractError("robots." + robot_id, "Select at least one profile")
            selected = []
            for item in profiles:
                if isinstance(item, str):
                    default = self.default_reference(robot_id, item)
                    if default is not None:
                        item = default
                    else:
                        item = self._saved(item, "profile")
                        if item["robot_id"] != robot_id:
                            raise ContractError("profiles", "Profile belongs to another robot")
                value = mapping(item, "profile")
                selected.append(self.resolve_profile(robot_id, value))
            accepted["robots"][robot_id] = selected
        for world_id, spawns in mapping(raw.get("worlds"), "worlds").items():
            if world_id not in self.worlds:
                raise ContractError("worlds", "Unknown world: " + world_id)
            if self.worlds[world_id].spawn_policy == "predefined":
                accepted["worlds"][world_id] = self._predefined_spawns(world_id, spawns, tuning)
                continue
            if not isinstance(spawns, list) or not spawns:
                raise ContractError("worlds." + world_id, "Select at least one spawn")
            default = self.worlds[world_id].default_spawn
            # Registry spawns (decision 60) resolve by ID before saved spawns; a saved spawn's ID is a
            # UUID, so a registry ID wins the (unlikely) tie.
            fixed = {spawn.id: registry_spawn(spawn) for spawn in self.worlds[world_id].spawns}
            selected = []
            for item in spawns:
                if isinstance(item, str) and item in fixed:
                    item = dict(fixed[item])
                elif (
                    isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                    and item["id"] in fixed
                ):
                    # A registry spawn given in full must be the registry's, and then gives the same
                    # specification as its ID (as on collection worlds).
                    known_keys(
                        item, {"id", "name", "x", "y", "yaw"}, f"worlds.{world_id}.{item['id']}"
                    )
                    check_registry_spawn(
                        "worlds." + world_id,
                        world_id,
                        item,
                        fixed[item["id"]],
                        "registry spawns run unchanged; give an edited pose its own spawn ID",
                    )
                    item = dict(fixed[item["id"]])
                elif isinstance(item, str):
                    if item in (*DEFAULT_SPAWN_NAMES, "default-" + world_id, default.id):
                        x, y, _, yaw = self.scenarios[world_id]["spawn"]
                        item = {
                            "id": default.id,
                            "name": default.name,
                            "x": x,
                            "y": y,
                            "yaw": yaw,
                        }
                    else:
                        try:
                            item = self._saved(item, "spawn")
                        except ContractError as exc:
                            if exc.code != "not_found":
                                raise
                            raise ContractError(
                                "worlds." + world_id,
                                f"{world_id} has no registry spawn {item} "
                                f"({', '.join(fixed) or 'none'}) and no saved spawn with that ID",
                                "not_found",
                            ) from exc
                        if item["world_id"] != world_id:
                            raise ContractError("spawns", "Spawn belongs to another world")
                        item = {
                            k: v for k, v in item.items() if k in ("id", "name", "x", "y", "yaw")
                        }
                selected.append(mapping(item, "spawn"))
            accepted["worlds"][world_id] = selected
        # Facts fixed at submission: the plan's hash and each robot's Default parameter hash.
        if isinstance(raw.get("study"), dict):
            study = Study.accepted(raw["study"])
            digest = self.plan_hash(study)
            # an accepted campaign names the hash the researcher validated: a plan changed since is refused
            if study.plan_sha256 is not None and study.plan_sha256 != digest:
                raise ContractError(
                    "study.plan_sha256",
                    "The plan changed above its Deviations heading since the campaign was validated; "
                    "validate it again (in a campaign file, remove study.plan_sha256 first)",
                    "plan_changed",
                )
            accepted["study"] = dict(study.dictionary(), plan_sha256=digest)
        # The rule's values; CampaignSpec.accepted checks the rule name, the global time_limit, given
        # limits, capped pairs and robots that disagree (contracts.resolve_time_limit_inputs).
        if raw.get("time_limit_rule") == TIME_LIMIT_RULE:
            limits, inputs = self._rule_time_limits(raw, accepted)
            accepted.setdefault("world_time_limits", limits)
            accepted["time_limit_inputs"] = inputs
        accepted["default_parameter_hashes"] = {
            robot_id: content_hash(
                self.resolve_profile(robot_id, {"name": "Default", "id": "default-" + robot_id})[
                    "effective"
                ]
            )
            for robot_id in accepted["robots"]
        }
        spec = CampaignSpec.accepted(accepted)
        exclusions = set(spec.exclusions)
        # Worlds outermost: each world's full reference is loaded at most once here (the LRU holds only
        # REFERENCE_CACHE of them, and this runs on the engine's request thread).
        for world_id, spawns in spec.worlds:
            loaded: Reference | None = None
            if self.native:
                loaded = self.reference(world_id)
                try:
                    loaded.validate_scenario(
                        self.root, self.scenarios[world_id], allow_custom_spawn=True
                    )
                except ValueError as exc:
                    raise ContractError(
                        "worlds." + world_id, str(exc), "reference_mismatch"
                    ) from exc
            if self.worlds[world_id].spawn_policy == "predefined" and not tuning:
                # A collection world runs all its predefined spawns: an exclusion removes a robot from
                # the world, never single spawns (I9 §3.4).
                for robot_id, _ in spec.robots:
                    removed = {s.id for s in spawns if (robot_id, world_id, s.id) in exclusions}
                    if removed and len(removed) != len(spawns):
                        raise ContractError(
                            "exclusions",
                            f"{world_id} runs all its predefined spawns: exclude every one of them "
                            f"for {robot_id} or none",
                        )
            for robot_id, profiles in spec.robots:
                for spawn in spawns:
                    if (robot_id, world_id, spawn.id) in exclusions:
                        continue
                    for profile in profiles:
                        robot = profile.dictionary()["robot"]
                        if world_id not in robot["worlds"]:
                            raise incompatible(robot_id, world_id)
                        if loaded is None:
                            loaded = self.reference(world_id)
                        try:
                            validate_spawn(robot, [spawn.x, spawn.y, 0.0, spawn.yaw], loaded)
                        except ValueError as exc:
                            raise ContractError("spawns." + spawn.id, str(exc)) from exc
        return spec

    def _predefined_spawns(self, world_id: str, value: Any, tuning: bool) -> list[dict[str, Any]]:
        """A collection world's spawns (I9 §3.4): exactly its predefined registry spawns, ids and poses
        compared exactly; none given, or the Default shorthand alone, means all of them. A tuning run
        names one of them."""
        world = self.worlds[world_id]
        field = "worlds." + world_id
        predefined = {spawn.id: registry_spawn(spawn) for spawn in world.spawns}
        if not predefined:
            raise ContractError(field, f"{world_id} has no predefined spawns in the registry")
        names = ", ".join(predefined)
        only = (
            f"{world_id} runs its predefined spawn{'s' if len(predefined) > 1 else ''} {names} only"
        )
        shorthand = (*DEFAULT_SPAWN_NAMES, "default-" + world_id, world.default_spawn.id)
        if (
            value is None
            or value == []
            or (isinstance(value, list) and len(value) == 1 and value[0] in shorthand)
        ):
            chosen = list(predefined)
        elif isinstance(value, list):
            chosen = []
            for item in value:
                if item in shorthand:
                    raise ContractError(
                        field, f"Default stands for all spawns of {world_id}; give it alone"
                    )
                identity = item if isinstance(item, str) else mapping(item, "spawn").get("id")
                if not isinstance(identity, str) or identity not in predefined:
                    raise ContractError(field, f"{only}; {identity!r} is not one of them")
                if isinstance(item, dict):
                    known_keys(item, {"id", "name", "x", "y", "yaw"}, f"{field}.{identity}")
                    check_registry_spawn(
                        field,
                        world_id,
                        item,
                        predefined[identity],
                        "collection worlds run their registry spawns unchanged",
                    )
                if identity in chosen:
                    raise ContractError(field, f"{world_id}: spawn {identity} is given twice")
                chosen.append(identity)
        else:
            raise ContractError(field, "Expected a list of spawns")
        if tuning:
            if len(chosen) != 1:
                raise ContractError(
                    field, f"A tuning run picks one predefined spawn of {world_id}: {names}"
                )
        elif set(chosen) != set(predefined):
            missing = [identity for identity in predefined if identity not in chosen]
            raise ContractError(
                field,
                f"{world_id} runs all its predefined spawns {names}; missing {', '.join(missing)}",
            )
        return [predefined[identity] for identity in predefined if identity in chosen]

    def _rule_time_limits(
        self, raw: dict[str, Any], accepted: dict[str, Any]
    ) -> tuple[dict[str, int], dict[str, dict[str, RuleValue]]]:
        """``time_limit_rule`` (I4 §3.3): each selected world's limit and the rule's values of the robots
        with a retained combination in it (fully excluded robot-world pairs are skipped). Only computes:
        CampaignSpec.accepted refuses what the values contradict."""
        exclusions = raw.get("exclusions")
        # Malformed exclusions are refused by CampaignSpec.accepted; here they exclude nothing.
        excluded = {
            tuple(entry)
            for entry in (exclusions if isinstance(exclusions, list) else [])
            if isinstance(entry, list) and all(isinstance(part, str) for part in entry)
        }
        limits: dict[str, int] = {}
        inputs: dict[str, dict[str, RuleValue]] = {}
        for world_id, spawns in accepted["worlds"].items():
            # an inline spawn without an ID gets Spawn.accepted's ID
            identities = [spawn.get("id", content_hash(spawn)) for spawn in spawns]
            values: dict[str, RuleValue] = {}
            for robot_id in accepted["robots"]:
                if all((robot_id, world_id, identity) in excluded for identity in identities):
                    continue
                if world_id not in self.robots[robot_id]["worlds"]:
                    raise incompatible(robot_id, world_id)
                value = self.rule_value(world_id, robot_id)
                if value is None:
                    raise ContractError(
                        "time_limit_rule",
                        f"{world_id} has no metric-version-{RULE_METRIC_VERSION} coverage reference",
                    )
                values[robot_id] = value
            inputs[world_id] = values
            if values:
                limits[world_id] = next(iter(values.values()))["seconds"]
        return limits, inputs

    def plan_hash(self, study: Study) -> str | None:
        """SHA-256 of a study's plan above its deviations log (RD-29), read from this checkout; its analysis
        block is checked on the way (RO-15), and a confirmatory plan must hold one."""
        if study.plan is None:
            return None
        path = (self.root / study.plan).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            raise ContractError("study.plan", "Plan not found in the checkout: " + study.plan)
        plan = plan_parts(path.read_text(encoding="utf-8"))
        confirmatory = study.evidence_level == "confirmatory"
        if plan.block is not None:
            validate(plan.block, confirmatory=confirmatory, worlds=self.world_facts())
        elif confirmatory:
            raise ContractError(
                "study.plan", "A confirmatory plan holds an analysis block (docs/MEASUREMENTS.md)"
            )
        return plan.sha256

    def _save(
        self, store: Store, kind: str, owner: str, value: dict[str, Any], family_id: str | None
    ) -> dict[str, Any]:
        """Save a profile or spawn under a new ID. ``family_id`` is a saved item's stable key: an edit
        names it, and the new row replaces the item's previous one, so the library holds one row per item
        while accepted campaigns keep their own copies. The item keeps its creation time."""
        if family_id is not None:
            family_id = text(family_id, "family_id")
        family = family_id or str(uuid.uuid4())
        previous = store.rows(
            "SELECT id,kind,owner,created_ns FROM library WHERE family_id=?", (family,)
        )
        if family_id and not previous:
            raise ContractError(
                "family_id", f"This {kind} was deleted; save it as a new {kind}", "not_found"
            )
        if any(row["owner"] != owner or row["kind"] != kind for row in previous):
            owner_kind = "robot" if kind == "profile" else "world"
            raise ContractError("family_id", f"This {kind} belongs to another {owner_kind}")
        created = min((row["created_ns"] for row in previous), default=time.time_ns())
        result = dict(value, id=str(uuid.uuid4()), family_id=family, created_ns=created)
        result.pop("hash", None)
        result["hash"] = content_hash(result)
        store.db.execute("DELETE FROM library WHERE family_id=?", (family,))
        # The schema's revision column predates the removal of revisions; every row now holds 1.
        store.db.execute(
            "INSERT INTO library(id,kind,owner,name,family_id,revision,snapshot,hash,created_ns) VALUES(?,?,?,?,?,1,?,?,?)",
            (
                result["id"],
                kind,
                owner,
                result["name"],
                family,
                canonical(result),
                result["hash"],
                result["created_ns"],
            ),
        )
        store.event(
            "library.saved",
            {
                "kind": kind,
                "id": result["id"],
                "family_id": family,
                "replaced": [row["id"] for row in previous],
            },
        )
        return result

    def handle(self, method: str, params: dict[str, Any], store: Store) -> Any:
        self.bind(store)
        if method == "parameters.resolve":
            robot = text(params.get("robot_id", self.default_robot), "robot_id")
            return self.resolve_profile(robot, params)
        if method in ("profiles.get", "spawns.get"):
            return self._saved(
                text(params.get("id"), "id"),
                "profile" if method.startswith("profiles") else "spawn",
            )
        if method in ("profiles.list", "spawns.list"):
            kind, field = (
                ("profile", "robot_id") if method.startswith("profiles") else ("spawn", "world_id")
            )
            filters: dict[str, Any] = {"kind": kind}
            if kind == "spawn" and "archived" in params:
                # The archive controls were removed on 16 October 2026; the flag only filters on request.
                if type(params["archived"]) is not bool:
                    raise ContractError("archived", "Expected true or false")
                filters["archived"] = int(params["archived"])
            if params.get(field):
                filters["owner"] = params[field]
            if params.get("tag"):
                raise ContractError("tag", "Tag filtering was removed; search by profile name")
            offset = integer(params.get("offset", 0), "offset", 0, 2**63 - 1)
            limit = integer(params.get("limit", 100), "limit", 1, 250)
            # One row per item: a family saved before revisions were removed may still hold older rows.
            latest = (
                "NOT EXISTS(SELECT 1 FROM library newer WHERE newer.family_id=library.family_id "
                "AND newer.revision>library.revision)"
            )
            clauses, arguments = ["kind=?", latest], [kind]
            if "archived" in filters:
                clauses.append("archived=?")
                arguments.append(filters["archived"])
            if "owner" in filters:
                clauses.append("owner=?")
                arguments.append(text(filters["owner"], field))
            if params.get("search"):
                clauses.append("instr(lower(name),lower(?))>0")
                arguments.append(text(params["search"], "search"))
            where = " AND ".join(clauses)
            summary = params.get("summary", False)
            if type(summary) is not bool:
                raise ContractError("summary", "Expected boolean")
            selection = (
                "id,kind,owner,name,family_id,hash,archived,created_ns"
                # a profile's explorer (null: frontier_exploration_ros2), so lists can tag it
                + (",json_extract(snapshot,'$.explorer') AS explorer" if kind == "profile" else "")
                + ("" if summary else ",snapshot")
            )
            rows = store.rows(
                "SELECT "
                + selection
                + " FROM library WHERE "
                + where
                + " ORDER BY created_ns DESC LIMIT ? OFFSET ?",
                tuple(arguments + [limit, offset]),
            )
            count_row = store.one("SELECT COUNT(*) n FROM library WHERE " + where, tuple(arguments))
            assert count_row is not None
            total = count_row["n"]
            for row in rows:
                if "snapshot" in row:
                    row["snapshot"] = json.loads(row["snapshot"])
                    row["snapshot"].pop("revision", None)
            return {
                "rows": rows,
                "total": total,
                "offset": offset,
                "next_offset": offset + len(rows) if offset + len(rows) < total else None,
            }
        if method == "profiles.save":
            robot = text(params.get("robot_id"), "robot_id")
            value = self.resolve_profile(robot, params)
            return self._save(store, "profile", robot, value, params.get("family_id"))
        if method == "profiles.export":
            value = self._saved(text(params.get("id"), "id"), "profile")
            if params.get("format", "explorer") == "explorer":
                # the profile's explorer's node scope (explore_node, roadmap_explorer_node, frontier_explorer)
                schema = self.explorer(value.get("explorer", DEFAULT_EXPLORER)).schema
                return {"yaml": schema.yaml(value["effective"])}
            if params["format"] != "suite":
                raise ContractError("format", "Use explorer or suite")
            return {
                "yaml": yaml.safe_dump(
                    {
                        "suite_profile_version": 1,
                        "profile": {
                            k: value[k]
                            for k in (
                                "name",
                                "description",
                                "robot_id",
                                "requested",
                                "baseline_hash",
                                "metadata_hash",
                            )
                        }
                        # an external explorer's profile names it; frontier_exploration_ros2's leave it out
                        | ({"explorer": value["explorer"]} if "explorer" in value else {}),
                    },
                    sort_keys=False,
                )
            }
        if method == "profiles.import":
            document = load_yaml(
                text(params.get("yaml_text"), "yaml_text", maximum=4 * 1024 * 1024)
            )
            known_keys(document, {"suite_profile_version", "profile"}, "suite_profile")
            if document.get("suite_profile_version") != 1:
                raise ContractError("suite_profile_version", "Unsupported suite profile version")
            value = mapping(document.get("profile"), "profile")
            robot = text(value.get("robot_id"), "robot_id")
            return self._save(store, "profile", robot, self.resolve_profile(robot, value), None)
        if method in ("spawns.export", "spawns.import"):
            if method.endswith("export"):
                value = self._saved(text(params.get("id"), "id"), "spawn")
                return {
                    "yaml": yaml.safe_dump(
                        {
                            "suite_spawn_version": 1,
                            "spawn": {k: value[k] for k in ("name", "world_id", "x", "y", "yaw")},
                        },
                        sort_keys=False,
                    )
                }
            document = load_yaml(
                text(params.get("yaml_text"), "yaml_text", maximum=4 * 1024 * 1024)
            )
            known_keys(document, {"suite_spawn_version", "spawn"}, "suite_spawn")
            if document.get("suite_spawn_version") != 1:
                raise ContractError("suite_spawn_version", "Unsupported suite spawn version")
            value = mapping(document.get("spawn"), "spawn")
            known_keys(value, {"name", "world_id", "x", "y", "yaw"}, "spawn")
            value["robot_ids"] = params.get("robot_ids", list(self.robots))
            return self.handle("spawns.save", value, store)
        if method == "spawns.save":
            world = text(params.get("world_id"), "world_id")
            if world not in self.worlds:
                raise ContractError("world_id", "Unknown world")
            if self.worlds[world].spawn_policy == "predefined":
                raise ContractError(
                    "world_id",
                    f"{world} runs its predefined spawns from the world collection; it takes no saved spawns",
                )
            pose = {
                "name": text(params.get("name"), "name"),
                "world_id": world,
                "x": number(params.get("x"), "x"),
                "y": number(params.get("y"), "y"),
                "yaw": number(params.get("yaw"), "yaw"),
            }
            robot_ids = params.get("robot_ids", list(self.robots))
            if not isinstance(robot_ids, list) or not robot_ids:
                raise ContractError("robot_ids", "Select robots to validate the placement")
            reference = self.reference(world)
            for identity in robot_ids:
                identity = text(identity, "robot_ids")
                try:
                    validate_spawn(
                        self.robot(identity),
                        [pose["x"], pose["y"], 0, pose["yaw"]],
                        reference,
                    )
                except ValueError as exc:
                    raise ContractError("spawn", str(exc)) from exc
            return self._save(store, "spawn", world, pose, params.get("family_id"))
        if method == "spawns.map":
            return self._map(text(params.get("world_id"), "world_id"))
        if method == "profiles.delete":
            return self._delete_family(params, store, "profile")
        if method == "spawns.delete":
            return self._delete_family(params, store, "spawn")
        raise ContractError("method", "Unsupported library operation")

    def _delete_family(self, params: dict[str, Any], store: Store, kind: str) -> dict[str, Any]:
        """Delete saved profiles or spawns by their stable keys, all or nothing; campaigns keep their own
        copies. A world's Default spawn is not a saved spawn and cannot be deleted."""
        known_keys(params, {"family_ids"}, "params")
        identities = params.get("family_ids")
        noun = kind + "s"
        if not isinstance(identities, list) or not 1 <= len(identities) <= PROFILE_DELETION_LIMIT:
            raise ContractError("family_ids", f"Choose 1–{PROFILE_DELETION_LIMIT} {noun}")
        identities = [text(identity, "family_ids") for identity in identities]
        if len(set(identities)) != len(identities):
            raise ContractError("family_ids", f"Each {kind} may appear once")
        # A family saved before revisions were removed may still hold older rows; they go too.
        rows = store.rows(
            "SELECT id,owner,name,family_id,hash FROM library WHERE kind=? "
            "AND family_id IN (SELECT value FROM json_each(?)) ORDER BY family_id,created_ns",
            (kind, canonical(identities)),
        )
        missing = sorted(set(identities) - {row["family_id"] for row in rows})
        if missing:
            raise ContractError(
                "family_ids",
                f"{kind.capitalize()} does not exist or was already deleted: " + ", ".join(missing),
                "not_found",
            )
        column = "profile_id" if kind == "profile" else "spawn_id"
        used = store.one(
            f"SELECT COUNT(DISTINCT campaign_id) n FROM experiments "
            f"WHERE {column} IN (SELECT value FROM json_each(?)) "
            f"AND campaign_id NOT IN (SELECT id FROM campaigns WHERE {TUNING_CAMPAIGN})",
            (canonical([row["id"] for row in rows]),),
        )
        referenced = used["n"] if used else 0
        store.db.execute(
            "DELETE FROM library WHERE kind=? AND family_id IN (SELECT value FROM json_each(?))",
            (kind, canonical(identities)),
        )
        store.event(
            "library.deleted",
            {
                "kind": kind,
                "family_ids": identities,
                "rows": rows,
                "referenced_campaigns": referenced,
            },
        )
        return {
            "family_ids": identities,
            "deleted": [{k: row[k] for k in ("id", "family_id", "owner", "name")} for row in rows],
            "referenced_campaigns": referenced,
        }

    def _map(self, world: str) -> dict[str, Any]:
        if world not in self.worlds:
            raise ContractError("world_id", "Unknown world")
        if world not in self.maps:
            self.maps[world] = self._render_map(world)
        return dict(self.maps[world])

    def _render_map(self, world: str) -> dict[str, Any]:
        ref = self.reference(world)
        return {
            "world_id": world,
            "image_png": png(reference_pixels(ref.collision, ref.target), optimize=True),
            "origin": ref.origin,
            "resolution": ref.resolution,
            "width": ref.metadata["width"],
            "height": ref.metadata["height"],
            "reference_id": ref.identity,
            "metadata_sha256": ref.metadata_hash,
        }
