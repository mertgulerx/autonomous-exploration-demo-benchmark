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

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from itertools import product
import json
import math
from pathlib import Path
import time
from typing import Any
import uuid

import yaml

from exploration_benchmark_runner.parameters import ParameterSchema, UniqueKeyLoader
from exploration_benchmark_runner.configuration import robot_parameter_schema, scenario
from exploration_benchmark_runner.coverage import Reference
from exploration_benchmark_robots.catalog import RobotCatalog, resolve_physical_settings
from exploration_benchmark_robots.preflight import validate_spawn

from .contracts import (
    CampaignSpec,
    ContractError,
    DEFAULT_LIMIT,
    HIDDEN_GROUPS,
    canonical,
    content_hash,
    integer,
    known_keys,
    mapping,
    number,
    text,
)
from .storage import file_hash
from .store import Store

PROFILE_FIELDS = {
    "id",
    "name",
    "robot_id",
    "requested",
    "overrides",
    "robot_overrides",
    "yaml_text",
    "description",
    "tags",
    "effective",
    "robot",
    "baseline_hash",
    "metadata_hash",
    "hash",
    "family_id",
    "revision",
    "parent_id",
    "sweep",
    "created_ns",
}


def load_yaml(source: str) -> dict[str, Any]:
    try:
        return mapping(yaml.load(source, Loader=UniqueKeyLoader), "yaml")
    except yaml.YAMLError as exc:
        raise ContractError("yaml", str(exc)) from exc


def sweep_values(axis: dict[str, Any], default: Any) -> list[Any]:
    known_keys(axis, {"values", "start", "stop", "step"}, "axis")
    if "values" in axis:
        if set(axis) != {"values"} or not isinstance(axis["values"], list) or not axis["values"]:
            raise ContractError("axis.values", "Provide one nonempty explicit list")
        values = axis["values"]
    else:
        if set(axis) != {"start", "stop", "step"} or type(default) not in (int, float):
            raise ContractError("axis", "Numeric ranges require start, stop and step")
        try:
            start, stop, step = (Decimal(str(axis[k])) for k in ("start", "stop", "step"))
            if (
                not all(v.is_finite() for v in (start, stop, step))
                or not step
                or (stop - start) * step < 0
            ):
                raise ContractError("axis", "Use a finite step pointing toward the stop value")
            count = int((stop - start) / step) + 1
            if count > 100000:
                raise ContractError(
                    "axis", "Split ranges exceeding 100000 values into separate sweeps"
                )
            decimals = [start + step * i for i in range(count)]
            if type(default) is int:
                if any(v != v.to_integral_value() for v in decimals):
                    raise ContractError("axis", "Integer parameters require integer range values")
                values = [int(v) for v in decimals]
            else:
                values = [float(v) for v in decimals]
        except (InvalidOperation, ValueError, OverflowError) as exc:
            raise ContractError("axis", "Invalid decimal range") from exc
    if any(type(v) is not type(default) for v in values):
        raise ContractError("axis", f"Values must have type {type(default).__name__}")
    if len({canonical(v) for v in values}) != len(values):
        raise ContractError("axis", "Duplicate values")
    return values


class Library:
    def __init__(self, root: Path, *, native: bool = True):
        self.root, self.native = root.resolve(), native
        self.store: Store | None = None
        self.baseline_file = self.root / "config/frontier_exploration_ros2/default.yaml"
        self.metadata_file = self.root / "exploration_benchmark_runner/config/parameters.yaml"
        self.schema = ParameterSchema(self.baseline_file, self.metadata_file)
        self.baseline_hash, self.metadata_hash = (
            file_hash(self.baseline_file),
            file_hash(self.metadata_file),
        )
        resource = self.root / "exploration_benchmark_robots"
        self.robots = yaml.safe_load((resource / "config/robots.yaml").read_text())["robots"]
        self.robot_catalog = RobotCatalog(resource) if native else None
        self.accepted_robots: dict[str, dict[str, Any]] = {}
        self.references = {
            world: Reference(
                self.root / "exploration_benchmark_runner/config/coverage" / (world + ".json")
            )
            for world in ("bookstore", "corridor", "warehouse")
        }
        self.scenarios = {world: scenario(self.root, world) for world in self.references}

    def bind(self, store: Store) -> None:
        self.store = store

    def robot(self, identity: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        if identity not in self.robots:
            raise ContractError("robot_id", "Unknown robot: " + identity)
        try:
            if identity not in self.accepted_robots:
                self.accepted_robots[identity] = (
                    self.robot_catalog.require(identity, self.robots[identity]["worlds"][0])
                    if self.robot_catalog
                    else deepcopy(self.robots[identity])
                )
            base = deepcopy(self.accepted_robots[identity])
            base.pop("benchmark_overrides", None)
            base.pop("benchmark_settings", None)
            return mapping(resolve_physical_settings(base, overrides), "physical_settings")
        except ValueError as exc:
            raise ContractError("robot_id", str(exc), "dependency_or_configuration") from exc

    def resolve_profile(self, robot_id: str, value: dict[str, Any]) -> dict[str, Any]:
        known_keys(value, PROFILE_FIELDS, "profile")
        if value.get("robot_id", robot_id) != robot_id:
            raise ContractError("robot_id", "Profile belongs to another robot")
        if (
            value.get("baseline_hash", self.baseline_hash) != self.baseline_hash
            or value.get("metadata_hash", self.metadata_hash) != self.metadata_hash
        ):
            raise ContractError(
                "profile",
                "Stored profile uses different defaults or metadata; import its explorer YAML to create an explicit revision",
            )
        requested = mapping(value.get("requested", value.get("overrides", {})), "overrides")
        physical = mapping(value.get("robot_overrides", {}), "robot_overrides")
        robot = self.robot(robot_id, physical)
        schema = robot_parameter_schema(self.schema, robot)
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
            "name": text(value.get("name", "Default"), "profile.name"),
            "robot_id": robot_id,
            "requested": deepcopy(requested),
            "effective": effective,
            "robot_overrides": deepcopy(physical),
            "robot": robot,
            "baseline_hash": self.baseline_hash,
            "metadata_hash": self.metadata_hash,
            "yaml_text": imported,
            "description": value.get("description", ""),
            "tags": value.get("tags", []),
        }
        if not isinstance(result["description"], str) or not isinstance(result["tags"], list):
            raise ContractError("profile", "Description must be text and tags a string list")
        if any(not isinstance(tag, str) for tag in result["tags"]):
            raise ContractError("tags", "Every tag must be text")
        if (
            len(result["description"]) > 4096
            or len(result["tags"]) > 32
            or any(len(tag) > 80 for tag in result["tags"])
        ):
            raise ContractError(
                "profile",
                "Use a description up to 4096 characters and at most 32 tags of 80 characters",
            )
        for key in ("family_id", "parent_id"):
            if key in value:
                result[key] = text(value[key], key)
        for key in ("revision", "created_ns"):
            if key in value:
                result[key] = integer(value[key], key, 1, 2**63 - 1)
        if "sweep" in value:
            sweep = mapping(value["sweep"], "sweep")
            known_keys(sweep, {"axes", "values"}, "sweep")
            axes, values = (
                mapping(sweep.get("axes"), "sweep.axes"),
                mapping(sweep.get("values"), "sweep.values"),
            )
            if not axes or set(axes) != set(values):
                raise ContractError("sweep", "Every sweep axis requires an accepted value")
            for key, axis in axes.items():
                candidates = sweep_values(
                    mapping(axis, "sweep.axes." + key), self.axis_default(robot_id, key)
                )
                chosen = values[key]
                actual = (
                    physical.get(
                        key[6:],
                        {"movement_speed": 0.8, "turning_speed": 1.0, "lidar_range": 10.0}.get(
                            key[6:]
                        ),
                    )
                    if key.startswith("robot.")
                    else effective.get(key)
                )
                if (
                    type(chosen) is not type(candidates[0])
                    or chosen not in candidates
                    or chosen != actual
                ):
                    raise ContractError(
                        "sweep.values." + key,
                        "Sweep provenance must match the accepted configuration",
                    )
            result["sweep"] = deepcopy(sweep)
        result["hash"] = content_hash(result)
        result["id"] = text(
            value.get(
                "id", "default-" + robot_id if not requested and not physical else result["hash"]
            ),
            "profile.id",
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
                    "physics_step": profile["physics_step"],
                    "worlds": profile["worlds"],
                    "defaults": defaults,
                }
            )
        worlds = []
        for identity, reference in self.references.items():
            x, y, _, yaw = self.scenarios[identity]["spawn"]
            worlds.append(
                {
                    "id": identity,
                    "label": identity.capitalize(),
                    "default_time_limit": DEFAULT_LIMIT,
                    "reference": reference.descriptor(),
                    "default_spawn": {
                        "id": "default-" + identity,
                        "name": "Default",
                        "x": x,
                        "y": y,
                        "yaw": yaw,
                    },
                    "origin": reference.origin,
                    "resolution": reference.resolution,
                    "width": reference.metadata["width"],
                    "height": reference.metadata["height"],
                }
            )
        return {
            "robots": robots,
            "worlds": worlds,
            "parameters": self.schema.metadata,
            "baseline_hash": self.baseline_hash,
            "metadata_hash": self.metadata_hash,
            "source": "native" if self.native else "synthetic",
            "defaults": {"concurrency": 1},
        }

    def _revision(self, identity: str, kind: str) -> dict[str, Any]:
        if self.store is None:
            raise RuntimeError("Library is not attached to the engine")
        row = self.store.one("SELECT * FROM library WHERE id=? AND kind=?", (identity, kind))
        if not row:
            raise ContractError("revision_id", "Library revision does not exist", "not_found")
        return mapping(json.loads(row["snapshot"]), "snapshot")

    def resolve_campaign(self, raw: dict[str, Any]) -> CampaignSpec:
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
                    if item in ("Default", "default", "default-" + robot_id):
                        item = {"name": "Default", "id": "default-" + robot_id}
                    else:
                        item = self._revision(item, "profile")
                        if item["robot_id"] != robot_id:
                            raise ContractError("profiles", "Profile belongs to another robot")
                value = mapping(item, "profile")
                selected.append(self.resolve_profile(robot_id, value))
            accepted["robots"][robot_id] = selected
        for world_id, spawns in mapping(raw.get("worlds"), "worlds").items():
            if world_id not in self.references:
                raise ContractError("worlds", "Unknown world: " + world_id)
            if not isinstance(spawns, list) or not spawns:
                raise ContractError("worlds." + world_id, "Select at least one spawn")
            selected = []
            for item in spawns:
                if isinstance(item, str):
                    if item in ("Default", "default", "default-" + world_id):
                        x, y, _, yaw = self.scenarios[world_id]["spawn"]
                        item = {
                            "id": "default-" + world_id,
                            "name": "Default",
                            "x": x,
                            "y": y,
                            "yaw": yaw,
                        }
                    else:
                        item = self._revision(item, "spawn")
                        if item["world_id"] != world_id:
                            raise ContractError("spawns", "Spawn belongs to another world")
                        item = {
                            k: v for k, v in item.items() if k in ("id", "name", "x", "y", "yaw")
                        }
                selected.append(mapping(item, "spawn"))
            accepted["worlds"][world_id] = selected
            if self.native:
                try:
                    self.references[world_id].validate_scenario(
                        self.root, self.scenarios[world_id], allow_custom_spawn=True
                    )
                except ValueError as exc:
                    raise ContractError(
                        "worlds." + world_id, str(exc), "reference_mismatch"
                    ) from exc
        spec = CampaignSpec.accepted(accepted)
        exclusions = set(spec.exclusions)
        for robot_id, profiles in spec.robots:
            for world_id, spawns in spec.worlds:
                for spawn in spawns:
                    if (robot_id, world_id, spawn.id) in exclusions:
                        continue
                    for profile in profiles:
                        robot = profile.dictionary()["robot"]
                        if world_id not in robot["worlds"]:
                            raise ContractError(
                                "compatibility",
                                f"{robot_id} cannot use {world_id}; explicitly exclude this combination",
                            )
                        try:
                            validate_spawn(
                                robot, [spawn.x, spawn.y, 0.0, spawn.yaw], self.references[world_id]
                            )
                        except ValueError as exc:
                            raise ContractError("spawns." + spawn.id, str(exc)) from exc
        return spec

    def _save(
        self, store: Store, kind: str, owner: str, value: dict[str, Any], family_id: str | None
    ) -> dict[str, Any]:
        if family_id is not None:
            family_id = text(family_id, "family_id")
        family = family_id or str(uuid.uuid4())
        previous = store.one(
            "SELECT * FROM library WHERE family_id=? ORDER BY revision DESC LIMIT 1", (family,)
        )
        if family_id and (not previous or previous["owner"] != owner or previous["kind"] != kind):
            raise ContractError("family_id", "Revision family does not match this library")
        revision = previous["revision"] + 1 if previous else 1
        result = dict(
            value,
            id=str(uuid.uuid4()),
            family_id=family,
            revision=revision,
            created_ns=time.time_ns(),
        )
        result.pop("hash", None)
        result["hash"] = content_hash(result)
        store.db.execute(
            "INSERT INTO library(id,kind,owner,name,family_id,revision,snapshot,hash,created_ns) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                result["id"],
                kind,
                owner,
                result["name"],
                family,
                revision,
                canonical(result),
                result["hash"],
                result["created_ns"],
            ),
        )
        store.event("library.saved", {"kind": kind, "id": result["id"], "revision": revision})
        return result

    def handle(self, method: str, params: dict[str, Any], store: Store) -> Any:
        self.bind(store)
        if method == "parameters.resolve":
            robot = text(params.get("robot_id", "mogi"), "robot_id")
            return self.resolve_profile(robot, params)
        if method in ("profiles.get", "spawns.get"):
            return self._revision(
                text(params.get("revision_id"), "revision_id"),
                "profile" if method.startswith("profiles") else "spawn",
            )
        if method in ("profiles.list", "spawns.list"):
            kind, field = (
                ("profile", "robot_id") if method.startswith("profiles") else ("spawn", "world_id")
            )
            filters: dict[str, Any] = {"kind": kind, "archived": int(params.get("archived", False))}
            if params.get(field):
                filters["owner"] = params[field]
            if not params.get("search") and not params.get("tag"):
                return store.query(
                    "library",
                    filters,
                    params.get("offset", 0),
                    params.get("limit", 100),
                    summary=params.get("summary", False),
                )
            offset = integer(params.get("offset", 0), "offset", 0, 2**63 - 1)
            limit = integer(params.get("limit", 100), "limit", 1, 250)
            clauses, arguments = ["kind=?", "archived=?"], [kind, filters["archived"]]
            if "owner" in filters:
                clauses.append("owner=?")
                arguments.append(text(filters["owner"], field))
            if params.get("search"):
                clauses.append("instr(lower(name),lower(?))>0")
                arguments.append(text(params["search"], "search"))
            if params.get("tag"):
                clauses.append(
                    "EXISTS(SELECT 1 FROM json_each(library.snapshot,'$.tags') WHERE value=?)"
                )
                arguments.append(text(params["tag"], "tag"))
            where = " AND ".join(clauses)
            summary = params.get("summary", False)
            if type(summary) is not bool:
                raise ContractError("summary", "Expected boolean")
            selection = (
                "id,kind,owner,name,family_id,revision,hash,archived,created_ns" if summary else "*"
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
            value = self._revision(text(params.get("revision_id"), "revision_id"), "profile")
            if params.get("format", "explorer") == "explorer":
                return {"yaml": self.schema.yaml(value["effective"])}
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
                                "tags",
                                "robot_id",
                                "requested",
                                "robot_overrides",
                                "baseline_hash",
                                "metadata_hash",
                            )
                        },
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
        if method == "profiles.diff":
            left = self._revision(text(params.get("left"), "left"), "profile")
            right = self._revision(text(params.get("right"), "right"), "profile")
            left_physical = left["robot"].get("benchmark_settings", {}).get("effective", {})
            right_physical = right["robot"].get("benchmark_settings", {}).get("effective", {})
            return {
                "same_robot": left["robot_id"] == right["robot_id"],
                "algorithm": {
                    key: [left["effective"].get(key), right["effective"].get(key)]
                    for key in left["effective"] | right["effective"]
                    if left["effective"].get(key) != right["effective"].get(key)
                },
                "physical": {
                    key: [left_physical.get(key), right_physical.get(key)]
                    for key in left_physical | right_physical
                    if left_physical.get(key) != right_physical.get(key)
                },
            }
        if method in ("spawns.export", "spawns.import"):
            if method.endswith("export"):
                value = self._revision(text(params.get("revision_id"), "revision_id"), "spawn")
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
            value["robot_ids"] = params.get("robot_ids", ["mogi"])
            return self.handle("spawns.save", value, store)
        if method == "spawns.save":
            world = text(params.get("world_id"), "world_id")
            if world not in self.references:
                raise ContractError("world_id", "Unknown world")
            pose = {
                "name": text(params.get("name"), "name"),
                "world_id": world,
                "x": number(params.get("x"), "x"),
                "y": number(params.get("y"), "y"),
                "yaw": number(params.get("yaw"), "yaw"),
            }
            robot_ids = params.get("robot_ids", ["mogi"])
            if not isinstance(robot_ids, list) or not robot_ids:
                raise ContractError("robot_ids", "Select robots to validate the placement")
            for identity in robot_ids:
                identity = text(identity, "robot_ids")
                try:
                    validate_spawn(
                        self.robot(identity),
                        [pose["x"], pose["y"], 0, pose["yaw"]],
                        self.references[world],
                    )
                except ValueError as exc:
                    raise ContractError("spawn", str(exc)) from exc
            return self._save(store, "spawn", world, pose, params.get("family_id"))
        if method in ("profiles.archive", "spawns.archive"):
            identity = text(params.get("revision_id"), "revision_id")
            kind = "profile" if method.startswith("profiles") else "spawn"
            self._revision(identity, kind)
            if type(params.get("archived", True)) is not bool:
                raise ContractError("archived", "Expected boolean")
            store.db.execute(
                "UPDATE library SET archived=? WHERE id=?",
                (int(params.get("archived", True)), identity),
            )
            return {"id": identity, "archived": params.get("archived", True)}
        if method == "spawns.map":
            return self._map(text(params.get("world_id"), "world_id"))
        if method in ("profiles.sweep", "profiles.sweep_preview"):
            return self._sweep(params, store, save=method == "profiles.sweep")
        raise ContractError("method", "Unsupported library operation")

    def _sweep(self, params: dict[str, Any], store: Store, *, save: bool) -> Any:
        robot_id = text(params.get("robot_id"), "robot_id")
        parent = (
            self._revision(text(params["parent_id"], "parent_id"), "profile")
            if params.get("parent_id")
            else self.resolve_profile(robot_id, {k: v for k, v in params.items() if k != "axes"})
        )
        if parent["robot_id"] != robot_id:
            raise ContractError("parent_id", "Profile belongs to another robot")
        axes = mapping(params.get("axes"), "axes")
        if not axes:
            raise ContractError("axes", "Select at least one sweep axis")
        keys, values = [], []
        for key, axis in axes.items():
            keys.append(key)
            values.append(sweep_values(mapping(axis, "axis"), self.axis_default(robot_id, key)))
        count = math.prod(len(v) for v in values)
        if count > 10000:
            raise ContractError(
                "axes", "Split sweeps exceeding 10000 profiles into separate sweeps"
            )
        variants: list[dict[str, Any]] = []
        revision_ids: list[str] = []
        seen: set[str] = set()
        for combination in product(*values):
            value = deepcopy(parent)
            value.pop("sweep", None)
            axis_values = dict(zip(keys, combination))
            for key, item in axis_values.items():
                value.setdefault(
                    "robot_overrides" if key.startswith("robot.") else "requested", {}
                )[key[6:] if key.startswith("robot.") else key] = item
            value.pop("id", None)
            value["name"] = (
                parent["name"] + " · " + ", ".join(f"{k}={v}" for k, v in axis_values.items())
            )[:160]
            resolved = self.resolve_profile(robot_id, value)
            identity = content_hash([resolved["effective"], resolved["robot_overrides"]])
            if identity in seen:
                raise ContractError("axes", "Sweep produces duplicate effective variants")
            seen.add(identity)
            resolved.update(parent_id=parent["id"], sweep={"axes": axes, "values": axis_values})
            if save:
                revision_ids.append(self._save(store, "profile", robot_id, resolved, None)["id"])
            elif len(variants) < 10:
                variants.append(resolved)
        return {"count": count, "profiles": variants, "revision_ids": revision_ids}

    def axis_default(self, robot_id: str, key: str) -> Any:
        physical_defaults = {"movement_speed": 0.8, "turning_speed": 1.0, "lidar_range": 10.0}
        if key.startswith("robot."):
            if robot_id != "mogi" or key[6:] not in physical_defaults:
                raise ContractError("axes." + key, "Only MOGI physical settings may be swept")
            return physical_defaults[key[6:]]
        metadata = self.schema.metadata.get(key)
        if not metadata or metadata["managed"] or metadata["group"] in HIDDEN_GROUPS:
            raise ContractError("axes." + key, "Parameter is not an editable sweep axis")
        return self.schema.defaults[key]

    def _map(self, world: str) -> dict[str, Any]:
        import base64
        from io import BytesIO
        import numpy as np
        from PIL import Image

        if world not in self.references:
            raise ContractError("world_id", "Unknown world")
        ref = self.references[world]
        pixels = np.full(ref.collision.shape, 170, dtype=np.uint8)
        pixels[ref.target] = 245
        pixels[ref.collision] = 30
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG", optimize=True)
        return {
            "world_id": world,
            "image_png": base64.b64encode(buffer.getvalue()).decode(),
            "origin": ref.origin,
            "resolution": ref.resolution,
            "width": ref.metadata["width"],
            "height": ref.metadata["height"],
            "reference_id": ref.identity,
            "metadata_sha256": ref.metadata_hash,
        }
