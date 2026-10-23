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
import json
from pathlib import Path
import time
from typing import Any
import uuid

import yaml

from exploration_benchmark_runner.parameters import ParameterSchema, UniqueKeyLoader
from exploration_benchmark_runner.configuration import robot_parameter_schema, scenario
from exploration_benchmark_runner.coverage import Reference, reference_file
from exploration_benchmark_robots.catalog import RobotCatalog, resolve_physical_settings
from exploration_benchmark_robots.preflight import validate_spawn

from .analysis_block import plan_parts, validate
from .contracts import (
    CampaignSpec,
    ContractError,
    DEFAULT_LIMIT,
    PROFILE_DELETION_LIMIT,
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

PROFILE_FIELDS = {
    "id",
    "name",
    "robot_id",
    "requested",
    "overrides",
    "robot_overrides",
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
# library revision numbers, 18 October 2026).
LEGACY_PROFILE_FIELDS = frozenset({"tags", "revision"})


def load_yaml(source: str) -> dict[str, Any]:
    try:
        return mapping(yaml.load(source, Loader=UniqueKeyLoader), "yaml")
    except yaml.YAMLError as exc:
        raise ContractError("yaml", str(exc)) from exc


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
                reference_file(self.root / "exploration_benchmark_runner/config/coverage", world)
            )
            for world in ("bookstore", "corridor", "warehouse")
        }
        self.scenarios = {world: scenario(self.root, world) for world in self.references}
        # spawns.map replies per world; the references above never change after loading.
        self.maps: dict[str, dict[str, Any]] = {}

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
        value = {k: v for k, v in value.items() if k not in LEGACY_PROFILE_FIELDS}
        known_keys(value, PROFILE_FIELDS, "profile")
        if value.get("robot_id", robot_id) != robot_id:
            raise ContractError("robot_id", "Profile belongs to another robot")
        if (
            value.get("baseline_hash", self.baseline_hash) != self.baseline_hash
            or value.get("metadata_hash", self.metadata_hash) != self.metadata_hash
        ):
            raise ContractError(
                "profile",
                "Stored profile uses different defaults or metadata; import its explorer YAML to save it as a new profile",
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
        }
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
                    "kinematics": profile["kinematics"],
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
            "study",
            "debug_streams",
            # an accepted campaign (Validate, then Add to queue; exported YAML) carries it: recomputed below
            "default_parameter_hashes",
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
                        item = self._saved(item, "profile")
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
                        item = self._saved(item, "spawn")
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
            validate(plan.block, confirmatory=confirmatory)
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
            robot = text(params.get("robot_id", "mogi"), "robot_id")
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
            selection = "id,kind,owner,name,family_id,hash,archived,created_ns" + (
                "" if summary else ",snapshot"
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
        if world not in self.references:
            raise ContractError("world_id", "Unknown world")
        if world not in self.maps:
            self.maps[world] = self._render_map(world)
        return dict(self.maps[world])

    def _render_map(self, world: str) -> dict[str, Any]:
        ref = self.references[world]
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
