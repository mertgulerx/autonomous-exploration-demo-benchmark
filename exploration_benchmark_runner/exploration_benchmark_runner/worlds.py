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

"""The world registry: every world the runner and the suite can run.

One data file (``config/worlds.yaml``) names each world's file, coverage settings,
pinned reference, spawns, robot fit and resources. The loader validates it completely
and returns frozen values. It imports no ROS package, so the suite's synthetic library,
host tools and scratch checks load it without a ROS environment (the GUI reads the
YAML itself; it has no runner package).
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

import yaml

from .parameters import UniqueKeyLoader

REGISTRY_VERSION = 1
# Repository-relative locations of the registry, the coverage references and the robot
# catalog (the suite reads the checkout, the manual runner the installed share).
REGISTRY_FILE = "exploration_benchmark_runner/config/worlds.yaml"
REFERENCE_DIRECTORY = "exploration_benchmark_runner/config/coverage"
ROBOT_CATALOG_FILE = "exploration_benchmark_robots/config/robots.yaml"
WORLD_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
SPAWN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TIERS = ("A", "SC", "F", "RP", "MO", "P")
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
ROLES = ("evaluation", "pilot", "development")
WORLD_FIELDS = (
    "label",
    "tier",
    "family",
    "role",
    "world_file",
    "source",
    "scenario",
    "reference",
    "default_spawn",
    "spawns",
    "robots",
    "fuel_models",
    "resource_paths",
    "moving_objects",
    "descriptors",
)
SOURCE_FIELDS = ("world_yaml", "world_yaml_sha256", "sdf_sha256", "worldgen_version")
# A world's official descriptor file and region raster in REFERENCE_DIRECTORY
# (I2 §3.9).
DESCRIPTORS_SUFFIX = "_descriptors.json"
REGIONS_SUFFIX = "_regions.png"


@dataclass(frozen=True, slots=True)
class Spawn:
    id: str
    name: str
    x: float
    y: float
    yaw: float
    z: float | None = None  # set for the default spawn only (the scenario height)

    def pose(self) -> list[float]:
        """``[x, y, z, yaw]`` as the runner's scenario holds it (default spawn only)."""
        if self.z is None:
            raise ValueError(f"Spawn {self.id} has no height")
        return [self.x, self.y, self.z, self.yaw]


@dataclass(frozen=True, slots=True)
class RobotFit:
    robot_id: str
    spawn_z: float | None  # None: the robot's catalog spawn_z


@dataclass(frozen=True, slots=True)
class Scenario:
    file: str  # repository-relative coverage settings file
    key: str | None  # entry under ``scenarios``; None for a flat per-world file


@dataclass(frozen=True, slots=True)
class ReferencePin:
    metric_version: int
    id: str
    metadata_sha256: str
    metadata_file: Path  # absolute path of the pinned metadata file


@dataclass(frozen=True, slots=True)
class Source:
    """Where a world_collection world comes from; see ``worlds_table``."""

    world_yaml: str
    world_yaml_sha256: str
    sdf_sha256: str
    worldgen_version: str | None


@dataclass(frozen=True, slots=True)
class Descriptors:
    file: str
    sha256: str

    @property
    def regions_file(self) -> str:
        """The region raster pinned by the descriptor file: ``<id>_descriptors.json`` ->
        ``<id>_regions.png`` beside it. (Inside the file, ``region_raster.file`` is the
        raster's name in a study export, ``regions.png``.)"""
        if not self.file.endswith(DESCRIPTORS_SUFFIX):
            raise ValueError(
                f"{self.file}: expected a name ending {DESCRIPTORS_SUFFIX}"
            )
        return self.file.removesuffix(DESCRIPTORS_SUFFIX) + REGIONS_SUFFIX


@dataclass(frozen=True, slots=True)
class World:
    id: str
    label: str
    tier: str
    family: str | None  # F1–F5 for tier F (the run record's world_family), else None
    role: str
    world_file: str
    source: Source | None
    scenario: Scenario
    reference: ReferencePin
    default_spawn: Spawn
    spawns: tuple[Spawn, ...]
    robots: tuple[RobotFit, ...]
    fuel_models: tuple[str, ...]
    resource_paths: tuple[str, ...]
    # The registry's moving-object block (I5 defines its content) as canonical JSON
    # text, so World stays immutable and hashable; ``moving_objects`` gives plain data.
    moving_objects_json: str | None
    descriptors: Descriptors | None

    @property
    def moving_objects(self) -> dict | None:
        """A fresh plain copy of the moving-object block, or None."""
        if self.moving_objects_json is None:
            return None
        return json.loads(self.moving_objects_json)

    @property
    def spawn_policy(self) -> str:
        """``custom`` for tier A; collection worlds run all their predefined spawns."""
        return "custom" if self.tier == "A" else "predefined"

    @property
    def robot_ids(self) -> tuple[str, ...]:
        return tuple(fit.robot_id for fit in self.robots)


@dataclass(frozen=True, slots=True)
class WorldRegistry:
    root: Path
    file: Path
    sha256: str
    version: int
    default_world: str
    worlds: tuple[World, ...]

    @classmethod
    def load(cls, root, path=None, *, references=None, robots=None, pinned=True):
        """Load and validate the registry.

        ``root`` is the repository root that every path is relative to; ``path`` the
        registry file (default: the checkout's); ``references`` the coverage reference
        directory (default: the checkout's); ``robots`` the known robot IDs (default:
        the checkout's robot catalog). ``pinned=False`` skips the reference file checks
        (reference generation for a world whose reference does not exist yet).
        """
        root = Path(root).resolve()
        file = Path(path) if path is not None else root / REGISTRY_FILE
        try:
            data = file.read_bytes()
        except OSError as exc:
            raise ValueError(f"World registry unreadable: {file}: {exc}") from exc
        try:
            raw = yaml.load(data, Loader=UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"World registry {file}: {exc}") from exc
        if robots is None:
            robots = _catalog_robot_ids(root / ROBOT_CATALOG_FILE)
        directory = (
            Path(references) if references is not None else root / REFERENCE_DIRECTORY
        )
        reader = _Reader(root, directory, frozenset(robots), pinned)
        _keys(raw, ("registry_version", "default_world", "worlds"), "registry")
        if type(raw["registry_version"]) is not int:
            raise ValueError("registry_version: expected an integer")
        if raw["registry_version"] != REGISTRY_VERSION:
            raise ValueError(
                f"registry_version: {raw['registry_version']} is not supported; "
                f"expected {REGISTRY_VERSION}"
            )
        entries = raw["worlds"]
        if not isinstance(entries, dict) or not entries:
            raise ValueError("worlds: expected a non-empty mapping")
        worlds = tuple(reader.world(key, value) for key, value in entries.items())
        default = raw["default_world"]
        if not isinstance(default, str) or default not in entries:
            raise ValueError(f"default_world: unknown world {default!r}")
        return cls(
            root,
            file,
            hashlib.sha256(data).hexdigest(),
            REGISTRY_VERSION,
            default,
            worlds,
        )

    def ids(self) -> tuple[str, ...]:
        return tuple(world.id for world in self.worlds)

    def runnable(self) -> tuple[World, ...]:
        """Worlds the runner and the suite offer (not development ones), in order."""
        return tuple(world for world in self.worlds if world.role != "development")

    def world(self, identity: str) -> World:
        for world in self.worlds:
            if world.id == identity:
                return world
        raise ValueError(
            f"Unknown world: {identity}; choose one of {', '.join(self.ids())}"
        )

    def path(self, relative: str) -> Path:
        """Absolute path of a repository-relative registry path."""
        return self.root / relative

    def fuel_models(self, identities=None) -> list[str]:
        """Pinned Fuel model URIs of the given worlds (default: all), deduplicated."""
        selected = (
            self.worlds if identities is None else [self.world(i) for i in identities]
        )
        seen, result = set(), []
        for world in selected:
            for uri in world.fuel_models:
                if _fuel_identity(uri) not in seen:
                    seen.add(_fuel_identity(uri))
                    result.append(uri)
        return result

    def worlds_table(self) -> list[dict]:
        """One row per world for the study export's ``worlds`` table (export version 2).

        Design point (``parameters``) and seed come from ``source.world_yaml`` of
        generated (tier F) worlds, whose bytes must still match the pinned hash; named
        worlds have ``{}`` and null. Every collection world names its world.yaml hash
        and generator version (``source.worldgen_version``); tier A has nulls.
        """
        rows = []
        for world in self.worlds:
            point, seed, generator, digest = {}, None, None, None
            if world.source is not None:
                digest = world.source.world_yaml_sha256
                generator = world.source.worldgen_version
            if world.source is not None and world.tier == "F":
                name = world.source.world_yaml
                data = self.path(name).read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f"{name}: changed since the registry was loaded")
                document = yaml.load(data, Loader=UniqueKeyLoader)
                if not isinstance(document, dict):
                    raise ValueError(f"{name}: expected a mapping")
                point = document.get("parameters") or {}
                seed = document.get("seed")
                if type(seed) is not int:
                    raise ValueError(f"{name}: seed must be an integer for tier F")
            rows.append(
                dict(
                    world_id=world.id,
                    tier=world.tier,
                    family=world.family,
                    role=world.role,
                    design_point_json=json.dumps(
                        point, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                    ),
                    seed=seed,
                    generator_version=generator,
                    world_yaml_sha256=digest,
                )
            )
        return rows


def _catalog_robot_ids(path):
    try:
        data = yaml.load(Path(path).read_bytes(), Loader=UniqueKeyLoader)
        return tuple(data["robots"])
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Robot catalog unreadable for the world registry: {exc}"
        ) from exc


def file_sha256(path):
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _keys(value, fields, name, optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a mapping")
    unknown = sorted(set(value) - set(fields) - set(optional))
    if unknown:
        raise ValueError(f"{name}: unknown field(s) {', '.join(unknown)}")
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{name}: missing field(s) {', '.join(missing)}")


def _text(value, name, pattern=None):
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{name}: expected a single-line text")
    if pattern is not None and not pattern.match(value):
        raise ValueError(f"{name}: {value!r} does not match {pattern.pattern}")
    return value


def _float(value, name):
    # Exact type: YAML integers and booleans are refused (floats keep a decimal point).
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name}: expected a finite float with a decimal point")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not SHA256.match(value):
        raise ValueError(f"{name}: expected a lower-case SHA-256")
    return value


def _fuel_identity(uri):
    """A Fuel model URI, unquoted and lower-case (Fuel resolves names so)."""
    return unquote(uri).lower()


def fuel_model(uri, name="fuel_models"):
    """A pinned Fuel model URI in the form the coverage asset resolver accepts.

    Paired with ``coverage.resolve_asset``: the same path rule (``/1.0/<owner>/models/
    <name>/<version>``), plus https and no query or fragment; change both together.
    """
    if not isinstance(uri, str):
        raise ValueError(f"{name}: expected a URI")
    parsed = urlparse(uri)
    parts = [unquote(part).lower() for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or len(parts) != 5
        or parts[0] != "1.0"
        or parts[2] != "models"
        or not parts[4].isdigit()
    ):
        raise ValueError(f"{name}: {uri!r} is not a versioned Fuel model URI")
    return uri


class _Reader:
    def __init__(self, root, references, robots, pinned):
        self.root, self.references, self.robots, self.pinned = (
            root,
            references,
            robots,
            pinned,
        )

    def file(self, value, name, directory=False):
        relative = _text(value, name)
        if Path(relative).is_absolute():
            raise ValueError(f"{name}: use a path relative to the repository root")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"{name}: {relative} lies outside the repository root")
        if not (path.is_dir() if directory else path.is_file()):
            kind = "directory" if directory else "file"
            raise ValueError(f"{name}: {kind} does not exist: {relative}")
        return relative

    def spawn(self, value, name, default):
        fields = (
            ("id", "name", "x", "y", "z", "yaw")
            if default
            else ("id", "name", "x", "y", "yaw")
        )
        _keys(value, fields, name)
        return Spawn(
            _text(value["id"], name + ".id", SPAWN_ID),
            _text(value["name"], name + ".name"),
            _float(value["x"], name + ".x"),
            _float(value["y"], name + ".y"),
            _float(value["yaw"], name + ".yaw"),
            _float(value["z"], name + ".z") if default else None,
        )

    def world(self, identity, value):
        name = f"worlds.{identity}"
        _text(identity, name, WORLD_ID)
        _keys(value, WORLD_FIELDS, name)
        tier = value["tier"]
        if tier not in TIERS:
            raise ValueError(f"{name}.tier: use one of {', '.join(TIERS)}")
        family = value["family"]
        if tier == "F":
            if family not in FAMILIES:
                raise ValueError(
                    f"{name}.family: use one of {', '.join(FAMILIES)} for tier F"
                )
        elif family is not None:
            raise ValueError(f"{name}.family: only tier F worlds have a family")
        if value["role"] not in ROLES:
            raise ValueError(f"{name}.role: use one of {', '.join(ROLES)}")
        world_file = self.file(value["world_file"], name + ".world_file")
        source = self.source(value["source"], name + ".source", world_file)
        if tier == "A" and source is not None:
            raise ValueError(
                f"{name}.source: tier A worlds are not from the world collection"
            )
        if tier != "A" and source is None:
            raise ValueError(f"{name}.source: collection worlds name their world.yaml")
        scenario = self.scenario(value["scenario"], name + ".scenario")
        fuel = self.fuel(value["fuel_models"], name + ".fuel_models")
        reference = self.reference(
            value["reference"], name + ".reference", identity, fuel
        )
        default_spawn = self.spawn(
            value["default_spawn"], name + ".default_spawn", True
        )
        spawns = value["spawns"]
        if not isinstance(spawns, list):
            raise ValueError(f"{name}.spawns: expected a list")
        spawns = tuple(
            self.spawn(item, f"{name}.spawns[{index}]", False)
            for index, item in enumerate(spawns)
        )
        identities = [default_spawn.id, *(spawn.id for spawn in spawns)]
        if len(set(identities)) != len(identities):
            raise ValueError(
                f"{name}.spawns: spawn IDs must be unique within the world"
            )
        return World(
            identity,
            _text(value["label"], name + ".label"),
            tier,
            family,
            value["role"],
            world_file,
            source,
            scenario,
            reference,
            default_spawn,
            spawns,
            self.robot_fit(value["robots"], name + ".robots"),
            fuel,
            self.resources(value["resource_paths"], name + ".resource_paths"),
            self.moving_objects(value["moving_objects"], name + ".moving_objects"),
            self.descriptors(value["descriptors"], name + ".descriptors"),
        )

    def source(self, value, name, world_file):
        if value is None:
            return None
        _keys(value, SOURCE_FIELDS, name)
        world_yaml = self.file(value["world_yaml"], name + ".world_yaml")
        result = Source(
            world_yaml,
            _digest(value["world_yaml_sha256"], name + ".world_yaml_sha256"),
            _digest(value["sdf_sha256"], name + ".sdf_sha256"),
            (
                None
                if value["worldgen_version"] is None
                else _text(value["worldgen_version"], name + ".worldgen_version")
            ),
        )
        if file_sha256(self.root / world_yaml) != result.world_yaml_sha256:
            raise ValueError(f"{name}.world_yaml_sha256: {world_yaml} changed")
        if file_sha256(self.root / world_file) != result.sdf_sha256:
            raise ValueError(f"{name}.sdf_sha256: {world_file} changed")
        return result

    def scenario(self, value, name):
        _keys(value, ("file", "key"), name)
        key = value["key"]
        if key is not None:
            _text(key, name + ".key")
        return Scenario(self.file(value["file"], name + ".file"), key)

    def reference(self, value, name, identity, fuel):
        _keys(value, ("metric_version", "id", "metadata_sha256"), name)
        version = value["metric_version"]
        if type(version) is not int or version < 1:
            raise ValueError(f"{name}.metric_version: expected a positive integer")
        pin = ReferencePin(
            version,
            _text(value["id"], name + ".id"),
            _digest(value["metadata_sha256"], name + ".metadata_sha256"),
            # The name rule of coverage.reference_file: <world>_v<metric version>.json.
            (self.references / f"{identity}_v{version}.json").resolve(),
        )
        if not self.pinned:
            return pin
        try:
            data = pin.metadata_file.read_bytes()
        except OSError as exc:
            raise ValueError(f"{name}: reference unreadable: {exc}") from exc
        if hashlib.sha256(data).hexdigest() != pin.metadata_sha256:
            raise ValueError(
                f"{name}.metadata_sha256: {pin.metadata_file.name} does not match"
            )
        try:
            metadata = json.loads(data)
        except ValueError as exc:
            raise ValueError(f"{name}: reference is not JSON: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{name}: reference metadata is not a mapping")
        for field, expected in (
            ("world", identity),
            ("id", pin.id),
            ("metric_version", version),
        ):
            if metadata.get(field) != expected:
                raise ValueError(
                    f"{name}: {pin.metadata_file.name} has {field} "
                    f"{metadata.get(field)!r}, expected {expected!r}"
                )
        # Every Fuel model the reference hashes must be pinned for fetching, and only
        # those: keys are "fuel:<model URI>#<file>" as the builder writes them.
        sources = metadata.get("source_sha256")
        if not isinstance(sources, dict):
            raise ValueError(f"{name}: {pin.metadata_file.name} has no source_sha256")
        hashed = {
            _fuel_identity(key[5:].split("#", 1)[0])
            for key in sources
            if key.startswith("fuel:")
        }
        if hashed != {_fuel_identity(uri) for uri in fuel}:
            raise ValueError(
                f"{name}: the world's fuel_models differ from the Fuel models of "
                f"{pin.metadata_file.name}: {', '.join(sorted(hashed)) or 'none'}"
            )
        return pin

    def robot_fit(self, value, name):
        if not isinstance(value, dict):
            raise ValueError(f"{name}: expected a mapping of robot IDs")
        fits = []
        for robot_id, fit in value.items():
            if robot_id not in self.robots:
                raise ValueError(f"{name}.{robot_id}: unknown robot (robots.yaml)")
            fit = {} if fit is None else fit
            _keys(fit, (), f"{name}.{robot_id}", optional=("spawn_z",))
            height = fit.get("spawn_z")
            fits.append(
                RobotFit(
                    robot_id,
                    (
                        None
                        if height is None
                        else _float(height, f"{name}.{robot_id}.spawn_z")
                    ),
                )
            )
        return tuple(fits)

    def fuel(self, value, name):
        if not isinstance(value, list):
            raise ValueError(f"{name}: expected a list")
        uris = tuple(fuel_model(uri, name) for uri in value)
        if len({_fuel_identity(uri) for uri in uris}) != len(uris):
            raise ValueError(f"{name}: duplicate model URIs")
        return uris

    def resources(self, value, name):
        if not isinstance(value, list):
            raise ValueError(f"{name}: expected a list")
        return tuple(
            self.file(item, f"{name}[{index}]", directory=True)
            for index, item in enumerate(value)
        )

    def moving_objects(self, value, name):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"{name}: expected a mapping or null")
        try:
            return json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: expected plain JSON data: {exc}") from exc

    def descriptors(self, value, name):
        if value is None:
            return None
        _keys(value, ("file", "sha256"), name)
        result = Descriptors(
            self.file(value["file"], name + ".file"),
            _digest(value["sha256"], name + ".sha256"),
        )
        if file_sha256(self.root / result.file) != result.sha256:
            raise ValueError(f"{name}.sha256: {result.file} changed")
        return result
