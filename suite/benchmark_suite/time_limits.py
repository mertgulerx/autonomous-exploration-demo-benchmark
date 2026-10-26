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

"""Read-only tool: the time-limit rule's limits (``TIME_LIMIT_RULE``) of one robot per world.

``python -m benchmark_suite.time_limits --robot tb4_standard [--world ID ...]`` (run from
``suite/``) prints the inputs and values as YAML comments, then a ``world_time_limits:``
block for a campaign YAML and a ``horizons_s:`` block for a study plan's ``analysis``
block. The whole output parses as YAML. The rule is defined in docs/MEASUREMENTS.md
("Time limits and outcomes"); the horizons serve the restricted-time analysis of
docs/research/METHODOLOGY.md §4.5.

The rule itself lives in ``exploration_benchmark_runner.time_limits``, with the target
area A (``target_area_m2``) and the reference metric version it reads
(``RULE_METRIC_VERSION``). Worlds and their pinned version-2 coverage references come
from the checkout's world registry (``exploration_benchmark_runner.worlds``,
``world_inputs``); R and v come from the robot catalog profile, the world fit from the
registry (``catalog.fill_robot_profiles``). The engine's catalog reply (``rule_time_limits``)
computes the same values.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType

import yaml

from .contracts import ContractError, integer, number
from .yaml_io import load_mapping

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path("exploration_benchmark_robots/config/robots.yaml")
FIXED_HORIZONS_S = (600, 1800)  # added where shorter than L (METHODOLOGY.md §4.5)


@dataclass(frozen=True, slots=True)
class WorldInput:
    world: str
    reference: str
    target_cells: int
    resolution_m: float


@dataclass(frozen=True, slots=True)
class RobotInput:
    robot: str
    range_m: float
    speed_mps: float
    worlds: tuple[str, ...]  # the registry worlds that fit the robot (``fill_robot_profiles``)


@dataclass(frozen=True, slots=True)
class Limit:
    """Mirrors the runner's ``time_limits.TimeLimit`` (the tool loads the rule module by path, not as a package)."""

    seconds: int
    uncapped_seconds: int
    capped: bool


@dataclass(frozen=True, slots=True)
class Row:
    world: WorldInput
    limit: Limit
    fits: bool  # the registry fits the robot to the world


def checkout_module(root: Path, package: str, name: str) -> ModuleType:
    """``package.name`` from the checkout that also provides the data.

    ``.venv-suite`` has no runner or robot package installed, so the checkout's package is
    loaded by path under a private name, ``_checkout_<package>`` (its modules must import
    without ROS; relative imports resolve inside it). A checkout without the file is
    refused rather than mixed with an installed copy.
    """
    qualified = f"{package}.{name}"
    directory = root / package / package
    private = f"_checkout_{package}"
    try:
        if not (directory / f"{name}.py").is_file():
            raise ImportError(f"{directory / (name + '.py')} is not in the checkout")
        if private not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                private, directory / "__init__.py", submodule_search_locations=[str(directory)]
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"no loader for {directory}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[private] = module  # dataclasses look their module up by name
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[private]
                raise
        return importlib.import_module(f"{private}.{name}")
    except (ImportError, OSError, SyntaxError) as exc:
        raise ContractError(qualified, f"Cannot load the module: {exc}") from exc


def _attribute(module: ModuleType, name: str) -> object:
    if not hasattr(module, name):
        raise ContractError(f"{module.__name__}.{name}", "Missing")
    value: object = getattr(module, name)
    return value


def _callable(module: ModuleType, name: str) -> Callable[..., object]:
    value = _attribute(module, name)
    if not callable(value):
        raise ContractError(f"{module.__name__}.{name}", "Expected a function")
    return value


def _int(module: ModuleType, name: str) -> int:
    value = _attribute(module, name)
    if type(value) is not int:
        raise ContractError(f"{module.__name__}.{name}", "Expected an integer")
    return value


@dataclass(frozen=True, slots=True)
class Rule:
    """Typed access to ``exploration_benchmark_runner.time_limits``."""

    identifier: str
    metric_version: int  # the coverage reference version the rule reads A from
    factor: int
    floor_s: int
    cap_s: int
    _limit: Callable[..., object]
    _area: Callable[..., object]

    @classmethod
    def load(cls, root: Path) -> Rule:
        module = checkout_module(root, "exploration_benchmark_runner", "time_limits")
        identifier = _attribute(module, "TIME_LIMIT_RULE")
        if not isinstance(identifier, str):
            raise ContractError(f"{module.__name__}.TIME_LIMIT_RULE", "Expected a string")
        return cls(
            identifier,
            _int(module, "RULE_METRIC_VERSION"),
            _int(module, "FACTOR"),
            _int(module, "FLOOR_S"),
            _int(module, "CAP_S"),
            _callable(module, "kinematic_time_limit"),
            _callable(module, "target_area_m2"),
        )

    def area(self, world: WorldInput) -> Fraction:
        """The rule's A of ``world`` in m², exactly (``target_area_m2``)."""
        value = self._area(world.target_cells, world.resolution_m)
        if not isinstance(value, Fraction):
            raise ContractError(self.identifier, "The rule returned no exact area")
        return value

    def limit(self, world: WorldInput, robot: RobotInput) -> Limit:
        result = self._limit(world.target_cells, world.resolution_m, robot.range_m, robot.speed_mps)
        values = tuple(getattr(result, key, None) for key in Limit.__slots__)
        if not (type(values[0]) is int and type(values[1]) is int and type(values[2]) is bool):
            raise ContractError(self.identifier, f"Unexpected rule result {result!r}")
        return Limit(values[0], values[1], values[2])


def _positive(value: object, field: str) -> float:
    try:
        result = number(value, field)
    except OverflowError as exc:
        raise ContractError(field, "Expected a finite number") from exc
    if result <= 0:
        raise ContractError(field, "Expected a positive number")
    return result


@dataclass(frozen=True, slots=True)
class RegistryWorld:
    """What the tool reads of a registry world: its ID and pinned reference."""

    world: str
    metric_version: int
    metadata_file: Path


def registry(root: Path) -> tuple[object, list[RegistryWorld]]:
    """The checkout's world registry (pins checked on load) and its runnable worlds."""
    worlds = checkout_module(root, "exploration_benchmark_runner", "worlds")
    try:
        loaded = getattr(_attribute(worlds, "WorldRegistry"), "load")(root)
        runnable = loaded.runnable()
        result = [
            RegistryWorld(
                str(world.id),
                int(world.reference.metric_version),
                Path(world.reference.metadata_file),
            )
            for world in runnable
        ]
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        raise ContractError("world registry", str(exc)) from exc
    return loaded, result


def world_inputs(
    root: Path, rule: Rule, worlds: Sequence[str], known: Sequence[RegistryWorld]
) -> list[WorldInput]:
    """Target cells and resolution of each world's pinned version-2 coverage reference.

    ``known`` is the registry's runnable worlds (``registry``); all of them with a
    version-2 reference when ``worlds`` is empty. Reads metadata only (no grid).
    """
    metric_version = rule.metric_version
    pinned = {item.world: item for item in known}
    available = [world for world, item in pinned.items() if item.metric_version == metric_version]
    if not available:
        raise ContractError("world", f"No registry world has a version-{metric_version} reference")
    unknown = [world for world in worlds if world not in available]
    if unknown:
        raise ContractError(
            "world",
            f"No registry world with a version-{metric_version} reference: {', '.join(unknown)}; "
            f"use {', '.join(available)}",
        )
    result = []
    for world in list(dict.fromkeys(worlds)) or available:
        path = pinned[world].metadata_file
        field = f"coverage reference {path.name}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(field, f"Unreadable: {exc}") from exc
        if not isinstance(data, dict):
            raise ContractError(field, "Expected a JSON object")
        if data.get("metric_version") != metric_version or data.get("world") != world:
            raise ContractError(field, f"Expected metric_version {metric_version} for {world}")
        cells = integer(data.get("target_cells"), f"{field}.target_cells", 1, sys.maxsize)
        resolution = _positive(data.get("resolution"), f"{field}.resolution")
        area = _positive(data.get("target_area_m2"), f"{field}.target_area_m2")
        identity = data.get("id")
        if not isinstance(identity, str):
            raise ContractError(f"{field}.id", "Expected a string")
        item = WorldInput(world, identity, cells, resolution)
        if not math.isclose(area, float(rule.area(item)), rel_tol=0, abs_tol=1e-8):
            raise ContractError(f"{field}.target_area_m2", "Disagrees with cells x resolution^2")
        result.append(item)
    return result


def robot_input(root: Path, robot: str, loaded: object) -> RobotInput:
    """R and v of the robot's catalog profile; the world fit from
    the registry ``loaded`` (``registry``)."""
    path = root / CATALOG
    try:
        data = load_mapping(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError("robot catalog", f"Unreadable {path}: {exc}") from exc
    catalog = checkout_module(root, "exploration_benchmark_robots", "catalog")
    try:
        robots = _callable(catalog, "fill_robot_profiles")(data, loaded)
    except ValueError as exc:
        raise ContractError("robot catalog", str(exc)) from exc
    if not isinstance(robots, dict):
        raise ContractError("robot catalog", "Expected a robots mapping")
    profile = robots.get(robot)
    if not isinstance(profile, dict):
        raise ContractError("robot", f"Unknown robot {robot}; use {', '.join(robots)}")
    lidar = profile.get("lidar")
    if not isinstance(lidar, dict):
        raise ContractError(f"robots.{robot}", "Incomplete profile")
    range_m, speed, worlds = (
        lidar.get("range_max"),
        profile.get("max_linear_speed"),
        profile.get("worlds"),
    )
    if not isinstance(worlds, list) or not all(isinstance(w, str) for w in worlds):
        raise ContractError(f"robots.{robot}.worlds", "Expected a list of world IDs")
    return RobotInput(
        robot,
        _positive(range_m, f"robots.{robot}.lidar.range_max"),
        _positive(speed, f"robots.{robot}.max_linear_speed"),
        tuple(worlds),
    )


def horizons(limit_s: int) -> list[int | float]:
    """L/4, L/2, 3L/4 and L, plus the fixed horizons shorter than L."""
    values = {Fraction(limit_s * k, 4) for k in (1, 2, 3, 4)}
    values.update(Fraction(h) for h in FIXED_HORIZONS_S if h < limit_s)
    # Quarters of a whole second are exact in binary floating point and in YAML.
    return [int(v) if v.denominator == 1 else float(v) for v in sorted(values)]


def _decimal(value: Fraction) -> str:
    return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def render(rule: Rule, robot: RobotInput, rows: Sequence[Row]) -> str:
    # Column widths fit the longest world ID and reference ID (collection IDs are long).
    wide = max([len("world"), *(len(row.world.world) for row in rows)]) + 2
    ref = max([len("reference"), *(len(row.world.reference) for row in rows)]) + 2
    lines = [
        f"# Rule {rule.identifier}: L = min({rule.cap_s}, max({rule.floor_s}, "
        f"ceil({rule.factor} * A / (2 * R * v)))) s",
        f"# Robot {robot.robot} (catalog profile): R = {robot.range_m:g} m, "
        f"v = {robot.speed_mps:g} m/s",
        "# A = target cells x resolution^2 of the registry-pinned version-2 coverage reference",
        "#",
        f"# {'world':<{wide}}{'reference':<{ref}}{'cells':>8}{'res_m':>7}{'A_m2':>14}"
        f"{'L_s':>8}{'uncapped_s':>12}  capped  fits",
    ]
    for row in rows:
        world, limit = row.world, row.limit
        area = rule.area(world)
        lines.append(
            f"# {world.world:<{wide}}{world.reference:<{ref}}{world.target_cells:>8}"
            f"{world.resolution_m:>7g}{_decimal(area):>14}{limit.seconds:>8}"
            f"{limit.uncapped_seconds:>12}  {'yes' if limit.capped else 'no':<6}  "
            f"{'yes' if row.fits else 'no'}"
        )
    listed = [row for row in rows if row.fits]
    outside = [row.world.world for row in rows if not row.fits]
    if outside:
        lines.append(
            f"# Not fitted to {robot.robot} in the registry, left out below: {', '.join(outside)}"
        )
    limits = {row.world.world: row.limit.seconds for row in listed}
    horizon_block = {row.world.world: horizons(row.limit.seconds) for row in listed}
    lines += [
        "",
        "# Campaign YAML:",
        yaml.safe_dump({"world_time_limits": limits}, sort_keys=False),
        "# Study plan, analysis block (with shorter_horizons in sensitivity):",
        yaml.safe_dump({"horizons_s": horizon_block}, sort_keys=False, default_flow_style=None),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark_suite.time_limits",
        description="Print the time-limit rule's limits and horizons (read-only).",
    )
    parser.add_argument("--robot", required=True, help="catalog robot ID, for example tb4_standard")
    parser.add_argument(
        "--world", action="append", default=[], help="world ID (repeatable; default: all)"
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="checkout root")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    try:
        rule = Rule.load(root)
        loaded, known = registry(root)
        robot = robot_input(root, arguments.robot, loaded)
        rows = [
            Row(world, rule.limit(world, robot), world.world in robot.worlds)
            for world in world_inputs(root, rule, arguments.world, known)
        ]
    except ValueError as exc:  # ContractError and the rule's input errors
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(render(rule, robot, rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
