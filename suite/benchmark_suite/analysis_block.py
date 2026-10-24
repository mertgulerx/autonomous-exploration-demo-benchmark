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

"""A study plan's text and its machine-readable analysis block (RESULTS_OVERHAUL RO-15; contract
docs/MEASUREMENTS.md, "Study analysis block").

A plan is Markdown. Everything above its first "Deviations" heading (levels 1–3, outside code fences) is the
registered plan, hashed into ``study.plan_sha256``. One fenced YAML block in that part whose only top-level key
is ``analysis`` holds the parameters the analysis module reads (``analysis-block-v1``; ``-v2`` with the optional
``event_tick`` and ``taxonomy`` keys of the figure extension, FX-19 and FX-20; ``-v3`` adding ``families``,
``density_contrasts`` and a hypothesis ``factor`` for ``family-sample-v1``, plans/worlds/I6 §3.8, §3.9). The engine
validates the block at submission and copies the plan and the parsed block into a study export; the analysis
module fills documented defaults for what a block leaves out (RO-16) and never reads the checkout.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import itertools
import json
import math
import re
from typing import Any

import yaml

from .contracts import ContractError
from .yaml_io import load_mapping

# The versions, keys, names and defaults below are mirrored in analysis/benchmark_analysis/settings.py (a separate
# environment that cannot import the suite): change both.
ANALYSIS_BLOCK_VERSIONS = (1, 2, 3)
DEVIATIONS_HEADING = re.compile(r"^#{1,3}\s+Deviations\b", re.IGNORECASE)
MILESTONES = (25, 50, 75, 90, 95)
# One estimator and test per endpoint (RESULTS_RESEARCH RD-3, RD-13, RD-52): restricted times use the mean
# paired ΔR with the sign-flip test; C(h), A_h, D(h) and CPU the Hodges–Lehmann estimate with the Wilcoxon
# test; success the exact McNemar test with the n10 : n01 interval.
TIME_ENDPOINTS = ("R_completion", *(f"R_{m}" for m in MILESTONES))
HORIZON_ENDPOINTS = ("C_h", "A_h", "D_h")
ENDPOINTS = (*TIME_ENDPOINTS, *HORIZON_ENDPOINTS, "success", "cpu")
SENSITIVITY = ("no_low_rtf", "exclusions_as_failures", "sequential_only", "shorter_horizons")
REPRESENTATIVE_RULES = ("SEL-1", "SEL-2", "SEL-3")
KEYS = {
    "version",
    "plan_registered_utc",
    "horizons_s",
    "milestones",
    "hypotheses",
    "alpha",
    "bootstrap",
    "seed",
    "min_n",
    "stall",
    "sensitivity",
    "representative",
}
V2_KEYS = {"event_tick", "taxonomy"}  # version 2 and later
V3_KEYS = {"families", "density_contrasts"}  # version 3 only
# A family with one named instance (F5-01) has no family estimand (METHODOLOGY §2.5); mirrors the analysis
# package's settings.SINGLE_INSTANCE_FAMILIES.
SINGLE_INSTANCE_FAMILIES = frozenset({"F5"})
FAMILY_KEYS = {"factors", "fixed", "pairs", "interval", "variance_split"}
# A family or density-contrast hypothesis (I6 O-4 (a)) tests rho (R_completion), C(L) or A_L at L.
FAMILY_ENDPOINTS = ("R_completion", "C_h", "A_h")
CONTRAST_ENDPOINTS = ("R_completion", "C_h")
POOLED = "pooled"  # reserved: the pooled hypotheses of RD-16
UNIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
CATEGORY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")  # a fenced code block's opening line (CommonMark)
YAML_INFO = ("yaml", "yml")
TOP_LEVEL_ANALYSIS = re.compile(r"^analysis\s*:", re.MULTILINE)


def _closes(match: re.Match[str], opening: str) -> bool:
    marker, rest = match.group(1), match.group(2)
    return marker[0] == opening[0] and len(marker) >= len(opening) and not rest.strip()


@dataclass(frozen=True, slots=True)
class WorldFacts:
    """What block validation reads of a registry world (I1): its tier, family (tier F only), ``world.yaml``
    ``parameters`` and spawn IDs."""

    tier: str
    family: str | None
    parameters: Mapping[str, Any]
    spawns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Plan:
    registered: str  # the plan above its deviations log
    sha256: str  # ``study.plan_sha256``: SHA-256 of ``registered``
    block: dict[str, Any] | None  # the raw ``analysis`` block, if the registered part holds one


def plan_parts(text: str) -> Plan:
    """The registered part of a plan (above its deviations log), its hash and its ``analysis`` block.

    Fences follow CommonMark: an opening line of at least three backticks or tildes indented at most three
    spaces, closed only by the same character at least as long. A ``yaml`` or ``yml`` fence whose text has a
    top-level ``analysis:`` key is the block; it must parse, without duplicate keys, to that one key.
    """
    kept: list[str] = []
    blocks: list[str] = []
    current: list[str] | None = None  # lines of the YAML fence being read
    opening: str | None = None  # the open fence's marker, None outside a fence
    for line in text.splitlines(keepends=True):
        match = FENCE.match(line.rstrip("\r\n"))
        if opening is None and match:
            opening = match.group(1)
            info = match.group(2).split()
            current = [] if info and info[0].lower() in YAML_INFO else None
        elif opening is not None and match and _closes(match, opening):
            if current is not None:
                blocks.append("".join(current))
            opening, current = None, None
        elif opening is None and DEVIATIONS_HEADING.match(line):
            break
        elif current is not None:
            current.append(line)
        kept.append(line)
    found = []
    for block in blocks:
        if not TOP_LEVEL_ANALYSIS.search(block):
            continue  # another YAML block of the plan (for example the campaign conditions)
        try:
            value = load_mapping(block)
        except (ValueError, yaml.YAMLError) as exc:
            raise ContractError("analysis", f"The analysis block is not valid YAML: {exc}") from exc
        if set(value) != {"analysis"}:
            raise ContractError("analysis", "The analysis block holds only the key analysis")
        found.append(value["analysis"])
    if len(found) > 1:
        raise ContractError("study.plan", "The plan holds more than one analysis block")
    if found and not isinstance(found[0], dict):
        raise ContractError("analysis", "The analysis block must be a mapping")
    registered = "".join(kept)
    return Plan(
        registered, hashlib.sha256(registered.encode()).hexdigest(), found[0] if found else None
    )


def _number(value: Any, field: str, low: float, high: float, *, integral: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ContractError(field, "Expected a finite number")
    if integral and not isinstance(value, int):
        raise ContractError(field, "Expected an integer")
    if not low <= value <= high:
        raise ContractError(field, f"Expected a value from {low:g} to {high:g}")


def _names(value: Any, field: str, allowed: tuple[Any, ...]) -> list[Any]:
    if (
        not isinstance(value, list)
        or len(set(map(repr, value))) != len(value)
        or any(v not in allowed for v in value)
    ):
        raise ContractError(field, "Use distinct values from: " + ", ".join(map(str, allowed)))
    return value


def validate(
    block: dict[str, Any],
    *,
    confirmatory: bool,
    worlds: Mapping[str, WorldFacts] | None = None,
) -> dict[str, Any]:
    """The block with every given field checked (defaults are the analysis module's, RO-16); an unquoted
    ``plan_registered_utc``, which YAML reads as a UTC time, comes back as its text. With ``worlds`` (the
    registry's worlds by ID), version 3's families and density contrasts are also checked against them."""
    version = block.get("version")
    if type(version) is not int or version not in ANALYSIS_BLOCK_VERSIONS:
        raise ContractError("analysis.version", "Use version 1, 2 or 3")
    allowed = KEYS | (V2_KEYS if version >= 2 else set()) | (V3_KEYS if version >= 3 else set())
    unknown = set(block) - allowed
    if unknown:
        raise ContractError("analysis", "Unknown keys: " + ", ".join(sorted(unknown)))
    registered = block.get("plan_registered_utc")
    if isinstance(registered, datetime) and registered.utcoffset() == timedelta(0):
        registered = registered.strftime("%Y-%m-%dT%H:%M:%SZ")
        block = dict(block, plan_registered_utc=registered)
    if registered is not None:
        if not isinstance(registered, str) or not re.fullmatch(
            r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", registered
        ):
            raise ContractError("analysis.plan_registered_utc", "Use UTC as YYYY-MM-DDTHH:MM:SSZ")
        try:
            datetime.strptime(registered, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ContractError("analysis.plan_registered_utc", "Not a valid time") from exc
    elif confirmatory:
        raise ContractError(
            "analysis.plan_registered_utc", "A confirmatory plan states when it was registered"
        )
    horizons = block.get("horizons_s", {})
    if not isinstance(horizons, dict):
        raise ContractError(
            "analysis.horizons_s", "Map each world to a list of horizons in seconds"
        )
    for world, values in horizons.items():
        if not isinstance(values, list) or not values:
            raise ContractError(f"analysis.horizons_s.{world}", "Expected a list of horizons")
        for value in values:
            _number(value, f"analysis.horizons_s.{world}", 1e-3, 1e7)
    milestones = block.get("milestones", list(MILESTONES))
    _names(milestones, "analysis.milestones", MILESTONES)
    families = _families(block.get("families", {}), worlds)
    contrasts = _density_contrasts(block.get("density_contrasts", []), worlds, set(families))
    hypotheses = block.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        raise ContractError("analysis.hypotheses", "Expected a list")
    seen = set()
    hypothesis_keys = {"id", "endpoint", "world", "profile", "horizon_s"}
    if version >= 3:
        hypothesis_keys.add("factor")
    for index, hypothesis in enumerate(hypotheses):
        field = f"analysis.hypotheses[{index}]"
        if not isinstance(hypothesis, dict) or set(hypothesis) - hypothesis_keys:
            raise ContractError(
                field,
                "Use id, endpoint, world, profile, for C_h, A_h and D_h horizon_s"
                + (" and, for a family, factor" if version >= 3 else ""),
            )
        identity = hypothesis.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ContractError(field + ".id", "Each hypothesis needs a distinct id")
        seen.add(identity)
        if hypothesis.get("endpoint") not in ENDPOINTS:
            raise ContractError(field + ".endpoint", "Use one of: " + ", ".join(ENDPOINTS))
        for key in ("world", "profile"):
            if not isinstance(hypothesis.get(key), str) or not hypothesis[key]:
                raise ContractError(
                    f"{field}.{key}", "Expected text (a world ID or pooled; a profile ID)"
                )
        if hypothesis["world"] in families or hypothesis["world"] in contrasts:
            _unit_hypothesis(hypothesis, field, families, contrasts)
            continue
        if "factor" in hypothesis:
            raise ContractError(field + ".factor", "Only a declared family takes a factor")
        if hypothesis["endpoint"] in HORIZON_ENDPOINTS:
            _number(hypothesis.get("horizon_s"), field + ".horizon_s", 1e-3, 1e7)
            declared = horizons.get(hypothesis["world"])
            if declared is not None and hypothesis["horizon_s"] not in declared:
                raise ContractError(
                    field + ".horizon_s", "Use one of the world's horizons in horizons_s"
                )
        elif "horizon_s" in hypothesis:
            raise ContractError(field + ".horizon_s", "Only C_h, A_h and D_h take a horizon")
        if hypothesis["endpoint"].startswith("R_") and hypothesis["endpoint"] != "R_completion":
            if int(hypothesis["endpoint"][2:]) not in milestones:
                raise ContractError(
                    field + ".endpoint", "Test only a milestone listed in milestones"
                )
    if confirmatory and not hypotheses:
        raise ContractError(
            "analysis.hypotheses", "A confirmatory plan lists its tested hypotheses"
        )
    if "alpha" in block:
        _number(block["alpha"], "analysis.alpha", 1e-6, 0.5)
    bootstrap = block.get("bootstrap", {})
    if not isinstance(bootstrap, dict) or set(bootstrap) - {"resamples", "interval"}:
        raise ContractError("analysis.bootstrap", "Use resamples and interval")
    if "resamples" in bootstrap:
        _number(
            bootstrap["resamples"],
            "analysis.bootstrap.resamples",
            10_000,
            10_000_000,
            integral=True,
        )
    if bootstrap.get("interval", "percentile") != "percentile":
        raise ContractError("analysis.bootstrap.interval", "The interval is percentile (RD-11)")
    if "seed" in block:
        _number(block["seed"], "analysis.seed", 0, 2**63 - 1, integral=True)
    if "min_n" in block:
        _number(block["min_n"], "analysis.min_n", 2, 10_000, integral=True)
    stall = block.get("stall", {})
    if not isinstance(stall, dict) or set(stall) - {"delta_pp", "window_s"}:
        raise ContractError("analysis.stall", "Use delta_pp and window_s")
    if "delta_pp" in stall:
        _number(stall["delta_pp"], "analysis.stall.delta_pp", 1e-6, 100)
    if stall.get("window_s") is not None:
        _number(stall["window_s"], "analysis.stall.window_s", 1e-3, 1e7)
    if "sensitivity" in block:
        _names(block["sensitivity"], "analysis.sensitivity", SENSITIVITY)
    representative = block.get("representative", {})
    if not isinstance(representative, dict) or set(representative) - {"rule", "endpoint"}:
        raise ContractError("analysis.representative", "Use rule and endpoint")
    if representative.get("rule", "SEL-1") not in REPRESENTATIVE_RULES:
        raise ContractError(
            "analysis.representative.rule", "Use one of: " + ", ".join(REPRESENTATIVE_RULES)
        )
    if representative.get("endpoint", "R_completion") not in TIME_ENDPOINTS:
        raise ContractError("analysis.representative.endpoint", "Use a restricted time endpoint")
    _event_tick(block.get("event_tick", {}))
    if "taxonomy" in block:
        _taxonomy(block["taxonomy"])
    return block


def _canonical(value: Any) -> str:
    """A level as compared with a world's parameters: integral floats as integers, lists and tuples alike,
    recursively (YAML 3.0 equals 3). Mirrored in ``analysis/benchmark_analysis/families.py``: change both."""

    def plain(item: Any) -> Any:
        if isinstance(item, bool):
            return item
        if isinstance(item, float) and item.is_integer():
            return int(item)
        if isinstance(item, list | tuple):
            return [plain(v) for v in item]
        if isinstance(item, dict):
            return {str(k): plain(v) for k, v in item.items()}
        return item

    return json.dumps(plain(value), sort_keys=True)


def _level(value: Any, field: str) -> None:
    scalar = (str, int, float)
    if isinstance(value, bool) or not (
        isinstance(value, scalar)
        or (
            isinstance(value, list)
            and value
            and all(isinstance(v, scalar) and not isinstance(v, bool) for v in value)
        )
    ):
        raise ContractError(field, "A level is text, a number or a list of numbers or texts")
    items = value if isinstance(value, list) else [value]
    if any(isinstance(v, float) and not math.isfinite(v) for v in items):
        raise ContractError(field, "Expected finite numbers")


def _families(value: Any, worlds: Mapping[str, WorldFacts] | None) -> dict[str, dict[str, Any]]:
    """``families`` (version 3, I6 §3.8): family ID -> {factors: {key: [low, high]}, fixed: {key: level},
    pairs: [[world, world]], interval, variance_split}; checked against the registry when it is given."""
    if not isinstance(value, dict):
        raise ContractError("analysis.families", "Map each family ID to its declaration")
    for name, spec in value.items():
        field = f"analysis.families.{name}"
        if not isinstance(name, str) or not UNIT_ID.fullmatch(name) or name == POOLED:
            raise ContractError(
                field, 'A family ID is letters, digits, _, . and - (at most 64), not "pooled"'
            )
        if not isinstance(spec, dict) or set(spec) - FAMILY_KEYS:
            raise ContractError(field, "Use " + ", ".join(sorted(FAMILY_KEYS)))
        factors = spec.get("factors", {})
        if not isinstance(factors, dict):
            raise ContractError(field + ".factors", "Map each parameter key to [low, high]")
        for key, levels in factors.items():
            if not isinstance(key, str) or not key:
                raise ContractError(field + ".factors", "A factor is a world.yaml parameters key")
            if not isinstance(levels, list) or len(levels) != 2:
                raise ContractError(f"{field}.factors.{key}", "Give two levels, low first")
            for level in levels:
                _level(level, f"{field}.factors.{key}")
            if _canonical(levels[0]) == _canonical(levels[1]):
                raise ContractError(f"{field}.factors.{key}", "The two levels must differ")
        fixed = spec.get("fixed", {})
        if not isinstance(fixed, dict) or any(not isinstance(k, str) or not k for k in fixed):
            raise ContractError(field + ".fixed", "Map parameter keys to their level")
        if fixed and not factors:
            raise ContractError(field + ".fixed", "Fixed parameters select a factorial's worlds")
        for key, level in fixed.items():
            if key in factors:
                raise ContractError(f"{field}.fixed.{key}", "A factor cannot also be fixed")
            _level(level, f"{field}.fixed.{key}")
        pairs = spec.get("pairs", [])
        if not isinstance(pairs, list):
            raise ContractError(field + ".pairs", "Expected a list of [world, world]")
        seen: set[frozenset[str]] = set()
        for index, pair in enumerate(pairs):
            where = f"{field}.pairs[{index}]"
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(w, str) and w for w in pair)
            ):
                raise ContractError(where, "A pair is two world IDs")
            if pair[0] == pair[1]:
                raise ContractError(where, "A pair names two different worlds")
            if frozenset(pair) in seen:
                raise ContractError(where, "Each pair is declared once")
            seen.add(frozenset(pair))
        for key in ("interval", "variance_split"):
            if key in spec and not isinstance(spec[key], bool):
                raise ContractError(f"{field}.{key}", "Expected true or false")
        if spec.get("variance_split") and len(factors) < 2:
            raise ContractError(
                field + ".variance_split",
                "The variance split fits the main effects of at least 2 factors (I6 §3.5)",
            )
        if worlds is not None:
            if name in worlds:
                raise ContractError(field, "A family ID cannot be a world ID")
            _family_in_registry(name, factors, fixed, pairs, worlds, field)
    return value


def _family_in_registry(
    name: str,
    factors: Mapping[str, list[Any]],
    fixed: Mapping[str, Any],
    pairs: list[list[str]],
    worlds: Mapping[str, WorldFacts],
    field: str,
) -> None:
    """A declared family matches the registry: tier-F worlds of that family, every factor and fixed key in
    their parameters, one world per factorial cell, pairs within the family."""
    if name in SINGLE_INSTANCE_FAMILIES:
        raise ContractError(
            field, f"{name} has one named world and no family estimand (METHODOLOGY §2.5)"
        )
    members = sorted(w for w, facts in worlds.items() if facts.family == name)
    if not members or any(worlds[w].tier != "F" for w in members):
        raise ContractError(field, f"{name} is not a generated family of the registry")
    for index, pair in enumerate(pairs):
        if any(w not in members for w in pair):
            raise ContractError(f"{field}.pairs[{index}]", f"Name two worlds of {name}")
    if not factors:
        return
    cells: dict[tuple[int, ...], list[str]] = {}
    for world in members:
        parameters = worlds[world].parameters
        for key in [*fixed, *factors]:
            if key not in parameters:
                part = "fixed" if key in fixed else "factors"
                raise ContractError(f"{field}.{part}.{key}", f"{world} has no parameter {key}")
        if any(_canonical(parameters[k]) != _canonical(v) for k, v in fixed.items()):
            continue
        coded = []
        for key, (low, high) in factors.items():
            level = _canonical(parameters[key])
            if level not in (_canonical(low), _canonical(high)):
                raise ContractError(
                    f"{field}.factors.{key}", f"{world} has {key} = {parameters[key]!r}"
                )
            coded.append(1 if level == _canonical(high) else -1)
        cells.setdefault(tuple(coded), []).append(world)
    expected = set(itertools.product((-1, 1), repeat=len(factors)))
    if set(cells) != expected or any(len(w) != 1 for w in cells.values()):
        raise ContractError(
            field + ".factors",
            "The factors must give one world per cell of a full factorial (declare fixed parameters)",
        )


def _density_contrasts(
    value: Any, worlds: Mapping[str, WorldFacts] | None, families: set[str]
) -> dict[str, dict[str, Any]]:
    """``density_contrasts`` (version 3, I6 §3.9): a list of {id, base, moving} with registry world IDs."""
    if not isinstance(value, list):
        raise ContractError("analysis.density_contrasts", "Expected a list of {id, base, moving}")
    found: dict[str, dict[str, Any]] = {}
    for index, contrast in enumerate(value):
        field = f"analysis.density_contrasts[{index}]"
        if not isinstance(contrast, dict) or set(contrast) != {"id", "base", "moving"}:
            raise ContractError(field, "Use id, base and moving")
        identity = contrast["id"]
        if not isinstance(identity, str) or not UNIT_ID.fullmatch(identity) or identity == POOLED:
            raise ContractError(
                field + ".id", 'Letters, digits, _, . and - (at most 64), not "pooled"'
            )
        if worlds is not None and identity in worlds:
            raise ContractError(field + ".id", "A contrast ID cannot be a world ID")
        if identity in found or identity in families:
            raise ContractError(
                field + ".id", "Each contrast ID is distinct from others and from families"
            )
        for key in ("base", "moving"):
            if not isinstance(contrast[key], str) or not contrast[key]:
                raise ContractError(f"{field}.{key}", "Expected a world ID")
        if contrast["base"] == contrast["moving"]:
            raise ContractError(field, "base and moving are two different worlds")
        if worlds is not None:
            for key in ("base", "moving"):
                if contrast[key] not in worlds:
                    raise ContractError(f"{field}.{key}", f"Unknown world {contrast[key]}")
            base, moving = worlds[contrast["base"]], worlds[contrast["moving"]]
            if set(base.spawns) != set(moving.spawns):
                raise ContractError(field, "base and moving must share their spawn IDs")
        found[identity] = contrast
    return found


def _unit_hypothesis(
    hypothesis: dict[str, Any],
    field: str,
    families: Mapping[str, dict[str, Any]],
    contrasts: Mapping[str, dict[str, Any]],
) -> None:
    """A hypothesis on a family (a profile contrast, or with ``factor`` a factorial effect on rho) or on a
    density contrast; tested at L, so without ``horizon_s`` (I6 O-4 (a))."""
    if "horizon_s" in hypothesis:
        raise ContractError(field + ".horizon_s", "A family or contrast hypothesis is tested at L")
    endpoint = hypothesis["endpoint"]
    if hypothesis["world"] in contrasts:
        if "factor" in hypothesis:
            raise ContractError(field + ".factor", "Only a declared family takes a factor")
        if endpoint not in CONTRAST_ENDPOINTS:
            raise ContractError(field + ".endpoint", "Use " + " or ".join(CONTRAST_ENDPOINTS))
        return
    family = families[hypothesis["world"]]
    factors = family.get("factors", {})
    if "factor" in hypothesis:
        factor = hypothesis["factor"]
        if not isinstance(factor, str) or factor not in factors:
            raise ContractError(field + ".factor", "Use a factor the family declares")
        if len(factors) < 3:
            raise ContractError(
                field + ".factor", "Lenth's test needs at least 3 factors (7 effects; I6 §3.4)"
            )
        if endpoint != "R_completion":
            raise ContractError(field + ".endpoint", "A factorial effect is tested on R_completion")
        return
    if endpoint not in FAMILY_ENDPOINTS:
        raise ContractError(field + ".endpoint", "Use " + ", ".join(FAMILY_ENDPOINTS))
    # the analysis's default (settings._family): an interval only for a family that declares factors
    if not family.get("interval", bool(factors)):
        raise ContractError(
            field, "The family has no interval, so a profile contrast cannot be tested"
        )


def _event_tick(value: Any) -> None:
    """SEL-4's rule for the decision tick a figure shows (FX-20): the first tick at or after ``after_s`` where the
    route's first target differs from the greedy first, else the first with at least ``min_candidates``."""
    if not isinstance(value, dict) or set(value) - {"after_s", "min_candidates"}:
        raise ContractError("analysis.event_tick", "Use after_s and min_candidates")
    if "after_s" in value:
        _number(value["after_s"], "analysis.event_tick.after_s", 0, 1e7)
    if "min_candidates" in value:
        _number(
            value["min_candidates"],
            "analysis.event_tick.min_candidates",
            1,
            10_000,
            integral=True,
        )


def _taxonomy(value: Any) -> None:
    """The behaviour codebook a study declares before its runs are coded (FX-19): 2 to 20 categories, each with an
    ID and a definition."""
    categories = value.get("categories") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != {"categories"}
        or not isinstance(categories, list)
    ):
        raise ContractError("analysis.taxonomy", "Use categories: a list of {id, definition}")
    if not 2 <= len(categories) <= 20:
        raise ContractError("analysis.taxonomy.categories", "Declare 2 to 20 categories")
    seen = set()
    for index, category in enumerate(categories):
        field = f"analysis.taxonomy.categories[{index}]"
        if not isinstance(category, dict) or set(category) != {"id", "definition"}:
            raise ContractError(field, "Use id and definition")
        identity, definition = category["id"], category["definition"]
        if not isinstance(identity, str) or not CATEGORY_ID.fullmatch(identity) or identity in seen:
            raise ContractError(
                field + ".id", "Use a distinct ID of letters, digits, _ and - (at most 64)"
            )
        if not isinstance(definition, str) or not 1 <= len(definition.strip()) <= 500:
            raise ContractError(field + ".definition", "Define the category in 1 to 500 characters")
        seen.add(identity)
