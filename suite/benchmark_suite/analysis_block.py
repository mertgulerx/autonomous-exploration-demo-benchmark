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
is ``analysis`` holds the parameters the analysis module reads (``analysis-block-v1``, or ``-v2`` with the optional
``event_tick`` and ``taxonomy`` keys of the figure extension, FX-19 and FX-20). The engine validates the
block at submission and copies the plan and the parsed block into a study export; the analysis module fills
documented defaults for what a block leaves out (RO-16) and never reads the checkout.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import math
import re
from typing import Any

import yaml

from .contracts import ContractError
from .yaml_io import load_mapping

# The versions, keys, names and defaults below are mirrored in analysis/benchmark_analysis/settings.py (a separate
# environment that cannot import the suite): change both.
ANALYSIS_BLOCK_VERSIONS = (1, 2)
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
V2_KEYS = {"event_tick", "taxonomy"}  # version 2 only
CATEGORY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")  # a fenced code block's opening line (CommonMark)
YAML_INFO = ("yaml", "yml")
TOP_LEVEL_ANALYSIS = re.compile(r"^analysis\s*:", re.MULTILINE)


def _closes(match: re.Match[str], opening: str) -> bool:
    marker, rest = match.group(1), match.group(2)
    return marker[0] == opening[0] and len(marker) >= len(opening) and not rest.strip()


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


def validate(block: dict[str, Any], *, confirmatory: bool) -> dict[str, Any]:
    """The block with every given field checked (defaults are the analysis module's, RO-16); an unquoted
    ``plan_registered_utc``, which YAML reads as a UTC time, comes back as its text."""
    version = block.get("version")
    if type(version) is not int or version not in ANALYSIS_BLOCK_VERSIONS:
        raise ContractError("analysis.version", "Use version 1 or 2")
    unknown = set(block) - KEYS - (V2_KEYS if version == 2 else set())
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
    hypotheses = block.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        raise ContractError("analysis.hypotheses", "Expected a list")
    seen = set()
    for index, hypothesis in enumerate(hypotheses):
        field = f"analysis.hypotheses[{index}]"
        if not isinstance(hypothesis, dict) or set(hypothesis) - {
            "id",
            "endpoint",
            "world",
            "profile",
            "horizon_s",
        }:
            raise ContractError(
                field, "Use id, endpoint, world, profile and, for C_h, A_h and D_h, horizon_s"
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
