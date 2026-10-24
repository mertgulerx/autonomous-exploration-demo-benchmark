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

"""The analysis parameters of a study: its plan's ``analysis`` block with documented defaults for what it leaves
out (contract ``analysis-block-v1``, ``-v2`` with ``event_tick`` and ``taxonomy``, and ``-v3`` with ``families``,
``density_contrasts`` and a hypothesis ``factor``; docs/MEASUREMENTS.md "Study analysis block"; RESULTS_OVERHAUL
RO-15, RO-16; FIGURE_EXTENSION FX-19, FX-20; plans/worlds/I6 §3.8, §3.9).

The engine validated the block at submission and again at export (``suite/benchmark_suite/analysis_block.py``,
whose constants these mirror); this module trusts that validation for value ranges, checks the version, the key
set and the names the analysis branches on (representative rule and endpoint, sensitivity sets), and records for
every value whether it came from the plan or the defaults. The family and density-contrast declarations are
checked against the export's world metadata by ``families``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from .errors import Refused
from .stats import BOOTSTRAP_MIN_CLUSTERS

BLOCK_VERSIONS = (1, 2, 3)
V2_KEYS = ("event_tick", "taxonomy")  # version 2 and later
V3_KEYS = ("families", "density_contrasts")  # version 3 only
MILESTONES = (25, 50, 75, 90, 95)
SENSITIVITY = ("no_low_rtf", "exclusions_as_failures", "sequential_only", "shorter_horizons")
REPRESENTATIVE_RULES = ("SEL-1", "SEL-2", "SEL-3")
TIME_ENDPOINTS = ("R_completion", *(f"R_{m}" for m in MILESTONES))
DEFAULTS: dict[str, Any] = dict(
    horizons_s={},
    milestones=list(MILESTONES),
    hypotheses=[],
    alpha=0.05,
    bootstrap=dict(resamples=10_000, interval="percentile"),
    seed=1234,
    min_n=5,
    stall=dict(delta_pp=0.1, window_s=None),
    sensitivity=list(SENSITIVITY[:3]),
    representative=dict(rule="SEL-1", endpoint="R_completion"),
    event_tick=dict(after_s=60.0, min_candidates=3),
    taxonomy=None,
    families={},
    density_contrasts=[],
)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    endpoint: str
    world: str  # a world ID or "pooled"
    profile: str
    horizon_s: float | None
    factor: str | None = None  # version 3: a family factor (a factorial effect on rho), else None


# A family parameter's level as ``world.yaml`` writes it: text, a number, or a list of them (``hall_m``).
Scalar = str | int | float
Level = Scalar | tuple[Scalar, ...]
# A family with one named instance (F5-01): its world is analysed like a named world, with no family estimand
# (METHODOLOGY §2.5, WORLD_COLLECTION §9 decision 55); mirrored by analysis_block.SINGLE_INSTANCE_FAMILIES.
SINGLE_INSTANCE_FAMILIES = frozenset({"F5"})
RESERVED_ID = "pooled"  # a family or contrast ID may not be "pooled" (estimates' pooled hypotheses)


@dataclass(frozen=True, slots=True)
class Factor:
    """A two-level family parameter: a ``world.yaml`` ``parameters`` key with its low and high level."""

    key: str
    low: Level
    high: Level


@dataclass(frozen=True, slots=True)
class Family:
    """A generated family analysed as a random sample of its declared design (``family-sample-v1``, I6 §3.8).

    ``factors`` span a full two-level factorial over the family's worlds whose parameters equal ``fixed``;
    ``pairs`` are descriptive world contrasts (second minus first); ``interval`` False reports the family average
    and profile contrasts without an interval. It defaults to True only for a family that declares factors (a
    complete factorial, which the analysis checks); a family without factors, such as F4 (3 of 4 cells), has
    no interval unless it declares one. ``variance_split`` asks for the pre-registered ANOVA variance split
    (I6 §3.5, O-5 (b))."""

    id: str
    factors: tuple[Factor, ...]
    fixed: Mapping[str, Level]
    pairs: tuple[tuple[str, str], ...]
    interval: bool
    variance_split: bool


@dataclass(frozen=True, slots=True)
class DensityContrast:
    """The moving-object density contrast of one base world (I6 §3.9): ``moving`` minus ``base``."""

    id: str
    base: str
    moving: str


@dataclass(frozen=True, slots=True)
class Settings:
    evidence_level: str
    plan_registered_utc: str | None
    horizons_s: Mapping[str, tuple[float, ...]]  # declared horizons per world; L is always added
    milestones: tuple[int, ...]
    hypotheses: tuple[Hypothesis, ...]
    alpha: float
    resamples: int
    seed: int
    min_n: int
    stall_delta_pp: float
    stall_window_s: float | None
    sensitivity: tuple[str, ...]
    representative_rule: str
    representative_endpoint: str
    event_after_s: float  # SEL-4 (FX-20): the earliest tick a decision figure may show
    event_min_candidates: int  # SEL-4's fallback: the first tick with at least this many candidates
    taxonomy: tuple[tuple[str, str], ...] | None  # the declared codebook (FX-19): (ID, definition)
    families: tuple[Family, ...]  # version 3 (I6 §3.8)
    density_contrasts: tuple[DensityContrast, ...]  # version 3 (I6 §3.9)
    sources: Mapping[str, str]  # key -> "plan" or "default"

    @property
    def confirmatory(self) -> bool:
        return self.evidence_level == "confirmatory"

    def document(self) -> dict[str, Any]:
        """The settings as recorded in every sidecar."""
        return dict(
            evidence_level=self.evidence_level,
            plan_registered_utc=self.plan_registered_utc,
            horizons_s={world: list(values) for world, values in sorted(self.horizons_s.items())},
            milestones=list(self.milestones),
            hypotheses=[
                dict(
                    id=h.id,
                    endpoint=h.endpoint,
                    world=h.world,
                    profile=h.profile,
                    horizon_s=h.horizon_s,
                    **({} if h.factor is None else dict(factor=h.factor)),
                )
                for h in self.hypotheses
            ],
            alpha=self.alpha,
            bootstrap=dict(
                resamples=self.resamples,
                interval="percentile",
                min_clusters=BOOTSTRAP_MIN_CLUSTERS,
            ),
            seed=self.seed,
            min_n=self.min_n,
            stall=dict(delta_pp=self.stall_delta_pp, window_s=self.stall_window_s),
            sensitivity=list(self.sensitivity),
            representative=dict(
                rule=self.representative_rule, endpoint=self.representative_endpoint
            ),
            event_tick=dict(after_s=self.event_after_s, min_candidates=self.event_min_candidates),
            taxonomy=None
            if self.taxonomy is None
            else [dict(id=identity, definition=text) for identity, text in self.taxonomy],
            families={
                family.id: dict(
                    factors={f.key: [f.low, f.high] for f in family.factors},
                    fixed=dict(family.fixed),
                    pairs=[list(pair) for pair in family.pairs],
                    interval=family.interval,
                    variance_split=family.variance_split,
                )
                for family in self.families
            },
            density_contrasts=[
                dict(id=c.id, base=c.base, moving=c.moving) for c in self.density_contrasts
            ],
            sources=dict(sorted(self.sources.items())),
        )


def settings(study: dict[str, Any]) -> Settings:
    """The settings of an export's ``study.yaml``; a confirmatory study without a block is refused (RO-16)."""
    evidence = str(study.get("evidence_level") or "exploratory")
    block = study.get("analysis")
    if block is None:
        if evidence == "confirmatory":
            raise Refused(
                "A confirmatory study needs its plan's analysis block; study.yaml has none"
            )
        block = {}
    elif (
        not isinstance(block, dict)
        or type(block.get("version")) is not int
        or block["version"] not in BLOCK_VERSIONS
    ):
        raise Refused("study.yaml: analysis block version 1, 2 or 3 expected")
    version: int = block["version"] if block else 1  # no block: only the defaults apply
    allowed = set(DEFAULTS) - (set() if version >= 2 else set(V2_KEYS))
    allowed -= set() if version >= 3 else set(V3_KEYS)
    unknown = set(block) - allowed - {"version", "plan_registered_utc"}
    if unknown:
        raise Refused("study.yaml: unknown analysis keys " + ", ".join(sorted(unknown)))
    sources = {key: "plan" if key in block else "default" for key in DEFAULTS}
    value = {**DEFAULTS, **block}
    bootstrap = {**DEFAULTS["bootstrap"], **value["bootstrap"]}
    stall = {**DEFAULTS["stall"], **value["stall"]}
    representative = {**DEFAULTS["representative"], **value["representative"]}
    event_tick = {**DEFAULTS["event_tick"], **value["event_tick"]}
    taxonomy = value["taxonomy"]
    # What the code below relies on; the engine checks the same (analysis_block.validate).
    if representative["rule"] not in REPRESENTATIVE_RULES:
        raise Refused(
            f"study.yaml: representative rule {representative['rule']} is not one of SEL-1, SEL-2, SEL-3"
        )
    if representative["endpoint"] not in TIME_ENDPOINTS:
        raise Refused(
            f"study.yaml: representative endpoint {representative['endpoint']} is not a restricted time"
        )
    unknown_sets = set(value["sensitivity"]) - set(SENSITIVITY)
    if unknown_sets:
        raise Refused("study.yaml: unknown sensitivity sets " + ", ".join(sorted(unknown_sets)))
    if version < 3 and any("factor" in h for h in value["hypotheses"]):
        raise Refused("study.yaml: a hypothesis factor needs analysis block version 3")
    families = tuple(_family(str(name), spec) for name, spec in value["families"].items())
    contrasts = tuple(
        DensityContrast(str(c["id"]), str(c["base"]), str(c["moving"]))
        for c in value["density_contrasts"]
    )
    _check_units(families, contrasts, value["hypotheses"])
    return Settings(
        evidence_level=evidence,
        plan_registered_utc=block.get("plan_registered_utc"),
        horizons_s=MappingProxyType(
            {
                str(world): tuple(sorted({float(h) for h in values}))
                for world, values in value["horizons_s"].items()
            }
        ),
        milestones=tuple(sorted(int(m) for m in value["milestones"])),
        hypotheses=tuple(
            Hypothesis(
                id=str(h["id"]),
                endpoint=str(h["endpoint"]),
                world=str(h["world"]),
                profile=str(h["profile"]),
                horizon_s=float(h["horizon_s"]) if h.get("horizon_s") is not None else None,
                factor=None if h.get("factor") is None else str(h["factor"]),
            )
            for h in value["hypotheses"]
        ),
        alpha=float(value["alpha"]),
        resamples=int(bootstrap["resamples"]),
        seed=int(value["seed"]),
        min_n=int(value["min_n"]),
        stall_delta_pp=float(stall["delta_pp"]),
        stall_window_s=None if stall["window_s"] is None else float(stall["window_s"]),
        sensitivity=tuple(value["sensitivity"]),
        representative_rule=str(representative["rule"]),
        representative_endpoint=str(representative["endpoint"]),
        event_after_s=float(event_tick["after_s"]),
        event_min_candidates=int(event_tick["min_candidates"]),
        taxonomy=None
        if taxonomy is None
        else tuple((str(c["id"]), str(c["definition"])) for c in taxonomy["categories"]),
        families=families,
        density_contrasts=contrasts,
        sources=MappingProxyType(sources),
    )


def _level(value: Any) -> Level:
    return tuple(value) if isinstance(value, list) else cast(Level, value)


def _family(name: str, spec: Mapping[str, Any]) -> Family:
    """One ``families`` declaration as the engine validated it (``analysis_block._families``). Factors are
    ordered by key, so effect and interaction names never depend on the YAML's key order."""
    if name in SINGLE_INSTANCE_FAMILIES:
        raise Refused(
            f"study.yaml: {name} has one named world and no family estimand (METHODOLOGY §2.5)"
        )
    factors = spec.get("factors") or {}
    if spec.get("fixed") and not factors:
        raise Refused(f"study.yaml: family {name} declares fixed parameters without factors")
    return Family(
        id=name,
        factors=tuple(
            Factor(str(key), _level(levels[0]), _level(levels[1]))
            for key, levels in sorted(factors.items())
        ),
        fixed=MappingProxyType({str(k): _level(v) for k, v in (spec.get("fixed") or {}).items()}),
        pairs=tuple((str(a), str(b)) for a, b in spec.get("pairs") or ()),
        interval=bool(spec.get("interval", bool(factors))),
        variance_split=bool(spec.get("variance_split", False)),
    )


def _check_units(
    families: tuple[Family, ...],
    contrasts: tuple[DensityContrast, ...],
    hypotheses: list[dict[str, Any]],
) -> None:
    """What the family analysis relies on; the engine checks the same (``analysis_block._families``,
    ``_density_contrasts``, ``_unit_hypothesis``): distinct unit IDs other than "pooled", and family
    hypotheses that can be tested (I6 §3.4)."""
    ids = [f.id for f in families] + [c.id for c in contrasts]
    if len(set(ids)) != len(ids) or RESERVED_ID in ids:
        raise Refused('study.yaml: family and contrast IDs must be distinct and not "pooled"')
    by_id = {f.id: f for f in families}
    for h in hypotheses:
        family = by_id.get(str(h["world"]))
        if family is None:
            continue
        if h.get("factor") is not None:
            if len(family.factors) < 3:
                raise Refused(
                    f"study.yaml: hypothesis {h['id']}: Lenth's test needs at least 3 factors "
                    f"(7 effects); {family.id} declares {len(family.factors)}"
                )
        elif not family.interval:
            raise Refused(
                f"study.yaml: hypothesis {h['id']}: family {family.id} has no interval, so no test"
            )
