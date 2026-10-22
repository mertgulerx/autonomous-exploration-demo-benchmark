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
out (contract ``analysis-block-v1``, docs/MEASUREMENTS.md "Study analysis block"; RESULTS_OVERHAUL RO-15, RO-16).

The engine validated the block at submission and again at export (``suite/benchmark_suite/analysis_block.py``,
whose constants these mirror); this module trusts that validation for value ranges, checks the version, the key
set and the names the analysis branches on (representative rule and endpoint, sensitivity sets), and records for
every value whether it came from the plan or the defaults.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .errors import Refused
from .stats import BOOTSTRAP_MIN_CLUSTERS

BLOCK_VERSION = 1
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
)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    endpoint: str
    world: str  # a world ID or "pooled"
    profile: str
    horizon_s: float | None


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
        or block["version"] != BLOCK_VERSION
    ):
        raise Refused(f"study.yaml: analysis block version {BLOCK_VERSION} expected")
    unknown = set(block) - set(DEFAULTS) - {"version", "plan_registered_utc"}
    if unknown:
        raise Refused("study.yaml: unknown analysis keys " + ", ".join(sorted(unknown)))
    sources = {key: "plan" if key in block else "default" for key in DEFAULTS}
    value = {**DEFAULTS, **block}
    bootstrap = {**DEFAULTS["bootstrap"], **value["bootstrap"]}
    stall = {**DEFAULTS["stall"], **value["stall"]}
    representative = {**DEFAULTS["representative"], **value["representative"]}
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
        sources=MappingProxyType(sources),
    )
