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
from typing import Any, Literal, Mapping, TypedDict

PROTOCOL_VERSION = 2  # 2: reports.generate removed (RESULTS_OVERHAUL R7)
# Versions whose requests mean the same here (2 only removed a method): the engine accepts both, and a client
# falls back to 1 for an engine image built before R7 (it may still run a campaign while the GUI is newer).
COMPATIBLE_PROTOCOLS = frozenset({1, 2})
# 10: receipt clock, odom_aligned_* names, goal_facts (RESULTS_OVERHAUL R2); 11: `navigation` rows with
# event "recovery" (world integration I5; no Arrow field changed)
DATASET_VERSION = 11
GAZEBO_SEED = 1234
DEFAULT_LIMIT = 1800.0
# Time-limit rules a campaign may name (``time_limit_rule``; I4 §3.3); CampaignSpec.accepted is the one
# check. TIME_LIMIT_RULE equals exploration_benchmark_runner.time_limits.TIME_LIMIT_RULE, the rule that
# Library.resolve_campaign computes (not imported: the GUI loads this module without the runner). A retired
# rule stays valid in stored specifications with their stored limits (Restart); new campaigns are refused it.
TIME_LIMIT_RULE = "kinematic-time-limit-v2"
RETIRED_TIME_LIMIT_RULES = frozenset({"kinematic-time-limit-v1"})
TIME_LIMIT_RULES = frozenset({TIME_LIMIT_RULE}) | RETIRED_TIME_LIMIT_RULES
NO_MOVEMENT_LIMIT = 180.0
HIDDEN_GROUPS = frozenset({"Suppression", "QoS", "Visualization", "Benchmark Integration"})
# Most campaigns or profiles one deletion request may name; the GUI checks the same bounds.
CAMPAIGN_DELETION_LIMIT = 100
PROFILE_DELETION_LIMIT = 250
# The final outcomes counted as reported (reported-outcomes-v3, docs/MEASUREMENTS.md "Reported population": the
# v1 classes plus stuck, stuck-stop-v1); run records apply it in run_record.report_class, which adds the explorer
# failures of reported-outcomes-v4 (explorer_failure below).
REPORTED_OUTCOMES = frozenset({"completed", "time_limit", "no_movement", "stuck"})
# Outcomes whose run was measured to its end (eligibility, completion and milestone censoring, C(L)); the same set
# as exploration_benchmark_runner.coverage.MEASURED_OUTCOMES (this module stays free of runner imports).
MEASURED_OUTCOMES = frozenset({"completed", "time_limit", "stuck"})
STOPPED_OUTCOMES = frozenset({"manual_stop"})
# Failure sources (contract C1, decision B1; run-record-v7): the worker names what caused a startup or runtime
# failure in the manifest's failure_source, "explorer" (the explorer process exited, returned an error result
# or refused its start protocol) or "stack" (simulation, Nav2, SLAM, streams, readiness). reported-outcomes-v4:
# a final explorer failure is a reported, unsuccessful outcome of its method (EXPLORER_FAILURE); a stack
# failure, or one whose manifest names no source, keeps its v3 class.
EXPLORER_SOURCE = "explorer"
# Our explorer's package, the explorer_id of its runs (run-record-v5).
OUR_EXPLORER = "frontier_exploration_ros2"
FAILURE_SOURCES = frozenset({EXPLORER_SOURCE, "stack"})
SOURCED_FAILURES = frozenset({"startup_failure", "runtime_failure"})
EXPLORER_FAILURE = "explorer_failure"


def explorer_failure(outcome: str | None, failure_source: object) -> bool:
    """A startup or runtime failure the worker attributed to the explorer (reported-outcomes-v4)."""
    return outcome in SOURCED_FAILURES and failure_source == EXPLORER_SOURCE


def measured_as(outcome: str | None, failure_source: object, motion_started: object) -> str:
    """The reported outcome whose measurement rules a run follows (C1): an explorer failure after movement
    onset as ``stuck`` (censored at L, coverage carried from its end to L), one before movement onset as
    ``no_movement`` (RD-8: pre-motion coverage held, R = L); every other outcome as itself."""
    if explorer_failure(outcome, failure_source):
        return "stuck" if motion_started is True else "no_movement"
    return outcome or ""


# RD-6's gap rule: a longer stretch without a valid coverage sample breaks a line and gives no value across it
# (the analysis package keeps its own copy, analysis/benchmark_analysis/data.py, with a 1e-9 float tolerance).
GAP_S = 2.0


def milestone_key(percent: float) -> str:
    """A coverage milestone's name key (coverage-stop-v2): "25" … "95", "99_9". Record fields are
    ``t{key}_s``, ``t{key}_event`` and ``t{key}_censor_s``, the export endpoint ``t{key}`` and the analysis
    endpoint ``R_{key}``."""
    return format(percent, "g").replace(".", "_")


def average_speed(distance_m: Any, elapsed_seconds: Any) -> float | None:
    """Average speed in m/s (``docs/MEASUREMENTS.md``): the raw-odometry distance over the simulation time since
    the first movement. None when either is unavailable or not finite, the distance is negative or no
    simulation time has passed (runs that never moved)."""
    values = (distance_m, elapsed_seconds)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        return None
    if not all(math.isfinite(v) for v in values) or distance_m < 0 or elapsed_seconds <= 0:
        return None
    return float(distance_m) / float(elapsed_seconds)


def report_exclusion(outcome: str | None) -> str | None:
    """None when the outcome is a reported one, otherwise "stopped" or "failure" (the Campaigns outcome bar)."""
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


class RuleValue(TypedDict):
    """One rule value of a robot in a world: the catalog's ``rule_time_limits`` entries and the accepted
    specification's ``time_limit_inputs`` entries (I4 §3.3)."""

    seconds: int
    uncapped_seconds: int
    capped: bool
    area_m2: float
    range_m: float
    speed_mps: float


RULE_VALUE_FIELDS = tuple(RuleValue.__annotations__)


def resolve_time_limit_inputs(
    value: Any,
    limits: Iterable[tuple[str, float]],
    retained: set[tuple[str, str]],
    *,
    rule: str | None,
) -> str | None:
    """The ``time_limit_inputs`` (world -> robot -> RuleValue) as canonical JSON, checked against the
    campaign: present exactly with a rule, one entry per selected world naming exactly the robots with a
    retained combination in it (``retained``: (robot, world) pairs), capped pairs run at the cap (robot set
    R5, 25 October 2026: their unfinished runs are right-censored at it), one limit per world, equal to ``world_time_limits``. ``Library.resolve_campaign`` computes the values;
    a stored specification (restart) is checked as stored. The one place these refusals live."""
    if rule is None:
        if value is not None:
            raise ContractError(
                "time_limit_inputs", "Only a campaign with time_limit_rule has them"
            )
        return None
    worlds = mapping(value, "time_limit_inputs")
    expected = dict(limits)
    if set(worlds) != set(expected):
        raise ContractError("time_limit_inputs", "Give the rule's inputs for every selected world")
    for world, robots in worlds.items():
        field = "time_limit_inputs." + world
        running = {robot for robot, used in retained if used == world}
        if not running:
            raise ContractError(
                "worlds." + world, f"Every combination of {world} is excluded; remove the world"
            )
        if set(mapping(robots, field)) != running:
            raise ContractError(
                field,
                "Give the rule's values of the robots that run it: " + ", ".join(sorted(running)),
            )
        for robot, entry in robots.items():
            where = f"{field}.{robot}"
            if set(mapping(entry, where)) != set(RULE_VALUE_FIELDS):
                raise ContractError(where, "Expected " + ", ".join(RULE_VALUE_FIELDS))
            seconds = integer(entry["seconds"], where + ".seconds", 1, 2**31 - 1)
            uncapped = integer(entry["uncapped_seconds"], where + ".uncapped_seconds", 1, 2**31 - 1)
            if not isinstance(entry["capped"], bool):
                raise ContractError(where + ".capped", "Expected true or false")
            if entry["capped"] and uncapped <= seconds:
                raise ContractError(where, "A capped limit lies below its uncapped value")
            if not entry["capped"] and uncapped != seconds:
                raise ContractError(where, "Below the cap, the uncapped value is the limit")
            for key in ("area_m2", "range_m", "speed_mps"):
                if number(entry[key], f"{where}.{key}") <= 0:
                    raise ContractError(f"{where}.{key}", "Expected a positive number")
        found = {robot: entry["seconds"] for robot, entry in robots.items()}
        if len(set(found.values())) > 1:
            listed = ", ".join(f"{robot} {seconds} s" for robot, seconds in found.items())
            raise ContractError(
                "world_time_limits." + world,
                f"The rule gives {world} different limits ({listed}); split the campaign by robot",
            )
        seconds = next(iter(found.values()))
        if seconds != expected[world]:
            raise ContractError(
                "world_time_limits." + world,
                f"{world}: {expected[world]:g} s differs from the rule's {seconds} s ({rule})",
            )
    return canonical(worlds)


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


STUDY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# The study of campaigns without a study block (exploratory; RD-29).
UNASSIGNED = "unassigned"
EvidenceLevel = Literal["exploratory", "confirmatory"]
EVIDENCE_LEVELS: tuple[EvidenceLevel, ...] = ("exploratory", "confirmatory")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class Study:
    """The study a campaign belongs to (RESULTS_RESEARCH RD-29, RD-30).

    The researcher sets ``id``, ``evidence_level`` and ``plan``; ``plan_sha256`` (the plan above its
    deviations log) and ``first_repetition`` (study-wide repetition numbers) are set at submission and
    kept by a restarted copy, which replaces its source's blocks.
    """

    id: str
    evidence_level: EvidenceLevel
    plan: str | None
    plan_sha256: str | None = None
    first_repetition: int = 0

    @classmethod
    def accepted(cls, value: dict[str, Any]) -> "Study":
        known_keys(
            value, {"id", "evidence_level", "plan", "plan_sha256", "first_repetition"}, "study"
        )
        identity = text(value.get("id"), "study.id", maximum=64)
        if not STUDY_ID.match(identity) or identity == UNASSIGNED:
            raise ContractError(
                "study.id", "Use letters, digits, '.', '_' or '-' (not 'unassigned'), at most 64"
            )
        level = value.get("evidence_level", "exploratory")
        if level == "exploratory":
            evidence: EvidenceLevel = "exploratory"
        elif level == "confirmatory":
            evidence = "confirmatory"
        else:
            raise ContractError("study.evidence_level", "Use exploratory or confirmatory")
        plan = value.get("plan")
        if plan is not None:
            plan = text(plan, "study.plan", maximum=300)
            if plan.startswith("/") or ".." in plan.split("/") or not plan.endswith(".md"):
                raise ContractError(
                    "study.plan", "Name a Markdown plan by its path in the checkout"
                )
        if evidence == "confirmatory" and plan is None:
            raise ContractError("study.plan", "A confirmatory study names its pre-registered plan")
        digest = value.get("plan_sha256")
        if digest is not None and (not isinstance(digest, str) or not SHA256.match(digest)):
            raise ContractError("study.plan_sha256", "Expected a SHA-256 hex digest")
        return cls(
            identity,
            evidence,
            plan,
            digest,
            integer(value.get("first_repetition", 0), "study.first_repetition", 0, 2**31 - 1),
        )

    def dictionary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "evidence_level": self.evidence_level,
            "plan": self.plan,
            "plan_sha256": self.plan_sha256,
            "first_repetition": self.first_repetition,
        }


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
    # A tuning run: set only by tuning.start (submitted specifications reject the key); hidden from lists
    # and replaced by the next tuning run.
    tuning: bool = False
    study: Study | None = None
    # Nav2 feedback and command rows (RESULTS_RESEARCH RD-42): a run condition held within a study.
    debug_streams: bool = False
    # The Default profile's parameter hash per robot, set at submission (``is_baseline`` in run records).
    default_parameter_hashes: tuple[tuple[str, str], ...] = ()
    # Each external explorer's Default parameter hash per robot, set at submission for the explorers the campaign
    # runs on that robot (profiles.yaml explorer ID -> robot -> hash; A7: a study's campaigns are compared by it).
    # Campaigns accepted before it have none.
    explorer_default_hashes: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    # ``time_limit_rule`` and its stored inputs (canonical JSON, world -> robot -> RULE_VALUE_FIELDS; I4 §3.3):
    # set at submission and kept as stored by a restarted copy.
    time_limit_rule: str | None = None
    time_limit_inputs_json: str | None = None

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
                "study",
                "debug_streams",
                "default_parameter_hashes",
                "explorer_default_hashes",
                "time_limit_rule",
                "time_limit_inputs",
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
        debug_streams = value.get("debug_streams", False)
        if not isinstance(debug_streams, bool):
            raise ContractError("debug_streams", "Expected true or false")
        hashes = mapping(value.get("default_parameter_hashes", {}), "default_parameter_hashes")
        for robot_id, digest in hashes.items():
            if not isinstance(digest, str) or not SHA256.match(digest):
                raise ContractError(
                    "default_parameter_hashes." + robot_id, "Expected a SHA-256 hex digest"
                )
        explorer_hashes = []
        for explorer, by_robot in mapping(
            value.get("explorer_default_hashes", {}), "explorer_default_hashes"
        ).items():
            field = "explorer_default_hashes." + explorer
            for robot_id, digest in mapping(by_robot, field).items():
                if not isinstance(digest, str) or not SHA256.match(digest):
                    raise ContractError(f"{field}.{robot_id}", "Expected a SHA-256 hex digest")
            explorer_hashes.append((explorer, tuple(sorted(by_robot.items()))))
        description = value.get("description", "")
        if not isinstance(description, str) or len(description) > 10000:
            raise ContractError("description", "Use at most 10000 characters")
        rule = value.get("time_limit_rule")
        if rule is not None and (not isinstance(rule, str) or rule not in TIME_LIMIT_RULES):
            raise ContractError("time_limit_rule", "Use " + ", ".join(sorted(TIME_LIMIT_RULES)))
        if rule is not None and "time_limit" in value:
            raise ContractError(
                "time_limit", "The rule sets each world's limit; remove the global time_limit"
            )
        limits = resolve_world_time_limits(value, (world for world, _ in worlds))
        skipped = set(excluded)
        retained = {
            (r, w)
            for r, _ in robots
            for w, spawns in worlds
            if any((r, w, s.id) not in skipped for s in spawns)
        }
        spec = cls(
            text(value.get("name"), "name"),
            description,
            tuple(sorted(robots)),
            tuple(sorted(worlds)),
            integer(value.get("repetitions", 1), "repetitions", 1, 2**31 - 1),
            integer(value.get("concurrency", 1), "concurrency", 1, 4),
            limits,
            tuple(sorted(excluded)),
            text(value["source_campaign_id"], "source_campaign_id")
            if value.get("source_campaign_id") is not None
            else None,
            study=Study.accepted(mapping(value["study"], "study"))
            if value.get("study") is not None
            else None,
            debug_streams=debug_streams,
            default_parameter_hashes=tuple(sorted(hashes.items())),
            explorer_default_hashes=tuple(sorted(explorer_hashes)),
            time_limit_rule=rule,
            time_limit_inputs_json=resolve_time_limit_inputs(
                value.get("time_limit_inputs"), limits, retained, rule=rule
            ),
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
            **({"tuning": True} if self.tuning else {}),
            **({"study": self.study.dictionary()} if self.study else {}),
            **({"debug_streams": True} if self.debug_streams else {}),
            **(
                {"default_parameter_hashes": dict(self.default_parameter_hashes)}
                if self.default_parameter_hashes
                else {}
            ),
            **(
                {
                    "explorer_default_hashes": {
                        explorer: dict(hashes) for explorer, hashes in self.explorer_default_hashes
                    }
                }
                if self.explorer_default_hashes
                else {}
            ),
            **(
                {
                    "time_limit_rule": self.time_limit_rule,
                    "time_limit_inputs": json.loads(self.time_limit_inputs_json or "{}"),
                }
                if self.time_limit_rule
                else {}
            ),
        }
