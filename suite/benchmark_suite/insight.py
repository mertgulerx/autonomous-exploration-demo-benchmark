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

"""Explorer insight for the Monitor: frontier analysis ticks, goal history and explored-cell snapshots.

Ticks come from the frontier insight node (``exploration_benchmark_insight``), a re-analysis of the
explorer's inputs with the explorer's own analysis code. Positions arrive in the SLAM map frame and are
placed in the world with the attempt's ``world_from_map`` alignment, an estimate frozen at readiness.
Unavailable values are ``None``; nothing here is a report measurement.
"""

import base64
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

SWITCH_WINDOW_NS = 60_000_000_000
SNAPSHOT_INTERVAL_NS = 5_000_000_000
KEYFRAME_INTERVAL_NS = 60_000_000_000
GOAL_CAPACITY = 5000
GOAL_PAGE = 500
# Telemetry navigation events that change the goal history.
GOAL_EVENTS = ("initialized", "initialization_unavailable", "preempted", "terminal")
# Per-candidate values shared by the live layer and the recorded rows.
CANDIDATE_VALUES = (
    "size_cells",
    "gain",
    "path_cost",
    "time_cost",
    "start_cost",
    "distance_m",
    "score_order",
    "pool_rank",
    "route_position",
    "greedy_position",
)


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 10**9 + int(stamp.nanosec)


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def rank(value: int) -> int | None:
    """Message ranks are 1-based with 0 for none."""
    return int(value) if value > 0 else None


def to_world(alignment: tuple[float, float, float], x: float, y: float) -> tuple[float, float]:
    wx, wy, yaw = alignment
    c, s = math.cos(yaw), math.sin(yaw)
    return wx + c * x - s * y, wy + s * x + c * y


@dataclass(frozen=True, slots=True)
class Candidate:
    """One frontier candidate; ``x``, ``y`` is the dispatch point in the map frame."""

    id: int
    x: float
    y: float
    world: tuple[float, float] | None
    size_cells: int
    gain: float | None
    path_cost: float | None
    time_cost: float | None
    start_cost: float | None
    distance_m: float | None
    score_order: int | None
    pool_rank: int | None
    route_position: int | None
    greedy_position: int | None

    def values(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in CANDIDATE_VALUES}


@dataclass(frozen=True, slots=True)
class ChunkGrid:
    """The insight node's decision-map chunk states (row-major from the map origin)."""

    size_cells: int
    resolution: float
    origin: tuple[float, float]
    origin_world: tuple[float, float, float] | None
    map_width: int
    map_height: int
    columns: int
    rows: int
    states: bytes
    hit: int
    rebuilt: int
    reset: int

    @property
    def hit_ratio(self) -> float | None:
        """Hit chunks over hit, rebuilt and reset chunks; ``None`` without chunks."""
        total = self.hit + self.rebuilt + self.reset
        return self.hit / total if total else None


@dataclass(frozen=True, slots=True)
class InsightTick:
    """One analysis tick in plain values, with world positions when an alignment is known."""

    tick: int
    simulation_ns: int
    map_source_ns: int
    mode: str | None
    robot: tuple[float, float, float]
    robot_world: tuple[float, float, float] | None
    raw_frontiers: int
    candidates: tuple[Candidate, ...]
    route: tuple[int, ...]
    route_cost: float | None
    analysis_wall_ms: float
    skipped_ticks: int
    chunks: ChunkGrid

    @property
    def target(self) -> int | None:
        return self.route[0] if self.route else None

    @property
    def greedy_first(self) -> int | None:
        return next((c.id for c in self.candidates if c.greedy_position == 1), None)

    @property
    def pool(self) -> int:
        return sum(c.pool_rank is not None for c in self.candidates)

    def candidate(self, identity: int | None) -> Candidate | None:
        return next((c for c in self.candidates if c.id == identity), None)


def place(
    robot: tuple[float, float, float],
    origin: tuple[float, float],
    alignment: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """World poses of the robot and the chunk-grid origin; ``None`` without an alignment."""
    if alignment is None:
        return None, None
    yaw = robot[2] + alignment[2]
    robot_world = (
        *to_world(alignment, robot[0], robot[1]),
        math.atan2(math.sin(yaw), math.cos(yaw)),
    )
    return robot_world, (*to_world(alignment, *origin), alignment[2])


def convert(message: Any, alignment: tuple[float, float, float] | None) -> InsightTick:
    """Plain, world-placed values of a ``FrontierInsight`` message."""

    def world(x: float, y: float) -> tuple[float, float] | None:
        return to_world(alignment, x, y) if alignment is not None else None

    candidates = tuple(
        Candidate(
            id=int(item.id),
            x=float(item.x),
            y=float(item.y),
            world=world(item.x, item.y),
            size_cells=int(item.size_cells),
            gain=finite(item.gain),
            path_cost=finite(item.path_cost),
            time_cost=finite(item.time_cost),
            start_cost=finite(item.start_cost),
            distance_m=finite(item.distance_m),
            score_order=rank(item.score_order),
            pool_rank=rank(item.pool_rank),
            route_position=rank(item.route_position),
            greedy_position=rank(item.greedy_position),
        )
        for item in message.candidates
    )
    robot = (float(message.robot_x), float(message.robot_y), float(message.robot_yaw))
    origin = (float(message.chunk_origin_x), float(message.chunk_origin_y))
    robot_world, origin_world = place(robot, origin, alignment)
    chunks = ChunkGrid(
        size_cells=int(message.chunk_size_cells),
        resolution=float(message.chunk_resolution),
        origin=origin,
        origin_world=origin_world,
        map_width=int(message.chunk_map_width),
        map_height=int(message.chunk_map_height),
        columns=int(message.chunk_columns),
        rows=int(message.chunk_rows),
        states=bytes(message.chunk_states),
        hit=int(message.chunk_hits),
        rebuilt=int(message.chunk_rebuilt),
        reset=int(message.chunk_resets),
    )
    return InsightTick(
        tick=int(message.tick),
        simulation_ns=stamp_ns(message.header.stamp),
        map_source_ns=stamp_ns(message.map_stamp),
        mode=str(message.mode) or None,
        robot=robot,
        robot_world=robot_world,
        raw_frontiers=int(message.raw_frontier_count),
        candidates=candidates,
        route=tuple(int(identity) for identity in message.route),
        route_cost=finite(message.route_cost),
        analysis_wall_ms=float(message.analysis_wall_ms),
        skipped_ticks=int(message.skipped_ticks),
        chunks=chunks,
    )


def decisions_layer(tick: InsightTick) -> dict[str, Any]:
    """Monitor ``decisions`` layer: candidates with world positions, the route and the tick's values."""
    if tick.robot_world is None:
        raise ValueError("No map-to-world alignment for the analysis")
    candidates = []
    for c in tick.candidates:
        assert c.world is not None  # placed with the same alignment as the robot
        candidates.append(dict(id=c.id, world_x=c.world[0], world_y=c.world[1], **c.values()))
    return dict(
        tick=tick.tick,
        simulation_ns=tick.simulation_ns,
        map_source_ns=tick.map_source_ns,
        mode=tick.mode,
        robot=list(tick.robot_world),
        raw_frontiers=tick.raw_frontiers,
        candidates=candidates,
        route=list(tick.route),
        route_cost=tick.route_cost,
        target=tick.target,
        greedy_first=tick.greedy_first,
        analysis_wall_ms=tick.analysis_wall_ms,
        skipped_ticks=tick.skipped_ticks,
    )


def chunks_layer(tick: InsightTick) -> dict[str, Any]:
    """Monitor ``chunks`` layer: the chunk grid placed in the world, one state byte per chunk (row-major,
    base64)."""
    chunks = tick.chunks
    if chunks.origin_world is None:
        raise ValueError("No map-to-world alignment for the chunk grid")
    return dict(
        tick=tick.tick,
        origin=list(chunks.origin_world),
        cell_m=chunks.size_cells * chunks.resolution,
        resolution=chunks.resolution,
        size_cells=chunks.size_cells,
        map_width=chunks.map_width,
        map_height=chunks.map_height,
        columns=chunks.columns,
        rows=chunks.rows,
        states=base64.b64encode(chunks.states).decode(),
        hit=chunks.hit,
        rebuilt=chunks.rebuilt,
        reset=chunks.reset,
    )


def decision_row(tick: InsightTick) -> dict[str, Any]:
    """Fields of one ``decisions`` stream row (identity and time columns come from the recorder).

    Positions are in the map frame; the attempt manifest's ``world_from_map`` places them in the world.
    """
    chunks = tick.chunks
    return dict(
        tick=tick.tick,
        mode=tick.mode,
        robot_x=tick.robot[0],
        robot_y=tick.robot[1],
        robot_yaw=tick.robot[2],
        raw_frontiers=tick.raw_frontiers,
        candidates=len(tick.candidates),
        pool=tick.pool,
        route=list(tick.route),
        route_cost=tick.route_cost,
        target=tick.target,
        greedy_first=tick.greedy_first,
        analysis_wall_ms=tick.analysis_wall_ms,
        skipped_ticks=tick.skipped_ticks,
        map_source_ns=tick.map_source_ns,
        chunk_size_cells=chunks.size_cells,
        chunk_resolution=chunks.resolution,
        chunk_origin_x=chunks.origin[0],
        chunk_origin_y=chunks.origin[1],
        chunk_map_width=chunks.map_width,
        chunk_map_height=chunks.map_height,
        chunk_columns=chunks.columns,
        chunk_rows=chunks.rows,
        chunk_states=chunks.states,
        chunk_hits=chunks.hit,
        chunk_rebuilt=chunks.rebuilt,
        chunk_resets=chunks.reset,
    )


def candidate_rows(tick: InsightTick) -> list[dict[str, Any]]:
    """Fields of the ``frontier_candidates`` stream rows of one tick (one row per candidate, map frame)."""
    return [
        dict(tick=tick.tick, candidate=c.id, x=c.x, y=c.y, **c.values()) for c in tick.candidates
    ]


def tick_from_rows(
    decision: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    alignment: tuple[float, float, float] | None,
) -> InsightTick:
    """Rebuild a tick from its recorded ``decisions`` row and ``frontier_candidates`` rows (replay).

    The inverse of ``decision_row`` and ``candidate_rows``: placed with the same alignment, it gives the live
    layers' values.
    """
    robot = (float(decision["robot_x"]), float(decision["robot_y"]), float(decision["robot_yaw"]))
    origin = (float(decision["chunk_origin_x"]), float(decision["chunk_origin_y"]))
    robot_world, origin_world = place(robot, origin, alignment)
    return InsightTick(
        tick=int(decision["tick"]),
        simulation_ns=int(decision["clock_ns"]),
        map_source_ns=int(decision["map_source_ns"]),
        mode=decision["mode"],
        robot=robot,
        robot_world=robot_world,
        raw_frontiers=int(decision["raw_frontiers"]),
        candidates=tuple(
            Candidate(
                id=int(row["candidate"]),
                x=float(row["x"]),
                y=float(row["y"]),
                world=to_world(alignment, row["x"], row["y"]) if alignment is not None else None,
                **{name: row[name] for name in CANDIDATE_VALUES},
            )
            for row in sorted(candidates, key=lambda row: int(row["candidate"]))
        ),
        route=tuple(int(identity) for identity in decision["route"]),
        route_cost=decision["route_cost"],
        analysis_wall_ms=float(decision["analysis_wall_ms"]),
        skipped_ticks=int(decision["skipped_ticks"]),
        chunks=ChunkGrid(
            size_cells=int(decision["chunk_size_cells"]),
            resolution=float(decision["chunk_resolution"]),
            origin=origin,
            origin_world=origin_world,
            map_width=int(decision["chunk_map_width"]),
            map_height=int(decision["chunk_map_height"]),
            columns=int(decision["chunk_columns"]),
            rows=int(decision["chunk_rows"]),
            states=bytes(decision["chunk_states"]),
            hit=int(decision["chunk_hits"]),
            rebuilt=int(decision["chunk_rebuilt"]),
            reset=int(decision["chunk_resets"]),
        ),
    )


class InsightSummary:
    """Run-level insight figures for the live status. Callers serialize access."""

    def __init__(self) -> None:
        self.latest: InsightTick | None = None
        self.ticks = 0
        self.lost = 0
        self.hit_ratio_sum = 0.0
        self.hit_ratio_ticks = 0

    def add(self, tick: InsightTick) -> None:
        if self.latest is not None and tick.tick > self.latest.tick + 1:
            self.lost += tick.tick - self.latest.tick - 1
        self.latest = tick
        self.ticks += 1
        ratio = tick.chunks.hit_ratio
        if ratio is not None:
            self.hit_ratio_sum += ratio
            self.hit_ratio_ticks += 1

    def figures(self, clock_ns: int, goal: list[float] | None) -> dict[str, Any]:
        tick = self.latest
        target = tick.candidate(tick.target) if tick else None
        goal_distance = None
        if target is not None and target.world is not None and goal is not None:
            goal_distance = math.hypot(target.world[0] - goal[0], target.world[1] - goal[1])
        return dict(
            age_simulation_seconds=(clock_ns - tick.simulation_ns) / 1e9 if tick else None,
            skipped_ticks=tick.skipped_ticks if tick else None,
            cache_hit_last=tick.chunks.hit_ratio if tick else None,
            cache_hit_run=self.hit_ratio_sum / self.hit_ratio_ticks
            if self.hit_ratio_ticks
            else None,
            target=tick.target if tick else None,
            target_goal_distance_m=goal_distance,
        )


class GoalHistory:
    """Bounded history of the goals Nav2 initialized, served by update cursor. Callers serialize access.

    Every start or end of a goal takes the next update number. A reader passes the last update it holds (the
    ``goals`` layer revision) and receives the current state of the goals updated since, oldest update first.
    """

    def __init__(self) -> None:
        self.goals: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.updates = 0
        self.dropped = 0
        self.recent: deque[int] = deque()
        self.first_initialized_ns: int | None = None
        self.initialized = 0
        # World position of the goal Nav2 initialized last, until its terminal event.
        self.current: tuple[float, float] | None = None

    def apply(
        self,
        event: str,
        goal: int,
        position: tuple[float, float] | None,
        terminal_status: str | None,
        simulation_ns: int,
    ) -> None:
        """Interpret one telemetry navigation event (``position`` in the world when the goal is valid)."""
        if event == "initialized":
            self.current = position
            x, y = position if position is not None else (None, None)
            self.start(goal, x, y, simulation_ns)
        elif event == "initialization_unavailable":
            self.start(goal, None, None, simulation_ns)
        elif event == "preempted":
            self.end(goal, False, simulation_ns)
        elif event == "terminal":
            self.current = None
            self.end(goal, terminal_status == "succeeded", simulation_ns)

    def start(self, goal: int, x: float | None, y: float | None, simulation_ns: int) -> None:
        self.updates += 1
        self.goals[goal] = dict(
            goal=goal,
            x=x,
            y=y,
            start_ns=simulation_ns,
            end_ns=None,
            reached=None,
            update=self.updates,
        )
        self.goals.move_to_end(goal)
        while len(self.goals) > GOAL_CAPACITY:
            self.goals.popitem(last=False)
            self.dropped += 1
        if self.first_initialized_ns is None:
            self.first_initialized_ns = simulation_ns
        self.initialized += 1
        self.recent.append(simulation_ns)
        while self.recent and self.recent[0] <= simulation_ns - SWITCH_WINDOW_NS:
            self.recent.popleft()

    def end(self, goal: int, reached: bool, simulation_ns: int) -> None:
        entry = self.goals.get(goal)
        if entry is None or entry["end_ns"] is not None:
            return
        self.updates += 1
        entry.update(end_ns=simulation_ns, reached=reached, update=self.updates)
        self.goals.move_to_end(goal)

    def page(self, after: int) -> dict[str, Any]:
        """Goals updated after ``after``; ``revision`` is the newest update the page holds."""
        changed = sorted(
            (entry for entry in self.goals.values() if entry["update"] > after),
            key=lambda entry: entry["update"],
        )[:GOAL_PAGE]
        return dict(
            after=after,
            revision=changed[-1]["update"] if changed else after,
            last_update=self.updates,
            dropped=self.dropped,
            goals=[dict(entry) for entry in changed],
        )

    def switch_rates(self, clock_ns: int) -> tuple[float | None, float | None]:
        """Goals initialized per simulated minute over the last 60 s and since the first goal.

        Both need 60 simulation s since the first goal; before that they are ``None``.
        """
        first = self.first_initialized_ns
        if first is None or clock_ns - first < SWITCH_WINDOW_NS:
            return None, None
        recent = sum(clock_ns - SWITCH_WINDOW_NS < stamp <= clock_ns for stamp in self.recent)
        return float(recent), self.initialized * 60e9 / (clock_ns - first)


class ExploredSnapshots:
    """Explored-cell snapshots for replay: every 5 simulation s, a keyframe every 60 s, changes between.

    A keyframe stores the free mask over the reference's target cells (reference order), bit-packed; a
    change row stores the bit-packed XOR with the previous snapshot. Callers serialize access.
    """

    def __init__(self) -> None:
        self.previous: np.ndarray[Any, Any] | None = None
        self.last_ns: int | None = None
        self.keyframe_ns: int | None = None

    def due(self, clock_ns: int) -> bool:
        return (
            self.last_ns is None
            or clock_ns - self.last_ns >= SNAPSHOT_INTERVAL_NS
            or clock_ns < self.last_ns
        )

    def restart(self) -> None:
        """Make the next snapshot a keyframe (after a snapshot row was not recorded)."""
        self.previous = self.keyframe_ns = None

    def take(self, free: np.ndarray[Any, Any], clock_ns: int) -> dict[str, Any]:
        mask = np.asarray(free, dtype=bool)
        previous = self.previous
        keyframe = (
            previous is None
            or self.keyframe_ns is None
            or previous.shape != mask.shape
            or clock_ns - self.keyframe_ns >= KEYFRAME_INTERVAL_NS
            or clock_ns < self.keyframe_ns
        )
        data = mask if keyframe or previous is None else mask ^ previous
        if keyframe:
            self.keyframe_ns = clock_ns
        self.previous, self.last_ns = mask, clock_ns
        return dict(
            kind="keyframe" if keyframe else "changes",
            reference_cells=int(mask.size),
            free_cells=int(np.count_nonzero(mask)),
            changed_cells=None if keyframe else int(np.count_nonzero(data)),
            mask=np.packbits(data).tobytes(),
        )
