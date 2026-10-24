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

"""The goal model every figure shares: the ``goals`` table read once, one encoding per goal status and per cancel
reason, and their legends (docs/plans/FIGURE_EXTENSION.md, FX3; FIGURE_STYLE §3, §6 item 4; MEASUREMENTS "Goal
and explorer-log tables", "Explorer log formats").

- ``read_goals``: each run's goals in sequence order, from whatever columns the export's ``goals`` table holds
  (an older table without ``cancel_reason`` reads it as None), with the goals Nav2 rejected from the raw
  ``navigation`` stream (``read_rejections``): a rejected goal never became active, so it has no row.
- ``STATUS_LOOKS``: one look per ``goals.status`` for both uses. On a map a goal is a point, the shape carrying
  the status: filled when reached, open when cancelled, preempted, open at the end or other, a failed goal a
  filled cross, and an arrival the explorer ended a half-filled circle (``ARRIVED`` below; FIGURE_STYLE §6 item 4;
  ``map_style``, ``map_handles``). On a time axis it is a bar from dispatch to end, in greys and hatches, never a
  profile hue (FIGURE_STYLE §3; ``draw_goal_bars``, ``bar_handles``).
- ``ARRIVED``: on a map, a goal the explorer cancelled with reason ``completion_distance`` (its close-enough guard,
  ``goal_preemption_complete_if_within_m``: the robot arrived) is a half-filled circle of its own, apart from the
  other cancels (``map_key``); the legend names the profiles' distance (``close_enough_text``).
- ``place``: a run's goals dispatched by a time, placed in the world with the run's frozen ``world_from_map``
  (``PlacedGoal``), and the others counted by map key; one rule for every map that draws goals (F17, F22, F23).
- ``REASON_MARKERS``: the explorer's cancel reason (log format 2 only) as a marker shape, none of them a status
  shape, drawn at a cancelled goal's end on a time axis (``reason_style``).
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import pyarrow.parquet as pq

from .data import TIME_EPSILON, Row
from .export import Export
from .streams import Reads, Triple, to_world
from .style import EDGE


@dataclass(frozen=True, slots=True)
class StatusLook:
    """How a goal of one status is drawn: as a map point (``marker``, filled or open) and as a bar."""

    label: str
    marker: str
    filled: bool  # reached goals only (FIGURE_STYLE §6 item 4)
    bar_fill: str
    bar_hatch: str | None


# goals.status (goal_facts statuses) -> look, in drawing and legend order; an unknown status is drawn as "other"
STATUS_LOOKS: dict[str, StatusLook] = {
    "succeeded": StatusLook("reached", "o", True, "#4D4D4D", None),
    "canceled": StatusLook("cancelled", "o", False, "#CCCCCC", None),
    "preempted": StatusLook(
        "preempted", "^", False, "#CCCCCC", "\\\\\\\\"
    ),  # replaced by the explorer
    "failed": StatusLook("failed", "X", True, "#FFFFFF", "xxxx"),
    "open": StatusLook("open at the end", "s", False, "#FFFFFF", "////"),
    "other": StatusLook("other end status", "D", False, "#FFFFFF", "...."),
}
ENDED_BY_EXPLORER = ("canceled", "preempted")  # the statuses a cancel reason can explain
# the explorer's cancel reason codes (log format 2); shapes distinct from every status marker
REASON_MARKERS = {
    "completion_distance": "v",
    "visible_gain_exhausted": ">",
    "blocked_goal": "P",
    "blocked_no_replacement": "*",
    "blocked_all_suppressed": "h",
    "no_progress": "<",
    "not_in_snapshot": "p",
    "suppression_cleared": "H",
    "session_stopped": "d",
    "unspecified": "8",
}
UNKNOWN_REASON_MARKER = "1"  # a code this package does not know
ARRIVED = "arrived"  # map key of a goal the explorer ended by its close-enough guard (``map_key``)
ARRIVED_REASON = "completion_distance"
CLOSE_ENOUGH_PARAMETER = "goal_preemption_complete_if_within_m"
NO_REASON_MARKER = "+"  # an explorer-ended goal without a reason (log format 1, or no cancel line)
# navigation events of a goal Nav2 refused (run_record's goals_rejected)
REJECTED_EVENTS = ("rejected", "preempt_rejected")
REJECTED_STYLE: dict[str, Any] = dict(marker="|", ms=7, mew=1.0, color="#000000", linestyle="none")
GOAL_PT = 4.0  # map marks, FIGURE_STYLE §6 item 4 (4–5 pt)
WANTED_COLUMNS = (
    "attempt_id",
    "goal_sequence",
    "dispatch_s",
    "end_s",
    "status",
    "cancel_reason",
    "x",
    "y",
    "frame",
)


@dataclass(frozen=True, slots=True)
class Goal:
    """One row of the ``goals`` table (movement-relative times, simulated s)."""

    sequence: int
    dispatch_s: float | None  # None when the run has no movement onset
    end_s: float | None  # None while open at the run's end
    status: str  # a key of STATUS_LOOKS (an unknown status reads as "other")
    reason: str | None  # cancel_reason: log format 2, and only for a goal the explorer ended
    x: float | None  # goal position in ``frame`` (the map frame, not world coordinates)
    y: float | None
    frame: str | None

    @property
    def look(self) -> StatusLook:
        return STATUS_LOOKS[self.status]

    @property
    def map_key(self) -> str:
        """How the goal is drawn on a map: ``ARRIVED`` for a cancel by the close-enough guard, else its status."""
        return (
            ARRIVED if self.status == "canceled" and self.reason == ARRIVED_REASON else self.status
        )


@dataclass(frozen=True, slots=True)
class Rejections:
    """Goals Nav2 rejected in one run."""

    times_s: tuple[float, ...]  # movement-relative times of those with one
    unplaced: (
        int  # rejections without a movement-relative time, or whose stream the export does not hold
    )
    counted: (
        int | None
    )  # the run record's goals_rejected; None when navigation telemetry was not recorded


@dataclass(frozen=True, slots=True)
class GoalRecord:
    """A run's goals in sequence order and its rejected goals."""

    goals: tuple[Goal, ...]
    rejections: Rejections


def read_rejections(reads: Reads, run: Row) -> Rejections:
    """Times of Nav2's rejections (``simulation_elapsed_ns`` of the raw ``navigation`` stream), read only when the
    run record counts some."""
    counted = run.get("goals_rejected")
    if not counted:
        return Rejections((), 0, None if counted is None else 0)
    table = reads.raw(run["attempt_id"], "navigation", ["event", "simulation_elapsed_ns"])
    if table is None:
        return Rejections((), int(counted), int(counted))
    stamps = [
        ns
        for event, ns in zip(
            table.column("event").to_pylist(),
            table.column("simulation_elapsed_ns").to_pylist(),
            strict=True,
        )
        if event in REJECTED_EVENTS
    ]
    placed = tuple(sorted(ns / 1e9 for ns in stamps if ns is not None))
    return Rejections(placed, len(stamps) - len(placed), int(counted))


def _goal(row: Mapping[str, Any]) -> Goal:
    status = str(row["status"])
    return Goal(
        int(row["goal_sequence"]),
        row.get("dispatch_s"),
        row.get("end_s"),
        status if status in STATUS_LOOKS else "other",
        row.get("cancel_reason"),
        row.get("x"),
        row.get("y"),
        row.get("frame"),
    )


def read_goals(reads: Reads, runs: Iterable[Row]) -> dict[str, GoalRecord]:
    """Each run's goals and rejections, keyed by attempt; one read of the ``goals`` table (only the columns its
    schema holds) for all runs. A run without rows, or an export without the table, has no goals."""
    wanted = {str(run["attempt_id"]): run for run in runs}
    by_run: dict[str, list[Goal]] = defaultdict(list)
    if "goals.parquet" in reads.export.files:
        schema = pq.read_schema(reads.export.directory / "goals.parquet")
        columns = [c for c in WANTED_COLUMNS if c in schema.names]
        for row in reads.rows("goals", columns):
            if row["attempt_id"] in wanted:
                by_run[row["attempt_id"]].append(_goal(row))
    return {
        attempt: GoalRecord(
            tuple(sorted(by_run.get(attempt, []), key=lambda g: g.sequence)),
            read_rejections(reads, run),
        )
        for attempt, run in wanted.items()
    }


# Map points ----------------------------------------------------------------------------------------------------

# map keys (``Goal.map_key``) in legend order: the statuses with the close-enough cancels after the other cancels
MAP_KEYS = (
    "succeeded",
    "canceled",
    ARRIVED,
    *(s for s in STATUS_LOOKS if s not in ("succeeded", "canceled")),
)


@dataclass(frozen=True, slots=True)
class PlacedGoal:
    """A goal placed in the world (``place``): the goal and its world position."""

    goal: Goal
    x: float
    y: float

    @property
    def key(self) -> str:
        return self.goal.map_key


def place(
    goals: Iterable[Goal], alignment: Triple | None, end_s: float | None
) -> tuple[tuple[PlacedGoal, ...], Counter[str]]:
    """The goals dispatched by ``end_s`` (all for None, and those without a dispatch time: a run without onset),
    placed in the world frame with the run's ``world_from_map`` ``alignment``, in the given order; and the others
    of them counted by map key: a goal outside the map frame, without a pose or without the alignment is counted,
    never drawn."""
    placed: list[PlacedGoal] = []
    unplaced: Counter[str] = Counter()
    for goal in goals:
        if (
            end_s is not None
            and goal.dispatch_s is not None
            and goal.dispatch_s > end_s + TIME_EPSILON
        ):
            continue
        if alignment is None or goal.frame != "map" or goal.x is None or goal.y is None:
            unplaced[goal.map_key] += 1
            continue
        placed.append(PlacedGoal(goal, *to_world(alignment, goal.x, goal.y)))
    return tuple(placed), unplaced


def map_style(key: str) -> dict[str, Any]:
    """Keyword arguments for ``ax.plot`` of goal points of one map key (``Goal.map_key``): the status's shape,
    filled black with a white edge when reached, open white with a black edge when not (FIGURE_STYLE §6 item 4);
    a failed goal's cross is filled, as in F17, because an open cross does not read; a close-enough cancel
    (``ARRIVED``) a circle filled on its left half: ended near the goal, not reached by Nav2."""
    if key == ARRIVED:
        return dict(
            marker="o",
            ms=GOAL_PT,
            mew=0.5,
            linestyle="none",
            fillstyle="left",
            mfc="#000000",
            mfcalt="#FFFFFF",
            mec="#000000",
        )
    look = STATUS_LOOKS.get(key, STATUS_LOOKS["other"])
    colours = (
        dict(mfc="#000000", mec="#FFFFFF") if look.filled else dict(mfc="#FFFFFF", mec="#000000")
    )
    return dict(marker=look.marker, ms=GOAL_PT, mew=0.5, linestyle="none", **colours)


def close_enough_text(export: Export, profiles: Iterable[str]) -> str:
    """The close-enough distance of the given profiles as a legend words it: "0.5 m" when they share one value
    (``profiles`` table, ``goal_preemption_complete_if_within_m``), else "the profile's close-enough distance"."""
    wanted, columns = set(profiles), ["profile_id", "parameter", "value"]
    unknown = "the profile's close-enough distance"
    if not set(columns) <= set(pq.read_schema(export.directory / "profiles.parquet").names):
        return unknown  # an export without the parameter rows
    table = export.table("profiles", columns)
    values = {
        str(value)
        for profile, parameter, value in zip(*(c.to_pylist() for c in table.columns), strict=True)
        if profile in wanted and parameter == CLOSE_ENOUGH_PARAMETER and value is not None
    }
    numbers = {float(v) for v in values if _number(v)}
    if len(numbers) == 1 and len(values) == 1:
        return f"{next(iter(numbers)):g} m"
    return unknown


def _number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def map_handles(
    counts: Mapping[str, int], close_enough: str = "the close-enough distance"
) -> list[Artist]:
    """Legend entries for goal points by map key, with the drawn count; reached and cancelled always, so the
    encoding is shown; ``close_enough`` names the distance of the ``ARRIVED`` entry (``close_enough_text``)."""
    labels = {key: STATUS_LOOKS[key].label for key in STATUS_LOOKS}
    labels[ARRIVED] = f"ended by the explorer\nwithin {close_enough}"  # two lines: a column's width
    labels["canceled"] = "cancelled otherwise" if counts.get(ARRIVED) else labels["canceled"]
    return [
        Line2D([], [], label=f"{labels[key]} ({counts.get(key, 0)})", **map_style(key))
        for key in MAP_KEYS
        if counts.get(key) or key in ("succeeded", "canceled")
    ]


# Time-axis bars ------------------------------------------------------------------------------------------------


def reason_style(reason: str | None) -> dict[str, Any]:
    """Keyword arguments for ``ax.plot`` of a cancel-reason mark (light grey, black edge; a missing reason a
    black cross)."""
    if reason is None:
        return dict(marker=NO_REASON_MARKER, ms=3.5, mec="#000000", mew=0.6, linestyle="none")
    marker = REASON_MARKERS.get(reason, UNKNOWN_REASON_MARKER)
    face = STATUS_LOOKS["canceled"].bar_fill
    return dict(marker=marker, ms=3.5, mfc=face, mec="#000000", mew=0.4, linestyle="none")


def bar_span(goal: Goal, end_s: float) -> tuple[float, float] | None:
    """A goal's bar on [0, ``end_s``]: from dispatch (0 for one dispatched before movement) to its end (an open
    goal to ``end_s``); None without a dispatch time or when nothing of it lies in the range."""
    if goal.dispatch_s is None:
        return None
    start = max(0.0, goal.dispatch_s)
    stop = min(end_s, goal.end_s if goal.end_s is not None else end_s)
    return (start, stop) if stop >= start else None


def draw_goal_bars(
    ax: Any,
    record: GoalRecord,
    end_s: float,
    *,
    reasons: bool = True,
    bottom: float = 0.0,
    height: float = 0.55,
) -> set[str]:
    """Draw a run's goals on a time axis over [0, ``end_s``] (the run's end): a bar per goal (``bar_span``)
    filled by status; with ``reasons``, the cancel reason of a goal the explorer ended, marked above the bar at
    its end when that end lies in the range; Nav2's rejections as ticks. Returns what was drawn, in the keys
    ``bar_handles`` takes: statuses, ``reason:<code>`` (``reason:none`` for a missing one) and ``rejected``."""
    used: set[str] = set()
    for goal in record.goals:
        span = bar_span(goal, end_s)
        if span is None:
            continue
        look = goal.look
        ax.add_patch(
            Rectangle(
                (span[0], bottom),
                span[1] - span[0],
                height,
                facecolor=look.bar_fill,
                hatch=look.bar_hatch,
                edgecolor=EDGE,
                lw=0.3,
            )
        )
        used.add(goal.status)
        ended = goal.end_s
        if (
            reasons
            and goal.status in ENDED_BY_EXPLORER
            and ended is not None
            and 0.0 <= ended <= end_s
        ):
            ax.plot([ended], [bottom + height + 0.22], clip_on=False, **reason_style(goal.reason))
            used.add(f"reason:{goal.reason or 'none'}")
    shown = [t for t in record.rejections.times_s if 0.0 <= t <= end_s]
    if shown:
        ax.plot(shown, [bottom + height / 2] * len(shown), **REJECTED_STYLE)
        used.add("rejected")
    return used


def bar_handles(used: Iterable[str]) -> list[Artist]:
    """Legend entries for what ``draw_goal_bars`` drew: statuses, then reasons, then rejections."""
    present = set(used)
    handles: list[Artist] = [
        Patch(
            facecolor=look.bar_fill,
            hatch=look.bar_hatch,
            edgecolor=EDGE,
            lw=0.3,
            label=f"goal {look.label}",
        )
        for status, look in STATUS_LOOKS.items()
        if status in present
    ]
    for code in sorted(u.removeprefix("reason:") for u in present if u.startswith("reason:")):
        reason = None if code == "none" else code
        label = "cancel reason not recorded" if reason is None else f"ended: {code}"
        handles.append(Line2D([], [], label=label, **reason_style(reason)))
    if "rejected" in present:
        handles.append(Line2D([], [], label="goal rejected", **REJECTED_STYLE | dict(ms=6)))
    return handles
