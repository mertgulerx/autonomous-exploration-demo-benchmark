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

"""Results, run view (23, 24): one run's map, key values, charts on one time axis and provenance
(RESULTS_STUDY.md decisions 1, 4 and 10; RESULTS_OVERHAUL RO-20).

The view replaces the study page while it is open ("Runs" goes back with the table as it was). It reads the
run record the table already holds, the provenance it names (``results.get``), the series of
``results.timeline`` (polled while the engine builds it), the final map (``results.map``), the run's block
partners and its campaign's experiment count (``results.query``) and, for a profile other than Default, the
profile snapshot (``profiles.get``) to count its differences. Every value is a recorded one: censored times read
"not reached", unavailable values "—", and the charts draw samples as recorded (no interpolation; a stretch
without samples longer than the 2 s gap rule breaks the line). The time axis is linear to just past the run's
end and compressed from there to L (decision 10's axis break).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any
import uuid

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_tokens
from .gui_cells import MONO_TEXT, OverlayScrollBar
from .gui_controls import StatusBadge, badge_look
from .gui_maps import LiveLayers, WorldMap, decode_grey
from .gui_mapview import LiveMap
from .gui_marks import CHART_LABEL, KeyValue, KeyValueList, tick_step
from .gui_profile_preview import profile_differences
from .gui_style import duration, fixed, number
from .gui_support import POLL_MS, Transport, stale_engine_message
from .gui_theme import ThemeManager, snap
from .gui_widgets import (
    BODY,
    CARD_GAP,
    PAGE_BOTTOM,
    PAGE_LEFT,
    PAGE_TOP,
    Button,
    Card,
    CardHead,
    FlexRow,
    Rule,
    TextLine,
    TextStyle,
    Themed,
    draw_text,
    paint_glyphs,
    place,
)
from .gui_wizard import BackLink
from .monitoring import PathCursor, accept_page

LEGEND = TextStyle(12, 400, 18)
REACHED_KEY = TextStyle(12.5, 400, 18)
REACHED_SHARE = TextStyle(11.5, 400, 14)
REACHED_VALUE = TextStyle(14, 600, 18, tabular=True)
TITLE = TextStyle(20, 600, 26)
KV_LABEL = TextStyle(12.5, 400, 18)
KV_VALUE = TextStyle(12.5, 500, 18, tabular=True)
KV_STRONG = TextStyle(12.5, 600, 18)
GROUP_TITLE = TextStyle(12, 600, 18)
TRACK_TITLE = TextStyle(12.5, 600, 18)
TRACK_TEXT = TextStyle(12, 400, 18)
MILESTONES = (50, 75, 90, 95)  # the first-reached times the view shows
GAP_S = 2.0  # a longer stretch without coverage samples breaks the line (RD-6's gap rule)
MAP_WIDTH = 520  # the left column (decision 4)


class RunMapView(LiveMap):
    """``.sum .stage .map``: the final map fitted 1 px inside the stage with a ``border.strong`` ring; the
    explored cells, the ground-truth path and the robot at its end; no scale bar. The wheel zooms, dragging
    pans and the reset button fits the map again, as on the Monitor."""

    PAD = 1
    RING = "border.strong"
    ROBOT = 7
    SCALE_BAR = False

    world_changed = QtCore.Signal()  # the summary sizes the map to the world's aspect

    def set_world(self, world: WorldMap | None) -> None:
        changed = world is not self.world
        super().set_world(world)
        if changed:
            self.world_changed.emit()

    def height_for(self, width: float) -> float:
        """The stage height that fits the world exactly at ``width`` (4:3 until the world is known)."""
        if self.world is None:
            return width * 0.75
        columns, rows = self.world.size
        return (width - 2 * self.PAD) * rows / max(1, columns) + 2 * self.PAD


@dataclass
class RunMapState:
    attempt_id: str | None = None
    cursor: PathCursor = field(default_factory=PathCursor)
    first: dict[str, Any] = field(default_factory=dict)
    ready: bool = (
        False  # the whole map arrived; anything else is asked again when the run is reopened
    )


class RunMapLoader(QtCore.QObject):
    """``results.map`` for the selected run (request IDs ``run-map:<uuid>``): one request in flight, stale
    replies re-requested, building and running maps polled; the status text goes to the legend line."""

    status = QtCore.Signal(str, bool)  # text, ready
    reference = QtCore.Signal(str)  # the coverage reference the map was measured against

    def __init__(self, transport: Transport, view: RunMapView) -> None:
        super().__init__(view)
        self.transport, self.view = transport, view
        self.state = RunMapState()
        self.layers = LiveLayers()
        self.in_flight = ""
        self.requested: str | None = None
        self.unsupported = False
        self.retry = QtCore.QTimer(self)
        self.retry.setSingleShot(True)
        self.retry.timeout.connect(self.request)

    @property
    def attempt_id(self) -> str | None:
        return self.state.attempt_id

    def show_attempt(self, identity: str | None, empty_text: str = "") -> None:
        if identity == self.state.attempt_id and identity is not None and self.state.ready:
            return
        self.clear()
        self.state.attempt_id = identity
        if identity is None:
            self.status.emit(empty_text or "Select a run to see its coverage and path map", False)
        elif not self.unsupported:
            self.status.emit("Loading the run map…", False)
            self.request()

    def clear(self) -> None:
        self.retry.stop()
        self.state = RunMapState()
        self.layers = LiveLayers()
        self.view.set_world(None)
        self.view.set_layers(self.layers)
        self.view.set_message("")
        if not self.unsupported:
            self.status.emit("Select a run to see its coverage and path map", False)

    def request(self) -> None:
        if self.unsupported or not self.state.attempt_id or self.in_flight:
            return
        self.in_flight = "run-map:" + uuid.uuid4().hex
        self.requested = self.state.attempt_id
        self.transport.submit(
            self.in_flight,
            "results.map",
            {"attempt_id": self.state.attempt_id, "after_sequence": self.state.cursor.applied},
        )

    def received(self, tag: str, result: Any) -> bool:
        if tag != self.in_flight:
            return tag.startswith("run-map:")
        self.in_flight = ""
        if result.get("attempt_id") != self.state.attempt_id:
            self.request()  # the selection changed while this request was out
            return True
        state = result.get("state")
        if state == "building":
            self.status.emit("Building the run map…", False)
            self.retry.start(250)
        elif state == "running":
            self.status.emit(str(result.get("reason", "Run in progress")), False)
            self.retry.start(2000)
        elif state in ("failed", "unavailable"):
            self.status.emit("Run map unavailable: " + str(result.get("reason", "")), False)
        elif state == "ready":
            try:
                self.apply(result)
            except (KeyError, TypeError, ValueError) as exc:
                self.status.emit(f"Run map unreadable: {exc}", False)
                return True
            if self.state.cursor.applied < result["last_sequence"]:
                self.request()
        return True

    def apply(self, result: dict[str, Any]) -> None:
        if result["after"] == 0:
            self.state.first = result
            reference = result.get("reference") or {}
            if reference.get("id"):
                self.reference.emit(str(reference["id"]))
            base = result.get("base")
            world = WorldMap.from_reply(base) if base else None
            self.view.set_world(world)
            explored = result.get("explored")
            pixels = decode_grey(explored["image_png"]) if explored else None
            if world is not None and pixels is not None and pixels.shape == world.pixels.shape:
                self.layers.explored = np.where(pixels > 0, 255, 0).astype(np.uint8)
            if world is None:
                self.view.set_message("The world reference changed; no base map")
        reset, vertices = accept_page(self.state.cursor, result)
        if reset:
            self.layers.path.clear()
        if vertices:
            self.layers.path.extend(vertices)
        if self.state.cursor.applied >= result["last_sequence"]:
            tail = result.get("tail")
            self.layers.tail = (float(tail[0]), float(tail[1])) if tail else None
            self.layers.pose = self.end_pose()
            self.state.ready = True
            self.status.emit(self.summary(result), True)
        self.view.set_layers(self.layers)

    def end_pose(self) -> tuple[float, float, float] | None:
        """The robot at the path's end, heading along its last step."""
        path = self.layers.path
        if not len(path):
            return None
        x, y = self.layers.tail if self.layers.tail else (path.x[-1], path.y[-1])
        for index in range(len(path) - 1, -1, -1):
            dx, dy = x - path.x[index], y - path.y[index]
            if math.hypot(dx, dy) > 1e-6:
                return x, y, math.atan2(dy, dx)
        return x, y, 0.0

    def summary(self, result: dict[str, Any]) -> str:
        path = self.layers.path
        parts = [
            f"{len(path):,} path vertices"
            if result.get("path_available")
            else "No ground-truth path"
        ]
        explored = self.state.first.get("explored")
        if explored:
            parts.append(f"explored {explored['share_percent']:.2f} % of reference cells")
        breaks = [f"{reason} {count}" for reason, count in result["breaks"].items() if count]
        if breaks:
            parts.append("breaks: " + ", ".join(breaks))
        if result["collision_samples"]:
            parts.append(f"{result['collision_samples']} samples in obstacle cells")
        if result["outside_samples"]:
            parts.append(f"{result['outside_samples']} samples outside the reference")
        dropped = result["dropped"] + self.state.cursor.dropped
        if dropped:
            parts.append(f"{dropped:,} vertices dropped")
        notes = list(result.get("notes") or [])
        return " · ".join(parts) + ("\n" + "\n".join(notes) if notes else "")

    def failed(self, tag: str, error: str) -> bool:
        if tag != self.in_flight:
            return tag.startswith("run-map:")
        self.in_flight = ""
        if self.requested != self.state.attempt_id:
            self.request()  # the failure belongs to a run that is no longer selected
            return True
        stale = stale_engine_message(
            error, "Unsupported operation: results.map", "the Results run map"
        )
        if stale:
            self.unsupported = True
            self.status.emit(stale, False)
        else:
            self.status.emit("Run map unavailable: " + error, False)
        return True

    def close_timers(self) -> None:
        self.retry.stop()


class MapLegend(Themed):
    """``.mleg``: "Ground-truth path" and "Explored in the final map" (12 px ``text.secondary``); while the
    map is not ready, its state instead. The full status is the tooltip."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.text, self.ready = "", False
        self.setFixedHeight(18)

    def set_status(self, text: str, ready: bool) -> None:
        self.text, self.ready = text, ready
        self.setToolTip(text)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        secondary = theme.color("text.secondary")
        if not self.ready:
            draw_text(painter, 0, 0, LEGEND.elide(self.text, self.width()), LEGEND, secondary)
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(theme.color("data.path"), 2.2)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QtCore.QPointF(1, 9), QtCore.QPointF(17, 9))
        painter.restore()
        x = 18 + 6 + draw_text(painter, 24, 0, "Ground-truth path", LEGEND, secondary) + 16
        coverage, free = theme.color("data.coverage"), theme.map_color("free")
        share = gui_tokens.EXPLORED_OPACITY[theme.name]

        def mix(amount: float) -> QtGui.QColor:
            return QtGui.QColor(
                round(coverage.red() * amount + free.red() * (1 - amount)),
                round(coverage.green() * amount + free.green() * (1 - amount)),
                round(coverage.blue() * amount + free.blue() * (1 - amount)),
            )

        box = QtCore.QRect(snap(x), 4, 10, 10)
        painter.fillRect(box, mix(0.45))
        painter.fillRect(box.adjusted(1, 1, -1, -1), mix(share))
        draw_text(painter, x + 16, 0, "Explored in the final map", LEGEND, secondary)


class FirstReached(Themed):
    """``.fr``: "First reached" in a 112 px column, then 50, 75, 90 and 95 % with the simulation time each was
    first reached ("not reached" when censored, "—" when unavailable); 44 px, rules above and below, a rule
    before each cell."""

    LABEL_WIDTH = 112

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.values: list[str] = ["—"] * len(MILESTONES)
        self.setFixedHeight(44)

    def set_values(self, values: Sequence[str]) -> None:
        self.values = list(values)
        self.setAccessibleDescription(
            "First reached: "
            + ", ".join(f"{share} % {value}" for share, value in zip(MILESTONES, self.values))
        )
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        line, secondary = theme.color("border.line"), theme.color("text.secondary")
        width = self.width()
        painter.fillRect(QtCore.QRect(0, 0, width, 1), line)
        painter.fillRect(QtCore.QRect(0, 43, width, 1), line)
        draw_text(painter, 0, 1 + (42 - 18) / 2, "First reached", REACHED_KEY, secondary)
        cell = (width - self.LABEL_WIDTH) / len(MILESTONES)
        block = 1 + (42 - 32) / 2
        ink = theme.color("text.ink")
        for index, (share, value) in enumerate(zip(MILESTONES, self.values, strict=True)):
            left = self.LABEL_WIDTH + index * cell
            painter.fillRect(QtCore.QRectF(snap(left), 1, 1, 42), line)
            draw_text(painter, left + 13, block, f"{share} %", REACHED_SHARE, secondary)
            if value in (
                "not reached",
                "—",
            ):  # censored or unavailable: ``.nr``, on the value's baseline
                baseline = REACHED_VALUE.baseline(block + 14)
                top = baseline - REACHED_KEY.baseline(0)
                draw_text(painter, left + 13, top, value, REACHED_KEY, secondary)
            else:
                draw_text(painter, left + 13, block + 14, value, REACHED_VALUE, ink)


# ---------- the shared time axis ----------


TICK_PX = 105  # a tick label per this many pixels of the axis' linear part at most


@dataclass(frozen=True, slots=True)
class TimeAxis:
    """Simulated seconds to x: linear from 0 to ``split``, then (when ``limit`` lies well past the run's end)
    compressed from ``split`` to ``limit`` in the last 16 % after a 10 px break (decision 10)."""

    left: float
    right: float
    split: float
    limit: float
    step: float
    LINEAR_SHARE = 0.84
    GAP = 10.0

    @classmethod
    def of(cls, end_s: float, limit_s: float | None, left: float, right: float) -> "TimeAxis":
        end = max(end_s, 1.0)
        ticks = max(3, int((right - left - cls.GAP) * cls.LINEAR_SHARE // TICK_PX))
        if limit_s is None or limit_s <= 0:  # no limit: linear to just past the end
            step = tick_step(end * 1.1, ticks)
            span = math.ceil(end * 1.1 / step) * step
            return cls(left, right, span, span, step)
        step = tick_step(end * 1.15, ticks)
        split = math.ceil(end * 1.15 / step) * step
        if split >= 0.6 * limit_s:  # the run fills most of its limit: linear to L
            return cls(left, right, limit_s, limit_s, tick_step(limit_s, ticks))
        return cls(left, right, split, limit_s, step)

    @property
    def compressed(self) -> bool:
        return self.limit > self.split

    @property
    def linear_end(self) -> float:
        """The x where the linear part ends (the right edge when not compressed)."""
        width = self.right - self.left
        return self.left + ((width - self.GAP) * self.LINEAR_SHARE if self.compressed else width)

    def x(self, seconds: float) -> float:
        width = self.right - self.left
        if not self.compressed:
            return self.left + width * seconds / self.split
        inner = width - self.GAP
        if seconds <= self.split:
            return self.left + inner * self.LINEAR_SHARE * seconds / self.split
        rest = (seconds - self.split) / (self.limit - self.split)
        return (
            self.left
            + inner * self.LINEAR_SHARE
            + self.GAP
            + inner * (1 - self.LINEAR_SHARE) * rest
        )

    def segments(self) -> list[tuple[float, float]]:
        """The axis' x ranges: one, or the linear and the compressed part."""
        if not self.compressed:
            return [(self.left, self.right)]
        return [(self.left, self.linear_end), (self.linear_end + self.GAP, self.right)]

    def ticks(self) -> list[float]:
        count = math.floor(self.split / self.step + 1e-9)  # never a tick past L
        return [index * self.step for index in range(count + 1)]


def nice_ceiling(value: float, steps: Sequence[float]) -> float:
    """The first of ``steps`` at or above ``value`` (the last one's multiple past them)."""
    for step in steps:
        if value <= step + 1e-9:
            return step
    return math.ceil(value / steps[-1]) * steps[-1]


# ---------- the Over time charts ----------


@dataclass(frozen=True, slots=True)
class Track:
    name: str
    height: int


TRACKS = (
    Track("Coverage", 196),
    Track("Goals", 34),
    Track("Dispatches", 44),
    Track("DP", 64),
    Track("Speed", 102),
)
LEGEND_HEIGHT, PAD_TOP, PAD_BOTTOM, PAD_SIDE = 30, 6, 8, 20
PLOT_LEFT, PLOT_RIGHT = 40, 12
GOAL_TOKENS = {
    "succeeded": ("data.completed", 1.0),
    "preempted": ("state.neutral", 0.55),
    "canceled": ("state.neutral", 0.55),
    "failed": ("data.failure", 1.0),
}
REACHED, ENDED, FAILED = ("succeeded",), ("preempted", "canceled"), ("failed",)


@dataclass
class Legend:
    """One ``.cl`` row: the track title, its legend items (a swatch painter and a label) and an aside."""

    title: str
    items: list[tuple[Callable[[QtGui.QPainter, float, float], float], str]]
    aside: str = ""


def rows_of(series: Any, width: int) -> list[list[Any]]:
    """A timeline series as rows of ``width`` values, rows of another shape left out."""
    if not isinstance(series, list):
        return []
    return [row for row in series if isinstance(row, list) and len(row) == width]


class RunCharts(Themed):
    """``.time .tb``: coverage, goals, explorer dispatches with preemption gates, the DP analysis and speed on
    one time axis (``TimeAxis``), each under its legend row; L as a dashed warning line through every track.

    Coverage: recorded samples as area and line (broken at invalid samples and at gaps over 2 s), milestone
    circles with drop lines, the last gain as a diamond and, for a completed run, its final value carried
    forward to L dashed (RD-6). Goals: one bar per goal, green reached, grey ended by the explorer, orange
    failed. Dispatches: a mark per explorer dispatch, the gate evaluations under them (from the R8 logging
    release on). DP: pool size and route length as steps, ticks where the DP's first frontier differs from
    greedy (an estimate). Speed: ground truth and odometry. A track without data says why."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.record: Mapping[str, Any] = {}
        self.timeline: Mapping[str, Any] | None = None
        self.message = ""
        self.setFixedHeight(PAD_TOP + sum(LEGEND_HEIGHT + t.height for t in TRACKS) + PAD_BOTTOM)

    def set_run(self, record: Mapping[str, Any]) -> None:
        self.record, self.timeline, self.message = record, None, "Loading the run's series…"
        self.update()

    def set_timeline(self, timeline: Mapping[str, Any]) -> None:
        self.timeline, self.message = timeline, ""
        self.update()

    def set_message(self, text: str) -> None:
        self.timeline, self.message = None, text
        self.update()

    # data

    def end_s(self) -> float:
        timeline = self.timeline or {}
        end = number(timeline.get("end_s"))
        if end is None:
            samples = rows_of(timeline.get("coverage"), 3)
            end = max((number(row[0]) or 0.0 for row in samples), default=0.0)
        return end

    def limit_s(self) -> float | None:
        """L on this axis; none on an activation-relative one (a run that never moved), since L counts from the
        first movement."""
        if (self.timeline or {}).get("time_origin") == "activation":
            return None
        limit = number(self.record.get("time_limit_s"))
        return limit if limit is not None and limit > 0 else None

    def axis(self) -> TimeAxis:
        left = PAD_SIDE + PLOT_LEFT
        right = self.width() - PAD_SIDE - PLOT_RIGHT
        return TimeAxis.of(self.end_s(), self.limit_s(), left, right)

    def carried_forward(self) -> bool:
        """A completed run's final value is carried forward to L (the defined carry-forward, RD-6), from its
        last sample, never across a final gap (as F1 and the traces)."""
        completion = number(self.record.get("completion_s"))
        last = next(
            (
                number(row[0])
                for row in reversed(rows_of((self.timeline or {}).get("coverage"), 3))
                if number(row[0]) is not None and number(row[1]) is not None
            ),
            None,
        )
        return (
            self.record.get("outcome") == "completed"
            and self.record.get("completion_event") is True
            and self.limit_s() is not None
            and completion is not None
            and last is not None
            and completion - last <= GAP_S
        )

    def goal_counts(self) -> tuple[int, int, int, int]:
        rows = rows_of((self.timeline or {}).get("goals"), 5)
        statuses = [row[3] for row in rows]
        reached = sum(s in REACHED for s in statuses)
        ended = sum(s in ENDED for s in statuses)
        failed = sum(s in FAILED for s in statuses)
        return reached, ended, failed, len(statuses) - reached - ended - failed

    def legends(self) -> list[Legend]:
        record = self.record
        coverage: list[tuple[Callable[[QtGui.QPainter, float, float], float], str]] = [
            (lambda p, x, y: self._line_swatch(p, x, y, "data.coverage", 2.0), "Recorded")
        ]
        if self.carried_forward():
            coverage.append(
                (
                    lambda p, x, y: self._line_swatch(
                        p, x, y, "data.coverage", 2.0, dash=(5, 4), opacity=0.75
                    ),
                    "Carried forward to L",
                )
            )
        last_gain = number(record.get("last_gain_s"))
        if last_gain is not None:
            coverage.append((self._diamond_swatch, f"Last gain {duration(last_gain)}"))
        idle = number(record.get("idle_tail_s"))
        reached, ended, failed, other = self.goal_counts()
        goals = [
            (
                lambda p, x, y: self._box_swatch(p, x, y, "data.completed", 1.0),
                f"Reached {reached}",
            ),
            (
                lambda p, x, y: self._box_swatch(p, x, y, "state.neutral", 0.55),
                f"Ended by the explorer {ended}",
            ),
            (lambda p, x, y: self._box_swatch(p, x, y, "data.failure", 1.0), f"Failed {failed}"),
        ]
        if other:
            goals.append(
                (lambda p, x, y: self._box_swatch(p, x, y, "state.neutral", 0.25), f"Open {other}")
            )
        timeline = self.timeline or {}
        kinds = {row[1] for row in rows_of(timeline.get("dispatches"), 3)}
        dispatches: list[tuple[Callable[[QtGui.QPainter, float, float], float], str]] = [
            (lambda p, x, y: self._mark_swatch(p, x, y, "frontier"), "New goal"),
            (lambda p, x, y: self._mark_swatch(p, x, y, "updated"), "Updated goal"),
            (
                lambda p, x, y: self._mark_swatch(p, x, y, "after_blocked_skip"),
                "After blocked skip",
            ),
        ]
        if kinds & {"return_to_start", "suppressed_return_to_start"}:
            dispatches.append(
                (lambda p, x, y: self._mark_swatch(p, x, y, "return_to_start"), "Return to start")
            )
        if rows_of(timeline.get("gates"), 4):
            dispatches += [
                (lambda p, x, y: self._gate_swatch(p, x, y, False), "Gate kept goal"),
                (lambda p, x, y: self._gate_swatch(p, x, y, True), "Gain exhausted"),
            ]
        dp = [
            (lambda p, x, y: self._line_swatch(p, x, y, "data.analysis", 1.6), "Pool"),
            (
                lambda p, x, y: self._line_swatch(p, x, y, "data.analysis", 1.4, dash=(3, 2)),
                "Route length",
            ),
            (self._differs_swatch, "DP first differs from greedy"),
        ]
        speed = [
            (lambda p, x, y: self._line_swatch(p, x, y, "data.path", 1.6), "Ground truth"),
            (
                lambda p, x, y: self._line_swatch(p, x, y, "text.secondary", 1.6, dash=(3, 2)),
                "Odometry",
            ),
        ]
        average = fixed(record.get("average_speed_mps"), 2)
        return [
            Legend("Coverage", coverage, f"Idle tail {duration(idle)}" if idle is not None else ""),
            Legend("Goals", goals),
            Legend("Dispatches", dispatches, "Explorer log"),
            Legend("DP", dp, "Analysis, an estimate"),
            Legend("Speed", speed, "m/s" + (f" · average {average}" if average else "")),
        ]

    # swatches: each paints at (x, middle of the row) and returns its width

    def _line_swatch(
        self,
        painter: QtGui.QPainter,
        x: float,
        middle: float,
        token: str,
        width: float,
        *,
        dash: tuple[float, float] | None = None,
        opacity: float = 1.0,
    ) -> float:
        color = self.color(token)
        color.setAlphaF(opacity)
        pen = QtGui.QPen(color, width)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        if dash:
            pen.setDashPattern([dash[0] / width, dash[1] / width])
        painter.setPen(pen)
        painter.drawLine(QtCore.QPointF(x + 1, middle), QtCore.QPointF(x + 17, middle))
        return 18.0

    def _box_swatch(
        self, painter: QtGui.QPainter, x: float, middle: float, token: str, opacity: float
    ) -> float:
        color = self.color(token)
        color.setAlphaF(opacity)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(QtCore.QRectF(x, middle - 5, 10, 10), 2, 2)
        return 10.0

    def _diamond_swatch(self, painter: QtGui.QPainter, x: float, middle: float) -> float:
        self._diamond(painter, x + 6, middle, filled=False, stroke=1.3)
        return 12.0

    def _mark_swatch(self, painter: QtGui.QPainter, x: float, middle: float, kind: str) -> float:
        width = 10.0 if kind in ("frontier", "return_to_start") else 12.0
        self._dispatch(painter, x + width / 2, middle, kind)
        return width

    def _gate_swatch(
        self, painter: QtGui.QPainter, x: float, middle: float, switched: bool
    ) -> float:
        if switched:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(self.color("text.ink"))
            painter.drawRoundedRect(QtCore.QRectF(x + 2.4, middle - 5, 3.2, 10), 1, 1)
            return 8.0
        painter.setPen(QtGui.QPen(self.color("text.tertiary"), 1.2))
        painter.drawLine(QtCore.QPointF(x + 5, middle - 4), QtCore.QPointF(x + 5, middle + 4))
        return 10.0

    def _differs_swatch(self, painter: QtGui.QPainter, x: float, middle: float) -> float:
        painter.setPen(QtGui.QPen(self.color("data.analysis"), 2))
        painter.drawLine(QtCore.QPointF(x + 4, middle - 4), QtCore.QPointF(x + 4, middle + 5))
        return 8.0

    # marks

    def _diamond(
        self, painter: QtGui.QPainter, cx: float, cy: float, *, filled: bool, stroke: float
    ) -> None:
        path = QtGui.QPainterPath(QtCore.QPointF(cx, cy - 5))
        for dx, dy in ((5, 0), (0, 5), (-5, 0)):
            path.lineTo(cx + dx, cy + dy)
        path.closeSubpath()
        ink = self.color("text.ink")
        if filled:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(ink)
        else:
            painter.setPen(QtGui.QPen(ink, stroke))
            painter.setBrush(self.color("surface.card"))
        painter.drawPath(path)

    def _dispatch(self, painter: QtGui.QPainter, x: float, y: float, kind: str) -> None:
        ink = self.color("text.ink")
        if kind == "updated":
            self._diamond(painter, x, y, filled=True, stroke=0)
        elif kind == "after_blocked_skip":
            path = QtGui.QPainterPath(QtCore.QPointF(x, y - 4.5))
            path.lineTo(x + 4.8, y + 4)
            path.lineTo(x - 4.8, y + 4)
            path.closeSubpath()
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(ink)
            painter.drawPath(path)
        elif kind in ("return_to_start", "suppressed_return_to_start"):
            painter.setPen(QtGui.QPen(ink, 1.4))
            painter.setBrush(self.color("surface.card"))
            painter.drawEllipse(QtCore.QPointF(x, y), 3.3, 3.3)
        else:  # a new frontier goal (and any kind the parser adds later)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(ink)
            painter.drawEllipse(QtCore.QPointF(x, y), 3.6, 3.6)

    # painting

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.timeline is None:
            text = BODY.elide(self.message, self.width() - 2 * PAD_SIDE)
            draw_text(painter, PAD_SIDE, PAD_TOP + 6, text, BODY, self.color("text.secondary"))
            return
        axis = self.axis()
        top = float(PAD_TOP)
        painters = (self._coverage, self._goals, self._dispatches, self._dp, self._speed)
        for legend, track, paint in zip(self.legends(), TRACKS, painters, strict=True):
            self._legend(painter, legend, top)
            top += LEGEND_HEIGHT
            painter.save()
            painter.translate(0, top)
            paint(painter, axis, track.height)
            painter.restore()
            top += track.height

    def _legend(self, painter: QtGui.QPainter, legend: Legend, top: float) -> None:
        secondary = self.color("text.secondary")
        x = float(PAD_SIDE)
        text_top = top + (LEGEND_HEIGHT - 18) / 2
        x += draw_text(painter, x, text_top, legend.title, TRACK_TITLE, self.color("text.ink"))
        x += 4  # ``.cl b`` margin-right
        for swatch, label in legend.items:
            x += 14
            painter.save()
            x += swatch(painter, x, top + LEGEND_HEIGHT / 2)
            painter.restore()
            x += 6 + draw_text(painter, x + 6, text_top, label, TRACK_TEXT, secondary)
        right = self.width() - PAD_SIDE
        # the aside only when it clears the legend (a narrow window drops it, the legend stays whole)
        if legend.aside and x + 14 <= right - TRACK_TEXT.advance(legend.aside):
            draw_text(
                painter,
                right - TRACK_TEXT.advance(legend.aside),
                text_top,
                legend.aside,
                TRACK_TEXT,
                secondary,
            )

    def _label(
        self,
        painter: QtGui.QPainter,
        x: float,
        baseline: float,
        text: str,
        token: str = "text.tertiary",
        *,
        anchor: str = "start",
        bold: bool = False,
    ) -> None:
        """SVG ``<text>``: placed by its baseline, anchored at its start, middle or end."""
        style = TextStyle(10.5, 600, 13, tabular=True) if bold else CHART_LABEL
        width = style.advance(text)
        if anchor == "middle":
            x -= width / 2
        elif anchor == "end":
            x -= width
        painter.save()
        painter.setPen(self.color(token))
        paint_glyphs(painter, x, baseline, text, style)
        painter.restore()

    def _frame(
        self,
        painter: QtGui.QPainter,
        axis: TimeAxis,
        height: int,
        top: float,
        bottom: float,
        maximum: float,
        ticks: Sequence[float],
        text: Callable[[float], str],
    ) -> Callable[[float], float]:
        """Grid lines (the axis line at 0), value labels, the axis break and the L line; returns y(value)."""
        plot = height - top - bottom

        def y(value: float) -> float:
            return top + plot * (1 - value / maximum)

        for value in ticks:
            token = "chart.axis" if value == 0 else "chart.grid"
            for start, end in axis.segments():
                painter.fillRect(
                    QtCore.QRectF(start, y(value) - 0.5, end - start, 1), self.color(token)
                )
            self._label(painter, axis.left - 8, y(value) + 3.5, text(value), anchor="end")
        self._break(painter, axis, y(0))
        self._limit(painter, axis, top, y(0))
        return y

    def _break(self, painter: QtGui.QPainter, axis: TimeAxis, baseline: float) -> None:
        if not axis.compressed:
            return
        x = axis.linear_end + axis.GAP / 2
        painter.setPen(QtGui.QPen(self.color("text.tertiary"), 1.1))
        for start in (x - 4, x + 1):
            painter.drawLine(
                QtCore.QPointF(start, baseline + 4), QtCore.QPointF(start + 3, baseline - 4)
            )

    def _limit(self, painter: QtGui.QPainter, axis: TimeAxis, top: float, bottom: float) -> None:
        if self.limit_s() is None:
            return
        pen = QtGui.QPen(self.color("state.warn"), 1.2)
        pen.setDashPattern([3 / 1.2, 3 / 1.2])
        pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        x = axis.x(axis.limit)
        painter.drawLine(QtCore.QPointF(x, top), QtCore.QPointF(x, bottom))

    def _clip(self, painter: QtGui.QPainter, axis: TimeAxis, height: int) -> None:
        """Data stays on its plot: rows before the time origin (pre-motion coverage, goals sent before the
        first movement) are cut at the axis' left edge, never drawn over the value labels."""
        painter.setClipRect(
            QtCore.QRectF(axis.left - 2.5, -4, axis.right - axis.left + 9, height + 8)
        )

    def _baseline(self, painter: QtGui.QPainter, axis: TimeAxis, height: int) -> None:
        painter.fillRect(
            QtCore.QRectF(axis.left, height - 1.5, axis.linear_end - axis.left, 1),
            self.color("chart.grid"),
        )

    def _note(self, painter: QtGui.QPainter, axis: TimeAxis, height: int, text: str) -> None:
        """A track without data: why, in its middle."""
        draw_text(
            painter, axis.left, (height - 18) / 2, text, TRACK_TEXT, self.color("text.tertiary")
        )

    def _coverage(self, painter: QtGui.QPainter, axis: TimeAxis, height: int) -> None:
        y = self._frame(painter, axis, height, 14, 6, 100, (0, 25, 50, 75, 100), lambda v: f"{v:g}")
        self._clip(painter, axis, height)
        coverage = self.color("data.coverage")
        timeline = self.timeline or {}
        segments: list[list[QtCore.QPointF]] = [[]]
        samples: list[tuple[float, float]] = []  # the drawn samples, (s, %)
        previous: float | None = None
        # A thinned series (every k-th sample) carries its gaps as null rows from the engine
        # (timelines.thinned_coverage), so its own spacing is never a gap; an unthinned one gets the 2 s rule.
        thinned = int((timeline.get("every_kth") or {}).get("coverage") or 1) > 1
        gap = math.inf if thinned else GAP_S
        for row in rows_of(timeline.get("coverage"), 3):
            t, value = number(row[0]), number(row[1])
            if t is None or value is None or (previous is not None and t - previous > gap):
                if segments[-1]:
                    segments.append([])
                if t is None or value is None:
                    previous = None
                    continue
            segments[-1].append(QtCore.QPointF(axis.x(t), y(value)))
            samples.append((t, value))
            previous = t
        segments = [points for points in segments if points]
        if not segments:
            self._note(painter, axis, height, "No coverage samples were recorded")
            return
        tint = QtGui.QColor(coverage)
        tint.setAlphaF(0.10)
        line = QtGui.QPen(coverage, 2)
        line.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        for points in segments:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(tint)
            painter.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(points[0].x(), y(0)),
                        *points,
                        QtCore.QPointF(points[-1].x(), y(0)),
                    ]
                )
            )
            painter.setPen(line)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPolyline(QtGui.QPolygonF(points))
        end = segments[-1][-1]
        if self.carried_forward():
            carry = QtGui.QColor(coverage)
            carry.setAlphaF(0.75)
            dashed = QtGui.QPen(carry, 2)
            dashed.setDashPattern([5 / 2, 4 / 2])
            dashed.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
            painter.setPen(dashed)
            painter.drawLine(end, QtCore.QPointF(axis.x(axis.limit), end.y()))
            completion = number(self.record.get("completion_s"))
            painter.setClipping(
                False
            )  # the label may reach past the plot's edge, as before the clip
            self._label(
                painter,
                end.x() + 8,
                end.y() + 16,
                f"Completed {duration(completion)}",
                "text.secondary",
            )
        card = self.color("surface.card")
        for share in MILESTONES:
            seconds = number(self.record.get(f"t{share}_s"))
            if seconds is None:
                continue
            # on the drawn line at the recorded time: the first sample at or above the share, whose value
            # may lie well above it when coverage jumped between samples
            point = QtCore.QPointF(axis.x(seconds), y(value_at(samples, seconds, share)))
            drop = QtGui.QColor(coverage)
            drop.setAlphaF(0.5)
            dashed = QtGui.QPen(drop, 1)
            dashed.setDashPattern([2, 2])
            dashed.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
            painter.setPen(dashed)
            painter.drawLine(point, QtCore.QPointF(point.x(), y(0)))
            painter.setPen(QtGui.QPen(coverage, 1.6))
            painter.setBrush(card)
            painter.drawEllipse(point, 3.5, 3.5)
        gain_s = number(self.record.get("last_gain_s"))
        gain = number(self.record.get("last_gain_coverage_pct"))
        if gain_s is not None and gain is not None:
            self._diamond(painter, axis.x(gain_s), y(gain) - 4, filled=False, stroke=1.4)
        painter.setPen(QtGui.QPen(card, 1.5))
        painter.setBrush(coverage)
        painter.drawEllipse(end, 4, 4)
        if self.limit_s() is not None:
            label = f"L {duration(axis.limit)}"
            self._label(
                painter, axis.x(axis.limit) - 5, 12, label, "state.warn", anchor="end", bold=True
            )

    def _goals(self, painter: QtGui.QPainter, axis: TimeAxis, height: int) -> None:
        self._clip(painter, axis, height)
        self._limit(painter, axis, 2, height - 2)
        self._baseline(painter, axis, height)
        goals = (self.timeline or {}).get("goals")
        if goals is None:
            self._note(painter, axis, height, "Goals not derived for this run yet")
            return
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for _, start, end, status, _ in rows_of(goals, 5):
            a, b = number(start), number(end)
            if a is None:
                continue
            b = b if b is not None else self.end_s()  # an open goal runs to the run's end
            token, opacity = GOAL_TOKENS.get(status, ("state.neutral", 0.25))
            color = self.color(token)
            color.setAlphaF(opacity)
            painter.setBrush(color)
            x0, x1 = axis.x(a), axis.x(b)
            painter.drawRoundedRect(QtCore.QRectF(x0 + 0.8, 8, max(2.0, x1 - x0 - 1.6), 16), 2, 2)

    def _dispatches(self, painter: QtGui.QPainter, axis: TimeAxis, height: int) -> None:
        self._clip(painter, axis, height)
        self._limit(painter, axis, 2, height - 2)
        self._baseline(painter, axis, height)
        timeline = self.timeline or {}
        if timeline.get("dispatches") is None:
            self._note(painter, axis, height, "No explorer log was parsed for this run")
            return
        for t, kind, _ in rows_of(timeline.get("dispatches"), 3):
            seconds = number(t)
            if seconds is not None:
                self._dispatch(painter, axis.x(seconds), 12, str(kind))
        for t, result, _, _ in rows_of(timeline.get("gates"), 4):
            seconds = number(t)
            if seconds is None:
                continue
            x = axis.x(seconds)
            # log format 2: "exhausted" (the explorer cancels the goal, or waits out the minimum interval)
            # and "keep"/"keep_small_cluster"; an "unavailable" gate decides nothing (the snapshot does), so
            # it gets no mark
            if result == "unavailable":
                continue
            if result == "exhausted":
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(self.color("text.ink"))
                painter.drawRoundedRect(QtCore.QRectF(x - 1.6, 26, 3.2, 10), 1, 1)
            else:
                painter.setPen(QtGui.QPen(self.color("text.tertiary"), 1.2))
                painter.drawLine(QtCore.QPointF(x, 27), QtCore.QPointF(x, 35))

    def _steps(
        self,
        rows: Sequence[Sequence[Any]],
        column: int,
        axis: TimeAxis,
        y: Callable[[float], float],
    ) -> list[list[QtCore.QPointF]]:
        """A value per row as steps to the next row (the last to the run's end); missing values break."""
        lines: list[list[QtCore.QPointF]] = [[]]
        end = self.end_s()
        for index, row in enumerate(rows):
            t, value = number(row[0]), number(row[column])
            if t is None or value is None:
                lines.append([])
                continue
            following = number(rows[index + 1][0]) if index + 1 < len(rows) else None
            until = following if following is not None else max(end, t)
            lines[-1] += [
                QtCore.QPointF(axis.x(t), y(value)),
                QtCore.QPointF(axis.x(until), y(value)),
            ]
        return [line for line in lines if line]

    def _dp(self, painter: QtGui.QPainter, axis: TimeAxis, height: int) -> None:
        timeline = self.timeline or {}
        rows = rows_of(timeline.get("analysis"), 5)
        largest = max((number(row[c]) or 0.0 for row in rows for c in (2, 3)), default=0.0)
        maximum = nice_ceiling(largest, (4, 8, 12, 16, 20, 40, 80))
        y = self._frame(
            painter, axis, height, 6, 6, maximum, (0, maximum / 2, maximum), lambda v: f"{v:g}"
        )
        self._clip(painter, axis, height)
        if not timeline.get("analysis_recorded") or not rows:
            self._note(painter, axis, height, "No explorer insight was recorded for this run")
            return
        analysis = self.color("data.analysis")
        route = QtGui.QColor(analysis)
        route.setAlphaF(0.8)
        dashed = QtGui.QPen(route, 1.4)
        dashed.setDashPattern([3 / 1.4, 2 / 1.4])
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for column, pen in ((2, QtGui.QPen(analysis, 1.6)), (3, dashed)):
            painter.setPen(pen)
            for line in self._steps(rows, column, axis, y):
                painter.drawPolyline(QtGui.QPolygonF(line))
        painter.setPen(QtGui.QPen(analysis, 2))
        for row in rows:
            seconds = number(row[0])
            if row[4] is True and seconds is not None:
                x = axis.x(seconds)
                painter.drawLine(QtCore.QPointF(x, y(0) - 7), QtCore.QPointF(x, y(0)))

    def _speed(self, painter: QtGui.QPainter, axis: TimeAxis, height: int) -> None:
        timeline = self.timeline or {}
        series = [rows_of(timeline.get(key), 2) for key in ("speed_odometry", "speed_ground_truth")]
        largest = max((number(row[1]) or 0.0 for rows in series for row in rows), default=0.0)
        maximum = nice_ceiling(largest, (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0))
        y = self._frame(
            painter, axis, height, 8, 22, maximum, (0, maximum / 2, maximum), lambda v: f"{v:.1f}"
        )
        for tick in axis.ticks():
            self._label(painter, axis.x(tick), height - 5, duration(tick), anchor="middle")
        if axis.compressed:
            self._label(painter, axis.x(axis.limit), height - 5, duration(axis.limit), anchor="end")
        if not any(series):
            self._note(painter, axis, height - 14, "No speed samples were recorded")
            return
        self._clip(painter, axis, height)
        odometry = QtGui.QPen(self.color("text.secondary"), 1.5)
        odometry.setDashPattern([3 / 1.5, 2 / 1.5])
        ground = QtGui.QPen(self.color("data.path"), 1.5)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for rows, pen in zip(series, (odometry, ground), strict=True):
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            line: list[QtCore.QPointF] = []
            for t, value in rows:
                seconds, speed = number(t), number(value)
                if seconds is None or speed is None:
                    if len(line) > 1:
                        painter.drawPolyline(QtGui.QPolygonF(line))
                    line = []
                    continue
                line.append(QtCore.QPointF(axis.x(seconds), y(speed)))
            if len(line) > 1:
                painter.drawPolyline(QtGui.QPolygonF(line))


# ---------- values ----------


def number_text(value: Any, digits: int) -> str:
    return fixed(value, digits) or "—"


def short_hash(value: Any, length: int = 8) -> str:
    text = str(value) if value else ""
    return text.removeprefix("sha256:")[:length] if text else "—"


def first_reached(record: Mapping[str, Any]) -> list[str]:
    """Each milestone's time, "not reached" when right-censored, "—" when the record holds neither."""
    values = []
    for share in MILESTONES:
        seconds, event = record.get(f"t{share}_s"), record.get(f"t{share}_event")
        if event is True and number(seconds) is not None:
            values.append(duration(seconds))
        elif event is False:
            values.append("not reached")
        else:
            values.append("—")
    return values


def key_values(record: Mapping[str, Any]) -> tuple[list[KeyValue], list[KeyValue]]:
    """The two columns under the map (decision 10: the minimum 5 s RTF and the wall time below 0.90 here)."""
    start, end = record.get("wall_start_ns"), record.get("wall_end_ns")
    wall = (end - start) / 1e9 if isinstance(start, int) and isinstance(end, int) else None
    rtf_token = "state.warn" if record.get("low_rtf") else "text.ink"
    left = [
        KeyValue("C(L)", number_text(record.get("coverage_at_limit_pct"), 1), "%"),
        KeyValue("Covered area", number_text(record.get("covered_m2"), 1), "m²"),
        KeyValue("Ground-truth path", number_text(record.get("gt_path_length_m"), 2), "m"),
        KeyValue("Odometry", number_text(record.get("odom_distance_m"), 2), "m"),
        KeyValue(
            "Minimum RTF, 5 s", number_text(record.get("rtf_min_5s"), 3), value_token=rtf_token
        ),
    ]
    recoveries = record.get("recoveries")
    right = [
        KeyValue("Explorer CPU", number_text(record.get("explorer_cpu_tw_pct"), 1), "%"),
        KeyValue("Explorer RSS peak", number_text(record.get("explorer_rss_peak_mib"), 1), "MiB"),
        KeyValue("Recoveries", str(recoveries) if isinstance(recoveries, int) else "—"),
        KeyValue("Wall elapsed", duration(wall)),
        KeyValue("Wall below 0.90 RTF", number_text(record.get("wall_s_below_090"), 0), "s"),
    ]
    return left, right


EXIT_TEXT = {
    "exited_0": "Exit 0",
    "exited_nonzero": "Exit {code}",
    "signalled": "Signal {code}",
    "stopped_by_worker": "Stopped by the worker",
    "not_started": "Not started",
}
ELIGIBILITY = ("motion", "coverage", "timing", "navigation")


def repository_text(state: Any) -> str:
    if not isinstance(state, Mapping) or not state.get("commit"):
        return "—"
    dirty = state.get("dirty")
    return f"{str(state['commit'])[:7]} · " + (
        "unknown" if dirty is None else "dirty" if dirty else "clean"
    )


def host_text(host: Any) -> str:
    """Cores and memory as Docker reports them (``deployment.HOST_FACTS``); the CPU model is the tooltip."""
    docker = host.get("docker") if isinstance(host, Mapping) else None
    if not isinstance(docker, Mapping):
        return "—"
    parts = []
    if isinstance(docker.get("NCPU"), int):
        parts.append(f"{docker['NCPU']} cores")
    if isinstance(docker.get("MemTotal"), int):
        parts.append(f"{docker['MemTotal'] / 2**30:.1f} GiB")
    return " · ".join(parts) or "—"


@dataclass
class Context:
    """What the provenance card shows beyond the record: names and the replies of its own reads."""

    profile: str = "—"
    differences: str = "—"
    partners: str = "—"
    experiments: int | None = None


def provenance_columns(
    record: Mapping[str, Any], document: Mapping[str, Any] | None, context: Context
) -> list[tuple[str, list[KeyValue]]]:
    """Run conditions, code and images, profile and block (screen 24)."""
    mono = MONO_TEXT
    slots, alone = record.get("parallel_max"), number(record.get("concurrent_wall_fraction"))
    parallel = str(slots) if isinstance(slots, int) else "—"
    alone_text = f"· alone {round((1 - alone) * 100)} %" if alone is not None else ""
    diagnostics = record.get("diagnostics_enabled")
    streams = record.get("debug_streams")
    conditions = " · ".join(
        (
            "—" if diagnostics is None else "On" if diagnostics else "Off",
            ", ".join(streams) if isinstance(streams, list) and streams else "off",
        )
    )
    started, early = record.get("insight_started"), record.get("insight_exited_before_stop")
    insight = (
        "—"
        if started is None
        else ("Started" + (" · ended early" if early else "") if started else "Not started")
    )
    status, code = record.get("explorer_exit_status"), record.get("explorer_exit_code")
    exit_text = EXIT_TEXT.get(str(status), "—").format(code=code if code is not None else "?")
    if record.get("explorer_exited_before_stop") and status in EXIT_TEXT:
        exit_text += " before stop"
    flags = {name: record.get(f"{name}_eligible") for name in ELIGIBILITY}
    failing = [name for name, value in flags.items() if value is False]
    if failing:
        eligibility = KeyValue(
            f"Eligibility {version_tail(record.get('eligibility_version'))}",
            "not " + ", ".join(failing),
            value_token="state.warn",
            value_style=KV_STRONG,
        )
    elif all(flags.values()):
        eligibility = KeyValue(
            f"Eligibility {version_tail(record.get('eligibility_version'))}",
            ", ".join(ELIGIBILITY),
            value_token="state.ok",
            value_style=KV_STRONG,
        )
    else:
        eligibility = KeyValue("Eligibility", "—")
    run = [
        KeyValue("Time limit L", duration(record.get("time_limit_s"))),
        KeyValue("Workers in parallel", parallel, alone_text),
        KeyValue("Runs since engine start", known(record.get("runs_since_engine_start"))),
        KeyValue("Diagnostics · debug streams", conditions),
        KeyValue("Explorer insight", insight),
        KeyValue("Explorer exit", exit_text),
        eligibility,
    ]
    document = document or {}
    repositories = document.get("repositories") or {}
    suite = document.get("suite_image") or {}
    labels = suite.get("labels") or {}
    version = labels.get("org.benchmark.suite.version")
    suite_text = f"sha256:{short_hash(suite.get('id'))}" + (f" · v{version}" if version else "")
    runtime = document.get("runtime_image") or {}
    host = document.get("host") or {}
    engine_host = host.get("engine") if isinstance(host, Mapping) else None
    code = [
        KeyValue("Checkout", repository_text(repositories.get("checkout")), value_style=mono),
        KeyValue(
            "Explorer",
            repository_text(repositories.get("exploration_packages/frontier_exploration_ros2")),
            value_style=mono,
        ),
        KeyValue("Suite image", suite_text if suite.get("id") else "—", value_style=mono),
        KeyValue(
            "Runtime image",
            f"sha256:{short_hash(runtime.get('id'))}" if runtime.get("id") else "—",
            value_style=mono,
        ),
        KeyValue(
            "Host",
            host_text(host),
            tooltip=str(engine_host.get("cpu_model") or "")
            if isinstance(engine_host, Mapping)
            else "",
        ),
        KeyValue(
            f"Comparison key v{record.get('comparison_key_version', '?')}",
            short_hash(record.get("comparison_key")),
            value_style=mono,
        ),
        KeyValue(
            "Record · schema · timing",
            f"v{known(record.get('run_record_version'))} · {known(record.get('dataset_schema'))} · "
            f"rule {version_tail(record.get('timing_rule'))}",
            value_style=mono,
        ),
    ]
    position = record.get("campaign_position")
    attempts = record.get("experiment_attempts")
    block = [
        KeyValue("Profile", context.profile),
        KeyValue(
            "Parameter hash",
            short_hash(record.get("parameter_hash"))
            + (" · baseline" if record.get("is_baseline") else ""),
            value_style=mono,
        ),
        KeyValue("Differences from Default", context.differences),
        KeyValue("Block", short_hash(record.get("block_id")), value_style=mono),
        KeyValue("Block partner", context.partners),
        KeyValue(
            "Experiment position",
            (f"{position + 1}" if isinstance(position, int) else "—")
            + (f" of {context.experiments}" if context.experiments else ""),
        ),
        KeyValue(
            "Attempt",
            f"{record.get('attempt_number', '?')} of "
            + (str(attempts) if isinstance(attempts, int) else "?"),
        ),
    ]
    return [("Run conditions", run), ("Code and images", code), ("Profile and block", block)]


def value_at(samples: Sequence[tuple[float, float]], seconds: float, fallback: float) -> float:
    """The drawn coverage line's value at ``seconds``: the sample there (times are sent to the millisecond),
    else the straight segment between its neighbours; ``fallback`` outside the samples."""
    near = min(samples, key=lambda sample: abs(sample[0] - seconds), default=None)
    if near is not None and abs(near[0] - seconds) <= 5e-4:
        return near[1]
    for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
        if t0 <= seconds <= t1:
            return v0 if t1 == t0 else v0 + (v1 - v0) * (seconds - t0) / (t1 - t0)
    return fallback


def known(value: Any) -> str:
    """A recorded value as text; "—" when it is unavailable (null)."""
    return "—" if value is None else str(value)


def version_tail(value: Any) -> str:
    """ "v2" of "timing-rule-v2" and "eligibility-v2"."""
    text = str(value or "")
    return text.rsplit("-", 1)[-1] if "-" in text else text or "?"


# ---------- cards ----------


class MapCard(Card):
    """``.card.side``: "Final map" with its reference, the map stage (300 px), the legend, first reached and
    the key values (two columns, 31 px rows)."""

    def __init__(self, themes: ThemeManager, transport: Transport) -> None:
        super().__init__(themes)
        self.head = CardHead(themes, "Final map", self)
        self.reference = TextLine(themes, "", BODY, "text.secondary")
        self.head.add(self.reference)
        self.map = RunMapView(themes, self)
        self.loader = RunMapLoader(transport, self.map)
        self.legend = MapLegend(themes, self)
        self.reached = FirstReached(themes, self)
        self.left = KeyValueList(
            themes, row_height=31, label_style=KV_LABEL, value_style=KV_VALUE, last_open=True
        )
        self.right = KeyValueList(
            themes, row_height=31, label_style=KV_LABEL, value_style=KV_VALUE, last_open=True
        )
        for kv in (self.left, self.right):
            kv.setParent(self)
        self.loader.status.connect(self.legend.set_status)
        self.loader.reference.connect(lambda name: self.reference.setText(f"Reference {name}"))
        self.map.world_changed.connect(self._layout)

    def show_record(self, record: Mapping[str, Any]) -> None:
        self.reference.setText("")
        self.loader.show_attempt(str(record.get("attempt_id")))
        self.reached.set_values(first_reached(record))
        left, right = key_values(record)
        self.left.set_rows(left)
        self.right.set_rows(right)

    def _layout(self) -> None:
        width = self.width()
        self.head.setGeometry(1, 1, width - 2, 52)
        inner = width - 40
        stage_top, stage = 1 + 52 + 14, 300.0
        columns, rows = self.map.world.size if self.map.world else (4, 3)
        map_width = min(inner, (stage - 2) * columns / max(1, rows) + 2)
        map_height = min(stage, self.map.height_for(map_width))
        place(
            self.map,
            20 + (inner - map_width) / 2,
            stage_top + (stage - map_height) / 2,
            map_width,
            map_height,
        )
        top = stage_top + stage + 10
        self.legend.setGeometry(20, round(top), inner, 18)
        top += 18 + 12
        self.reached.setGeometry(20, round(top), inner, 44)
        top += 44
        half = (inner - 24) / 2
        place(self.left, 20, top, half, self.left.height())
        place(self.right, 20 + half + 24, top, half, self.right.height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()


class TimeCard(Card):
    """``.card.time``: "Over time" with the axis note, then the charts."""

    def __init__(self, themes: ThemeManager) -> None:
        super().__init__(themes)
        self.head = CardHead(themes, "Over time", self)
        self.note = TextLine(themes, "", BODY, "text.secondary")
        self.head.add(self.note)
        self.charts = RunCharts(themes, self)

    def show_record(self, record: Mapping[str, Any]) -> None:
        self.charts.set_run(record)
        self._note()

    def set_timeline(self, timeline: Mapping[str, Any]) -> None:
        self.charts.set_timeline(timeline)
        self._note()

    def _note(self) -> None:
        timeline = self.charts.timeline
        if timeline is None:
            self.note.setText("")
            return
        origin = "first movement" if timeline.get("time_origin") != "activation" else "activation"
        axis = self.charts.axis()
        text = f"Simulated time since {origin}"
        if axis.compressed:
            text += f" · compressed after {duration(axis.split)}"
        self.note.setText(text)

    def natural_height(self) -> int:
        return 1 + 52 + self.charts.height() + 1

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.head.setGeometry(1, 1, self.width() - 2, 52)
        self.charts.setGeometry(1, 53, self.width() - 2, self.charts.height())
        self._note()


class ProvenanceColumn(Themed):
    """One ``.prov`` column: its ``h4`` (12 px 600 secondary, 22 px with 10 px above) over 31 px rows."""

    def __init__(self, themes: ThemeManager, title: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(themes, parent)
        self.title = title
        self.rows = KeyValueList(
            themes, row_height=31, label_style=KV_LABEL, value_style=KV_VALUE, last_open=True
        )
        self.rows.setParent(self)

    def set_rows(self, rows: Sequence[KeyValue]) -> None:
        self.rows.set_rows(rows)
        self.setFixedHeight(10 + 22 + 2 + self.rows.height())
        self.rows.setGeometry(0, 34, self.width(), self.rows.height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.rows.setGeometry(0, 34, self.width(), self.rows.height())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        draw_text(painter, 0, 10 + 2, self.title, GROUP_TITLE, self.color("text.secondary"))


class ProvenanceCard(Card):
    """``.card.prov``: "Provenance" with the document's ID, then three columns 32 px apart."""

    def __init__(self, themes: ThemeManager) -> None:
        super().__init__(themes)
        self.head = CardHead(themes, "Provenance", self)
        self.aside = TextLine(themes, "", BODY, "text.secondary")
        self.head.add(self.aside)
        self.columns = [
            ProvenanceColumn(themes, title, self)
            for title in ("Run conditions", "Code and images", "Profile and block")
        ]

    def show_run(
        self, record: Mapping[str, Any], document: Mapping[str, Any] | None, context: Context
    ) -> None:
        identity = record.get("provenance_id")
        self.aside.setText(
            "Recorded with the run · provenance " + short_hash(identity)
            if document
            else "Loading the provenance…"
            if identity
            else "No provenance recorded"
        )
        for column, (_, rows) in zip(
            self.columns, provenance_columns(record, document, context), strict=True
        ):
            column.set_rows(rows)
        self.setFixedHeight(self.natural_height())
        self._layout()

    def natural_height(self) -> int:
        return 1 + 52 + 6 + max((c.height() for c in self.columns), default=0) + 14 + 1

    def _layout(self) -> None:
        width = self.width()
        self.head.setGeometry(1, 1, width - 2, 52)
        column = (width - 2 - 40 - 2 * 32) / 3
        for index, widget in enumerate(self.columns):
            place(widget, 1 + 20 + index * (column + 32), 1 + 52 + 6, column, widget.height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()


class RunHead(FlexRow):
    """``.rhead``: "Runs" back link, a rule, the robot (20 px 600), the run's context and its outcome badge; at
    the right Replay (RO-20), Block partner (disabled with its reason without one) and Files."""

    def __init__(self, themes: ThemeManager) -> None:
        super().__init__(gap=16)
        self.back = BackLink(themes, "Runs")
        self.add(self.back)
        rule = Rule(themes, vertical=True)
        rule.setFixedHeight(24)
        self.add(rule, width=1)
        self.title = TextLine(themes, "", TITLE, "text.ink")
        self.add(self.title)
        self.context = TextLine(themes, "", BODY, "text.secondary")
        self.add(self.context, gap=12)
        self.badge = StatusBadge(themes)
        self.add(self.badge, gap=12)
        self.add_stretch()
        self.replay = Button(themes, "Replay", icon="play")
        self.replay.setToolTip(
            "Replay this run on the Monitor: map, explorer decisions and coverage over time"
        )
        self.partner = Button(themes, "Block partner", icon="route")
        self.files = Button(themes, "Files", icon="folder")
        for button in (self.replay, self.partner, self.files):
            self.add(button, gap=8)

    def relayout(self) -> None:
        """The context line takes the room the tools leave and is elided in it."""
        if hasattr(self, "files"):
            used = sum(
                self.width_of(widget, width) + gap
                for widget, width, gap, _ in self.items
                if widget is not self.context and widget.isVisibleTo(self)
            )
            room = max(0.0, self.width() - used - 12 - 16)
            self.items = [
                (w, min(room, self.context.natural_width()) if w is self.context else width, g, m)
                for w, width, g, m in self.items
            ]
        super().relayout()


# ---------- the view ----------


class RunView(QtWidgets.QScrollArea):
    """The run view: the head, the map and time cards side by side and the provenance card below, scrolled as
    one page (decision 4). ``show_run`` opens a final run's record; replies of an earlier run are dropped."""

    back = QtCore.Signal()
    replay_requested = QtCore.Signal(str)  # attempt id
    files_requested = QtCore.Signal(str)  # attempt id
    open_record = QtCore.Signal(object)  # a block partner's record

    def __init__(
        self,
        themes: ThemeManager,
        transport: Transport,
        names: Callable[[str, Any], str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.themes, self.transport, self.names = themes, transport, names
        self.catalog: Mapping[str, Any] = {}
        self.record: dict[str, Any] = {}
        self.document: Mapping[str, Any] | None = None
        self.context = Context()
        self.partner_records: list[dict[str, Any]] = []
        self.token = ""
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(False)
        self.page = QtWidgets.QWidget()
        self.page.setAutoFillBackground(False)
        self.viewport().setAutoFillBackground(False)
        self.setStyleSheet("QScrollArea { background: transparent; }")
        self.head = RunHead(themes)
        self.map_card = MapCard(themes, transport)
        self.time_card = TimeCard(themes)
        self.provenance = ProvenanceCard(themes)
        for widget in (self.head, self.map_card, self.time_card, self.provenance):
            widget.setParent(self.page)
        self.setWidget(self.page)
        OverlayScrollBar(themes, self)
        self.head.back.clicked.connect(self.back.emit)
        self.head.replay.clicked.connect(
            lambda: self.replay_requested.emit(str(self.record.get("attempt_id")))
        )
        self.head.files.clicked.connect(
            lambda: self.files_requested.emit(str(self.record.get("attempt_id")))
        )
        self.head.partner.clicked.connect(self._open_partner)
        self.poll = QtCore.QTimer(self)
        self.poll.setSingleShot(True)
        self.poll.setInterval(POLL_MS)
        self.poll.timeout.connect(self._request_timeline)

    # opening a run

    def set_catalog(self, catalog: Mapping[str, Any]) -> None:
        self.catalog = catalog

    def show_run(self, record: Mapping[str, Any]) -> None:
        self.record = dict(record)
        self.document = None
        self.partner_records = []
        self.token = uuid.uuid4().hex
        self.poll.stop()
        identity = str(record.get("attempt_id"))
        baseline = bool(record.get("is_baseline")) or str(record.get("profile_id", "")).startswith(
            "default-"
        )
        self.context = Context(
            profile=self.names("profile", record.get("profile_id")),
            differences="None" if baseline else "—",
        )
        self._head()
        self.map_card.show_record(record)
        self.time_card.show_record(record)
        self._provenance()
        self.verticalScrollBar().setValue(0)
        self.transport.submit(
            "run-view-get:" + self.token,
            "results.get",
            {"table": "attempts", "id": identity, "include": ["provenance"]},
        )
        self._request_timeline()
        if record.get("block_id"):
            self.transport.submit(
                "run-view-block:" + self.token,
                "results.query",
                {
                    "table": "runs",
                    # the study's rule: a campaign a restarted copy replaces shares the block, never a partner
                    "filters": {
                        "block_id": record["block_id"],
                        "study_id": record.get("study_id"),
                        "is_final": True,
                    },
                    "offset": 0,
                    "limit": 50,
                },
            )
        if record.get("campaign_id"):
            self.transport.submit(
                "run-view-experiments:" + self.token,
                "results.query",
                {
                    "table": "experiments",
                    "filters": {"campaign_id": record["campaign_id"]},
                    "offset": 0,
                    "limit": 1,
                },
            )
        if not baseline and record.get("profile_id"):
            self.transport.submit(
                "run-view-profile:" + self.token, "profiles.get", {"id": record["profile_id"]}
            )

    def _request_timeline(self) -> None:
        self.transport.submit(
            "run-view-timeline:" + self.token,
            "results.timeline",
            {"attempt_id": str(self.record.get("attempt_id"))},
        )

    def _head(self) -> None:
        record = self.record
        self.head.title.setText(self.names("robot", record.get("robot_id")))
        repetition = record.get("repetition")
        parts = [
            self.names("world", record.get("world_id")),
            self.names("spawn", record.get("spawn_id")) + " spawn",
            self.names("profile", record.get("profile_id")),
            f"rep {repetition + 1}" if isinstance(repetition, int) else "",
            self.names("campaign", record.get("campaign_id")),
        ]
        self.head.context.setText(" · ".join(part for part in parts if part))
        excluded = record.get("report_class") not in ("reported", None)
        self.head.badge.set_look(badge_look("outcome", record.get("outcome"), muted=excluded))
        self.head.replay.setEnabled(record.get("record_kind") == "full")
        self._partner_button()
        self.head.relayout()

    def _partner_button(self) -> None:
        partners = self.partner_records
        button = self.head.partner
        if not self.record.get("block_id"):
            button.setEnabled(False)
            button.setToolTip("The run belongs to no block")
        elif not partners:
            button.setEnabled(False)
            button.setToolTip("The only profile in its block")
        else:
            button.setEnabled(True)
            button.setToolTip(
                "Open the block partner: "
                + ", ".join(self.names("profile", r.get("profile_id")) for r in partners)
            )

    def _open_partner(self) -> None:
        """The block's Default run when this run is not Default, else its first partner in plan order."""
        if self.partner_records:
            baseline = next((r for r in self.partner_records if r.get("is_baseline")), None)
            chosen = baseline if baseline and not self.record.get("is_baseline") else None
            self.open_record.emit(chosen or self.partner_records[0])

    def _provenance(self) -> None:
        self.provenance.show_run(self.record, self.document, self.context)
        self._layout()

    # replies

    def received(self, tag: str, result: Any) -> bool:
        if self.map_card.loader.received(tag, result):
            return True
        if not tag.startswith("run-view-"):
            return False
        kind, _, token = tag.removeprefix("run-view-").partition(":")
        if token != self.token:
            return True  # an earlier run's reply
        if kind == "get":
            self.document = result.get("provenance") if isinstance(result, Mapping) else None
            self._provenance()
        elif kind == "timeline":
            state = result.get("state")
            if state == "loading":
                self.poll.start()
            elif state == "ready":
                self.time_card.set_timeline(result)
            else:
                self.time_card.charts.set_message(
                    "The run's series could not be read: " + str(result.get("reason", state))
                )
        elif kind == "block":
            rows = [
                row
                for row in result.get("rows") or []
                if row.get("experiment_id")
                != self.record.get("experiment_id")  # its own final run too
            ]
            self.partner_records = rows
            self.context.partners = (
                ", ".join(self.names("profile", r.get("profile_id")) for r in rows)
                if rows
                else "None · one profile"
            )
            self._partner_button()
            self._provenance()
        elif kind == "experiments":
            total = result.get("total")
            self.context.experiments = total if isinstance(total, int) else None
            self._provenance()
        elif kind == "profile":
            self._count_differences(result)
        return True

    def _count_differences(self, snapshot: Mapping[str, Any]) -> None:
        robot = next(
            (r for r in self.catalog.get("robots", []) if r.get("id") == snapshot.get("robot_id")),
            None,
        )
        if robot is not None and isinstance(snapshot.get("effective"), dict):
            count = len(
                profile_differences(robot, self.catalog.get("parameters", {}), dict(snapshot))
            )
            self.context.differences = (
                "None" if count == 0 else f"{count} parameter" + ("" if count == 1 else "s")
            )
            self._provenance()

    def failed(self, tag: str, error: str) -> bool:
        if self.map_card.loader.failed(tag, error):
            return True
        if not tag.startswith("run-view-"):
            return False
        kind, _, token = tag.removeprefix("run-view-").partition(":")
        if token != self.token:
            return True
        if kind == "timeline":
            stale = stale_engine_message(
                error, "Unsupported operation: results.timeline", "the run view's charts"
            )
            self.time_card.charts.set_message(
                stale or "The run's series could not be read: " + error
            )
        elif kind == "get":
            self.provenance.aside.setText("Provenance unavailable: " + error)
        return True  # the cards say what is missing; nothing else depends on these reads

    # layout

    def _layout(self) -> None:
        width = self.viewport().width()
        inner = width - 2 * PAGE_LEFT
        self.head.setGeometry(PAGE_LEFT, PAGE_TOP, inner, 36)
        top = PAGE_TOP + 36 + CARD_GAP
        height = self.time_card.natural_height()
        self.map_card.setGeometry(PAGE_LEFT, top, MAP_WIDTH, height)
        self.time_card.setGeometry(
            PAGE_LEFT + MAP_WIDTH + CARD_GAP, top, inner - MAP_WIDTH - CARD_GAP, height
        )
        top += height + CARD_GAP
        self.provenance.setGeometry(PAGE_LEFT, top, inner, self.provenance.natural_height())
        self.page.resize(width, top + self.provenance.natural_height() + PAGE_BOTTOM)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def close_timers(self) -> None:
        self.poll.stop()
        self.map_card.loader.close_timers()
