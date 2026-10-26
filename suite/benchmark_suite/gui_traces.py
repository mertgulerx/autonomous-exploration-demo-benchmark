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

"""The Study tab's coverage traces (27; RESULTS_STUDY.md decision 5, RD-6): every reported run of one stratum as a
raw line, our explorer in ink and every compared method in its own colour (contract C2); the one view the GUI draws
itself, with no band and no statistic."""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import math
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from .contracts import GAP_S, OUR_EXPLORER, measured_as
from .gui_marks import CHART_LABEL
from .gui_menus import show_tooltip
from .gui_run_detail import TimeAxis, carries_forward
from .gui_style import duration, number
from .gui_theme import ThemeManager
from .gui_widgets import TextStyle, Themed, paint_glyphs

# Colours by explorer (contract C2): frontier_exploration_ros2 in ink, each external explorer one fixed Okabe–Ito
# hue in registry order (profiles.yaml after ours: blue, vermilion, bluish green, reddish purple, orange), the
# same in every robot and study, lighter in the dark theme. The GUI draws this view, so it uses the GUI's colours
# (RESULTS_STUDY.md batch 2).
TRACE_COLOURS = (
    ("#0072B2", "#5FA8DE"),
    ("#D55E00", "#E58448"),
    ("#009E73", "#3FB98F"),
    ("#CC79A7", "#D99CC0"),
    ("#E69F00", "#EDB33F"),
)
# Variants of one method (contract C2): its Default profile solid, its other profiles in these dash patterns
# (px at width 1; scaled by the pen), in (parameter hash, profile ID) order per explorer and robot.
VARIANT_DASHES: tuple[tuple[float, ...], ...] = ((), (6.0, 2.5), (6.0, 2.0, 1.5, 2.0), (11.0, 3.0))
TRACE_LEFT, TRACE_RIGHT, TRACE_TOP, TRACE_BOTTOM = 44, 14, 14, 30
Style = tuple[int, int]  # (colour index, -1 for ink; dash index)


def trace_styles(runs: Iterable[Mapping[str, Any]], explorers: Sequence[str]) -> dict[str, Style]:
    """Each profile's (colour, dash) by its explorer (contract C2). ``explorers``: the catalog's run-record
    explorer IDs in registry order; frontier_exploration_ros2 is ink (-1), an external explorer its position
    among the others, one the catalog does not list after them. Within an explorer and robot the Default profile
    (``default-…`` or the baseline) is solid, the others take the dash patterns in (parameter hash, ID) order."""
    external = [explorer for explorer in explorers if explorer != OUR_EXPLORER]
    members: dict[tuple[str, str], set[tuple[bool, str, str]]] = defaultdict(set)
    for run in runs:
        profile = str(run.get("profile_id"))
        default = bool(run.get("is_baseline")) or profile.startswith("default-")
        members[(str(run.get("explorer_id") or OUR_EXPLORER), str(run.get("robot_id")))].add(
            (not default, str(run.get("parameter_hash")), profile)
        )
    unlisted = sorted({e for e, _ in members if e != OUR_EXPLORER and e not in external})
    styles: dict[str, Style] = {}
    for (explorer, _), found in members.items():
        colour = (
            -1
            if explorer == OUR_EXPLORER
            else external.index(explorer)
            if explorer in external
            else len(external) + unlisted.index(explorer)
        )
        for dash, (_, _, profile) in enumerate(sorted(found)):
            styles[profile] = (colour, dash)
    return styles


def run_end(run: Mapping[str, Any]) -> float:
    """When a run stops exploring on the axis: its completion, a stuck run's stop (stuck-stop-v1; an explorer
    failure's after movement onset, ``contracts.measured_as``), else its limit (a run that never moved never
    explored)."""
    outcome = measured_as(run.get("outcome"), run.get("failure_source"), run.get("motion_started"))
    if outcome == "no_movement":
        return 0.0
    completion = number(run.get("completion_s"))
    if outcome == "completed" and completion is not None:
        return completion
    stopped = number(run.get("elapsed_s"))
    if outcome == "stuck" and stopped is not None:
        return stopped
    return number(run.get("time_limit_s")) or 0.0


class TracesChart(Themed):
    """Every reported run's coverage in one stratum as a raw line (no band, no statistic), our explorer in ink and
    each compared method in its colour, variants of a method in dash patterns (contract C2), completed and stuck
    runs (and explorer failures after movement onset) carried forward to L dashed; the axis of the run view
    (linear past the last end, compressed to L). Hovering names a run, a click opens it (``opened``)."""

    opened = QtCore.Signal(str)  # attempt id

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.runs: list[dict[str, Any]] = []
        self.styles: dict[str, Style] = {}  # profile id -> (colour index, -1: ink; dash index)
        # the catalog's run-record explorer IDs in registry order (StudyTab.set_explorers)
        self.explorers: list[str] = []
        self.hovered = -1
        self.describe: Callable[[dict[str, Any]], str] = lambda run: str(run.get("attempt_id"))
        self.setMouseTracking(True)

    def set_runs(
        self, runs: Sequence[dict[str, Any]], styles: Mapping[str, Style] | None = None
    ) -> None:
        """``styles``: the study's profiles' styles (``trace_styles`` of its final runs), so a profile keeps its
        colour and dash in every world; without it, this reply's."""
        self.runs = list(runs)
        self.styles = {**trace_styles(runs, self.explorers), **(styles or {})}
        self.hovered = -1
        self.update()

    def trace_style(self, run: Mapping[str, Any]) -> Style:
        return self.styles.get(str(run.get("profile_id")), (-1, 0))

    def colour(self, run: Mapping[str, Any]) -> QtGui.QColor:
        index = self.trace_style(run)[0]
        if index < 0:
            return self.color("text.ink")
        light, dark = TRACE_COLOURS[index % len(TRACE_COLOURS)]
        return QtGui.QColor(dark if self.theme.name == "dark" else light)

    def dashes(self, run: Mapping[str, Any]) -> tuple[float, ...]:
        """The run's variant dash pattern in px (empty: solid)."""
        return VARIANT_DASHES[self.trace_style(run)[1] % len(VARIANT_DASHES)]

    def limit(self) -> float | None:
        limits = [number(r.get("time_limit_s")) for r in self.runs]
        found = [value for value in limits if value]
        return max(found) if found else None

    def axis(self) -> TimeAxis:
        end = max((run_end(r) for r in self.runs), default=60.0)
        return TimeAxis.of(end, self.limit(), TRACE_LEFT, self.width() - TRACE_RIGHT)

    def value_y(self, value: float) -> float:
        plot = self.height() - TRACE_TOP - TRACE_BOTTOM
        return TRACE_TOP + plot * (1 - value / 100)

    def lines(self, axis: TimeAxis) -> list[list[list[QtCore.QPointF]]]:
        """Each run's segments: a stretch without samples over 2 s breaks its line (a thinned run carries those
        stretches as null rows from the engine, so its own spacing is never one); a run that never moved is its
        pre-motion coverage held from 0 to L, as F1 draws it (nothing when that value is unavailable)."""
        found = []
        for run in self.runs:
            held = self.held(run)
            if held is not None:
                end = axis.limit if self.limit() is not None else axis.split
                y = self.value_y(held)
                found.append([[QtCore.QPointF(axis.x(0.0), y), QtCore.QPointF(axis.x(end), y)]])
                continue
            gap = math.inf if int(number(run.get("every_kth")) or 1) > 1 else GAP_S
            segments: list[list[QtCore.QPointF]] = [[]]
            previous: float | None = None
            for t, value in run.get("samples") or []:
                seconds, coverage = number(t), number(value)
                if seconds is None or coverage is None:
                    if segments[-1]:
                        segments.append([])
                    previous = None
                    continue
                if previous is not None and seconds - previous > gap and segments[-1]:
                    segments.append([])
                segments[-1].append(QtCore.QPointF(axis.x(seconds), self.value_y(coverage)))
                previous = seconds
            found.append([segment for segment in segments if segment])
        return found

    @staticmethod
    def held(run: Mapping[str, Any]) -> float | None:
        """A reported run that never moved (an explorer failure before movement onset too): its pre-motion
        coverage, held (no line of its own otherwise)."""
        outcome = measured_as(
            run.get("outcome"), run.get("failure_source"), run.get("motion_started")
        )
        if outcome != "no_movement" or run.get("samples"):
            return None
        return number(run.get("pre_motion_coverage_pct"))  # unavailable stays unavailable, never 0

    def _nearest(self, position: QtCore.QPointF) -> int:
        axis = self.axis()
        best, chosen = 6.0, -1
        for index, segments in enumerate(self.lines(axis)):
            for segment in segments:
                # a sample alone between two gaps is a point to hit
                pairs = (
                    zip(segment, segment[1:]) if len(segment) > 1 else [(segment[0], segment[0])]
                )
                for a, b in pairs:
                    distance = _distance(position, a, b)
                    if distance < best:
                        best, chosen = distance, index
        return chosen

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self._nearest(event.position())
        if index != self.hovered:
            self.hovered = index
            self.setCursor(
                QtCore.Qt.CursorShape.PointingHandCursor
                if index >= 0
                else QtCore.Qt.CursorShape.ArrowCursor
            )
            self.update()
        if index >= 0:
            show_tooltip(self, event.globalPosition().toPoint(), self.describe(self.runs[index]))

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hovered = -1
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self._nearest(event.position())
        if index >= 0 and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.opened.emit(str(self.runs[index].get("attempt_id")))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        axis = self.axis()
        bottom = self.value_y(0)
        for value in (0, 25, 50, 75, 100):
            token = "chart.axis" if value == 0 else "chart.grid"
            for start, stop in axis.segments():
                painter.fillRect(
                    QtCore.QRectF(start, self.value_y(value) - 0.5, stop - start, 1),
                    self.color(token),
                )
            self._label(painter, TRACE_LEFT - 8, self.value_y(value) + 3.5, str(value), "end")
        if axis.compressed:
            x = axis.linear_end + axis.GAP / 2
            painter.setPen(QtGui.QPen(self.color("text.tertiary"), 1.1))
            for start in (x - 4, x + 1):
                painter.drawLine(
                    QtCore.QPointF(start, bottom + 4), QtCore.QPointF(start + 3, bottom - 4)
                )
        for tick in axis.ticks():
            self._label(painter, axis.x(tick), self.height() - 8, duration(tick), "middle")
        limit = self.limit()
        if axis.compressed:
            self._label(
                painter, axis.x(axis.limit), self.height() - 8, duration(axis.limit), "middle"
            )
        if limit is not None:
            pen = QtGui.QPen(self.color("state.warn"), 1.2)
            pen.setDashPattern([3 / 1.2, 3 / 1.2])
            painter.setPen(pen)
            x = axis.x(axis.limit)
            painter.drawLine(QtCore.QPointF(x, TRACE_TOP), QtCore.QPointF(x, bottom))
            self._label(
                painter,
                x - 4,
                TRACE_TOP + 8,
                f"L {duration(limit)}",
                "end",
                "state.warn",
                bold=True,
            )
        lines = self.lines(axis)
        order = [i for i in range(len(self.runs)) if i != self.hovered] + (
            [self.hovered] if self.hovered >= 0 else []
        )
        for index in order:
            run, segments = self.runs[index], lines[index]
            if not segments:
                continue
            hovered = index == self.hovered
            colour = self.colour(run)
            colour.setAlphaF(1.0 if hovered else 0.55)
            pen = QtGui.QPen(colour, 2.6 if hovered else 1.2)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            width = pen.widthF()
            if self.held(run) is not None:
                pen.setDashPattern([1, 2.5])  # dotted: held, not recorded progress (F1)
            elif dashes := self.dashes(run):
                pen.setDashPattern([value / width for value in dashes])  # a variant (C2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            for segment in segments:
                if len(segment) == 1:  # a sample alone between two gaps
                    painter.drawPoint(segment[0])
                painter.drawPolyline(QtGui.QPolygonF(segment))
            end = segments[-1][-1]
            samples = run.get("samples") or []
            last = number(samples[-1][0]) if samples else None
            if carries_forward(run, last, limit):
                carried = self.colour(run)
                carried.setAlphaF(0.9 if hovered else 0.35)
                width = 2.6 if hovered else 1.2
                dashed = QtGui.QPen(carried, width)
                dashed.setDashPattern([4 / width, 3 / width])
                painter.setPen(dashed)
                painter.drawLine(end, QtCore.QPointF(axis.x(axis.limit), end.y()))
            if hovered:
                painter.setPen(QtGui.QPen(self.color("surface.card"), 1.5))
                painter.setBrush(colour)
                painter.drawEllipse(end, 4, 4)

    def _label(
        self,
        painter: QtGui.QPainter,
        x: float,
        baseline: float,
        text: str,
        anchor: str,
        token: str = "text.tertiary",
        *,
        bold: bool = False,
    ) -> None:
        style = TextStyle(10.5, 600, 13, tabular=True) if bold else CHART_LABEL
        width = style.advance(text)
        x -= width if anchor == "end" else width / 2 if anchor == "middle" else 0
        painter.save()
        painter.setPen(self.color(token))
        paint_glyphs(painter, x, baseline, text, style)
        painter.restore()


def _distance(point: QtCore.QPointF, a: QtCore.QPointF, b: QtCore.QPointF) -> float:
    dx, dy = b.x() - a.x(), b.y() - a.y()
    length = dx * dx + dy * dy
    share = (
        0.0
        if length == 0
        else max(0.0, min(1.0, ((point.x() - a.x()) * dx + (point.y() - a.y()) * dy) / length))
    )
    return math.hypot(point.x() - a.x() - share * dx, point.y() - a.y() - share * dy)
