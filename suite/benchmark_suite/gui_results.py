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

"""Results (16, 17): the selected campaign's runs and its scientific reports.

Runs: the paged attempt or experiment table (filters, excluded outcomes muted and marked, censored time limits)
with a footer counting reported and excluded runs over the current filter, and the run summary (path map from
``results.map``, recorded coverage from ``measurements.query``, first-reached times and values from
``results.get``). Reports: the report list, its details from the manifest, its artifacts and a preview.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_tokens
from .contracts import REPORTED_OUTCOMES
from .gui_cells import (
    CELL_ROLE,
    Cell,
    ColumnStyle,
    OverlayScrollBar,
    TableFooter,
    TableLook,
    style_table,
)
from .gui_controls import KeySelect, StatusBadge, UnderlineTabs, badge_look
from .gui_maps import LiveLayers, WorldMap, decode_grey
from .gui_mapview import LiveMap
from .gui_marks import CoverageChart, CoverageSample, KeyValue, KeyValueList
from .gui_monitor import (
    MILESTONES,
    chart_end,
    coverage_samples,
    measured,
    milestone_marks,
    reached_by_share,
    run_average_speed,
)
from .gui_style import cell_text, duration, fixed, humanize, number
from .gui_support import PagedTable, PagedTableModel, Transport, stale_engine_message
from .gui_theme import ThemeManager, css_color, snap
from .gui_widgets import (
    BODY,
    PAGE_TITLE,
    Button,
    Card,
    CardHead,
    CardSplit,
    FlexRow,
    Rule,
    SegmentedControl,
    TextLine,
    TextStyle,
    Themed,
    draw_text,
    place,
    plural,
)
from .monitoring import PathCursor, accept_page

OUTCOMES = (
    "completed",
    "time_limit",
    "no_movement",
    "startup_failure",
    "runtime_failure",
    "timing_rejection",
    "engine_interruption",
    "manual_discard",
    "manual_stop",
    "storage_failure",
)
LEGEND = TextStyle(12, 400, 18)
SECTION = TextStyle(12, 600, 18)
KV_LABEL = TextStyle(12.5, 400, 18)
REACHED_KEY = TextStyle(12.5, 400, 18)
REACHED_SHARE = TextStyle(11.5, 400, 14)
REACHED_VALUE = TextStyle(14, 600, 18, tabular=True)
FOOT = TextStyle(12.5, 400, 18)
FOOT_COUNT = TextStyle(12.5, 600, 18)
COUNT = TextStyle(13, 600, 18)
MONO_SMALL = TextStyle(11.5, 400, 18, mono=True)
COVERAGE_PAGE = 250


# ---------- the runs table ----------

RUN_COLUMNS = [
    ("robot_id", "Robot"),
    ("outcome", "Outcome"),
    ("world_id", "World"),
    ("profile_name", "Profile"),
    ("spawn_name", "Spawn"),
    ("repetition", "Rep."),
    ("coverage_percent", "Coverage"),
    ("simulation_elapsed", "Sim time"),
    ("distance_m", "Distance"),
    ("average_speed_mps", "Avg speed"),
]
RUN_LOOK = TableLook(
    key="results.runs",
    columns=[
        ColumnStyle("name", width=168, padding_left=20, padding_right=10),
        ColumnStyle("badge", width=204, padding_left=10, padding_right=10),
        ColumnStyle("text", width=84, padding_left=10, padding_right=10),
        ColumnStyle("text", padding_left=10, padding_right=10),
        ColumnStyle("text", width=70, padding_left=10, padding_right=10),
        ColumnStyle("number", width=40, padding_left=10, padding_right=10),
        ColumnStyle("number", width=78, padding_left=10, padding_right=10),
        ColumnStyle("number", width=96, padding_left=10, padding_right=10),
        ColumnStyle("number", width=84, padding_left=10, padding_right=10),
        ColumnStyle("number", width=96, padding_left=10, padding_right=20),
    ],
)


def excluded(record: Mapping[str, Any]) -> bool:
    """A terminal outcome that reports leave out (``reported-outcomes-v1``)."""
    outcome = record.get("outcome")
    return isinstance(outcome, str) and outcome not in REPORTED_OUTCOMES


class RunsModel(PagedTableModel):
    """Run pages with design cells: excluded outcomes muted with the "excluded" mark and their row in
    ``text.secondary``; censored time limits marked "limit"; unavailable values "—"."""

    def __init__(self) -> None:
        super().__init__(RUN_COLUMNS)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == CELL_ROLE and index.isValid():
            record = self.record(index.row())
            key = self.columns[index.column()][0]
            if record is None:
                return Cell("…") if index.column() == 0 else Cell(kind="empty")
            return self.design_cell(record, key)
        return super().data(index, role)

    def design_cell(self, record: Mapping[str, Any], key: str) -> Cell:
        out = excluded(record)
        token = "text.secondary" if out else None
        missing = Cell("—", "number", token="text.tertiary")
        if key in ("robot_id", "world_id"):
            raw = str(record.get(key, ""))
            name = self.names.get(key, {}).get(raw, humanize(raw))
            if key == "robot_id":
                return Cell(name, "text" if out else "name", token=token)
            return Cell(name, token=token)
        if key == "outcome":
            outcome = record.get("outcome")
            if not outcome:
                state = record.get("state") or record.get("phase") or ""
                return Cell(humanize(str(state)), token="text.secondary")
            look = badge_look("outcome", str(outcome), muted=out)
            if look is None:
                return Cell(humanize(str(outcome)), token=token)
            return Cell(kind="badge", badge=look, note="excluded" if out else "")
        if key in ("profile_name", "spawn_name"):
            return Cell(str(record.get(key) or "—"), token=token)
        if key == "repetition":
            value = record.get("repetition")
            return (
                Cell(str(value + 1), "number", token=token) if isinstance(value, int) else missing
            )
        if key == "coverage_percent":
            text = fixed(record.get("coverage_percent"), 1)
            return Cell(text, "number", unit="%", token=token) if text else missing
        if key == "simulation_elapsed":
            seconds = number(record.get("simulation_elapsed"))
            if seconds is None:
                return missing
            limit = record.get("outcome") == "time_limit"
            return Cell(
                duration(seconds),
                "number",
                note="limit" if limit else "",
                note_kind="limit",
                token=token,
            )
        if key == "distance_m":
            text = fixed(record.get("distance_m"), 2)
            return Cell(text, "number", unit="m", token=token) if text else missing
        if key == "average_speed_mps":
            text = fixed(run_average_speed(record), 2)
            return Cell(text, "number", unit="m/s", token=token) if text else missing
        return Cell(kind="empty")


class RunsTable(PagedTable):
    """The paged runs table (``results.query`` attempts or experiments, request ID ``result-table``)."""

    def __init__(self, transport: Transport, themes: ThemeManager) -> None:
        model = RunsModel()
        super().__init__(transport, "result-table", "attempts", RUN_COLUMNS, model=model)
        self.label.hide()
        layout = self.layout()
        if layout is not None:
            layout.setSpacing(0)
        style_table(self.view, themes, RUN_LOOK)


class RunsFooter(TableFooter):
    """``.tbl-foot``: "<b>7</b> reported · <b>3</b> excluded from reports" (counts over the current filter)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent, padding=20)
        self.counts: tuple[int | None, int | None] = (None, None)

    def set_counts(self, reported: int | None, excluded_runs: int | None) -> None:
        if (reported, excluded_runs) != self.counts:
            self.counts = (reported, excluded_runs)
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(QtCore.QRect(0, 0, self.width(), 1), self.color("border.line"))
        top = 1 + (39 - 18) / 2
        x = 20.0
        ink, secondary = self.color("text.ink"), self.color("text.secondary")
        for index, (count, words) in enumerate(
            zip(self.counts, (" reported", " excluded from reports"), strict=True)
        ):
            if index:
                painter.save()
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(self.color("text.tertiary"))
                painter.drawEllipse(QtCore.QPointF(x + 8 + 1.5, top + 9), 1.5, 1.5)
                painter.restore()
                x += 8 + 3 + 8
            text = "—" if count is None else f"{count:,}"
            x += draw_text(painter, x, top, text, FOOT_COUNT, ink)
            x += draw_text(painter, x, top, words, FOOT, secondary)


class RunsCard(Card):
    """``.runs``: the filter bar (Attempts · Experiments, then the robot, world, outcome, profile and spawn
    selects at 132 px), the table and the footer."""

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.bar = FlexRow(self, gap=8, left=20, right=12)
        self.kind = SegmentedControl(themes, [("Attempts", None), ("Experiments", None)])
        self.bar.add(self.kind)
        rule = Rule(themes, vertical=True)
        rule.setFixedHeight(27)
        self.bar.add(rule, width=1, margin=4)
        self.robot, self.world, self.outcome = (
            KeySelect(themes),
            KeySelect(themes),
            KeySelect(themes),
        )
        self.profile, self.spawn = KeySelect(themes), KeySelect(themes)
        for select in (self.robot, self.world, self.outcome, self.profile, self.spawn):
            self.bar.add(select, width=132)
        self.robot.addItem("All robots", None)
        self.world.addItem("All worlds", None)
        self.profile.addItem("All profiles", None)
        self.spawn.addItem("All spawns", None)
        self.outcome.addItem("All outcomes", None)
        for value in OUTCOMES:
            self.outcome.addItem(humanize(value), value)
        self.table = RunsTable(transport, themes)
        self.table.setParent(self)
        # Narrow windows: the columns keep their widths and the table scrolls sideways.
        self.table.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.footer = RunsFooter(themes, self)
        self.selects = (self.robot, self.world, self.outcome, self.profile, self.spawn)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        # The selects are 132 px; in a narrow card they share the room left (at least 80 px each).
        room = width - 2 - 20 - 12 - self.kind.natural_width() - 8 - 9 - 8 * len(self.selects)
        select = max(80.0, min(132.0, room / len(self.selects)))
        for widget in self.selects:
            self.bar.set_width(widget, select)
        self.bar.setGeometry(1, 1, width - 2, 51)
        self.table.setGeometry(1, 53, width - 2, max(0, height - 53 - 41))
        self.footer.setGeometry(1, height - 41, width - 2, 40)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        QtGui.QPainter(self).fillRect(
            QtCore.QRect(1, 52, self.width() - 2, 1), self.color("border.line")
        )


# ---------- the run summary ----------


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


class RunMapLoader(QtCore.QObject):
    """``results.map`` for the selected run (request IDs ``run-map:<uuid>``): one request in flight, stale
    replies re-requested, building and running maps polled; the status text goes to the legend line."""

    status = QtCore.Signal(str, bool)  # text, ready

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
        if identity == self.state.attempt_id and identity is not None:
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
    """``.fr``: 25, 50, 75, 90 and 95 % with the simulation time each was first reached (44 px,
    rules above and below, a rule before each cell)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.values: list[str] = ["—"] * len(MILESTONES)
        self.setFixedHeight(44)

    def set_values(self, values: Sequence[str]) -> None:
        self.values = list(values)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        line = theme.color("border.line")
        width = self.width()
        painter.fillRect(QtCore.QRect(0, 0, width, 1), line)
        painter.fillRect(QtCore.QRect(0, 43, width, 1), line)
        cell = width / len(MILESTONES)
        block = 1 + (42 - 32) / 2
        for index, (percent, value) in enumerate(zip(MILESTONES, self.values, strict=True)):
            left = index * cell
            inset = 13 if index else 0
            if index:
                painter.fillRect(QtCore.QRectF(snap(left), 1, 1, 42), line)
            secondary = theme.color("text.secondary")
            draw_text(painter, left + inset, block, f"{percent} %", REACHED_SHARE, secondary)
            ink = theme.color("text.ink")
            draw_text(painter, left + inset, block + 14, value, REACHED_VALUE, ink)


class RunSummary(Card):
    """``.sum``: the run's robot and Open folder; the map and its legend; "Coverage over simulation time"; the
    first-reached times; Covered area, Real-time factor | CPU average, RSS average, Coverage status (values the
    row does not show). The map and the chart share the card's height: by default the map takes ``MAP_SHARE``
    of it, at most what the world's aspect needs at the card's width; dragging the gap under the map resizes
    it (remembered; a double click restores the default; up to the world's aspect) and the chart takes the rest.
    The body under the head scrolls when the map and the chart need more than the card's height."""

    MAP_MIN, CHART_MIN = 160, 112
    MAP_SHARE = 0.62
    # Everything in the body except the map and the chart: margins, legend, chart title, first-reached, lists.
    FIXED = 14 + 10 + 18 + 16 + 18 + 6 + 4 + 44 + 96 + 6

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "Run summary", self)
        self.replay = Button(themes, "Replay", icon="play", variant="ghost", small=True)
        self.replay.setToolTip(
            "Replay this run on the Monitor: map, explorer decisions and coverage over time"
        )
        self.replay.setEnabled(False)
        self.head.add(self.replay)
        self.open_folder = Button(themes, "Open folder", icon="folder", variant="ghost", small=True)
        self.open_folder.setEnabled(False)
        self.head.add(self.open_folder)
        self.area = QtWidgets.QScrollArea(self)
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.area.viewport().setAutoFillBackground(False)
        self.body = QtWidgets.QWidget()
        self.body.setAutoFillBackground(False)
        self.area.setWidget(self.body)
        # The card paints its own fill: the area, its viewport and the body stay transparent.
        self.area.viewport().setObjectName("runSummaryViewport")
        self.body.setObjectName("runSummaryBody")
        self.area.setStyleSheet(
            "QScrollArea, #runSummaryViewport, #runSummaryBody { background: transparent; }"
        )
        OverlayScrollBar(themes, self.area)
        body = self.body
        self.map = RunMapView(themes, body)
        self.map.world_changed.connect(self._layout)
        self.map_split = CardSplit(themes, body, "results/summary-map", self._layout, across=False)
        self.legend = MapLegend(themes, body)
        self.chart_title = TextLine(
            themes, "Coverage over simulation time", SECTION, "text.secondary", body
        )
        self.chart = CoverageChart(themes, body)
        self.reached = FirstReached(themes, body)
        self.lists = [
            KeyValueList(themes, row_height=32, label_style=KV_LABEL, last_open=True, parent=body)
            for _ in range(2)
        ]
        self.directory = ""
        self.clear_values()

    def clear_values(self) -> None:
        self.reached.set_values(["—"] * len(MILESTONES))
        self.chart.set_data([])
        self.set_values({})

    def set_values(self, metrics: Mapping[str, Any]) -> None:
        status = metrics.get("coverage_status")
        self.lists[0].set_rows(
            [
                measured("Covered area", metrics.get("coverage_covered_m2"), 1, "m²"),
                KeyValue("Real-time factor", fixed(metrics.get("rtf"), 3) or "—"),
            ]
        )
        self.lists[1].set_rows(
            [
                measured("CPU average", metrics.get("cpu_average_percent"), 1, "%"),
                measured("RSS average", metrics.get("rss_average_mib"), 1, "MiB"),
                KeyValue("Coverage status", humanize(str(status)) if status else "—"),
            ]
        )
        self.reached.set_values(
            [
                "—"
                if item.reached is None
                else duration(item.seconds)
                if item.reached
                else "Not reached"
                for item in reached_by_share(metrics)
            ]
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def _layout(self) -> None:
        width, height = self.width(), self.height()
        self.head.setGeometry(1, 1, width - 2, 52)
        self.area.setGeometry(1, 53, width - 2, max(0, height - 54))
        # Body coordinates: the card's 21 px content edge is 20 px into the body (1 px border).
        inner = width - 42
        free = height - 54 - self.FIXED  # for the map and the chart
        aspect = max(80, round(self.map.height_for(inner)))
        high = max(self.MAP_MIN, aspect)  # dragged taller than the card fits, the body scrolls
        default = min(aspect, max(self.MAP_MIN, free * self.MAP_SHARE))
        stage = round(self.map_split.resolve(default, self.MAP_MIN, high))
        chart = max(self.CHART_MIN, free - stage)
        self.map.setGeometry(20, 14, inner, stage)
        self.map_split.place(20, 14 + stage, inner, 10)
        y = 14 + stage + 10
        self.legend.setGeometry(20, y, inner, 18)
        y += 18 + 16
        place(self.chart_title, 20, y, inner, 18)
        y += 18 + 6
        self.chart.setGeometry(20, y, inner, chart)
        y += chart + 4
        self.reached.setGeometry(20, y, inner, 44)
        y += 44
        half = (inner - 24) / 2
        place(self.lists[0], 20, y, half, 96)
        place(self.lists[1], 20 + half + 24, y, half, 96)
        self.body.setMinimumHeight(y + 96 + 6)


# ---------- reports ----------

REPORT_COLUMNS = [("created_ns", "Created"), ("state", "State"), ("detail", "Details")]
REPORT_LOOK = TableLook(
    key="results.reports",
    columns=[
        ColumnStyle("number", align="left", width=170, padding_left=20),
        ColumnStyle("badge", width=118),
        ColumnStyle("text", padding_right=20),
    ],
)
ARTIFACT_COLUMNS = [("kind", "Artifact"), ("bytes", "Size"), ("path", "File")]
ARTIFACT_LOOK = TableLook(
    key="results.artifacts",
    columns=[
        ColumnStyle("rep", padding_left=20),
        ColumnStyle("number", width=96),
        ColumnStyle("mono", width=206, padding_right=20),
    ],
)
ARTIFACT_ICONS = {".pdf": "file", ".png": "image", ".parquet": "table", ".json": "braces"}


class ReportsModel(PagedTableModel):
    def __init__(self) -> None:
        super().__init__(REPORT_COLUMNS)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == CELL_ROLE and index.isValid():
            record = self.record(index.row())
            key = self.columns[index.column()][0]
            if record is None:
                return Cell("…") if index.column() == 0 else Cell(kind="empty")
            if key == "created_ns":
                return Cell(cell_text("created_ns", record.get("created_ns")), "number")
            if key == "state":
                look = badge_look("report", str(record.get("state", "")))
                return (
                    Cell(kind="badge", badge=look)
                    if look
                    else Cell(humanize(str(record.get("state", ""))))
                )
            return Cell(
                str(record.get("detail") or "—"),
                token=None if record.get("detail") else "text.tertiary",
            )
        return super().data(index, role)


class ArtifactsModel(PagedTableModel):
    def __init__(self) -> None:
        super().__init__(ARTIFACT_COLUMNS)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == CELL_ROLE and index.isValid():
            record = self.record(index.row())
            key = self.columns[index.column()][0]
            if record is None:
                return Cell("…") if index.column() == 0 else Cell(kind="empty")
            path = str(record.get("path", ""))
            if key == "kind":
                icon = ARTIFACT_ICONS.get(Path(path).suffix.lower(), "file")
                return Cell(
                    humanize(str(record.get("kind", ""))), "rep", icon=icon, token="text.ink"
                )
            if key == "bytes":
                return Cell(cell_text("bytes", record.get("bytes")), "number")
            return Cell(Path(path).name, "mono", tooltip=path)
        return super().data(index, role)


class TableBox(Card):
    """A card with a ruled head over a design table (and an optional footer)."""

    def __init__(
        self,
        themes: ThemeManager,
        title: str,
        table: QtWidgets.QWidget,
        *,
        footer: TableFooter | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, title, self)
        self.table = table
        table.setParent(self)
        self.footer = footer
        if footer is not None:
            footer.setParent(self)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.head.setGeometry(1, 1, width - 2, 52)
        bottom = height - 1 - (40 if self.footer is not None else 0)
        self.table.setGeometry(1, 53, width - 2, max(0, bottom - 53))
        if self.footer is not None:
            self.footer.setGeometry(1, bottom, width - 2, 40)


class ReportDetails(Card):
    """Report details: the policy version at the right of the head; "Reported" and "Excluded", each with its
    count (600 ink) and, when the manifest has them, the outcomes behind it (D16; older manifests: the total
    only). Rows 40 px, labels in a 110 px column."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "Report details", self)
        self.policy = TextLine(themes, "", MONO_SMALL, "text.tertiary")
        self.head.add(self.policy)
        self.rows: list[tuple[str, str, str]] = []

    def show_manifest(self, report: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
        policy = (manifest.get("selection") or {}).get("report_policy") or {}
        self.policy.setText(str(policy.get("version") or ""))
        self.head.row.relayout()

        def breakdown(counts: Any) -> str:
            if not isinstance(counts, Mapping):
                return ""
            order = {outcome: index for index, outcome in enumerate(OUTCOMES)}
            ranked = sorted(counts.items(), key=lambda item: order.get(str(item[0]), len(order)))
            return ", ".join(
                f"{count} {humanize(str(outcome)).lower()}" for outcome, count in ranked if count
            )

        reported = policy.get("reported_final_experiments")
        excluded_counts = policy.get("excluded")
        total = (
            sum(v for v in excluded_counts.values() if isinstance(v, int))
            if isinstance(excluded_counts, Mapping)
            else None
        )
        self.rows = [
            (
                "Reported",
                f"{reported:,}" if isinstance(reported, int) else "—",
                breakdown(policy.get("reported")),
            ),
            ("Excluded", "—" if total is None else f"{total:,}", breakdown(excluded_counts)),
        ]
        detail = str(report.get("detail") or "")
        if detail and report.get("state") == "FAILED":
            self.rows.append(("Message", "", detail))
        self.update()

    def clear(self) -> None:
        self.policy.setText("")
        self.rows = []
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.head.setGeometry(1, 1, self.width() - 2, 52)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        right = self.width() - 21
        for index, (label, count, detail) in enumerate(self.rows):
            top = 53 + index * 40
            last = index == len(self.rows) - 1
            if not last:
                painter.fillRect(
                    QtCore.QRectF(21, top + 39, right - 21, 1), self.color("border.line")
                )
            line_top = top + ((40 if last else 39) - 18) / 2
            draw_text(painter, 21, line_top, label, BODY, self.color("text.secondary"))
            x = 21 + 110.0
            if count:
                x += draw_text(painter, x, line_top, count, COUNT, self.color("text.ink")) + 6
            draw_text(
                painter, x, line_top, BODY.elide(detail, right - x), BODY, self.color("text.body")
            )


class PreviewCard(Card):
    """The selected artifact: its name and Open file in the head; a PNG fitted with smooth scaling (dark theme:
    on the report paper), otherwise "No preview for this file type"."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "Preview", self)
        self.open_file = Button(themes, "Open file", icon="aupright", variant="ghost", small=True)
        self.open_file.setEnabled(False)
        self.head.add(self.open_file)
        self.image: QtGui.QImage | None = None
        self.message = "Select an artifact to preview it"

    def show_message(self, text: str) -> None:
        self.image, self.message = None, text
        self.update()

    def show_image(self, image: QtGui.QImage) -> None:
        self.image, self.message = image, ""
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.head.setGeometry(1, 1, self.width() - 2, 52)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        body = QtCore.QRectF(1, 53, self.width() - 2, self.height() - 54)
        if self.image is None or self.image.isNull():
            text = BODY.elide(self.message, body.width() - 40)
            x = body.x() + (body.width() - BODY.advance(text)) / 2
            draw_text(painter, x, body.center().y() - 9, text, BODY, self.color("text.secondary"))
            return
        if self.theme.name == "dark":
            # The generated figure is white: it sits on the report paper (``.paper``, margin 12 16 16).
            paper = body.adjusted(16, 12, -16, -16)
            painter.save()
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(css_color(gui_tokens.REPORT_PAPER_DARK))
            painter.drawRoundedRect(paper, 4, 4)
            painter.restore()
            area = paper.adjusted(7, 12, -7, -12)
        else:
            area = body.adjusted(20, 16, -20, -16)
        size = QtCore.QSizeF(self.image.size()).scaled(
            area.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        target = QtCore.QRectF(
            area.x() + (area.width() - size.width()) / 2, area.y(), size.width(), size.height()
        )
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self.image)


class ReportBrowser(QtWidgets.QWidget):
    """Reports (17): the report list (``report-table``), its details, its artifacts (``report-artifacts``,
    ``reports.get``) and the preview of a PNG (``local.read_image``); Generate reports is in the page head."""

    def __init__(self, themes: ThemeManager, transport: Transport) -> None:
        super().__init__()
        self.themes, self.transport = themes, transport
        self.campaign_id: str | None = None
        self.report_id: str | None = None
        self.artifact: dict[str, Any] | None = None
        self.preview_tag = ""
        self.generation_tag = ""
        self.reports = PagedTable(
            transport, "report-table", "reports", REPORT_COLUMNS, model=ReportsModel()
        )
        self.reports.label.hide()
        style_table(self.reports.view, themes, REPORT_LOOK)
        header = self.reports.view.horizontalHeader()
        header.setSortIndicatorShown(True)  # newest first (the engine's order), as 17 marks it
        header.setSortIndicator(0, QtCore.Qt.SortOrder.DescendingOrder)
        self.reports.selected.connect(self.select_report)
        self.artifact_table = PagedTable(
            transport,
            "report-artifacts",
            "artifacts",
            ARTIFACT_COLUMNS,
            method="reports.get",
            result_key="artifacts",
            model=ArtifactsModel(),
        )
        self.artifact_table.label.hide()
        style_table(self.artifact_table.view, themes, ARTIFACT_LOOK)
        self.artifact_table.setEnabled(False)
        self.artifact_table.sync_timer.stop()
        self.artifacts, self.model = self.artifact_table.view, self.artifact_table.model
        self.artifact_table.selected.connect(self.select_artifact)
        self.artifact_table.loaded.connect(self.show_report)
        self.artifacts.doubleClicked.connect(lambda _: self.open_artifact())
        self.list_card = TableBox(themes, "Scientific reports", self.reports, parent=self)
        self.details = ReportDetails(themes, self)
        self.files_footer = TableFooter(themes, padding=20)
        self.files_card = TableBox(
            themes, "Artifacts", self.artifact_table, footer=self.files_footer, parent=self
        )
        self.generated = TextLine(themes, "", BODY, "text.secondary")
        self.files_card.head.add(self.generated)
        self.preview = PreviewCard(themes, self)
        # The gaps between the cards can be dragged (remembered; a double click restores the design).
        self.left = CardSplit(themes, self, "reports/left", self.relayout, across=True)
        self.top = CardSplit(themes, self, "reports/top", self.relayout, across=False)
        self.preview.open_file.clicked.connect(self.open_artifact)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def relayout(self) -> None:
        width, height = self.width(), self.height()
        default = 560 if width >= 1200 else max(360, (width - 16) * 0.4)
        left = self.left.resolve(default, 360, width - 16 - 360)
        top = self.top.resolve(170, 2 + 52 + 36 + 40, height - 16 - 200)
        place(self.list_card, 0, 0, left, top)
        place(self.details, left + 16, 0, width - left - 16, top)
        place(self.files_card, 0, top + 16, left, max(0, height - top - 16))
        place(self.preview, left + 16, top + 16, width - left - 16, max(0, height - top - 16))
        self.left.place(left, 0, 16, height)
        self.top.place(0, top, width, 16)

    def set_campaign(self, identity: str) -> None:
        if identity == self.campaign_id:
            return
        self.campaign_id = identity
        self.report_id = self.artifact = None
        self.preview_tag = self.generation_tag = ""
        self.artifact_table.setEnabled(False)
        self.artifact_table.sync_timer.stop()
        self.artifact_table.filters = {}
        self.model.replace([])
        self.artifacts.clearSelection()
        self.preview.show_message("Select an artifact to preview it")
        self.preview.open_file.setEnabled(False)
        self.details.clear()
        self.generated.setText("")
        self.files_footer.set_text("")
        self.reports.filters = {"campaign_id": identity}
        self.reports.model.replace([])
        self.reports.view.clearSelection()
        self.reports.reset_page()

    def select_report(self, row: dict[str, Any]) -> None:
        expected = self.reports.filters.get("campaign_id")
        if expected and row["campaign_id"] != expected:
            return
        if self.report_id == row["id"]:
            return
        self.report_id = row["id"]
        self.campaign_id = row["campaign_id"]
        self.artifact = None
        self.preview_tag = ""
        self.preview.show_message("Select an artifact to preview it")
        self.preview.open_file.setEnabled(False)
        self.artifact_table.setEnabled(True)
        self.artifact_table.filters = {"report_id": self.report_id}
        self.artifact_table.reset_page()
        self.artifact_table.sync_timer.start()

    def select_artifact(self, row: dict[str, Any]) -> None:
        self.artifact = row
        self.preview_tag = ""
        path = Path(str(row["path"]))
        self.preview.head.title.setText(humanize(str(row.get("kind") or path.name)))
        self.preview.open_file.setEnabled(True)
        self.preview.head.row.relayout()
        if path.suffix == ".png":
            self.preview.show_message("Loading the preview…")
            self.preview_tag = "report-preview-" + uuid.uuid4().hex
            self.transport.submit(self.preview_tag, "local.read_image", {"path": str(path)})
        else:
            self.preview.show_message("No preview for this file type")

    def open_artifact(self) -> None:
        if self.artifact:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.artifact["path"]))

    def generate(self) -> None:
        if self.campaign_id:
            self.generation_tag = "report-generate:" + self.campaign_id + ":" + uuid.uuid4().hex
            self.transport.submit(
                self.generation_tag, "reports.generate", {"campaign_id": self.campaign_id}
            )

    def show_report(self, result: dict[str, Any]) -> None:
        report, manifest = result["report"], result.get("manifest") or {}
        self.details.show_manifest(report, manifest)
        start, end = report.get("created_ns"), report.get("finished_ns")
        parts = []
        if isinstance(start, int):
            parts.append(cell_text("created_ns", start))
            if isinstance(end, int):
                parts[-1] += " – " + cell_text("finished_ns", end).split(" · ")[-1]
        elapsed = manifest.get("elapsed_wall_seconds")
        if number(elapsed) is not None:
            parts.append(duration(elapsed))
        self.generated.setText(("Generated " + " · ".join(parts)) if parts else "")
        self.files_card.head.row.relayout()
        self.files_footer.set_text(plural(int(result.get("total") or 0), "file"))

    def received(self, tag: str, result: Any) -> None:
        if tag == self.preview_tag:
            image = QtGui.QImage.fromData(result["image"])
            if image.isNull():
                self.preview.show_message("Preview could not be decoded")
            else:
                self.preview.show_image(image)
        elif tag == self.generation_tag:
            self.reports.refresh()

    def failed(self, tag: str, message: str) -> None:
        if tag == self.preview_tag:
            self.preview.show_message(message)


# ---------- the page ----------


class RunsPane(QtWidgets.QWidget):
    """The Runs tab: the runs card and, 16 px to its right, the 420 px run summary."""

    def __init__(self, runs: RunsCard, summary: RunSummary) -> None:
        super().__init__()
        self.runs, self.summary = runs, summary
        runs.setParent(self)
        summary.setParent(self)
        self.side = CardSplit(
            runs.themes, self, "results/summary", self.relayout, across=True, sign=-1
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def relayout(self) -> None:
        width, height = self.width(), self.height()
        default = 420 if width >= 1300 else 360  # narrow windows: a narrower summary, a wider table
        side = self.side.resolve(default, 320, width - 16 - 480)
        place(self.runs, 0, 0, max(0, width - side - 16), height)
        place(self.summary, width - side, 0, side, height)
        self.side.place(width - side - 16, 0, 16, height)


class ResultsPage(QtWidgets.QWidget):
    """Results: head (campaign select, its state, Generate reports), Runs and Reports tabs."""

    campaign_chosen = QtCore.Signal(str, str)  # id, name
    # The selected run's attempt and directory: the Monitor's Logs view reads it until a live run is chosen
    # there (as before the redesign).
    attempt_selected = QtCore.Signal(str, object)
    replay_requested = QtCore.Signal(str)  # attempt id
    notify = QtCore.Signal(str, str)  # text, kind

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes, self.transport = themes, transport
        self.campaign_id: str | None = None
        self.campaign_name = ""
        self.campaign_state = ""
        self.campaigns: list[dict[str, Any]] = []
        self.result_selection: str | None = None
        self.result_row: dict[str, Any] = {}
        self.count_generation = 0
        self.counted_total: int | None = None
        self.counts: dict[str, int] = {}
        self.coverage_tag = ""
        self.coverage_attempt: str | None = None
        self.coverage_offset = 0
        self.samples: list[CoverageSample] = []
        self.gap = False
        self.metrics: dict[str, Any] = {}

        self.head = FlexRow(self, gap=12)
        self.head.add(TextLine(themes, "Results", PAGE_TITLE, "text.ink"))
        rule = Rule(themes, vertical=True)
        rule.setFixedHeight(24)
        self.head.add(rule, width=1, margin=8)
        self.campaign = KeySelect(themes, "Campaign")
        self.campaign.addItem("No campaign selected", None)
        self.head.add(self.campaign, width=305)
        self.state = StatusBadge(themes)
        self.head.add(self.state)
        self.head.add_stretch()
        self.generate = Button(themes, "Generate reports", icon="retry", variant="primary")
        self.head.add(self.generate)
        self.tabs = UnderlineTabs(themes, [("Runs", "table"), ("Reports", "file")], self)
        self.stack = QtWidgets.QStackedWidget(self)
        self.runs = RunsCard(themes, transport)
        self.summary = RunSummary(themes)
        self.stack.addWidget(RunsPane(self.runs, self.summary))
        self.report_browser = ReportBrowser(themes, transport)
        self.stack.addWidget(self.report_browser)
        self.tabs.changed.connect(self._tab)
        self.results = self.runs.table
        self.reports = self.report_browser.reports
        self.reports.loaded.connect(self._reports_loaded)
        self.loader = RunMapLoader(transport, self.summary.map)
        self.loader.status.connect(self.summary.legend.set_status)
        self.summary.open_folder.clicked.connect(self.open_folder)
        self.replay_attempt: str | None = None
        self.summary.replay.clicked.connect(self._replay)
        self.generate.clicked.connect(self._generate)
        self.campaign.currentIndexChanged.connect(self._campaign_picked)
        self.results.selected.connect(self.select_result)
        self.results.loaded.connect(self._page_loaded)
        self.filter_timer = QtCore.QTimer(self)
        self.filter_timer.setSingleShot(True)
        self.filter_timer.setInterval(0)
        self.filter_timer.timeout.connect(self.filter_results)
        self.runs.kind.changed.connect(lambda _: self.filter_timer.start())
        for select in (
            self.runs.robot,
            self.runs.world,
            self.runs.outcome,
            self.runs.profile,
            self.runs.spawn,
        ):
            select.currentIndexChanged.connect(lambda _: self.filter_timer.start())
        self.campaign_sync = QtCore.QTimer(self)
        self.campaign_sync.setInterval(5000)
        self.campaign_sync.timeout.connect(
            lambda: self.load_campaigns() if self.isVisible() else None
        )
        self.campaign_sync.start()

    # campaigns (D17)

    def load_campaigns(self) -> None:
        self.transport.submit(
            "results-campaigns",
            "results.query",
            {"table": "campaigns", "filters": {}, "offset": 0, "limit": 250, "summary": True},
        )

    def _campaigns_received(self, result: dict[str, Any]) -> None:
        rows = sorted(result.get("rows") or [], key=lambda row: -(row.get("created_ns") or 0))
        self.campaigns = rows
        self.campaign.blockSignals(True)
        self.campaign.clear()
        self.campaign.addItem("No campaign selected", None)
        for row in rows:
            self.campaign.addItem(str(row.get("name") or str(row["id"])[:8]), row["id"])
        self._ensure_listed()
        index = self.campaign.findData(self.campaign_id)
        self.campaign.setCurrentIndex(max(0, index))
        self.campaign.blockSignals(False)
        self._show_state()

    def _campaign_picked(self, index: int) -> None:
        identity = self.campaign.itemData(index)
        if identity and identity != self.campaign_id:
            self.campaign_chosen.emit(str(identity), self.campaign.itemText(index))
        elif not identity:  # "No campaign selected" is not a choice: back to the selected campaign
            self.campaign.blockSignals(True)
            self.campaign.setCurrentIndex(max(0, self.campaign.findData(self.campaign_id)))
            self.campaign.blockSignals(False)

    def _show_state(self) -> None:
        row = next((r for r in self.campaigns if r["id"] == self.campaign_id), None)
        if row is not None:
            self.campaign_state = str(row["state"])
        look = badge_look("campaign", self.campaign_state) if self.campaign_state else None
        self.state.set_look(look)
        self.state.setVisible(look is not None)
        self._size_select()

    def set_state(self, state: str) -> None:
        """The selected campaign's state from the window's ``campaign.status`` poll."""
        if state != self.campaign_state:
            self.campaign_state = state
            self._show_state()

    def _size_select(self) -> None:
        """``.select`` at least 300 px, as wide as its key and current text (and its chevron)."""
        text = BODY.advance("Campaign") + 2 + BODY.advance(" " + self.campaign.currentText())
        self.head.set_width(self.campaign, max(300, math.ceil(2 + 10 + text + 8 + 16 + 10)))

    def _ensure_listed(self) -> None:
        """The selected campaign stays in the select even when the list does not hold it (older than the 250
        newest)."""
        if self.campaign_id and self.campaign.findData(self.campaign_id) < 0:
            self.campaign.addItem(self.campaign_name or str(self.campaign_id)[:8], self.campaign_id)

    def set_campaign(self, identity: str | None, name: str = "") -> None:
        self.campaign_id, self.campaign_name, self.campaign_state = identity, name, ""
        self.campaign.blockSignals(True)
        if self.campaign.findData(identity) < 0 and identity:
            self.load_campaigns()
        self._ensure_listed()
        index = self.campaign.findData(identity)
        self.campaign.setCurrentIndex(max(0, index))
        self.campaign.blockSignals(False)
        self._show_state()
        self.report_browser.set_campaign(identity or "")
        self.generate.setEnabled(bool(identity))
        self.filter_results()

    # catalog and filter options

    def set_names(self, catalog: Mapping[str, Any]) -> None:
        self.results.model.set_names(
            {
                "robot_id": {r["id"]: r["label"] for r in catalog["robots"]},
                "world_id": {w["id"]: w["label"] for w in catalog["worlds"]},
            }
        )
        for select, items, everything in (
            (self.runs.robot, catalog["robots"], "All robots"),
            (self.runs.world, catalog["worlds"], "All worlds"),
        ):
            select.blockSignals(True)
            select.clear()
            select.addItem(everything, None)
            for item in items:
                select.addItem(item["label"], item["id"])
            select.blockSignals(False)

    def set_filter_options(
        self, profiles: Sequence[tuple[str, Any]], spawns: Sequence[tuple[str, Any]]
    ) -> None:
        """The profile and spawn filters (label, value); the current choice is kept when still offered."""
        for select, options, everything in (
            (self.runs.profile, profiles, "All profiles"),
            (self.runs.spawn, spawns, "All spawns"),
        ):
            chosen = select.currentData()
            select.blockSignals(True)
            select.clear()
            select.addItem(everything, None)
            for text, value in options:
                select.addItem(text, value)
            select.setCurrentIndex(max(0, select.findData(chosen)))
            select.blockSignals(False)

    # runs

    def kind_table(self) -> str:
        return "experiments" if self.runs.kind.index() == 1 else "attempts"

    def filter_results(self) -> None:
        self.result_selection = None
        self.result_row = {}
        self.summary.clear_values()
        self.summary.head.title.setText("Run summary")
        self.summary.open_folder.setEnabled(False)
        self.loader.clear()
        filters: dict[str, Any] = {"campaign_id": self.campaign_id} if self.campaign_id else {}
        for key, select in (
            ("robot_id", self.runs.robot),
            ("world_id", self.runs.world),
            ("profile_id", self.runs.profile),
            ("spawn_id", self.runs.spawn),
        ):
            value = select.currentData()
            if value:
                filters[key] = value
        if self.runs.outcome.currentIndex() > 0:
            filters["outcome"] = self.runs.outcome.currentData()
        self.results.table = self.kind_table()
        # Experiment rows carry no run values (only attempt summaries do): those columns are left out.
        for column in (6, 7, 8, 9):
            self.results.view.setColumnHidden(column, self.results.table == "experiments")
        self.counted_total = None
        self.results.filters, self.results.offset = filters, 0
        self.results.model.replace([])
        self.results.refresh()
        self.recount()

    def recount(self, *, keep: bool = False) -> None:
        """The footer's reported and excluded counts over the current filter: one count per outcome. A recount
        after the table refreshed (``keep``) shows the previous counts until all new ones are in."""
        self.count_generation += 1
        self.counts = {}
        if not keep:
            self.runs.footer.set_counts(None, None)
        chosen = self.results.filters.get("outcome")
        for outcome in OUTCOMES:
            if chosen and outcome != chosen:
                self.counts[outcome] = 0
                continue
            self.transport.submit(
                f"result-count:{self.count_generation}:{outcome}",
                "results.query",
                {
                    "table": self.results.table,
                    "filters": dict(self.results.filters, outcome=outcome),
                    "offset": 0,
                    "limit": 1,
                    "summary": True,
                },
            )
        self._show_counts()

    def _show_counts(self) -> None:
        if len(self.counts) < len(OUTCOMES):
            return
        reported = sum(
            count for outcome, count in self.counts.items() if outcome in REPORTED_OUTCOMES
        )
        out = sum(
            count for outcome, count in self.counts.items() if outcome not in REPORTED_OUTCOMES
        )
        self.runs.footer.set_counts(reported, out)

    def _reports_loaded(self, result: dict[str, Any]) -> None:
        total = result.get("total")
        if isinstance(total, int):
            self.tabs.set_count(1, total)

    def _page_loaded(self, result: dict[str, Any]) -> None:
        total = result.get("total")
        if isinstance(total, int):
            self.tabs.set_count(0, total)
            if total != self.counted_total:
                # Runs ended or started since the counts were read (the table re-reads every 2 s).
                self.counted_total = total
                self.recount(keep=True)

    def select_result(self, row: dict[str, Any]) -> None:
        self.result_selection = row["id"]
        self.result_row = row
        attempts = self.results.table == "attempts"
        identity = row["id"] if attempts else row.get("final_attempt_id")
        robot = str(row.get("robot_id", ""))
        self.summary.head.title.setText(
            self.results.model.names.get("robot_id", {}).get(robot, humanize(robot))
        )
        self.summary.directory = str(row.get("directory") or "")
        self.summary.open_folder.setEnabled(bool(self.summary.directory))
        self.replay_attempt = identity
        self.summary.replay.setEnabled(bool(identity))
        self.summary.clear_values()
        self.loader.show_attempt(identity, "No finished attempt for this experiment")
        self.transport.submit(
            "result-detail", "results.get", {"table": self.results.table, "id": row["id"]}
        )
        self._read_coverage(identity)
        if identity:
            self.attempt_selected.emit(identity, row.get("directory"))

    def _replay(self) -> None:
        if self.replay_attempt:
            self.replay_requested.emit(self.replay_attempt)

    def _read_coverage(self, attempt: str | None, offset: int = 0) -> None:
        if offset == 0:
            self.samples, self.gap = [], False
        self.coverage_attempt, self.coverage_offset = attempt, offset
        if attempt is None:
            self.coverage_tag = ""
            return
        self.coverage_tag = "result-coverage:" + uuid.uuid4().hex
        self.transport.submit(
            self.coverage_tag,
            "measurements.query",
            {"attempt_id": attempt, "kind": "coverage", "offset": offset, "limit": COVERAGE_PAGE},
        )

    def _coverage_received(self, result: dict[str, Any]) -> None:
        rows = result.get("rows") or []
        samples, self.gap = coverage_samples(rows, self.gap)
        self.samples.extend(samples)
        if rows and result.get("next_offset") is not None:
            self._read_coverage(self.coverage_attempt, self.coverage_offset + len(rows))
            return
        self.coverage_tag = ""
        self._update_chart()

    def _update_chart(self) -> None:
        milestones = milestone_marks(self.metrics, self.samples)
        end = chart_end(self.samples[-1].seconds) if self.samples else None
        self.summary.chart.set_data(self.samples, milestones, x_end=end)

    def _generate(self) -> None:
        """Generate reports: one request at a time; its reply or error re-enables the button."""
        if self.campaign_id:
            self.generate.setEnabled(False)
            self.report_browser.generate()

    def open_folder(self) -> None:
        if self.summary.directory:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.summary.directory))

    # replies

    def received(self, tag: str, result: Any) -> bool:
        if self.loader.received(tag, result):
            return True
        if tag == "results-campaigns":
            self._campaigns_received(result)
            return True
        if tag.startswith("report-generate:") and tag == self.report_browser.generation_tag:
            self.generate.setEnabled(bool(self.campaign_id))
            self.notify.emit("Report queued", "ok")
            return False  # the report browser refreshes its list too
        if tag.startswith("result-count:"):
            _, generation, outcome = tag.split(":", 2)
            if int(generation) == self.count_generation:
                self.counts[outcome] = int(result.get("total") or 0)
                self._show_counts()
            return True
        if tag.startswith("result-coverage:"):
            if tag == self.coverage_tag:
                self._coverage_received(result)
            return True
        if tag == "result-detail":
            if result.get("id") == self.result_selection:
                self.metrics = dict(result.get("metrics") or {})
                self.summary.set_values(self.metrics)
                directory = result.get("directory")
                if directory and not self.summary.directory:
                    self.summary.directory = str(directory)
                    self.summary.open_folder.setEnabled(True)
                self._update_chart()
            return True
        return False

    def failed(self, tag: str, error: str) -> bool:
        if self.loader.failed(tag, error):
            return True
        if tag.startswith("result-count:") or tag == "results-campaigns":
            return True  # the footer keeps "—"; the select keeps its campaigns
        if tag.startswith("report-preview-"):
            return True  # the preview card shows it
        if tag.startswith("report-generate:") and tag == self.report_browser.generation_tag:
            self.generate.setEnabled(bool(self.campaign_id))
            return False  # the window toasts the engine's reason
        if tag.startswith("result-coverage:"):
            if tag == self.coverage_tag:
                self.coverage_tag = ""
                self._update_chart()
            return True
        return False

    # layout

    def _tab(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for table in (self.results, self.reports):
            if table.isVisible():
                table.refresh()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.head.setGeometry(0, 0, width, 36)
        place(self.tabs, 0, 48, width, 39)
        self.stack.setGeometry(0, 103, width, max(0, height - 103))

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.load_campaigns()

    def close_timers(self) -> None:
        self.loader.close_timers()
        self.campaign_sync.stop()
