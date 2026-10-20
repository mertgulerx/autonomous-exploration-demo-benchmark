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

"""The Monitor page (HANDOFF §8.9–§8.13, screens 09–12).

The page head names the active workers' campaign and opens it on Campaigns. The worker strip has one card per
active worker (slot order): its tag, robot, attempt badge and experiment, then Coverage, Sim time and Distance,
or the readiness detail while it starts. The run card shows the selected run: the view switch (Map & metrics,
Logs, Events), the attempt's worker, number and start, and Stop run. Map & metrics is the live map (fed by
``monitor.get`` at 10 Hz, one request in flight, while the page is visible) with its legend line, and the run
measurements: coverage with its chart (the attempt's recorded samples from ``measurements.query``, read 250 at a
time and again every 5 s, plus the live value), first-reached times and two pairs of key-value lists. When the
run ends, its card keeps the last snapshot. Stop run asks in a dialog and sends ``attempt.stop`` with the fixed
request ID ``attempt-stop:<attempt>``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import time
from typing import Any, cast
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .contracts import average_speed
from .gui_cells import OverlayScrollBar
from .gui_controls import BadgeLook, StatusBadge, badge_look, paint_badge
from .gui_dialogs import Consequences, Run, bullet, wrap_text
from .gui_events import AttemptSlots, EventStream, EventView
from .gui_logs import LogView
from .gui_mapview import LiveMap, MapLegend
from .gui_maps import LiveLayers, WorldMap, apply_monitor_reply
from .gui_marks import (
    CoverageChart,
    CoverageSample,
    KeyValue,
    KeyValueList,
    paint_worker_tag,
)
from .gui_menus import show_tooltip
from .gui_style import cell_text, duration, fixed, humanize, number, worker_values
from .gui_support import Transport, stale_engine_message
from .gui_theme import ThemeManager, snap
from .gui_viewers import TITLES, Kind, RunViewers
from .gui_widgets import (
    BODY,
    CARD_TITLE,
    PAGE_TITLE,
    Button,
    Card,
    CssBox,
    FlexRow,
    LinkButton,
    ModalHost,
    SegmentedControl,
    TextLine,
    TextStyle,
    Themed,
    css_rect,
    draw_text,
    paint_glyphs,
    place,
    rounded_path,
)
from .monitoring import MONITOR_LAYERS, PathCursor

IDLE_TEXT = "No active workers. Start a queued campaign to begin monitoring."
CHOOSE_TEXT = "Select a worker above to monitor its run."
ENDED_STATUS = "Run ended: the last received path and pose stay on the map"
MILESTONES = (25, 50, 75, 90, 95)
COVERAGE_PAGE = 250
COVERAGE_REFRESH = 5.0  # s: the recording's checkpoint interval


def measured(label: str, value: Any, digits: int, unit: str) -> KeyValue:
    text = fixed(value, digits)
    return KeyValue(label, text or "—", unit if text is not None else "")


def run_average_speed(metrics: Mapping[str, Any]) -> float | None:
    """The run's average speed: recorded by the worker, else derived the same way (runs recorded before it)."""
    value = metrics.get("average_speed_mps")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    elapsed = metrics.get("simulation_elapsed") if metrics.get("motion_started", True) else None
    return average_speed(metrics.get("distance_m"), elapsed)


def readiness_text(detail: str) -> str:
    """The worker's readiness detail as the strip shows it ("Waiting for clock, scan, …")."""
    return detail.replace("Waiting for: ", "Waiting for ", 1)


def experiment_text(experiment: Mapping[str, Any]) -> str:
    return (
        f"{humanize(experiment['world_id'])} · {experiment['profile_name']} · "
        f"{experiment['spawn_name']} · repetition {experiment['repetition'] + 1}"
    )


def attempt_meta(attempt: Mapping[str, Any]) -> str:
    """The attempt facts of the run head ("Worker 1 · attempt 1 · started 16 Oct 2026 · 20:56"; the
    reservation time until the attempt starts)."""
    text = f"Worker {attempt['slot'] + 1} · attempt {attempt['number']}"
    if attempt.get("started_ns"):
        return text + " · started " + cell_text("started_ns", attempt["started_ns"])
    if attempt.get("reserved_ns"):
        return text + " · reserved " + cell_text("reserved_ns", attempt["reserved_ns"])
    return text


def health_badge(metrics: Mapping[str, Any]) -> tuple[BadgeLook | None, str]:
    """The timing and stream health badge and its tooltip (every flag)."""
    flags = [humanize(str(flag)) for flag in metrics.get("quality_flags") or []]
    if flags:
        word = flags[0] if len(flags) == 1 else f"{len(flags)} flags"
        return badge_look("health", "flags", word=word), "\n".join(flags)
    if metrics.get("rtf") is None:
        return badge_look("health", "pending", word="Pending"), ""
    return badge_look("health", "healthy", word="Healthy"), ""


def chart_end(seconds: float) -> float:
    """The chart's time axis: about 10 % after the last sample, on 5 s (45 s for a 41 s run, as 09)."""
    end = 5.0 * round(seconds * 1.1 / 5)
    return max(10.0, end if end > seconds else end + 5.0)


# ---------- worker strip ----------

WORKER_NAME = TextStyle(14, 600, 18)
WORKER_LABEL = TextStyle(11.5, 400, 14)
WORKER_VALUE = TextStyle(16, 600, 22, tabular=True)
WORKER_UNIT = TextStyle(12, 400, 22, tabular=True)
WAIT_TEXT = TextStyle(12, 400, 16)
VALUE_COLUMNS = (72.0, 64.0, 72.0)  # ``.m3``: 16 px apart, right-aligned
WAIT_WIDTH = 256.0


class WorkerCard(Themed, CssBox):
    """``.wk``: a 72 px card (padding 0 20, radius 8; the selected one with a 2 px ink edge): the worker tag,
    robot (14/600), attempt badge and the experiment in ``text.secondary``; at the right Coverage, Sim time
    and Distance (11.5 px labels over 16/600 values), or, while the worker starts, the hourglass and its
    readiness detail (12/16, two lines at most). Click, Enter or Space selects it."""

    chosen = QtCore.Signal(str)

    def __init__(self, themes: ThemeManager, identity: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(themes, parent)
        self.identity = identity
        self.run: dict[str, Any] = {}
        self.robot = ""
        self.look: BadgeLook | None = None
        self.values = worker_values({})
        self.waiting: str | None = None
        self.selected = False
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusRadius", 8)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def set_run(self, run: dict[str, Any], robot: str, waiting: str | None) -> None:
        self.run, self.robot, self.waiting = run, robot, waiting
        attempt = run["attempt"]
        self.look = (
            badge_look("campaign", "STOPPING", word="Stopping")
            if run.get("stop_requested")
            else badge_look("attempt", attempt["phase"])
        )
        self.values = worker_values(run.get("metrics") or {})
        experiment = run["experiment"]
        self.setAccessibleName(
            f"W{attempt['slot'] + 1} {robot}: {experiment_text(experiment)}"
            + (f"; {waiting}" if waiting else "")
        )
        self.update()

    def set_selected(self, selected: bool) -> None:
        if selected != self.selected:
            self.selected = selected
            self.update()

    def content(self) -> tuple[float, float]:
        box = css_rect(self)
        return box.x() + 21, box.width() - 42

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.chosen.emit(self.identity)
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
            QtCore.Qt.Key.Key_Space,
        ):
            self.chosen.emit(self.identity)
            event.accept()
            return
        super().keyPressEvent(event)

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            tips = []
            if self.run:
                tips.append(experiment_text(self.run["experiment"]))
            if self.waiting:
                tips.append(self.waiting)
            show_tooltip(self, event.globalPos(), "\n".join(tips))
            return True
        return super().event(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = self.theme
        outer = QtCore.QRectF(self.rect())
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(theme.color("surface.card"))
        painter.drawPath(rounded_path(outer, 8))
        if self.selected:  # 1 px ink border plus a 1 px ink inset ring
            painter.setPen(QtGui.QPen(theme.color("text.ink"), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(rounded_path(outer.adjusted(1, 1, -1, -1), 7))
        else:
            painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(rounded_path(outer.adjusted(0.5, 0.5, -0.5, -0.5), 7.5))
        if not self.run:
            return
        left, width = self.content()
        right = left + width
        side = WAIT_WIDTH if self.waiting else sum(VALUE_COLUMNS) + 32
        if self.waiting:
            self._paint_wait(painter, right - WAIT_WIDTH)
        else:
            self._paint_values(painter, right - side)
        self._paint_identity(painter, left, max(0.0, width - side - 20))

    def _paint_identity(self, painter: QtGui.QPainter, left: float, width: float) -> None:
        """``.id``: line 1 (22 px) at 14 px, line 2 four px below it."""
        theme = self.theme
        attempt, experiment = self.run["attempt"], self.run["experiment"]
        tag = f"W{attempt['slot'] + 1}"
        x = left + paint_worker_tag(painter, left, 14 + 1.5, tag, theme) + 8
        badge = self.look.width if self.look is not None else 0.0
        room = max(0.0, left + width - x - (8 + badge if badge else 0))
        name = WORKER_NAME.elide(self.robot, room)
        x += draw_text(painter, x, 16, name, WORKER_NAME, theme.color("text.ink"))
        if self.look is not None:
            paint_badge(painter, x + 8, 14, self.look, theme)
        text = BODY.elide(experiment_text(experiment), width)
        draw_text(painter, left, 40, text, BODY, theme.color("text.secondary"))

    def _paint_values(self, painter: QtGui.QPainter, left: float) -> None:
        theme = self.theme
        x = left
        for (label, value, unit), column in zip(self.values, VALUE_COLUMNS, strict=True):
            right = x + column
            draw_text(
                painter,
                right - WORKER_LABEL.advance(label),
                16,
                label,
                WORKER_LABEL,
                theme.color("text.secondary"),
            )
            unit_width = 2 + WORKER_UNIT.advance(unit) if unit else 0.0
            value_x = right - unit_width - WORKER_VALUE.advance(value)
            draw_text(painter, value_x, 33, value, WORKER_VALUE, theme.color("text.ink"))
            if unit:
                # The unit shares the value's baseline.
                painter.save()
                painter.setPen(theme.color("text.tertiary"))
                paint_glyphs(
                    painter, right - unit_width + 2, WORKER_VALUE.baseline(33), unit, WORKER_UNIT
                )
                painter.restore()
            x = right + 16

    def _paint_wait(self, painter: QtGui.QPainter, left: float) -> None:
        theme = self.theme
        lines = wrap_text(self.waiting or "", WAIT_TEXT, WAIT_WIDTH - 24, limit=2)
        top = (72 - 16 * len(lines)) / 2 + css_rect(self).y()
        gui_icons.draw(
            painter,
            "hourglass",
            theme.color("text.tertiary"),
            QtCore.QPointF(snap(left), snap(top + 1)),
        )
        for index, line in enumerate(lines):
            color = theme.color("text.secondary")
            draw_text(painter, left + 24, top + 16 * index, line, WAIT_TEXT, color)


class WorkerStrip(QtWidgets.QWidget):
    """``.workers``: the cards in slot order, up to four in a row of equal columns 16 px apart."""

    chosen = QtCore.Signal(str)
    COLUMNS = 4

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.themes = themes
        self.cards: dict[str, WorkerCard] = {}
        self.order: list[str] = []
        self.selected: str | None = None
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.setFixedHeight(0)

    def update_runs(
        self,
        runs: Sequence[dict[str, Any]],
        robots: Mapping[str, str],
        waiting: Mapping[str, str | None],
    ) -> None:
        ordered = sorted(runs, key=lambda run: int(run["attempt"]["slot"]))
        identities = [run["attempt"]["id"] for run in ordered]
        for identity in set(self.cards) - set(identities):
            self.cards.pop(identity).deleteLater()
        for run in ordered:
            identity = run["attempt"]["id"]
            card = self.cards.get(identity)
            if card is None:
                card = WorkerCard(self.themes, identity, self)
                card.chosen.connect(self.chosen)
                card.show()
                self.cards[identity] = card
            robot_id = run["experiment"]["robot_id"]
            card.set_run(run, robots.get(robot_id, robot_id), waiting.get(identity))
            card.set_selected(identity == self.selected)
        changed = identities != self.order
        self.order = identities
        if changed:
            rows = math.ceil(len(identities) / self.COLUMNS)
            self.setFixedHeight(rows * 72 + max(0, rows - 1) * 16)
            self.setVisible(bool(identities))
            self.relayout()

    def select(self, identity: str | None) -> None:
        self.selected = identity
        for key, card in self.cards.items():
            card.set_selected(key == identity)

    def relayout(self) -> None:
        count = len(self.order)
        if not count:
            return
        columns = min(count, self.COLUMNS)
        width = (self.width() - 16 * (columns - 1)) / columns
        for index, identity in enumerate(self.order):
            row, column = divmod(index, columns)
            place(self.cards[identity], column * (width + 16), row * 88, width, 72)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()


# ---------- measurements ----------

MEASURE_HEAD = TextStyle(12.5, 400, 18)
SECTION_TEXT = TextStyle(12, 600, 18)
SECTION_NOTE = TextStyle(12, 400, 18, tabular=True)
BIG_VALUE = TextStyle(30, 600, 34, tabular=True, spacing_em=-0.01)
BIG_UNIT = TextStyle(16, 500, 18)
AREA_VALUE = TextStyle(13, 600, 18, tabular=True)
AREA_TEXT = TextStyle(13, 400, 18, tabular=True)
REACHED_KEY = TextStyle(12.5, 400, 16)
REACHED_SMALL = TextStyle(11.5, 400, 14)
REACHED_VALUE = TextStyle(14, 600, 18, tabular=True)
REACHED_NONE = TextStyle(12.5, 400, 18)
KV_LABEL = TextStyle(12.5, 400, 18)
CONTENT_HEIGHT = 582  # 269 to 851 at 1440×900
MEASUREMENTS_WIDTH = 520  # ``.run-body``: ``grid-template-columns: 1fr 520px``


@dataclass(frozen=True, slots=True)
class Reached:
    percent: float
    reached: bool | None  # None: no milestone data yet
    seconds: float | None = None  # simulation time when reached (None: unavailable)


def reached_by_share(metrics: Mapping[str, Any]) -> list[Reached]:
    """The 50, 75, 90 and 95 % milestones of a run's metrics (``coverage_milestones``)."""
    milestones: dict[float, Mapping[str, Any]] = {}
    for entry in metrics.get("coverage_milestones") or []:
        share = number(entry.get("percent")) if isinstance(entry, Mapping) else None
        if share is not None:
            milestones[share] = entry
    reached = []
    for percent in MILESTONES:
        found = milestones.get(float(percent))
        if found is None:
            reached.append(Reached(percent, None))
        else:
            done = bool(found.get("reached"))
            seconds = number(found.get("simulation_elapsed")) if done else None
            reached.append(Reached(percent, done, seconds))
    return reached


def coverage_samples(
    rows: Sequence[Mapping[str, Any]], gap: bool
) -> tuple[list[CoverageSample], bool]:
    """Recorded coverage rows (``measurements.query``) as chart samples: valid samples after the first
    movement; a sample without a valid value starts a new segment (no gap is bridged). Returns the samples and
    whether the next one starts a segment."""
    samples = []
    for row in rows:
        elapsed = row.get("simulation_elapsed_ns")
        if not isinstance(elapsed, int):
            continue  # before the first movement: no exploration time yet
        percent = number(row.get("coverage_percent"))
        if not (row.get("valid") and row.get("status") == "valid") or percent is None:
            gap = True
            continue
        samples.append(CoverageSample(elapsed / 1e9, percent, gap))
        gap = False
    return samples, gap


def milestone_marks(
    metrics: Mapping[str, Any], samples: Sequence[CoverageSample]
) -> list[CoverageSample]:
    """The chart's milestone circles: on the sample that reached the share (its own value), else on the
    share."""
    marks = []
    for entry in metrics.get("coverage_milestones") or []:
        if not isinstance(entry, Mapping) or not entry.get("reached"):
            continue
        moment, share = number(entry.get("simulation_elapsed")), number(entry.get("percent"))
        if moment is None or share is None:
            continue
        value = next((s.percent for s in samples if abs(s.seconds - moment) < 1e-3), share)
        marks.append(CoverageSample(moment, value))
    return marks


class MeasurementsContent(Themed):
    """``.meas`` content (padding 0 20): the head with the timing and stream health badge; Coverage with the
    value and covered area, the chart and the first-reached times; Time & motion and Compute & timing as two
    32 px key-value lists each, 32 px apart."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.chart = CoverageChart(themes, self)
        self.lists = [
            KeyValueList(themes, row_height=32, label_style=KV_LABEL, last_open=True, parent=self)
            for _ in range(4)
        ]
        self.health: BadgeLook | None = None
        self.health_tip = ""
        self.coverage: str | None = None
        self.covered: str | None = None
        self.target: str | None = None
        self.status = ""
        self.reached = [Reached(p, None) for p in MILESTONES]
        self.ended = False
        self.setMinimumHeight(CONTENT_HEIGHT)

    def set_metrics(self, metrics: Mapping[str, Any], *, ended: bool) -> None:
        self.ended = ended
        self.health, self.health_tip = health_badge(metrics)
        self.coverage = fixed(metrics.get("coverage_percent"), 1)
        self.covered = fixed(metrics.get("coverage_covered_m2"), 1)
        self.target = fixed(metrics.get("coverage_target_m2"), 1)
        status = metrics.get("coverage_status")
        self.status = humanize(str(status)) if status and status != "valid" else ""
        self.reached = reached_by_share(metrics)
        turning = metrics.get("angular_speed")
        rows = (
            [
                KeyValue("Simulation elapsed", duration(metrics.get("simulation_elapsed"))),
                measured("Distance", metrics.get("distance_m"), 1, "m"),
                KeyValue("Real-time factor", fixed(metrics.get("rtf"), 3) or "—"),
            ],
            [
                measured("Average speed", run_average_speed(metrics), 2, "m/s"),
                measured("Speed", metrics.get("speed"), 2, "m/s"),
                measured("Turning", turning, 2, "rad/s"),
            ],
            [
                measured("Explorer CPU", metrics.get("cpu_percent"), 1, "%"),
                measured("Explorer CPU mean", metrics.get("cpu_average_percent"), 1, "%"),
                measured("Explorer CPU time", metrics.get("cpu_seconds"), 2, "s"),
            ],
            [
                measured("Explorer RAM", metrics.get("rss_mib"), 1, "MiB"),
                measured("Explorer RAM mean", metrics.get("rss_average_mib"), 1, "MiB"),
                measured("SLAM map age", metrics.get("coverage_map_age"), 1, "s"),
            ],
        )
        for widget, values in zip(self.lists, rows, strict=True):
            widget.set_rows(values)
        self.setToolTip(self.health_tip)
        self.update()

    def clear(self) -> None:
        self.set_metrics({}, ended=False)
        self.chart.set_data([])

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width = self.width() - 40
        half = (width - 32) / 2
        place(self.chart, 20, 130, width, 112)
        for index, widget in enumerate(self.lists):
            column, section = index % 2, index // 2
            place(widget, 20 + column * (half + 32), 341 + section * 145, half, 96)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        right = self.width() - 20.0
        rule = theme.color("border.line")
        secondary, ink = theme.color("text.secondary"), theme.color("text.ink")
        # Head: 52 px with a rule.
        draw_text(painter, 20, 15.5, "Run measurements", CARD_TITLE, ink)
        if self.health is not None:
            badge_x = right - self.health.width
            paint_badge(painter, badge_x, 14.5, self.health, theme)
            label = "Timing / streams"
            draw_text(
                painter,
                badge_x - 8 - MEASURE_HEAD.advance(label),
                16.5,
                label,
                MEASURE_HEAD,
                secondary,
            )
        painter.fillRect(QtCore.QRectF(20, 51, right - 20, 1), rule)
        # Coverage.
        self._section(painter, 64, "Coverage", self.status)
        baseline = BIG_VALUE.baseline(88)
        value = self.coverage or "—"
        painter.save()
        painter.setPen(ink)
        paint_glyphs(painter, 20, baseline, value, BIG_VALUE)
        if self.coverage is not None:
            painter.setPen(secondary)
            paint_glyphs(painter, 20 + BIG_VALUE.advance(value) + 8, baseline, "%", BIG_UNIT)
        if self.target is not None:
            rest = f" of {self.target} m²"
            strong = self.covered or "—"
            x = right - AREA_TEXT.advance(rest) - AREA_VALUE.advance(strong)
            painter.setPen(ink)
            paint_glyphs(painter, x, baseline, strong, AREA_VALUE)
            painter.setPen(secondary)
            paint_glyphs(painter, x + AREA_VALUE.advance(strong), baseline, rest, AREA_TEXT)
        painter.restore()
        self._paint_reached(painter, 246, right)
        for top, title in ((306, "Time & motion"), (451, "Compute & timing")):
            painter.fillRect(QtCore.QRectF(20, top, right - 20, 1), rule)
            self._section(painter, top + 13, title, "")

    def _section(self, painter: QtGui.QPainter, top: float, title: str, note: str) -> None:
        """``.msh``: 12 px 600 ``text.secondary`` (an optional 400 note at the right)."""
        color = self.color("text.secondary")
        draw_text(painter, 20, top, title, SECTION_TEXT, color)
        if note:
            x = self.width() - 20 - SECTION_NOTE.advance(note)
            draw_text(painter, x, top, note, SECTION_NOTE, color)

    def _paint_reached(self, painter: QtGui.QPainter, top: float, right: float) -> None:
        """A 46 px row with a top rule: one cell per milestone (25 to 95 %) with the simulation time it was
        first reached, rules between the cells."""
        theme = self.theme
        secondary = theme.color("text.secondary")
        painter.fillRect(QtCore.QRectF(20, top, right - 20, 1), theme.color("border.line"))
        width = (right - 20) / max(1, len(self.reached))
        cell_top = top + 1 + (45 - 32) / 2
        for index, reached in enumerate(self.reached):
            x = 20 + index * width
            if index:
                painter.fillRect(
                    QtCore.QRectF(snap(x), snap(cell_top), 1, snap(cell_top + 32) - snap(cell_top)),
                    theme.color("border.line"),
                )
            inset = 13 if index else 0
            draw_text(
                painter, x + inset, cell_top, f"{reached.percent:g} %", REACHED_SMALL, secondary
            )
            if reached.reached is False:
                text = "Not reached" if self.ended else "Not yet"
                style, token = REACHED_NONE, "text.tertiary"
            else:  # reached ("—" when its time is unavailable) or no data yet
                text, style, token = duration(reached.seconds), REACHED_VALUE, "text.ink"
            draw_text(painter, x + inset, cell_top + 14, text, style, theme.color(token))


class Measurements(QtWidgets.QScrollArea):
    """The measurements column: the content scrolls when the run card is shorter than it."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.content = MeasurementsContent(themes)
        self.setWidget(self.content)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().setAutoFillBackground(False)
        self.content.setAutoFillBackground(False)
        OverlayScrollBar(themes, self)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        # Below the natural height the content scrolls, with 20 px under its last row.
        height = self.viewport().height()
        self.content.setMinimumHeight(
            CONTENT_HEIGHT if height >= CONTENT_HEIGHT else CONTENT_HEIGHT + 20
        )
        super().resizeEvent(event)


# ---------- run card ----------

RUN_META = TextStyle(13, 400, 18, tabular=True)
VIEWER_TIPS: dict[Kind, str] = {
    "gazebo": (
        "Open the Gazebo GUI on the selected run's simulation; it closes when the run ends. Pausing, moving or "
        "deleting entities changes the run, and rendering adds host load: watch the real-time factor"
    ),
    "rviz": (
        "Open RViz on the selected run (the navigation view without Exploration Control, with the traveled "
        "path from when it opens; simulation time); it closes when the run ends. Rendering adds host load: "
        "watch the real-time factor"
    ),
}


class RunCard(Card):
    """``.card.run``: the head (52 px, rule: view switch, attempt facts, Launch RViz, Stop run) over the selected
    view.
    Without a run it shows only a centred message in ``text.secondary``."""

    def __init__(
        self,
        themes: ThemeManager,
        map_view: QtWidgets.QWidget,
        logs: QtWidgets.QWidget,
        events: QtWidgets.QWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.views = SegmentedControl(
            themes,
            [("Map & metrics", "map"), ("Logs", "terminal"), ("Events", "list")],
            icon_size=14,
            parent=self,
        )
        self.views.changed.connect(self.show_view)
        self.ended = StatusBadge(themes, None, self)
        self.meta = TextLine(themes, "", RUN_META, "text.secondary", self)
        self.gazebo = Button(themes, "Launch Gazebo", icon="aupright", parent=self)
        self.rviz = Button(themes, "Launch RViz", icon="aupright", parent=self)
        self.viewer_buttons: dict[Kind, Button] = {"gazebo": self.gazebo, "rviz": self.rviz}
        for kind, button in self.viewer_buttons.items():
            button.setToolTip(VIEWER_TIPS[kind])
            button.setEnabled(False)
        self.stop = Button(themes, "Stop run", icon="stop", variant="danger", parent=self)
        self.stop.setToolTip(
            "End the selected run; its data is kept and campaign reports exclude it"
        )
        self.stop.setEnabled(False)
        self.bodies = (map_view, logs, events)
        for body in self.bodies:
            body.setParent(self)
        self.message = IDLE_TEXT
        self.has_run = False
        self.show_view(0)
        self.set_run_shown(False)

    def set_run_shown(self, shown: bool) -> None:
        self.has_run = shown
        for widget in (self.views, self.meta, self.gazebo, self.rviz, self.stop):
            widget.setVisible(shown)
        self.ended.setVisible(shown and self.ended.look is not None)
        for index, body in enumerate(self.bodies):
            body.setVisible(shown and index == self.views.index())
        self.relayout()
        self.update()

    def set_message(self, text: str) -> None:
        if text != self.message:
            self.message = text
            self.update()

    def set_meta(self, text: str, ended: BadgeLook | None) -> None:
        self.meta.setText(text)
        self.ended.set_look(ended)
        self.ended.setVisible(self.has_run and ended is not None)
        self.relayout()

    def show_view(self, index: int) -> None:
        self.views.set_index(index)
        for position, body in enumerate(self.bodies):
            body.setVisible(self.has_run and position == index)

    def css_changed(self) -> None:
        self.relayout()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def relayout(self) -> None:
        box = css_rect(self)
        left, top, width = box.x() + 1, box.y() + 1, box.width() - 2
        # ``.run-head``: padding 0 12, gap 16; items centred in the 51 px above the rule.
        seg = self.views.natural_width()
        place(self.views, left + 12, top + (51 - 32) / 2, seg, 32)
        stop = self.stop.natural_width()
        place(self.stop, left + width - 12 - stop, top + (51 - 32) / 2, stop, 32)
        right = left + width - 12 - stop
        for button in (
            self.rviz,
            self.gazebo,
        ):  # right to left: Launch Gazebo, Launch RViz, Stop run
            natural = button.natural_width()
            right -= 8 + natural
            place(button, right, top + (51 - 32) / 2, natural, 32)
        actions = left + width - 12 - right
        x = left + 12 + seg + 16
        if self.ended.isVisible() and self.ended.look is not None:
            badge = self.ended.look.width
            place(self.ended, x, top + (51 - 22) / 2, badge, 22)
            x += badge + 12
        place(self.meta, x, top + (51 - 18) / 2, max(0.0, left + width - 12 - actions - 16 - x), 18)
        body_top = top + 52
        for body in self.bodies:
            body.setGeometry(
                round(left),
                round(body_top),
                round(width),
                max(0, round(box.bottom() - 1 - body_top)),
            )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        box = css_rect(self)
        if self.has_run:
            painter.fillRect(
                QtCore.QRectF(box.x() + 1, box.y() + 52, box.width() - 2, 1),
                self.color("border.line"),
            )
            return
        text = BODY.elide(self.message, max(0.0, box.width() - 48))
        x = box.x() + (box.width() - BODY.advance(text)) / 2
        y = box.y() + (box.height() - 18) / 2
        draw_text(painter, x, y, text, BODY, self.color("text.secondary"))


class MapPane(Themed):
    """``.run-body``: the map column (stage and legend, right rule) and the measurements column."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.map = LiveMap(themes, self)
        self.legend = MapLegend(themes, self)
        self.measurements = Measurements(themes, self)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        column = max(0, width - MEASUREMENTS_WIDTH)
        self.map.setGeometry(0, 0, max(0, column - 1), max(0, height - 40))
        self.legend.setGeometry(0, height - 40, max(0, column - 1), 40)
        self.measurements.setGeometry(column, 0, width - column, height)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        column = max(0, self.width() - MEASUREMENTS_WIDTH)
        painter.fillRect(QtCore.QRect(column - 1, 0, 1, self.height()), self.color("border.line"))


def confirm_stop_run(modal: ModalHost, themes: ThemeManager, context: str) -> bool:
    """``12``: "Stop this run?", the run, three consequences, the footnote; Cancel and Stop run."""
    card = Consequences(themes, "Stop this run?")
    card.add_context(context)
    stopped = Run("Stopped", TextStyle(13, 600, 18), "text.ink")
    card.add_bullets(
        [
            bullet("The run ends as ", stopped, "; its recorded data is kept for inspection."),
            bullet("It is excluded from campaign reports and statistics and is not run again."),
            bullet("The campaign continues with the next experiment."),
        ]
    )
    card.add_footnote(
        "If the run ends on its own before the stop takes effect, its own outcome is kept.",
        ruled=True,
    )
    card.add_button("Cancel", False, default=True)
    card.add_button("Stop run", True, variant="danger-solid", icon="stop")
    return bool(modal.exec_card(card, escape_result=False))


def show_stop_unavailable(modal: ModalHost, themes: ThemeManager, text: str) -> None:
    card = Consequences(themes, "Stop run unavailable")
    card.add_sentence([Run("The run continues. " + text, token="text.body")], 4)
    card.add_button("Close", False, default=True)
    modal.exec_card(card, escape_result=False)


# ---------- the page ----------


@dataclass(slots=True)
class CoverageHistory:
    """The selected attempt's recorded coverage samples, read in pages of 250 and again every 5 s."""

    attempt_id: str
    samples: list[CoverageSample] = field(default_factory=list)
    offset: int = 0
    pending: str = ""
    complete: bool = False
    read_at: float = 0.0
    gap: bool = False  # the next valid sample starts a new segment


class MonitorPage(QtWidgets.QWidget):
    """The Monitor (see the module docstring). The window forwards ``engine.status`` with ``update_status``
    and the transport replies with ``received`` and ``failed`` (False: not this page's)."""

    open_campaign = QtCore.Signal(str, str)
    notify = QtCore.Signal(str, str)  # text, kind

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes, self.transport = themes, transport
        self.status: dict[str, Any] = {}
        self.catalog: dict[str, Any] = {}
        self.active_page = False
        self.selected_attempt: str | None = None
        self.shown_run: dict[str, Any] | None = None  # the run the card shows (kept after it ends)
        self.monitor_id: str | None = None
        self.monitor_requested: str | None = None
        self.monitor_pending = False
        self.monitor_cursor = PathCursor()
        self.live = LiveLayers()
        self.layer_revisions: dict[str, int] = {}
        self.worlds: dict[str, WorldMap] = {}
        self.reference_mismatch = False
        self.stop_pending: set[str] = set()
        self.stop_requested_runs: set[str] = set()
        self.coverage: CoverageHistory | None = None
        self.campaign_names: dict[str, str] = {}
        self.tuning_campaigns: set[str] = set()  # tuning runs have no Open campaign link
        self.campaign_requests: set[str] = set()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        # Page head (``.page-head``): title, campaign, Open campaign.
        self.head = FlexRow(gap=12)
        self.head.setFixedHeight(36)
        self.head.add(TextLine(themes, "Monitor", PAGE_TITLE, "text.ink"))
        self.context = TextLine(themes, "", BODY, "text.secondary")
        self.head.add(self.context)
        self.head.add_stretch()
        self.open_link = LinkButton(themes, "Open campaign")
        self.open_link.clicked.connect(self._open_campaign)
        self.head.add(self.open_link)
        layout.addWidget(self.head)
        self.strip = WorkerStrip(themes)
        self.strip.chosen.connect(self.choose_worker)
        self.strip.hide()
        layout.addWidget(self.strip)
        self.map_pane = MapPane(themes)
        self.live_map = self.map_pane.map
        self.legend = self.map_pane.legend
        self.measurements = self.map_pane.measurements.content
        self.events = EventStream(transport, self)
        self.slots = AttemptSlots(transport, self)
        self.log_view = LogView(themes, transport)
        self.event_view = EventView(themes, self.events, self.slots)
        self.card = RunCard(themes, self.map_pane, self.log_view, self.event_view)
        self.card.stop.clicked.connect(self.stop_selected_run)
        self.viewers = RunViewers(self)
        self.viewers.done.connect(self._viewer_done)
        self.card.gazebo.clicked.connect(lambda: self.launch_viewer("gazebo"))
        self.card.rviz.clicked.connect(lambda: self.launch_viewer("rviz"))
        layout.addWidget(self.card, 1)
        self.events.changed.connect(self._refresh_strip)  # readiness details come from events
        self.monitor_timer = QtCore.QTimer(self)
        self.monitor_timer.setInterval(100)
        self.monitor_timer.timeout.connect(self.poll_monitor)
        self.feed_timer = QtCore.QTimer(self)
        self.feed_timer.setInterval(1000)
        self.feed_timer.timeout.connect(self.poll_feeds)
        self._update_head()

    # ---------- window hooks ----------

    def activate(self) -> None:
        """The page became current."""
        self.active_page = True
        self.feed_timer.start()
        self.follow_single_worker()
        self.inspect_selected()
        self.poll_feeds()

    def deactivate(self) -> None:
        self.active_page = False
        self.feed_timer.stop()
        self.close_monitor()

    def active_runs(self) -> list[dict[str, Any]]:
        return list(self.status.get("active", []))

    def run_of(self, identity: str | None) -> dict[str, Any] | None:
        return next((r for r in self.active_runs() if r["attempt"]["id"] == identity), None)

    def robot_footprint(self, robot_id: str) -> tuple[tuple[float, float], ...]:
        """The robot's catalog footprint (robot frame, metres); empty when the catalog has none."""
        robot: Mapping[str, Any] = next(
            (r for r in self.catalog.get("robots", []) if r["id"] == robot_id), {}
        )
        try:
            return tuple((float(x), float(y)) for x, y in robot.get("footprint") or [])
        except (TypeError, ValueError):
            return ()

    def robot_labels(self) -> dict[str, str]:
        return {r["id"]: r.get("label", r["id"]) for r in self.catalog.get("robots", [])}

    def update_status(self, status: dict[str, Any], catalog: dict[str, Any]) -> None:
        self.status, self.catalog = status, catalog
        active = self.active_runs()
        for row in (status.get("campaigns") or {}).get("rows") or []:
            if isinstance(row, dict) and row.get("id"):
                self.campaign_names[row["id"]] = row.get("name", "")
        self.stop_requested_runs |= {r["attempt"]["id"] for r in active if r.get("stop_requested")}
        live = {r["attempt"]["id"] for r in active}
        for attempt_id in self.viewers.attempts() - live:  # a run's viewers close when the run ends
            self.viewers.stop(attempt_id)
        if self.active_page:
            self.follow_single_worker()
        self._update_head()  # a new campaign resets the slots before the active runs' are added
        for run in active:
            self.slots.add(run["attempt"])
        self._refresh_strip()
        selected = self.run_of(self.selected_attempt)
        if selected is not None:
            self.shown_run = selected
            self._show_run(selected)
            if self.active_page:
                self.inspect_selected()
        elif self.monitor_id or (self.shown_run and self.selected_attempt):
            self._run_ended()
        self._update_head()
        self._update_card()

    def _refresh_strip(self) -> None:
        waiting: dict[str, str | None] = {}
        for run in self.active_runs():
            phase = run["attempt"]["phase"]
            if phase in ("STARTING", "READY"):
                state = self.events.latest_state(run["attempt"]["id"]) or {}
                detail = str(state.get("detail") or "")
                waiting[run["attempt"]["id"]] = (
                    readiness_text(detail) if detail else "Waiting for readiness"
                )
            else:
                waiting[run["attempt"]["id"]] = None
        self.strip.update_runs(self.active_runs(), self.robot_labels(), waiting)

    def _update_head(self) -> None:
        campaign = self.campaign_id()
        name = self.campaign_names.get(campaign, "") if campaign else ""
        if campaign and not name and campaign not in self.campaign_requests:
            self.campaign_requests.add(campaign)
            self.transport.submit(
                "monitor-campaign:" + campaign, "campaign.status", {"campaign_id": campaign}
            )
        self.context.setText(name)
        self.open_link.setVisible(campaign is not None and campaign not in self.tuning_campaigns)
        self.head.relayout()
        self.events.set_campaign(campaign)
        self.slots.set_campaign(campaign)

    def campaign_id(self) -> str | None:
        """The campaign of the run on the card, else the engine's current one."""
        if self.shown_run is not None:
            return str(self.shown_run["attempt"]["campaign_id"])
        current = self.status.get("current_campaign")
        return str(current) if current else None

    def _open_campaign(self) -> None:
        campaign = self.campaign_id()
        if campaign:
            self.open_campaign.emit(campaign, self.campaign_names.get(campaign, ""))

    def _update_card(self) -> None:
        if self.shown_run is None:
            self.card.set_message(IDLE_TEXT if not self.active_runs() else CHOOSE_TEXT)
            self.card.set_run_shown(False)
            return
        self.card.set_run_shown(True)
        self.update_stop_button()

    # ---------- selection ----------

    def follow_single_worker(self) -> None:
        active = self.active_runs()
        if len(active) == 1 and self.selected_attempt != active[0]["attempt"]["id"]:
            self.choose_worker(active[0]["attempt"]["id"], focus_map=False)

    def choose_worker(self, identity: str, *, focus_map: bool = True) -> None:
        active = self.run_of(identity)
        if active is None:
            return
        changed = identity != self.selected_attempt
        self.selected_attempt = identity
        self.shown_run = active
        self.strip.select(identity)
        if changed:
            self.measurements.clear()
            self.coverage = CoverageHistory(identity)
            self._read_coverage()
            self.event_view.set_attempt(identity)
        if self.log_view.attempt_id != identity:  # Results may have shown another attempt's log
            self.log_view.select_attempt(identity, active["attempt"]["directory"])
        self._show_run(active)
        if focus_map:
            self.card.show_view(0)
        self._update_head()
        self._update_card()
        self.inspect_selected()

    def _show_run(self, run: dict[str, Any]) -> None:
        self.card.set_meta(attempt_meta(run["attempt"]), None)
        self.measurements.set_metrics(run.get("metrics") or {}, ended=False)
        self._update_chart(run.get("metrics") or {})

    def _run_ended(self) -> None:
        """The run on the card left the engine's active list: keep its last snapshot."""
        stopped = self.monitor_id in self.stop_requested_runs or (
            self.shown_run is not None
            and self.shown_run["attempt"]["id"] in self.stop_requested_runs
        )
        self.close_monitor()
        self.selected_attempt = None
        self.strip.select(None)
        if self.shown_run is None:
            return
        word = "Run stopped" if stopped else "Run ended"
        self.card.set_meta(
            attempt_meta(self.shown_run["attempt"]), BadgeLook(word, "neutral", None)
        )
        self.measurements.set_metrics(self.shown_run.get("metrics") or {}, ended=True)
        self.set_legend_status(ENDED_STATUS, False)

    def selected_run(self) -> dict[str, Any] | None:
        return self.run_of(self.selected_attempt)

    def update_stop_button(self) -> None:
        active = self.selected_run()
        live = active is not None and active["attempt"]["phase"] in ("STARTING", "READY", "RUNNING")
        self.card.stop.setEnabled(
            active is not None
            and live
            and not active.get("stop_requested")
            and not active["attempt"].get("discard_intent")
            and active["attempt"]["id"] not in self.stop_pending
        )
        for kind, button in self.card.viewer_buttons.items():
            button.setEnabled(
                active is not None
                and live
                and (kind, active["attempt"]["id"]) not in self.viewers.pending
            )

    def launch_viewer(self, kind: Kind) -> None:
        active = self.selected_run()
        if active is None:
            return
        robot = str(active["experiment"]["robot_id"])
        title = "Run · " + self.stop_context(active) + " - " + TITLES[kind]
        self.viewers.launch(kind, active["attempt"], robot, title)
        self.update_stop_button()
        self.notify.emit(f"Starting {TITLES[kind]}…", "neutral")

    def _viewer_done(self, kind: str, attempt_id: str, error: str) -> None:
        self.viewers.finished(cast(Kind, kind), attempt_id, error)
        if self.run_of(attempt_id) is None:
            self.viewers.stop(attempt_id)  # the run ended while the viewer started
        self.update_stop_button()
        if error:
            self.notify.emit(error, "bad")

    def stop_context(self, active: Mapping[str, Any]) -> str:
        experiment, attempt = active["experiment"], active["attempt"]
        robot = self.robot_labels().get(experiment["robot_id"], experiment["robot_id"])
        return f"{robot} · {experiment_text(experiment)} · attempt {attempt['number']}"

    def stop_selected_run(self) -> None:
        active = self.selected_run()
        modal = self.window().findChild(ModalHost)
        if active is None or modal is None:
            return
        identity = active["attempt"]["id"]
        self.card.stop.set_opened(True)
        try:
            confirmed = confirm_stop_run(modal, self.themes, self.stop_context(active))
        finally:
            self.card.stop.set_opened(False)
        current = self.selected_run()
        if not confirmed or current is None or current["attempt"]["id"] != identity:
            return  # cancelled, or the run ended or another was chosen while the dialog was open
        self.stop_pending.add(identity)
        self.update_stop_button()
        # A fixed request ID makes a repeated click return the stored answer.
        self.transport.submit(
            "stop-run:" + identity,
            "attempt.stop",
            {"attempt_id": identity},
            request_id="attempt-stop:" + identity,
        )

    # ---------- live map ----------

    def inspect_selected(self) -> None:
        if not self.selected_attempt or self.monitor_id == self.selected_attempt:
            return
        active = self.selected_run()
        if active is None:
            return
        # A request still in flight for the previous attempt clears monitor_pending on arrival and its
        # reply is ignored, so at most one request is outstanding.
        self.monitor_id = self.selected_attempt
        self.monitor_cursor = PathCursor()
        self.live = LiveLayers(footprint=self.robot_footprint(active["experiment"]["robot_id"]))
        self.layer_revisions = {}
        self.reference_mismatch = False
        self.live_map.set_layers(self.live)
        world = active["experiment"]["world_id"]
        self.live_map.set_world(self.worlds.get(world))
        if world not in self.worlds:
            self.live_map.set_message("Loading map…")
            self.transport.submit("monitor-map:" + world, "spawns.map", {"world_id": world})
        self.set_legend_status("Waiting for the worker", False)
        self.monitor_timer.start()

    def close_monitor(self) -> None:
        self.monitor_id = None
        self.monitor_timer.stop()

    def poll_monitor(self) -> None:
        if not self.monitor_id or self.monitor_pending or not self.active_page:
            return
        layers = [n for n in MONITOR_LAYERS if not (n == "explored" and self.reference_mismatch)]
        self.monitor_pending = True
        self.monitor_requested = self.monitor_id
        self.transport.submit(
            "monitor",
            "monitor.get",
            {
                "attempt_id": self.monitor_id,
                "after_sequence": self.monitor_cursor.applied,
                "layers": layers,
                "revisions": {
                    name: self.layer_revisions[name]
                    for name in layers
                    if name in self.layer_revisions
                },
            },
        )

    def set_legend_status(self, text: str, live: bool) -> None:
        self.legend.set_status(text, live)

    def apply_monitor(self, result: dict[str, Any]) -> None:
        if "history" not in result:
            self.set_legend_status(result.get("reason") or "Waiting for the worker", False)
            return
        try:
            facts = apply_monitor_reply(
                self.live, self.monitor_cursor, self.layer_revisions, result, self.live_map.world
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.set_legend_status(f"Monitor reply rejected: {exc}", False)
            return
        self.reference_mismatch = facts.mismatch
        self.live_map.update()
        live = facts.status == "available" and facts.pose_age is not None
        if live:
            text = (
                f"{self.monitor_cursor.applied:,} path vertices · pose {facts.pose_age:.1f} s old"
            )
        elif facts.status == "stale":
            text = f"Ground-truth pose stale: {facts.reason}"
        else:
            text = f"Ground-truth pose unavailable: {facts.reason}"
        counted = {key: count for key, count in facts.breaks.items() if count > 0}
        if counted:
            listed = ", ".join(f"{key} {count}" for key, count in counted.items())
            text += f" · {sum(counted.values())} breaks ({listed})"
        dropped = self.monitor_cursor.dropped + self.live.path.dropped
        if dropped > 0:
            text += f" · {dropped:,} older vertices dropped"
        if facts.collisions > 0:
            text += f" · {facts.collisions} samples in obstacle cells"
        if facts.outside > 0:
            text += f" · {facts.outside} samples outside the reference region"
        if facts.mismatch:
            text += " · World map differs from this attempt's reference; explored cells hidden"
        self.set_legend_status(" · ".join([text, *facts.notes]), live)

    # ---------- coverage history ----------

    def _read_coverage(self) -> None:
        history = self.coverage
        if history is None or history.pending:
            return
        history.pending = f"monitor-coverage:{uuid.uuid4().hex}"
        self.transport.submit(
            history.pending,
            "measurements.query",
            {
                "attempt_id": history.attempt_id,
                "kind": "coverage",
                "offset": history.offset,
                "limit": COVERAGE_PAGE,
            },
        )

    def _coverage_received(self, result: dict[str, Any]) -> None:
        history = self.coverage
        assert history is not None
        history.pending = ""
        rows = result.get("rows") or []
        samples, history.gap = coverage_samples(rows, history.gap)
        history.samples.extend(samples)
        history.offset += len(rows)
        if rows and result.get("next_offset") is not None:
            history.complete = False
            self._read_coverage()  # the next page
        else:
            history.complete = True
            history.read_at = time.monotonic()
        run = self.shown_run
        self._update_chart((run or {}).get("metrics") or {})

    def _update_chart(self, metrics: Mapping[str, Any]) -> None:
        history = self.coverage
        if history is None or self.shown_run is None:
            self.measurements.chart.set_data([])
            return
        samples = list(history.samples)
        # The live value continues the recorded samples once all of them are here (no gap is bridged).
        seconds = number(metrics.get("simulation_elapsed"))
        percent = number(metrics.get("coverage_percent"))
        if (
            history.complete
            and seconds is not None
            and percent is not None
            and metrics.get("coverage_status", "valid") == "valid"
            and (not samples or seconds > samples[-1].seconds)
        ):
            samples.append(CoverageSample(seconds, percent, history.gap))
        milestones = milestone_marks(metrics, samples)
        end = chart_end(samples[-1].seconds) if samples else None
        self.measurements.chart.set_data(samples, milestones, x_end=end)

    # ---------- feeds ----------

    def poll_feeds(self) -> None:
        """Every second on this page: new events, the log while followed, recorded coverage every 5 s."""
        if not self.active_page:
            return
        self.events.poll()
        self.log_view.poll()
        history = self.coverage
        if (
            history is not None
            and history.complete
            and not history.pending
            and self.selected_attempt == history.attempt_id
            and time.monotonic() - history.read_at >= COVERAGE_REFRESH
        ):
            self._read_coverage()

    # ---------- replies ----------

    def received(self, tag: str, result: Any) -> bool:
        if (
            self.events.received(tag, result)
            or self.slots.received(tag, result)
            or self.log_view.received(tag, result)
        ):
            return True
        if tag == "monitor":
            self.monitor_pending = False
            if self.monitor_id and result.get("attempt_id") == self.monitor_id:
                self.apply_monitor(result)
            return True
        if tag.startswith("monitor-map:"):
            world_id = tag.removeprefix("monitor-map:")
            try:
                world = WorldMap.from_reply(result)
            except (KeyError, TypeError, ValueError) as exc:
                self._world_failed(world_id, f"World map unreadable: {exc}")
                return True
            self.worlds[world_id] = world
            run = self.selected_run()
            if run is not None and run["experiment"]["world_id"] == world_id:
                self.live_map.set_world(world)
            return True
        if tag.startswith("monitor-coverage:"):
            if self.coverage is not None and tag == self.coverage.pending:
                self._coverage_received(result)
            return True
        if tag.startswith("monitor-campaign:"):
            campaign = result.get("campaign") if isinstance(result, dict) else None
            if isinstance(campaign, dict) and campaign.get("id"):
                self.campaign_names[campaign["id"]] = campaign.get("name", "")
                specification = campaign.get("specification")
                if isinstance(specification, dict) and specification.get("tuning") is True:
                    self.tuning_campaigns.add(campaign["id"])  # not on Campaigns or Results
                self._update_head()
            return True
        if tag.startswith("stop-run:"):
            self.stop_pending.discard(tag.removeprefix("stop-run:"))
            self.notify.emit(
                "Stop already requested; the run is finalizing"
                if result.get("already_requested")
                else "Stop requested; the run is finalizing",
                "neutral",
            )
            return True
        return False

    def _world_failed(self, world_id: str, message: str) -> None:
        run = self.selected_run()
        if run is not None and run["experiment"]["world_id"] == world_id:
            self.live_map.set_message(message)
            self.set_legend_status(message, False)

    def failed(self, tag: str, error: str) -> bool:
        if (
            self.events.failed(tag, error)
            or self.slots.failed(tag, error)
            or self.log_view.failed(tag, error)  # shown in the Logs footer
        ):
            return True
        if tag == "monitor":
            # The Monitor reports its own errors; polling at 10 Hz must not flood the toasts.
            self.monitor_pending = False
            stale = stale_engine_message(
                error, "Unsupported operation: monitor.get", "the ground-truth Monitor"
            )
            if stale:
                self.set_legend_status(stale, False)
                self.monitor_timer.stop()  # until another run is chosen
            elif self.monitor_id and self.monitor_requested == self.monitor_id:
                self.set_legend_status(f"Monitor unavailable: {error}", False)
            return True
        if tag.startswith("monitor-map:"):
            self._world_failed(tag.removeprefix("monitor-map:"), f"World map unavailable: {error}")
            return True
        if tag.startswith("monitor-coverage:"):
            if self.coverage is not None and tag == self.coverage.pending:
                self.coverage.pending = ""
                self.coverage.complete = True
                self.coverage.read_at = time.monotonic()
            return True
        if tag.startswith("monitor-campaign:"):
            return True
        if tag.startswith("stop-run:"):
            self.stop_pending.discard(tag.removeprefix("stop-run:"))
            self.update_stop_button()
            stale = stale_engine_message(error, "Unsupported operation: attempt.stop", "Stop run")
            if stale:
                # A dialog, not a toast: the run keeps going.
                modal = self.window().findChild(ModalHost)
                if modal is not None:
                    show_stop_unavailable(modal, self.themes, stale)
                return True
            self.notify.emit(error, "bad")
            return True
        return False
