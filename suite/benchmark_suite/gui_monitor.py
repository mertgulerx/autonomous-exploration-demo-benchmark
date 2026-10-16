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

import base64
from collections import deque
import math
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .gui_style import COLORS, Badge, duration, humanize, label
from .gui_support import MapView
from .monitoring import IN_COLLISION, SEGMENT_START


class LiveMap(MapView):
    CHUNK = 4096
    MAXIMUM_VERTICES = 200_000

    def __init__(self) -> None:
        super().__init__()
        self.image.setZValue(-30)
        self.explored = pg.ImageItem(axisOrder="row-major")
        self.addItem(self.explored)
        self.explored.setZValue(-20)
        self.path_pen = pg.mkPen(COLORS["accent.default"], width=2)
        self.tail = self.plot(pen=self.path_pen, connect="finite")
        self.disagreements = self.plot(
            pen=None, symbol="x", symbolPen=COLORS["status.error"], symbolSize=9
        )
        self.goal = self.plot(pen=None, symbol="x", symbolPen=COLORS["status.info"], symbolSize=12)
        self.plan = self.plot(
            pen=pg.mkPen(COLORS["text.body"], width=1, style=QtCore.Qt.PenStyle.DashLine)
        )
        for item in (self.tail, self.disagreements):
            item.setZValue(1)
        for item in (self.goal, self.plan):
            item.setZValue(2)
        self.footprint.setZValue(3)
        self.arrow.setZValue(3)
        self.reference_hash: str | None = None
        self.reference_world: str | None = None
        # Frozen items retain only their real vertex count; only the current arrays are mutable.
        self.chunks: deque[tuple[Any, int]] = deque()
        self.current: Any = None
        self.current_x: list[float] = []
        self.current_y: list[float] = []
        self.current_real = 0
        self.real_vertices = 0
        self.view_dropped = 0
        self.markers: deque[tuple[float, float]] = deque(maxlen=2000)
        self.hide_pose()

    def set_reference(self, result: dict[str, Any]) -> None:
        self.set_map(result)
        self.reference_hash = result.get("metadata_sha256")
        self.reference_world = result["world_id"]

    @staticmethod
    def place(item: Any, origin: list[float], resolution: float) -> None:
        x, y, yaw = origin
        c, s = math.cos(yaw) * resolution, math.sin(yaw) * resolution
        item.setTransform(QtGui.QTransform(c, s, -s, c, x, y))

    @staticmethod
    def pixels(data: dict[str, Any]) -> np.ndarray[Any, Any]:
        image = QtGui.QImage.fromData(base64.b64decode(data["image_png"]))
        if image.isNull():
            raise ValueError("Unreadable monitor layer image")
        image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        array = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
            image.height(), image.bytesPerLine()
        )
        return (
            array[:, : image.width() * 4].reshape(image.height(), image.width(), 4)[:, :, 0].copy()
        )

    def set_explored(self, data: dict[str, Any] | None) -> None:
        if data is None:
            self.explored.clear()
            return
        pixels = self.pixels(data)
        rgba = np.zeros((*pixels.shape, 4), dtype=np.uint8)
        color = QtGui.QColor(COLORS["status.info"])
        rgba[:, :, :3] = (color.red(), color.green(), color.blue())
        rgba[:, :, 3] = np.where(pixels == 255, 70, 0)
        self.explored.setImage(rgba, autoLevels=False)
        self.place(self.explored, data["origin"], data["resolution"])

    def set_navigation(self, data: dict[str, Any] | None) -> None:
        goal = data.get("goal") if data is not None else None
        plan = data.get("plan", []) if data is not None else []
        self.goal.setData([goal[0]] if goal else [], [goal[1]] if goal else [])
        self.plan.setData([point[0] for point in plan], [point[1] for point in plan])

    def _new_chunk(self, seed: tuple[float, float] | None) -> None:
        if self.current is not None:
            self.current.setData(self.current_x, self.current_y, connect="finite")
            self.chunks.append((self.current, self.current_real))
        self.current = self.plot(pen=self.path_pen, connect="finite")
        self.current.setZValue(0)
        self.current_x, self.current_y = [], []
        self.current_real = 0
        if seed is not None:
            # Repeats the previous chunk's last vertex to keep the line continuous; not counted.
            self.current_x.append(seed[0])
            self.current_y.append(seed[1])

    def append_vertices(self, vertices: list[tuple[float, float, int]]) -> None:
        if not vertices:
            return
        marked = False
        for x, y, flags in vertices:
            start = bool(flags & SEGMENT_START)
            if self.current is None or len(self.current_x) == self.CHUNK:
                seed = (
                    (self.current_x[-1], self.current_y[-1])
                    if self.current_x and not start
                    else None
                )
                self._new_chunk(seed)
            if start and self.current_x:
                self.current_x.append(math.nan)
                self.current_y.append(math.nan)
                if len(self.current_x) == self.CHUNK:
                    self._new_chunk(None)
            self.current_x.append(x)
            self.current_y.append(y)
            self.current_real += 1
            self.real_vertices += 1
            if flags & IN_COLLISION:
                self.markers.append((x, y))
                marked = True
            while self.real_vertices > self.MAXIMUM_VERTICES and self.chunks:
                item, count = self.chunks.popleft()
                self.removeItem(item)
                self.real_vertices -= count
                self.view_dropped += count
        self.current.setData(self.current_x, self.current_y, connect="finite")
        if marked:
            self.disagreements.setData(
                [point[0] for point in self.markers], [point[1] for point in self.markers]
            )

    def set_tail(self, point: list[float] | None) -> None:
        if point is None or not self.current_x or not math.isfinite(self.current_x[-1]):
            self.tail.setData([], [])
        else:
            self.tail.setData([self.current_x[-1], point[0]], [self.current_y[-1], point[1]])

    def clear_path(self) -> None:
        for item, _ in self.chunks:
            self.removeItem(item)
        self.chunks.clear()
        if self.current is not None:
            self.removeItem(self.current)
        self.current = None
        self.current_x, self.current_y = [], []
        self.current_real = self.real_vertices = self.view_dropped = 0
        self.markers.clear()
        self.tail.setData([], [])
        self.disagreements.setData([], [])

    def clear_view(self) -> None:
        super().clear_view()
        self.clear_path()
        self.set_explored(None)
        self.set_navigation(None)
        self.reference_hash = self.reference_world = None

    def set_footprint(self, vertices: list[list[float]]) -> None:
        self.vertices = vertices
        self.hide_pose()


def number(value: Any, unit: str = "", digits: int = 1) -> str:
    return f"{value:.{digits}f}{unit}" if isinstance(value, (float, int)) else "—"


class WorkerCard(QtWidgets.QFrame):
    chosen = QtCore.Signal(str)

    def __init__(self, identity: str):
        super().__init__()
        self.identity = identity
        self.setProperty("worker", True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        head = QtWidgets.QHBoxLayout()
        self.title = label("", "section")
        self.title.setWordWrap(True)
        self.phase = Badge()
        head.addWidget(self.title, 1)
        head.addWidget(self.phase)
        layout.addLayout(head)
        self.context = label("", "muted")
        self.context.setWordWrap(True)
        layout.addWidget(self.context)
        metrics = QtWidgets.QGridLayout()
        self.coverage, self.elapsed, self.distance = (label("—") for _ in range(3))
        for column, title, value in (
            (0, "Coverage", self.coverage),
            (1, "Elapsed · sim", self.elapsed),
            (2, "Distance", self.distance),
        ):
            value.setWordWrap(True)
            metrics.addWidget(label(title, "muted"), 0, column)
            metrics.addWidget(value, 1, column)
        layout.addLayout(metrics)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )
        for text in self.findChildren(QtWidgets.QLabel):
            text.installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            isinstance(watched, QtWidgets.QLabel)
            and isinstance(event, QtGui.QMouseEvent)
            and event.type() == QtCore.QEvent.Type.MouseButtonRelease
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and not watched.hasSelectedText()
        ):
            self.chosen.emit(self.identity)
        return super().eventFilter(watched, event)

    def update_run(self, active: dict[str, Any], catalog: dict[str, Any]) -> None:
        experiment, metrics, attempt = active["experiment"], active["metrics"], active["attempt"]
        robot = next(
            (r["label"] for r in catalog.get("robots", []) if r["id"] == experiment["robot_id"]),
            experiment["robot_id"],
        )
        self.title.setText(robot)
        self.context.setText(
            f"{humanize(experiment['world_id'])} · {experiment['profile_name']}\n{experiment['spawn_name']} · repetition {experiment['repetition'] + 1} · attempt {attempt['number']}"
        )
        self.phase.set_state(
            humanize(attempt["phase"]), "active" if attempt["phase"] == "RUNNING" else "neutral"
        )
        coverage = number(metrics.get("coverage_percent", metrics.get("coverage")), "%")
        status = metrics.get("coverage_status", "valid")
        self.coverage.setText(coverage + (" · " + humanize(status) if status != "valid" else ""))
        self.elapsed.setText(duration(metrics.get("simulation_elapsed")))
        self.distance.setText(number(metrics.get("distance_m"), " m"))
        self.setAccessibleName(robot + ": " + self.context.text())

    def set_selected(self, selected: bool) -> None:
        if self.property("selected") != selected:
            self.setProperty("selected", selected)
            self.style().unpolish(self)
            self.style().polish(self)

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
        else:
            super().keyPressEvent(event)


class WorkerCards(QtWidgets.QWidget):
    chosen = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.cards: dict[str, WorkerCard] = {}
        self.selected: str | None = None
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(12)
        self.empty = label(
            "No active workers. Start a queued campaign to begin monitoring.", "muted"
        )
        self.grid.addWidget(self.empty, 0, 0)
        self.columns = 0

    def update_runs(self, active: list[dict[str, Any]], catalog: dict[str, Any]) -> None:
        identities = {a["attempt"]["id"] for a in active}
        changed = identities != set(self.cards)
        for identity in set(self.cards) - identities:
            card = self.cards.pop(identity)
            self.grid.removeWidget(card)
            card.deleteLater()
        for run in active:
            identity = run["attempt"]["id"]
            if identity not in self.cards:
                card = WorkerCard(identity)
                card.chosen.connect(self.chosen)
                self.cards[identity] = card
            self.cards[identity].update_run(run, catalog)
            self.cards[identity].set_selected(identity == self.selected)
        self.empty.setVisible(not active)
        if changed:
            self.reflow()

    def select(self, identity: str | None) -> None:
        self.selected = identity
        for key, card in self.cards.items():
            card.set_selected(key == identity)

    def reflow(self) -> None:
        self.columns = 4 if self.width() >= 960 else 2 if self.width() >= 580 else 1
        for i, card in enumerate(self.cards.values()):
            self.grid.addWidget(card, i // self.columns, i % self.columns)
        for column in range(4):
            self.grid.setColumnStretch(
                column, 1 if column < min(self.columns, len(self.cards)) else 0
            )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        columns = 4 if event.size().width() >= 960 else 2 if event.size().width() >= 580 else 1
        if columns != self.columns:
            self.reflow()
        super().resizeEvent(event)


class RunMetrics(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.grid = QtWidgets.QFormLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setVerticalSpacing(12)
        self.values: dict[str, QtWidgets.QLabel] = {}
        for key, title in (
            ("coverage", "Coverage"),
            ("milestones", "First reached · sim"),
            ("simulation", "Simulation elapsed"),
            ("wall", "Wall elapsed"),
            ("distance", "Distance"),
            ("speed", "Speed"),
            ("turn", "Turning"),
            ("cpu", "Explorer CPU"),
            ("ram", "Explorer RAM"),
            ("rtf", "RTF"),
            ("health", "Timing / stream health"),
        ):
            value = label("—")
            value.setWordWrap(True)
            self.values[key] = value
            self.grid.addRow(label(title, "muted"), value)

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        values = {
            "coverage": number(metrics.get("coverage_percent", metrics.get("coverage")), "%"),
            "simulation": duration(metrics.get("simulation_elapsed")),
            "wall": duration(metrics.get("wall_elapsed")),
            "distance": number(metrics.get("distance_m"), " m"),
            "speed": number(metrics.get("speed"), " m/s", 2),
            "turn": number(metrics.get("angular_speed"), " rad/s", 2),
            "cpu": number(metrics.get("cpu_percent"), "%"),
            "ram": number(metrics.get("rss_mib"), " MiB"),
            "rtf": number(metrics.get("rtf"), digits=3),
            "health": ", ".join(humanize(s) for s in metrics.get("quality_flags", []))
            or ("Pending" if metrics.get("rtf") is None else "Healthy"),
            "milestones": "\n".join(
                f"{entry['percent']:g}%  {duration(entry.get('simulation_elapsed')) if entry.get('reached') else 'Not reached'}"
                for entry in metrics.get("coverage_milestones", [])
            )
            or "Pending",
        }
        for key, text in values.items():
            self.values[key].setText(text)
