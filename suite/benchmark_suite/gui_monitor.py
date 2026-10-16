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

from typing import Any
from PySide6 import QtCore, QtGui, QtWidgets

from .gui_style import Badge, duration, humanize, label


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
