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
from copy import deepcopy
import math
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from .gui_style import COLORS, label
from .gui_support import FloatEdit, Transport, button
from .contracts import DEFAULT_LIMIT


def spawn_key(world: str, reference: Any) -> str:
    if isinstance(reference, dict):
        return str(reference["id"])
    return "default-" + world if reference in ("Default", "default") else str(reference)


def pose_text(pose: dict[str, Any]) -> str:
    return f"({pose['x']:.2f}, {pose['y']:.2f}) m · {math.degrees(pose['yaw']):.1f}°"


class _MapPreview(QtWidgets.QWidget):
    clicked = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(200)
        self.setMaximumHeight(260)
        self.pixmap = QtGui.QPixmap()
        self.metadata: dict[str, Any] = {}
        self.poses: list[dict[str, Any]] = []
        self.message = "Loading map…"
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def set_map(self, result: dict[str, Any]) -> None:
        image = QtGui.QImage.fromData(base64.b64decode(result["image_png"]))
        if image.isNull():
            raise ValueError("Unreadable world map")
        image = image.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
        image = image.convertToFormat(QtGui.QImage.Format.Format_Indexed8)
        paper = QtGui.QColor(COLORS["surface.raised"])
        image.setColorTable(
            [
                QtGui.QColor(
                    round(paper.red() * i / 255),
                    round(paper.green() * i / 255),
                    round(paper.blue() * i / 255),
                ).rgb()
                for i in range(256)
            ]
        )
        self.pixmap = QtGui.QPixmap.fromImage(image.transformed(QtGui.QTransform().scale(1, -1)))
        self.metadata = result
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        if self.pixmap.isNull():
            painter.setPen(QtGui.QColor(COLORS["text.muted"]))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self.message)
            return
        size = self.pixmap.size().scaled(
            self.size() - QtCore.QSize(16, 16), QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        target = QtCore.QRectF(
            (self.width() - size.width()) / 2,
            (self.height() - size.height()) / 2,
            size.width(),
            size.height(),
        )
        painter.drawPixmap(target, self.pixmap, QtCore.QRectF(self.pixmap.rect()))
        x, y, yaw = self.metadata["origin"]
        resolution = self.metadata["resolution"]
        c, s = math.cos(yaw), math.sin(yaw)
        for pose in self.poses:
            dx, dy = pose["x"] - x, pose["y"] - y
            u, v = (c * dx + s * dy) / resolution, (-s * dx + c * dy) / resolution
            point = QtCore.QPointF(
                target.x() + u / self.pixmap.width() * target.width(),
                target.bottom() - v / self.pixmap.height() * target.height(),
            )
            if not target.contains(point):
                continue
            painter.save()
            painter.translate(point)
            painter.rotate(-math.degrees(pose["yaw"] - yaw))
            painter.setPen(QtGui.QPen(QtGui.QColor(COLORS["surface.raised"]), 2))
            painter.setBrush(QtGui.QColor(COLORS["accent.default"]))
            painter.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(9, 0),
                        QtCore.QPointF(-6, -6),
                        QtCore.QPointF(-3, 0),
                        QtCore.QPointF(-6, 6),
                    ]
                )
            )
            painter.restore()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class _SpawnPopup(QtWidgets.QFrame):
    changed = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget, world: dict[str, Any]) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Popup)
        self.world_id = str(world["id"])
        self.references: list[Any] = ["Default"]
        self.setObjectName("worldSpawnsPopup")
        self.resize(470, 380)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(label("Spawns · " + world["label"], "section"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search starting positions")
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Starting position", "Position · heading"])
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(40)
        layout.addWidget(self.table, 1)
        self.empty = label("No matching starting positions", "muted")
        layout.addWidget(self.empty)
        self.more = label(
            "First 250 revisions shown. The Worlds & Spawns page lists all saved spawns.", "muted"
        )
        self.more.setWordWrap(True)
        layout.addWidget(self.more)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch()
        self.done = button("Done", self.hide, footer, role="primary")
        layout.addLayout(footer)
        self.search.textChanged.connect(self.filter_rows)
        self.table.itemChanged.connect(self.toggle)

    def refresh(
        self, entries: list[tuple[Any, str, str]], selected: list[Any], has_more: bool
    ) -> None:
        self.references = deepcopy(selected) or ["Default"]
        keys = {spawn_key(self.world_id, ref) for ref in self.references}
        current = self.table.currentItem()
        current_entry = self.table.item(current.row(), 0) if current else None
        current_key = (
            spawn_key(self.world_id, current_entry.data(QtCore.Qt.ItemDataRole.UserRole))
            if current_entry
            else None
        )
        blocker = QtCore.QSignalBlocker(self.table)
        self.table.setRowCount(len(entries))
        for row, (reference, name, detail) in enumerate(entries):
            item = QtWidgets.QTableWidgetItem(name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, deepcopy(reference))
            key = spawn_key(self.world_id, reference)
            last = key in keys and len(keys) == 1
            flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
            item.setFlags(flags if last else flags | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked if key in keys else QtCore.Qt.CheckState.Unchecked
            )
            item.setToolTip(
                "At least one starting position is required. Select another first."
                if last
                else name
            )
            self.table.setItem(row, 0, item)
            coordinates = QtWidgets.QTableWidgetItem(detail)
            coordinates.setFlags(flags)
            self.table.setItem(row, 1, coordinates)
            if key == current_key:
                self.table.setCurrentCell(row, 0)
        del blocker
        self.more.setVisible(has_more)
        self.done.setText(f"Done · {len(self.references)} selected")
        self.filter_rows(self.search.text())

    def toggle(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        reference = item.data(QtCore.Qt.ItemDataRole.UserRole)
        key = spawn_key(self.world_id, reference)
        remaining = [ref for ref in self.references if spawn_key(self.world_id, ref) != key]
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            if len(remaining) == len(self.references):
                self.references.append(deepcopy(reference))
        elif not remaining:
            blocker = QtCore.QSignalBlocker(self.table)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            del blocker
            return
        else:
            self.references = remaining
        self.changed.emit(deepcopy(self.references))

    def filter_rows(self, text: str) -> None:
        visible = 0
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            match = item is not None and text.casefold() in item.text().casefold()
            self.table.setRowHidden(row, not match)
            visible += int(match)
        self.empty.setVisible(not visible)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.hide()
            return
        super().keyPressEvent(event)


class _WorldCard(QtWidgets.QFrame):
    def __init__(self, world: dict[str, Any]) -> None:
        super().__init__()
        self.setProperty("worldCard", True)
        self.setMinimumWidth(280)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(label(world["label"], "section"), 1)
        self.select = QtWidgets.QCheckBox()
        self.select.setAccessibleName("Select world: " + world["label"])
        heading.addWidget(self.select)
        layout.addLayout(heading)
        width, height = world["width"] * world["resolution"], world["height"] * world["resolution"]
        layout.addWidget(label(f"{width:g} × {height:g} m", "muted"))
        self.preview = _MapPreview()
        self.preview.setAccessibleName(world["label"] + " map and starting positions")
        self.preview.clicked.connect(self.select.click)
        layout.addWidget(self.preview)
        self.summary = label("", "muted")
        self.summary.setMinimumHeight(20)
        self.summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.summary)
        footer = QtWidgets.QHBoxLayout()
        self.spawns = button("Spawns · 1", lambda: None, footer, role="outlined")
        self.spawns.setAccessibleName("Select starting positions for " + world["label"])
        footer.addStretch()
        layout.addLayout(footer)
        budget = QtWidgets.QHBoxLayout()
        budget.addWidget(label("Time limit · seconds", "muted"))
        budget.addStretch()
        self.time_limit = FloatEdit(world.get("default_time_limit", DEFAULT_LIMIT))
        self.time_limit.setFixedWidth(110)
        self.time_limit.setAccessibleName(world["label"] + " time limit in simulation seconds")
        self.time_limit.setToolTip(
            "Starts at first planar movement. Set 0 to disable this world's limit."
        )
        validator = self.time_limit.validator()
        if isinstance(validator, QtGui.QDoubleValidator):
            validator.setBottom(0)
        budget.addWidget(self.time_limit)
        layout.addLayout(budget)
        self.select.installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.select and event.type() in (
            QtCore.QEvent.Type.FocusIn,
            QtCore.QEvent.Type.FocusOut,
        ):
            self.setProperty("focused", event.type() == QtCore.QEvent.Type.FocusIn)
            self.style().unpolish(self)
            self.style().polish(self)
        return bool(super().eventFilter(watched, event))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.summary.setText(
            self.summary.fontMetrics().elidedText(
                self.summary.toolTip(),
                QtCore.Qt.TextElideMode.ElideRight,
                max(self.width() - 24, 1),
            )
        )


class WorldSelector(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, transport: Transport) -> None:
        super().__init__()
        self.transport = transport
        self._worlds: dict[str, dict[str, Any]] = {}
        self._cards: dict[str, _WorldCard] = {}
        self._selected: set[str] = set()
        self._references: dict[str, list[Any]] = {}
        self._saved: dict[str, dict[str, dict[str, Any]]] = {}
        self._poses: dict[tuple[str, str], dict[str, Any]] = {}
        self._maps: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, tuple[str, str | None]] = {}
        self._popups: dict[str, _SpawnPopup] = {}
        self._has_more: dict[str, bool] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll_area = QtWidgets.QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.content = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.content)
        self.grid.setContentsMargins(2, 2, 2, 2)
        self.grid.setSpacing(12)
        self.grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self._scroll_area.setWidget(self.content)
        outer.addWidget(self._scroll_area)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)

    def set_catalog(self, worlds: list[dict[str, Any]]) -> None:
        budget_text = {world: card.time_limit.text() for world, card in self._cards.items()}
        accepted = {str(world["id"]): deepcopy(world) for world in worlds}
        for identity in tuple(self._cards):
            if identity not in accepted or self._worlds.get(identity) != accepted[identity]:
                self._cards.pop(identity).deleteLater()
                popup = self._popups.pop(identity, None)
                if popup:
                    popup.hide()
                    popup.deleteLater()
                self._maps.pop(identity, None)
                self._pending = {
                    tag: pending for tag, pending in self._pending.items() if pending[0] != identity
                }
                self._poses = {key: pose for key, pose in self._poses.items() if key[0] != identity}
        self._worlds = accepted
        self._selected.intersection_update(accepted)
        for identity, world in accepted.items():
            self._references.setdefault(identity, ["Default"])
            if identity not in self._cards:
                card = _WorldCard(world)
                if identity in budget_text:
                    card.time_limit.setText(budget_text[identity])
                card.time_limit.textChanged.connect(lambda text: self.changed.emit())
                card.select.toggled.connect(
                    lambda checked, wid=identity: self.select_world(wid, checked)
                )
                card.spawns.clicked.connect(
                    lambda checked=False, wid=identity: self.show_spawns(wid)
                )
                self._cards[identity] = card
            self._sync(identity)
            if self.isVisible() and identity not in self._maps:
                self._request(identity)
        self._layout_cards()

    def selections(self) -> dict[str, list[Any]]:
        return {
            world: deepcopy(self._references[world])
            for world in self._worlds
            if world in self._selected
        }

    def set_selections(self, selected: dict[str, list[Any]]) -> None:
        accepted = set(selected).intersection(self._worlds)
        if not accepted and self._selected:
            return
        self._selected = accepted
        for world in self._selected:
            self._references[world] = deepcopy(selected[world]) or ["Default"]
        for world in self._worlds:
            self._sync(world)

    def time_limits(self) -> dict[str, float]:
        limits = {}
        for world in self._worlds:
            if world not in self._selected:
                continue
            try:
                limit = self._cards[world].time_limit.value()
                if limit < 0:
                    raise ValueError("Negative limit")
            except ValueError as exc:
                raise ValueError(
                    f"{self._worlds[world]['label']}: enter a finite, nonnegative time limit in seconds."
                ) from exc
            limits[world] = limit
        return limits

    def set_time_limits(self, limits: dict[str, float]) -> None:
        for world, card in self._cards.items():
            card.time_limit.setValue(
                limits.get(world, self._worlds[world].get("default_time_limit", DEFAULT_LIMIT))
            )

    def set_spawns(self, world: str, rows: list[dict[str, Any]], has_more: bool = False) -> None:
        if world not in self._worlds:
            return
        previous = self._saved.get(world, {})
        current = {str(row["id"]): deepcopy(row) for row in rows}
        for key, row in current.items():
            pose = row.get("snapshot", row)
            if all(field in pose for field in ("x", "y", "yaw")):
                self._poses[world, key] = deepcopy(pose)
        for ref in self._references.get(world, []):
            key = spawn_key(world, ref)
            if key not in current and key in previous:
                current[key] = previous[key]
        self._saved[world] = current
        self._has_more[world] = has_more
        self._sync(world)

    def spawn_label(self, world: str, reference: Any) -> str:
        if isinstance(reference, dict):
            return str(reference["name"])
        if spawn_key(world, reference) == "default-" + world:
            return "Default"
        row = self._saved.get(world, {}).get(str(reference), {})
        return str(row.get("name") or str(reference)[:8])

    def spawn_options(self) -> list[tuple[str, str, Any, str]]:
        return [
            (world, info["label"], deepcopy(ref), name)
            for world, info in self._worlds.items()
            for ref, name, detail in self._entries(world)
        ]

    def select_world(self, world: str, checked: bool) -> None:
        if world not in self._worlds:
            return
        if not checked and self._selected == {world}:
            self._sync(world)
            return
        if checked:
            self._selected.add(world)
        else:
            self._selected.discard(world)
            popup = self._popups.get(world)
            if popup:
                popup.hide()
        self._sync(world)
        self.changed.emit()

    def _pose(self, world: str, reference: Any) -> dict[str, Any] | None:
        if isinstance(reference, dict):
            return reference
        key = spawn_key(world, reference)
        return (
            self._worlds[world]["default_spawn"]
            if key == "default-" + world
            else self._poses.get((world, key))
        )

    def _entries(self, world: str) -> list[tuple[Any, str, str]]:
        references: list[Any] = ["Default", *self._saved.get(world, {})]
        known = {spawn_key(world, ref) for ref in references}
        for ref in self._references.get(world, []):
            if spawn_key(world, ref) not in known or isinstance(ref, dict):
                references = [r for r in references if spawn_key(world, r) != spawn_key(world, ref)]
                references.append(deepcopy(ref))
                known.add(spawn_key(world, ref))
        entries = []
        for ref in references:
            row = self._saved.get(world, {}).get(spawn_key(world, ref), {})
            pose = self._pose(world, ref)
            revision = f" · r{row['revision']}" if row.get("revision") is not None else ""
            entries.append(
                (ref, self.spawn_label(world, ref) + revision, pose_text(pose) if pose else "")
            )
        return entries

    def _request(self, world: str, revision: str | None = None) -> None:
        if (world, revision) in self._pending.values():
            return
        tag = "world-selection-" + uuid.uuid4().hex
        self._pending[tag] = (world, revision)
        self.transport.submit(
            tag,
            "spawns.get" if revision else "spawns.map",
            {"revision_id": revision} if revision else {"world_id": world},
        )

    def _sync(self, world: str) -> None:
        card = self._cards[world]
        blocker = QtCore.QSignalBlocker(card.select)
        card.select.setChecked(world in self._selected)
        del blocker
        card.setProperty("selected", world in self._selected)
        card.style().unpolish(card)
        card.style().polish(card)
        references = self._references[world]
        card.spawns.setText(f"Spawns · {len(references)}")
        card.spawns.setEnabled(world in self._selected)
        card.time_limit.setEnabled(world in self._selected)
        names = ", ".join(self.spawn_label(world, ref) for ref in references)
        card.summary.setText(
            card.fontMetrics().elidedText(
                names, QtCore.Qt.TextElideMode.ElideRight, max(card.width() - 24, 180)
            )
        )
        card.summary.setToolTip(names)
        card.preview.poses = []
        for ref in references:
            pose = self._pose(world, ref)
            if pose is not None:
                card.preview.poses.append(pose)
            elif world in self._selected:
                self._request(world, spawn_key(world, ref))
        card.preview.update()
        popup = self._popups.get(world)
        if popup and popup.isVisible():
            popup.refresh(self._entries(world), references, self._has_more.get(world, False))
        for identity, other_card in self._cards.items():
            last = self._selected == {identity}
            other_card.select.setEnabled(not last)
            other_card.select.setToolTip(
                "Select another world before deselecting this one." if last else ""
            )

    def show_spawns(self, world: str) -> None:
        if world not in self._selected:
            return
        popup = self._popups.get(world)
        if popup is None:
            popup = _SpawnPopup(self, self._worlds[world])
            popup.changed.connect(lambda refs, wid=world: self._set_spawns(wid, refs))
            self._popups[world] = popup
        popup.refresh(
            self._entries(world), self._references[world], self._has_more.get(world, False)
        )
        anchor = self._cards[world].spawns
        position = anchor.mapToGlobal(QtCore.QPoint(0, anchor.height()))
        bounds = anchor.screen().availableGeometry()
        position.setX(max(bounds.left(), min(position.x(), bounds.right() - popup.width())))
        if position.y() + popup.height() > bounds.bottom():
            position.setY(
                max(bounds.top(), anchor.mapToGlobal(QtCore.QPoint()).y() - popup.height())
            )
        popup.move(position)
        popup.show()
        popup.search.setFocus()

    def _set_spawns(self, world: str, references: list[Any]) -> None:
        if not references:
            return
        self._references[world] = deepcopy(references)
        self._sync(world)
        self.changed.emit()

    def received(self, tag: str, result: Any) -> None:
        pending = self._pending.pop(tag, None)
        if pending is None:
            return
        world, revision = pending
        if world not in self._worlds or result.get("world_id") != world:
            return
        if revision:
            if result.get("id") != revision:
                return
            self._poses[world, revision] = deepcopy(result)
            self._sync(world)
        else:
            try:
                self._cards[world].preview.set_map(result)
            except ValueError as exc:
                self._cards[world].preview.message = str(exc)
                self._cards[world].preview.update()
                return
            self._maps[world] = result

    def failed(self, tag: str, error: str) -> None:
        pending = self._pending.pop(tag, None)
        if pending and pending[0] in self._cards:
            preview = self._cards[pending[0]].preview
            if pending[1] is None:
                preview.message = "Map unavailable"
                preview.setToolTip(error)
                preview.update()
            else:
                self._cards[pending[0]].summary.setToolTip(error)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        for world in self._worlds:
            if world not in self._maps:
                self._request(world)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        for popup in self._popups.values():
            popup.hide()
        super().hideEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._layout_cards)

    def _layout_cards(self) -> None:
        columns = (
            3
            if self._scroll_area.viewport().width() >= 900
            else 2
            if self._scroll_area.viewport().width() >= 590
            else 1
        )
        while self.grid.count():
            self.grid.takeAt(0)
        for index, world in enumerate(self._worlds):
            card = self._cards[world]
            self.grid.addWidget(card, index // columns, index % columns)
        for column in range(3):
            self.grid.setColumnStretch(column, int(column < columns))
