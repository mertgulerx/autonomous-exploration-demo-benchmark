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
from collections import OrderedDict
import math
from pathlib import Path
import queue
import subprocess
from typing import Any, Iterator
import uuid

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .contracts import ContractError, canonical
from .protocol import Client
from .storage import atomic_bytes
from .gui_style import COLORS, cell_text, icon
from .gui_table import TableTextDelegate


class TextSelection(QtCore.QObject):
    @staticmethod
    def copy_table(view: QtWidgets.QTableView) -> None:
        delegate = view.itemDelegate()
        if isinstance(delegate, TableTextDelegate) and delegate.selected_text():
            QtGui.QGuiApplication.clipboard().setText(delegate.selected_text())
            return
        values = {(index.row(), index.column()): index.data() for index in view.selectedIndexes()}
        if not values:
            return
        rows = sorted({row for row, column in values})
        columns = sorted({column for row, column in values})
        text = "\n".join(
            "\t".join(
                str(values[row, column]) if values.get((row, column)) is not None else ""
                for column in columns
            )
            for row in rows
        )
        QtGui.QGuiApplication.clipboard().setText(text)

    def table_menu(self, view: QtWidgets.QTableView, position: QtCore.QPoint) -> None:
        index = view.indexAt(position)
        delegate = view.itemDelegate()
        selecting_text = isinstance(delegate, TableTextDelegate) and bool(delegate.selected_text())
        if not selecting_text and index.isValid() and not view.selectionModel().isSelected(index):
            view.setCurrentIndex(index)
        menu = QtWidgets.QMenu(view)
        copy = menu.addAction("Copy")
        copy.setShortcut(QtGui.QKeySequence.StandardKey.Copy)
        copy.setEnabled(selecting_text or bool(view.selectedIndexes()))
        copy.triggered.connect(lambda: self.copy_table(view))
        menu.exec(view.viewport().mapToGlobal(position))
        menu.deleteLater()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Polish and isinstance(watched, QtWidgets.QLabel):
            watched.setTextInteractionFlags(
                watched.textInteractionFlags()
                | QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
        elif event.type() == QtCore.QEvent.Type.Polish and isinstance(
            watched, QtWidgets.QTableView
        ):
            if not isinstance(watched.itemDelegate(), TableTextDelegate):
                watched.setItemDelegate(TableTextDelegate(watched))
            if watched.contextMenuPolicy() != QtCore.Qt.ContextMenuPolicy.CustomContextMenu:
                watched.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
                watched.customContextMenuRequested.connect(
                    lambda position, view=watched: self.table_menu(view, position)
                )
        elif (
            isinstance(watched, QtWidgets.QTableView)
            and isinstance(event, QtGui.QKeyEvent)
            and event.type() == QtCore.QEvent.Type.KeyPress
            and event.matches(QtGui.QKeySequence.StandardKey.Copy)
        ):
            delegate = watched.itemDelegate()
            if watched.selectedIndexes() or (
                isinstance(delegate, TableTextDelegate) and delegate.bounds() is not None
            ):
                self.copy_table(watched)
                return True
        return super().eventFilter(watched, event)


class FloatEdit(QtWidgets.QLineEdit):
    valueChanged = QtCore.Signal(float)

    def __init__(self, value: float = 0.0):
        super().__init__()
        validator = QtGui.QDoubleValidator(-1e308, 1e308, 17, self)
        validator.setLocale(QtCore.QLocale.c())
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        self.setValidator(validator)
        self.setValue(value)
        self.textChanged.connect(self.emit_value)

    def setValue(self, value: float) -> None:
        self.setText(repr(float(value)))

    def value(self) -> float:
        value = float(self.text())
        if not math.isfinite(value):
            raise ValueError("Enter a finite numerical value")
        return value

    def emit_value(self) -> None:
        if self.hasAcceptableInput():
            self.valueChanged.emit(self.value())


def button(
    label: str, action: Any, layout: Any, *, role: str = "secondary", glyph: str | None = None
) -> QtWidgets.QPushButton:
    widget = QtWidgets.QPushButton(label)
    widget.setProperty("role", role)
    if role == "danger" and glyph is None:
        glyph = "alert"
    if glyph:
        widget.setProperty("glyph", glyph)
        color = (
            COLORS["text.on-accent"]
            if role == "primary"
            else COLORS["status.error"]
            if role == "danger"
            else COLORS["surface.base"]
            if role == "secondary"
            else COLORS["text.muted"]
        )
        widget.setIcon(icon(glyph, color))
    widget.clicked.connect(action)
    layout.addWidget(widget)
    return widget


class Transport(QtCore.QThread):
    received = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str)

    def __init__(self, socket: Path):
        super().__init__()
        self.client = Client(socket, timeout=120)
        self.requests: queue.Queue[tuple[str, str, dict[str, Any], str]] = queue.Queue(128)

    def submit(self, tag: str, method: str, params: dict[str, Any] | None = None) -> None:
        try:
            self.requests.put_nowait((tag, method, params or {}, str(uuid.uuid4())))
        except queue.Full:
            self.failed.emit(tag, "Too many pending operations; wait for the engine")

    def run(self) -> None:
        while not self.isInterruptionRequested():
            try:
                tag, method, params, identity = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                result: Any
                if method == "local.read":
                    path = Path(params["path"])
                    if path.stat().st_size > 4 * 1024 * 1024:
                        raise ValueError("Import exceeds 4 MiB")
                    result = {"text": path.read_text()}
                elif method == "local.read_image":
                    path = Path(params["path"])
                    if path.stat().st_size > 8 * 1024 * 1024:
                        raise ValueError("Preview exceeds 8 MiB")
                    result = {"image": path.read_bytes()}
                elif method == "local.write":
                    atomic_bytes(Path(params["path"]), params["text"].encode())
                    result = {"path": params["path"]}
                else:
                    result = self.client.call(method, params, identity)
                self.received.emit(tag, result)
            except (
                ContractError,
                OSError,
                ValueError,
                EOFError,
                subprocess.CalledProcessError,
            ) as exc:
                self.failed.emit(tag, str(exc))

    def close(self) -> bool:
        self.requestInterruption()
        self.client.close()
        return self.wait(1500)


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, columns: list[tuple[str, str]]):
        super().__init__()
        self.columns = columns
        self.rows: list[dict[str, Any]] = []
        self.names: dict[str, dict[str, str]] = {}

    def replace(self, rows: list[dict[str, Any]]) -> None:
        if len(rows) == len(self.rows) and [r.get("id") for r in rows] == [
            r.get("id") for r in self.rows
        ]:
            self.rows = rows
            if rows and self.columns:
                self.dataChanged.emit(
                    self.index(0, 0), self.index(len(rows) - 1, len(self.columns) - 1)
                )
            return
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def set_columns(self, columns: list[tuple[str, str]]) -> None:
        if columns != self.columns:
            self.beginResetModel()
            self.columns = columns
            self.rows = []
            self.endResetModel()

    def record(self, row: int) -> dict[str, Any] | None:
        return self.rows[row] if 0 <= row < len(self.rows) else None

    def rowCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None
        record = self.record(index.row())
        if record is None:
            return None
        key = self.columns[index.column()][0]
        if key == "actions":
            return None
        value = record.get(key)
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if key == "outcome" and value is None:
                return cell_text(
                    "phase",
                    record.get("state") or record.get("phase"),
                )
            if key in self.names and value in self.names[key]:
                return self.names[key][value]
            return cell_text(key, value)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return str(value) if value is not None else "Unavailable"
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float)):
            return QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if (
            role == QtCore.Qt.ItemDataRole.DisplayRole
            and orientation == QtCore.Qt.Orientation.Horizontal
        ):
            return self.columns[section][1]
        return None


class PagedTableModel(TableModel):
    page_needed = QtCore.Signal(int)
    pages_changed = QtCore.Signal()
    page_size = 100
    cache_size = 4

    def __init__(self, columns: list[tuple[str, str]]):
        super().__init__(columns)
        self.pages: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self.total = 0

    def replace(self, rows: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self.rows, self.total = rows, len(rows)
        self.pages.clear()
        if rows:
            self.pages[0] = rows
        self.endResetModel()
        self.pages_changed.emit()

    def set_columns(self, columns: list[tuple[str, str]]) -> None:
        if columns != self.columns:
            self.beginResetModel()
            self.columns = columns
            self.rows, self.total = [], 0
            self.pages.clear()
            self.endResetModel()
            self.pages_changed.emit()

    def record(self, row: int) -> dict[str, Any] | None:
        offset, position = divmod(row, self.page_size)
        rows = self.pages.get(offset * self.page_size, [])
        return rows[position] if 0 <= row < self.total and position < len(rows) else None

    def records(self) -> Iterator[tuple[int, dict[str, Any]]]:
        for offset, rows in sorted(self.pages.items()):
            for position, record in enumerate(rows):
                yield offset + position, record

    def rowCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else self.total

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if index.isValid() and self.record(index.row()) is None:
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                self.page_needed.emit(index.row() // self.page_size * self.page_size)
                return "…" if index.column() == 0 else None
            return None
        return super().data(index, role)

    def set_page(
        self, offset: int, rows: list[dict[str, Any]], total: int, *, invalidate: bool = False
    ) -> None:
        if invalidate:
            self.pages.clear()
        if total != self.total:
            self.pages.clear()
            if total > self.total:
                self.beginInsertRows(QtCore.QModelIndex(), self.total, total - 1)
                self.total = total
                self.endInsertRows()
            else:
                self.beginRemoveRows(QtCore.QModelIndex(), total, self.total - 1)
                self.total = total
                self.endRemoveRows()
        self.rows = rows
        self.pages[offset] = rows
        self.pages.move_to_end(offset)
        while len(self.pages) > self.cache_size:
            self.pages.popitem(last=False)
        if self.total and self.columns:
            self.dataChanged.emit(
                self.index(0, 0), self.index(self.total - 1, len(self.columns) - 1)
            )
        self.pages_changed.emit()


class PagedTable(QtWidgets.QWidget):
    selected = QtCore.Signal(object)
    loaded = QtCore.Signal(object)

    def __init__(
        self,
        transport: Transport,
        tag: str,
        table: str,
        columns: list[tuple[str, str]],
        *,
        method: str = "results.query",
        result_key: str = "rows",
    ):
        super().__init__()
        self.transport, self.tag, self.table = transport, tag, table
        self.method, self.result_key = method, result_key
        self.filters: dict[str, Any] = {}
        self.sort_by: str | None = None
        self.sort_descending = False
        self.offset, self.total, self.busy = 0, 0, False
        self.refresh_pending = False
        self.context: tuple[str, str] | None = None
        self.requested: tuple[str, int, str] | None = None
        self.invalidate_pages = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.view = QtWidgets.QTableView()
        self.model = PagedTableModel(columns)
        self.view.setModel(self.model)
        self.view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.view.setSortingEnabled(False)
        self.view.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Interactive
        )
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.verticalHeader().hide()
        self.view.setAlternatingRowColors(True)
        self.view.setShowGrid(False)
        self.view.verticalHeader().setDefaultSectionSize(40)
        self.view.horizontalHeader().setDefaultSectionSize(150)
        self.view.selectionModel().currentRowChanged.connect(self.select_row)
        layout.addWidget(self.view)
        self.label = QtWidgets.QLabel("0 records")
        self.label.setProperty("kind", "muted")
        layout.addWidget(self.label)
        self.load_timer = QtCore.QTimer(self)
        self.load_timer.setSingleShot(True)
        self.load_timer.setInterval(30)
        self.load_timer.timeout.connect(self.load_visible)
        self.model.page_needed.connect(lambda _: self.schedule_visible())
        self.view.verticalScrollBar().valueChanged.connect(self.schedule_visible)
        self.view.viewport().installEventFilter(self)
        self.sync_timer = QtCore.QTimer(self)
        self.sync_timer.setInterval(2000)
        self.sync_timer.timeout.connect(lambda: self.refresh() if self.isVisible() else None)
        self.sync_timer.start()
        transport.received.connect(self._received)
        transport.failed.connect(self._failed)

    def select_row(self, current: QtCore.QModelIndex, previous: QtCore.QModelIndex) -> None:
        record = self.model.record(current.row()) if current.isValid() else None
        if record is not None:
            self.selected.emit(record)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() in (QtCore.QEvent.Type.Show, QtCore.QEvent.Type.Resize):
            self.schedule_visible()
        return super().eventFilter(watched, event)

    def schedule_visible(self, *args: Any) -> None:
        self.load_timer.start()

    def visible_offsets(self) -> list[int]:
        first = max(0, self.view.rowAt(0))
        last = self.view.rowAt(self.view.viewport().height() - 1)
        if last < 0:
            last = min(self.model.rowCount() - 1, first + 100)
        return list(range(first // 100 * 100, last // 100 * 100 + 1, 100))

    def load_visible(self) -> None:
        if self.busy or not self.isVisible() or not self.isEnabled():
            return
        for offset in self.visible_offsets():
            if offset not in self.model.pages:
                self.request_page(offset)
                return

    def reset_page(self) -> None:
        self.offset = 0
        self.model.replace([])
        self.view.scrollToTop()
        if self.busy:
            self.refresh_pending = True
        self.refresh()

    def page_context(self) -> str:
        return canonical(
            {
                "filters": self.filters,
                "sort_by": self.sort_by,
                "sort_descending": self.sort_descending,
            }
        )

    def refresh(self) -> None:
        if not self.isEnabled():
            return
        context = self.table, self.page_context()
        if self.busy and self.context == context:
            return
        if self.context is not None and self.context != context:
            self.offset = 0
            self.model.replace([])
        elif self.isVisible() and self.model.rowCount():
            self.offset = self.visible_offsets()[0]
        self.request_page(self.offset, invalidate=True)

    def request_page(self, offset: int, *, invalidate: bool = False) -> None:
        if not self.busy:
            self.busy = True
            self.offset = offset
            self.invalidate_pages = invalidate
            self.context = self.table, self.page_context()
            self.requested = self.table, offset, self.context[1]
            params: dict[str, Any] = (
                {"table": self.table, "filters": self.filters}
                if self.method == "results.query"
                else dict(self.filters)
            )
            if self.sort_by is not None:
                params.update(sort_by=self.sort_by, sort_descending=self.sort_descending)
            self.transport.submit(
                self.tag,
                self.method,
                dict(
                    params,
                    offset=self.offset,
                    limit=100,
                    summary=True,
                ),
            )

    def _received(self, tag: str, result: Any) -> None:
        if tag == self.tag:
            self.busy = False
            if not self.isEnabled():
                self.refresh_pending = False
                return
            if self.refresh_pending:
                self.refresh_pending = False
                self.refresh()
                return
            if self.requested != (
                self.table,
                self.offset,
                self.page_context(),
            ):
                self.refresh()
                return
            self.total = result["total"]
            current = self.view.currentIndex()
            record = self.model.record(current.row()) if current.isValid() else None
            identity = record.get("id", record.get("path")) if record else None
            self.model.set_page(
                self.offset, result[self.result_key], self.total, invalidate=self.invalidate_pages
            )
            if identity is not None:
                match = next(
                    (i for i, r in self.model.records() if r.get("id", r.get("path")) == identity),
                    None,
                )
                if match is not None:
                    target = self.model.index(match, current.column())
                    if self.view.currentIndex() != target:
                        self.view.setCurrentIndex(target)
                    elif self.model.record(match) != record:
                        self.selected.emit(self.model.record(match))
                elif (
                    self.offset <= current.row() < self.offset + 100 or current.row() >= self.total
                ):
                    self.view.clearSelection()
                    self.view.setCurrentIndex(QtCore.QModelIndex())
            elif current.isValid():
                loaded = self.model.record(current.row())
                if loaded is not None:
                    self.selected.emit(loaded)
            self.label.setText(f"{self.total:,} record{'s' if self.total != 1 else ''}")
            self.loaded.emit(result)
            self.schedule_visible()

    def _failed(self, tag: str, message: str) -> None:
        if tag == self.tag:
            self.busy, self.refresh_pending = False, False
            if not self.isEnabled():
                return
            if self.context != (self.table, self.page_context()):
                self.refresh()
                return
            self.label.setText(message)


class HeadingArrow(pg.ArrowItem):  # type: ignore[misc]
    def __init__(self, view: "MapView"):
        super().__init__(
            angle=0, headLen=15, tipAngle=30, brush=COLORS["text.strong"], pen=COLORS["text.strong"]
        )
        self.view = view

    def hoverEvent(self, event: Any) -> None:
        if self.view.editable:
            event.acceptDrags(QtCore.Qt.MouseButton.LeftButton)

    def mouseDragEvent(self, event: Any) -> None:
        if not self.view.editable or event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return
        event.accept()
        point = self.view.plotItem.vb.mapSceneToView(event.scenePos())
        x, y, _ = self.view.pose
        if math.hypot(point.x() - x, point.y() - y) > 1e-6:
            yaw = math.atan2(point.y() - y, point.x() - x)
            self.view.first_click = None
            self.view.set_pose(x, y, yaw)
            self.view.pose_selected.emit(x, y, yaw)


class MapView(pg.PlotWidget):  # type: ignore[misc]
    pose_selected = QtCore.Signal(float, float, float)

    def __init__(self) -> None:
        super().__init__()
        self.plotItem.hideButtons()
        self.setAspectLocked()
        self.setLabel("bottom", "World X", units="m")
        self.setLabel("left", "World Y", units="m")
        self.showGrid(x=True, y=True, alpha=0.12)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.addItem(self.image)
        self.footprint = self.plot(pen=pg.mkPen(COLORS["text.heading"], width=2))
        self.arrow = HeadingArrow(self)
        self.addItem(self.arrow)
        self.pose = [0.0, 0.0, 0.0]
        self.vertices: list[list[float]] = []
        self.editable = False
        self.first_click: tuple[float, float] | None = None
        self.map_initialized = False
        self.scene().sigMouseClicked.connect(self._click)
        self.setBackground(COLORS["surface.sunken"])
        for axis_name in ("bottom", "left"):
            axis = self.getAxis(axis_name)
            axis.setPen(COLORS["border.control"])
            axis.setTextPen(COLORS["text.muted"])
            axis.setLabel(color=COLORS["text.muted"])
            axis.setTickFont(QtGui.QFont(QtGui.QGuiApplication.font().family(), 9))

    def clear_view(self) -> None:
        self.map_initialized = False
        self.image.clear()
        self.footprint.setData([], [])

    def hide_pose(self) -> None:
        self.footprint.setData([], [])
        self.arrow.hide()

    def set_map(self, result: dict[str, Any]) -> None:
        image = QtGui.QImage.fromData(base64.b64decode(result["image_png"]))
        if image.isNull():
            raise ValueError("Unreadable map image")
        image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        array = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
            image.height(), image.bytesPerLine()
        )
        pixels = array[:, : image.width() * 4].reshape(image.height(), image.width(), 4).copy()
        ink_color = QtGui.QColor(COLORS["text.heading"])
        paper_color = QtGui.QColor(COLORS["surface.raised"])
        ink = np.array((ink_color.red(), ink_color.green(), ink_color.blue()), dtype=np.uint16)
        paper = np.array(
            (paper_color.red(), paper_color.green(), paper_color.blue()), dtype=np.uint16
        )
        pixels[:, :, :3] = ink + pixels[:, :, :3].astype(np.uint16) * (paper - ink) // 255
        self.image.setImage(pixels, autoLevels=False)
        x, y, yaw = result["origin"]
        resolution = result["resolution"]
        c, s = math.cos(yaw) * resolution, math.sin(yaw) * resolution
        self.image.setTransform(QtGui.QTransform(c, s, -s, c, x, y))
        if not self.map_initialized:
            self.map_initialized = True
            self.setRange(rect=self.image.mapRectToParent(self.image.boundingRect()), padding=0.04)

    def set_pose(
        self, x: float, y: float, yaw: float, vertices: list[list[float]] | None = None
    ) -> None:
        self.pose = [x, y, yaw]
        if vertices is not None:
            self.vertices = vertices
        self.arrow.show()
        self.arrow.setPos(x, y)
        self.arrow.setStyle(angle=180 - math.degrees(yaw))
        if self.vertices:
            a = np.asarray(self.vertices + self.vertices[:1])
            c, s = math.cos(yaw), math.sin(yaw)
            a = a @ np.array([[c, s], [-s, c]]) + [x, y]
            self.footprint.setData(a[:, 0], a[:, 1])

    def _click(self, event: Any) -> None:
        if not self.editable or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        point = self.plotItem.vb.mapSceneToView(event.scenePos())
        if self.first_click is None:
            self.first_click = point.x(), point.y()
            self.set_pose(point.x(), point.y(), self.pose[2])
            self.pose_selected.emit(*self.pose)
        else:
            x, y = self.first_click
            yaw = math.atan2(point.y() - y, point.x() - x)
            self.first_click = None
            self.set_pose(x, y, yaw)
            self.pose_selected.emit(x, y, yaw)
