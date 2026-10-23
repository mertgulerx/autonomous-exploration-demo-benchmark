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

"""Results, Files tab (29, 30; RESULTS_STUDY.md decision 7; RESULTS_RESEARCH RD-41, A14).

Left, the root ("Study exports", or a run its run view's Files button chose), a breadcrumb and the folder's entries
(folders first, each with its kind's icon and size; ``files.list``). Right, the chosen file as its kind wants:
a Parquet or CSV table in pages of 100 rows with the export codebook's units under the column names
(``files.table``), text and logs in 64 KiB windows with search (``files.log``; zstd and gzip logs decompressed by
the engine), JSON and YAML as text (JSON indented once the whole file is loaded), PDFs, images and ``.npz`` maps
read from the host path the listing names (read off the GUI thread; a map is decoded and another file hashed on
it, at most 16 MiB), and any other file as its size and SHA-256 with
"Open externally". The engine builds these reads off its loop and answers ``loading`` first, so each is polled;
a reply for an earlier root, folder or file is dropped.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any
import uuid

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .gui_cells import (
    CELL_ROLE,
    MONO_TEXT,
    SUBLINE_ROLE,
    Cell,
    CellDelegate,
    ColumnStyle,
    OverlayScrollBar,
    TableLook,
    apply_columns,
    style_table,
)
from .gui_controls import KeySelect, SearchField
from .gui_pdf import Well
from .gui_support import POLL_MS, TableModel, Transport
from .gui_theme import ThemeManager, snap
from .gui_widgets import (
    BODY,
    LOG_TEXT,
    Button,
    Card,
    CardHead,
    FlexRow,
    TextLine,
    TextStyle,
    Themed,
    draw_text,
)

ROW_TEXT = TextStyle(13, 400, 18)
DIR_TEXT = TextStyle(13, 500, 18)
SIZE_TEXT = TextStyle(12, 400, 18, tabular=True)
CRUMB = TextStyle(12.5, 400, 18)
CRUMB_LAST = TextStyle(12.5, 500, 18)
FOOT = TextStyle(12.5, 400, 18)
ICONS = {
    "table": "table",
    "json": "braces",
    "yaml": "file",
    "text": "file",
    "log": "terminal",
    "pdf": "file",
    "image": "image",
    "map": "map",
}
WINDOW = 65536  # a log window (64 KiB)
RUN_ROOTS = 5  # run directories kept in the root select
NUMERIC = re.compile(
    r"^(U?(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT)|U?INT\d*|FLOAT|DOUBLE|DECIMAL.*|REAL)$", re.I
)


def size_text(size: Any) -> str:
    if not isinstance(size, int):
        return ""
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MiB"
    return f"{size / 1024:.1f} KiB"


@dataclass(frozen=True, slots=True)
class Root:
    """A tree the tab shows: a study's directory or a finished run's directory."""

    kind: str  # "study" or "attempt"
    identity: str
    label: str  # the select's text
    home: str  # the breadcrumb's first folder name


# ---------- the folder ----------


class Crumb(Themed):
    """``.crumb``: a folder icon, then the path's folders split by chevrons, the last in ink 500; a click on a
    folder goes there (``chosen`` carries its relative path)."""

    chosen = QtCore.Signal(str)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.home, self.path = "", ""
        self.boxes: list[tuple[QtCore.QRectF, str]] = []
        self.setFixedHeight(40)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def set_path(self, home: str, path: str) -> None:
        self.home, self.path = home, path
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        for box, path in self.boxes:
            if box.contains(event.position()):
                self.chosen.emit(path)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        painter.fillRect(QtCore.QRect(0, 39, self.width(), 1), theme.color("border.line"))
        top = (39 - 18) / 2
        gui_icons.draw(
            painter, "folder", theme.color("text.tertiary"), QtCore.QPointF(20, top + 2.5), 13
        )
        x = 20.0 + 13 + 4
        parts = [p for p in self.path.split("/") if p]
        # the root's name until a folder is open, then the path from its first folder
        names = [(part, "/".join(parts[: i + 1])) for i, part in enumerate(parts)] or [
            (self.home, "")
        ]
        self.boxes = []
        for index, (name, path) in enumerate(names):
            last = index == len(names) - 1
            if index:
                gui_icons.draw(
                    painter,
                    "cright",
                    theme.color("text.tertiary"),
                    QtCore.QPointF(x, top + 2.5),
                    13,
                )
                x += 13 + 4
            style = CRUMB_LAST if last else CRUMB
            width = draw_text(
                painter,
                x,
                top,
                style.elide(name, self.width() - 20 - x),
                style,
                theme.color("text.ink" if last else "text.secondary"),
            )
            self.boxes.append((QtCore.QRectF(x, 0, width, 39), path))
            x += width + 4


class Entries(Themed):
    """``.frow``: 34 px rows (icon, name, size), folders first and named in ink 500; the chosen file on
    ``surface.selected`` with a 2 px ink bar. ``opened`` carries the entry."""

    opened = QtCore.Signal(object)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.entries: list[dict[str, Any]] = []
        self.current: str | None = None
        self.hovered = -1
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Folder")

    def set_entries(self, entries: Sequence[dict[str, Any]]) -> None:
        self.entries = list(entries)
        self.setFixedHeight(34 * len(self.entries))
        self.update()

    def _row_at(self, y: float) -> int:
        row = int(y // 34)
        return row if 0 <= row < len(self.entries) else -1

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        row = self._row_at(event.position().y())
        if row != self.hovered:
            self.hovered = row
            self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hovered = -1
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        row = self._row_at(event.position().y())
        if row >= 0 and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.opened.emit(self.entries[row])

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        paths = [e["path"] for e in self.entries]
        index = paths.index(self.current) if self.current in paths else -1
        if key in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Up) and self.entries:
            # the arrows move the highlight (a file opens as it is reached; a folder waits for Enter)
            start = (
                index
                if index >= 0
                else (-1 if key == QtCore.Qt.Key.Key_Down else len(self.entries))
            )
            step = 1 if key == QtCore.Qt.Key.Key_Down else -1
            entry = self.entries[(start + step) % len(self.entries)]
            self.current = str(entry["path"])
            if entry.get("type") != "directory":
                self.opened.emit(entry)
            self.update()
        elif key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) and index >= 0:
            self.opened.emit(self.entries[index])
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        width = self.width()
        for index, entry in enumerate(self.entries):
            top = index * 34
            folder = entry.get("type") == "directory"
            selected = entry.get("path") == self.current
            if selected:
                painter.fillRect(QtCore.QRect(0, top, width, 33), theme.color("surface.selected"))
                painter.fillRect(QtCore.QRect(0, top, 2, 33), theme.color("text.ink"))
            elif index == self.hovered:
                painter.fillRect(QtCore.QRect(0, top, width, 33), theme.color("surface.hover"))
            painter.fillRect(QtCore.QRect(0, top + 33, width, 1), theme.color("border.line"))
            icon = "folder" if folder else ICONS.get(str(entry.get("kind")), "file")
            token = "text.ink" if selected else "text.tertiary"
            gui_icons.draw(painter, icon, theme.color(token), QtCore.QPointF(20, top + 9.5), 15)
            text_top = top + (33 - 18) / 2
            size = size_text(entry.get("size"))
            right = width - 16
            if size:
                draw_text(
                    painter,
                    right - SIZE_TEXT.advance(size),
                    text_top,
                    size,
                    SIZE_TEXT,
                    theme.color("text.tertiary"),
                )
            style = DIR_TEXT if folder else ROW_TEXT
            room = right - (SIZE_TEXT.advance(size) + 10 if size else 0) - (20 + 18 + 10)
            name = style.elide(str(entry.get("name")), room)
            draw_text(
                painter,
                20 + 18 + 10,
                text_top,
                name,
                style,
                theme.color("text.ink" if folder else "text.body"),
            )


class FolderCard(Card):
    """``.card.fl`` (360 px): the root select, the breadcrumb, the entries (scrolling) and their count."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.root = KeySelect(themes, "Root")
        self.root.setParent(self)
        self.crumb = Crumb(themes, self)
        self.area = QtWidgets.QScrollArea(self)
        self.area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.area.setWidgetResizable(False)
        self.area.viewport().setAutoFillBackground(False)
        self.area.setStyleSheet("QScrollArea { background: transparent; }")
        self.entries = Entries(themes)
        self.area.setWidget(self.entries)
        OverlayScrollBar(themes, self.area)
        self.foot = TextLine(themes, "", FOOT, "text.secondary", self)
        self.total = 0
        self.error = ""  # why the folder could not be listed; shown instead of the count
        self.area.verticalScrollBar().valueChanged.connect(lambda _: self._count())
        self.area.verticalScrollBar().rangeChanged.connect(lambda *_: self._count())

    def set_error(self, reason: str) -> None:
        self.total, self.error = 0, reason
        self._count()

    def set_total(self, total: int) -> None:
        self.total, self.error = total, ""
        QtCore.QTimer.singleShot(0, self._count)  # once the list has its height in the scroll area

    def _count(self) -> None:
        """ "16 of 31 entries · scroll for more": the rows in view of the folder's entries."""
        if self.error:
            self.foot.setText(self.error)
            return
        shown = len(self.entries.entries)
        viewport = self.area.viewport().height()
        start = self.area.verticalScrollBar().value() // 34
        visible = min(shown - start, max(0, math.floor(viewport / 34)))
        last = start + visible
        more = last < self.total
        self.foot.setText(
            f"{last} of {self.total} entries" + (" · scroll for more" if more else "")
            if self.total
            else "Empty folder"
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.root.setGeometry(20, 1 + 14, width - 20 - 16, 32)
        self.crumb.setGeometry(1, 1 + 14 + 32, width - 2, 40)
        top = 1 + 14 + 32 + 40
        self.area.setGeometry(1, top, width - 2, max(0, height - 1 - 40 - top))
        self.entries.resize(width - 2, self.entries.height())
        self.foot.setGeometry(21, height - 1 - 40 + 11, width - 42, 18)
        self._count()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        self._count()  # the list may have been filled while the tab was hidden
        QtGui.QPainter(self).fillRect(
            QtCore.QRect(1, self.height() - 1 - 40, self.width() - 2, 1), self.color("border.line")
        )


# ---------- the viewers ----------


class DataModel(TableModel):
    """A ``files.table`` page: values as written ("null" for none), units from the codebook under the names."""

    def __init__(self) -> None:
        super().__init__([])
        self.types: list[str] = []
        self.units: list[str] = []

    def set_page(
        self,
        columns: Sequence[Mapping[str, Any]],
        rows: Sequence[Sequence[Any]],
        units: Mapping[str, str],
    ) -> None:
        self.beginResetModel()
        self.columns = [(str(c.get("name")), str(c.get("name"))) for c in columns]
        self.types = [str(c.get("type") or "") for c in columns]
        self.units = [units.get(str(c.get("name")), "") for c in columns]
        self.rows = [
            dict(zip((name for name, _ in self.columns), row, strict=False)) for row in rows
        ]
        self.endResetModel()

    def numeric(self, column: int) -> bool:
        return 0 <= column < len(self.types) and bool(NUMERIC.match(self.types[column]))

    def data(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, role: int = 0) -> Any:
        if not index.isValid():
            return None
        value = self.rows[index.row()].get(self.columns[index.column()][0])
        if role == CELL_ROLE:
            if value is None:
                return Cell("null", "mono", token="text.tertiary")  # ``td.nul``
            text = json.dumps(value) if isinstance(value, bool) else str(value)
            return Cell(
                text, "mono", tooltip=text if len(text) > 8 else ""
            )  # shortened values read whole
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return "null" if value is None else str(value)
        return None

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = 0) -> Any:
        if orientation == QtCore.Qt.Orientation.Horizontal and role == SUBLINE_ROLE:
            return self.units[section] if 0 <= section < len(self.units) else None
        return super().headerData(section, orientation, role)


IDENTITY = re.compile(r"^(?=.*[a-f])[0-9a-f-]{16,}$")  # hashes and UUIDs; never a number


def column_width(name: str, unit: str, values: Sequence[str]) -> int:
    """A column as wide as its name, unit and values need, 64–200 px (longer values are cut with "…"); a
    column of identities (hashes, UUIDs) shows their first 8 characters."""
    if values and all(IDENTITY.match(v) for v in values if v != "null"):
        values = [MONO_TEXT.elide(v, MONO_TEXT.advance("00000000…")) for v in values]
    widest = max(
        [TextStyle(12, 500, 18).advance(name), TextStyle(11, 400, 13).advance(unit)]
        + [MONO_TEXT.advance(v) for v in values[:100]]
    )
    return int(min(200, max(64, math.ceil(widest) + 28)))  # ``.dt th`` padding 0 14


class TableView(Themed):
    """A table file: the page's columns (mono, numbers right-aligned), 44 px headers with units, and the footer
    with the row and column range and Previous and Next."""

    page = QtCore.Signal(int)  # the offset asked for

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.view = QtWidgets.QTableView(self)
        self.model = DataModel()
        self.view.setModel(self.model)
        self.look = TableLook(columns=[], header_height=44, row_height=32, key="")
        style_table(self.view, themes, self.look)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.view.horizontalScrollBar().valueChanged.connect(lambda _: self.update())
        self.footer = FlexRow(self, gap=8, left=20, right=12)
        self.rows_text = TextLine(themes, "", FOOT, "text.secondary")
        self.footer.add(self.rows_text)
        self.footer.add_stretch()
        self.previous = Button(themes, "Previous", icon="cleft", small=True)
        self.next = Button(themes, "Next", icon="cright", small=True, trailing=True)
        self.footer.add(self.previous)
        self.footer.add(self.next)
        self.offset: int = 0
        self.total: int = 0
        self.previous.clicked.connect(lambda: self.page.emit(max(0, self.offset - 100)))
        self.next.clicked.connect(lambda: self.page.emit(self.offset + 100))

    def set_page(self, reply: Mapping[str, Any], units: Mapping[str, str]) -> None:
        columns = list(reply.get("columns") or [])
        rows = list(reply.get("rows") or [])
        self.model.set_page(columns, rows, units)
        styles = []
        for index, column in enumerate(columns):
            name = str(column.get("name"))
            values = [
                self.model.data(self.model.index(r, index), QtCore.Qt.ItemDataRole.DisplayRole)
                for r in range(min(len(rows), 100))
            ]
            width = column_width(name, self.model.units[index], [str(v) for v in values])
            styles.append(
                ColumnStyle(
                    "mono",
                    align="right" if self.model.numeric(index) else "left",
                    padding_left=14,
                    padding_right=14,
                    width=width,
                )
            )
        self.look = TableLook(columns=styles, header_height=44, row_height=32, key="")
        apply_columns(self.view, self.themes, self.look)
        self.view.horizontalHeader().setFixedHeight(44)
        previous = self.view.itemDelegate()
        self.view.setItemDelegate(CellDelegate(self.view, self.themes, self.look))
        if previous is not None and previous.parent() is self.view:
            previous.deleteLater()
        self.offset = int(reply.get("offset") or 0)
        self.total = int(reply.get("total") or 0)
        self.previous.setEnabled(self.offset > 0)
        self.next.setEnabled(reply.get("next_offset") is not None)
        self._footer()

    def _footer(self) -> None:
        count = self.model.rowCount()
        header = self.view.horizontalHeader()
        columns = self.model.columnCount()
        first = header.logicalIndexAt(0) + 1 if columns else 0
        last = header.logicalIndexAt(self.view.viewport().width() - 1)
        last = columns if last < 0 else last + 1
        text = f"Rows {self.offset + 1 if count else 0}–{self.offset + count} of {self.total}"
        self.rows_text.setText(
            text + (f"  ·  columns {first}–{last} of {columns}" if columns else "")
        )
        self.footer.relayout()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        self._footer()
        QtGui.QPainter(self).fillRect(
            QtCore.QRect(0, self.height() - 44, self.width(), 1), self.color("border.line")
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.view.setGeometry(0, 0, self.width(), self.height() - 44)
        self.footer.setGeometry(0, self.height() - 43, self.width(), 43)


class TextPane(Themed):
    """``.lw``: the loaded text on the sunken fill, 12/20 px mono, one line per row (cut with "…"), search
    matches marked (``accent.soft``; the current one in ink)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.lines: list[str] = []
        self.marks: list[tuple[int, int, int]] = []  # line, start, length
        self.current = -1

    def set_text(self, text: str) -> None:
        self.lines = text.split("\n")
        if self.lines and self.lines[-1] == "":
            self.lines.pop()
        self.setFixedHeight(10 + 20 * len(self.lines) + 10)
        self.update()

    def mark(self, needle: str) -> None:
        self.marks = []
        if needle:
            lowered = needle.lower()
            for number, line in enumerate(self.lines):
                start = line.lower().find(lowered)
                while start >= 0:
                    self.marks.append((number, start, len(needle)))
                    start = line.lower().find(lowered, start + len(needle))
        self.current = 0 if self.marks else -1
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        painter.fillRect(self.rect(), theme.color("surface.sunken"))
        room = self.width() - 40
        area = event.rect()
        first = max(0, (area.top() - 10) // 20)
        last = min(len(self.lines), (area.bottom() - 10) // 20 + 1)
        body = theme.color("text.body")
        marks_by_line: dict[int, list[tuple[int, int, int]]] = {}
        for index, (row, start, length) in enumerate(self.marks):
            marks_by_line.setdefault(row, []).append((start, length, index))
        for number in range(first, last):
            line = LOG_TEXT.elide(self.lines[number], room)
            top = 10 + 20 * number
            for start, length, index in marks_by_line.get(number, []):
                if start >= len(line):
                    continue
                x0 = 20 + LOG_TEXT.advance(line[:start])
                x1 = 20 + LOG_TEXT.advance(line[: start + length])
                current = index == self.current
                box = QtCore.QRectF(x0 - 1, top + 1, x1 - x0 + 2, 18)
                painter.fillRect(box, theme.color("text.ink" if current else "accent.soft"))
            draw_text(painter, 20, top, line, LOG_TEXT, body)
            for start, length, index in marks_by_line.get(number, []):
                if index == self.current and start < len(line):
                    x0 = 20 + LOG_TEXT.advance(line[:start])
                    draw_text(
                        painter,
                        x0,
                        top,
                        line[start : start + length],
                        LOG_TEXT,
                        theme.color("text.onInk"),
                    )
                elif start < len(line):
                    x0 = 20 + LOG_TEXT.advance(line[:start])
                    draw_text(
                        painter,
                        x0,
                        top,
                        line[start : start + length],
                        LOG_TEXT,
                        theme.color("accent.softText"),
                    )


class TextView(Themed):
    """Text and logs: the search bar (52 px; matches in the loaded windows, the next ones searched for by the
    engine), the text, and the footer with the loaded byte range and "Load next 64 KiB"."""

    more = QtCore.Signal()
    further = QtCore.Signal(str)  # search past the loaded windows

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.bar = FlexRow(self, gap=8, left=20, right=16)
        self.search = SearchField(themes, "Search the loaded text")
        self.bar.add(self.search, width=300)
        self.count = TextLine(themes, "", FOOT, "text.secondary")
        self.bar.add(self.count)
        self.up = Button(themes, icon="cup", small=True)
        self.down = Button(themes, icon="cdown", small=True)
        self.up.setToolTip("Previous match")
        self.down.setToolTip("Next match")
        self.bar.add(self.up)
        self.bar.add(self.down)
        self.area = QtWidgets.QScrollArea(self)
        self.area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.area.setWidgetResizable(False)
        self.pane = TextPane(themes)
        self.area.setWidget(self.pane)
        OverlayScrollBar(themes, self.area)
        self.footer = FlexRow(self, gap=8, left=20, right=12)
        self.range = TextLine(themes, "", FOOT, "text.secondary")
        self.footer.add(self.range)
        self.footer.add_stretch()
        self.load = Button(themes, "Load next 64 KiB", small=True)
        self.footer.add(self.load)
        self.load.clicked.connect(self.more.emit)
        self.search.textChanged.connect(self._find)
        self.search.returnPressed.connect(lambda: self._step(1))
        self.up.clicked.connect(lambda: self._step(-1))
        self.down.clicked.connect(lambda: self._step(1))
        self.text, self.start, self.end_offset, self.finished = "", 0, 0, True

    def set_text(self, text: str, start: int, end: int, finished: bool) -> None:
        self.text, self.start, self.end_offset, self.finished = text, start, end, finished
        self.pane.set_text(text)
        self.pane.resize(self.area.viewport().width(), self.pane.height())
        self.load.setEnabled(not finished)

        def kib(value: int) -> str:
            return f"{value / 1024:.0f} KiB" if value else "0"

        self.range.setText(
            f"Bytes {kib(start)}–{kib(end)} of the text"
            + ("" if finished else "  ·  search runs over the loaded windows")
        )
        self.footer.relayout()
        self._find()

    def _find(self, *_: object) -> None:
        self.pane.mark(self.search.text())
        self._count()
        self._reveal()

    def _count(self) -> None:
        marks = self.pane.marks
        needle = self.search.text()
        self.count.setText(
            ""
            if not needle
            else f"{self.pane.current + 1} of {len(marks)} in this window"
            if marks
            else "No match in the loaded text"
        )
        self.bar.relayout()

    def _step(self, step: int) -> None:
        marks = self.pane.marks
        if not marks:
            if self.search.text() and not self.finished:
                self.further.emit(self.search.text())
            return
        following = self.pane.current + step
        if following >= len(marks) and not self.finished:
            self.further.emit(self.search.text())
            return
        self.pane.current = following % len(marks)
        self.pane.update()
        self._count()
        self._reveal()

    def _reveal(self) -> None:
        if 0 <= self.pane.current < len(self.pane.marks):
            line = self.pane.marks[self.pane.current][0]
            self.area.ensureVisible(0, 10 + 20 * line + 10, 0, 40)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        line = self.color("border.line")
        painter.fillRect(QtCore.QRect(0, 51, self.width(), 1), line)
        painter.fillRect(QtCore.QRect(0, self.height() - 44, self.width(), 1), line)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.bar.setGeometry(0, 0, width, 51)
        self.area.setGeometry(0, 52, width, max(0, height - 52 - 44))
        self.pane.resize(width, self.pane.height())
        self.footer.setGeometry(0, height - 43, width, 43)


class ImageView(Themed):
    """An image or a map's grid on the sunken fill, fitted and centred, with a note above it."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.image: QtGui.QImage | None = None
        self.note = ""

    def set_image(self, image: QtGui.QImage | None, note: str = "") -> None:
        self.image, self.note = image, note
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        painter.fillRect(self.rect(), theme.color("surface.sunken"))
        top = 18
        if self.note:
            for line in self.note.split("\n"):
                draw_text(
                    painter,
                    20,
                    top,
                    BODY.elide(line, self.width() - 40),
                    BODY,
                    theme.color("text.secondary"),
                )
                top += 20
            top += 8
        if self.image is None or self.image.isNull():
            return
        room = QtCore.QSizeF(self.width() - 40, self.height() - top - 18)
        size = QtCore.QSizeF(self.image.size()).scaled(
            room, QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        if size.width() > self.image.width():  # never enlarged past 4× (a small grid stays crisp)
            size = QtCore.QSizeF(self.image.size()) * min(4.0, size.width() / self.image.width())
        box = QtCore.QRectF(
            snap((self.width() - size.width()) / 2), top, size.width(), size.height()
        )
        painter.fillRect(box.adjusted(-1, -1, 1, 1), theme.color("border.line"))
        painter.drawImage(box, self.image)


def map_image(data: bytes) -> tuple[QtGui.QImage | None, str]:
    """A ``.npz`` file's arrays listed, and its first two-dimensional array drawn in greys (as stored)."""
    loaded = np.load(io.BytesIO(data), allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("not an .npz archive")
    with loaded as arrays:
        found = {name: arrays[name] for name in arrays.files}  # each array decompressed once
    notes = [f"{name}: {a.dtype} {' × '.join(map(str, a.shape))}" for name, a in found.items()]
    grid = next((a for a in found.values() if a.ndim == 2), None)
    if grid is None:
        return None, "\n".join(notes)
    values = grid.astype(np.float64)
    if grid.dtype == np.int8 and values.size and values.min() >= -1 and values.max() <= 100:
        # an occupancy grid: free white, occupied black by its probability, unknown (-1) grey
        pixels = np.where(values < 0, 217.0, 255 - values / 100 * 255)
    else:
        finite = values[np.isfinite(values)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        scaled = np.zeros(values.shape) if high == low else (values - low) / (high - low)
        pixels = np.nan_to_num(255 - scaled * 255)
    pixels = np.ascontiguousarray(pixels.astype(np.uint8)[::-1])
    rows, columns = pixels.shape
    image = QtGui.QImage(
        pixels.data, columns, rows, columns, QtGui.QImage.Format.Format_Grayscale8
    ).copy()
    return image, "\n".join(notes)


def unknown_note(name: str, data: bytes, size: int) -> str:
    digest = hashlib.sha256(data).hexdigest()
    return f"{name}: {size_text(size)}\nSHA-256 {digest}\nNo viewer for this kind; Open externally shows it."


TABLE, TEXT, PDF, IMAGE = 0, 1, 2, 3


class ViewerCard(Card):
    """``.card.vw``: the file's name and facts, Open externally, then its viewer."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "", self)
        self.meta = TextLine(themes, "", FOOT, "text.secondary")
        self.head.add(self.meta)
        self.head.row.add_stretch()
        self.open = Button(themes, "Open externally", icon="aupright", variant="ghost", small=True)
        self.head.add(self.open)
        self.stack = QtWidgets.QStackedWidget(self)
        self.table = TableView(themes)
        self.text = TextView(themes)
        self.pdf = Well(themes)
        self.image = ImageView(themes)
        for widget in (self.table, self.text, self.pdf, self.image):
            self.stack.addWidget(widget)

    def set_head(self, name: str, meta: str) -> None:
        self.head.title.setText(name)
        self.meta.setText(meta)
        self.open.setVisible(bool(name))
        self.head.row.relayout()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.head.setGeometry(1, 1, self.width() - 2, 52)
        self.stack.setGeometry(1, 53, self.width() - 2, max(0, self.height() - 54))


# ---------- the tab ----------


class FilesTab(QtWidgets.QWidget):
    """The Files tab: the folder card (360 px) and the viewer, 16 px apart."""

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes, self.transport = themes, transport
        self.folder = FolderCard(themes, self)
        self.viewer = ViewerCard(themes, self)
        self.roots: list[Root] = []
        self.root: Root | None = None
        self.path = ""
        self.host_root: Path | None = None  # the root's directory on this host (from the listing)
        self.entries: list[dict[str, Any]] = []
        self.file: dict[str, Any] | None = None
        self.units: dict[
            str, dict[str, str]
        ] = {}  # table -> column -> unit (the export's codebook)
        self.folder_token = self.file_token = ""
        self.pending: dict[
            str, tuple[str, dict[str, Any]]
        ] = {}  # tag -> request (repeated while loading)
        self.poll = QtCore.QTimer(self)
        self.poll.setSingleShot(True)
        self.poll.setInterval(POLL_MS)
        self.poll.timeout.connect(self._repeat)
        self.loaded_text = ""
        self.jump = False  # the study root's first listing opens the newest export
        self.listing_failed = False  # its listing failed; retried when the tab shows
        self.units_folder = ""  # the export folder whose codebook.csv gave ``units``
        self.table_reply: Mapping[str, Any] | None = (
            None  # the shown table page (redrawn with its units)
        )
        self.search_from: int | None = None  # where an unfinished search goes on (its scan cap)
        self.search_needle = ""  # the text that search was for
        self.folder.root.currentIndexChanged.connect(self._root_picked)
        self.folder.crumb.chosen.connect(self.open_folder)
        self.folder.entries.opened.connect(self._entry)
        self.viewer.open.clicked.connect(self._open_externally)
        self.viewer.table.page.connect(self._table_page)
        self.viewer.text.more.connect(self._more_text)
        self.viewer.text.further.connect(self._search_further)
        self.viewer.set_head("", "Choose a file")

    # roots

    def set_study(self, study_id: str | None) -> None:
        """The study whose exports the first root shows (the page's study select)."""
        roots = [r for r in self.roots if r.kind == "attempt"]
        if study_id:
            roots.insert(0, Root("study", study_id, "Study exports", "study " + study_id))
        self._set_roots(roots, roots[0] if roots else None)

    def study_changed(self, study_id: str) -> None:
        """A new export of ``study_id``: its root, when shown at its top or at ``exports/``, lists again (opening the
        newest export when no file is open); a folder inside an export stays as it is."""
        if self.root is None or self.root.kind != "study" or self.root.identity != study_id:
            return
        if self.listing_failed or (self.path in ("", "exports") and self.file is None):
            self._change_root(self.root)
        elif self.path in ("", "exports"):
            self.open_folder(self.path)

    def retry_failed(self) -> None:
        """The tab shown again: a listing that failed (a study without exports yet) is asked for again."""
        if self.root is not None and self.listing_failed:
            if self.root.kind == "study" and self.file is None:
                self._change_root(self.root)
            else:
                self.open_folder(self.path)

    def show_run(self, attempt_id: str, label: str) -> None:
        """A run's directory as the root (the run view's Files button)."""
        root = Root("attempt", attempt_id, "Run · " + label, "run " + attempt_id[:8])
        runs = [r for r in self.roots if r.kind == "attempt" and r.identity != attempt_id]
        # the study root and the last few runs viewed (the select does not grow over a session)
        roots = [r for r in self.roots if r.kind != "attempt"] + runs[-(RUN_ROOTS - 1) :] + [root]
        self._set_roots(roots, root)

    def _set_roots(self, roots: list[Root], chosen: Root | None) -> None:
        select = self.folder.root
        select.blockSignals(True)
        select.clear()
        for root in roots:
            select.addItem(root.label, root.identity)
        self.roots = roots
        if chosen is not None:
            select.setCurrentIndex(roots.index(chosen))
        select.blockSignals(False)
        if chosen != self.root:
            self._change_root(chosen)

    def _root_picked(self, index: int) -> None:
        if 0 <= index < len(self.roots) and self.roots[index] != self.root:
            self._change_root(self.roots[index])

    def _change_root(self, root: Root | None) -> None:
        """A new root: nothing of the old one stays (its file, units, entries and pending replies)."""
        self.root, self.host_root, self.units, self.units_folder = root, None, {}, ""
        self.file, self.file_token, self.loaded_text = None, uuid.uuid4().hex, ""
        self.entries = []
        self.folder.entries.current = ""
        self.folder.entries.set_entries([])
        self.viewer.stack.setCurrentIndex(IMAGE)
        self.viewer.image.set_image(None, "")
        self.viewer.set_head("", "Choose a file")
        # a study root opens on its newest finished export once, when it is chosen
        self.jump = root is not None and root.kind == "study"
        self.open_folder("")

    # requests

    def _request(self, kind: str, token: str, method: str, params: dict[str, Any]) -> None:
        assert self.root is not None
        tag = f"files-tab-{kind}:{token}"
        full = dict(root=self.root.kind, id=self.root.identity, **params)
        self.pending[tag] = (method, full)
        self.transport.submit(tag, method, full)

    def _repeat(self) -> None:
        for tag, (method, params) in list(self.pending.items()):
            if tag.endswith(self.folder_token) or tag.endswith(self.file_token):
                self.transport.submit(tag, method, params)

    # folders

    def open_folder(self, path: str) -> None:
        if self.root is None:
            self.folder.entries.set_entries([])
            self.folder.set_total(0)
            return
        self.path = path
        self.listing_failed = False
        self.folder_token = uuid.uuid4().hex
        self.folder.crumb.set_path(self.root.home, path)
        self._request(
            "list", self.folder_token, "files.list", {"path": path, "offset": 0, "limit": 250}
        )

    def _listed(self, reply: Mapping[str, Any]) -> None:
        entries = list(reply.get("entries") or [])
        if reply.get("root"):
            self.host_root = Path(str(reply["root"]))
        if reply.get("offset", 0):
            entries = self.entries + entries
        self.entries = entries
        self.folder.entries.set_entries(entries)
        self.folder.entries.resize(
            self.folder.area.viewport().width(), self.folder.entries.height()
        )
        self.folder.set_total(int(reply.get("total") or len(entries)))
        if isinstance(reply.get("next_offset"), int):
            self._request(
                "list",
                self.folder_token,
                "files.list",
                {"path": self.path, "offset": reply["next_offset"], "limit": 250},
            )
            return
        # The study root opens once on its newest finished export (folders named by their UTC time; a staging
        # or partial folder is still being written); the crumb then reaches every folder
        if self.jump and self.root is not None and self.root.kind == "study":
            folders = [
                str(e["name"])
                for e in entries
                if e.get("type") == "directory"
                and not str(e["name"]).endswith(".staging")
                and ".partial-" not in str(e["name"])
            ]
            wanted = "exports" if self.path == "" else max(folders, default=None)
            if self.path in ("", "exports") and wanted is not None and wanted in folders:
                self.jump = self.path == ""  # into exports/, then once into the newest export
                self.open_folder(f"{self.path}/{wanted}".lstrip("/"))
                return
            self.jump = False
        if (
            self.root is not None
            and self.root.kind == "study"
            and self.path.startswith("exports/")
            and self.units_folder != self.path
        ):
            self.units, self.units_folder = {}, self.path
            codebook = f"{self.path}/codebook.csv"
            if any(e.get("path") == codebook for e in entries):
                self._request(
                    "codebook",
                    self.folder_token,
                    "files.table",
                    {"path": codebook, "offset": 0, "limit": 250},
                )

    def _codebook(self, reply: Mapping[str, Any]) -> None:
        names = [str(c.get("name")) for c in reply.get("columns") or []]
        for row in reply.get("rows") or []:
            record = dict(zip(names, row, strict=False))
            unit = str(record.get("unit") or "")
            if unit and record.get("type") not in ("string", "bool"):
                self.units.setdefault(str(record.get("table")), {})[str(record.get("column"))] = (
                    unit
                )
        if isinstance(reply.get("next_offset"), int):
            self._request(
                "codebook",
                self.folder_token,
                "files.table",
                {"path": f"{self.path}/codebook.csv", "offset": reply["next_offset"], "limit": 250},
            )
        elif (
            self.table_reply is not None
            and (self.file or {}).get("kind") == "table"
            and str((self.file or {}).get("path", "")).rpartition("/")[0] == self.units_folder
        ):
            # a table of this folder opened before its codebook arrived gets its units
            self._table(self.table_reply)

    def _entry(self, entry: Mapping[str, Any]) -> None:
        if entry.get("type") == "directory":
            self.open_folder(str(entry.get("path")))
        else:
            self.open_file(dict(entry))

    # files

    def open_file(self, entry: dict[str, Any]) -> None:
        self.file = entry
        self.file_token = uuid.uuid4().hex
        self.search_from, self.table_reply = None, None
        self.folder.entries.current = str(entry.get("path"))
        self.folder.entries.update()
        kind, name, path = str(entry.get("kind")), str(entry.get("name")), str(entry.get("path"))
        viewer = self.viewer
        if kind == "table":
            viewer.set_head(name, "Reading…")
            viewer.stack.setCurrentIndex(TABLE)
            self._table_page(0)
        elif kind in ("text", "log", "json", "yaml"):
            viewer.set_head(name, "Reading…")
            viewer.stack.setCurrentIndex(TEXT)
            self.loaded_text = ""
            viewer.text.search.clear()
            self._request(
                "text", self.file_token, "files.log", {"path": path, "offset": 0, "limit": WINDOW}
            )
        else:
            viewer.set_head(name, size_text(entry.get("size")))
            viewer.stack.setCurrentIndex(PDF if kind == "pdf" else IMAGE)
            viewer.pdf.pages.clear("Reading…")
            viewer.image.set_image(None, "Reading…")
            if (
                self.host_root is None
            ):  # the listing that names the folder's host path has not arrived
                viewer.pdf.pages.clear("Waiting for the folder listing; open the file again")
                viewer.image.set_image(None, "Waiting for the folder listing; open the file again")
            else:
                self.transport.submit(
                    f"files-tab-bytes:{self.file_token}",
                    "local.read_bytes",
                    {"path": str(self.host_root / path)},
                )

    def _table_page(self, offset: int) -> None:
        if self.file is not None:
            self._request(
                "table",
                self.file_token,
                "files.table",
                {"path": str(self.file.get("path")), "offset": offset, "limit": 100},
            )

    def _more_text(self) -> None:
        self.search_from = None  # the window moves on: a search goes on after it
        if self.file is not None:
            self._request(
                "text",
                self.file_token,
                "files.log",
                {
                    "path": str(self.file.get("path")),
                    "offset": self.viewer.text.end_offset,
                    "limit": WINDOW,
                },
            )

    def _search_further(self, needle: str) -> None:
        if self.file is not None:
            if needle != self.search_needle:  # another search starts after the loaded text again
                self.search_from, self.search_needle = None, needle
            offset = self.viewer.text.end_offset
            if self.search_from is not None and self.search_from > offset:
                offset = self.search_from
            self._request(
                "found",
                self.file_token,
                "files.log",
                {
                    "path": str(self.file.get("path")),
                    "offset": offset,
                    "limit": WINDOW,
                    "search": needle,
                },
            )

    def _table(self, reply: Mapping[str, Any]) -> None:
        self.table_reply = reply
        name = str((self.file or {}).get("name"))
        table = name.rsplit(".", 1)[0]
        # the codebook describes the tables beside it, not a raw file of the same name further down
        folder = str((self.file or {}).get("path", "")).rpartition("/")[0]
        units = self.units.get(table, {}) if folder == self.units_folder else {}
        self.viewer.table.set_page(reply, units)
        columns = len(reply.get("columns") or [])
        self.viewer.set_head(
            name,
            f"{reply.get('total', 0)} rows · {columns} columns"
            + (" · units from codebook.csv" if units else ""),
        )

    def _text(self, reply: Mapping[str, Any], *, replace: bool) -> None:
        text = str(reply.get("text") or "")
        offset = int(reply.get("offset") or 0)
        if not replace and self.loaded_text and offset != self.viewer.text.end_offset:
            return  # a window for an earlier position (a search moved the view meanwhile)
        if replace or not self.loaded_text:
            self.loaded_text, start = text, offset
        else:
            self.loaded_text += text
            start = self.viewer.text.start
        end = int(reply.get("next_offset") or offset + len(text.encode()))
        finished = bool(reply.get("end"))
        shown = self.loaded_text
        kind = str((self.file or {}).get("kind"))
        if kind == "json" and finished and start == 0:
            try:
                shown = json.dumps(json.loads(self.loaded_text), indent=2, ensure_ascii=False)
            except ValueError:
                pass  # shown as written
        self.viewer.text.set_text(shown, start, end, finished)
        name = str((self.file or {}).get("name"))
        stored = reply.get("stored_bytes")
        packed = name.endswith((".zst", ".gz"))
        self.viewer.set_head(
            name,
            ("zstd · " if name.endswith(".zst") else "gzip · " if name.endswith(".gz") else "")
            + (f"{size_text(stored)} stored" if packed else size_text(stored)),
        )

    def _bytes(self, data: bytes) -> None:
        entry = self.file or {}
        kind, name = str(entry.get("kind")), str(entry.get("name"))
        if kind == "pdf":
            self.viewer.pdf.pages.clear()
            self.viewer.pdf.pages.add(data)
        elif kind == "image":
            image = QtGui.QImage.fromData(data)
            self.viewer.image.set_image(
                image, "" if not image.isNull() else "The image could not be decoded"
            )
        elif kind == "map":
            try:
                grid, note = map_image(data)
            except Exception as exc:  # noqa: BLE001 (a damaged or foreign file: zip, format, value errors)
                grid, note = None, f"The map could not be read: {exc}"
            self.viewer.image.set_image(grid, note)
        else:
            self.viewer.image.set_image(None, unknown_note(name, data, len(data)))

    def _open_externally(self) -> None:
        if self.host_root is not None and self.file is not None:
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(self.host_root / str(self.file.get("path"))))
            )

    # replies

    def received(self, tag: str, result: Any) -> bool:
        if not tag.startswith("files-tab-"):
            return False
        kind, _, token = tag.removeprefix("files-tab-").partition(":")
        if token not in (self.folder_token, self.file_token):
            self.pending.pop(tag, None)
            return True  # an earlier folder's or file's reply
        if kind == "bytes":
            self._bytes(bytes(result.get("bytes") or b""))
            return True
        state = result.get("state")
        if state == "loading":
            self.poll.start()
            return True
        self.pending.pop(tag, None)
        if state == "failed":
            self._failed(kind, str(result.get("reason") or "Not readable"))
        elif kind == "list":
            self._listed(result)
        elif kind == "codebook":
            self._codebook(result)
        elif kind == "table":
            self._table(result)
        elif kind == "text":
            self._text(result, replace=False)
        elif kind == "found":
            if result.get("match_offset") is None:
                reached = result.get("searched_to")
                if isinstance(reached, int) and not result.get("end"):
                    # the engine stops each search at its scan cap: the next search goes on from there
                    self.search_from = reached
                    text = f"No match up to {size_text(reached)}; search again to go on"
                else:
                    self.search_from = None
                    text = "No match in the rest of the text"
                self.viewer.text.count.setText(text)
                self.viewer.text.bar.relayout()
            else:
                self.search_from = None
                self._text(result, replace=True)
        return True

    def failed(self, tag: str, error: str) -> bool:
        if not tag.startswith("files-tab-"):
            return False
        self.pending.pop(tag, None)
        kind, _, token = tag.removeprefix("files-tab-").partition(":")
        if token in (self.folder_token, self.file_token):
            self._failed(kind, error)
        return True

    def _failed(self, kind: str, reason: str) -> None:
        if kind == "list":
            self.jump, self.listing_failed = False, True
            self.folder.entries.set_entries([])
            self.folder.set_error(reason)
        elif kind in ("table", "text", "found", "bytes"):
            name = str((self.file or {}).get("name"))
            self.viewer.set_head(name, "Not readable: " + reason)
            self.viewer.image.set_image(None, "Not readable: " + reason)
            self.viewer.stack.setCurrentIndex(IMAGE)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.folder.setGeometry(0, 0, 360, self.height())
        self.viewer.setGeometry(376, 0, max(0, self.width() - 376), self.height())

    def close_timers(self) -> None:
        self.poll.stop()
