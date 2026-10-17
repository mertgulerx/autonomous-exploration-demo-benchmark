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

"""The Monitor's feeds (HANDOFF §8.11, §8.12): the toolbar and painted table both feeds share, and the Logs
view: a component's log read 64 KiB at a time into a 2,000-line buffer, parsed into wall time, level and
message, filtered by severity and text, followed every second while "Follow newest" is on, read on when the
table is scrolled to its end, copied with Copy, and opened in its folder from the table's context menu (D4).
"""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from .gui_cells import PaintedColumn, PaintedTable, TableFooter
from .gui_controls import KeySelect, SearchField, Switch
from .gui_menus import Menu
from .gui_support import Transport
from .gui_theme import ThemeManager
from .gui_widgets import Button, FlexRow, PopupHost, TextStyle, Themed, draw_text

# ---------- shared feed parts ----------


class FeedBar(FlexRow):
    """``.lf``: 52 px with a bottom rule, padding 0 12 0 20, controls 8 px apart; ``add_stretch`` before the
    right-hand group (Follow newest and Copy)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, gap=8, left=20, right=12)
        self.themes = themes
        self.inner_height = 51  # controls are centred above the rule
        self.setFixedHeight(52)
        themes.changed.connect(self.update)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(
            QtCore.QRect(0, self.height() - 1, self.width(), 1),
            self.themes.current.color("border.line"),
        )


def copy_text(text: str) -> None:
    clipboard = QtGui.QGuiApplication.clipboard()
    if clipboard is not None:
        clipboard.setText(text)


def open_context_menu(
    widget: QtWidgets.QWidget,
    themes: ThemeManager,
    position: QtCore.QPoint,
    items: Sequence[tuple[str, str, Callable[[], None], bool]],
) -> None:
    """An in-window menu at ``position`` (widget coordinates): (text, icon, callback, enabled) items."""
    host = widget.window().findChild(PopupHost)
    if host is None:
        return
    menu = Menu(themes)
    for text, icon, callback, enabled in items:
        menu.add_item(text, callback, icon=icon, enabled=enabled)
    point = widget.mapTo(host.window_widget, position)
    menu.popup_at(host, QtCore.QRect(point, QtCore.QSize(1, 1)), align="left", opener=widget)


MenuItems = list[tuple[str, str, Callable[[], None], bool]]  # text, icon, callback, enabled


class FeedView(Themed):
    """A feed of the run card (``.run-body.single``): the toolbar (the view's filters, then Follow newest and Copy
    at the right), the table and its 40 px footer (``.lfoot``). Logs and Events share it."""

    def __init__(
        self, themes: ThemeManager, table: PaintedTable, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.bar = FeedBar(themes, self)
        self.table = table
        table.setParent(self)
        table.follow = True  # Follow newest starts on
        table.menu_requested.connect(self._context_menu)
        self.footer = TableFooter(themes, self, padding=20)
        self.follow = Switch(themes, "Follow newest", label_token="text.secondary")
        self.follow.setChecked(True)
        self.follow.toggled.connect(self._follow_toggled)
        self.copy = Button(themes, "Copy", icon="copy")
        self.copy.clicked.connect(self.copy_shown)

    def add_filters(self, *filters: tuple[QtWidgets.QWidget, float | None]) -> None:
        """The view's filters (widget and width) from the left; Follow newest and Copy close the bar."""
        for widget, width in filters:
            self.bar.add(widget, width=width)
        self.bar.add_stretch()
        self.bar.add(self.follow)
        self.bar.add(self.copy)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.bar.setGeometry(0, 0, width, 52)
        self.footer.setGeometry(0, height - 40, width, 40)
        self.table.setGeometry(0, 52, width, max(0, height - 52 - 40))

    def _follow_toggled(self, checked: bool) -> None:
        self.table.follow = checked
        if checked:
            self.table.scroll_to_end()
            self.followed()

    def followed(self) -> None:
        """Follow newest was switched on."""

    def copy_shown(self) -> None:
        raise NotImplementedError

    def menu_items(self, row: int) -> MenuItems:
        """The context menu for ``row`` (−1: no row)."""
        raise NotImplementedError

    def _context_menu(self, position: QtCore.QPoint) -> None:
        items = self.menu_items(self.table.row_at(position.y()))
        open_context_menu(self.table.viewport(), self.themes, position, items)


# ---------- log lines ----------

COMPONENTS = (
    ("worker", "Worker"),
    ("simulation", "Simulation"),
    ("robot", "Robot"),
    ("navigation", "Navigation"),
    ("slam", "SLAM"),
    ("explorer", "Explorer"),
)
SEVERITIES = ("All", "DEBUG", "INFO", "WARN", "ERROR", "FATAL")
LEVEL = re.compile(r"\[(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL)\]")
# ``[process-1] [LEVEL] [stamp] [node]: message`` (the process prefix is optional).
LINE = re.compile(
    r"^(?:\[[^\]]*\]\s+)?\[(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL)\]\s+\[(\d+(?:\.\d+)?)\]"
    r"\s+\[[^\]]*\]:\s?(.*)$"
)
BUFFER_LINES = 2000
READ_BYTES = 65536


@dataclass(frozen=True, slots=True)
class LogLine:
    raw: str
    time: str  # local wall time of the ROS stamp ("" when the line does not parse)
    level: str  # DEBUG, INFO, WARN, ERROR or FATAL as parsed ("" when it does not parse)
    message: str  # the text after the node prefix, else the raw line
    severity: str  # the level for the severity filter (also found in lines that do not parse)


def wall_time(stamp: str) -> str:
    """``%H:%M:%S.mmm`` in local time of a ROS stamp in seconds, rounded to the millisecond."""
    milliseconds = round(float(stamp) * 1000)
    moment = datetime.fromtimestamp(milliseconds // 1000)
    return moment.strftime("%H:%M:%S") + f".{milliseconds % 1000:03d}"


def parse_log_line(raw: str) -> LogLine:
    found = LINE.match(raw)
    if found is not None:
        level = found[1].replace("WARNING", "WARN")
        try:
            time = wall_time(found[2])
        except (OverflowError, OSError, ValueError):
            time = ""
        return LogLine(raw, time, level, found[3], level)
    anywhere = LEVEL.search(raw)
    severity = anywhere[1].replace("WARNING", "WARN") if anywhere else ""
    return LogLine(raw, "", "", raw, severity)


LOG_TEXT = TextStyle(12, 400, 24, mono=True)
LOG_LEVEL = TextStyle(12, 600, 24, mono=True)
LOG_COLUMNS = (  # ``.lh``: 112 and 64 px columns inside a 20 px row padding
    PaintedColumn("Wall time", 132, 20),
    PaintedColumn("Level", 64, 0),
    PaintedColumn("Message", None, 0),
)


def level_look(level: str) -> tuple[str | None, str, str]:
    """Row fill, level colour and message colour of a level (``.ll``, ``.ll.warn``, ``.ll.error``)."""
    if level == "WARN":
        return "state.warnSoft", "state.warn", "text.body"
    if level in ("ERROR", "FATAL"):
        return "state.badSoft", "state.bad", "state.bad"
    if level == "DEBUG":
        return None, "text.tertiary", "text.tertiary"
    return None, "text.secondary", "text.body"


class LogTable(PaintedTable):
    """``.lh`` and ``.ll``: 24 px rows in Ubuntu Sans Mono 12 px, padding 0 20; wall time (112 px,
    ``text.tertiary``), level (64 px, 600) and the message (elided; the whole line in the tooltip). WARN rows
    are tinted ``state.warnSoft``, ERROR and FATAL rows ``state.badSoft`` with the message in ``state.bad``;
    DEBUG rows are ``text.tertiary``."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, LOG_COLUMNS, parent, key="monitor.logs")
        self.lines: list[LogLine] = []

    def set_lines(self, lines: Sequence[LogLine]) -> None:
        self.lines = list(lines)
        self.rows_changed(len(self.lines))

    def row_height(self, index: int) -> int:
        return 24

    def paint_row(self, painter: QtGui.QPainter, index: int, rect: QtCore.QRect) -> None:
        theme = self.themes.current
        line = self.lines[index]
        fill, level_token, message_token = level_look(line.level)
        if fill is not None:
            painter.fillRect(rect, theme.color(fill))
        (time_x, _), (level_x, _), (message_x, message_width) = self.column_spans()
        top = rect.top()
        if line.time:
            draw_text(painter, time_x + 20, top, line.time, LOG_TEXT, theme.color("text.tertiary"))
        if line.level:
            draw_text(painter, level_x, top, line.level, LOG_LEVEL, theme.color(level_token))
        message = LOG_TEXT.elide(line.message, self.message_room())
        draw_text(painter, message_x, top, message, LOG_TEXT, theme.color(message_token))

    def message_room(self) -> float:
        """The message column inside the row's 20 px right padding."""
        _, width = self.column_spans()[2]
        return max(0.0, width - 20)

    def tooltip_at(self, index: int, position: QtCore.QPoint) -> str:
        line = self.lines[index]
        return line.raw if LOG_TEXT.advance(line.message) > self.message_room() else ""


class LogView(FeedView):
    """The Logs view of the run card (``10``): Component, Severity and the text filter over the log table. Replies
    come through the Monitor page (``received`` and ``failed``: False when a tag is not the view's); errors show
    in the footer."""

    TAG = "component-log-"

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        table = LogTable(themes)
        super().__init__(themes, table, parent)
        self.log_table = table
        self.transport = transport
        self.attempt_id: str | None = None
        self.directory: str | None = None
        self.offset = 0
        self.pending = ""
        self.partial = ""
        self.lines: deque[LogLine] = deque(maxlen=BUFFER_LINES)
        self.shown: list[LogLine] = []
        self.component = KeySelect(themes, "Component")
        for key, title in COMPONENTS:
            self.component.addItem(title, key)
        self.component.currentIndexChanged.connect(self.reset)
        self.severity = KeySelect(themes, "Severity")
        self.severity.addItems(list(SEVERITIES))
        self.severity.currentIndexChanged.connect(self.render)
        self.search = SearchField(themes, "Filter messages")
        self.search.textChanged.connect(self.render)
        self.add_filters((self.component, 180), (self.severity, 180), (self.search, 260))
        # D4: reading on when the table is scrolled to its end replaces Load more.
        self.table.reached_end.connect(self.load)
        self.footer.set_text("Select an active or retained attempt")

    # data

    def component_key(self) -> str:
        return str(self.component.currentData() or "worker")

    def select_attempt(self, identity: str, directory: str | None = None) -> None:
        self.attempt_id, self.directory = identity, directory
        self.reset()

    def reset(self, *_: object) -> None:
        self.offset, self.partial, self.pending = 0, "", ""
        self.lines.clear()
        self.render()
        self.load()

    def poll(self) -> None:
        """Every second while the Monitor shows: the next 64 KiB while Follow newest is on."""
        if self.isVisible() and self.follow.isChecked():
            self.load()

    def load(self) -> None:
        if self.attempt_id and not self.pending:
            self.pending = self.TAG + uuid.uuid4().hex
            self.transport.submit(
                self.pending,
                "logs.read",
                {
                    "attempt_id": self.attempt_id,
                    "component": self.component_key(),
                    "offset": self.offset,
                    "limit": READ_BYTES,
                },
            )

    def received(self, tag: str, result: Any) -> bool:
        if not tag.startswith(self.TAG):
            return False
        if tag != self.pending:
            return True  # a reply for an earlier attempt or component
        self.pending = ""
        self.offset = result["next_offset"]
        parts = (self.partial + result["text"]).split("\n")
        self.partial = parts.pop()[-READ_BYTES:]
        self.lines.extend(parse_log_line(part) for part in parts)
        self.footer.set_text(
            f"{self.offset:,} bytes read · {len(self.lines):,} buffered lines"
            if result["available"]
            else "No stored log for this component"
        )
        self.render()
        return True

    def failed(self, tag: str, message: str) -> bool:
        if not tag.startswith(self.TAG):
            return False
        if tag == self.pending:
            self.pending = ""
            self.footer.set_text(message)
        return True

    def render(self, *_: object) -> None:
        needle = self.search.text().casefold()
        severity = self.severity.currentText()
        lines = []
        partial = parse_log_line(self.partial) if self.partial else None
        for line in (*self.lines, *([partial] if partial is not None else [])):
            if needle not in line.raw.casefold():
                continue
            if self.severity.currentIndex() > 0 and line.severity != severity:
                continue
            lines.append(line)
        self.shown = lines
        self.log_table.set_lines(lines)

    # actions

    def followed(self) -> None:
        self.load()

    def copy_shown(self) -> None:
        copy_text("\n".join(line.raw for line in self.shown))

    def open_directory(self) -> None:
        if self.directory:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(self.directory))))

    def menu_items(self, row: int) -> MenuItems:
        items: MenuItems = []
        if row >= 0:
            raw = self.log_table.lines[row].raw
            items.append(("Copy line", "copy", lambda: copy_text(raw), True))
        items.append(("Copy shown lines", "copy", self.copy_shown, bool(self.shown)))
        items.append(("Open log folder", "folder", self.open_directory, bool(self.directory)))
        return items
